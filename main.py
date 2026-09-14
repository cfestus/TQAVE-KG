#!/usr/bin/env python3
"""
main.py
-------
Main entry point for the TQAVE Semantic Knowledge Graph Pipeline.

Usage:
    python main.py [--config config.json] [--input data.json] [--output output_dir]
"""
import argparse
import os
import sys
import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

class SHACLValidationError(RuntimeError):
    pass


class GraphDBUploadError(RuntimeError):
    pass


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='TQAVE Semantic Knowledge Graph Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory path'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Disable reasoning'
    )
    parser.add_argument(
        '--no-wikidata',
        action='store_true',
        help='Disable Wikidata linking'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM-only extraction (requires OpenAI API key)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
        default='gpt-4o-mini',
        help='LLM model to use for extraction'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    print("=" * 70)
    print("  TQAVE Semantic Knowledge Graph Pipeline")
    print("  Version 1.0.0")
    print("=" * 70)

    args = parse_arguments()

    # Lazy imports: keep --help usable even when optional runtime deps are missing.
    from pipeline import EnhancedSemanticPipeline
    from serializers.rdf_serializer import RDFSerializer
    from validation.shacl_validator import SHACLValidator
    from utils.graphdb_client import GraphDBClient
    from utils.helpers import (
        create_output_directory,
        save_json,
        load_json,
        validate_input_json,
    )
    from utils.semantic_quality_events import write_semantic_quality_report

    # ---------------------------------------------------------
    # Load / override configuration
    # ---------------------------------------------------------
    if args.config:
        Config.load_from_file(Path(args.config))

    if args.input:
        Config.INPUT_JSON = args.input
    if args.output:
        Config.OUTPUT_DIR = Path(args.output)
    if args.debug:
        Config.DEBUG = True
    if args.no_reasoning:
        Config.USE_REASONING = False
    if args.no_wikidata:
        Config.USE_WIKIDATA_LINKING = False
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size

    # Extraction mode: we now use LLM-only entity extraction
    # The --use-llm flag is kept for future extension, but we default to LLM anyway.
    Config.USE_LLM_EXTRACTION = True
    Config.USE_HYBRID_EXTRACTION = False
    if hasattr(Config, "USE_BERT_EXTRACTION"):
        Config.USE_BERT_EXTRACTION = False
    print("LLM-only extraction enforced")
    if not Config.USE_LLM_EXTRACTION or Config.USE_HYBRID_EXTRACTION:
        raise RuntimeError(
            "LLM-only extraction enforcement failed: USE_LLM_EXTRACTION must be true and "
            "USE_HYBRID_EXTRACTION must be false."
        )

    # ---------------------------------------------------------
    # Validate configuration
    # ---------------------------------------------------------
    if not Config.validate():
        print("\n❌ Configuration validation failed")
        sys.exit(1)

    Config.print_config()

    # ---------------------------------------------------------
    # IO setup
    # ---------------------------------------------------------
    create_output_directory(Config.OUTPUT_DIR)

    print(f"\n📂 Loading data from {Config.INPUT_JSON}.")
    input_path = Path(Config.INPUT_JSON)

    if not input_path.exists():
        print(f"❌ File not found: {Config.INPUT_JSON}")
        sys.exit(1)

    try:
        data = load_json(input_path)
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        sys.exit(1)

    try:
        validate_input_json(data)
    except Exception as e:
        print(f"❌ Input JSON validation failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Initialise pipeline
    # ---------------------------------------------------------
    try:
        pipeline = EnhancedSemanticPipeline(Config)
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        logger.exception("Pipeline initialization failed")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  Processing Documents")
    print("=" * 70)

    # ---------------------------------------------------------
    # Process documents
    # ---------------------------------------------------------
    try:
        annotations = pipeline.process_batch(data)
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
        logger.exception("Document processing failed")
        sys.exit(1)

    # ---------------------------------------------------------
    # Normalise annotations to a dict for downstream stability
    # ---------------------------------------------------------
    if isinstance(annotations, dict):
        annotations_dict = annotations
    elif hasattr(annotations, "to_dict"):
        annotations_dict = annotations.to_dict()
    else:
        raise TypeError(f"Unsupported annotations type: {type(annotations)}")

    # ---------------------------------------------------------
    # Reasoning (before RDF build)
    # ---------------------------------------------------------
    inferred_relations = pipeline.apply_reasoning(annotations_dict)
    if inferred_relations:
        annotations_dict.setdefault("metadata", {})
        annotations_dict["metadata"]["inferred_relations"] = [r.to_dict() for r in inferred_relations]

    # ---------------------------------------------------------
    # Conflict detection
    # ---------------------------------------------------------
    try:
        conflicts = pipeline.detect_conflicts(annotations_dict)
    except Exception as e:
        print(f"⚠️  Conflict detection failed: {e}")
        logger.exception("Conflict detection failed")
        conflicts = []

    # ---------------------------------------------------------
    # Save outputs and serialize RDF
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Saving Outputs")
    print("=" * 70)

    output_paths = Config.get_output_paths()

    try:
        # Run identity, decided ONCE, here -- before anything below can
        # fail -- and shared by the checkpoint, the failure diagnostics
        # (if needed), and the final RDFSerializer (which picks up
        # annotations_dict["metadata"]["runId"] automatically; see
        # RDFSerializer._init_export_metadata), so all three always agree
        # on which run they belong to.
        from datetime import datetime, timezone
        from utils.helpers import safe_hash

        run_meta = annotations_dict.setdefault("metadata", {})
        dataset_id = run_meta.get("datasetId") or Path(str(Config.INPUT_JSON)).stem
        run_id = run_meta.get("runId")
        if not run_id:
            now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"run-{safe_hash(str(dataset_id) + '|' + now)}"
            run_meta["runId"] = run_id
        run_context = {
            "dataset_id": str(dataset_id),
            "pipeline_version": str(getattr(Config, "PIPELINE_VERSION", None) or "5.1"),
            "git_commit": os.getenv("GIT_COMMIT") or "unknown",
            "generated_at_time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

        # Crash-safe checkpoint of the complete in-memory extraction
        # result, written BEFORE the provenance gate runs -- so a
        # RuntimeError from that gate (as happened in the 2026-09
        # controlled-regeneration failure this was built to prevent a
        # repeat of) does not discard the already-completed, expensive
        # LLM-extraction work. Explicitly non-canonical -- never treated
        # as, or substituted for, the final TTL/annotations output.
        from utils.checkpoint import write_pre_provenance_checkpoint

        try:
            checkpoint_path = write_pre_provenance_checkpoint(
                output_paths["pre_provenance_checkpoint"],
                annotations_dict=annotations_dict,
                entities=pipeline.global_registry.get_all_entities(),
                run_id=run_id,
                run_context=run_context,
            )
            print(f"\n💾 Pre-provenance checkpoint written (non-canonical): {checkpoint_path}")
        except Exception:
            logger.exception(
                "Failed to write pre-provenance checkpoint -- continuing "
                "(this is a safety net, not a required step)."
            )

        # Authoritative (field, discourse_id) -> raw source-field text
        # lookup, built once from the RAW INPUT DATASET (not the
        # annotations_dict, whose own per-record metadata.text is
        # reliably populated for some record types but not others) --
        # passed explicitly into the provenance fill pass rather than
        # having that lower-level function open the file itself.
        from utils.source_text_lookup import SourceTextLookup

        try:
            source_text_lookup = SourceTextLookup.from_json_file(Config.INPUT_JSON)
        except Exception:
            logger.exception(
                "Failed to build the authoritative source-text lookup -- "
                "provenance recovery will fall back to annotations "
                "metadata.text alone."
            )
            source_text_lookup = None

        provenance_fill_fn = getattr(pipeline, "fill_missing_relation_provenance", None)
        if callable(provenance_fill_fn):
            provenance_fill = provenance_fill_fn(
                annotations_dict,
                source_text_lookup=source_text_lookup,
                failures_output_path=output_paths["provenance_grounding_failures"],
                run_id=run_id,
                run_context=run_context,
            )
            print(
                "\nProvenance fill pass: "
                f"filled_span_map={provenance_fill.get('filled_from_span_map', 0)}, "
                f"filled_single_chunk={provenance_fill.get('filled_from_single_chunk_discourse', 0)}, "
                f"cross_chunk={provenance_fill.get('cross_chunk_evidence', 0)}, "
                f"unknown={provenance_fill.get('unknown_chunk_index_count', 0)}"
            )
        print("\n🧱 Building RDF graph (single pass).")
        serializer = RDFSerializer(
            annotations_dict,
            pipeline.global_registry,
            Config.NAMESPACES,
            quality_events=getattr(pipeline, "quality_events", None),
            # C.2/C.3: lets the serializer verify each entity's emitted
            # (field, discourse_id) span against the real source text and
            # withhold offsets it cannot ground, instead of emitting a
            # drifted or over-capturing span.
            source_text_lookup=source_text_lookup,
        )
        g = serializer.graph

        # Root-cause fix (bounded preflight, item 4): write the
        # semantic-quality report IMMEDIATELY after the graph is built,
        # not after SHACL/upload/serialize-to-file -- if SHACL fails
        # with FAIL_ON_SHACL_VIOLATIONS (or GraphDB upload fails with
        # FAIL_ON_UPLOAD_FAILURE), the outer except clause below exits
        # before ever reaching the ORIGINAL write call further down,
        # meaning a quality-gate-failing run left no audit trail at
        # exactly the moment it is most needed. Written here, the report
        # survives regardless of what happens afterward. Includes only
        # non-secret run/configuration identity (see docstring).
        print("\n💾 Saving semantic-quality instrumentation report.")
        write_semantic_quality_report(
            serializer.quality_events,
            str(output_paths["semantic_quality_report"]),
            run_id=getattr(serializer, "run_id", None),
            run_context={
                "dataset_id": getattr(serializer, "dataset_id", None),
                "pipeline_version": getattr(serializer, "pipeline_version", None),
                "git_commit": getattr(serializer, "git_commit", None),
                "generated_at_time": getattr(serializer, "generated_at_time", None),
                "domain_profile": getattr(getattr(serializer, "domain_profile", None), "name", None),
            },
        )

        print("\n🔎 Running SHACL validation.")
        validator = SHACLValidator(
            output_dir=Config.OUTPUT_DIR,
        )
        shacl_result = validator.validate_graph(g)
        if not shacl_result.get("conforms", False):
            msg = (
                "SHACL validation failed. "
                f"Report: {shacl_result.get('report_path')}"
            )
            if Config.FAIL_ON_SHACL_VIOLATIONS:
                raise SHACLValidationError(msg)
            print(f"⚠️ {msg}")
        else:
            print("✅ SHACL validation passed.")

        if Config.USE_GRAPHDB_UPLOAD:
            graphdb = GraphDBClient()
            conn = graphdb.check_connection()
            if not conn.get("success"):
                msg = f"GraphDB connection failed: {conn.get('error')}"
                if Config.FAIL_ON_UPLOAD_FAILURE:
                    raise GraphDBUploadError(msg)
                print(f"⚠️ {msg}")
            else:
                up = graphdb.upload_rdf(g)
                if not up.get("success"):
                    msg = f"GraphDB upload failed: {up.get('error')}"
                    if Config.FAIL_ON_UPLOAD_FAILURE:
                        raise GraphDBUploadError(msg)
                    print(f"⚠️ {msg}")
                else:
                    print("✅ GraphDB upload completed.")

        print("\n💾 Serializing to RDF formats.")
        serializer.serialize_to_turtle(output_paths['ttl'])
        serializer.serialize_to_rdf(output_paths['rdf'])
        serializer.serialize_to_jsonld(output_paths['jsonld'])
        serializer.serialize_to_owl(output_paths['owl'])
        print("✓ RDF serialization complete")

        print("\n💾 Saving annotations.")
        save_json(annotations_dict, output_paths["annotations"])

        # ---------------------------------------------------------
        # Quality metrics (post RDF/SHACL/upload)
        # ---------------------------------------------------------
        quality_metrics = pipeline.compute_quality_metrics(annotations_dict)

        print("\n💾 Saving quality metrics.")
        save_json(quality_metrics, output_paths['metrics'])

        if conflicts:
            print("\n💾 Saving conflicts.")
            save_json({'conflicts': conflicts}, output_paths['conflicts'])

    except (SHACLValidationError, GraphDBUploadError):
        logger.exception("Fatal pipeline validation/upload failure")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error saving outputs: {e}")
        logger.exception("Output persistence failed")
        sys.exit(1)


if __name__ == '__main__':
    main()


