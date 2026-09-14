"""
Replayability: proves the chain
    extraction checkpoint -> provenance completion -> RDF serialization
can run to completion using ONLY a persisted pre-provenance checkpoint,
with no LLM/network call anywhere in the path. This does NOT replay the
actual crashed run from the 2026-09 investigation -- that run's complete
checkpoint does not exist (nothing like this mechanism existed at the
time) -- it demonstrates the mechanism works for a fresh, synthetic
checkpoint, which is what a FUTURE crash-safe run would leave behind.
"""
import tempfile
import unittest
from pathlib import Path

from config import Config
from models import Entity
from utils.checkpoint import (
    PRE_PROVENANCE_CHECKPOINT_STATUS,
    load_pre_provenance_checkpoint,
    write_pre_provenance_checkpoint,
)

ETD = Config.NAMESPACES["etd"]


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


class ReplayFromCheckpointTests(unittest.TestCase):
    def test_full_chain_replays_without_an_llm_call(self):
        # ------------------------------------------------------------
        # Stage 0: stand in for a COMPLETED extraction (no LLM call is
        # made anywhere in this test -- these are hand-built objects,
        # exactly the shape pipeline.process_batch would have produced).
        # ------------------------------------------------------------
        source_uri = f"{ETD}entity_source_replay"
        target_uri = f"{ETD}entity_target_replay"
        entities = [
            Entity(uri=source_uri, label="Source Thing", entity_type="Concept", confidence=0.9,
                   metadata={"surface_form": "Source Thing", "discourse_id": "d1", "field": "answer.description"}),
            Entity(uri=target_uri, label="Target Thing", entity_type="Concept", confidence=0.9,
                   metadata={"surface_form": "Target Thing", "discourse_id": "d1", "field": "answer.description"}),
        ]
        relation = {
            "source": source_uri, "relation": "relatedTo", "target": target_uri, "confidence": 0.9,
            "metadata": {
                "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                "extractor": "llm_relation_extractor",
                "subject_surface_form": "Source Thing", "object_surface_form": "Target Thing",
                "evidence": "Source Thing relates to Target Thing.",
                "char_start": 0, "char_end": 39,
            },
        }
        annotations_dict = {
            "themes": {}, "questions": {},
            "answers": {"d1": {"relations": [relation], "metadata": {"text": "Source Thing relates to Target Thing."}}},
            "votes": {}, "experts": {}, "documents": {}, "metadata": {"runId": "run-replay-test"},
        }

        # ------------------------------------------------------------
        # Stage 1: checkpoint the complete extraction result (this is
        # what main.py now does BEFORE the provenance gate runs).
        # ------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "pre_provenance_checkpoint.json"
            write_pre_provenance_checkpoint(
                checkpoint_path, annotations_dict=annotations_dict, entities=entities,
                run_id="run-replay-test", run_context={"dataset_id": "ai_ethics_datasets"},
            )

            # ------------------------------------------------------------
            # Stage 2: REPLAY -- load the checkpoint back (simulating a
            # fresh process with none of the original in-memory state,
            # and no LLM/network access) and reconstruct everything it
            # takes to continue.
            # ------------------------------------------------------------
            checkpoint = load_pre_provenance_checkpoint(checkpoint_path)
            self.assertEqual(checkpoint["status"], PRE_PROVENANCE_CHECKPOINT_STATUS)
            self.assertFalse(checkpoint["canonical"])
            replayed_annotations = checkpoint["annotations"]
            replayed_entities = [Entity.from_dict(d) for d in checkpoint["entities"]]
            self.assertEqual(len(replayed_entities), 2)

        # ------------------------------------------------------------
        # Stage 3: provenance completion, using ONLY the replayed data
        # (this is pipeline.fill_missing_relation_provenance -- a pure,
        # deterministic function of annotations_dict; it makes no LLM or
        # network call).
        # ------------------------------------------------------------
        import types
        from pipeline import EnhancedSemanticPipeline
        from utils.helpers import GlobalEntityRegistry

        p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        p.config = types.SimpleNamespace(STRICT_PROVENANCE_GROUNDING=False, MAX_UNKNOWN_CHUNK_INDEX_PCT=0.0)
        p.global_registry = GlobalEntityRegistry()
        for ent in replayed_entities:
            p.global_registry.get_or_create(ent)
        p._discourse_chunk_spans = {
            "d1": [{"discourse_id": "d1", "chunk_index": 0, "start_offset": 0, "end_offset": 60}],
        }
        p._provenance_diagnostics = {}

        result = p.fill_missing_relation_provenance(replayed_annotations)  # must NOT raise
        self.assertEqual(result["unknown_chunk_index_count"], 0)

        # ------------------------------------------------------------
        # Stage 4: RDF serialization from the (now provenance-complete)
        # replayed annotations + replayed entity registry -- no LLM/
        # network call here either.
        # ------------------------------------------------------------
        from serializers.rdf_serializer import RDFSerializer
        from rdflib import Namespace, URIRef

        serializer = RDFSerializer(
            annotations_dict=replayed_annotations,
            registry=_Registry(replayed_entities),
            namespaces=Config.NAMESPACES,
        )
        etd_ns = Namespace(ETD)
        rel_nodes = list(serializer.graph.subjects(etd_ns.predicate, etd_ns.relatedTo))
        self.assertEqual(len(rel_nodes), 1, "the replayed relation must survive all the way to the serialized graph")
        subjects = list(serializer.graph.objects(rel_nodes[0], etd_ns.subject))
        self.assertEqual(subjects, [URIRef(source_uri)])


if __name__ == "__main__":
    unittest.main()
