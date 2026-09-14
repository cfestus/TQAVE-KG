"""
Tests for utils/checkpoint.py (crash-safe pre-provenance checkpoint +
provenance-grounding failure diagnostics) and their wiring into
fill_missing_relation_provenance.
"""
import json
import os
import tempfile
import types
import unittest
from pathlib import Path

from models import Entity, Relation
from pipeline import EnhancedSemanticPipeline
from utils.checkpoint import (
    PRE_PROVENANCE_CHECKPOINT_STATUS,
    build_provenance_failure_record,
    load_pre_provenance_checkpoint,
    write_pre_provenance_checkpoint,
    write_provenance_grounding_failures,
)
from utils.helpers import GlobalEntityRegistry

ETD = "https://alfie-project.eu/etd-hub#"


def _stub_pipeline(threshold=0.0):
    p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    p.config = types.SimpleNamespace(STRICT_PROVENANCE_GROUNDING=False, MAX_UNKNOWN_CHUNK_INDEX_PCT=threshold)
    p.global_registry = GlobalEntityRegistry()
    p._discourse_chunk_spans = {}
    p._provenance_diagnostics = {}
    return p


def _relation(*, source, target, relation="is", discourse_id, field="answer.description", extractor="llm_relation_extractor"):
    return {"source": source, "target": target, "relation": relation,
            "metadata": {"discourse_id": discourse_id, "field": field, "extractor": extractor}}


def _annotations(discourse_id, relations, text=""):
    return {
        "themes": {}, "questions": {},
        "answers": {discourse_id: {"relations": relations, "metadata": {"text": text}}},
        "votes": {}, "experts": {}, "documents": {}, "metadata": {},
    }


class PreProvenanceCheckpointTests(unittest.TestCase):
    def test_written_checkpoint_is_explicitly_marked_non_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pre_provenance_checkpoint.json"
            ann = {"answers": {"11": {"relations": [], "metadata": {}}}}
            ent = Entity(uri=f"{ETD}e1", label="X", entity_type="Concept", confidence=0.9, metadata={})
            write_pre_provenance_checkpoint(
                path, annotations_dict=ann, entities=[ent], run_id="run-test-1",
                run_context={"dataset_id": "ai_ethics_datasets"},
            )
            loaded = load_pre_provenance_checkpoint(path)
        self.assertEqual(loaded["canonical"], False)
        self.assertEqual(loaded["validation_complete"], False)
        self.assertEqual(loaded["status"], PRE_PROVENANCE_CHECKPOINT_STATUS)
        self.assertEqual(loaded["run_id"], "run-test-1")
        self.assertEqual(loaded["run_context"]["dataset_id"], "ai_ethics_datasets")
        self.assertEqual(loaded["annotations"], ann)
        self.assertEqual(len(loaded["entities"]), 1)
        self.assertEqual(loaded["entities"][0]["uri"], f"{ETD}e1")

    def test_entities_round_trip_via_to_dict_from_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            ent = Entity(uri=f"{ETD}e1", label="Widget", entity_type="Concept", confidence=0.75,
                         metadata={"surface_form": "Widget", "discourse_id": "11"})
            write_pre_provenance_checkpoint(path, annotations_dict={}, entities=[ent])
            loaded = load_pre_provenance_checkpoint(path)
        restored = Entity.from_dict(loaded["entities"][0])
        self.assertEqual(restored.uri, ent.uri)
        self.assertEqual(restored.label, "Widget")
        self.assertEqual(restored.metadata.get("discourse_id"), "11")

    def test_atomic_write_leaves_no_temp_file_behind_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            write_pre_provenance_checkpoint(path, annotations_dict={}, entities=[])
            leftover = [f for f in os.listdir(tmp) if f != path.name]
        self.assertEqual(leftover, [])

    def test_a_write_failure_does_not_leave_a_partial_file_at_the_target_path(self):
        """Simulates a crash mid-write: json.dump raising partway through
        must not leave anything (partial or otherwise) at the FINAL target
        path -- only the temp file (which is then cleaned up) is ever
        touched before the atomic rename."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"

            class _Unserializable:
                pass

            with self.assertRaises(TypeError):
                write_pre_provenance_checkpoint(path, annotations_dict={"bad": _Unserializable()}, entities=[])
            self.assertFalse(path.exists())
            # No stray .tmp files left behind either.
            leftovers = [f for f in os.listdir(tmp)]
            self.assertEqual(leftovers, [])

    def test_never_confused_with_a_valid_checkpoint_by_a_careful_caller(self):
        """A caller that intends to replay a checkpoint must be able to
        positively confirm it is one before trusting it."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            write_pre_provenance_checkpoint(path, annotations_dict={}, entities=[])
            loaded = load_pre_provenance_checkpoint(path)
        is_valid_checkpoint = (
            loaded.get("status") == PRE_PROVENANCE_CHECKPOINT_STATUS and loaded.get("canonical") is False
        )
        self.assertTrue(is_valid_checkpoint)


class ProvenanceGroundingFailuresWiringTests(unittest.TestCase):
    def test_failures_file_written_before_the_raise_when_gate_fails(self):
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["11"] = [
            {"discourse_id": "11", "chunk_index": 0, "start_offset": 0, "end_offset": 50},
            {"discourse_id": "11", "chunk_index": 1, "start_offset": 50, "end_offset": 120},
        ]
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="11")
        ann = _annotations("11", [rel], text="")
        with tempfile.TemporaryDirectory() as tmp:
            failures_path = Path(tmp) / "provenance_grounding_failures.json"
            with self.assertRaises(RuntimeError):
                p.fill_missing_relation_provenance(
                    ann, failures_output_path=failures_path, run_id="run-x", run_context={"dataset_id": "d"},
                )
            self.assertTrue(failures_path.exists())
            with open(failures_path, encoding="utf-8") as f:
                dumped = json.load(f)
        self.assertEqual(dumped["run_id"], "run-x")
        self.assertEqual(len(dumped["failures"]), 1)
        record = dumped["failures"][0]
        self.assertEqual(record["subject"], f"{ETD}s1")
        self.assertEqual(record["predicate"], "is")
        self.assertEqual(record["object"], f"{ETD}o1")
        self.assertEqual(record["failure_classification"], "A_missing_relation_and_entity_spans")
        self.assertIn("id", record)
        self.assertIn("grounding_evidence_available", record)

    def test_no_file_written_when_gate_passes(self):
        p = _stub_pipeline(threshold=1.0)  # permissive -- gate passes even with an unknown
        p._discourse_chunk_spans["11"] = [
            {"discourse_id": "11", "chunk_index": 0, "start_offset": 0, "end_offset": 50},
            {"discourse_id": "11", "chunk_index": 1, "start_offset": 50, "end_offset": 120},
        ]
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="11")
        ann = _annotations("11", [rel], text="")
        with tempfile.TemporaryDirectory() as tmp:
            failures_path = Path(tmp) / "provenance_grounding_failures.json"
            p.fill_missing_relation_provenance(ann, failures_output_path=failures_path)
            self.assertFalse(failures_path.exists())

    def test_no_secrets_or_raw_metadata_dict_leak_into_the_dump(self):
        """Only the explicit allow-listed fields ever go into a failure
        record -- an unexpected key on the relation's metadata (standing
        in for something sensitive a future extractor might attach) must
        not appear."""
        record = build_provenance_failure_record(
            identifier="rel-00001", section="answers",
            rel_dict={
                "source": f"{ETD}s1", "relation": "is", "target": f"{ETD}o1",
                "metadata": {"discourse_id": "11", "field": "answer.description", "api_key": "sk-should-not-appear"},
            },
            source_span=None, target_span=None, chunk_spans_count=1,
            failure_reason="A_missing_relation_and_entity_spans",
        )
        dumped = json.dumps(record)
        self.assertNotIn("sk-should-not-appear", dumped)
        self.assertNotIn("api_key", record["metadata"])

    def test_write_provenance_grounding_failures_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.json"
            write_provenance_grounding_failures(path, failures=[{"id": "rel-00001"}], run_id="r1")
            leftover = [f for f in os.listdir(tmp) if f != path.name]
        self.assertEqual(leftover, [])


if __name__ == "__main__":
    unittest.main()
