"""
Tests for two provenance-grounding fixes:

1. Endpoint-span safety (_entity_span_from_uri): an entity's mention span
   must be scoped to the discourse_id (and, where recorded, field) of the
   relation actually being grounded -- never an arbitrary span belonging
   to a DIFFERENT discourse just because the entity was later
   consolidated with mentions from elsewhere.

2. Authoritative source-text lookup: the A_missing_relation_and_entity_spans
   recovery path (evidence-quote search) must be able to use an
   explicitly-injected (field, discourse_id) -> raw text lookup
   (utils.source_text_lookup.SourceTextLookup) when a record's own
   annotations metadata.text is missing, WITHOUT _ground_relation_metadata
   ever opening a file itself. Recovery only happens when the evidence
   occurrence maps unambiguously to exactly one semantic chunk.
"""
import types
import unittest

from models import Entity
from pipeline import EnhancedSemanticPipeline
from utils.helpers import GlobalEntityRegistry
from utils.source_text_lookup import SourceTextLookup

ETD = "https://alfie-project.eu/etd-hub#"


def _stub_pipeline(threshold=0.0):
    p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    p.config = types.SimpleNamespace(
        STRICT_PROVENANCE_GROUNDING=False,
        MAX_UNKNOWN_CHUNK_INDEX_PCT=threshold,
    )
    p.global_registry = GlobalEntityRegistry()
    p._discourse_chunk_spans = {}
    p._provenance_diagnostics = {}
    return p


def _two_chunk_spans(discourse_id):
    return [
        {"discourse_id": discourse_id, "chunk_index": 0, "start_offset": 0, "end_offset": 50},
        {"discourse_id": discourse_id, "chunk_index": 1, "start_offset": 50, "end_offset": 120},
    ]


def _relation(*, source, target, relation="is", discourse_id, field="answer.description",
              evidence=None, extractor="llm_relation_extractor"):
    meta = {"discourse_id": discourse_id, "field": field, "extractor": extractor}
    if evidence is not None:
        meta["evidence_quote"] = evidence
    return {"source": source, "target": target, "relation": relation, "metadata": meta}


def _annotations(discourse_id, relations, text=""):
    return {
        "themes": {}, "questions": {},
        "answers": {discourse_id: {"relations": relations, "metadata": {"text": text}}},
        "votes": {}, "experts": {}, "documents": {}, "metadata": {},
    }


def _entity_with_mentions(uri, mentions):
    return Entity(uri=uri, label="X", entity_type="Concept", confidence=0.9, metadata={"mentions": mentions})


class EndpointSpanDiscourseSafetyTests(unittest.TestCase):
    """Direct unit tests of _entity_span_from_uri's scoping."""

    def test_entity_mentioned_only_in_a_different_discourse_is_not_used(self):
        p = _stub_pipeline()
        ent = _entity_with_mentions(
            f"{ETD}e1", [{"discourse_id": "d2", "chunk_index": 0, "start_char": 10, "end_char": 20}],
        )
        p.global_registry.entities[f"{ETD}e1"] = ent
        self.assertIsNone(p._entity_span_from_uri(f"{ETD}e1", "d1"))
        # The OTHER discourse's span is still correctly retrievable on its own terms.
        self.assertEqual(p._entity_span_from_uri(f"{ETD}e1", "d2"), (10, 20))

    def test_entity_with_mentions_in_multiple_discourses_uses_only_the_requested_one(self):
        """The core case the task calls out: an entity consolidated across
        several discourses must contribute only ITS OWN span for the
        discourse actually being grounded, never a span combining or
        borrowed from another discourse's mention."""
        p = _stub_pipeline()
        ent = _entity_with_mentions(
            f"{ETD}e1",
            [
                {"discourse_id": "d1", "chunk_index": 0, "start_char": 5, "end_char": 15},
                {"discourse_id": "d2", "chunk_index": 0, "start_char": 500, "end_char": 600},
                {"discourse_id": "d3", "chunk_index": 1, "start_char": 9000, "end_char": 9010},
            ],
        )
        p.global_registry.entities[f"{ETD}e1"] = ent
        self.assertEqual(p._entity_span_from_uri(f"{ETD}e1", "d1"), (5, 15))
        self.assertEqual(p._entity_span_from_uri(f"{ETD}e1", "d2"), (500, 600))
        self.assertEqual(p._entity_span_from_uri(f"{ETD}e1", "d3"), (9000, 9010))
        # A discourse it was never mentioned in at all -> None, not some
        # blend of the above.
        self.assertIsNone(p._entity_span_from_uri(f"{ETD}e1", "d4"))

    def test_scalar_discourse_id_actively_disagreeing_is_rejected(self):
        """No mentions list, but a top-level scalar discourse_id that
        names a DIFFERENT discourse (the get_or_create cross-mention
        drift shape) -- must not be used."""
        p = _stub_pipeline()
        ent = Entity(
            uri=f"{ETD}e1", label="X", entity_type="Concept", confidence=0.9,
            metadata={"discourse_id": "d2", "start_char": 500, "end_char": 600},
        )
        p.global_registry.entities[f"{ETD}e1"] = ent
        self.assertIsNone(p._entity_span_from_uri(f"{ETD}e1", "d1"))
        self.assertEqual(p._entity_span_from_uri(f"{ETD}e1", "d2"), (500, 600))

    def test_entity_with_no_discourse_information_at_all_is_still_usable(self):
        """Backward-compatible: an entity that never recorded ANY
        discourse_id (no mentions list, no scalar discourse_id) is missing
        evidence, not confirmed to belong elsewhere -- still usable,
        unlike the active-mismatch case above."""
        p = _stub_pipeline()
        ent = Entity(uri=f"{ETD}e1", label="X", entity_type="Concept", confidence=0.9,
                     metadata={"start_char": 2, "end_char": 5})
        p.global_registry.entities[f"{ETD}e1"] = ent
        self.assertEqual(p._entity_span_from_uri(f"{ETD}e1", "d1"), (2, 5))

    def test_field_mismatch_rejected_when_both_sides_record_a_field(self):
        """discourse_id ranges are not globally unique across record types
        (a Theme and a Question can share the same numeric id) -- field is
        a second scoping key precisely for that reason."""
        p = _stub_pipeline()
        ent = _entity_with_mentions(
            f"{ETD}e1",
            [{"discourse_id": "14", "field": "theme.description", "chunk_index": 0, "start_char": 5, "end_char": 15}],
        )
        p.global_registry.entities[f"{ETD}e1"] = ent
        # Same discourse_id "14", but a DIFFERENT record type (Question,
        # not Theme) -- must not be treated as the same mention.
        self.assertIsNone(p._entity_span_from_uri(f"{ETD}e1", "14", field="question.body"))
        self.assertEqual(p._entity_span_from_uri(f"{ETD}e1", "14", field="theme.description"), (5, 15))

    def test_missing_field_on_the_mention_does_not_block_a_match(self):
        """A mention recorded before field-tracking existed (no "field"
        key at all) must not be rejected just because the caller supplies
        one -- missing evidence, not confirmed incompatibility."""
        p = _stub_pipeline()
        ent = _entity_with_mentions(
            f"{ETD}e1", [{"discourse_id": "d1", "chunk_index": 0, "start_char": 5, "end_char": 15}],
        )
        p.global_registry.entities[f"{ETD}e1"] = ent
        self.assertEqual(p._entity_span_from_uri(f"{ETD}e1", "d1", field="answer.description"), (5, 15))


class EndToEndRelationGroundingUsesOnlyTheRightDiscourseTests(unittest.TestCase):
    """Same safety property, exercised through the real
    fill_missing_relation_provenance entry point rather than the helper
    directly."""

    def test_relation_grounded_in_d1_ignores_source_entitys_d2_mention(self):
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        ent = _entity_with_mentions(
            f"{ETD}s1",
            [{"discourse_id": "d2", "chunk_index": 0, "start_char": 500, "end_char": 600}],
        )
        p.global_registry.entities[f"{ETD}s1"] = ent
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1")
        ann = _annotations("d1", [rel], text="")
        # The entity's ONLY span belongs to a different discourse -- must
        # stay unknown, not silently borrow it.
        with self.assertRaises(RuntimeError):
            p.fill_missing_relation_provenance(ann)
        self.assertTrue(rel["metadata"]["chunk_index_unknown"])


class AuthoritativeSourceTextRecoveryTests(unittest.TestCase):
    """Reproduces the A_missing_relation_and_entity_spans shape with an
    injected SourceTextLookup standing in for the real dataset file."""

    def _lookup_with(self, field, discourse_id, text):
        return SourceTextLookup({
            "Theme": [], "Question": [], "Answer": [{"id": int(discourse_id), "description": text}],
        }) if field == "answer.description" else SourceTextLookup({})

    def test_metadata_text_missing_but_authoritative_text_available_recovers(self):
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["11"] = _two_chunk_spans("11")
        text = ("x" * 10) + "The special phrase appears here." + ("y" * 77)
        lookup = self._lookup_with("answer.description", "11", text)
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="11", evidence="The special phrase")
        ann = _annotations("11", [rel], text="")  # metadata.text is EMPTY
        result = p.fill_missing_relation_provenance(ann, source_text_lookup=lookup)
        self.assertEqual(result["unknown_chunk_index_count"], 0)
        meta = rel["metadata"]
        self.assertFalse(meta["chunk_index_unknown"])
        self.assertEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["provenance_text_source"], "authoritative_source_text_lookup")

    def test_evidence_in_two_chunks_via_authoritative_text_stays_unknown(self):
        p = _stub_pipeline(threshold=0.0)
        chunk_spans = [
            {"discourse_id": "11", "chunk_index": 0, "start_offset": 0, "end_offset": 20},
            {"discourse_id": "11", "chunk_index": 1, "start_offset": 70, "end_offset": 110},
            {"discourse_id": "11", "chunk_index": 2, "start_offset": 110, "end_offset": 150},
        ]
        p._discourse_chunk_spans["11"] = chunk_spans
        quote = "SHARED EVIDENCE PHRASE"
        # First occurrence in the gap [20,70) (out of chunk range);
        # second occurrence wholly in chunk1; third wholly in chunk2 --
        # genuinely ambiguous once mapped per-chunk.
        text = ("x" * 30) + quote
        text = text.ljust(80) + quote
        text = text.ljust(120) + quote
        text = text.ljust(150, "z")
        lookup = self._lookup_with("answer.description", "11", text)
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="11", evidence=quote)
        ann = _annotations("11", [rel], text="")
        with self.assertRaises(RuntimeError):
            p.fill_missing_relation_provenance(ann, source_text_lookup=lookup)
        self.assertTrue(rel["metadata"]["chunk_index_unknown"])

    def test_evidence_absent_from_authoritative_text_stays_unknown(self):
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["11"] = _two_chunk_spans("11")
        lookup = self._lookup_with("answer.description", "11", "completely unrelated text padded out nicely for the chunks")
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="11", evidence="text that never appears")
        ann = _annotations("11", [rel], text="")
        with self.assertRaises(RuntimeError):
            p.fill_missing_relation_provenance(ann, source_text_lookup=lookup)

    def test_no_source_text_lookup_supplied_stays_unknown_same_as_before(self):
        """Without an injected lookup at all, behavior is unchanged from
        before this fix -- metadata.text missing means genuinely
        unrecoverable, not silently repaired by reaching for a file this
        function was never given."""
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["11"] = _two_chunk_spans("11")
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="11", evidence="The special phrase")
        ann = _annotations("11", [rel], text="")
        with self.assertRaises(RuntimeError):
            p.fill_missing_relation_provenance(ann)  # no source_text_lookup kwarg at all
        self.assertTrue(rel["metadata"]["chunk_index_unknown"])

    def test_zero_threshold_still_raises_on_genuinely_unrecoverable_case_with_lookup_present(self):
        """A lookup being available at all does not weaken the gate --
        a relation with no relation/entity spans AND no matching evidence
        anywhere still fails at threshold 0.0."""
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["11"] = _two_chunk_spans("11")
        lookup = self._lookup_with("answer.description", "11", "x" * 120)
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="11")  # no evidence_quote at all
        ann = _annotations("11", [rel], text="")
        with self.assertRaises(RuntimeError):
            p.fill_missing_relation_provenance(ann, source_text_lookup=lookup)


if __name__ == "__main__":
    unittest.main()
