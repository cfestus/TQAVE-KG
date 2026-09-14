"""
Regression tests for the provenance-recovery gap found investigating a
real controlled-regeneration failure (2026-09): fill_missing_relation_
provenance raised RuntimeError with 2 relations unknown out of 285
checked (rate ~0.007018 > threshold 0.0), classified as
A_missing_relation_and_entity_spans (1) and D_other (1).

Root cause traced in _ground_relation_metadata (pipeline.py):
  - Priority (c) "subject/object mention span" was all-or-nothing: if
    only ONE endpoint had a registered span, the relation got NO span at
    all, even though that single endpoint still deterministically
    anchors it to a real chunk. This is exactly the shape of a
    D_other classification (not A, because one endpoint HAD a span).
  - Priority (d) "uniquely matching semantic unit/chunk" did not exist as
    a general fallback -- only a narrow special case (discourse has
    exactly ONE chunk total) did, and the existing whole-text evidence-
    quote search (priority b) only ever looks at the FIRST occurrence of
    the quote, which can straddle a chunk boundary (or hit a
    non-representative occurrence) even when a LATER occurrence sits
    cleanly inside a single chunk.

Both are fixed here as new, still-conservative fallback tiers (never a
nearest-text/label guess): a single available endpoint span is now used
(labelled distinctly, "single_entity_span"), and -- only once every
earlier attempt has failed to produce a clean single-chunk mapping -- an
evidence quote that occurs in exactly one candidate chunk's own text is
now used ("unique_chunk_evidence_match"). A genuinely unrecoverable case
(per this session's actual failure -- A_missing_relation_and_entity_spans,
or an ambiguous/absent-evidence D_other-shaped case) still correctly
stays unknown and still fails the gate -- these fixes never force a guess.
"""
import types
import unittest

from models import Entity
from pipeline import EnhancedSemanticPipeline
from utils.helpers import GlobalEntityRegistry

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


class AMissingRelationAndEntitySpansTests(unittest.TestCase):
    """Reproduces the exact real failure shape: no relation span, no
    source span, no target span, no usable evidence text -- genuinely
    unrecoverable, must stay unknown and fail the gate at threshold 0.0."""

    def test_classified_unknown_and_gate_fails(self):
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        relations = [_relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1")]
        ann = _annotations("d1", relations, text="")  # no discourse text at all -- (b) can't run
        with self.assertRaises(RuntimeError) as ctx:
            p.fill_missing_relation_provenance(ann)
        self.assertIn("Provenance grounding gate failed", str(ctx.exception))
        rel_meta = relations[0]["metadata"]
        self.assertTrue(rel_meta["chunk_index_unknown"])
        self.assertEqual(rel_meta["chunk_index"], -1)

    def test_verified_no_relation_span_no_entity_spans_no_evidence(self):
        """Explicit verification of each of the sub-checks the task asked
        for: relation span absent, subject span absent, object span
        absent, evidence text not usable."""
        p = _stub_pipeline(threshold=1.0)  # don't raise, just inspect
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1")
        ann = _annotations("d1", [rel], text="")
        p.fill_missing_relation_provenance(ann)
        meta = rel["metadata"]
        self.assertEqual(p._extract_relation_span_from_metadata(meta), None)
        self.assertIsNone(p._entity_span_from_uri(f"{ETD}s1", "d1"))
        self.assertIsNone(p._entity_span_from_uri(f"{ETD}o1", "d1"))
        self.assertTrue(meta["chunk_index_unknown"])


class SingleEndpointSpanRecoveryTests(unittest.TestCase):
    """The fixed gap: exactly ONE endpoint has a registered span -- under
    the OLD logic this fell through to unknown (D_other-shaped: not A,
    since one endpoint DID have a span); now it must be recovered via the
    new single_entity_span tier."""

    def _register_entity_with_span(self, p, uri, discourse_id, start, end):
        ent = Entity(
            uri=uri, label="X", entity_type="Concept", confidence=0.9,
            metadata={"mentions": [{"discourse_id": discourse_id, "chunk_index": 0, "start_char": start, "end_char": end}]},
        )
        p.global_registry.entities[uri] = ent

    def test_only_source_has_span_now_recovers(self):
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        self._register_entity_with_span(p, f"{ETD}s1", "d1", 5, 15)  # inside chunk 0 (0-50)
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1")  # target has NO span
        ann = _annotations("d1", [rel], text="")
        result = p.fill_missing_relation_provenance(ann)  # must NOT raise
        self.assertEqual(result["unknown_chunk_index_count"], 0)
        meta = rel["metadata"]
        self.assertFalse(meta["chunk_index_unknown"])
        self.assertEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["groundingMethod"], "single_entity_span")

    def test_only_target_has_span_now_recovers(self):
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        self._register_entity_with_span(p, f"{ETD}o1", "d1", 60, 70)  # inside chunk 1 (50-120)
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1")
        ann = _annotations("d1", [rel], text="")
        result = p.fill_missing_relation_provenance(ann)
        self.assertEqual(result["unknown_chunk_index_count"], 0)
        meta = rel["metadata"]
        self.assertEqual(meta["chunk_index"], 1)
        self.assertEqual(meta["groundingMethod"], "single_entity_span")

    def test_single_endpoint_span_out_of_all_chunk_ranges_stays_unknown(self):
        """The recovery must not guess when even the single available span
        doesn't actually fall inside any known chunk -- still fails
        conservatively rather than picking a nearest chunk."""
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        self._register_entity_with_span(p, f"{ETD}s1", "d1", 500, 510)  # outside both chunks (0-120)
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1")
        ann = _annotations("d1", [rel], text="")
        with self.assertRaises(RuntimeError):
            p.fill_missing_relation_provenance(ann)
        self.assertTrue(rel["metadata"]["chunk_index_unknown"])

    def test_both_endpoints_still_prefer_the_original_two_span_derivation(self):
        """Confirms the new single-endpoint branch does not fire, and does
        not change behavior, when BOTH endpoints have spans (unaffected,
        pre-existing path)."""
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        self._register_entity_with_span(p, f"{ETD}s1", "d1", 5, 15)
        self._register_entity_with_span(p, f"{ETD}o1", "d1", 20, 30)
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1")
        ann = _annotations("d1", [rel], text="")
        p.fill_missing_relation_provenance(ann)
        meta = rel["metadata"]
        self.assertEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["groundingMethod"], "span_map")


class UniqueChunkEvidenceMatchTests(unittest.TestCase):
    """The other fixed gap: the existing whole-text evidence-quote search
    (priority b) only looks at the FIRST occurrence of the quote and maps
    THAT position through the chunk boundaries -- if that first occurrence
    straddles a chunk boundary (ambiguous/cross-chunk), the relation was
    previously left unknown even when a LATER occurrence of the same
    quote sits cleanly inside a single chunk. The new fallback checks
    each candidate chunk's own text directly and uses it ONLY when the
    match is unique across chunks."""

    def _build_straddling_then_clean_text(self):
        # chunk0 = [0,30), chunk1 = [30,90), chunk2 = [90,150)
        quote = "SHARED EVIDENCE PHRASE"  # 23 chars
        chunk_spans = [
            {"discourse_id": "d1", "chunk_index": 0, "start_offset": 0, "end_offset": 30},
            {"discourse_id": "d1", "chunk_index": 1, "start_offset": 30, "end_offset": 90},
            {"discourse_id": "d1", "chunk_index": 2, "start_offset": 90, "end_offset": 150},
        ]
        # Occurrence 1 starts at 20 (inside chunk0) and ends at 43 (inside
        # chunk1) -- straddles the chunk0/chunk1 boundary at 30, so a
        # whole-text .find() landing here maps to TWO chunks, not one.
        self.assertGreater(20 + len(quote), 30)
        text = ("x" * 20) + quote
        # Occurrence 2 sits wholly inside chunk1 [30,90), comfortably away
        # from both of its boundaries.
        text = text.ljust(60) + quote
        text = text.ljust(150, "y")
        return text, chunk_spans, quote

    def test_ambiguous_first_match_rescued_by_a_later_clean_occurrence(self):
        p = _stub_pipeline(threshold=0.0)
        text, chunk_spans, quote = self._build_straddling_then_clean_text()
        p._discourse_chunk_spans["d1"] = chunk_spans
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1", evidence=quote)
        ann = _annotations("d1", [rel], text=text)
        result = p.fill_missing_relation_provenance(ann)
        self.assertEqual(result["unknown_chunk_index_count"], 0)
        meta = rel["metadata"]
        self.assertEqual(meta["chunk_index"], 1)
        self.assertEqual(meta["groundingMethod"], "unique_chunk_evidence_match")

    def test_quote_wholly_contained_in_two_different_chunks_is_not_guessed(self):
        """If the evidence quote's FIRST occurrence falls entirely in a
        gap covered by no chunk at all (priority-b's mapping finds zero
        overlapping chunks -- genuinely out of range, not the pre-existing
        "legitimate cross-chunk evidence" case), and it ALSO occurs,
        wholly, inside two DIFFERENT chunks elsewhere, the new fallback
        must not pick either of those -- ambiguous stays unknown and the
        gate still fails."""
        p = _stub_pipeline(threshold=0.0)
        quote = "SHARED EVIDENCE PHRASE"  # 23 chars
        # chunk0=[0,20), GAP=[20,70) (no chunk defined here at all),
        # chunk1=[70,110), chunk2=[110,150).
        chunk_spans = [
            {"discourse_id": "d1", "chunk_index": 0, "start_offset": 0, "end_offset": 20},
            {"discourse_id": "d1", "chunk_index": 1, "start_offset": 70, "end_offset": 110},
            {"discourse_id": "d1", "chunk_index": 2, "start_offset": 110, "end_offset": 150},
        ]
        # Occurrence 1: wholly inside the gap [20,70) -- out of range of
        # every chunk, so priority-b's mapping yields ZERO segments (not
        # the cross-chunk case), and this is the position .find() returns
        # first.
        text = ("x" * 30) + quote  # occupies [30, 53) -- inside the gap
        # Occurrence 2: wholly inside chunk1 [70,110).
        text = text.ljust(80) + quote  # occupies [80, 103)
        # Occurrence 3: wholly inside chunk2 [110,150).
        text = text.ljust(120) + quote  # occupies [120, 143)
        text = text.ljust(150, "y")
        p._discourse_chunk_spans["d1"] = chunk_spans
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1", evidence=quote)
        ann = _annotations("d1", [rel], text=text)
        with self.assertRaises(RuntimeError):
            p.fill_missing_relation_provenance(ann)
        self.assertTrue(rel["metadata"]["chunk_index_unknown"])

    def test_evidence_absent_from_every_chunk_stays_unknown(self):
        p = _stub_pipeline(threshold=0.0)
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1", evidence="text that never appears")
        ann = _annotations("d1", [rel], text="completely unrelated content padded out to fill the chunks nicely")
        with self.assertRaises(RuntimeError):
            p.fill_missing_relation_provenance(ann)


class DOtherClassificationTests(unittest.TestCase):
    """D_other is the catch-all reason when a relation is counted unknown
    but doesn't match A (no relation/entity spans at all), B (relation
    span present, chunk spans present, zero evidence spans -- an
    out-of-range/mapping miss), or C (grounded cross-chunk span_map).
    Reproduces a D_other-shaped case that remains genuinely unrecoverable
    even after the new fallbacks (single endpoint span maps to ZERO
    chunks -- present but useless -- with no usable evidence text
    either), proving D_other is not automatically a bug -- only cases
    where a deterministic recovery actually exists were fixed."""

    def test_single_endpoint_span_present_but_unmappable_is_d_other_and_stays_unknown(self):
        p = _stub_pipeline(threshold=1.0)  # inspect classification without raising
        p._discourse_chunk_spans["d1"] = _two_chunk_spans("d1")
        ent = Entity(
            uri=f"{ETD}s1", label="X", entity_type="Concept", confidence=0.9,
            metadata={"mentions": [{"discourse_id": "d1", "chunk_index": 0, "start_char": 999, "end_char": 1005}]},
        )
        p.global_registry.entities[f"{ETD}s1"] = ent
        rel = _relation(source=f"{ETD}s1", target=f"{ETD}o1", discourse_id="d1")
        ann = _annotations("d1", [rel], text="")
        result = p.fill_missing_relation_provenance(ann)
        self.assertEqual(result["unknown_chunk_index_count"], 1)
        self.assertIn("D_other", result["unknown_non_excluded_by_reason"])
        self.assertTrue(rel["metadata"]["chunk_index_unknown"])


if __name__ == "__main__":
    unittest.main()
