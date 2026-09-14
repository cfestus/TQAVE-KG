import importlib.util
import unittest

from config import Config
from models import Entity
from utils.evidence_alignment import (
    align_evidence_span,
    STATUS_EXACT,
    STATUS_NORMALIZED_EXACT,
    STATUS_FUZZY_RECOVERED,
    STATUS_UNRESOLVED,
)

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class AlignmentUnitTests(unittest.TestCase):
    def test_exact_source_evidence_remains_unchanged(self):
        ctx = "The system was trained on a large dataset of records."
        r = align_evidence_span("trained on a large dataset", ctx)
        self.assertEqual(r.status, STATUS_EXACT)
        self.assertEqual(r.text, "trained on a large dataset")
        self.assertEqual(ctx[r.context_char_start:r.context_char_end], r.text)

    def test_markdown_bold_source_aligns_to_cleaned_candidate(self):
        ctx = "**Content**: some descriptive value about the subject matter."
        cand = "Content: some descriptive value about the subject matter"
        r = align_evidence_span(cand, ctx)
        self.assertEqual(r.status, STATUS_NORMALIZED_EXACT)
        self.assertEqual(ctx[r.context_char_start:r.context_char_end], r.text)
        # Recovered text must be copied verbatim from source, not the cleaned candidate.
        self.assertNotEqual(r.text, cand)

    def test_escaped_punctuation_aligns_correctly(self):
        ctx = "predicting income \\>$50K/year with several features"
        cand = "predicting income >$50K/year with several features"
        r = align_evidence_span(cand, ctx)
        self.assertEqual(r.status, STATUS_NORMALIZED_EXACT)
        self.assertEqual(ctx[r.context_char_start:r.context_char_end], r.text)

    def test_repeated_whitespace_and_newlines_align_correctly(self):
        ctx = "Some Subject clearly\r\n   exhibits a specific   verifiable property in this sentence."
        cand = "exhibits a specific verifiable property"
        r = align_evidence_span(cand, ctx)
        self.assertEqual(r.status, STATUS_NORMALIZED_EXACT)
        self.assertEqual(ctx[r.context_char_start:r.context_char_end], r.text)

    def test_unicode_quote_and_dash_variation_aligns_back_to_raw_source(self):
        ctx = "The report says \u201cbias occurs\u201d \u2014 frequently in practice."
        cand = '"bias occurs" - frequently'
        r = align_evidence_span(cand, ctx)
        self.assertEqual(r.status, STATUS_NORMALIZED_EXACT)
        self.assertEqual(ctx[r.context_char_start:r.context_char_end], r.text)

    def test_unique_mild_paraphrase_recovers_only_above_similarity_threshold(self):
        ctx = "Content: ~12,800 essays from grades 7 to 10 across 8 different prompts collected over years."
        cand = "Content: ~12,800 essays from grades 7-10 across 8 different prompts"
        r = align_evidence_span(cand, ctx)
        self.assertEqual(r.status, STATUS_FUZZY_RECOVERED)
        self.assertEqual(ctx[r.context_char_start:r.context_char_end], r.text)
        self.assertGreaterEqual(float(r.method.rsplit("=", 1)[-1].rstrip(")")), 0.85)

    def test_weak_paraphrase_below_threshold_is_unresolved_not_recovered(self):
        ctx = "The dataset contains business reviews and ratings with geographic and demographic patterns."
        cand = "reviews and ratings that reflect geographic & demographic patterns"
        r = align_evidence_span(cand, ctx)
        self.assertEqual(r.status, STATUS_UNRESOLVED)
        self.assertEqual(r.text, cand, "unresolved text must remain the original candidate, not a fabricated recovery")

    def test_ambiguous_repeated_phrase_does_not_choose_first_occurrence(self):
        ctx = "the word test appears here in the first sentence. Later on, the word test appears again in a different spot."
        cand = "the word test appears"
        second_occurrence = ctx.rfind(cand)
        r_no_hint = align_evidence_span(cand, ctx)
        self.assertEqual(r_no_hint.status, STATUS_UNRESOLVED)
        r_with_hint = align_evidence_span(cand, ctx, hint_char_offset=second_occurrence)
        self.assertEqual(r_with_hint.status, STATUS_EXACT)
        self.assertEqual(r_with_hint.context_char_start, second_occurrence)

    def test_unresolved_candidate_is_flagged_not_serialized_as_fake_quote(self):
        ctx = "A completely different sentence about something else entirely unrelated to the candidate."
        cand = "shows geographic bias in outcomes across the board"
        r = align_evidence_span(cand, ctx)
        self.assertEqual(r.status, STATUS_UNRESOLVED)
        self.assertIsNone(r.context_char_start)
        self.assertIsNone(r.context_char_end)

    def test_offsets_satisfy_exact_substring_invariant_across_all_resolved_tiers(self):
        cases = [
            ("trained on a dataset", "The model was trained on a dataset of records."),
            ("Content: value here", "**Content**: value here in the source."),
            ("a b c", "prefix\n\n  a   b\tc  suffix"),
        ]
        for cand, ctx in cases:
            r = align_evidence_span(cand, ctx)
            if r.status != STATUS_UNRESOLVED:
                self.assertEqual(ctx[r.context_char_start:r.context_char_end], r.text)

    def test_missing_source_context_is_unresolved(self):
        r = align_evidence_span("some candidate text", "")
        self.assertEqual(r.status, STATUS_UNRESOLVED)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for evidence-alignment serializer tests")
class _RDFSerializerHarness(unittest.TestCase):
    class _Registry:
        def __init__(self, entities):
            self.entities = {e.uri: e for e in entities}

        def get_all_entities(self):
            return list(self.entities.values())

    def _relation_dict(
        self, uri, source, relation, target, *, evidence_quote, source_context,
        field="theme.description", discourse_id="d1", chunk_index=0,
    ):
        meta = {
            "evidence_quote": evidence_quote,
            "evidence": evidence_quote,
            "field": field,
            "source_field": field,
            "discourse_id": discourse_id,
            "source_doc_id": discourse_id,
            "origin_key": f"{field}|{discourse_id}",
            "chunk_index": chunk_index,
            "source": "llm_relation_extractor",
            "source_context": source_context,
        }
        return {
            "uri": uri, "source": source, "relation": relation, "target": target,
            "confidence": 0.9, "is_inferred": False,
            "target_literal": target, "target_datatype": None, "target_is_literal": True,
            "object_type": "string",
            "qualifiers": {}, "predicate_text": relation, "predicate_mapping_status": "mapped",
            "negated": False, "temporal_information": {}, "quantitative_information": {},
            "metadata": meta,
        }

    def _make_graph(self, entities, relations_by_record):
        from serializers.rdf_serializer import RDFSerializer

        reg = self._Registry(entities)
        themes = {}
        for rec_id, rels in relations_by_record.items():
            themes[rec_id] = {"entities": [], "relations": rels}
        ann = {"themes": themes, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}}
        s = RDFSerializer(annotations_dict=ann, registry=reg, namespaces=Config.NAMESPACES, prune_isolated_entities=False)
        return s.graph, s

    def _subject(self, local="subject_x"):
        return Entity(
            uri=f"https://alfie-project.eu/etd-hub#{local}", label="Some Subject",
            entity_type="Concept", confidence=0.9, metadata={},
        )


class EvidenceIdentityRegressionTests(_RDFSerializerHarness):
    def test_two_identical_texts_at_different_offsets_remain_distinct(self):
        from rdflib import Namespace, RDF
        subj = self._subject("distinct_offsets")
        ctx = "identical value appears here and identical value appears there too in this passage."
        rel_a = self._relation_dict(
            "https://kg/r/off_a", subj.uri, "hasAttribute", "identical value",
            evidence_quote="identical value", source_context=ctx, discourse_id="d1", chunk_index=0,
        )
        rel_b = self._relation_dict(
            "https://kg/r/off_b", subj.uri, "hasDescription", "identical value",
            evidence_quote="identical value", source_context=ctx, discourse_id="d1", chunk_index=1,
        )
        g, s = self._make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        evidence_nodes = list(g.subjects(RDF.type, etd.Evidence))
        # Different relations/chunk provenance -> distinct Evidence nodes,
        # even though the resolved evidenceText content is identical.
        self.assertEqual(len(evidence_nodes), 2)

    def test_task8_one_evidence_text_per_node_invariant_intact(self):
        from rdflib import Namespace, RDF
        subj = self._subject("task8_invariant")
        ctx = "**Content**: a descriptive value with \\-escaped punctuation in it."
        rel = self._relation_dict(
            "https://kg/r/t8", subj.uri, "hasDescription", "a descriptive value",
            evidence_quote="Content: a descriptive value with -escaped punctuation in it", source_context=ctx,
        )
        g, s = self._make_graph([subj], {"t1": [rel]})
        etd = Namespace(Config.NAMESPACES["etd"])
        for ev in g.subjects(RDF.type, etd.Evidence):
            texts = list(g.objects(ev, etd.evidenceText))
            self.assertEqual(len(texts), 1)

    def test_task98_duplicate_consolidation_still_preserves_evidence_correctly(self):
        from rdflib import Namespace, RDF
        subj = self._subject("task98_dedup")
        ctx = "**Content**: a shared literal target described in this passage of the source."
        rel_a = self._relation_dict(
            "https://kg/r/dup_a", subj.uri, "hasDescription", "a shared literal target",
            evidence_quote="Content: a shared literal target", source_context=ctx,
        )
        rel_b = self._relation_dict(
            "https://kg/r/dup_b", subj.uri, "hasDescription", "a shared literal target",
            evidence_quote="described in this passage of the source", source_context=ctx,
        )
        g, s = self._make_graph([subj], {"t1": [rel_a, rel_b]})
        etd = Namespace(Config.NAMESPACES["etd"])
        relation_nodes = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(relation_nodes), 1)
        evidence_nodes = list(g.objects(relation_nodes[0], etd.hasEvidence))
        self.assertEqual(len(evidence_nodes), 2)
        for ev in evidence_nodes:
            texts = list(g.objects(ev, etd.evidenceText))
            self.assertEqual(len(texts), 1)
            cstart = list(g.objects(ev, etd.context_char_start))
            cend = list(g.objects(ev, etd.context_char_end))
            if cstart and cend:
                self.assertEqual(ctx[int(cstart[0]):int(cend[0])], str(texts[0]))


class UnresolvedNotSerializedAsExactTests(_RDFSerializerHarness):
    def test_unresolved_candidate_flagged_with_status_metadata(self):
        from rdflib import Namespace, RDF
        subj = self._subject("unresolved_case")
        ctx = "A completely unrelated sentence describing something else in the source passage."
        rel = self._relation_dict(
            "https://kg/r/unresolved", subj.uri, "hasAttribute", "shows a distinct unrelated outcome pattern",
            evidence_quote="shows a distinct unrelated outcome pattern", source_context=ctx,
        )
        g, s = self._make_graph([subj], {"t1": [rel]})
        etd = Namespace(Config.NAMESPACES["etd"])
        for ev in g.subjects(RDF.type, etd.Evidence):
            status = [str(v) for v in g.objects(ev, etd.evidence_alignment_status)]
            self.assertEqual(status, ["unresolved"])


if __name__ == "__main__":
    unittest.main()
