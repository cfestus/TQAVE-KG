import types
import unittest

from extractors.llm_relation_extractor import LLMRelationExtractor
from models import Entity
from pipeline import EnhancedSemanticPipeline


class LLMRelationExtractorSpanTests(unittest.TestCase):
    def _extractor(self) -> LLMRelationExtractor:
        # Avoid OpenAI client init in unit tests.
        return LLMRelationExtractor.__new__(LLMRelationExtractor)

    def test_single_match_maps_to_offsets(self):
        ex = self._extractor()
        text = "Alpha supports Beta."
        source = "https://alfie-project.eu/etd-hub#entity_alpha"
        target = "https://alfie-project.eu/etd-hub#entity_beta"
        rels = [
            {
                "source_id": source,
                "relation": "supports",
                "target_id": target,
                "confidence": 0.91,
                "evidence_quote": "supports",
            }
        ]
        out = ex._convert_to_relations(rels, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(out), 1)
        m = out[0].metadata
        expected = text.find("supports")
        self.assertEqual(m.get("char_start"), expected)
        self.assertEqual(m.get("char_end"), expected + len("supports"))
        self.assertEqual(m.get("start_offset"), expected)
        self.assertEqual(m.get("end_offset"), expected + len("supports"))
        self.assertEqual(m.get("groundingStatus"), "grounded")
        self.assertEqual(m.get("groundingMethod"), "evidence_quote_span")

    def test_no_match_keeps_relation_ungrounded(self):
        ex = self._extractor()
        text = "Alpha supports Beta."
        source = "https://alfie-project.eu/etd-hub#entity_alpha"
        target = "https://alfie-project.eu/etd-hub#entity_beta"
        rels = [
            {
                "source_id": source,
                "relation": "supports",
                "target_id": target,
                "confidence": 0.91,
                "evidence_quote": "not present quote",
            }
        ]
        out = ex._convert_to_relations(rels, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(out), 1)
        m = out[0].metadata
        self.assertEqual(m.get("char_start"), -1)
        self.assertEqual(m.get("char_end"), -1)
        self.assertEqual(m.get("groundingStatus"), "ungrounded")
        self.assertEqual(m.get("groundingMethod"), "none")

    def test_multiple_match_with_mentions_chooses_closest_to_pair_midpoint(self):
        ex = self._extractor()
        text = "Alpha supports Beta. Alpha supports Beta."
        source = "https://alfie-project.eu/etd-hub#entity_alpha"
        target = "https://alfie-project.eu/etd-hub#entity_beta"
        second_alpha = text.rfind("Alpha")
        second_beta = text.rfind("Beta")
        source_ent = Entity(
            uri=source,
            label="Alpha",
            entity_type="Entity",
            confidence=1.0,
            metadata={"mentions": [{"discourse_id": "d1", "chunk_index": 0, "start_char": second_alpha, "end_char": second_alpha + 5}]},
        )
        target_ent = Entity(
            uri=target,
            label="Beta",
            entity_type="Entity",
            confidence=1.0,
            metadata={"mentions": [{"discourse_id": "d1", "chunk_index": 0, "start_char": second_beta, "end_char": second_beta + 4}]},
        )
        rels = [
            {
                "source_id": source,
                "relation": "supports",
                "target_id": target,
                "confidence": 0.93,
                "evidence_quote": "supports",
            }
        ]
        out = ex._convert_to_relations(rels, discourse_id="d1", text=text, entities=[source_ent, target_ent])
        self.assertEqual(len(out), 1)
        expected = text.rfind("supports")
        self.assertEqual(out[0].metadata.get("char_start"), expected)

    def test_multiple_match_without_mentions_chooses_first(self):
        ex = self._extractor()
        text = "Alpha supports Beta. Alpha supports Beta."
        source = "https://alfie-project.eu/etd-hub#entity_alpha"
        target = "https://alfie-project.eu/etd-hub#entity_beta"
        rels = [
            {
                "source_id": source,
                "relation": "supports",
                "target_id": target,
                "confidence": 0.93,
                "evidence_quote": "supports",
            }
        ]
        out = ex._convert_to_relations(rels, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].metadata.get("char_start"), text.find("supports"))

    def test_strict_gate_passes_with_quote_grounded_relation(self):
        ex = self._extractor()
        text = "Alpha supports Beta."
        source = "https://alfie-project.eu/etd-hub#entity_alpha"
        target = "https://alfie-project.eu/etd-hub#entity_beta"
        rels = [
            {
                "source_id": source,
                "relation": "supports",
                "target_id": target,
                "confidence": 0.93,
                "evidence_quote": "supports",
            }
        ]
        converted = ex._convert_to_relations(rels, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(converted), 1)
        rel = converted[0]

        p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        p.config = types.SimpleNamespace(
            STRICT_PROVENANCE_GROUNDING=True,
            MAX_UNKNOWN_CHUNK_INDEX_PCT=0.0,
        )
        p._discourse_chunk_spans = {
            "d1": [{"discourse_id": "d1", "chunk_index": 0, "start_offset": 0, "end_offset": len(text)}]
        }
        p._provenance_diagnostics = {}
        p.global_registry = types.SimpleNamespace(entities={}, get_all_entities=lambda: [])

        annotations = {
            "themes": {"d1": {"relations": [rel.to_dict()]}},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {},
        }
        stats = p.fill_missing_relation_provenance(annotations)
        self.assertTrue(bool(stats.get("gate_passed")))
        self.assertEqual(int(stats.get("unknown_chunk_index_count", 1)), 0)
        self.assertAlmostEqual(float(stats.get("unknown_chunk_index_rate", 1.0)), 0.0, places=6)

    def test_low_confidence_related_to_is_not_dropped_before_validation(self):
        ex = self._extractor()
        text = "Alpha is related to Beta in the same sentence."
        source = "https://alfie-project.eu/etd-hub#entity_alpha"
        target = "https://alfie-project.eu/etd-hub#entity_beta"
        rels = [
            {
                "source_id": source,
                "relation": "relatedTo",
                "target_id": target,
                "confidence": 0.41,
                "evidence_quote": "related to",
            }
        ]
        out = ex._convert_to_relations(rels, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].relation, "relatedTo")
        self.assertAlmostEqual(out[0].confidence, 0.41, places=6)

    def test_empty_endpoint_payload_is_skipped_before_relation_construction(self):
        ex = self._extractor()
        text = "MIMIC-CXR Database contains 377,110 chest X-ray images."
        rels = [
            {
                "subject_text": "   ",
                "predicate": "imageCount",
                "object_type": "count",
                "target_literal": "",
                "object_text": "   ",
                "confidence": 0.77,
            },
            {
                "subject_text": "MIMIC-CXR Database",
                "predicate": "imageCount",
                "object_type": "count",
                "target_literal": "377110",
                "confidence": 0.91,
            },
        ]
        out = ex._convert_to_relations(rels, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "MIMIC-CXR Database")
        self.assertEqual(out[0].target, "377110")
        self.assertTrue(out[0].target_is_literal)


if __name__ == "__main__":
    unittest.main()
