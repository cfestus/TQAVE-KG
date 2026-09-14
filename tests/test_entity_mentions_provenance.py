import types
import unittest

from models import Entity, Relation
from pipeline import EnhancedSemanticPipeline
from utils.helpers import GlobalEntityRegistry


class EntityMentionsProvenanceTests(unittest.TestCase):
    def _stub_pipeline(self, strict=False):
        p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        p.config = types.SimpleNamespace(
            STRICT_PROVENANCE_GROUNDING=bool(strict),
            MAX_UNKNOWN_CHUNK_INDEX_PCT=0.0 if strict else 1.0,
        )
        p._discourse_chunk_spans = {}
        p._provenance_diagnostics = {}
        return p

    def test_registry_merge_preserves_mentions_across_discourses(self):
        reg = GlobalEntityRegistry()
        uri = "https://alfie-project.eu/etd-hub#entity_x"
        e1 = Entity(
            uri=uri,
            label="X",
            entity_type="Entity",
            confidence=0.8,
            metadata={
                "mentions": [
                    {"discourse_id": "d1", "chunk_index": 0, "start_char": 2, "end_char": 5}
                ]
            },
        )
        e2 = Entity(
            uri=uri,
            label="X entity",
            entity_type="Entity",
            confidence=0.9,
            metadata={
                "mentions": [
                    {"discourse_id": "d1", "chunk_index": 0, "start_char": 2, "end_char": 5},
                    {"discourse_id": "d2", "chunk_index": 1, "start_char": 20, "end_char": 28},
                    {"discourse_id": "", "chunk_index": 0, "start_char": 1, "end_char": 2},
                ]
            },
        )
        reg.get_or_create(e1)
        merged = reg.get_or_create(e2)
        mentions = merged.metadata.get("mentions", [])
        self.assertEqual(len(mentions), 2)
        self.assertEqual(
            mentions,
            [
                {"discourse_id": "d1", "chunk_index": 0, "start_char": 2, "end_char": 5},
                {"discourse_id": "d2", "chunk_index": 1, "start_char": 20, "end_char": 28},
            ],
        )

    def test_discourse_matching_fallback_uses_mentions(self):
        p = self._stub_pipeline()
        s_uri = "https://alfie-project.eu/etd-hub#entity_s"
        t_uri = "https://alfie-project.eu/etd-hub#entity_t"
        s_ent = Entity(
            uri=s_uri,
            label="S",
            entity_type="Entity",
            confidence=1.0,
            metadata={
                "mentions": [
                    {"discourse_id": "d1", "chunk_index": 0, "start_char": 1, "end_char": 4},
                    {"discourse_id": "d2", "chunk_index": 0, "start_char": 10, "end_char": 15},
                ]
            },
        )
        t_ent = Entity(
            uri=t_uri,
            label="T",
            entity_type="Entity",
            confidence=1.0,
            metadata={
                "mentions": [
                    {"discourse_id": "d1", "chunk_index": 0, "start_char": 6, "end_char": 8},
                    {"discourse_id": "d2", "chunk_index": 0, "start_char": 20, "end_char": 24},
                ]
            },
        )
        p.global_registry = types.SimpleNamespace(
            entities={s_uri: s_ent, t_uri: t_ent},
            get_all_entities=lambda: [s_ent, t_ent],
        )

        self.assertEqual(p._entity_span_from_uri(s_uri, "d2"), (10, 15))

        rel = Relation(
            uri="https://alfie-project.eu/etd-hub#rel_mentions",
            source=s_uri,
            relation="mentions",
            target=t_uri,
            confidence=0.8,
            metadata={"discourse_id": "d2", "evidence": "test"},
        )
        spans = [{"discourse_id": "d2", "chunk_index": 0, "start_offset": 0, "end_offset": 100}]
        counters = {
            "missing_span_offsets": 0,
            "derived_span_from_entities": 0,
            "span_out_of_range": 0,
            "cross_chunk_evidence": 0,
            "unknown_chunk_index_count": 0,
        }
        p._ensure_relation_chunk_provenance(rel, fallback_discourse_id="d2", chunk_spans=spans, counters=counters)
        self.assertEqual(rel.metadata.get("char_start"), 10)
        self.assertEqual(rel.metadata.get("char_end"), 24)
        self.assertEqual(rel.metadata.get("chunk_index"), 0)
        self.assertEqual(rel.metadata.get("groundingMethod"), "span_map")

    def test_strict_gate_unknown_rate_zero_with_mention_fallback(self):
        p = self._stub_pipeline(strict=True)
        s_uri = "https://alfie-project.eu/etd-hub#entity_a"
        t_uri = "https://alfie-project.eu/etd-hub#entity_b"
        s_ent = Entity(
            uri=s_uri,
            label="A",
            entity_type="Entity",
            confidence=1.0,
            metadata={"mentions": [{"discourse_id": "d1", "chunk_index": 0, "start_char": 12, "end_char": 18}]},
        )
        t_ent = Entity(
            uri=t_uri,
            label="B",
            entity_type="Entity",
            confidence=1.0,
            metadata={"mentions": [{"discourse_id": "d1", "chunk_index": 0, "start_char": 22, "end_char": 30}]},
        )
        p.global_registry = types.SimpleNamespace(
            entities={s_uri: s_ent, t_uri: t_ent},
            get_all_entities=lambda: [s_ent, t_ent],
        )
        p._discourse_chunk_spans = {
            "d1": [{"discourse_id": "d1", "chunk_index": 0, "start_offset": 0, "end_offset": 200}]
        }
        annotations = {
            "themes": {
                "d1": {
                    "relations": [
                        {
                            "source": s_uri,
                            "relation": "mentions",
                            "target": t_uri,
                            "metadata": {"discourse_id": "d1", "extractor": "llm_relation_extractor"},
                        }
                    ]
                }
            },
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {},
        }

        stats = p.fill_missing_relation_provenance(annotations)
        self.assertTrue(bool(stats.get("gate_passed")))
        self.assertEqual(int(stats.get("extracted_relations_checked", 0)), 1)
        self.assertEqual(int(stats.get("unknown_chunk_index_count", 0)), 0)
        self.assertAlmostEqual(float(stats.get("unknown_chunk_index_rate", 1.0)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

