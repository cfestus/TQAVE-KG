import types
import unittest
import importlib.util

from rdflib import Namespace
from rdflib.namespace import RDF, XSD

from models import Entity, Relation
from pipeline import EnhancedSemanticPipeline
from serializers.rdf_serializer import RDFSerializer

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class ChunkProvenanceHardeningTests(unittest.TestCase):
    def _stub_pipeline(self):
        p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        p.config = types.SimpleNamespace(
            STRICT_PROVENANCE_GROUNDING=False,
            MAX_UNKNOWN_CHUNK_INDEX_PCT=0.02,
        )
        p._discourse_chunk_spans = {}
        p._provenance_diagnostics = {}
        return p

    def test_span_to_chunk_mapping_sets_grounded_chunk_index(self):
        p = self._stub_pipeline()
        rel = Relation(
            uri="https://alfie-project.eu/etd-hub#rel_t1",
            source="https://alfie-project.eu/etd-hub#entity_a",
            relation="mentions",
            target="https://alfie-project.eu/etd-hub#entity_b",
            confidence=0.9,
            metadata={
                "discourse_id": "d1",
                "char_start": 5,
                "char_end": 9,
                "evidence": "sample",
            },
        )
        spans = [
            {"discourse_id": "d1", "chunk_index": 0, "start_offset": 0, "end_offset": 10},
            {"discourse_id": "d1", "chunk_index": 1, "start_offset": 10, "end_offset": 20},
        ]
        counters = {
            "missing_span_offsets": 0,
            "span_out_of_range": 0,
            "cross_chunk_evidence": 0,
            "unknown_chunk_index_count": 0,
        }
        p._ensure_relation_chunk_provenance(rel, fallback_discourse_id="d1", chunk_spans=spans, counters=counters)
        self.assertEqual(rel.metadata.get("chunk_index"), 0)
        self.assertFalse(rel.metadata.get("chunk_index_unknown"))
        self.assertEqual(rel.metadata.get("groundingStatus"), "grounded")
        self.assertEqual(rel.metadata.get("groundingMethod"), "span_map")

    def test_cross_chunk_span_creates_evidence_spans_and_unknown_relation_chunk(self):
        p = self._stub_pipeline()
        rel = Relation(
            uri="https://alfie-project.eu/etd-hub#rel_t2",
            source="https://alfie-project.eu/etd-hub#entity_a",
            relation="mentions",
            target="https://alfie-project.eu/etd-hub#entity_b",
            confidence=0.9,
            metadata={
                "discourse_id": "d2",
                "char_start": 5,
                "char_end": 15,
                "evidence": "cross span",
            },
        )
        spans = [
            {"discourse_id": "d2", "chunk_index": 0, "start_offset": 0, "end_offset": 10},
            {"discourse_id": "d2", "chunk_index": 1, "start_offset": 10, "end_offset": 20},
        ]
        counters = {
            "missing_span_offsets": 0,
            "span_out_of_range": 0,
            "cross_chunk_evidence": 0,
            "unknown_chunk_index_count": 0,
        }
        p._ensure_relation_chunk_provenance(rel, fallback_discourse_id="d2", chunk_spans=spans, counters=counters)
        self.assertEqual(rel.metadata.get("groundingMethod"), "span_map")
        self.assertEqual(rel.metadata.get("groundingStatus"), "grounded")
        self.assertEqual(rel.metadata.get("chunk_index"), -1)
        self.assertTrue(rel.metadata.get("chunk_index_unknown"))
        ev_spans = rel.metadata.get("evidence_spans", [])
        self.assertEqual(len(ev_spans), 2)
        self.assertEqual(sorted(int(s["chunk_index"]) for s in ev_spans), [0, 1])
        self.assertEqual(counters.get("cross_chunk_evidence"), 1)

    def test_entity_span_fallback_derives_relation_span_and_grounds(self):
        p = self._stub_pipeline()
        s_uri = "https://alfie-project.eu/etd-hub#entity_s"
        t_uri = "https://alfie-project.eu/etd-hub#entity_t"
        s_ent = Entity(
            uri=s_uri,
            label="S",
            entity_type="Entity",
            confidence=1.0,
            metadata={"start_char": 2, "end_char": 5},
        )
        t_ent = Entity(
            uri=t_uri,
            label="T",
            entity_type="Entity",
            confidence=1.0,
            metadata={"start_char": 6, "end_char": 9},
        )
        p.global_registry = types.SimpleNamespace(
            entities={s_uri: s_ent, t_uri: t_ent},
            get_all_entities=lambda: [s_ent, t_ent],
        )
        rel = Relation(
            uri="https://alfie-project.eu/etd-hub#rel_t_fallback",
            source=s_uri,
            relation="mentions",
            target=t_uri,
            confidence=0.9,
            metadata={"discourse_id": "d_fallback", "evidence": "fallback"},
        )
        spans = [{"discourse_id": "d_fallback", "chunk_index": 0, "start_offset": 0, "end_offset": 20}]
        counters = {
            "missing_span_offsets": 0,
            "derived_span_from_entities": 0,
            "span_out_of_range": 0,
            "cross_chunk_evidence": 0,
            "unknown_chunk_index_count": 0,
        }
        p._ensure_relation_chunk_provenance(rel, fallback_discourse_id="d_fallback", chunk_spans=spans, counters=counters)
        self.assertEqual(rel.metadata.get("char_start"), 2)
        self.assertEqual(rel.metadata.get("char_end"), 9)
        self.assertEqual(rel.metadata.get("chunk_index"), 0)
        self.assertFalse(rel.metadata.get("chunk_index_unknown"))
        self.assertEqual(rel.metadata.get("groundingMethod"), "span_map")
        self.assertEqual(counters.get("derived_span_from_entities"), 1)

    def test_fill_missing_relation_provenance_grounds_from_evidence_quote(self):
        p = self._stub_pipeline()
        p.global_registry = types.SimpleNamespace(entities={}, get_all_entities=lambda: [])
        text = "Alpha supports Beta in a local proposition."
        annotations = {
            "themes": {
                "d_quote": {
                    "entities": [],
                    "relations": [
                        {
                            "uri": "https://alfie-project.eu/etd-hub#rel_quote_grounded",
                            "source": "https://alfie-project.eu/etd-hub#entity_alpha",
                            "relation": "supports",
                            "target": "https://alfie-project.eu/etd-hub#entity_beta",
                            "confidence": 0.8,
                            "metadata": {
                                "discourse_id": "d_quote",
                                "field": "theme.description",
                                "source_field": "theme.description",
                                "extractor": "llm_relation_extractor",
                                "evidence_quote": "supports Beta",
                                "evidence": "supports Beta",
                            },
                        }
                    ],
                    "metadata": {"text": text},
                }
            },
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {},
        }
        p._discourse_chunk_spans = {
            "theme.description|d_quote": [
                {"discourse_id": "d_quote", "chunk_index": 0, "start_offset": 0, "end_offset": len(text)}
            ]
        }
        stats = p.fill_missing_relation_provenance(annotations)
        rel_meta = annotations["themes"]["d_quote"]["relations"][0]["metadata"]
        self.assertEqual(rel_meta.get("groundingMethod"), "span_map")
        self.assertEqual(rel_meta.get("chunk_index"), 0)
        self.assertFalse(rel_meta.get("chunk_index_unknown"))
        self.assertEqual(stats.get("unknown_chunk_index_count"), 0)

    def test_serializer_does_not_infer_chunk_index_when_missing(self):
        etd = "https://alfie-project.eu/etd-hub#"
        ETD = Namespace(etd)
        annotations = {
            "themes": {
                "d3": {
                    "entities": [
                        {"uri": f"{etd}entity_a", "label": "A", "entity_type": "Entity", "metadata": {"discourse_id": "d3"}},
                        {"uri": f"{etd}entity_b", "label": "B", "entity_type": "Entity", "metadata": {"discourse_id": "d3"}},
                    ],
                    "relations": [
                        {
                            "uri": f"{etd}rel_no_chunk",
                            "source": f"{etd}entity_a",
                            "relation": "mentions",
                            "target": f"{etd}entity_b",
                            "confidence": 0.8,
                            "metadata": {
                                "discourse_id": "d3",
                                "chunk_index": -1,
                                "chunk_index_unknown": True,
                                "groundingStatus": "ungrounded",
                                "groundingMethod": "none",
                                "evidence": "no span evidence",
                            },
                        }
                    ],
                    "metadata": {"text": "dummy"},
                }
            },
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {},
        }
        namespaces = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "prov": "http://www.w3.org/ns/prov#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "etd": etd,
            "onto": "https://alfie-project.eu/ontology#",
        }
        serializer = RDFSerializer(annotations, registry=None, namespaces=namespaces)
        g = serializer.graph
        evidence_nodes = list(g.subjects(RDF.type, ETD.Evidence))
        self.assertGreaterEqual(len(evidence_nodes), 1)
        vals = []
        for ev in evidence_nodes:
            for o in g.objects(ev, ETD.chunk_index):
                vals.append(int(str(o)))
        self.assertIn(-1, vals)

        unknown_flags = []
        for ev in evidence_nodes:
            for o in g.objects(ev, ETD.chunk_index_unknown):
                unknown_flags.append(str(o).lower() in ("true", "1"))
        self.assertIn(True, unknown_flags)

    def test_serializer_splits_cross_chunk_evidence_into_multiple_nodes(self):
        etd = "https://alfie-project.eu/etd-hub#"
        ETD = Namespace(etd)
        rel_uri = f"{etd}rel_cross_chunk"
        annotations = {
            "themes": {
                "d4": {
                    "entities": [
                        {"uri": f"{etd}entity_a", "label": "A", "entity_type": "Entity", "metadata": {"discourse_id": "d4"}},
                        {"uri": f"{etd}entity_b", "label": "B", "entity_type": "Entity", "metadata": {"discourse_id": "d4"}},
                    ],
                    "relations": [
                        {
                            "uri": rel_uri,
                            "source": f"{etd}entity_a",
                            "relation": "mentions",
                            "target": f"{etd}entity_b",
                            "confidence": 0.8,
                            "metadata": {
                                "discourse_id": "d4",
                                "chunk_index": -1,
                                "chunk_index_unknown": True,
                                "groundingStatus": "grounded",
                                "groundingMethod": "span_map",
                                "evidence": "cross chunk evidence",
                                "evidence_spans": [
                                    {"chunk_index": 0, "char_start": 5, "char_end": 10, "start_offset": 5, "end_offset": 10},
                                    {"chunk_index": 1, "char_start": 10, "char_end": 15, "start_offset": 10, "end_offset": 15},
                                ],
                            },
                        }
                    ],
                    "metadata": {"text": "dummy"},
                }
            },
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {},
        }
        namespaces = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "prov": "http://www.w3.org/ns/prov#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "etd": etd,
            "onto": "https://alfie-project.eu/ontology#",
        }
        serializer = RDFSerializer(annotations, registry=None, namespaces=namespaces)
        g = serializer.graph
        rel_node = ETD.rel_cross_chunk
        evidence_for_rel = list(g.objects(rel_node, ETD.hasEvidence))
        self.assertEqual(len(evidence_for_rel), 2)
        chunk_vals = []
        for ev in evidence_for_rel:
            for o in g.objects(ev, ETD.chunk_index):
                chunk_vals.append(int(str(o)))
        self.assertEqual(sorted(chunk_vals), [0, 1])

    def test_structural_unknown_evidence_gets_explicit_grounding_and_no_placeholder(self):
        etd = "https://alfie-project.eu/etd-hub#"
        ETD = Namespace(etd)
        annotations = {
            "votes": {
                "v1": {
                    "entities": [
                        {"uri": f"{etd}entity_a", "label": "A", "entity_type": "Entity", "metadata": {"discourse_id": "1"}},
                        {"uri": f"{etd}entity_b", "label": "B", "entity_type": "Entity", "metadata": {"discourse_id": "1"}},
                    ],
                    "relations": [
                        {
                            "uri": f"{etd}rel_structural_unknown",
                            "source": f"{etd}entity_a",
                            "relation": "mentions",
                            "target": f"{etd}entity_b",
                            "confidence": 1.0,
                            "metadata": {
                                "discourse_id": "1",
                                "field": "vote.vote_value",
                                "extractor": "structural",
                                "context_source": "dataset_vote",
                                "chunk_index": -1,
                                "chunk_index_unknown": True,
                            },
                        }
                    ],
                    "metadata": {"text": "dummy"},
                }
            },
            "themes": {},
            "questions": {},
            "answers": {},
            "experts": {},
            "documents": {},
            "metadata": {},
        }
        namespaces = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "prov": "http://www.w3.org/ns/prov#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "etd": etd,
            "onto": "https://alfie-project.eu/ontology#",
        }
        serializer = RDFSerializer(annotations, registry=None, namespaces=namespaces)
        g = serializer.graph
        evidence_nodes = list(g.subjects(RDF.type, ETD.Evidence))
        self.assertGreaterEqual(len(evidence_nodes), 1)
        ev = evidence_nodes[0]
        self.assertEqual(str(next(g.objects(ev, ETD.groundingStatus))), "ungrounded")
        self.assertEqual(str(next(g.objects(ev, ETD.groundingMethod))), "structural")
        self.assertEqual(int(str(next(g.objects(ev, ETD.chunk_index)))), -1)
        self.assertEqual(str(next(g.objects(ev, ETD.chunk_index_unknown))).lower(), "true")
        evidence_text = str(next(g.objects(ev, ETD.evidenceText)))
        self.assertNotIn("Evidence unavailable", evidence_text)

    def test_chunk_grounded_evidence_clears_unknown_flag(self):
        etd = "https://alfie-project.eu/etd-hub#"
        ETD = Namespace(etd)
        annotations = {
            "answers": {
                "a1": {
                    "entities": [
                        {"uri": f"{etd}entity_a", "label": "A", "entity_type": "Entity", "metadata": {"discourse_id": "1"}},
                        {"uri": f"{etd}entity_b", "label": "B", "entity_type": "Entity", "metadata": {"discourse_id": "1"}},
                    ],
                    "relations": [
                        {
                            "uri": f"{etd}rel_chunk_grounded",
                            "source": f"{etd}entity_a",
                            "relation": "mentions",
                            "target": f"{etd}entity_b",
                            "confidence": 0.8,
                            "metadata": {
                                "discourse_id": "1",
                                "field": "answer.description",
                                "extractor": "structural",
                                "chunk_index": 0,
                                "chunk_index_unknown": True,
                                "groundingStatus": "grounded",
                                "groundingMethod": "single_chunk_discourse",
                                "char_start": -1,
                                "char_end": -1,
                            },
                        }
                    ],
                    "metadata": {"text": "dummy"},
                }
            },
            "themes": {},
            "questions": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {},
        }
        namespaces = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "prov": "http://www.w3.org/ns/prov#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "etd": etd,
            "onto": "https://alfie-project.eu/ontology#",
        }
        serializer = RDFSerializer(annotations, registry=None, namespaces=namespaces)
        g = serializer.graph
        ev = next(g.subjects(RDF.type, ETD.Evidence))
        self.assertEqual(int(str(next(g.objects(ev, ETD.chunk_index)))), 0)
        self.assertEqual(str(next(g.objects(ev, ETD.chunk_index_unknown))).lower(), "false")
        self.assertEqual(str(next(g.objects(ev, ETD.groundingStatus))), "chunk_grounded")
        self.assertEqual(str(next(g.objects(ev, ETD.groundingMethod))), "single_chunk_discourse")

    def test_gate_excludes_inferred_reasoner_relations_from_rate(self):
        p = self._stub_pipeline()
        p.config.MAX_UNKNOWN_CHUNK_INDEX_PCT = 1.0
        p._discourse_chunk_spans = {}
        p.global_registry = types.SimpleNamespace(entities={}, get_all_entities=lambda: [])

        annotations = {
            "themes": {
                "d1": {
                    "relations": [
                        {
                            "source": "https://alfie-project.eu/etd-hub#entity_a",
                            "relation": "mentions",
                            "target": "https://alfie-project.eu/etd-hub#entity_b",
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
            "metadata": {
                "inferred_relations": [
                    {
                        "source": "https://alfie-project.eu/etd-hub#entity_a",
                        "relation": "relatedTo",
                        "target": "https://alfie-project.eu/etd-hub#entity_c",
                        "is_inferred": True,
                        "metadata": {"discourse_id": "d1", "extractor": "reasoner"},
                    }
                ]
            },
        }

        stats = p.fill_missing_relation_provenance(annotations)
        self.assertEqual(stats.get("total_relations"), 2)
        self.assertEqual(stats.get("extracted_relations_checked"), 1)
        self.assertEqual(stats.get("total_relations_checked"), 1)
        self.assertEqual(stats.get("inferred_relations_excluded_from_gate"), 1)
        self.assertEqual(stats.get("unknown_chunk_index_count"), 1)
        self.assertAlmostEqual(float(stats.get("unknown_chunk_index_rate", 0.0)), 1.0, places=6)

    def test_gate_excludes_non_textual_structural_vote_lightweight_relations(self):
        p = self._stub_pipeline()
        p.config.MAX_UNKNOWN_CHUNK_INDEX_PCT = 1.0
        p._discourse_chunk_spans = {
            "d1": [{"discourse_id": "d1", "chunk_index": 0, "start_offset": 0, "end_offset": 50}]
        }
        p.global_registry = types.SimpleNamespace(entities={}, get_all_entities=lambda: [])

        annotations = {
            "themes": {
                "d1": {
                    "relations": [
                        {
                            "source": "https://alfie-project.eu/etd-hub#entity_a",
                            "relation": "mentions",
                            "target": "https://alfie-project.eu/etd-hub#entity_b",
                            "metadata": {
                                "discourse_id": "d1",
                                "extractor": "llm_relation_extractor",
                                "char_start": 5,
                                "char_end": 10,
                            },
                        }
                    ]
                }
            },
            "questions": {
                "d2": {
                    "relations": [
                        {
                            "source": "https://alfie-project.eu/etd-hub#question_1",
                            "relation": "belongsToTheme",
                            "target": "https://alfie-project.eu/etd-hub#theme_1",
                            "metadata": {"discourse_id": "d2", "extractor": "structural"},
                        }
                    ]
                }
            },
            "answers": {
                "d3": {
                    "relations": [
                        {
                            "source": "https://alfie-project.eu/etd-hub#entity_x",
                            "relation": "relatedTo",
                            "target": "https://alfie-project.eu/etd-hub#entity_y",
                            "metadata": {"discourse_id": "d3", "extractor": "lightweight"},
                        }
                    ]
                }
            },
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {},
        }

        stats = p.fill_missing_relation_provenance(annotations)
        self.assertEqual(stats.get("total_relations"), 3)
        self.assertEqual(stats.get("extracted_relations_checked"), 1)
        self.assertEqual(stats.get("total_relations_checked"), 1)
        self.assertEqual(stats.get("non_textual_relations_excluded_from_gate"), 2)
        self.assertEqual(stats.get("unknown_chunk_index_count"), 0)
        self.assertAlmostEqual(float(stats.get("unknown_chunk_index_rate", 1.0)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
