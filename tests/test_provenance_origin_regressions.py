import unittest
from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import RDF

from models import Entity, Relation
from serializers.rdf_serializer import RDFSerializer, consolidate_duplicate_reified_relations
from validation.shacl_validator import SHACLValidator


class TestProvenanceOriginRegressions(unittest.TestCase):
    def setUp(self) -> None:
        self.etd = "https://alfie-project.eu/etd-hub#"
        self.namespaces = {
            "etd": self.etd,
            "prov": "http://www.w3.org/ns/prov#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dcterms": "http://purl.org/dc/terms/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "schema": "https://schema.org/",
            "org": "http://www.w3.org/ns/org#",
            "dcat": "http://www.w3.org/ns/dcat#",
            "dpv": "https://w3id.org/dpv#",
            "aipo": "https://w3id.org/aipo/ontology#",
            "airo": "https://w3id.org/airo#",
            "vair": "https://w3id.org/vair#",
            "relaieo": "https://w3id.org/relaieo#",
            "fmo": "https://w3id.org/fmo#",
            "hudock": "https://w3id.org/hudock#",
        }

    def _serializer(self, annotations):
        class _Cfg:
            NAMESPACES = self.namespaces
            PRUNE_ISOLATED_ENTITIES = False
            STRICT_ETD_PROPERTY_REGISTRY = True
            MATERIALIZE_RAW_FIELDS = False
            MATERIALIZE_SENSITIVE_FIELDS = False
            STRICT_RELATION_BACKBONE = True
            PIPELINE_VERSION = "test"

        return RDFSerializer(annotations, registry=None, namespaces=self.namespaces, config=_Cfg)

    def test_discourse_id_collision_requires_disambiguating_provenance(self):
        # Theme/Question/Answer can share discourse_id=14; relation must carry
        # additional origin information to stay unambiguous.
        theme_uri = f"{self.etd}theme_14"
        question_uri = f"{self.etd}question_14"
        answer_uri = f"{self.etd}answer_14"
        risk_uri = f"{self.etd}risk_x"

        annotations = {
            "themes": {
                "14": {
                    "entities": [
                        {"uri": theme_uri, "label": "Theme 14", "entity_type": "Theme", "metadata": {"discourse_id": "14", "field": "theme.description"}},
                        {"uri": risk_uri, "label": "Bias", "entity_type": "Risk", "metadata": {"discourse_id": "14", "field": "theme.description"}},
                    ],
                    "relations": [
                        {
                            "uri": f"{self.etd}rel_theme_14",
                            "source": theme_uri,
                            "relation": "hasRisk",
                            "target": risk_uri,
                            "confidence": 0.9,
                            "metadata": {"discourse_id": "14", "extractor": "llm_relation_extractor", "context_source": "llm_json"},
                        }
                    ],
                    "metadata": {"discourse_id": "14", "field": "theme.description"},
                }
            },
            "questions": {
                "14": {
                    "entities": [
                        {"uri": question_uri, "label": "Question 14", "entity_type": "Question", "metadata": {"discourse_id": "14", "field": "question.body"}},
                    ],
                    "relations": [],
                    "metadata": {"discourse_id": "14", "field": "question.body"},
                }
            },
            "answers": {
                "14": {
                    "entities": [
                        {"uri": answer_uri, "label": "Answer 14", "entity_type": "Answer", "metadata": {"discourse_id": "14", "field": "answer.description"}},
                    ],
                    "relations": [],
                    "metadata": {"discourse_id": "14", "field": "answer.description"},
                }
            },
            "votes": {},
            "experts": {},
            "metadata": {},
        }

        s = self._serializer(annotations)
        g = s.graph
        etd = Namespace(self.etd)
        rel = URIRef(f"{self.etd}rel_theme_14")

        # Regression expectation (currently fails):
        # relation must carry explicit field or direct origin link for disambiguation.
        has_field = any(g.objects(rel, etd.field))
        has_origin_link = any(g.objects(rel, etd.extractedFrom))
        self.assertTrue(has_field or has_origin_link)

    def test_unique_origin_completeness_for_relation_and_evidence(self):
        a = f"{self.etd}a"
        b = f"{self.etd}b"
        rel_uri = f"{self.etd}rel_origin"
        annotations = {
            "themes": {
                "1": {
                    "entities": [
                        {"uri": a, "label": "A", "entity_type": "Entity", "metadata": {"discourse_id": "1", "field": "theme.description"}},
                        {"uri": b, "label": "B", "entity_type": "Entity", "metadata": {"discourse_id": "1", "field": "theme.description"}},
                    ],
                    "relations": [
                        {
                            "uri": rel_uri,
                            "source": a,
                            "relation": "relatedTo",
                            "target": b,
                            "confidence": 0.7,
                            "metadata": {"discourse_id": "1", "extractor": "llm_relation_extractor", "context_source": "llm_json"},
                        }
                    ],
                    "metadata": {"discourse_id": "1", "field": "theme.description"},
                }
            },
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "metadata": {},
        }
        s = self._serializer(annotations)
        g = s.graph
        etd = Namespace(self.etd)
        rel = URIRef(rel_uri)
        evs = list(g.objects(rel, etd.hasEvidence))
        self.assertTrue(evs)

        # Regression expectation (currently fails):
        # each relation/evidence should map to exactly one origin IRI.
        self.assertEqual(len(list(g.objects(rel, etd.extractedFrom))), 1)
        for ev in evs:
            self.assertEqual(len(list(g.objects(ev, etd.extractedFrom))), 1)

    def test_no_cross_discourse_spo_merge(self):
        g = self._serializer({"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "metadata": {}}).graph
        etd = Namespace(self.etd)

        s = URIRef(f"{self.etd}entity_s")
        p = URIRef(f"{self.etd}relatedTo")
        o = URIRef(f"{self.etd}entity_o")

        r1 = URIRef(f"{self.etd}rel_a")
        r2 = URIRef(f"{self.etd}rel_b")

        for r, did in ((r1, 6), (r2, 11)):
            g.add((r, RDF.type, etd.Relation))
            g.add((r, etd.subject, s))
            g.add((r, etd.predicate, p))
            g.add((r, etd.object, o))
            g.add((r, etd.discourse_id, Literal(did)))

        # Regression expectation (currently fails because merge key is only SPO):
        metrics = consolidate_duplicate_reified_relations(g, etd)
        self.assertEqual(metrics["relations_removed_count"], 0)
        self.assertIn((r1, RDF.type, etd.Relation), g)
        self.assertIn((r2, RDF.type, etd.Relation), g)

    def test_synthetic_castby_provenance_not_null(self):
        # Semantic vote relation triggers synthetic castBy/reifies; provenance must never be null.
        expert_uri = f"{self.etd}expert_1"
        answer_uri = f"{self.etd}answer_1"
        annotations = {
            "votes": {
                "60": {
                    "entities": [
                        {"uri": expert_uri, "label": "Expert 1", "entity_type": "Expert", "metadata": {"discourse_id": "60"}},
                        {"uri": answer_uri, "label": "Answer 1", "entity_type": "Answer", "metadata": {"discourse_id": "60"}},
                    ],
                    "relations": [
                        {
                            "uri": f"{self.etd}rel_vote_60",
                            "source": expert_uri,
                            "relation": "endorses",
                            "target": answer_uri,
                            "confidence": 0.9,
                            "metadata": {"discourse_id": "60", "extractor": "vote", "context_source": "dataset_vote"},
                        }
                    ],
                    "metadata": {"discourse_id": "60"},
                }
            },
            "themes": {},
            "questions": {},
            "answers": {},
            "experts": {},
            "metadata": {},
        }

        s = self._serializer(annotations)
        g = s.graph
        etd = Namespace(self.etd)
        cast_by_rels = sorted(
            [r for r in g.subjects(RDF.type, etd.Relation) if any(o == etd.castBy for o in g.objects(r, etd.predicate))],
            key=str,
        )
        self.assertTrue(cast_by_rels)

        # Regression expectation (currently fails):
        for r in cast_by_rels:
            self.assertEqual(len(list(g.objects(r, etd.discourse_id))), 1)
            evs = list(g.objects(r, etd.hasEvidence))
            self.assertTrue(evs)
            for ev in evs:
                self.assertEqual(len(list(g.objects(ev, etd.discourse_id))), 1)

    def test_relation_field_is_serialized_when_present(self):
        x = f"{self.etd}x"
        y = f"{self.etd}y"
        rel_uri = f"{self.etd}rel_field"
        annotations = {
            "themes": {
                "2": {
                    "entities": [
                        {"uri": x, "label": "X", "entity_type": "Entity", "metadata": {"discourse_id": "2", "field": "theme.description"}},
                        {"uri": y, "label": "Y", "entity_type": "Entity", "metadata": {"discourse_id": "2", "field": "theme.description"}},
                    ],
                    "relations": [
                        {
                            "uri": rel_uri,
                            "source": x,
                            "relation": "relatedTo",
                            "target": y,
                            "confidence": 0.5,
                            "metadata": {"discourse_id": "2", "field": "theme.description", "extractor": "lightweight", "context_source": "lightweight_mentions"},
                        }
                    ],
                    "metadata": {"discourse_id": "2", "field": "theme.description"},
                }
            },
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "metadata": {},
        }
        s = self._serializer(annotations)
        g = s.graph
        etd = Namespace(self.etd)
        rel = URIRef(rel_uri)

        # Regression expectation (currently fails): relation-level field is serialised.
        fields = [str(v) for v in g.objects(rel, etd.field)]
        self.assertEqual(fields, ["theme.description"])


if __name__ == "__main__":
    unittest.main()
