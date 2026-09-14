import unittest

from rdflib import Namespace, URIRef
from rdflib.namespace import OWL, RDF, SKOS

from config import Config
from models import Entity
from serializers.rdf_serializer import RDFSerializer


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


class _BaseConfig:
    NAMESPACES = Config.NAMESPACES
    PRUNE_ISOLATED_ENTITIES = False
    MATERIALIZE_RAW_FIELDS = False
    MATERIALIZE_SENSITIVE_FIELDS = False
    MEDIA_BASE_URL = ""
    STRICT_RELATION_BACKBONE = False


class _StrictConfig(_BaseConfig):
    STRICT_RELATION_BACKBONE = True


class RelationBackboneAtomicityTests(unittest.TestCase):
    def _serialize(self, entities, relations, config_cls=None):
        cfg = config_cls or _BaseConfig
        ann = {
            "themes": {
                "d1": {
                    "entities": [],
                    "relations": relations,
                    "metadata": {},
                }
            },
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {"datasetId": "ds", "runId": "run"},
        }
        return RDFSerializer(
            annotations_dict=ann,
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            config=cfg,
        )

    def test_external_object_uri_is_replaced_by_internal_canonical_uri(self):
        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)
        wd_uri = URIRef("http://www.wikidata.org/entity/Q123332980")

        src = Entity(uri=f"{etd_base}entity_discourse_16", label="Doc 16", entity_type="DiscourseUnit", confidence=1.0)
        obj_local = Entity(
            uri=f"{etd_base}entity_algorithmic_bias",
            label="Algorithmic Bias",
            entity_type="Risk",
            confidence=1.0,
            metadata={"wikidata": {"qid": "Q123332980", "label": "Algorithmic bias"}},
        )

        rels = [
            {
                "uri": f"{etd_base}rel_causes_1",
                "source_id": src.uri,
                "relation": "causes",
                "target_id": str(wd_uri),
                "confidence": 0.9,
                "metadata": {},
            }
        ]
        s = self._serialize([src, obj_local], rels)
        g = s.graph

        # heuristic (non-authoritative) Wikidata alignment: kept as the
        # project alignment predicate + skos:closeMatch, never owl:sameAs.
        self.assertIn((URIRef(obj_local.uri), etd.wikidata, wd_uri), g)
        self.assertIn((URIRef(obj_local.uri), SKOS.closeMatch, wd_uri), g)
        self.assertNotIn((URIRef(obj_local.uri), OWL.sameAs, wd_uri), g)

        rel_nodes = [r for r in g.subjects(RDF.type, etd.Relation)]
        self.assertGreaterEqual(len(rel_nodes), 1)
        matched = 0
        for r in rel_nodes:
            pred = next(iter(g.objects(r, etd.predicate)), None)
            subj = next(iter(g.objects(r, etd.subject)), None)
            if pred == etd.causes and subj == URIRef(src.uri):
                objs = set(g.objects(r, etd.object))
                self.assertEqual(len(objs), 1)
                self.assertEqual(next(iter(objs)), URIRef(obj_local.uri))
                matched += 1
        self.assertEqual(matched, 1)

    def test_external_subject_uri_is_replaced_by_internal_canonical_uri(self):
        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)
        wd_uri = URIRef("http://www.wikidata.org/entity/Q42")

        src_local = Entity(
            uri=f"{etd_base}entity_expert_42",
            label="Expert 42",
            entity_type="Expert",
            confidence=1.0,
            metadata={"wikidata": {"qid": "Q42", "label": "Douglas Adams"}},
        )
        obj = Entity(uri=f"{etd_base}entity_answer_1", label="Answer 1", entity_type="Answer", confidence=1.0)

        rels = [
            {
                "uri": f"{etd_base}rel_supports_1",
                "source_id": str(wd_uri),
                "relation": "supports",
                "target_id": obj.uri,
                "confidence": 0.8,
                "metadata": {},
            }
        ]
        s = self._serialize([src_local, obj], rels)
        g = s.graph

        self.assertIn((URIRef(src_local.uri), etd.wikidata, wd_uri), g)
        self.assertIn((URIRef(src_local.uri), SKOS.closeMatch, wd_uri), g)
        self.assertNotIn((URIRef(src_local.uri), OWL.sameAs, wd_uri), g)

        rel_nodes = [r for r in g.subjects(RDF.type, etd.Relation)]
        self.assertGreaterEqual(len(rel_nodes), 1)
        matched = 0
        for r in rel_nodes:
            pred = next(iter(g.objects(r, etd.predicate)), None)
            obj_val = next(iter(g.objects(r, etd.object)), None)
            if pred == etd.supports and obj_val == URIRef(obj.uri):
                subs = set(g.objects(r, etd.subject))
                self.assertEqual(len(subs), 1)
                self.assertEqual(next(iter(subs)), URIRef(src_local.uri))
                matched += 1
        self.assertEqual(matched, 1)

    def test_conflicting_write_remints_relation_uri_and_spine_is_atomic(self):
        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)
        fixed_rel = f"{etd_base}rel_collision"

        src = Entity(uri=f"{etd_base}entity_src", label="Source", entity_type="Question", confidence=1.0)
        obj1 = Entity(uri=f"{etd_base}entity_obj1", label="Object 1", entity_type="Answer", confidence=1.0)
        obj2 = Entity(uri=f"{etd_base}entity_obj2", label="Object 2", entity_type="Answer", confidence=1.0)

        rels = [
            {"uri": fixed_rel, "source_id": src.uri, "relation": "uses", "target_id": obj1.uri, "confidence": 0.9, "metadata": {}},
            {"uri": fixed_rel, "source_id": src.uri, "relation": "uses", "target_id": obj2.uri, "confidence": 0.9, "metadata": {}},
        ]
        s = self._serialize([src, obj1, obj2], rels)
        g = s.graph

        relation_nodes = [r for r in g.subjects(RDF.type, etd.Relation)]
        uses_nodes = []
        for r in relation_nodes:
            p_vals = set(g.objects(r, etd.predicate))
            s_vals = set(g.objects(r, etd.subject))
            o_vals = set(g.objects(r, etd.object))
            self.assertEqual(len(s_vals), 1, f"subjectCount != 1 for {r}")
            self.assertEqual(len(p_vals), 1, f"predicateCount != 1 for {r}")
            self.assertEqual(len(o_vals), 1, f"objectCount != 1 for {r}")
            if next(iter(p_vals)) == etd.uses and next(iter(s_vals)) == URIRef(src.uri):
                uses_nodes.append(r)

        self.assertEqual(len(uses_nodes), 2)
        self.assertIn(URIRef(fixed_rel), uses_nodes)

    def test_conflicting_write_raises_in_strict_mode(self):
        etd_base = Config.NAMESPACES["etd"]
        fixed_rel = f"{etd_base}rel_collision_strict"

        src = Entity(uri=f"{etd_base}entity_src_s", label="Source", entity_type="Question", confidence=1.0)
        obj1 = Entity(uri=f"{etd_base}entity_obj1_s", label="Object 1", entity_type="Answer", confidence=1.0)
        obj2 = Entity(uri=f"{etd_base}entity_obj2_s", label="Object 2", entity_type="Answer", confidence=1.0)

        rels = [
            {"uri": fixed_rel, "source_id": src.uri, "relation": "uses", "target_id": obj1.uri, "confidence": 0.9, "metadata": {}},
            {"uri": fixed_rel, "source_id": src.uri, "relation": "uses", "target_id": obj2.uri, "confidence": 0.9, "metadata": {}},
        ]

        with self.assertRaises(RuntimeError):
            self._serialize([src, obj1, obj2], rels, config_cls=_StrictConfig)


if __name__ == "__main__":
    unittest.main()

