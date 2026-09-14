import importlib.util
import unittest

from config import Config
from models import Entity, Relation
from reasoning.reasoner import MultiHopReasoner


HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for inference-rule serialization test")
class ReasonerInferenceRuleTests(unittest.TestCase):
    def test_inferred_relations_have_stable_rule_id_and_no_unspecified_fallback(self):
        from rdflib import Literal, Namespace, RDF
        from rdflib.namespace import XSD
        from serializers.rdf_serializer import RDFSerializer

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        a = f"{etd_base}entity_a"
        b = f"{etd_base}entity_b"
        c = f"{etd_base}entity_c"

        base_relations = [
            Relation(source=a, relation="uses", target=b, confidence=0.9, metadata={"discourse_id": "d1", "field": "answer.description"}),
            Relation(source=b, relation="uses", target=c, confidence=0.9, metadata={"discourse_id": "d1", "field": "answer.description"}),
        ]

        reasoner = MultiHopReasoner(min_confidence=0.5, max_iterations=2, debug=False)
        inferred = reasoner.infer_relations(base_relations)
        self.assertGreaterEqual(len(inferred), 1, "Expected at least one inferred relation")
        self.assertTrue(all(r.metadata.get("inference_rule") for r in inferred))

        entities = [
            Entity(uri=a, label="A", entity_type="Entity", confidence=1.0, metadata={}),
            Entity(uri=b, label="B", entity_type="Entity", confidence=1.0, metadata={}),
            Entity(uri=c, label="C", entity_type="Entity", confidence=1.0, metadata={}),
        ]

        ann = {
            "themes": {},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {"inferred_relations": [r.to_dict() for r in inferred]},
        }

        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph

        inferred_nodes = [
            n for n in g.subjects(RDF.type, etd.Relation)
            if (n, etd.isInferred, Literal(True, datatype=XSD.boolean)) in g
        ]
        self.assertGreaterEqual(len(inferred_nodes), 1)

        unspecified = Literal("unspecified_inference_rule", datatype=XSD.string)
        for n in inferred_nodes:
            self.assertNotIn((n, etd.inferenceRule, unspecified), g)
            rules = list(g.objects(n, etd.inferenceRule))
            self.assertGreaterEqual(len(rules), 1, f"Inferred relation missing inferenceRule: {n}")


if __name__ == "__main__":
    unittest.main()
