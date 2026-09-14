import importlib.util
import unittest

from config import Config
from models import Entity


HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class _Registry:
    def __init__(self, entities):
        self.entities = {e.uri: e for e in entities}

    def get_all_entities(self):
        return list(self.entities.values())


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF grounding test")
class RDFGroundingTests(unittest.TestCase):
    def test_mapped_ontology_class_materialized_as_rdf_type(self):
        from rdflib import RDF, URIRef
        from serializers.rdf_serializer import RDFSerializer

        ent = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_x",
            label="Demo Model",
            entity_type="AISystem",
            ontology_uri="https://w3id.org/aipo#AISystem",
            confidence=0.9,
            metadata={"ontology_mapping_source": "unit_test", "ontology_mapping_confidence": 0.88},
        )
        reg = _Registry([ent])

        ann = {
            "themes": {"t1": {"entities": [], "relations": []}},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
        }
        s = RDFSerializer(
            annotations_dict=ann,
            registry=reg,
            namespaces=Config.NAMESPACES,
            prune_isolated_entities=False,
        )
        g = s.graph

        self.assertIn((URIRef(ent.uri), RDF.type, URIRef("https://w3id.org/aipo#AISystem")), g)

    def test_curie_ontology_uri_always_materialized_as_rdf_type(self):
        from rdflib import RDF, URIRef
        from serializers.rdf_serializer import RDFSerializer

        ent = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_y",
            label="Policy Ref",
            entity_type="PolicyFramework",
            ontology_uri="dcat:Dataset",
            confidence=0.9,
            metadata={},
        )
        reg = _Registry([ent])

        ann = {
            "themes": {"t1": {"entities": [], "relations": []}},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
        }
        s = RDFSerializer(
            annotations_dict=ann,
            registry=reg,
            namespaces=Config.NAMESPACES,
            prune_isolated_entities=False,
        )
        g = s.graph

        expected = URIRef(f"{Config.NAMESPACES['dcat']}Dataset")
        self.assertIn((URIRef(ent.uri), RDF.type, expected), g)


if __name__ == "__main__":
    unittest.main()

