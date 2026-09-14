import unittest

from rdflib import Namespace, URIRef
from rdflib.namespace import OWL, RDF

from config import Config
from models import Entity
from serializers.rdf_serializer import RDFSerializer
from utils.entity_normalizer import EntityNormalizer


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


class _Config:
    NAMESPACES = Config.NAMESPACES
    PRUNE_ISOLATED_ENTITIES = False
    MATERIALIZE_RAW_FIELDS = False
    MATERIALIZE_SENSITIVE_FIELDS = False
    MEDIA_BASE_URL = ""


class ClassNameCanonicalizationTests(unittest.TestCase):
    def test_entity_normalizer_type_aliases_map_to_canonical(self):
        n = EntityNormalizer()
        self.assertEqual(n.normalize_type("AiSystem"), "AISystem")
        self.assertEqual(n.normalize_type("Aisystem"), "AISystem")
        self.assertEqual(n.normalize_type("AISystem"), "AISystem")
        self.assertEqual(n.normalize_type("Algorithmicbias"), "AlgorithmicBias")
        self.assertEqual(n.normalize_type("Personaldata"), "PersonalData")

    def test_variant_entity_types_are_canonicalized(self):
        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)

        entities = [
            Entity(uri=f"{etd_base}e_ai_1", label="AI one", entity_type="AiSystem", confidence=1.0),
            Entity(uri=f"{etd_base}e_ai_2", label="AI two", entity_type="Aisystem", confidence=1.0),
            Entity(uri=f"{etd_base}e_ab_1", label="Algo bias", entity_type="Algorithmicbias", confidence=1.0),
            Entity(uri=f"{etd_base}e_pd_1", label="Personal data", entity_type="Personaldata", confidence=1.0),
        ]

        ann = {
            "themes": {},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {"datasetId": "d1", "runId": "r1"},
        }

        s = RDFSerializer(
            annotations_dict=ann,
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
            config=_Config,
        )
        g = s.graph

        # Canonical class declarations exist.
        self.assertIn((etd.AISystem, RDF.type, OWL.Class), g)
        self.assertIn((etd.AlgorithmicBias, RDF.type, OWL.Class), g)
        self.assertIn((etd.PersonalData, RDF.type, OWL.Class), g)

        # Variant class declarations must not exist.
        self.assertNotIn((etd.AiSystem, RDF.type, OWL.Class), g)
        self.assertNotIn((etd.Aisystem, RDF.type, OWL.Class), g)
        self.assertNotIn((etd.Algorithmicbias, RDF.type, OWL.Class), g)
        self.assertNotIn((etd.Personaldata, RDF.type, OWL.Class), g)

        # Instances typed only with canonical ETD class spellings.
        ai_1 = URIRef(f"{etd_base}e_ai_1")
        ai_2 = URIRef(f"{etd_base}e_ai_2")
        ab_1 = URIRef(f"{etd_base}e_ab_1")
        pd_1 = URIRef(f"{etd_base}e_pd_1")
        self.assertIn((ai_1, RDF.type, etd.AISystem), g)
        self.assertIn((ai_2, RDF.type, etd.AISystem), g)
        self.assertIn((ab_1, RDF.type, etd.AlgorithmicBias), g)
        self.assertIn((pd_1, RDF.type, etd.PersonalData), g)
        self.assertNotIn((ai_1, RDF.type, etd.AiSystem), g)
        self.assertNotIn((ai_2, RDF.type, etd.Aisystem), g)
        self.assertNotIn((ab_1, RDF.type, etd.Algorithmicbias), g)
        self.assertNotIn((pd_1, RDF.type, etd.Personaldata), g)


if __name__ == "__main__":
    unittest.main()
