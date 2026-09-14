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


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for alias/label modeling test")
class AliasLabelModelingTests(unittest.TestCase):
    def test_canonical_label_with_alt_and_hidden_aliases(self):
        from rdflib import Namespace, URIRef
        from rdflib.namespace import RDFS, SKOS
        from serializers.rdf_serializer import RDFSerializer

        etd = Namespace(Config.NAMESPACES["etd"])
        ent = Entity(
            uri=f"{etd}entity_alias",
            label="Apple Inc",
            entity_type="Organization",
            confidence=0.95,
            metadata={
                "aliases": ["Apple Incorporated", "Apple, Inc.", "APPLE INC"],
                "normalized_label": "apple inc",
                "surface_form": "Apple INc",
            },
        )

        ann = {
            "themes": {"t1": {"entities": [], "relations": []}},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
        }
        serializer = RDFSerializer(
            annotations_dict=ann,
            registry=_Registry([ent]),
            namespaces=Config.NAMESPACES,
            prune_isolated_entities=False,
        )
        g = serializer.graph
        node = URIRef(ent.uri)

        labels = list(g.objects(node, RDFS.label))
        self.assertEqual(len(labels), 1)
        self.assertEqual(str(labels[0]), "Apple Inc")
        self.assertEqual(labels[0].language, "en")

        alt_labels = {str(v) for v in g.objects(node, SKOS.altLabel)}
        hidden_labels = {str(v) for v in g.objects(node, SKOS.hiddenLabel)}

        self.assertIn("Apple Incorporated", alt_labels)
        self.assertIn("Apple, Inc.", hidden_labels)
        self.assertIn("apple inc", hidden_labels)
        self.assertIn("Apple INc", hidden_labels)


if __name__ == "__main__":
    unittest.main()

