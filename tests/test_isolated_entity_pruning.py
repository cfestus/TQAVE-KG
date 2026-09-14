import unittest

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

from serializers.rdf_serializer import prune_isolated_entities


class IsolatedEntityPruningTests(unittest.TestCase):
    def test_prune_removes_only_isolated_etd_entity_nodes(self):
        etd = Namespace("https://alfie-project.eu/etd-hub#")
        g = Graph()

        connected = URIRef(f"{etd}entity_connected")
        other = URIRef(f"{etd}entity_other")
        isolated = URIRef(f"{etd}entity_isolated")
        rel = URIRef(f"{etd}rel_1")

        # Entities
        for e, label in ((connected, "Connected"), (other, "Other"), (isolated, "Isolated")):
            g.add((e, RDF.type, etd.Entity))
            g.add((e, RDFS.label, Literal(label, lang="en")))

        # Connected relation linking two entities
        g.add((rel, RDF.type, etd.Relation))
        g.add((rel, etd.subject, connected))
        g.add((rel, etd.predicate, etd.relatedTo))
        g.add((rel, etd.object, other))

        metrics = prune_isolated_entities(g, etd)

        self.assertEqual(metrics["isolated_entities_found"], 1)
        self.assertEqual(metrics["isolated_entities_removed"], 1)
        self.assertGreater(metrics["triples_removed"], 0)

        # Connected entities remain
        self.assertTrue((connected, RDF.type, etd.Entity) in g)
        self.assertTrue((other, RDF.type, etd.Entity) in g)
        # Isolated removed entirely (subject + inbound)
        self.assertFalse((isolated, None, None) in g)
        self.assertFalse((None, None, isolated) in g)


if __name__ == "__main__":
    unittest.main()

