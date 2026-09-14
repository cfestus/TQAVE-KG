import unittest

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from serializers.rdf_serializer import enforce_single_en_labels


class LabelEnforcementTests(unittest.TestCase):
    def setUp(self):
        self.etd = Namespace("https://alfie-project.eu/etd-hub#")

    def test_missing_label_generates_en_label(self):
        g = Graph()
        e = URIRef(f"{self.etd}entity_1")
        g.add((e, RDF.type, self.etd.Entity))

        metrics = enforce_single_en_labels(g, self.etd)
        self.assertGreaterEqual(metrics["labels_generated"], 1)

        labels = list(g.objects(e, RDFS.label))
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].language, "en")

    def test_string_label_converted_to_en(self):
        g = Graph()
        e = URIRef(f"{self.etd}entity_2")
        g.add((e, RDF.type, self.etd.Entity))
        g.add((e, RDFS.label, Literal("Entity Two", datatype=XSD.string)))

        metrics = enforce_single_en_labels(g, self.etd)
        self.assertGreaterEqual(metrics["labels_converted_to_en"], 1)

        labels = list(g.objects(e, RDFS.label))
        self.assertEqual(len(labels), 1)
        self.assertEqual(str(labels[0]), "Entity Two")
        self.assertEqual(labels[0].language, "en")

    def test_multiple_labels_reduced_deterministically(self):
        g = Graph()
        e = URIRef(f"{self.etd}entity_3")
        g.add((e, RDF.type, self.etd.Entity))
        g.add((e, RDFS.label, Literal("Beta", lang="en")))
        g.add((e, RDFS.label, Literal("Alpha", lang="en")))
        g.add((e, RDFS.label, Literal("Gamma", datatype=XSD.string)))

        metrics = enforce_single_en_labels(g, self.etd)
        self.assertGreaterEqual(metrics["labels_deduped"], 1)

        labels = list(g.objects(e, RDFS.label))
        self.assertEqual(len(labels), 1)
        self.assertEqual(str(labels[0]), "Alpha")
        self.assertEqual(labels[0].language, "en")

    def test_relation_label_uses_spo_localnames(self):
        g = Graph()
        rel = URIRef(f"{self.etd}rel_1")
        s = URIRef(f"{self.etd}source_a")
        p = URIRef(f"{self.etd}supports")
        o = URIRef(f"{self.etd}target_b")

        g.add((rel, RDF.type, self.etd.Relation))
        g.add((rel, self.etd.subject, s))
        g.add((rel, self.etd.predicate, p))
        g.add((rel, self.etd.object, o))

        enforce_single_en_labels(g, self.etd)

        labels = list(g.objects(rel, RDFS.label))
        self.assertEqual(len(labels), 1)
        self.assertEqual(str(labels[0]), "Relation: supports (source_a -> target_b)")
        self.assertEqual(labels[0].language, "en")

    def test_vote_label_from_reified_edges_and_evidence_clamp(self):
        g = Graph()

        vote = URIRef(f"{self.etd}vote_1")
        expert = URIRef(f"{self.etd}expert_1")
        answer = URIRef(f"{self.etd}answer_1")
        rel_cast = URIRef(f"{self.etd}rel_cast")
        rel_recv = URIRef(f"{self.etd}rel_recv")
        ev = URIRef(f"{self.etd}evidence_1")

        g.add((vote, RDF.type, self.etd.Vote))
        g.add((expert, RDF.type, self.etd.Expert))
        g.add((answer, RDF.type, self.etd.Answer))

        g.add((rel_cast, RDF.type, self.etd.Relation))
        g.add((rel_cast, self.etd.subject, vote))
        g.add((rel_cast, self.etd.predicate, self.etd.castBy))
        g.add((rel_cast, self.etd.object, expert))

        g.add((rel_recv, RDF.type, self.etd.Relation))
        g.add((rel_recv, self.etd.subject, vote))
        g.add((rel_recv, self.etd.predicate, self.etd.receivesVote))
        g.add((rel_recv, self.etd.object, answer))

        g.add((ev, RDF.type, self.etd.Evidence))
        g.add((ev, self.etd.source, Literal("https://very.long.source.example.com/path/that/should/not/dominate/label")))
        g.add((ev, self.etd.snippet_id, Literal("snippet-123")))

        enforce_single_en_labels(g, self.etd)

        vote_labels = list(g.objects(vote, RDFS.label))
        self.assertEqual(len(vote_labels), 1)
        self.assertEqual(str(vote_labels[0]), "Vote by expert_1 for answer_1")

        ev_labels = list(g.objects(ev, RDFS.label))
        self.assertEqual(len(ev_labels), 1)
        self.assertTrue(str(ev_labels[0]).startswith("Evidence: snippet-123"))
        self.assertLessEqual(len(str(ev_labels[0])), 80)


if __name__ == "__main__":
    unittest.main()

