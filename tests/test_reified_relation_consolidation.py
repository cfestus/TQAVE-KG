import unittest

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from serializers.rdf_serializer import consolidate_duplicate_reified_relations


class ReifiedRelationConsolidationTests(unittest.TestCase):
    def test_duplicate_spo_consolidation_preserves_evidence(self):
        etd = Namespace("https://alfie-project.eu/etd-hub#")
        g = Graph()

        s = URIRef(f"{etd}entity_s")
        p = URIRef(f"{etd}supports")
        o = URIRef(f"{etd}entity_o")

        r1 = URIRef(f"{etd}rel_a")
        r2 = URIRef(f"{etd}rel_b")
        ev1 = URIRef(f"{etd}evidence_1")
        ev2 = URIRef(f"{etd}evidence_2")

        for rel, conf, ev in ((r1, 0.9, ev1), (r2, 0.7, ev2)):
            g.add((rel, RDF.type, etd.Relation))
            g.add((rel, etd.subject, s))
            g.add((rel, etd.predicate, p))
            g.add((rel, etd.object, o))
            g.add((rel, etd.confidence, Literal(conf, datatype=XSD.float)))
            g.add((rel, etd.isInferred, Literal(False, datatype=XSD.boolean)))
            g.add((rel, etd.hasEvidence, ev))

        metrics = consolidate_duplicate_reified_relations(g, etd)
        self.assertEqual(metrics["duplicate_spo_groups_count"], 1)
        self.assertEqual(metrics["relations_removed_count"], 1)

        # The surviving node of an actual duplicate-merge
        # group is re-minted onto a URI computed from the canonical
        # (discourse_scope, subject, predicate, object) identity rather than
        # kept at whichever raw occurrence's URI happened to win the
        # confidence tie-break -- so neither raw r1 nor r2 need survive
        # literally; exactly one Relation node must remain, carrying the
        # union of both occurrences' evidence.
        remaining = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(remaining), 1)
        winner = remaining[0]
        self.assertFalse((r1, RDF.type, etd.Relation) in g)
        self.assertFalse((r2, RDF.type, etd.Relation) in g)

        evidence = set(g.objects(winner, etd.hasEvidence))
        self.assertEqual(evidence, {ev1, ev2}, "Winner must keep union of evidence links")

    def test_tie_break_prefers_asserted_then_lexicographic_uri(self):
        etd = Namespace("https://alfie-project.eu/etd-hub#")
        g = Graph()

        s = URIRef(f"{etd}entity_s")
        p = URIRef(f"{etd}supports")
        o = URIRef(f"{etd}entity_o")

        r_true = URIRef(f"{etd}rel_true")
        r_false = URIRef(f"{etd}rel_false")
        for rel, inferred in ((r_true, True), (r_false, False)):
            g.add((rel, RDF.type, etd.Relation))
            g.add((rel, etd.subject, s))
            g.add((rel, etd.predicate, p))
            g.add((rel, etd.object, o))
            g.add((rel, etd.confidence, Literal(0.5, datatype=XSD.float)))
            g.add((rel, etd.isInferred, Literal(inferred, datatype=XSD.boolean)))

        metrics = consolidate_duplicate_reified_relations(g, etd)
        self.assertEqual(metrics["duplicate_spo_groups_count"], 1)
        # Tie-break still prefers the non-inferred occurrence's metadata to
        # survive onto the (now re-minted) surviving node.
        remaining = list(g.subjects(RDF.type, etd.Relation))
        self.assertEqual(len(remaining), 1)
        self.assertEqual(list(g.objects(remaining[0], etd.isInferred))[0].toPython(), False)

        # second group: identical confidence/inferred (true tie) -- result
        # must still be deterministic regardless of extraction order.
        def _build(order):
            gx = Graph()
            for rel in order:
                gx.add((rel, RDF.type, etd.Relation))
                gx.add((rel, etd.subject, s))
                gx.add((rel, etd.predicate, p))
                gx.add((rel, etd.object, o))
                gx.add((rel, etd.confidence, Literal(0.5, datatype=XSD.float)))
                gx.add((rel, etd.isInferred, Literal(False, datatype=XSD.boolean)))
            consolidate_duplicate_reified_relations(gx, etd)
            return sorted(str(u) for u in gx.subjects(RDF.type, etd.Relation))

        r_a = URIRef(f"{etd}rel_a")
        r_b = URIRef(f"{etd}rel_b")
        result_ab = _build((r_a, r_b))
        result_ba = _build((r_b, r_a))
        self.assertEqual(len(result_ab), 1)
        self.assertEqual(result_ab, result_ba)


if __name__ == "__main__":
    unittest.main()

