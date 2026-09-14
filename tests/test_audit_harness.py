import tempfile
import unittest
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from scripts.audit_harness import (
    _duplicate_reified_relation_spo,
    _evidence_chunk_unknown_stats,
    _isolated_entities,
    _vote_typing_issues,
    _cross_chunk_duplicate_labels,
)


class AuditHarnessTests(unittest.TestCase):
    def test_duplicate_reified_spo_detection(self):
        etd = "https://alfie-project.eu/etd-hub#"
        g = Graph()

        relation_cls = URIRef(f"{etd}Relation")
        s_pred = URIRef(f"{etd}subject")
        p_pred = URIRef(f"{etd}predicate")
        o_pred = URIRef(f"{etd}object")

        rel1 = URIRef(f"{etd}rel_1")
        rel2 = URIRef(f"{etd}rel_2")
        s = URIRef(f"{etd}entity_a")
        p = URIRef(f"{etd}supports")
        o = URIRef(f"{etd}entity_b")

        for rel in (rel1, rel2):
            g.add((rel, RDF.type, relation_cls))
            g.add((rel, s_pred, s))
            g.add((rel, p_pred, p))
            g.add((rel, o_pred, o))

        with tempfile.TemporaryDirectory() as td:
            out = _duplicate_reified_relation_spo(g, etd, Path(td))
            self.assertEqual(out["duplicate_relation_spo_groups_count"], 1)
            self.assertEqual(out["duplicate_relation_spo_total_relations_involved"], 2)
            out_file = Path(out["duplicate_relation_spo_report_path"])
            self.assertTrue(out_file.exists())

    def test_vote_castby_from_reified_only_not_missing(self):
        etd = "https://alfie-project.eu/etd-hub#"
        g = Graph()

        vote = URIRef(f"{etd}vote_1")
        expert = URIRef(f"{etd}expert_1")
        answer = URIRef(f"{etd}answer_1")

        g.add((vote, RDF.type, URIRef(f"{etd}Vote")))
        g.add((expert, RDF.type, URIRef(f"{etd}Expert")))
        g.add((answer, RDF.type, URIRef(f"{etd}Answer")))

        # castBy only reified (no direct vote castBy expert triple)
        rel_cast = URIRef(f"{etd}rel_cast")
        g.add((rel_cast, RDF.type, URIRef(f"{etd}Relation")))
        g.add((rel_cast, URIRef(f"{etd}subject"), vote))
        g.add((rel_cast, URIRef(f"{etd}predicate"), URIRef(f"{etd}castBy")))
        g.add((rel_cast, URIRef(f"{etd}object"), expert))

        # receivesVote only reified (Answer -> Vote)
        rel_recv = URIRef(f"{etd}rel_recv")
        g.add((rel_recv, RDF.type, URIRef(f"{etd}Relation")))
        g.add((rel_recv, URIRef(f"{etd}subject"), answer))
        g.add((rel_recv, URIRef(f"{etd}predicate"), URIRef(f"{etd}receivesVote")))
        g.add((rel_recv, URIRef(f"{etd}object"), vote))

        issues = _vote_typing_issues(g, etd)
        self.assertEqual(issues["missing_castBy"], 0)
        self.assertEqual(issues["missing_receivesVote"], 0)

    def test_evidence_chunk_unknown_counters(self):
        etd = "https://alfie-project.eu/etd-hub#"
        g = Graph()

        ev1 = URIRef(f"{etd}evidence_1")
        ev2 = URIRef(f"{etd}evidence_2")
        g.add((ev1, RDF.type, URIRef(f"{etd}Evidence")))
        g.add((ev2, RDF.type, URIRef(f"{etd}Evidence")))
        g.add((ev1, URIRef(f"{etd}chunk_index"), Literal(-1, datatype=XSD.integer)))
        g.add((ev2, URIRef(f"{etd}chunk_index"), Literal(0, datatype=XSD.integer)))

        stats = _evidence_chunk_unknown_stats(g, etd)
        self.assertEqual(stats["evidence_total_count"], 2)
        self.assertEqual(stats["evidence_chunk_index_unknown_count"], 1)
        self.assertEqual(stats["evidence_chunk_index_unknown_pct"], 50.0)

    def test_cross_chunk_collision_and_merge_details(self):
        etd = "https://alfie-project.eu/etd-hub#"
        wd = "http://www.wikidata.org/entity/"
        g = Graph()

        # Two local entities with same label, different chunks, same external ID.
        e1 = URIRef(f"{etd}entity_1")
        e2 = URIRef(f"{etd}entity_2")
        for e in (e1, e2):
            g.add((e, RDF.type, URIRef(f"{etd}Entity")))
            g.add((e, RDFS.label, Literal("UNESCO", lang="en")))
            g.add((e, URIRef("http://www.w3.org/2002/07/owl#sameAs"), URIRef(f"{wd}Q7809")))

        # Build relation+evidence chunk contexts so entities are seen in different chunks.
        rel1 = URIRef(f"{etd}rel_1")
        ev1 = URIRef(f"{etd}ev_1")
        g.add((rel1, RDF.type, URIRef(f"{etd}Relation")))
        g.add((rel1, URIRef(f"{etd}subject"), e1))
        g.add((rel1, URIRef(f"{etd}predicate"), URIRef(f"{etd}mentions")))
        g.add((rel1, URIRef(f"{etd}object"), e1))
        g.add((rel1, URIRef(f"{etd}hasEvidence"), ev1))
        g.add((ev1, RDF.type, URIRef(f"{etd}Evidence")))
        g.add((ev1, URIRef(f"{etd}discourse_id"), Literal("doc1", datatype=XSD.string)))
        g.add((ev1, URIRef(f"{etd}chunk_index"), Literal(0, datatype=XSD.integer)))

        rel2 = URIRef(f"{etd}rel_2")
        ev2 = URIRef(f"{etd}ev_2")
        g.add((rel2, RDF.type, URIRef(f"{etd}Relation")))
        g.add((rel2, URIRef(f"{etd}subject"), e2))
        g.add((rel2, URIRef(f"{etd}predicate"), URIRef(f"{etd}mentions")))
        g.add((rel2, URIRef(f"{etd}object"), e2))
        g.add((rel2, URIRef(f"{etd}hasEvidence"), ev2))
        g.add((ev2, RDF.type, URIRef(f"{etd}Evidence")))
        g.add((ev2, URIRef(f"{etd}discourse_id"), Literal("doc1", datatype=XSD.string)))
        g.add((ev2, URIRef(f"{etd}chunk_index"), Literal(1, datatype=XSD.integer)))

        with tempfile.TemporaryDirectory() as td:
            out = _cross_chunk_duplicate_labels(g, etd, Path(td))
            self.assertEqual(out["cross_chunk_collisions_count"], 1)
            self.assertEqual(out["cross_chunk_merges_count"], 1)
            self.assertIn("cross_chunk_collisions_details", out)
            self.assertIn("cross_chunk_merges_details", out)
            self.assertTrue(Path(out["cross_chunk_linking_report_path"]).exists())

    def test_isolated_entities_excludes_has_source_entity_anchors(self):
        etd = "https://alfie-project.eu/etd-hub#"
        g = Graph()

        source_anchor = URIRef(f"{etd}entity_source_anchor")
        dangling = URIRef(f"{etd}entity_dangling")
        rel = URIRef(f"{etd}rel_1")

        for ent, label in ((source_anchor, "Source Anchor"), (dangling, "Dangling")):
            g.add((ent, RDF.type, URIRef(f"{etd}Entity")))
            g.add((ent, RDFS.label, Literal(label, lang="en")))

        g.add((rel, RDF.type, URIRef(f"{etd}Relation")))
        g.add((rel, URIRef(f"{etd}hasSourceEntity"), source_anchor))

        entity_nodes = {source_anchor, dangling}
        isolated = _isolated_entities(g, entity_nodes, etd)

        self.assertEqual(isolated, [str(dangling)])


if __name__ == "__main__":
    unittest.main()

