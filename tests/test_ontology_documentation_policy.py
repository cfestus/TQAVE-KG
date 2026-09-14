import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from config import Config
from ontology.add_schema import add_schema
from scripts.kg_quality_gate import (
    _etd_property_lifecycle_coverage,
    _etd_term_documentation_coverage,
    run_quality_gate,
)
from serializers.rdf_serializer import RDFSerializer


class _Registry:
    def __init__(self):
        self.entities = {}

    def get_all_entities(self):
        return []


class OntologyDocumentationPolicyTests(unittest.TestCase):
    def _base_graph(self) -> Graph:
        etd = Namespace(Config.NAMESPACES["etd"])
        g = Graph()
        add_schema(g, etd)
        export = URIRef(f"{etd}export_test")
        g.add((export, RDF.type, etd.Export))
        g.add((export, RDFS.label, Literal("Export: d (r)", lang="en")))
        g.add((export, etd.datasetId, Literal("d")))
        g.add((export, etd.runId, Literal("r")))
        return g

    def _run_gate(self, graph: Graph):
        etd_ns = Config.NAMESPACES["etd"]
        with tempfile.TemporaryDirectory() as td:
            ttl_path = Path(td) / "g.ttl"
            out_dir = Path(td) / "out"
            shapes_path = Path(td) / "shapes.ttl"
            out_dir.mkdir(parents=True, exist_ok=True)
            shapes_path.write_text("@prefix sh: <http://www.w3.org/ns/shacl#> .", encoding="utf-8")
            graph.serialize(destination=str(ttl_path), format="turtle")

            with patch(
                "scripts.kg_quality_gate.SHACLValidator.validate_graph",
                return_value={
                    "conforms": True,
                    "counts": {"violations": 0, "warnings": 0, "infos": 0},
                    "report_path": str(out_dir / "shacl_report.txt"),
                },
            ), patch(
                "scripts.kg_quality_gate._entity_nodes", return_value=set()
            ), patch(
                "scripts.kg_quality_gate._isolated_entities", return_value=set()
            ), patch(
                "scripts.kg_quality_gate._vote_typing_issues",
                return_value={"missing_castBy": 0, "missing_receivesVote": 0, "multiple_castBy": 0, "multiple_receivesVote": 0},
            ), patch(
                "scripts.kg_quality_gate._duplicate_reified_relation_spo",
                return_value={"duplicate_relation_spo_groups_count": 0, "duplicate_relation_spo_total_relations_involved": 0},
            ), patch(
                "scripts.kg_quality_gate._same_chunk_duplicate_labels_count",
                return_value=0,
            ), patch(
                "scripts.kg_quality_gate._cross_chunk_duplicate_labels",
                return_value={"cross_chunk_collisions_count": 0, "cross_chunk_merges_count": 0, "cross_chunk_linking_report_path": None},
            ):
                return run_quality_gate(
                    ttl_path=ttl_path,
                    shapes_path=shapes_path,
                    output_dir=out_dir,
                    etd_namespace=etd_ns,
                )

    def test_declared_terms_have_labels_comments_and_lifecycle(self):
        g = self._base_graph()
        etd_ns = Config.NAMESPACES["etd"]

        doc_cov = _etd_term_documentation_coverage(g, etd_ns)
        self.assertEqual(int(doc_cov.get("etd_terms_missing_label_count", 0)), 0)
        self.assertEqual(int(doc_cov.get("etd_terms_missing_comment_count", 0)), 0)

        lifecycle_cov = _etd_property_lifecycle_coverage(g, etd_ns)
        self.assertEqual(int(lifecycle_cov.get("etd_property_lifecycle_missing_count", 0)), 0)
        self.assertEqual(int(lifecycle_cov.get("etd_property_lifecycle_invalid_count", 0)), 0)
        self.assertEqual(int(lifecycle_cov.get("etd_property_lifecycle_note_missing_count", 0)), 0)

    def test_source_class_and_source_property_are_both_documented(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = self._base_graph()

        class_comment = next(g.objects(etd.Source, RDFS.comment), None)
        prop_comment = next(g.objects(etd.source, RDFS.comment), None)
        self.assertIsNotNone(class_comment)
        self.assertIsNotNone(prop_comment)
        self.assertNotEqual(str(class_comment), str(prop_comment))

    def test_dynamic_alignment_class_gets_label_and_comment(self):
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
            registry=_Registry(),
            namespaces=Config.NAMESPACES,
            prune_isolated_entities=False,
        )

        etd = Namespace(Config.NAMESPACES["etd"])
        serializer._class_alignment_targets["Biasrisk"] = URIRef("https://w3id.org/airo#BiasRisk")
        serializer._add_alignment_axioms()

        cls = URIRef(f"{etd}Biasrisk")
        self.assertIn((cls, RDF.type, OWL.Class), serializer.graph)
        self.assertTrue(any(True for _ in serializer.graph.objects(cls, RDFS.label)))
        self.assertTrue(any(True for _ in serializer.graph.objects(cls, RDFS.comment)))

    def test_quality_gate_fails_when_term_comment_missing(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = self._base_graph()
        for obj in list(g.objects(etd.relatedTo, RDFS.comment)):
            g.remove((etd.relatedTo, RDFS.comment, obj))

        ok, _metrics, reasons = self._run_gate(g)
        self.assertFalse(ok)
        self.assertIn("etd_terms_missing_comment", reasons)


if __name__ == "__main__":
    unittest.main()
