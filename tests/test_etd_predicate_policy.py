import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from config import Config
from ontology.add_schema import add_schema
from scripts.kg_quality_gate import run_quality_gate


class EtdPredicatePolicyTests(unittest.TestCase):
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

    def test_gate_fails_on_undeclared_etd_predicate_usage(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = self._base_graph()
        node = URIRef(f"{etd}entity_1")
        g.add((node, RDF.type, etd.Entity))
        g.add((node, etd.unregisteredPredicateX, Literal("x")))

        ok, metrics, reasons = self._run_gate(g)
        self.assertFalse(ok)
        self.assertIn("undeclared_etd_predicates_used", reasons)
        self.assertGreater(int(metrics.get("undeclared_etd_predicates_used_count", 0)), 0)

    def test_gate_fails_on_banned_dynamic_raw_predicate(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = self._base_graph()
        p = etd.dynamic_bad_raw
        g.add((p, RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(f"{etd}e1"), p, Literal("x")))

        ok, metrics, reasons = self._run_gate(g)
        self.assertFalse(ok)
        self.assertIn("banned_dynamic_etd_raw_predicates_present", reasons)
        self.assertEqual(int(metrics.get("banned_dynamic_etd_raw_predicates_count", 0)), 1)

    def test_gate_fails_when_declared_kind_mismatches_usage(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = self._base_graph()
        # hasEvidence is declared as object property; use a literal to force mismatch.
        rel = URIRef(f"{etd}rel_1")
        g.add((rel, RDF.type, etd.Relation))
        g.add((rel, etd.hasEvidence, Literal("not-an-iri")))

        ok, metrics, reasons = self._run_gate(g)
        self.assertFalse(ok)
        self.assertIn("etd_declared_vs_observed_kind_mismatch", reasons)
        self.assertGreater(int(metrics.get("etd_declared_vs_observed_kind_mismatch_count", 0)), 0)

    def test_gate_fails_when_external_namespace_used_without_matching_import(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = Graph()
        export = URIRef(f"{etd}export_test")
        g.add((export, RDF.type, etd.Export))
        g.add((export, RDFS.label, Literal("Export: d (r)", lang="en")))
        g.add((export, etd.datasetId, Literal("d")))
        g.add((export, etd.runId, Literal("r")))

        # Use external namespace without declaring required owl:imports.
        node = URIRef(f"{etd}entity_1")
        g.add((node, RDF.type, URIRef("https://w3id.org/aipo#AISystem")))

        ok, metrics, reasons = self._run_gate(g)
        self.assertFalse(ok)
        self.assertIn("missing_external_owl_imports", reasons)
        self.assertGreater(int(metrics.get("missing_external_imports_count", 0)), 0)


if __name__ == "__main__":
    unittest.main()
