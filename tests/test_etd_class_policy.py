import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from config import Config
from ontology.add_schema import add_schema
from scripts.kg_quality_gate import run_quality_gate


class EtdClassPolicyTests(unittest.TestCase):
    def test_add_schema_has_no_deprecated_class_declarations(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = Graph()
        add_schema(g, etd)
        self.assertNotIn((etd.AiSystem, RDF.type, OWL.Class), g)
        self.assertNotIn((etd.Aisystem, RDF.type, OWL.Class), g)
        self.assertNotIn((etd.Personaldata, RDF.type, OWL.Class), g)

    def test_quality_gate_rejects_deprecated_variant_even_without_collision_pair(self):
        etd_ns = Config.NAMESPACES["etd"]
        etd = Namespace(etd_ns)

        g = Graph()
        # Deprecated variant only (no canonical pair) => no lexical collision pair.
        g.add((etd.AiSystem, RDF.type, OWL.Class))
        g.add((URIRef(f"{etd}instance_x"), RDF.type, etd.AiSystem))
        # Keep export requirement satisfied so failure reason is from class policy.
        export = URIRef(f"{etd}export_test")
        g.add((export, RDF.type, etd.Export))
        g.add((export, RDFS.label, Literal("Export: d (r)", lang="en")))
        g.add((export, etd.datasetId, Literal("d")))
        g.add((export, etd.runId, Literal("r")))

        with tempfile.TemporaryDirectory() as td:
            ttl_path = Path(td) / "g.ttl"
            out_dir = Path(td) / "out"
            shapes_path = Path(td) / "shapes.ttl"
            out_dir.mkdir(parents=True, exist_ok=True)
            shapes_path.write_text("@prefix sh: <http://www.w3.org/ns/shacl#> .", encoding="utf-8")
            g.serialize(destination=str(ttl_path), format="turtle")

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
                ok, metrics, reasons = run_quality_gate(
                    ttl_path=ttl_path,
                    shapes_path=shapes_path,
                    output_dir=out_dir,
                    etd_namespace=etd_ns,
                )

            self.assertFalse(ok)
            self.assertEqual(int(metrics.get("etd_class_name_collision_groups_count", 0)), 0)
            self.assertGreater(int(metrics.get("deprecated_etd_class_declarations_count", 0)), 0)
            self.assertGreater(int(metrics.get("deprecated_etd_instance_type_usage_count", 0)), 0)
            self.assertIn("deprecated_etd_class_declarations_present", reasons)
            self.assertIn("deprecated_etd_instance_types_present", reasons)


if __name__ == "__main__":
    unittest.main()

