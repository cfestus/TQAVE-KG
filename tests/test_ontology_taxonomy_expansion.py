import importlib.util
import tempfile
import unittest
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from config import Config
from ontology.add_schema import add_schema

HAS_PYSHACL = importlib.util.find_spec("pyshacl") is not None

NEW_CLASSES = [
    "ResearchOutput",
    "ResearchStudy",
    "Analysis",
    "Publication",
    "SoftwareProduct",
    "AuditProcess",
    "Concept",
]

EXPECTED_SUPERCLASS = {
    "ResearchOutput": "Resource",
    "ResearchStudy": "ResearchOutput",
    "Analysis": "ResearchOutput",
    "Publication": "ResearchOutput",
    "SoftwareProduct": "System",
    "AuditProcess": "Evaluation",
    "Concept": "Entity",
    "AlgorithmicBias": "Bias",
    "RacialBias": "Bias",
    "DemographicParity": "EthicalPrinciple",
    "EthicalPrinciple": "Entity",
}


def _schema_graph() -> Graph:
    etd = Namespace(Config.NAMESPACES["etd"])
    g = Graph()
    add_schema(g, etd)
    return g


class NewClassesExistTests(unittest.TestCase):
    def test_every_new_class_is_declared_as_owl_class(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for cls in NEW_CLASSES:
            self.assertIn(
                (etd[cls], RDF.type, OWL.Class), g,
                f"expected etd:{cls} to be declared as owl:Class",
            )

    def test_every_new_class_has_label_and_comment(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for cls in NEW_CLASSES:
            labels = list(g.objects(etd[cls], RDFS.label))
            comments = list(g.objects(etd[cls], RDFS.comment))
            self.assertTrue(labels, f"etd:{cls} missing rdfs:label")
            self.assertTrue(comments, f"etd:{cls} missing rdfs:comment")
            # Comments must be specific enough to disambiguate, not the
            # generic auto-generated fallback.
            self.assertNotIn("ETD class representing", str(comments[0]))


class SuperclassRelationshipTests(unittest.TestCase):
    def test_expected_superclass_relationships(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for child, parent in EXPECTED_SUPERCLASS.items():
            self.assertIn(
                (etd[child], RDFS.subClassOf, etd[parent]), g,
                f"expected etd:{child} rdfs:subClassOf etd:{parent}",
            )

    def test_research_output_family_all_root_in_resource_then_entity(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for cls in ("ResearchStudy", "Analysis", "Publication"):
            chain = set()
            current = etd[cls]
            for _ in range(10):
                parents = list(g.objects(current, RDFS.subClassOf))
                etd_parents = [p for p in parents if str(p).startswith(str(etd))]
                if not etd_parents:
                    break
                current = etd_parents[0]
                chain.add(str(current))
            self.assertIn(str(etd.ResearchOutput), chain)
            self.assertIn(str(etd.Resource), chain)
            self.assertIn(str(etd.Entity), chain)

    def test_dataset_and_research_output_are_distinct_siblings(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        self.assertIn((etd.Dataset, RDFS.subClassOf, etd.Resource), g)
        self.assertIn((etd.ResearchOutput, RDFS.subClassOf, etd.Resource), g)
        self.assertNotIn((etd.Dataset, RDFS.subClassOf, etd.ResearchOutput), g)
        self.assertNotIn((etd.ResearchOutput, RDFS.subClassOf, etd.Dataset), g)


class NoAccidentalCyclesTests(unittest.TestCase):
    def test_no_cycles_in_etd_subclass_graph(self):
        etd_str = Config.NAMESPACES["etd"]
        g = _schema_graph()
        edges = {}
        for s, _p, o in g.triples((None, RDFS.subClassOf, None)):
            if str(s).startswith(etd_str) and str(o).startswith(etd_str):
                edges.setdefault(str(s), set()).add(str(o))

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in edges}
        for targets in edges.values():
            for m in targets:
                color.setdefault(m, WHITE)

        def dfs(node):
            color[node] = GRAY
            for nxt in edges.get(node, ()):
                if color.get(nxt, WHITE) == GRAY:
                    return False
                if color.get(nxt, WHITE) == WHITE and not dfs(nxt):
                    return False
            color[node] = BLACK
            return True

        for node in list(edges.keys()):
            if color[node] == WHITE:
                self.assertTrue(dfs(node), f"cycle detected reachable from {node}")


class ExistingDeclarationsRemainValidTests(unittest.TestCase):
    def test_person_bias_aisystem_populationgroup_demographicgroup_still_declared(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for cls in ("Person", "Bias", "AISystem", "PopulationGroup", "DemographicGroup"):
            self.assertIn((etd[cls], RDF.type, OWL.Class), g, f"etd:{cls} must remain declared")

        self.assertIn((etd.Person, RDFS.subClassOf, etd.Entity), g)
        self.assertIn((etd.Bias, RDFS.subClassOf, etd.Entity), g)
        self.assertIn((etd.AISystem, RDFS.subClassOf, etd.System), g)
        self.assertIn((etd.PopulationGroup, RDFS.subClassOf, etd.Entity), g)
        self.assertIn((etd.DemographicGroup, RDFS.subClassOf, etd.PopulationGroup), g)

    def test_no_deprecated_class_variants_reintroduced(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        self.assertNotIn((etd.AiSystem, RDF.type, OWL.Class), g)
        self.assertNotIn((etd.Aisystem, RDF.type, OWL.Class), g)
        self.assertNotIn((etd.Personaldata, RDF.type, OWL.Class), g)


class ExternalAlignmentTests(unittest.TestCase):
    def test_justified_external_alignments_present(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        self.assertIn(
            (etd.Person, OWL.equivalentClass, URIRef("http://xmlns.com/foaf/0.1/Person")), g,
        )
        self.assertIn(
            (etd.Dataset, RDFS.subClassOf, URIRef("http://www.w3.org/ns/dcat#Dataset")), g,
        )
        self.assertIn(
            (etd.Organization, RDFS.subClassOf, URIRef("http://schema.org/Organization")), g,
        )
        self.assertIn(
            (etd.SoftwareProduct, RDFS.subClassOf, URIRef("http://schema.org/SoftwareApplication")), g,
        )
        self.assertIn(
            (etd.Publication, RDFS.subClassOf, URIRef("http://purl.org/ontology/bibo/Document")), g,
        )
        self.assertIn(
            (etd.Concept, RDFS.subClassOf, URIRef("http://www.w3.org/2004/02/skos/core#Concept")), g,
        )

    def test_no_forced_alignment_for_research_study_or_analysis(self):
        # These categories were explicitly NOT forced onto an approximately
        # related external class -- they remain ETD-only.
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for cls in ("ResearchStudy", "Analysis", "AuditProcess"):
            external_parents = [
                o for o in g.objects(etd[cls], RDFS.subClassOf)
                if not str(o).startswith(Config.NAMESPACES["etd"])
            ]
            self.assertEqual(external_parents, [], f"etd:{cls} should not have a forced external alignment")


class SerializationAndSchemaLoadingTests(unittest.TestCase):
    def test_schema_serializes_to_turtle_without_error(self):
        g = _schema_graph()
        turtle = g.serialize(format="turtle")
        self.assertIn("ResearchOutput", turtle)
        self.assertIn("AuditProcess", turtle)

    def test_schema_round_trips_through_parse(self):
        g = _schema_graph()
        turtle = g.serialize(format="turtle")
        reparsed = Graph()
        reparsed.parse(data=turtle, format="turtle")
        self.assertEqual(len(g), len(reparsed))

    @unittest.skipUnless(HAS_PYSHACL, "pyshacl is required for SHACL validation")
    def test_shacl_shapes_still_load_and_validate_against_schema_plus_backbone(self):
        from validation.shacl_validator import SHACLValidator

        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for pfx, uri in Config.NAMESPACES.items():
            g.bind(pfx, Namespace(uri))
        export = URIRef(f"{etd}export_test")
        g.add((export, RDF.type, etd.Export))
        g.add((export, RDFS.label, Literal("Export: d (r)", lang="en")))
        g.add((export, etd.datasetId, Literal("d")))
        g.add((export, etd.runId, Literal("r")))

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            validator = SHACLValidator(output_dir=out_dir)
            result = validator.validate_graph(g)
            self.assertTrue(result.get("success", True) or "conforms" in result)
            self.assertTrue(result.get("conforms", False), result.get("results_text", "SHACL failed to load/validate"))


if __name__ == "__main__":
    unittest.main()
