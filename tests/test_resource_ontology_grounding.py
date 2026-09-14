"""Synthetic tests for the corrected Resource ontology grounding.

No dataset IDs, record IDs, repository/platform names, or domain vocabulary
are used -- only generic placeholder labels, so these tests apply identically
to any future dataset/domain.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from config import Config
from models import Entity
from ontology.add_schema import add_schema
from ontology.mapper import OntologyMapper

HAS_PYSHACL = importlib.util.find_spec("pyshacl") is not None

FOAF_DOCUMENT = URIRef("http://xmlns.com/foaf/0.1/Document")
DCAT_RESOURCE = URIRef("http://www.w3.org/ns/dcat#Resource")
DCAT_DATASET = URIRef("http://www.w3.org/ns/dcat#Dataset")
BIBO_DOCUMENT = URIRef("http://purl.org/ontology/bibo/Document")


def _schema_graph() -> Graph:
    etd = Namespace(Config.NAMESPACES["etd"])
    g = Graph()
    add_schema(g, etd)
    return g


class ResourceInstanceMappingTests(unittest.TestCase):
    def test_generic_repository_platform_resource_does_not_become_foaf_document(self):
        mapper = OntologyMapper()
        entity = Entity(
            uri="https://alfie-project.eu/etd-hub#resource_x",
            label="Some Hosting Platform",
            entity_type="Resource",
            confidence=1.0,
            metadata={},
        )
        mapped = mapper.map_entity(entity)
        self.assertNotIn("foaf:Document", mapped)
        self.assertIn("dcat:Resource", mapped)

    def test_dataset_retains_its_own_dataset_grounding(self):
        mapper = OntologyMapper()
        entity = Entity(
            uri="https://alfie-project.eu/etd-hub#dataset_x",
            label="Some Collection Of Records",
            entity_type="Dataset",
            confidence=1.0,
            metadata={},
        )
        mapped = mapper.map_entity(entity)
        self.assertIn("dcat:Dataset", mapped)
        self.assertNotIn("dcat:Resource", mapped)
        self.assertNotIn("foaf:Document", mapped)

    def test_publication_still_grounds_to_a_document_oriented_class(self):
        mapper = OntologyMapper()
        entity = Entity(
            uri="https://alfie-project.eu/etd-hub#publication_x",
            label="Some Paper",
            entity_type="Publication",
            confidence=1.0,
            metadata={},
        )
        mapped = mapper.map_entity(entity)
        self.assertIn("bibo:Document", mapped)

    def test_unmapped_resource_sibling_type_does_not_inherit_resource_grounding(self):
        # A class that subclasses Resource in the TBox (SoftwarePackage) but
        # has no explicit ENTITY_TYPE_TO_ONTOLOGY entry of its own must not
        # silently pick up Resource's dcat:Resource mapping (instance
        # mapping is a flat lookup, not TBox-inheriting) -- and must
        # certainly not receive an unjustified, more-specific external class.
        mapper = OntologyMapper()
        entity = Entity(
            uri="https://alfie-project.eu/etd-hub#softwarepackage_x",
            label="Some Package",
            entity_type="SoftwarePackage",
            confidence=1.0,
            metadata={},
        )
        mapped = mapper.map_entity(entity)
        self.assertNotIn("dcat:Resource", mapped)
        self.assertNotIn("foaf:Document", mapped)


class ResourceSuperclassRelationshipTests(unittest.TestCase):
    def test_resource_subclasses_etd_entity(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        self.assertIn((etd.Resource, RDFS.subClassOf, etd.Entity), g)

    def test_resource_aligns_to_dcat_resource_via_subclassof_not_equivalence(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        self.assertIn((etd.Resource, RDFS.subClassOf, DCAT_RESOURCE), g)
        self.assertNotIn((etd.Resource, OWL.equivalentClass, DCAT_RESOURCE), g)
        self.assertNotIn((etd.Resource, RDFS.subClassOf, FOAF_DOCUMENT), g)
        self.assertNotIn((etd.Resource, OWL.equivalentClass, FOAF_DOCUMENT), g)

    def test_dataset_does_not_transitively_become_a_document(self):
        # Dataset -> Resource -> dcat:Resource must never route through
        # foaf:Document. Walk the full etd-local subclass chain from
        # Dataset and confirm foaf:Document never appears as an etd-local
        # or directly-asserted parent.
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        self.assertIn((etd.Dataset, RDFS.subClassOf, etd.Resource), g)

        visited = set()
        frontier = [etd.Dataset]
        while frontier:
            node = frontier.pop()
            if node in visited:
                continue
            visited.add(node)
            for parent in g.objects(node, RDFS.subClassOf):
                self.assertNotEqual(
                    str(parent), str(FOAF_DOCUMENT),
                    "Dataset must not transitively become a foaf:Document via Resource",
                )
                if str(parent).startswith(str(etd)):
                    frontier.append(parent)

    def test_research_output_family_does_not_transitively_become_a_document(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for cls in ("ResearchOutput", "ResearchStudy", "Analysis"):
            visited = set()
            frontier = [etd[cls]]
            while frontier:
                node = frontier.pop()
                if node in visited:
                    continue
                visited.add(node)
                for parent in g.objects(node, RDFS.subClassOf):
                    self.assertNotEqual(str(parent), str(FOAF_DOCUMENT))
                    if str(parent).startswith(str(etd)):
                        frontier.append(parent)


class NoAccidentalCyclesTests(unittest.TestCase):
    def test_no_cycles_in_etd_subclass_graph_after_resource_alignment_change(self):
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


class SerializationAndSHACLValidityTests(unittest.TestCase):
    def test_schema_serializes_with_corrected_resource_alignment(self):
        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        turtle = g.serialize(format="turtle")
        self.assertIn("Resource", turtle)
        reparsed = Graph()
        reparsed.parse(data=turtle, format="turtle")
        self.assertIn((etd.Resource, RDFS.subClassOf, DCAT_RESOURCE), reparsed)

    @unittest.skipUnless(HAS_PYSHACL, "pyshacl is required for SHACL validation")
    def test_shacl_shapes_still_validate_with_a_resource_instance(self):
        from validation.shacl_validator import SHACLValidator

        etd = Namespace(Config.NAMESPACES["etd"])
        g = _schema_graph()
        for pfx, uri in Config.NAMESPACES.items():
            g.bind(pfx, Namespace(uri))

        node = URIRef(f"{etd}resource_synthetic_test")
        g.add((node, RDF.type, etd.Resource))
        g.add((node, RDF.type, DCAT_RESOURCE))
        g.add((node, RDFS.label, Literal("Some Hosting Platform", lang="en")))

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
