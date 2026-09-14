import importlib.util
import unittest

from config import Config
from models import Entity


HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
HAS_PYSHACL = importlib.util.find_spec("pyshacl") is not None


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for inferred provenance test")
class InferredProvenanceTests(unittest.TestCase):
    def test_all_inferred_relations_have_rule_or_derivation(self):
        from rdflib import Literal, Namespace, RDF, URIRef
        from rdflib.namespace import XSD
        from serializers.rdf_serializer import RDFSerializer

        etd_base = Config.NAMESPACES["etd"]
        etd = Namespace(etd_base)
        prov = Namespace(Config.NAMESPACES["prov"])

        entities = [
            Entity(uri=f"{etd_base}entity_a", label="A", entity_type="Entity", confidence=1.0, metadata={}),
            Entity(uri=f"{etd_base}entity_b", label="B", entity_type="Entity", confidence=1.0, metadata={}),
            Entity(uri=f"{etd_base}entity_c", label="C", entity_type="Entity", confidence=1.0, metadata={}),
        ]

        ann = {
            "themes": {},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {
                "inferred_relations": [
                    {
                        "uri": f"{etd_base}rel_inf_1",
                        "source": f"{etd_base}entity_a",
                        "relation": "relatedTo",
                        "target": f"{etd_base}entity_b",
                        "confidence": 0.7,
                        "is_inferred": True,
                        "metadata": {"inference_type": "composition", "discourse_id": "x1"},
                    },
                    {
                        "uri": f"{etd_base}rel_inf_2",
                        "source": f"{etd_base}entity_b",
                        "relation": "relatedTo",
                        "target": f"{etd_base}entity_c",
                        "confidence": 0.7,
                        "is_inferred": True,
                        "metadata": {
                            "supporting_relation_uris": [f"{etd_base}rel_support_1", f"{etd_base}rel_support_2"],
                            "discourse_id": "x2",
                        },
                    },
                ]
            },
        }

        serializer = RDFSerializer(
            annotations_dict=ann,
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
        )
        g = serializer.graph

        inferred_nodes = []
        for rel_node in g.subjects(RDF.type, etd.Relation):
            if (rel_node, etd.isInferred, Literal(True, datatype=XSD.boolean)) in g:
                inferred_nodes.append(rel_node)

        self.assertGreaterEqual(len(inferred_nodes), 2, "Expected inferred relations in RDF output")

        for rel_node in inferred_nodes:
            has_rule = any(True for _ in g.objects(rel_node, etd.inferenceRule))
            has_derivation = any(True for _ in g.objects(rel_node, prov.wasDerivedFrom))
            self.assertTrue(
                has_rule or has_derivation,
                f"Inferred relation missing provenance metadata: {rel_node}",
            )

        # Namespace/provenance predicate guarantee:
        # - prov prefix is bound
        # - prov:wasDerivedFrom URI is exactly the PROV namespace URI
        prefixes = dict(g.namespaces())
        self.assertIn("prov", prefixes)
        self.assertEqual(str(prefixes["prov"]), "http://www.w3.org/ns/prov#")

        derived_preds = {
            str(p) for _, p, _ in g.triples((None, prov.wasDerivedFrom, None))
        }
        for p in derived_preds:
            self.assertEqual(p, "http://www.w3.org/ns/prov#wasDerivedFrom")

        ttl = g.serialize(format="turtle")
        if isinstance(ttl, bytes):
            ttl = ttl.decode("utf-8")
        self.assertIn("@prefix prov:", ttl)

    @unittest.skipUnless(HAS_PYSHACL, "pyshacl is required for SHACL inferred provenance validation test")
    def test_inferred_provenance_passes_shacl_rule(self):
        import tempfile
        from pathlib import Path

        from serializers.rdf_serializer import RDFSerializer
        from validation.shacl_validator import SHACLValidator

        etd_base = Config.NAMESPACES["etd"]
        entities = [
            Entity(uri=f"{etd_base}entity_a", label="A", entity_type="Entity", confidence=1.0, metadata={}),
            Entity(uri=f"{etd_base}entity_b", label="B", entity_type="Entity", confidence=1.0, metadata={}),
        ]
        ann = {
            "themes": {},
            "questions": {},
            "answers": {},
            "votes": {},
            "experts": {},
            "documents": {},
            "metadata": {
                "inferred_relations": [
                    {
                        "uri": f"{etd_base}rel_inf_ok",
                        "source": f"{etd_base}entity_a",
                        "relation": "relatedTo",
                        "target": f"{etd_base}entity_b",
                        "confidence": 0.7,
                        "is_inferred": True,
                        "metadata": {"inference_type": "transitive", "discourse_id": "z1"},
                    }
                ]
            },
        }

        serializer = RDFSerializer(
            annotations_dict=ann,
            registry=_Registry(entities),
            namespaces=Config.NAMESPACES,
        )
        # Test-isolation fix: SHACLValidator writes shacl_report.* into its
        # output_dir. Config.OUTPUT_DIR here resolved to the real
        # production output/ directory, silently overwriting its live
        # shacl_report.ttl with this test's synthetic 3-entity graph.
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = SHACLValidator(output_dir=Path(tmpdir))
            result = validator.validate_graph(serializer.graph)
        self.assertTrue(result.get("success", False))
        self.assertTrue(result.get("conforms", False), result.get("results_text", "SHACL did not conform"))


if __name__ == "__main__":
    unittest.main()
