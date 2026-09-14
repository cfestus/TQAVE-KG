import importlib.util
import tempfile
import unittest
from pathlib import Path

from rdflib import Namespace
from rdflib.namespace import RDF

from serializers.rdf_serializer import RDFSerializer

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
HAS_PYSHACL = importlib.util.find_spec("pyshacl") is not None

ETD = "https://alfie-project.eu/etd-hub#"
NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov": "http://www.w3.org/ns/prov#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "etd": ETD,
    "onto": "https://alfie-project.eu/ontology#",
}


def _duplicate_relation_annotations(evidence_a: str, evidence_b: str) -> dict:
    """Two relation dicts for the SAME subject/predicate/object/discourse_id
    -- e.g. the same proposition independently extracted from two different
    semantic units in the same discourse -- neither carrying an explicit
    'uri', so both fall back to the identical deterministic hash in
    RDFSerializer._get_all_relations and resolve to the same reified
    relation node. Reproduces the real-world duplicate-evidence condition
    without depending on chunking/extraction internals."""
    return {
        "themes": {
            "d1": {
                "entities": [
                    {"uri": f"{ETD}entity_a", "label": "A", "entity_type": "Group", "metadata": {"discourse_id": "d1"}},
                ],
                "relations": [
                    {
                        "source": f"{ETD}entity_a",
                        "relation": "charged",
                        "target": "significantly higher fares",
                        "target_is_literal": True,
                        "confidence": 0.9,
                        "metadata": {
                            "discourse_id": "d1",
                            "field": "theme.description",
                            "extractor": "llm_relation_extractor",
                            "evidence": evidence_a,
                        },
                    },
                    {
                        "source": f"{ETD}entity_a",
                        "relation": "charged",
                        "target": "significantly higher fares",
                        "target_is_literal": True,
                        "confidence": 0.9,
                        "metadata": {
                            "discourse_id": "d1",
                            "field": "theme.description",
                            "extractor": "llm_relation_extractor",
                            "evidence": evidence_b,
                        },
                    },
                ],
                "metadata": {"text": "dummy"},
            }
        },
        "questions": {},
        "answers": {},
        "votes": {},
        "experts": {},
        "documents": {},
        "metadata": {},
    }


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required")
class DuplicateEvidenceIdentityTests(unittest.TestCase):
    """Regression tests for the SHACL sh:MaxCountConstraintComponent
    violations on etd:evidenceText: the same reified relation node received
    multiple evidenceText values because the Evidence node's identity key
    did not fold in the evidence text itself."""

    def test_different_evidence_for_same_relation_gets_two_distinct_nodes(self):
        ann = _duplicate_relation_annotations(
            evidence_a="Neighborhoods with larger non-white populations charged significantly higher fares",
            evidence_b="A different local sentence also stating the same proposition, charged differently phrased",
        )
        serializer = RDFSerializer(ann, registry=None, namespaces=NAMESPACES)
        g = serializer.graph
        ETD_NS = Namespace(ETD)

        rel_nodes = list(g.subjects(RDF.type, ETD_NS.Relation))
        self.assertEqual(len(rel_nodes), 1, "the two relation dicts must resolve to ONE reified relation node")
        rel_node = rel_nodes[0]

        evidence_nodes = list(g.objects(rel_node, ETD_NS.hasEvidence))
        self.assertEqual(
            len(evidence_nodes), 2,
            "two genuinely different evidence strings must produce two distinct Evidence nodes, not be collapsed",
        )

        # Every Evidence node must satisfy the SHACL constraint: exactly one
        # evidenceText literal.
        texts = set()
        for ev in evidence_nodes:
            ev_texts = list(g.objects(ev, ETD_NS.evidenceText))
            self.assertEqual(len(ev_texts), 1, f"{ev} must have exactly one evidenceText")
            texts.add(str(ev_texts[0]))

        # Both original, distinct evidence strings must be preserved
        # verbatim -- not merged, not dropped.
        self.assertEqual(
            texts,
            {
                "Neighborhoods with larger non-white populations charged significantly higher fares",
                "A different local sentence also stating the same proposition, charged differently phrased",
            },
        )

        # The internal audit-grade contract check must not raise.
        serializer._assert_evidence_contract()

    def test_identical_evidence_for_same_relation_collapses_to_one_node(self):
        same_text = "Neighborhoods with larger non-white populations charged significantly higher fares"
        ann = _duplicate_relation_annotations(evidence_a=same_text, evidence_b=same_text)
        serializer = RDFSerializer(ann, registry=None, namespaces=NAMESPACES)
        g = serializer.graph
        ETD_NS = Namespace(ETD)

        rel_node = next(g.subjects(RDF.type, ETD_NS.Relation))
        evidence_nodes = list(g.objects(rel_node, ETD_NS.hasEvidence))
        self.assertEqual(
            len(evidence_nodes), 1,
            "truly identical evidence must not create redundant duplicate Evidence nodes",
        )
        ev_texts = list(g.objects(evidence_nodes[0], ETD_NS.evidenceText))
        self.assertEqual(len(ev_texts), 1)
        self.assertEqual(str(ev_texts[0]), same_text)

        serializer._assert_evidence_contract()

    @unittest.skipUnless(HAS_PYSHACL, "pyshacl is required for SHACL validation")
    def test_shacl_validation_reports_no_evidence_text_violations(self):
        from validation.shacl_validator import SHACLValidator

        ann = _duplicate_relation_annotations(
            evidence_a="Neighborhoods with larger non-white populations charged significantly higher fares",
            evidence_b="A different local sentence also stating the same proposition, charged differently phrased",
        )
        serializer = RDFSerializer(ann, registry=None, namespaces=NAMESPACES)

        with tempfile.TemporaryDirectory() as tmpdir:
            validator = SHACLValidator(output_dir=Path(tmpdir))
            result = validator.validate_graph(serializer.graph)

        results_graph = result["results_graph"]
        SH = Namespace("http://www.w3.org/ns/shacl#")
        for res in results_graph.subjects(RDF.type, SH.ValidationResult):
            path = next(results_graph.objects(res, SH.resultPath), None)
            component = next(results_graph.objects(res, SH.sourceConstraintComponent), None)
            if path is not None and str(path) == f"{ETD}evidenceText":
                self.assertNotEqual(
                    str(component),
                    "http://www.w3.org/ns/shacl#MaxCountConstraintComponent",
                    f"unexpected evidenceText MaxCount violation: {res}",
                )


if __name__ == "__main__":
    unittest.main()
