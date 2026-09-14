"""Grounding-method gate synchronization.

The pipeline deterministically emits ``groundingMethod = "single_entity_span"``
for the priority-(c) single-endpoint span-derivation tier
(``pipeline.py`` -> ``span_source = "single_entity_span"`` ->
``method = span_source``). That is a legitimate producer-side grounding
method, weaker than a two-endpoint ``span_map`` derivation but still a
genuine, non-guessed character span.

``scripts/kg_quality_gate.py``'s ``_provenance_completeness_metrics``
allow-list originally omitted it, so every such Evidence node counted as
``evidence_unexpected_method`` and failed the full gate. These tests pin
the fix: ``single_entity_span`` is accepted, and a genuinely unknown
grounding method is still rejected.
"""
import pathlib
import unittest

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF

from scripts.kg_quality_gate import _provenance_completeness_metrics

ETD = Namespace("https://alfie-project.eu/etd-hub#")


def _graph_with_single_evidence(method: str) -> Graph:
    g = Graph()
    ev = URIRef("https://alfie-project.eu/etd-hub#evidence_test_0001")
    du = URIRef("https://alfie-project.eu/etd-hub#du_test_0001")
    rel = URIRef("https://alfie-project.eu/etd-hub#rel_test_0001")
    g.add((rel, RDF.type, ETD.Relation))
    g.add((rel, ETD.hasEvidence, ev))
    g.add((ev, RDF.type, ETD.Evidence))
    g.add((ev, ETD.groundingStatus, Literal("grounded")))
    g.add((ev, ETD.groundingMethod, Literal(method)))
    g.add((ev, ETD.chunk_index, Literal(0)))
    g.add((ev, ETD.chunk_index_unknown, Literal(False)))
    g.add((ev, ETD.extractedFrom, du))
    g.add((ev, ETD.char_start, Literal(0)))
    g.add((ev, ETD.char_end, Literal(5)))
    g.add((ev, ETD.evidenceText, Literal("hello")))
    g.add((ev, ETD.source, Literal("llm_relation_extractor")))
    g.add((ev, ETD.field, Literal("theme.description")))
    return g


class GroundingMethodGateSyncTests(unittest.TestCase):
    def test_single_entity_span_is_an_intentional_producer_side_method(self):
        """The pipeline itself assigns groundingMethod = single_entity_span."""
        src = pathlib.Path(__file__).resolve().parents[1] / "pipeline.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn('span_source = "single_entity_span"', text)
        self.assertIn("method = span_source", text)

    def test_valid_grounded_single_entity_span_evidence_passes(self):
        metrics = _provenance_completeness_metrics(
            _graph_with_single_evidence("single_entity_span"), ETD
        )
        self.assertEqual(metrics["evidence_unexpected_method_count"], 0)
        # sanity: the node is otherwise a well-formed grounded Evidence node
        self.assertEqual(metrics["evidence_missing_grounding_count"], 0)
        self.assertEqual(metrics["orphan_evidence_count"], 0)
        self.assertEqual(metrics["evidence_inconsistent_chunk_unknown_count"], 0)

    def test_unknown_grounding_method_still_fails(self):
        metrics = _provenance_completeness_metrics(
            _graph_with_single_evidence("totally_made_up_method"), ETD
        )
        self.assertEqual(metrics["evidence_unexpected_method_count"], 1)

    def test_known_baseline_methods_still_pass(self):
        for method in (
            "span_map",
            "single_chunk_discourse",
            "structural",
            "reasoner",
            "vote",
            "lightweight",
            "none",
        ):
            metrics = _provenance_completeness_metrics(
                _graph_with_single_evidence(method), ETD
            )
            self.assertEqual(
                metrics["evidence_unexpected_method_count"], 0, method
            )


if __name__ == "__main__":
    unittest.main()
