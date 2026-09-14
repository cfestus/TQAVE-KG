"""Predicate-kind gate consistency.

`serializers/rdf_serializer.py::_declare_local_relation_predicates` infers
a dynamic local relation predicate's property kind from the object shape
across ALL of its surviving reified `etd:Relation` instances (any resource
object -> owl:ObjectProperty; literal objects still permitted).

`scripts/kg_quality_gate.py`'s declared-vs-observed kind check previously
re-derived the "observed" shape from the DIRECT-SPO projection, which
legitimately omits negated / non-asserted relations (direct-SPO assertion
invariant). A predicate whose only resource-valued relation was
non-asserted therefore looked literal-only and, absent a
`PREDICATE_REGISTRY.literal_allowed=True` entry, was reported as a
declared/observed kind mismatch (e.g. `favored`).

The gate now validates against the same reified relation population the
declaration was inferred from (`_reified_relation_object_shape`), keeping
the direct-SPO shape only as a diagnostic. No serialization / direct-SPO
policy change. Named predicates below are synthetic fixtures.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from config import Config
from ontology.add_schema import add_schema
from scripts.kg_quality_gate import (
    _observed_predicate_value_shape,
    _reified_relation_object_shape,
    run_quality_gate,
)

ETD = Namespace(Config.NAMESPACES["etd"])


def _add_reified_relation(g, rel_uri, subj, pred_uri, obj, *, emit_direct):
    g.add((rel_uri, RDF.type, ETD.Relation))
    g.add((rel_uri, ETD.subject, subj))
    g.add((rel_uri, ETD.predicate, pred_uri))
    g.add((rel_uri, ETD.object, obj))
    if emit_direct:
        g.add((subj, pred_uri, obj))


class ReifiedRelationObjectShapeUnitTests(unittest.TestCase):
    def _g(self):
        return Graph()

    def test_literal_only_reified_objects_is_datatype(self):
        g = self._g()
        p = ETD.px
        _add_reified_relation(g, ETD.r1, ETD.s1, p, Literal("a"), emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, Literal("b"), emit_direct=True)
        self.assertEqual(_reified_relation_object_shape(g, ETD, p), "datatype")

    def test_resource_only_reified_objects_is_object(self):
        g = self._g()
        p = ETD.px
        _add_reified_relation(g, ETD.r1, ETD.s1, p, ETD.o1, emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, ETD.o2, emit_direct=True)
        self.assertEqual(_reified_relation_object_shape(g, ETD, p), "object")

    def test_both_reified_object_kinds_is_mixed(self):
        g = self._g()
        p = ETD.px
        _add_reified_relation(g, ETD.r1, ETD.s1, p, Literal("a"), emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, ETD.o2, emit_direct=True)
        self.assertEqual(_reified_relation_object_shape(g, ETD, p), "mixed")

    def test_mixed_even_when_resource_relation_emits_no_direct_triple(self):
        # `favored`-shaped example: 2 literal (direct emitted) + 1 resource
        # (direct suppressed because negated/non-asserted). Reified shape is
        # still mixed; the direct-SPO projection alone would say datatype.
        g = self._g()
        p = ETD.px
        _add_reified_relation(g, ETD.r1, ETD.s1, p, Literal("a"), emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, Literal("b"), emit_direct=True)
        _add_reified_relation(g, ETD.r3, ETD.s2, p, ETD.grp, emit_direct=False)
        self.assertEqual(_reified_relation_object_shape(g, ETD, p), "mixed")
        self.assertEqual(_observed_predicate_value_shape(g, p), "datatype")

    def test_none_when_no_reified_relation_uses_predicate(self):
        g = self._g()
        p = ETD.px
        # predicate only ever appears as a plain triple, never as etd:predicate
        g.add((ETD.s1, p, Literal("x")))
        self.assertEqual(_reified_relation_object_shape(g, ETD, p), "none")

    def test_non_relation_node_is_not_counted(self):
        g = self._g()
        p = ETD.px
        # a node carrying etd:predicate but NOT typed etd:Relation is ignored
        g.add((ETD.n1, ETD.predicate, p))
        g.add((ETD.n1, ETD.object, ETD.o1))
        self.assertEqual(_reified_relation_object_shape(g, ETD, p), "none")

    def test_schema_triples_not_counted_as_usage(self):
        g = self._g()
        add_schema(g, ETD)
        # trainedOn is a schema-declared predicate; with no reified relation
        # instances its reified shape is "none", not derived from any
        # rdfs:domain/range or other schema triple.
        self.assertEqual(_reified_relation_object_shape(g, ETD, ETD.trainedOn), "none")


class GateReconciliationEndToEndTests(unittest.TestCase):
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
                return_value={"conforms": True,
                              "counts": {"violations": 0, "warnings": 0, "infos": 0},
                              "report_path": str(out_dir / "shacl_report.txt")},
            ), patch("scripts.kg_quality_gate._entity_nodes", return_value=set()), \
                 patch("scripts.kg_quality_gate._isolated_entities", return_value=set()), \
                 patch("scripts.kg_quality_gate._vote_typing_issues",
                       return_value={"missing_castBy": 0, "missing_receivesVote": 0,
                                     "multiple_castBy": 0, "multiple_receivesVote": 0}), \
                 patch("scripts.kg_quality_gate._duplicate_reified_relation_spo",
                       return_value={"duplicate_relation_spo_groups_count": 0,
                                     "duplicate_relation_spo_total_relations_involved": 0}), \
                 patch("scripts.kg_quality_gate._same_chunk_duplicate_labels_count", return_value=0), \
                 patch("scripts.kg_quality_gate._cross_chunk_duplicate_labels",
                       return_value={"cross_chunk_collisions_count": 0, "cross_chunk_merges_count": 0,
                                     "cross_chunk_linking_report_path": None}):
                return run_quality_gate(ttl_path=ttl_path, shapes_path=shapes_path,
                                        output_dir=out_dir, etd_namespace=etd_ns)

    def _base_graph(self) -> Graph:
        g = Graph()
        add_schema(g, ETD)
        export = URIRef(f"{ETD}export_test")
        g.add((export, RDF.type, ETD.Export))
        g.add((export, RDFS.label, Literal("Export: d (r)", lang="en")))
        g.add((export, ETD.datasetId, Literal("d")))
        g.add((export, ETD.runId, Literal("r")))
        return g

    def _declare_local_object_property(self, g, local):
        p = URIRef(f"{ETD}{local}")
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.label, Literal(local, lang="en")))
        g.add((p, RDFS.comment, Literal("local", lang="en")))
        g.add((p, ETD.lifecycleStatus, Literal("reserved")))
        g.add((p, ETD.lifecycleNote, Literal("x", lang="en")))
        return p

    # (a)
    def test_reified_mixed_with_direct_literal_only_no_mismatch(self):
        g = self._base_graph()
        p = self._declare_local_object_property(g, "dynMix")
        _add_reified_relation(g, ETD.r1, ETD.s1, p, Literal("certain genders"), emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, Literal("certain ages"), emit_direct=True)
        _add_reified_relation(g, ETD.r3, ETD.s2, p, ETD.someGroup, emit_direct=False)  # non-asserted -> suppressed
        ok, metrics, reasons = self._run_gate(g)
        self.assertNotIn("etd_declared_vs_observed_kind_mismatch", reasons)
        self.assertEqual(int(metrics["etd_declared_vs_observed_kind_mismatch_count"]), 0)
        self.assertEqual(metrics["etd_declared_vs_observed_kind_reconciled"].get(str(p), {}).get("observed"), "mixed")
        shapes = metrics["etd_predicate_object_shapes"][str(p)]
        self.assertEqual(shapes["reified_object_shape"], "mixed")
        self.assertEqual(shapes["direct_spo_shape"], "datatype")
        self.assertEqual(shapes["validated_against"], "reified")

    # (b)
    def test_reified_object_only_and_direct_object_only_no_mismatch(self):
        g = self._base_graph()
        p = self._declare_local_object_property(g, "dynObj")
        _add_reified_relation(g, ETD.r1, ETD.s1, p, ETD.o1, emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, ETD.o2, emit_direct=True)
        ok, metrics, reasons = self._run_gate(g)
        self.assertNotIn("etd_declared_vs_observed_kind_mismatch", reasons)
        self.assertEqual(int(metrics["etd_declared_vs_observed_kind_mismatch_count"]), 0)

    # (c)
    def test_reified_datatype_only_with_datatype_property_no_mismatch(self):
        g = self._base_graph()
        p = URIRef(f"{ETD}dynLit")
        g.add((p, RDF.type, OWL.DatatypeProperty))
        g.add((p, RDFS.label, Literal("dynLit", lang="en")))
        g.add((p, RDFS.comment, Literal("local", lang="en")))
        g.add((p, ETD.lifecycleStatus, Literal("reserved")))
        g.add((p, ETD.lifecycleNote, Literal("x", lang="en")))
        _add_reified_relation(g, ETD.r1, ETD.s1, p, Literal("v1"), emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, Literal("v2"), emit_direct=True)
        ok, metrics, reasons = self._run_gate(g)
        self.assertNotIn("etd_declared_vs_observed_kind_mismatch", reasons)
        self.assertEqual(int(metrics["etd_declared_vs_observed_kind_mismatch_count"]), 0)

    # (d)
    def test_true_declaration_mismatch_still_fails(self):
        g = self._base_graph()
        # declared DatatypeProperty but every reified object is a resource
        p = URIRef(f"{ETD}dynBad")
        g.add((p, RDF.type, OWL.DatatypeProperty))
        g.add((p, RDFS.label, Literal("dynBad", lang="en")))
        g.add((p, RDFS.comment, Literal("local", lang="en")))
        g.add((p, ETD.lifecycleStatus, Literal("reserved")))
        g.add((p, ETD.lifecycleNote, Literal("x", lang="en")))
        _add_reified_relation(g, ETD.r1, ETD.s1, p, ETD.o1, emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, ETD.o2, emit_direct=True)
        ok, metrics, reasons = self._run_gate(g)
        self.assertIn("etd_declared_vs_observed_kind_mismatch", reasons)
        self.assertEqual(metrics["etd_declared_vs_observed_kind_mismatch"][str(p)],
                         {"declared": "datatype", "observed": "object"})

    def test_true_mismatch_via_direct_fallback_when_no_reified_population(self):
        # hasEvidence (schema ObjectProperty) used with a literal on a
        # Relation node but NOT as etd:predicate -> no reified population ->
        # direct-SPO fallback still catches the mismatch (mirrors the
        # pre-existing test_etd_predicate_policy fixture).
        g = self._base_graph()
        rel = URIRef(f"{ETD}rel_x")
        g.add((rel, RDF.type, ETD.Relation))
        g.add((rel, ETD.hasEvidence, Literal("not-an-iri")))
        ok, metrics, reasons = self._run_gate(g)
        self.assertIn("etd_declared_vs_observed_kind_mismatch", reasons)

    # (e)
    def test_registered_literal_allowed_reconciliation_still_works(self):
        g = self._base_graph()
        # trainedOn is schema-declared ObjectProperty AND PREDICATE_REGISTRY
        # literal_allowed=True; all-literal reified usage must reconcile.
        p = ETD.trainedOn
        _add_reified_relation(g, ETD.r1, ETD.s1, p, Literal("a public web corpus"), emit_direct=True)
        _add_reified_relation(g, ETD.r2, ETD.s1, p, Literal("internal logs"), emit_direct=True)
        ok, metrics, reasons = self._run_gate(g)
        self.assertNotIn("etd_declared_vs_observed_kind_mismatch", reasons)
        rec = metrics["etd_declared_vs_observed_kind_reconciled"].get(str(p), {})
        self.assertEqual(rec.get("observed"), "datatype")
        self.assertIn("literal_allowed=True", rec.get("reason", ""))

    # (f)
    def test_gate_does_not_depend_on_direct_spo_emission(self):
        # A fully mixed reified population with ZERO direct SPO triples for
        # the predicate still reconciles -> the gate no longer requires the
        # serializer to emit any direct triple, so direct-SPO suppression
        # policy needs no change.
        g = self._base_graph()
        p = self._declare_local_object_property(g, "dynNoDirect")
        _add_reified_relation(g, ETD.r1, ETD.s1, p, Literal("lit"), emit_direct=False)
        _add_reified_relation(g, ETD.r2, ETD.s2, p, ETD.res, emit_direct=False)
        ok, metrics, reasons = self._run_gate(g)
        self.assertNotIn("etd_declared_vs_observed_kind_mismatch", reasons)
        shapes = metrics["etd_predicate_object_shapes"][str(p)]
        self.assertEqual(shapes["reified_object_shape"], "mixed")
        self.assertEqual(shapes["direct_spo_shape"], "none")
        self.assertEqual(shapes["validated_against"], "reified")


if __name__ == "__main__":
    unittest.main()
