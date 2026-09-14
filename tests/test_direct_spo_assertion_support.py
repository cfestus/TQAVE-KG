"""Direct-SPO assertion invariant.

A plain direct <s> <p> <o> triple may exist only when at least one
surviving reified relation for that exact SPO provides POSITIVE ASSERTED
support:
  * not negated; AND
  * epistemic status is an asserted proposition (explicit 'asserted', or
    missing/legacy-empty for compatibility); AND
  * modality_tags carry no non-asserted modality (questioned / hypothetical
    / uncertain / conditional / normative / reported).

The SAME rule (RDFSerializer._relation_provides_positive_asserted_support)
gates both the initial emission and _enforce_direct_spo_assertion_invariant.
Reified relations + Evidence are always preserved regardless.

Synthetic entities only; no LLM / network.
"""
import importlib.util
import unittest

from config import Config
from models import Entity

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None
ETD = Config.NAMESPACES["etd"]


class _Registry:
    def __init__(self, entities):
        self._entities = list(entities)

    def get_all_entities(self):
        return self._entities


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class DirectSPOAssertionSupportTests(unittest.TestCase):
    _S = f"{ETD}entity_dspo_subject"
    _T = f"{ETD}entity_dspo_object"

    def _serialize(self, rel_metas):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer

        entities = [
            Entity(uri=self._S, label="Alpha System", entity_type="Concept", confidence=0.8, metadata={}),
            Entity(uri=self._T, label="Some Outcome", entity_type="Concept", confidence=0.8, metadata={}),
        ]
        relations = []
        for i, extra in enumerate(rel_metas):
            meta = {
                "discourse_id": "d1", "field": "theme.description", "chunk_index": 0,
                "source": "llm_relation_extractor", "relation_kind": "extracted_semantic",
                "subject_surface_form": "Alpha System", "object_surface_form": "Some Outcome",
                "evidence": "Alpha System predicted some outcome in the source text.",
                "evidence_quote": "Alpha System predicted some outcome in the source text.",
            }
            meta.update(extra)
            relations.append({
                "uri": f"{ETD}rel_dspo_{i}",
                "source": self._S, "relation": "predicted", "target": self._T,
                "confidence": 0.9, "metadata": meta,
            })
        ann = {
            "themes": {}, "questions": {},
            "answers": {"d1": {"relations": relations, "metadata": {"text": "some source text"}}},
            "votes": {}, "experts": {}, "documents": {},
            "metadata": {"datasetId": "unit_ds", "runId": "run-unit"},
        }
        s = RDFSerializer(annotations_dict=ann, registry=_Registry(entities),
                          namespaces=Config.NAMESPACES, prune_isolated_entities=False)
        return s, Namespace(ETD)

    def _direct_present(self, s, ns):
        from rdflib import URIRef
        return (URIRef(self._S), ns.predicted, URIRef(self._T)) in s.graph

    def _reified_and_evidence_ok(self, s, ns, expected_rel_count):
        from rdflib import RDF, URIRef
        rels = [r for r in s.graph.subjects(RDF.type, ns.Relation)
                if (r, ns.predicate, ns.predicted) in s.graph
                and (URIRef(self._S) in set(s.graph.objects(r, ns.subject)))]
        self.assertEqual(len(rels), expected_rel_count,
                         f"expected {expected_rel_count} reified predicted relations, got {len(rels)}")
        for r in rels:
            self.assertEqual(set(s.graph.objects(r, ns.object)), {URIRef(self._T)})
            ev = list(s.graph.objects(r, ns.hasEvidence))
            self.assertTrue(ev, f"{r}: reified relation lost its Evidence node")
            for e in ev:
                self.assertTrue(list(s.graph.objects(e, ns.evidenceText)))

    # ---- single-relation scenarios ----

    def test_asserted_only_direct_spo_present(self):
        s, ns = self._serialize([{"negated": False, "epistemic_status": "asserted", "modality_tags": "asserted"}])
        self.assertTrue(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 1)

    def test_reported_only_suppressed(self):
        s, ns = self._serialize([{"negated": False, "epistemic_status": "reported", "modality_tags": "reported"}])
        self.assertFalse(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 1)

    def test_conditional_only_suppressed(self):
        s, ns = self._serialize([{"negated": False, "epistemic_status": "conditional", "modality_tags": "conditional"}])
        self.assertFalse(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 1)

    def test_normative_only_suppressed(self):
        s, ns = self._serialize([{"negated": False, "epistemic_status": "normative", "modality_tags": "normative"}])
        self.assertFalse(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 1)

    def test_questioned_hypothetical_uncertain_suppressed(self):
        for st in ("questioned", "hypothetical", "uncertain"):
            with self.subTest(status=st):
                s, ns = self._serialize([{"negated": False, "epistemic_status": st, "modality_tags": st}])
                self.assertFalse(self._direct_present(s, ns))
                self._reified_and_evidence_ok(s, ns, 1)

    def test_negated_asserted_suppressed(self):
        s, ns = self._serialize([{"negated": True, "epistemic_status": "asserted", "modality_tags": "asserted"}])
        self.assertFalse(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 1)

    def test_epistemic_asserted_but_modality_tags_non_asserted_suppressed(self):
        s, ns = self._serialize([{"negated": False, "epistemic_status": "asserted",
                                  "modality_tags": "questioned|conditional"}])
        self.assertFalse(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 1)

    def test_legacy_empty_status_remains_compatible(self):
        # no epistemic_status, no modality_tags, not negated -> unchanged
        # (missing status treated as asserted for backward compatibility)
        s, ns = self._serialize([{"negated": False}])
        self.assertTrue(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 1)

    def test_legacy_empty_status_with_non_asserted_modality_tags_suppressed(self):
        s, ns = self._serialize([{"negated": False, "modality_tags": "reported"}])
        self.assertFalse(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 1)

    # ---- group scenarios (same exact SPO) ----

    def test_asserted_plus_conditional_sibling_retains_direct_spo(self):
        s, ns = self._serialize([
            {"negated": False, "epistemic_status": "asserted", "modality_tags": "asserted"},
            {"negated": False, "epistemic_status": "conditional", "modality_tags": "conditional"},
        ])
        self.assertTrue(self._direct_present(s, ns))
        # both reified relations survive (distinct qualification)
        self._reified_and_evidence_ok(s, ns, 2)

    def test_reported_plus_conditional_siblings_only_suppressed(self):
        s, ns = self._serialize([
            {"negated": False, "epistemic_status": "reported", "modality_tags": "reported"},
            {"negated": False, "epistemic_status": "conditional", "modality_tags": "conditional"},
        ])
        self.assertFalse(self._direct_present(s, ns))
        self._reified_and_evidence_ok(s, ns, 2)


@unittest.skipUnless(HAS_RDFLIB, "rdflib required")
class SharedRuleConsistencyTests(unittest.TestCase):
    """The emission gate and the cleanup invariant must agree because they
    call the SAME helper."""

    def test_same_helper_used_by_emission_and_invariant(self):
        from serializers.rdf_serializer import RDFSerializer
        import inspect
        emit_src = inspect.getsource(RDFSerializer._direct_spo_suppressed_by_assertion_status)
        inv_src = inspect.getsource(RDFSerializer._enforce_direct_spo_assertion_invariant)
        self.assertIn("_relation_provides_positive_asserted_support", emit_src)
        self.assertIn("_relation_provides_positive_asserted_support", inv_src)

    def test_helper_classifications(self):
        from serializers.rdf_serializer import RDFSerializer
        s = RDFSerializer.__new__(RDFSerializer)
        f = s._relation_provides_positive_asserted_support
        # positive
        self.assertTrue(f(negated=False, epistemic_status="asserted", modality_tags="asserted"))
        self.assertTrue(f(negated=False, epistemic_status="", modality_tags=""))
        self.assertTrue(f(negated=None, epistemic_status=None, modality_tags=None))
        # non-positive
        for st in ("reported", "conditional", "normative", "questioned", "hypothetical", "uncertain"):
            self.assertFalse(f(negated=False, epistemic_status=st, modality_tags=""), st)
        self.assertFalse(f(negated=True, epistemic_status="asserted", modality_tags="asserted"))
        self.assertFalse(f(negated=False, epistemic_status="asserted", modality_tags="questioned|conditional"))
        self.assertFalse(f(negated=False, epistemic_status="", modality_tags=["reported"]))


if __name__ == "__main__":
    unittest.main()
