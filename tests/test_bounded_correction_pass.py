"""Regression tests for the bounded root-cause correction pass driven by
the 2026-09-07 semantic-regeneration audit.

Sections (matching the audit report's item numbering):

  1. Endpoint/surface-form mismatches -- ordering fix (subject_text/
     object_text -> subject_surface_form/object_surface_form
     reconciliation now runs BEFORE the compatibility check) plus the
     final, recovery-capable invariant pass that runs after
     consolidation.
  2. Non-asserted direct SPO / consolidation -- duplicate-group identity
     now includes negated/epistemic_status, and a final invariant pass
     removes any direct triple with no surviving positively-asserted
     justification.
  3. Same-chunk identical-referent duplicate entities -- discourse-
     scoped entity reuse in _resolve_surface_entity.
  4. Correction lifecycle -- respectively-split correction is now
     instrumented, and a stale, unsplit duplicate from a DIFFERENT
     occurrence of the same statistic in the same discourse is
     superseded rather than left to coexist.
  5. Targeted deterministic spot fixes -- bare expletive "there",
     hosting-platform predicate correction for literal objects, the
     passive-dimension-as-agent guard's evidence-text fallback.
  6. Semantic-quality instrumentation report -- events are recorded and
     output/semantic_quality_report.json is actually written.

No test calls the live OpenAI API; all extractor output is hand-built.
"""

import importlib.util
import json
import os
import tempfile
import unittest

from config import Config
from models import Entity, Relation

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


def _base_annotations(relations, discourse_id="d1"):
    return {
        "themes": {}, "questions": {},
        "answers": {discourse_id: {"relations": relations}},
        "votes": {}, "experts": {}, "documents": {},
        "metadata": {},
    }


def _make_bare_pipeline():
    from pipeline import EnhancedSemanticPipeline
    from utils.helpers import GlobalEntityRegistry
    from ontology.mapper import OntologyMapper
    from utils.entity_normalizer import EntityNormalizer

    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    return pipeline  # quality_events / _chunk_entity_identity_index are lazy @property-backed


def _register_entity(pipeline, *, uri, label, entity_type="Concept"):
    ent = Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.9, metadata={"discourse_id": "d1"})
    pipeline._ensure_entity_ontology_grounding(ent)
    return pipeline.global_registry.get_or_create(ent)


# ---------------------------------------------------------------------
# 1. Endpoint/surface-form mismatch ordering + final invariant pass
# ---------------------------------------------------------------------

@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class EndpointSurfaceMismatchOrderingTests(unittest.TestCase):
    """The compatibility check must see the reconciled surface form even
    for a relation whose predicate is not in SEMANTIC_RELATION_PREDICATES
    (so SemanticRelationPolicy.validate() never ran and never populated
    subject_surface_form/object_surface_form itself) -- previously the
    check saw an EMPTY surface form for such relations and always passed
    them through regardless of a genuine mismatch in subject_text."""

    etd_base = Config.NAMESPACES["etd"]

    def _build(self, *, subject_label, subject_text, predicate="flagged"):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        source_uri = f"{self.etd_base}entity_source_ord00"
        target_uri = f"{self.etd_base}entity_target_ord01"
        entities = [
            Entity(uri=source_uri, label=subject_label, entity_type="Concept", confidence=0.8, metadata={"surface_form": subject_label}),
            Entity(uri=target_uri, label="Some Outcome", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Some Outcome"}),
        ]
        relations = [
            {
                "source": source_uri,
                "relation": predicate,  # deliberately NOT in PREDICATE_REGISTRY
                "target": target_uri,
                "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1",
                    "field": "answer.description",
                    "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    "subject_text": subject_text,
                    "object_text": "Some Outcome",
                },
            }
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        return serializer, etd_ns, source_uri, target_uri

    def test_unregistered_predicate_mismatch_is_now_caught(self):
        """Subject entity is 'COMPAS' but subject_text says 'Black
        defendants' -- previously slipped through because 'flagged' is
        not a registered predicate, so subject_surface_form was empty
        when the compatibility check ran."""
        serializer, etd_ns, s, t = self._build(subject_label="COMPAS", subject_text="Black defendants")
        from rdflib import URIRef
        rel_nodes = list(serializer.graph.subjects(etd_ns.subject, URIRef(s)))
        self.assertEqual(rel_nodes, [], "mismatched subject must not survive as a reified relation")

    def test_unregistered_predicate_compatible_case_is_kept(self):
        serializer, etd_ns, s, t = self._build(subject_label="COMPAS", subject_text="COMPAS")
        from rdflib import URIRef
        rel_nodes = list(serializer.graph.subjects(etd_ns.subject, URIRef(s)))
        self.assertEqual(len(rel_nodes), 1)

    def test_reconciled_surface_form_is_persisted(self):
        serializer, etd_ns, s, t = self._build(subject_label="COMPAS", subject_text="COMPAS")
        from rdflib import URIRef
        rel_nodes = list(serializer.graph.subjects(etd_ns.subject, URIRef(s)))
        self.assertEqual(len(rel_nodes), 1)
        forms = [str(v) for v in serializer.graph.objects(rel_nodes[0], etd_ns.subject_surface_form)]
        self.assertEqual(forms, ["COMPAS"])


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class EndpointSurfaceCompatibilityFinalInvariantTests(unittest.TestCase):
    """The post-consolidation final pass (_enforce_endpoint_surface_
    compatibility_final) must recover a valid grounded endpoint when an
    unambiguous same-discourse candidate exists, and must only drop when
    no such candidate exists."""

    etd_base = Config.NAMESPACES["etd"]

    def test_final_pass_recovers_valid_endpoint_within_discourse(self):
        """Two relations in the same discourse: one correctly bound, one
        whose subject got bound to the WRONG entity (simulating a stale
        rebind) but whose surface form matches a DIFFERENT entity that is
        already part of this same discourse -- must be repointed, not
        dropped."""
        from rdflib import Namespace, URIRef, RDFS, RDF
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        wrong_uri = f"{self.etd_base}entity_wrong_final00"
        right_uri = f"{self.etd_base}entity_right_final01"
        obj_uri = f"{self.etd_base}entity_obj_final02"
        entities = [
            Entity(uri=wrong_uri, label="COMPAS", entity_type="Concept", confidence=0.8, metadata={"surface_form": "COMPAS"}),
            Entity(uri=right_uri, label="Black Defendants", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Black Defendants"}),
            Entity(uri=obj_uri, label="High Risk", entity_type="Concept", confidence=0.8, metadata={"surface_form": "High Risk"}),
        ]
        relations = [
            {
                "source": right_uri,
                "relation": "relatedTo",
                "target": obj_uri,
                "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    "subject_surface_form": "Black Defendants", "object_surface_form": "High Risk",
                },
            },
            {
                "source": wrong_uri,
                "relation": "flagged",
                "target": obj_uri,
                "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    # Deliberately a slight VARIANT of the recovered
                    # entity's own canonical label (not byte-identical),
                    # so a test that overwrote surface_form with the
                    # canonical label instead of preserving the original
                    # extracted text would be caught.
                    "subject_surface_form": "the Black defendants", "object_surface_form": "High Risk",
                    "evidence": "COMPAS flagged the Black defendants as high risk.",
                    "start_char": 12, "end_char": 60,
                },
            },
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph

        flagged_rels = list(g.subjects(etd_ns.predicate, etd_ns.flagged))
        self.assertEqual(len(flagged_rels), 1, "the mismatched relation must survive via recovery, not be dropped")
        rel_node = flagged_rels[0]
        subj = list(g.objects(rel_node, etd_ns.subject))
        self.assertEqual(subj, [URIRef(right_uri)], "must be repointed to the discourse-scoped matching entity")
        self.assertGreaterEqual(
            serializer.endpoint_surface_compatibility_final["relations_recovered"], 1,
        )

        # Item 2 (bounded preflight): the ORIGINAL extracted surface text
        # and evidence/span must survive the rebind verbatim -- recovery
        # must never overwrite genuine source evidence with the
        # canonical entity label merely to make the two agree.
        surface_forms = [str(v) for v in g.objects(rel_node, etd_ns.subject_surface_form)]
        self.assertEqual(surface_forms, ["the Black defendants"], "original captured surface text must be preserved verbatim, not rewritten to the canonical label")
        evidence_nodes = list(g.objects(rel_node, etd_ns.hasEvidence))
        self.assertEqual(len(evidence_nodes), 1, "the original evidence link must survive the rebind")
        evidence_props = {str(p) for p in g.predicates(evidence_nodes[0], None)}
        evidence_text_pred = next((p for p in evidence_props if p.endswith("evidenceText") or p.endswith("#evidence")), None)
        self.assertIsNotNone(evidence_text_pred, f"expected an evidence-text predicate on the evidence node; found {sorted(evidence_props)}")

    def test_final_pass_drops_when_no_recoverable_endpoint(self):
        from rdflib import Namespace, URIRef
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        wrong_uri = f"{self.etd_base}entity_wrong_final10"
        obj_uri = f"{self.etd_base}entity_obj_final11"
        entities = [
            Entity(uri=wrong_uri, label="COMPAS", entity_type="Concept", confidence=0.8, metadata={"surface_form": "COMPAS"}),
            Entity(uri=obj_uri, label="High Risk", entity_type="Concept", confidence=0.8, metadata={"surface_form": "High Risk"}),
        ]
        relations = [
            {
                "source": wrong_uri,
                "relation": "flagged",
                "target": obj_uri,
                "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    "subject_surface_form": "Black Defendants", "object_surface_form": "High Risk",
                },
            },
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph
        flagged_rels = list(g.subjects(etd_ns.predicate, etd_ns.flagged))
        self.assertEqual(flagged_rels, [], "no recoverable endpoint exists -- relation must be dropped, not left mismatched")
        self.assertGreaterEqual(serializer.endpoint_surface_compatibility_final["relations_dropped"], 1)


# ---------------------------------------------------------------------
# 2. Non-asserted direct SPO / consolidation
# ---------------------------------------------------------------------

@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class ConsolidationAssertionStatusTests(unittest.TestCase):
    etd_base = Config.NAMESPACES["etd"]

    def _build(self, rel_metas):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        source_uri = f"{self.etd_base}entity_source_cons00"
        target_uri = f"{self.etd_base}entity_target_cons01"
        entities = [
            Entity(uri=source_uri, label="GPT-4o", entity_type="Concept", confidence=0.8, metadata={}),
            Entity(uri=target_uri, label="Some Outcome", entity_type="Concept", confidence=0.8, metadata={}),
        ]
        relations = []
        for i, extra in enumerate(rel_metas):
            meta = {
                "discourse_id": "d1", "field": "theme.description", "chunk_index": 0,
                "source": "llm_relation_extractor",
                "subject_surface_form": "GPT-4o", "object_surface_form": "Some Outcome",
            }
            meta.update(extra)
            relations.append({
                "uri": f"{self.etd_base}rel_cons_test_{i}",
                "source": source_uri, "relation": "predicted", "target": target_uri,
                "confidence": 0.9, "metadata": meta,
            })
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        return serializer, etd_ns, source_uri, target_uri

    def test_positive_and_negated_duplicate_are_both_preserved_distinct(self):
        """Section 2's confirmed defect shape: two occurrences of the
        identical SPO, one positively asserted, one negated/reported --
        both must survive as distinct reified relations, and the direct
        triple must exist (justified by the positive one) without ever
        having carried the negated one's status."""
        from rdflib import URIRef
        serializer, etd_ns, s, t = self._build([
            {"negated": False, "epistemic_status": "asserted"},
            {"negated": True, "epistemic_status": "reported"},
        ])
        g = serializer.graph
        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(s)))
        self.assertEqual(len(rel_nodes), 2, "positive and negated versions of the same SPO must NOT be merged")

        direct = list(g.triples((URIRef(s), etd_ns.predicted, URIRef(t))))
        self.assertEqual(len(direct), 1, "direct triple must exist, justified by the surviving positive relation")

    def test_true_duplicate_same_status_still_merges(self):
        """Two occurrences with IDENTICAL negated/epistemic_status are a
        genuine redundant re-extraction and must still merge as before."""
        from rdflib import URIRef
        serializer, etd_ns, s, t = self._build([
            {"negated": False, "epistemic_status": "asserted", "confidence": 0.9},
            {"negated": False, "epistemic_status": "asserted"},
        ])
        g = serializer.graph
        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(s)))
        self.assertEqual(len(rel_nodes), 1, "true duplicates (same assertion status) must still be consolidated")

    def test_only_negated_occurrence_leaves_no_direct_triple(self):
        from rdflib import URIRef
        serializer, etd_ns, s, t = self._build([{"negated": True, "epistemic_status": "reported"}])
        g = serializer.graph
        direct = list(g.triples((URIRef(s), etd_ns.predicted, URIRef(t))))
        self.assertEqual(direct, [], "a lone negated relation must never leave an orphaned direct triple")

    def test_two_negated_duplicates_still_consolidate(self):
        """Two occurrences that AGREE on negated=True (a genuine redundant
        re-extraction of the same negated fact) must still merge into
        one reified relation, and must still have no direct triple."""
        from rdflib import URIRef
        serializer, etd_ns, s, t = self._build([
            {"negated": True, "epistemic_status": "reported", "confidence": 0.9},
            {"negated": True, "epistemic_status": "reported"},
        ])
        g = serializer.graph
        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(s)))
        self.assertEqual(len(rel_nodes), 1, "two negated duplicates with identical status must still consolidate")
        direct = list(g.triples((URIRef(s), etd_ns.predicted, URIRef(t))))
        self.assertEqual(direct, [])

    def test_two_hypothetical_duplicates_still_consolidate(self):
        """Two occurrences that AGREE on epistemic_status=hypothetical
        must still merge, and must still have no direct triple."""
        from rdflib import URIRef
        serializer, etd_ns, s, t = self._build([
            {"negated": False, "epistemic_status": "hypothetical", "confidence": 0.9},
            {"negated": False, "epistemic_status": "hypothetical"},
        ])
        g = serializer.graph
        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(s)))
        self.assertEqual(len(rel_nodes), 1, "two hypothetical duplicates with identical status must still consolidate")
        direct = list(g.triples((URIRef(s), etd_ns.predicted, URIRef(t))))
        self.assertEqual(direct, [])

    def test_questioned_and_positive_are_preserved_distinct(self):
        """A questioned restatement and a positively-asserted occurrence
        of the same SPO are DISTINCT assertions (one is a checklist-style
        query, the other a fact) -- both must survive, and the direct
        triple must exist, justified by the positive one."""
        from rdflib import URIRef
        serializer, etd_ns, s, t = self._build([
            {"negated": False, "epistemic_status": "asserted"},
            {"negated": False, "epistemic_status": "questioned"},
        ])
        g = serializer.graph
        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(s)))
        self.assertEqual(len(rel_nodes), 2)
        direct = list(g.triples((URIRef(s), etd_ns.predicted, URIRef(t))))
        self.assertEqual(len(direct), 1)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class DirectSPOAssertionInvariantEnforcementTests(unittest.TestCase):
    """Directly exercises _enforce_direct_spo_assertion_invariant as the
    final, authoritative backstop -- simulates an orphaned direct triple
    surviving some upstream mechanism, independent of consolidation."""

    etd_base = Config.NAMESPACES["etd"]

    def test_orphaned_direct_triple_with_no_positive_relation_is_removed(self):
        from rdflib import Namespace, URIRef, RDF, RDFS, Literal, XSD
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        source_uri = f"{self.etd_base}entity_source_orphan00"
        entities = [Entity(uri=source_uri, label="System X", entity_type="Concept", confidence=0.8, metadata={})]
        relations = [
            {
                "source": source_uri, "relation": "predicted", "target": "bad outcome",
                "target_literal": "bad outcome", "target_is_literal": True,
                "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "theme.description", "chunk_index": 0,
                    "source": "llm_relation_extractor", "negated": True, "epistemic_status": "reported",
                    "relation_kind": "extracted_semantic",
                    "subject_surface_form": "System X", "object_surface_form": "bad outcome",
                },
            },
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph

        # Simulate an orphaned direct triple reaching the graph despite
        # suppression (e.g. via some future code path this invariant is
        # meant to guard against generally, not just the one confirmed
        # consolidation-driven cause).
        obj_literal = Literal("bad outcome", datatype=XSD.string)
        g.add((URIRef(source_uri), etd_ns.predicted, obj_literal))
        result = serializer._enforce_direct_spo_assertion_invariant(etd_ns)

        direct = list(g.triples((URIRef(source_uri), etd_ns.predicted, obj_literal)))
        self.assertEqual(direct, [], "orphaned direct triple with no positively-asserted relation must be removed")
        self.assertGreaterEqual(result["orphaned_direct_triples_removed"], 1)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class OrphanEvidenceReconciliationTests(unittest.TestCase):
    """Targeted correction pass, item 1: _reconcile_orphan_evidence_nodes
    must delete an Evidence node only when NO surviving etd:Relation
    still references it via etd:hasEvidence, preserving shared Evidence
    and its provenance, and the full RDFSerializer pipeline must leave
    zero orphan Evidence nodes end to end (matching kg_quality_gate.py's
    own orphan_evidence_count definition exactly)."""

    etd_base = Config.NAMESPACES["etd"]

    def _build(self, *, subject_label, subject_surface, predicate, mismatched):
        """One relation whose subject is deliberately incompatible with
        its captured surface form and has no recoverable candidate in
        its discourse -- guaranteed to be dropped by
        _enforce_endpoint_surface_compatibility_final, exercising the
        real drop path this fix targets (not a hand-inserted triple)."""
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        source_uri = f"{self.etd_base}entity_source_{subject_label.lower().replace(' ', '_')}"
        target_uri = f"{self.etd_base}entity_target_{subject_label.lower().replace(' ', '_')}"
        entities = [
            Entity(uri=source_uri, label=subject_label, entity_type="Concept", confidence=0.8, metadata={"surface_form": subject_label}),
            Entity(uri=target_uri, label="Outcome", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Outcome"}),
        ]
        relations = [
            {
                "source": source_uri, "relation": predicate, "target": target_uri, "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    "subject_surface_form": subject_surface, "object_surface_form": "Outcome",
                    "evidence": f"{subject_surface} {predicate} Outcome.", "evidence_quote": f"{subject_surface} {predicate} Outcome.",
                },
            },
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        return serializer, etd_ns

    def test_sole_owner_relation_dropped_evidence_removed(self):
        serializer, etd_ns = self._build(
            subject_label="COMPAS", subject_surface="totally unrelated text", predicate="flagged", mismatched=True,
        )
        from rdflib import RDF
        g = serializer.graph
        self.assertGreaterEqual(serializer.endpoint_surface_compatibility_final["relations_dropped"], 1)
        evidence_nodes = list(g.subjects(RDF.type, etd_ns.Evidence))
        self.assertEqual(evidence_nodes, [], "the sole owner's Evidence node must be removed, not left orphaned")
        self.assertGreaterEqual(serializer.orphan_evidence_reconciliation["orphaned_evidence_removed"], 1)

    def test_one_of_two_owners_dropped_shared_evidence_retained(self):
        """Two relations sharing ONE Evidence node (same hasEvidence
        target): one gets dropped by endpoint-mismatch, the other
        survives -- the shared Evidence node must NOT be deleted, and
        the surviving relation's link to it must remain intact."""
        from rdflib import Namespace, RDF, URIRef
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        good_uri = f"{self.etd_base}entity_good_shared"
        bad_uri = f"{self.etd_base}entity_bad_shared"
        target_uri = f"{self.etd_base}entity_target_shared"
        entities = [
            Entity(uri=good_uri, label="Good Subject", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Good Subject"}),
            Entity(uri=bad_uri, label="COMPAS", entity_type="Concept", confidence=0.8, metadata={"surface_form": "COMPAS"}),
            Entity(uri=target_uri, label="Outcome", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Outcome"}),
        ]
        shared_evidence_text = "Good Subject and something unrelated both relate to Outcome."
        relations = [
            {
                "uri": f"{self.etd_base}rel_shared_good",
                "source": good_uri, "relation": "relatedTo", "target": target_uri, "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    "subject_surface_form": "Good Subject", "object_surface_form": "Outcome",
                    "evidence": shared_evidence_text, "evidence_quote": shared_evidence_text,
                    "char_start": 0, "char_end": len(shared_evidence_text),
                },
            },
            {
                "uri": f"{self.etd_base}rel_shared_bad",
                "source": bad_uri, "relation": "flagged", "target": target_uri, "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    "subject_surface_form": "totally unrelated text", "object_surface_form": "Outcome",
                    "evidence": shared_evidence_text, "evidence_quote": shared_evidence_text,
                    "char_start": 0, "char_end": len(shared_evidence_text),
                },
            },
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph

        # The bad relation must have been dropped (mismatched, no
        # recoverable candidate); the good one must survive.
        good_rels = list(g.subjects(etd_ns.subject, URIRef(good_uri)))
        bad_rels = list(g.subjects(etd_ns.subject, URIRef(bad_uri)))
        self.assertEqual(len(good_rels), 1, "the good relation must survive")
        self.assertEqual(bad_rels, [], "the mismatched relation must be dropped")

        # Evidence nodes reaching serialization would only be identical
        # (shared) if _add_relations()'s evidence-key computation
        # produces the same URI for both -- confirm that happened, then
        # confirm it survived because the good relation still needs it.
        evidence_nodes = list(g.subjects(RDF.type, etd_ns.Evidence))
        self.assertEqual(len(evidence_nodes), 1, "both relations shared one Evidence node by construction (identical evidence/char span)")
        remaining_links = list(g.objects(good_rels[0], etd_ns.hasEvidence))
        self.assertEqual(remaining_links, evidence_nodes, "the surviving relation's evidence link and text must remain intact")
        evidence_text = list(g.objects(evidence_nodes[0], etd_ns.evidenceText))
        self.assertEqual([str(v) for v in evidence_text], [shared_evidence_text])

    def test_final_graph_contains_zero_orphan_evidence_nodes(self):
        """Mirrors kg_quality_gate.py's own orphan_evidence_count metric
        directly: after a mix of surviving, recovered, and dropped
        relations, zero Evidence nodes may have zero inbound
        etd:hasEvidence references."""
        from rdflib import Namespace, RDF
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        entities = [
            Entity(uri=f"{self.etd_base}e_ok", label="OK Subject", entity_type="Concept", confidence=0.8, metadata={"surface_form": "OK Subject"}),
            Entity(uri=f"{self.etd_base}e_bad", label="COMPAS", entity_type="Concept", confidence=0.8, metadata={"surface_form": "COMPAS"}),
            Entity(uri=f"{self.etd_base}e_target", label="Outcome", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Outcome"}),
        ]
        relations = [
            {
                "source": f"{self.etd_base}e_ok", "relation": "relatedTo", "target": f"{self.etd_base}e_target", "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0, "source": "llm_relation_extractor",
                    "subject_surface_form": "OK Subject", "object_surface_form": "Outcome",
                    "evidence": "OK Subject relates to Outcome.", "evidence_quote": "OK Subject relates to Outcome.",
                },
            },
            {
                "source": f"{self.etd_base}e_bad", "relation": "flagged", "target": f"{self.etd_base}e_target", "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0, "source": "llm_relation_extractor",
                    "subject_surface_form": "nonsense mismatched text", "object_surface_form": "Outcome",
                    "evidence": "Something entirely unrelated happened.", "evidence_quote": "Something entirely unrelated happened.",
                },
            },
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph

        evidence_nodes = set(g.subjects(RDF.type, etd_ns.Evidence))
        inbound = {}
        for rel in g.subjects(RDF.type, etd_ns.Relation):
            for ev in g.objects(rel, etd_ns.hasEvidence):
                inbound[ev] = inbound.get(ev, 0) + 1
        orphans = [ev for ev in evidence_nodes if inbound.get(ev, 0) == 0]
        self.assertEqual(orphans, [], f"final graph must contain zero orphan Evidence nodes, found: {orphans}")

    def test_surviving_relation_evidence_and_provenance_intact(self):
        """A relation untouched by any drop/recovery mechanism must keep
        its full evidence (text, span) and hasSourceEntity provenance
        exactly as before this fix -- the reconciliation pass must not
        perturb anything about relations/evidence it doesn't need to
        touch."""
        from rdflib import Namespace, RDF, URIRef
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        source_uri = f"{self.etd_base}entity_intact_subject"
        target_uri = f"{self.etd_base}entity_intact_target"
        theme_uri = f"{self.etd_base}theme_intact"
        entities = [
            Entity(uri=source_uri, label="Intact Subject", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Intact Subject"}),
            Entity(uri=target_uri, label="Intact Outcome", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Intact Outcome"}),
        ]
        evidence_text = "Intact Subject relates to Intact Outcome, genuinely."
        relations = [
            {
                "source": source_uri, "relation": "relatedTo", "target": target_uri, "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0, "source": "llm_relation_extractor",
                    "subject_surface_form": "Intact Subject", "object_surface_form": "Intact Outcome",
                    "evidence": evidence_text, "evidence_quote": evidence_text,
                    "source_entity_uri": theme_uri,
                },
            },
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph

        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(source_uri)))
        self.assertEqual(len(rel_nodes), 1)
        evidence_nodes = list(g.objects(rel_nodes[0], etd_ns.hasEvidence))
        self.assertEqual(len(evidence_nodes), 1)
        self.assertEqual([str(v) for v in g.objects(evidence_nodes[0], etd_ns.evidenceText)], [evidence_text])
        self.assertIn(URIRef(theme_uri), list(g.objects(rel_nodes[0], etd_ns.hasSourceEntity)))
        self.assertEqual(serializer.orphan_evidence_reconciliation["orphaned_evidence_removed"], 0)


# ---------------------------------------------------------------------
# 3. Same-chunk identical-referent duplicate entities
# ---------------------------------------------------------------------

class SameChunkEntityReuseTests(unittest.TestCase):
    def test_identical_label_within_discourse_is_reused_across_units(self):
        """Simulates the $765M case: two different semantic units of the
        SAME discourse each resolve an object surface form with the
        IDENTICAL label but a DIFFERENT preferred_type -- must resolve to
        ONE entity, not two."""
        pipeline = _make_bare_pipeline()
        local_entities_unit1 = []
        ent1 = pipeline._resolve_surface_entity(
            surface="$765M+ Annually In Excessive Interest Payments",
            preferred_type="money",
            local_entities=local_entities_unit1,
            discourse_id="d9",
            field="theme.description",
            origin_key="unit1",
            unit={"section": "Public Datasets", "start_offset": 0},
            unit_text="Results in $765M+ annually in excessive interest payments.",
        )
        local_entities_unit2 = []  # a DIFFERENT unit's own local list
        ent2 = pipeline._resolve_surface_entity(
            surface="$765M+ Annually In Excessive Interest Payments",
            preferred_type="EconomicImpact",
            local_entities=local_entities_unit2,
            discourse_id="d9",
            field="theme.description",
            origin_key="unit2",
            unit={"section": "Ethical Challenges", "start_offset": 0},
            unit_text="Results in $765M+ annually in excessive interest payments.",
        )
        self.assertEqual(str(ent1.uri), str(ent2.uri), "identical label within one discourse must resolve to ONE entity")
        reuse_events = [e for e in pipeline.quality_events if e["category"] == "same_chunk_entity_reused"]
        self.assertEqual(len(reuse_events), 1)

    def test_conflicting_family_same_label_same_discourse_is_not_merged(self):
        """Negative test (bounded preflight, item 1): a bare label that
        happens to be reused for two STRUCTURALLY DIFFERENT kinds of
        referent within the same discourse (e.g. "Model" once meaning a
        specific research paper/analysis, once meaning a person) must
        NOT be merged just because the type-bucket-override mechanism
        exists for the $765M-style case -- a family conflict (paper_or_
        analysis vs person) is much stronger evidence of a genuinely
        different referent than a mere type-label disagreement within
        the same family."""
        pipeline = _make_bare_pipeline()
        local_entities_unit1 = []
        ent1 = pipeline._resolve_surface_entity(
            surface="Smith", preferred_type="Person",
            local_entities=local_entities_unit1, discourse_id="d9",
            field="theme.description", origin_key="unit1",
            unit={"section": "Experts", "start_offset": 0}, unit_text="Smith raised the concern.",
        )
        local_entities_unit2 = []
        ent2 = pipeline._resolve_surface_entity(
            surface="Smith", preferred_type="Dataset",
            local_entities=local_entities_unit2, discourse_id="d9",
            field="theme.description", origin_key="unit2",
            unit={"section": "Public Datasets", "start_offset": 0}, unit_text="The Smith dataset was released in 2020.",
        )
        self.assertNotEqual(
            str(ent1.uri), str(ent2.uri),
            "a Person and a Dataset sharing a bare label must never be merged by this mechanism",
        )
        reuse_events = [e for e in pipeline.quality_events if e["category"] == "same_chunk_entity_reused"]
        self.assertEqual(reuse_events, [], "no reuse should have fired across conflicting families")

    def test_identical_label_in_different_discourse_is_not_merged(self):
        """The discourse-scoped reuse index must never fire across
        discourses. Uses two DIFFERENT preferred_type hints (mirroring
        the real $765M case), so that -- unlike same-label-same-type
        mentions, which EntityNormalizer.canonical_uri() already
        deduplicates globally by design, independent of this mechanism
        -- the two mentions would only ever collide via the discourse-
        scoped reuse index under test."""
        pipeline = _make_bare_pipeline()
        ent1 = pipeline._resolve_surface_entity(
            surface="Model Performance", preferred_type="metric",
            local_entities=[], discourse_id="d1", field="theme.description",
            origin_key="u1", unit={"section": "A", "start_offset": 0}, unit_text="Model Performance is high.",
        )
        ent2 = pipeline._resolve_surface_entity(
            surface="Model Performance", preferred_type="analyticaldimension",
            local_entities=[], discourse_id="d2", field="theme.description",
            origin_key="u2", unit={"section": "A", "start_offset": 0}, unit_text="Model Performance is high.",
        )
        self.assertNotEqual(str(ent1.uri), str(ent2.uri), "reuse must never cross discourse boundaries")
        reuse_events = [e for e in pipeline.quality_events if e["category"] == "same_chunk_entity_reused"]
        self.assertEqual(reuse_events, [], "no reuse should have fired across two different discourses")


# ---------------------------------------------------------------------
# 4. Correction lifecycle: respectively-split + supersession
# ---------------------------------------------------------------------

class RespectivelySplitSupersessionTests(unittest.TestCase):
    def _relation(self, *, subject_text, obj_text, discourse_id="d9", section="A"):
        return Relation(
            source=f"placeholder-subject-{subject_text}", relation="pay", target=obj_text,
            target_literal=obj_text, target_is_literal=True,
            confidence=0.9,
            metadata={
                "subject_text": subject_text, "object_text": obj_text,
                "discourse_id": discourse_id, "section": section,
                "evidence": "Black and Latino borrowers pay 7.9 and 3.6 basis points more respectively.",
                "evidence_quote": "Black and Latino borrowers pay 7.9 and 3.6 basis points more respectively.",
            },
        )

    def test_correction_event_is_recorded(self):
        pipeline = _make_bare_pipeline()
        rel_a = self._relation(subject_text="Black Borrowers", obj_text="7.9 and 3.6 basis points more")
        rel_b = self._relation(subject_text="Latino Borrowers", obj_text="7.9 and 3.6 basis points more")
        relations = [rel_a, rel_b]
        pipeline._split_shared_respectively_values(relations)

        self.assertEqual(rel_a.metadata["object_text"], "7.9 basis points more")
        self.assertEqual(rel_b.metadata["object_text"], "3.6 basis points more")
        corrections = [e for e in pipeline.quality_events if e["category"] == "respectively_split_correction"]
        self.assertEqual(len(corrections), 2)

    def test_stale_unsplit_duplicate_from_another_section_is_superseded(self):
        """Two occurrences of the SAME statistic in two different
        sections of one discourse: one gets correctly split, the other
        (extracted separately, never seen by the split pass together
        with its own sibling) keeps the stale compound value -- the
        reconciliation pass must supersede the stale one, not leave both
        as independent facts."""
        pipeline = _make_bare_pipeline()

        # Section "Public Datasets": correctly split pair.
        good_a = self._relation(subject_text="Black Borrowers", obj_text="7.9 and 3.6 basis points more", section="Public Datasets")
        good_b = self._relation(subject_text="Latino Borrowers", obj_text="7.9 and 3.6 basis points more", section="Public Datasets")
        pipeline._split_shared_respectively_values([good_a, good_b])

        # Section "Ethical Challenges": a stale, never-split duplicate of
        # the SAME statistic (simulating the un-split occurrence).
        stale = self._relation(subject_text="Black Borrowers", obj_text="7.9 and 3.6 basis points more", section="Ethical Challenges")

        all_relations = [good_a, good_b, stale]
        result = pipeline._supersede_stale_respectively_duplicates(all_relations, "d9")

        self.assertNotIn(stale, result, "the stale unsplit duplicate must be superseded/removed")
        self.assertIn(good_a, result)
        self.assertIn(good_b, result)
        superseded = [e for e in pipeline.quality_events if e["category"] == "respectively_split_supersession"]
        self.assertEqual(len(superseded), 1)
        self.assertIn(str(good_a.uri) if getattr(good_a, "uri", None) else "", superseded[0]["replacement_relation_ids"] + [""])

    def test_genuinely_distinct_facts_are_not_removed(self):
        """A relation that merely shares predicate/discourse but has NO
        respectively_split_from sibling anywhere must never be touched --
        guards against over-broad same-evidence deduplication."""
        pipeline = _make_bare_pipeline()
        distinct = self._relation(subject_text="White Borrowers", obj_text="2.1 basis points more")
        result = pipeline._supersede_stale_respectively_duplicates([distinct], "d9")
        self.assertEqual(result, [distinct])

    def test_entity_binding_is_corrected_not_just_surface_form(self):
        """Directly reproduces the confirmed SV100/SV108 defect shape
        (verified against the real annotations.json record for this
        exact sentence, see scratchpad/repro_respectively2.py): both
        sibling relations' `target` are ALREADY bound to ONE shared
        entity URI (minted by the primary entity-extraction pass for the
        unsplit "7.9 and 3.6 basis points more" phrase) at the point the
        split pass runs -- not raw text. The previous correction updated
        only metadata["object_text"]/object_surface_form; the actual
        etd:object binding survived unchanged. This test operates one
        level below full RDF serialization (directly on
        _split_shared_respectively_values) to isolate the binding logic
        itself; test_respectively_serialized_graph_has_distinct_object_
        bindings below confirms the same thing end-to-end through
        RDFSerializer."""
        pipeline = _make_bare_pipeline()
        shared_entity_uri = f"{Config.NAMESPACES['etd']}entity_shared_7_9_and_3_6"
        evidence = "Black and Latino borrowers pay 7.9 and 3.6 basis points more respectively."
        rel_a = Relation(
            source="placeholder-black-borrowers", relation="pay", target=shared_entity_uri,
            confidence=0.92,
            metadata={
                "subject_text": "Black borrowers", "object_text": "7.9 and 3.6 basis points more",
                "discourse_id": "d9", "evidence": evidence, "evidence_quote": evidence,
            },
        )
        rel_b = Relation(
            source="placeholder-latino-borrowers", relation="pay", target=shared_entity_uri,
            confidence=0.92,
            metadata={
                "subject_text": "Latino borrowers", "object_text": "7.9 and 3.6 basis points more",
                "discourse_id": "d9", "evidence": evidence, "evidence_quote": evidence,
            },
        )
        self.assertEqual(rel_a.target, rel_b.target, "sanity check: both start bound to the identical shared entity")

        pipeline._split_shared_respectively_values([rel_a, rel_b])

        self.assertNotEqual(
            rel_a.target, shared_entity_uri,
            "the stale shared-entity binding must not survive on the corrected relation",
        )
        self.assertNotEqual(
            rel_b.target, shared_entity_uri,
            "the stale shared-entity binding must not survive on the corrected relation",
        )
        self.assertNotEqual(rel_a.target, rel_b.target, "the two relations must no longer share one binding")
        self.assertEqual(rel_a.metadata["object_text"], "7.9 basis points more")
        self.assertEqual(rel_b.metadata["object_text"], "3.6 basis points more")
        self.assertFalse(rel_a.target_is_literal)
        self.assertIsNone(rel_a.target_literal)


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class RespectivelyEndToEndSerializationTests(unittest.TestCase):
    """End-to-end: build the real source sentence's entities/relations
    exactly as the extractor + primary entity-extraction pass produce
    them (both siblings' object already bound to ONE shared entity),
    run the ACTUAL pipeline object-resolution + RDFSerializer, and query
    the SERIALIZED graph's real etd:subject/etd:object bindings -- not
    just surface-form metadata. Confirms:
      Black borrowers -> 7.9 basis points [more]
      Latino borrowers -> 3.6 basis points [more]
    with genuinely distinct object bindings, and that the original
    compound relation is superseded rather than left standing alongside
    the correction."""

    def test_respectively_serialized_graph_has_distinct_object_bindings(self):
        from rdflib import Namespace, RDF, RDFS, URIRef
        from serializers.rdf_serializer import RDFSerializer
        from utils.helpers import GlobalEntityRegistry
        from ontology.mapper import OntologyMapper
        from utils.entity_normalizer import EntityNormalizer
        from pipeline import EnhancedSemanticPipeline

        pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        pipeline.config = Config
        pipeline.global_registry = GlobalEntityRegistry()
        pipeline.ontology_mapper = OntologyMapper()
        pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)

        etd_base = Config.NAMESPACES["etd"]
        evidence = "Black and Latino borrowers pay 7.9 and 3.6 basis points more respectively"

        def register_entity(label, entity_type="PopulationGroup"):
            slug = "".join(c.lower() if c.isalnum() else "-" for c in label)
            ent = Entity(uri=f"{etd_base}placeholder-{slug}", label=label, entity_type=entity_type, confidence=0.9, metadata={"discourse_id": "9"})
            pipeline._ensure_entity_ontology_grounding(ent)
            ent.uri = pipeline.entity_normalizer.canonical_uri(label=ent.label, entity_type=ent.entity_type, ontology_uri=getattr(ent, "ontology_uri", None))
            return pipeline.global_registry.get_or_create(ent)

        black_borrowers = register_entity("Black Borrowers")
        latino_borrowers = register_entity("Latino Borrowers")
        # The pre-existing, shared compound entity both relations'
        # `target` are already bound to -- exactly the confirmed real
        # defect shape (see repro_respectively2.py).
        compound_entity = register_entity("7.9 and 3.6 basis points more", entity_type="Metric")
        local_entities = [black_borrowers, latino_borrowers, compound_entity]

        rel_a = Relation(
            source=black_borrowers.uri, relation="pay", target=str(compound_entity.uri),
            confidence=0.92,
            metadata={
                "subject_text": "Black borrowers", "object_text": "7.9 and 3.6 basis points more",
                "discourse_id": "9", "field": "theme.description",
                "evidence": evidence, "evidence_quote": evidence,
            },
        )
        rel_b = Relation(
            source=latino_borrowers.uri, relation="pay", target=str(compound_entity.uri),
            confidence=0.92,
            metadata={
                "subject_text": "Latino borrowers", "object_text": "7.9 and 3.6 basis points more",
                "discourse_id": "9", "field": "theme.description",
                "evidence": evidence, "evidence_quote": evidence,
            },
        )
        unit = {"start_offset": 0, "text": evidence, "section": "Ethical Challenges"}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[rel_a, rel_b], local_entities=local_entities, discourse_text=evidence,
            discourse_id="9", field="theme.description", origin_key="theme.description|9",
            unit=unit, unit_text=evidence,
        )

        # Build the annotations_dict RDFSerializer expects, from the
        # resolved Relation objects (mirrors how pipeline.py's own
        # annotations get assembled).
        def _rel_dict(rel):
            return {
                "uri": rel.uri, "source": rel.source, "relation": rel.relation, "target": rel.target,
                "confidence": rel.confidence, "target_literal": rel.target_literal,
                "target_is_literal": rel.target_is_literal, "metadata": rel.metadata,
            }
        ann = _base_annotations([_rel_dict(r) for r in resolved], discourse_id="9")
        registry_entities = list(pipeline.global_registry.get_all_entities())
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(registry_entities), namespaces=Config.NAMESPACES)
        g = serializer.graph
        etd_ns = Namespace(etd_base)

        pay_rels = list(g.subjects(RDF.type, etd_ns.Relation))
        pay_rels = [r for r in pay_rels if (r, etd_ns.predicate, etd_ns.pay) in g]
        self.assertEqual(len(pay_rels), 2, "exactly two distinct 'pay' relations must survive, sharing no object binding")

        by_subject = {}
        for rel in pay_rels:
            subj = list(g.objects(rel, etd_ns.subject))[0]
            obj = list(g.objects(rel, etd_ns.object))[0]
            obj_label = None
            if isinstance(obj, URIRef):
                labels = [str(v) for v in g.objects(obj, RDFS.label)]
                obj_label = labels[0] if labels else None
            by_subject[str(subj)] = {"object": obj, "object_label": obj_label}

        black_obj = by_subject[str(black_borrowers.uri)]
        latino_obj = by_subject[str(latino_borrowers.uri)]

        self.assertNotEqual(
            black_obj["object"], latino_obj["object"],
            "Black borrowers and Latino borrowers must NOT be bound to the same object entity/literal",
        )
        self.assertNotEqual(
            black_obj["object"], compound_entity.uri,
            "Black borrowers must not retain the original stale shared '7.9 and 3.6' compound binding",
        )
        self.assertNotEqual(
            latino_obj["object"], compound_entity.uri,
            "Latino borrowers must not retain the original stale shared '7.9 and 3.6' compound binding",
        )
        # Each side's own number must appear in whatever the object
        # resolves to (a literal value or an entity's own label) --
        # confirms Black -> 7.9, Latino -> 3.6, not swapped.
        black_repr = str(black_obj["object_label"] or black_obj["object"])
        latino_repr = str(latino_obj["object_label"] or latino_obj["object"])
        self.assertIn("7.9", black_repr, f"Black borrowers' object must represent 7.9, got {black_repr!r}")
        self.assertIn("3.6", latino_repr, f"Latino borrowers' object must represent 3.6, got {latino_repr!r}")
        self.assertNotIn("3.6", black_repr)
        self.assertNotIn("7.9", latino_repr)

        # The original compound entity must not remain bound as the
        # object of any surviving 'pay' relation -- it may still exist
        # as an isolated node (pruning is a separate, already-tested
        # concern) but it must not be what either corrected relation
        # asserts.
        surviving_objects = {str(v["object"]) for v in by_subject.values()}
        self.assertNotIn(str(compound_entity.uri), surviving_objects)


# ---------------------------------------------------------------------
# 5. Targeted deterministic spot fixes
# ---------------------------------------------------------------------

class ExpletiveThereEntityTests(unittest.TestCase):
    def test_bare_there_is_treated_as_unresolved_coreference(self):
        from pipeline import EnhancedSemanticPipeline
        self.assertTrue(EnhancedSemanticPipeline._is_unresolved_coreference_candidate("There"))
        self.assertTrue(EnhancedSemanticPipeline._is_unresolved_coreference_candidate("there"))

    def test_there_used_as_locative_word_within_a_longer_phrase_is_unaffected(self):
        from pipeline import EnhancedSemanticPipeline
        # A genuine multi-word phrase containing "there" must not be
        # caught by this narrow, single-token-only check.
        self.assertFalse(EnhancedSemanticPipeline._is_expletive_only_token("living there permanently"))

    def test_reason_code_is_specific(self):
        from pipeline import EnhancedSemanticPipeline
        self.assertEqual(EnhancedSemanticPipeline._unresolved_referential_reason("there"), "unresolved_expletive_token")


class HostingPredicateLiteralObjectTests(unittest.TestCase):
    def test_bare_copula_with_literal_hosting_platform_object_is_corrected(self):
        pipeline = _make_bare_pipeline()
        rel = Relation(
            source="placeholder-subject", relation="is", target="UCI Machine Learning Repository",
            target_literal="UCI Machine Learning Repository", target_is_literal=True,
            confidence=0.9,
            metadata={"subject_text": "Adult/Census Income Dataset", "discourse_id": "d1"},
        )
        local_entities = []
        pipeline._correct_hosting_platform_predicate(
            rel, rel.metadata, local_entities, "is", discourse_id="d1",
            field="answer.description", origin_key="answer.description|d1",
            unit={"section": "A", "start_offset": 0}, unit_text="Adult/Census Income Dataset is UCI Machine Learning Repository.",
        )
        self.assertEqual(rel.relation, "hostedBy")
        self.assertFalse(
            rel.target_is_literal,
            "hostedBy is an owl:ObjectProperty -- its object must be a real entity, not a literal",
        )
        self.assertTrue(str(rel.target).startswith("http"), "target must now be a minted entity URI")
        self.assertEqual(len(local_entities), 1)
        self.assertEqual(local_entities[0].label, "UCI Machine Learning Repository")
        corrections = [e for e in pipeline.quality_events if e["category"] == "hosting_predicate_correction"]
        self.assertEqual(len(corrections), 1)

    def test_bare_copula_with_ordinary_literal_object_is_untouched(self):
        pipeline = _make_bare_pipeline()
        rel = Relation(
            source="placeholder-subject", relation="is", target="a machine learning model",
            target_literal="a machine learning model", target_is_literal=True,
            confidence=0.9,
            metadata={"subject_text": "GPT-4o", "discourse_id": "d1"},
        )
        pipeline._correct_hosting_platform_predicate(rel, rel.metadata, [], "is", discourse_id="d1")
        self.assertEqual(rel.relation, "is")


class PassiveDimensionEvidenceFallbackTests(unittest.TestCase):
    """The G3 guard's object_text check now also falls back to the
    relation's own evidence text, so an object whose head noun ('bias')
    was truncated out of object_text ('medical image classifiers') by
    the extractor is still recognised as an abstract-finding breakdown."""

    def test_object_text_truncation_still_matches_via_evidence_fallback(self):
        from pipeline import EnhancedSemanticPipeline
        pipeline = _make_bare_pipeline()
        matches_direct = pipeline._matches_semantic_category(
            category="abstract_finding", text="medical image classifiers", uri=None, local_entities=None,
        )
        self.assertFalse(matches_direct, "sanity check: the truncated object text alone must NOT already match")
        matches_evidence = pipeline._matches_semantic_category(
            category="abstract_finding",
            text="Documented bias in medical image classifiers by race, sex, age, and insurance type.",
            uri=None, local_entities=None,
        )
        self.assertTrue(matches_evidence, "the fuller evidence text must match abstract_finding")


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class EntityClassCorrectionOntologyConsistencyTests(unittest.TestCase):
    """Bounded preflight, item 5: when a class-correction rule fires,
    the entity's now-stale EXTERNAL ontology alignment (computed
    upstream from the pre-correction entity_type, entirely independent
    of _corrected_entity_class's rule engine) must not survive via any
    code path -- confirmed bug found during this preflight: suppressing
    only the immediate rdf:type emission left etd:ontology_uri itself
    still asserting the stale external class via a LATER, separate
    metadata-materialization pass that re-reads the entity's own
    (unmutated) metadata dict. A genuinely uncorrected entity's
    legitimate external alignment must be completely unaffected."""

    etd_base = Config.NAMESPACES["etd"]

    def _build_and_ground(self, *, label, entity_type):
        from pipeline import EnhancedSemanticPipeline
        from ontology.mapper import OntologyMapper
        from serializers.rdf_serializer import RDFSerializer
        from rdflib import Namespace

        slug = "".join(c.lower() if c.isalnum() else "_" for c in label)
        uri = f"{self.etd_base}entity_{slug}_onto_test"
        ent = Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.9, metadata={"surface_form": label})
        bare = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        bare.config = Config
        bare.ontology_mapper = OntologyMapper()
        bare._ensure_entity_ontology_grounding(ent)

        # A connecting relation so prune_isolated_entities does not
        # remove the entity before we can inspect it.
        ann = _base_annotations([{
            "source": uri, "relation": "relatedTo", "target": "x",
            "target_literal": "x", "target_is_literal": True, "confidence": 0.9,
            "metadata": {
                "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                "source": "llm_relation_extractor",
                "subject_surface_form": label, "object_surface_form": "x",
            },
        }])
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry([ent]), namespaces=Config.NAMESPACES)
        return serializer.graph, Namespace(self.etd_base), uri

    def test_corrected_entity_loses_stale_external_alignment_everywhere(self):
        from rdflib import URIRef, RDF
        g, etd_ns, uri = self._build_and_ground(label="Model Cards", entity_type="AISystem")
        types = list(g.objects(URIRef(uri), RDF.type))
        self.assertNotIn(URIRef("https://w3id.org/aipo#AISystem"), types, "stale external class must not appear as rdf:type")
        onto = list(g.objects(URIRef(uri), etd_ns.ontology_uri))
        self.assertEqual(onto, [], "stale etd:ontology_uri must also be gone, not just its rdf:type twin")
        self.assertIn(etd_ns.Resource, types, "the CORRECTED local class must still be asserted")

    def test_uncorrected_entity_keeps_its_legitimate_external_alignment(self):
        from rdflib import URIRef, RDF
        g, etd_ns, uri = self._build_and_ground(label="GPT-4o", entity_type="AISystem")
        types = list(g.objects(URIRef(uri), RDF.type))
        self.assertIn(URIRef("https://w3id.org/aipo#AISystem"), types, "a genuine AISystem must keep its real external alignment")
        onto = list(g.objects(URIRef(uri), etd_ns.ontology_uri))
        self.assertEqual(onto, [URIRef("https://w3id.org/aipo#AISystem")])


# ---------------------------------------------------------------------
# Integration test (bounded preflight, item 3): endpoint recovery,
# consolidation, direct-SPO reconciliation, reified endpoints, evidence
# links, provenance, and orphan cleanup all in one realistic discourse.
# ---------------------------------------------------------------------

@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class EndToEndCorrectionConsolidationIntegrationTests(unittest.TestCase):
    """One discourse exercising the full post-_add_relations pipeline:
    _enforce_endpoint_surface_compatibility_final (recovery) ->
    consolidate_duplicate_reified_relations -> _enforce_direct_spo_
    assertion_invariant, in that order (see RDFSerializer.__init__).

    Scenario: "COMPAS flagged Black defendants as high risk" is
    extracted twice -- once correctly bound to the "Black Defendants"
    entity, once mis-bound to the "COMPAS" entity (simulating the
    confirmed real-world defect shape) -- plus a genuinely distinct,
    negated restatement of the identical (subject, predicate, object)
    from a different sentence ("COMPAS did NOT flag ..." reported
    elsewhere). Expected end state:
      - the mis-bound occurrence is REPOINTED (recovered), not dropped,
        which makes its (s, p, o) identical to the correctly-bound one;
      - consolidation THEN merges those two into one reified relation,
        unioning their evidence links and preserving hasSourceEntity
        provenance;
      - the negated occurrence remains a SEPARATE, distinct reified
        relation (different negated status => different identity);
      - exactly one direct SPO triple survives, justified by the
        surviving positive relation;
      - no orphaned triples remain for the entity that was recovered
        away from (no relation node still points at "COMPAS" via this
        predicate/object pair) and no stray relation-node triples
        remain for the consolidated-away duplicate.
    """

    etd_base = Config.NAMESPACES["etd"]

    def test_recovery_then_consolidation_then_direct_spo_reconciliation(self):
        from rdflib import Namespace, URIRef, RDF
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        compas_uri = f"{self.etd_base}entity_compas_integ00"
        black_def_uri = f"{self.etd_base}entity_black_defendants_integ01"
        high_risk_uri = f"{self.etd_base}entity_high_risk_integ02"
        theme_uri = f"{self.etd_base}theme_integ_d5"

        entities = [
            Entity(uri=compas_uri, label="COMPAS", entity_type="Concept", confidence=0.8, metadata={"surface_form": "COMPAS"}),
            Entity(uri=black_def_uri, label="Black Defendants", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Black Defendants"}),
            Entity(uri=high_risk_uri, label="High Risk", entity_type="Concept", confidence=0.8, metadata={"surface_form": "High Risk"}),
        ]

        def _rel(uri_suffix, source, subject_surface, evidence_text, start_char, negated=False, epistemic_status="asserted"):
            return {
                "uri": f"{self.etd_base}{uri_suffix}",
                "source": source,
                "relation": "flagged",
                "target": high_risk_uri,
                "confidence": 0.9,
                "negated": negated,
                "metadata": {
                    "discourse_id": "d5", "field": "answer.description", "chunk_index": 0,
                    "source": "llm_relation_extractor", "relation_kind": "extracted_semantic",
                    "negated": negated, "epistemic_status": epistemic_status,
                    "subject_surface_form": subject_surface, "object_surface_form": "High Risk",
                    "source_entity_uri": theme_uri,
                    "evidence": evidence_text, "evidence_quote": evidence_text,
                    "start_char": start_char, "end_char": start_char + len(evidence_text),
                },
            }

        relations = [
            # R1: correctly bound, from one sentence/section.
            _rel("rel_integ_r1", black_def_uri, "Black Defendants",
                 "COMPAS flagged Black defendants as high risk in the pretrial report.", 0),
            # R2: mis-bound to COMPAS, but surface form still names the
            # true referent -- must be recovered to black_def_uri, which
            # then makes it an exact duplicate of R1. Distinct evidence
            # span (a different sentence re-stating the same fact) so
            # the merge's evidence-union behaviour is genuinely exercised.
            _rel("rel_integ_r2", compas_uri, "Black Defendants",
                 "The pretrial report noted Black defendants were flagged as high risk.", 200),
            # R3: genuinely distinct -- same eventual SPO once R2 is
            # recovered, but NEGATED, so it must remain separate.
            _rel("rel_integ_r3", black_def_uri, "Black Defendants",
                 "Contrary to some claims, COMPAS did not flag Black defendants as high risk in this case.", 400,
                 negated=True, epistemic_status="reported"),
        ]
        ann = _base_annotations(relations, discourse_id="d5")
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph

        # --- Reified endpoints after recovery + consolidation ---
        flagged_rels = [
            r for r in g.subjects(RDF.type, etd_ns.Relation)
            if (r, etd_ns.predicate, etd_ns.flagged) in g
        ]
        self.assertEqual(len(flagged_rels), 2, "exactly 2 distinct assertions must survive: one positive (merged), one negated")

        positive_rels = [r for r in flagged_rels if not any(str(v).lower() == "true" for v in g.objects(r, etd_ns.negated))]
        negated_rels = [r for r in flagged_rels if any(str(v).lower() == "true" for v in g.objects(r, etd_ns.negated))]
        self.assertEqual(len(positive_rels), 1, "R1 and recovered-R2 must consolidate into exactly one positive relation")
        self.assertEqual(len(negated_rels), 1, "R3 must survive as its own, separate, negated relation")

        positive_rel = positive_rels[0]
        negated_rel = negated_rels[0]
        self.assertEqual(list(g.objects(positive_rel, etd_ns.subject)), [URIRef(black_def_uri)])
        self.assertEqual(list(g.objects(negated_rel, etd_ns.subject)), [URIRef(black_def_uri)])

        # --- No orphaned reference to the recovered-away entity ---
        compas_flagged_rels = [r for r in flagged_rels if (r, etd_ns.subject, URIRef(compas_uri)) in g]
        self.assertEqual(compas_flagged_rels, [], "no surviving relation may still point at the mis-bound COMPAS entity")
        self.assertEqual(
            list(g.triples((URIRef(compas_uri), etd_ns.flagged, None))), [],
            "no orphaned direct triple may remain for the recovered-away entity",
        )

        # --- Evidence links: R1 + recovered-R2 must be unioned onto the survivor ---
        evidence_nodes = list(g.objects(positive_rel, etd_ns.hasEvidence))
        self.assertEqual(len(evidence_nodes), 2, "evidence from both consolidated occurrences must be preserved, not dropped")

        # --- Provenance: hasSourceEntity must survive consolidation ---
        source_entities = list(g.objects(positive_rel, etd_ns.hasSourceEntity))
        self.assertIn(URIRef(theme_uri), source_entities)

        # --- Direct-SPO reconciliation: exactly one direct triple, justified by the positive relation ---
        direct = list(g.triples((URIRef(black_def_uri), etd_ns.flagged, URIRef(high_risk_uri))))
        self.assertEqual(len(direct), 1, "exactly one direct triple must exist, backed by the surviving positive assertion")

        # --- Orphan cleanup: the merged-away duplicate's own relation URI must carry no triples at all ---
        loser_uri = URIRef(f"{self.etd_base}rel_integ_r2")
        self.assertEqual(
            list(g.triples((loser_uri, None, None))), [],
            "the consolidated-away duplicate's own relation node must be fully cleaned up",
        )
        self.assertEqual(list(g.triples((None, None, loser_uri))), [])


# ---------------------------------------------------------------------
# 6. Semantic-quality instrumentation report
# ---------------------------------------------------------------------

class SemanticQualityReportTests(unittest.TestCase):
    def test_report_is_written_with_recorded_events(self):
        from utils.semantic_quality_events import record_event, write_semantic_quality_report

        events = []
        record_event(
            events, category="self_reference_drop", action="dropped",
            discourse_id="d1", source_relation_id="rel_1",
            subject_surface="Audit Trails", predicate_surface="make", object_surface="Audit Trails",
            evidence_text="Audit trails make audit trails more robust.",
            reason="subject and object endpoints denote the same referent",
        )
        record_event(events, category="hosting_predicate_correction", action="corrected", discourse_id="d2")

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "semantic_quality_report.json")
            report = write_semantic_quality_report(events, out_path, run_id="run-test123")
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)

        self.assertEqual(on_disk["run_id"], "run-test123")
        self.assertEqual(on_disk["summary"]["total_events"], 2)
        self.assertEqual(on_disk["summary"]["by_category"]["self_reference_drop"], 1)
        self.assertEqual(len(on_disk["events"]), 2)
        self.assertEqual(report["summary"]["total_events"], 2)

    def test_long_evidence_text_is_clipped_not_dropped(self):
        from utils.semantic_quality_events import record_event

        events = []
        long_text = "x" * 1000
        record_event(events, category="c", action="dropped", evidence_text=long_text)
        self.assertLess(len(events[0]["evidence_text"]), 1000)
        self.assertTrue(events[0]["evidence_text"].endswith("…"))

    @unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
    def test_pipeline_and_serializer_events_share_one_sink(self):
        """main.py wires quality_events=pipeline.quality_events into
        RDFSerializer so pre- and post-serialization events land in ONE
        report; this test exercises that same wiring directly, without a
        live LLM call."""
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer

        pipeline = _make_bare_pipeline()
        pipeline.quality_events.append({"category": "self_reference_drop", "action": "dropped"})

        etd_base = Config.NAMESPACES["etd"]
        source_uri = f"{etd_base}entity_source_sink00"
        target_uri = f"{etd_base}entity_target_sink01"
        entities = [
            Entity(uri=source_uri, label="COMPAS", entity_type="Concept", confidence=0.8, metadata={"surface_form": "COMPAS"}),
            Entity(uri=target_uri, label="Outcome", entity_type="Concept", confidence=0.8, metadata={"surface_form": "Outcome"}),
        ]
        relations = [
            {
                "source": source_uri, "relation": "flagged", "target": target_uri, "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1", "field": "answer.description", "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    "subject_text": "totally different text", "object_text": "Outcome",
                },
            },
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(
            annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES,
            quality_events=pipeline.quality_events,
        )
        self.assertIs(serializer.quality_events, pipeline.quality_events)
        categories = {e["category"] for e in pipeline.quality_events}
        self.assertIn("self_reference_drop", categories)
        self.assertIn("endpoint_surface_mismatch_drop", categories)

    def test_report_carries_run_identity_and_no_secrets(self):
        """Bounded preflight, item 4: the report must include non-secret
        run/configuration identity, and must never carry anything that
        looks like a credential -- record_event's fixed field set
        cannot structurally accept a raw config/environment dump, and
        this test asserts that guarantee explicitly."""
        from utils.semantic_quality_events import record_event, write_semantic_quality_report

        events = []
        record_event(
            events, category="hosting_predicate_correction", action="corrected",
            discourse_id="d1", source_relation_id="rel_1",
            subject_surface="Adult/Census Income Dataset", predicate_surface="is",
            object_surface="UCI Machine Learning Repository",
            reason="hosting_platform_object_on_identity_or_description_predicate",
        )
        run_context = {
            "dataset_id": "ai_ethics_datasets",
            "pipeline_version": "5.1",
            "git_commit": "abc123",
            "generated_at_time": "2026-09-09T00:00:00+00:00",
            "domain_profile": "ai_ethics",
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "semantic_quality_report.json")
            report = write_semantic_quality_report(events, out_path, run_id="run-abc", run_context=run_context)
            with open(out_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)

        self.assertEqual(on_disk["run_context"], run_context)
        for key in ("dataset_id", "pipeline_version", "git_commit", "generated_at_time", "domain_profile"):
            self.assertIn(key, on_disk["run_context"])

        # No-secrets check: scan the entire serialized report text for
        # common credential-shaped substrings/field names. This report's
        # only inputs are `events` (fixed, narrow fields -- see
        # record_event's signature, which has no way to accept an
        # arbitrary blob) and `run_context` (explicit, non-secret
        # identity fields) -- so nothing else could leak in regardless.
        raw = json.dumps(report).lower()
        for forbidden in ("api_key", "apikey", "secret", "password", "authorization", "bearer ", "token"):
            self.assertNotIn(forbidden, raw, f"report must never contain a credential-shaped field: {forbidden!r}")

    def test_report_is_written_before_a_shacl_failure_aborts_the_run(self):
        """Bounded preflight, item 4: a quality-gate-failing run (SHACL
        non-conformance with FAIL_ON_SHACL_VIOLATIONS) must still leave
        the semantic-quality report behind -- it is written immediately
        after the graph is built, before SHACL/upload/serialize-to-file,
        so a later hard failure cannot prevent it."""
        import types as _types
        from unittest.mock import patch
        import importlib.util as _ilu
        if _ilu.find_spec("rdflib") is None:
            self.skipTest("rdflib is required for main orchestration tests")
        import main

        class DummyPipeline:
            def __init__(self):
                self.global_registry = object()
                self.quality_events = [{"category": "self_reference_drop", "action": "dropped"}]

            def process_batch(self, _data):
                return {"themes": {}, "questions": {}, "answers": {}, "votes": {}, "experts": {}, "documents": {}}

            def apply_reasoning(self, _ann):
                return []

            def detect_conflicts(self, _ann):
                return []

            def compute_quality_metrics(self, _ann):
                return {}

        class DummySerializer:
            def __init__(self, annotations_dict, registry, namespaces, **kwargs):
                events = kwargs.get("quality_events")
                self.quality_events = events if isinstance(events, list) else []
                self.graph = object()
                self.run_id = "run-shacl-fail-test"
                self.dataset_id = "ai_ethics_datasets"
                self.pipeline_version = "5.1"
                self.git_commit = "unknown"
                self.generated_at_time = "2026-09-09T00:00:00+00:00"
                self.domain_profile = _types.SimpleNamespace(name="ai_ethics")

            def serialize_to_turtle(self, _):
                pass

            def serialize_to_rdf(self, _):
                pass

            def serialize_to_jsonld(self, _):
                pass

            def serialize_to_owl(self, _):
                pass

        report_calls = []

        def _fake_write_report(events, path, *, run_id=None, run_context=None):
            report_calls.append({"events": list(events), "path": path, "run_id": run_id, "run_context": run_context})
            return {}

        with patch("main.parse_arguments", return_value=self._args()), \
             patch.object(main.Config, "validate", return_value=True), \
             patch.object(main.Config, "print_config", return_value=None), \
             patch("builtins.print", return_value=None), \
             patch("utils.helpers.create_output_directory", return_value=None), \
             patch("utils.helpers.load_json", return_value={"themes": [], "questions": [], "answers": []}), \
             patch("utils.helpers.validate_input_json", return_value=True), \
             patch("utils.helpers.save_json", return_value=None), \
             patch("main.Path.exists", return_value=True), \
             patch("pipeline.EnhancedSemanticPipeline", return_value=DummyPipeline()), \
             patch("serializers.rdf_serializer.RDFSerializer", DummySerializer), \
             patch("utils.semantic_quality_events.write_semantic_quality_report", side_effect=_fake_write_report), \
             patch("validation.shacl_validator.SHACLValidator") as shacl_cls:
            shacl_cls.return_value.validate_graph.return_value = {"conforms": False, "report_path": "x"}
            main.Config.FAIL_ON_SHACL_VIOLATIONS = True
            main.Config.USE_GRAPHDB_UPLOAD = False
            with self.assertRaises(SystemExit):
                main.main()

        self.assertEqual(len(report_calls), 1, "the report must be written exactly once, even though SHACL failure aborts the run")
        self.assertEqual(report_calls[0]["run_context"]["dataset_id"], "ai_ethics_datasets")
        self.assertEqual(report_calls[0]["run_context"]["domain_profile"], "ai_ethics")

    def _args(self):
        import types
        return types.SimpleNamespace(
            config=None, input="data/ai_ethics_datasets.json", output="output",
            debug=False, no_reasoning=False, no_wikidata=True, batch_size=1,
            use_llm=False, llm_model="gpt-4o-mini",
        )


if __name__ == "__main__":
    unittest.main()
