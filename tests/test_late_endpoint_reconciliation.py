"""
C.4 -- late relation-endpoint creation / reconciliation.

Root cause fixed here: a relation endpoint could be left pointing at an
INTERMEDIATE entity URI produced by a multi-step consolidation merge
(A -> B in one pass, B -> C in a later one). ``EntityConsolidator`` stored
the raw hop and ``pipeline._apply_entity_redirects_to_annotations`` applied
redirects with a single dict lookup, so the endpoint landed on B -- a URI no
surviving entity holds and the later ``CanonicalLinker`` has no redirect for.
The RDF serializer then materialised a bare, provenance-less ``etd:Entity``
stub for it (no datasetId/runId/field/source_doc_id/span).

The fixes, all generic (no dependency on any label / QID / dataset / discourse):
  1. ``EntityConsolidator`` keeps its redirect map FLAT -- every stored value
     is a terminal URI, never itself a redirect key.
  2. ``_apply_entity_redirects_to_annotations`` resolves redirect chains
     transitively for entity URIs and relation endpoints alike.
  3. ``run_canonical_linking`` ends with ``_reconcile_relation_endpoints``:
     every non-literal endpoint URI must be a URI a registered canonical
     entity actually holds; a stranded one is repaired via the union of all
     redirect maps (transitively) or recorded as unresolved -- never left to
     be silently materialised.
  4. Synthetic ``Proposition`` endpoints no longer carry
     ``surface_form="proposition"`` (a type label masquerading as a verbatim
     mention); they are marked ``mention_grounding="constructed"`` and keep
     their clause text + offsets as source grounding.

No example label / URI from the production graph is hardcoded below.
"""
import tempfile
import types
import unittest
from pathlib import Path

from rdflib import Namespace, RDF, URIRef

from config import Config
from models import Entity, Relation
from pipeline import EnhancedSemanticPipeline
from serializers.rdf_serializer import RDFSerializer
from utils.entity_consolidator import EntityConsolidator
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry

ETD = Config.NAMESPACES["etd"]
ETD_NS = Namespace(ETD)


def _entity(local, label, *, conf=0.9, meta=None):
    return Entity(
        uri=f"{ETD}{local}",
        label=label,
        entity_type="Concept",
        confidence=conf,
        metadata=dict(meta or {}),
    )


def _blank_pipeline():
    p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    p.global_registry = GlobalEntityRegistry()
    p._provenance_diagnostics = {}
    p.canonical_linker = types.SimpleNamespace(redirects={})
    return p


def _annotations_stub(relations):
    pr = types.SimpleNamespace(relations=list(relations), entities=[], metadata={"text": ""})
    return types.SimpleNamespace(
        themes={}, questions={}, answers={"d1": pr},
        votes={}, experts={}, documents={},
    )


class RedirectChainFlatteningTests(unittest.TestCase):
    """The consolidator's redirect map never contains a value that is
    itself a key -- a two-step merge collapses to a direct edge."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cons = EntityConsolidator(
            registry=GlobalEntityRegistry(),
            normalizer=EntityNormalizer(),
            output_dir=Path(self._tmp.name),
        )

    def test_two_step_merge_produces_a_flat_map(self):
        a = _entity("term_variant_a", "Signal Term")
        b = _entity("term_variant_b", "Signal Term")
        c = _entity("term_super_c", "Signal Term Broad")

        # Step 1 (e.g. exact-key pass): A folds into B.
        self.cons._merge_entity_into_canonical(b, a, reason="exact_key", score=1.0)
        # Step 2 (e.g. later fuzzy pass): B folds into C.
        self.cons._merge_entity_into_canonical(c, b, reason="fuzzy", score=0.95)

        # Every redirect value must be terminal -- NOT itself a redirect key.
        for src, dst in self.cons.redirects.items():
            self.assertNotIn(
                dst, self.cons.redirects,
                f"redirect {src} -> {dst} is a chain; value {dst} is still a key",
            )
        # A and B both resolve straight to C.
        self.assertEqual(self.cons.redirects.get(a.uri), c.uri)
        self.assertEqual(self.cons.redirects.get(b.uri), c.uri)

    def test_terminal_redirect_target_is_cycle_safe(self):
        self.cons.redirects = {f"{ETD}x": f"{ETD}y", f"{ETD}y": f"{ETD}x"}
        # Must terminate rather than loop forever.
        self.assertIn(self.cons._terminal_redirect_target(f"{ETD}x"), {f"{ETD}x", f"{ETD}y"})


class ApplyEntityRedirectsTransitiveTests(unittest.TestCase):
    """_apply_entity_redirects_to_annotations follows the whole chain."""

    def test_relation_endpoint_follows_multi_hop_chain(self):
        p = _blank_pipeline()
        a, b, c = f"{ETD}ep_a", f"{ETD}ep_b", f"{ETD}ep_c"
        rel = Relation(source=a, relation="relatedTo", target=a, confidence=0.9,
                       metadata={"discourse_id": "d1"})
        ann = _annotations_stub([rel])

        p._apply_entity_redirects_to_annotations(ann, {a: b, b: c})

        # Landed on the terminal URI, not the intermediate.
        self.assertEqual(rel.source, c)
        self.assertEqual(rel.target, c)
        # And the cumulative union is remembered for later reconciliation.
        self.assertEqual(p._entity_redirects.get(a), b)

    def test_entity_uri_follows_multi_hop_chain(self):
        p = _blank_pipeline()
        a, b, c = f"{ETD}u_a", f"{ETD}u_b", f"{ETD}u_c"
        ent = _entity("u_a", "Whatever")
        pr = types.SimpleNamespace(relations=[], entities=[ent], metadata={})
        ann = types.SimpleNamespace(
            themes={}, questions={}, answers={"d1": pr},
            votes={}, experts={}, documents={},
        )
        p._apply_entity_redirects_to_annotations(ann, {a: b, b: c})
        self.assertEqual(ent.uri, c)


class ReconcileRelationEndpointsTests(unittest.TestCase):
    """The post-canonical-linking invariant."""

    def test_stranded_endpoint_is_repaired_to_the_registered_canonical(self):
        p = _blank_pipeline()
        canonical = _entity("kept_canonical", "Kept Canonical",
                            meta={"surface_form": "kept canonical", "field": "answer.description",
                                  "discourse_id": "d1"})
        p.global_registry.get_or_create(canonical)

        stale_a, stale_b = f"{ETD}stale_a", f"{ETD}stale_b"
        p._entity_redirects = {stale_a: stale_b, stale_b: canonical.uri}
        rel = Relation(source=stale_a, relation="relatedTo", target="a literal value",
                       confidence=0.9, target_literal="a literal value", target_is_literal=True,
                       metadata={"discourse_id": "d1", "subject_text": "kept canonical"})
        ann = _annotations_stub([rel])

        report = p._reconcile_relation_endpoints(ann)

        self.assertEqual(rel.source, canonical.uri)
        self.assertEqual(report["repaired"], 1)
        self.assertEqual(report["unresolved"], 0)
        self.assertEqual(p._provenance_diagnostics["reconciled_relation_endpoints"], 1)
        self.assertEqual(p._provenance_diagnostics["unresolved_relation_endpoints"], [])

    def test_reuse_of_existing_canonical_entity_no_new_uri_minted(self):
        p = _blank_pipeline()
        canonical = _entity("shared_canonical", "Shared Canonical",
                            meta={"surface_form": "shared canonical"})
        p.global_registry.get_or_create(canonical)
        stale = f"{ETD}shared_stale"
        p._entity_redirects = {stale: canonical.uri}

        # One relation already on the canonical URI, one still on the stale hop.
        r1 = Relation(source=canonical.uri, relation="relatedTo", target=f"{ETD}other1",
                      confidence=0.9, metadata={"discourse_id": "d1"})
        r2 = Relation(source=stale, relation="relatedTo", target=f"{ETD}other2",
                      confidence=0.9, metadata={"discourse_id": "d1"})
        p.global_registry.get_or_create(_entity("other1", "Other One"))
        p.global_registry.get_or_create(_entity("other2", "Other Two"))
        ann = _annotations_stub([r1, r2])

        p._reconcile_relation_endpoints(ann)

        # Both relations converge on the SAME pre-existing canonical URI --
        # nothing new was minted for the stale endpoint.
        self.assertEqual(r1.source, canonical.uri)
        self.assertEqual(r2.source, canonical.uri)
        endpoint_uris = {r1.source, r2.source}
        self.assertEqual(endpoint_uris, {canonical.uri})
        self.assertNotIn(stale, endpoint_uris)

    def test_genuinely_unresolvable_endpoint_is_flagged_not_invented(self):
        p = _blank_pipeline()
        p.global_registry.get_or_create(_entity("only_real", "Only Real"))
        orphan = f"{ETD}orphan_endpoint"  # in no registry, in no redirect map
        rel = Relation(source=orphan, relation="relatedTo", target=f"{ETD}only_real",
                       confidence=0.9, metadata={"discourse_id": "d1", "subject_text": "ghost"})
        ann = _annotations_stub([rel])

        report = p._reconcile_relation_endpoints(ann)

        self.assertEqual(report["unresolved"], 1)
        self.assertEqual(rel.source, orphan)  # left as-is, not silently rewritten
        flagged = p._provenance_diagnostics["unresolved_relation_endpoints"]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["endpoint_uri"], orphan)
        self.assertEqual(flagged[0]["side"], "source")
        self.assertTrue(
            any(ev.get("category") == "late_endpoint_unreconciled" for ev in p.quality_events)
        )


class SerializedGraphHasNoBareStubTests(unittest.TestCase):
    """End to end: after reconciliation the relation subject is the
    registered canonical entity (with full provenance) and NO bare
    ``etd:Entity`` stub exists for the stranded intermediate URI."""

    def _serialize(self, registry_entities, annotations_dict):
        reg = types.SimpleNamespace(get_all_entities=lambda: list(registry_entities))
        return RDFSerializer(
            annotations_dict=annotations_dict,
            registry=reg,
            namespaces=Config.NAMESPACES,
            dataset_id="unit_dataset",
            run_id="run-unit-test",
        )

    def test_no_stub_and_full_provenance_after_reconcile(self):
        p = _blank_pipeline()
        canonical = _entity(
            "grounded_canonical", "Grounded Canonical",
            meta={
                "surface_form": "Grounded Canonical",
                "field": "answer.description",
                "source_field": "answer.description",
                "discourse_id": "d1",
                "source_doc_id": "d1",
                "start_char": 0,
                "end_char": 18,
                "context_source": "semantic_assertion_resolution",
                "source": "semantic_assertion_resolution",
                "chunk_index": -1,
                "chunk_index_unknown": True,
            },
        )
        p.global_registry.get_or_create(canonical)
        other = _entity("obj_entity", "Object Entity", meta={"surface_form": "Object Entity"})
        p.global_registry.get_or_create(other)

        stale_hop = f"{ETD}stranded_intermediate"
        p._entity_redirects = {stale_hop: canonical.uri}

        rel = Relation(
            source=stale_hop, relation="relatedTo", target=other.uri, confidence=0.92,
            metadata={"discourse_id": "d1", "field": "answer.description",
                      "subject_text": "Grounded Canonical", "object_text": "Object Entity",
                      "chunk_index": 0},
        )
        ann = _annotations_stub([rel])
        p._reconcile_relation_endpoints(ann)
        self.assertEqual(rel.source, canonical.uri)

        annotations_dict = {
            "themes": {}, "questions": {},
            "answers": {"d1": {"relations": [rel.to_dict()],
                               "metadata": {"text": "Grounded Canonical relates to it."}}},
            "votes": {}, "experts": {}, "documents": {},
            "metadata": {"datasetId": "unit_dataset", "runId": "run-unit-test"},
        }
        serializer = self._serialize([canonical, other], annotations_dict)
        g = serializer.graph

        # No bare stub for the stranded intermediate URI.
        self.assertNotIn((URIRef(stale_hop), RDF.type, ETD_NS.Entity), g)
        self.assertEqual(list(g.predicate_objects(URIRef(stale_hop))), [])

        # The canonical subject carries real dataset/run/field provenance.
        canon = URIRef(canonical.uri)
        self.assertIn(canonical.uri, {str(s) for s in g.subjects(ETD_NS.datasetId, None)})
        self.assertEqual(str(next(g.objects(canon, ETD_NS.runId))), "run-unit-test")
        self.assertEqual(str(next(g.objects(canon, ETD_NS.field))), "answer.description")
        self.assertTrue(list(g.objects(canon, ETD_NS.source_doc_id)))


class SyntheticPropositionEndpointTests(unittest.TestCase):
    """A Proposition endpoint is a CONSTRUCTED entity, not a pretend
    verbatim mention."""

    def _prop_pipeline(self):
        p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        p.global_registry = GlobalEntityRegistry()
        p.config = types.SimpleNamespace(USE_ONTOLOGY_MAPPING=False)
        return p

    def test_proposition_entity_is_marked_constructed_with_clause_provenance(self):
        p = self._prop_pipeline()
        clause = "the described mechanism increases downstream error rates"
        unit = {"unit_id": "U:1", "start_offset": 40, "end_offset": 40 + len(clause),
                "text": clause, "unit_type": "sentence"}

        ent = p._get_or_create_unit_proposition_entity(
            unit=unit, discourse_id="d9", origin_key="some.field|d9",
            field="some.field", derived_from_relation_uris=[],
        )
        self.assertIsNotNone(ent)
        meta = ent.metadata

        # Does NOT pretend a type label is a verbatim source mention.
        self.assertNotEqual(str(meta.get("surface_form") or "").strip().lower(), "proposition")
        # Explicitly flagged as constructed, distinct from mention-grounded.
        self.assertEqual(meta.get("mention_grounding"), "constructed")
        self.assertEqual(meta.get("constructed_from"), "semantic_clause")
        # Clause / source provenance is preserved.
        self.assertEqual(meta.get("text"), clause)
        self.assertEqual(meta.get("start_char"), 40)
        self.assertEqual(meta.get("end_char"), 40 + len(clause))
        self.assertEqual(meta.get("source_doc_id"), "d9")

    def test_serialized_proposition_has_no_proposition_surface_form_literal(self):
        p = self._prop_pipeline()
        clause = "this claim is contested by later work"
        unit = {"unit_id": "U:2", "start_offset": 5, "end_offset": 5 + len(clause),
                "text": clause, "unit_type": "sentence"}
        ent = p._get_or_create_unit_proposition_entity(
            unit=unit, discourse_id="d3", origin_key="f|d3", field="f",
            derived_from_relation_uris=[],
        )
        reg = types.SimpleNamespace(get_all_entities=lambda: [ent])
        annotations_dict = {
            "themes": {}, "questions": {}, "answers": {}, "votes": {},
            "experts": {}, "documents": {},
            "metadata": {"datasetId": "unit_dataset", "runId": "run-unit-test"},
        }
        serializer = RDFSerializer(
            annotations_dict=annotations_dict, registry=reg,
            namespaces=Config.NAMESPACES, dataset_id="unit_dataset", run_id="run-unit-test",
        )
        g = serializer.graph
        surface_literals = {str(o).strip().lower() for o in g.objects(URIRef(ent.uri), ETD_NS.surface_form)}
        self.assertNotIn("proposition", surface_literals)
        from rdflib.namespace import SKOS
        hidden_literals = {str(o).strip().lower() for o in g.objects(URIRef(ent.uri), SKOS.hiddenLabel)}
        self.assertNotIn("proposition", hidden_literals)


if __name__ == "__main__":
    unittest.main()
