"""Same-chunk entity collision: identical label + affirmative same
concrete-chunk evidence on BOTH sides must not survive as two entities
merely because two stages inferred incompatible coarse types for it.

Root cause: CanonicalLinker.link_entities Pass 2 keys its exact
grouping on (coarse_type, normalized_label, scope). Two no-external-ID
mentions of the same surface form in the same discourse/chunk -- one
typed Concept by a coverage-recovery extraction, one typed Dataset by the
semantic-assertion path -- landed in different groups and each minted its
own canonical URI, so scripts/kg_quality_gate.py's
_same_chunk_duplicate_labels_count reported a collision.

Pass 1.5 (_unify_same_chunk_cross_type) unifies a purely inferred
coarse-type disagreement onto the resolved (most specific) type before
Pass 2. Safety review: the same-chunk test now requires a shared
*concrete* (discourse, chunk) affirmatively evidenced for BOTH mentions
-- from entity metadata OR caller-injected grounded-relation provenance
(the mechanism a Dataset endpoint actually has). "Same discourse +
one concrete chunk + one unknown chunk" is NOT sufficient. Named
benchmarks below are synthetic fixtures only.
"""
import tempfile
import unittest

from models import Entity, Relation
from pipeline import EnhancedSemanticPipeline
from utils.canonical_linker import CanonicalLinker

_OUT = tempfile.mkdtemp(prefix="skg_test_same_chunk_entity_collision_")

_ETD = "https://alfie-project.eu/etd-hub#"


class _Registry:
    def __init__(self):
        self.entities = {}

    def get_all_entities(self):
        return list(self.entities.values())

    def replace_entity(self, old_uri, new_uri):
        if old_uri in self.entities and new_uri not in self.entities:
            self.entities[new_uri] = self.entities[old_uri]


def _mk(uri, label, entity_type, *, did=None, chunk=None, chunk_unknown=None,
        confidence=0.9, metadata=None):
    meta = dict(metadata or {})
    if did is not None:
        meta.setdefault("discourse_id", str(did))
    if chunk is not None:
        meta.setdefault("chunk_index", int(chunk))
    if chunk_unknown is not None:
        meta.setdefault("chunk_index_unknown", chunk_unknown)
    return Entity(uri=_ETD + uri, label=label, entity_type=entity_type,
                  confidence=confidence, metadata=meta)


def _link(*entities, participation=None):
    reg = _Registry()
    reg.entities = {e.uri: e for e in entities}
    linker = CanonicalLinker(registry=reg, output_dir=_OUT)
    if participation:
        linker.set_entity_chunk_participation(participation)
    out = linker.run_full_registry()
    return linker, out


class SameChunkCrossTypeCollapseTests(unittest.TestCase):
    def test_kg46_shape_collapses_when_dataset_side_has_grounded_relation_chunk(self):
        # concept mention carries the concrete chunk in its own metadata;
        # the assertion-derived dataset mention has NO span of its own but
        # its grounded relations place it in the same concrete chunk --
        # supplied here exactly as run_canonical_linking would inject it.
        concept = _mk("entity/concept/x-1", "Some Benchmark Suite", "Concept",
                      did="12", chunk=0, confidence=0.9)
        dataset = _mk("entity/dataset/x-2", "Some Benchmark Suite", "Dataset",
                      did="12", chunk=None, confidence=0.86,
                      metadata={"context_source": "semantic_assertion_resolution"})
        linker, out = _link(concept, dataset,
                            participation={dataset.uri: {("12", 0)}})

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "Dataset")  # deepest declared type
        self.assertIn("dataset", out[0].uri.lower())
        self.assertEqual(concept.entity_type, "Dataset")
        self.assertEqual(concept.metadata.get("same_chunk_type_reconciled_from"), "Concept")
        targets = {linker._resolve_redirect(concept.uri), linker._resolve_redirect(dataset.uri)}
        self.assertEqual(len(targets), 1)

    def test_same_concrete_chunk_in_both_entities_metadata_collapses(self):
        a = _mk("entity/concept/y-1", "Shared Corpus Name", "Concept", did="5", chunk=2)
        b = _mk("entity/dataset/y-2", "Shared Corpus Name", "Dataset", did="5", chunk=2)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "Dataset")

    def test_compatible_specificity_difference_promotes_to_specific(self):
        a = _mk("entity/resource/z-1", "Shared Repo Name", "Resource", did="7", chunk=0, confidence=0.7)
        b = _mk("entity/dataset/z-2", "Shared Repo Name", "Dataset", did="7", chunk=0, confidence=0.6)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "Dataset")
        self.assertEqual(a.entity_type, "Dataset")

    def test_uncertain_side_does_not_block_collapse(self):
        confident = _mk("entity/dataset/u-1", "Shared Set Name", "Dataset", did="8", chunk=1,
                        metadata={"population_group_typing_reason": "llm_classification_confident"})
        ambiguous = _mk("entity/concept/u-2", "Shared Set Name", "Concept", did="8", chunk=1,
                        confidence=0.4,
                        metadata={"population_group_typing_reason": "llm_classification_ambiguous"})
        _linker, out = _link(confident, ambiguous)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "Dataset")


class UnknownChunkEvidenceSafetyTests(unittest.TestCase):
    def test_unknown_chunk_with_no_independent_evidence_stays_separate(self):
        # Same discourse, same label, A@chunk0, B@unknown, and NO
        # grounded-relation / metadata chunk evidence for B -> MUST remain
        # two entities (the old "one known chunk + one unknown" shortcut is
        # gone).
        a = _mk("entity/concept/k-1", "Ambiguous Chunkless", "Concept", did="7", chunk=0)
        b = _mk("entity/dataset/k-2", "Ambiguous Chunkless", "Dataset", did="7", chunk=None)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 2)
        self.assertEqual({o.entity_type for o in out}, {"Concept", "Dataset"})

    def test_unknown_chunk_collapses_once_grounded_relation_participation_supplied(self):
        a = _mk("entity/concept/k-3", "Ambiguous With Relation", "Concept", did="7", chunk=0)
        b = _mk("entity/dataset/k-4", "Ambiguous With Relation", "Dataset", did="7", chunk=None)
        _linker, out = _link(a, b, participation={b.uri: {("7", 0)}})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "Dataset")

    def test_metadata_chunk_index_unknown_flag_is_not_concrete_evidence(self):
        a = _mk("entity/concept/k-5", "Flagged Unknown", "Concept", did="8", chunk=0)
        b = _mk("entity/dataset/k-6", "Flagged Unknown", "Dataset", did="8", chunk=0,
                chunk_unknown=True)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 2)

    def test_grounded_relation_in_different_chunk_does_not_merge(self):
        a = _mk("entity/concept/k-7", "Cross Chunk Relation", "Concept", did="9", chunk=0)
        b = _mk("entity/dataset/k-8", "Cross Chunk Relation", "Dataset", did="9", chunk=None)
        _linker, out = _link(a, b, participation={b.uri: {("9", 1)}})
        self.assertEqual(len(out), 2)


class DistinctReferentEvidenceTests(unittest.TestCase):
    def test_explicit_distinct_referent_flag_keeps_two_entities(self):
        concept = _mk("entity/concept/d-1", "Overloaded Term", "Concept", did="12", chunk=0)
        dataset = _mk("entity/dataset/d-2", "Overloaded Term", "Dataset", did="12", chunk=0,
                      metadata={"distinct_referent": True})
        _linker, out = _link(concept, dataset)
        self.assertEqual(len(out), 2)
        self.assertEqual({o.entity_type for o in out}, {"Concept", "Dataset"})

    def test_distinct_referent_from_naming_other_member_keeps_two(self):
        a = _mk("entity/concept/d-3", "Overloaded Term Two", "Concept", did="4", chunk=0)
        b = _mk("entity/dataset/d-4", "Overloaded Term Two", "Dataset", did="4", chunk=0,
                metadata={"distinct_referent_from": [_ETD + "entity/concept/d-3"]})
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 2)


class NoCrossContextForcedMergeTests(unittest.TestCase):
    def test_different_chunk_same_label_not_merged_across_types(self):
        a = _mk("entity/concept/c-1", "Repeated Label", "Concept", did="5", chunk=0)
        b = _mk("entity/dataset/c-2", "Repeated Label", "Dataset", did="5", chunk=1)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 2)
        self.assertEqual({o.entity_type for o in out}, {"Concept", "Dataset"})

    def test_different_discourse_same_label_not_merged_across_types(self):
        a = _mk("entity/concept/c-3", "Repeated Label Two", "Concept", did="1", chunk=0)
        b = _mk("entity/dataset/c-4", "Repeated Label Two", "Dataset", did="2", chunk=0)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 2)
        self.assertEqual({o.entity_type for o in out}, {"Concept", "Dataset"})

    def test_both_sides_chunkless_same_discourse_not_forced(self):
        a = _mk("entity/concept/c-5", "Vague Grounding", "Concept", did="9", chunk=None)
        b = _mk("entity/dataset/c-6", "Vague Grounding", "Dataset", did="9", chunk=None)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 2)


class ExistingBehaviourIntactTests(unittest.TestCase):
    def test_same_chunk_same_type_still_reuses_one_entity(self):
        a = _mk("entity/concept/s-1", "Ordinary Concept", "Concept", did="3", chunk=0)
        b = _mk("entity/concept/s-2", "Ordinary Concept", "Concept", did="3", chunk=0)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type, "Concept")

    def test_incompatible_types_without_shared_chunk_still_kept_separate(self):
        a = _mk("entity/person/p-1", "Ambiguous Name", "Person", did="1", chunk=0, confidence=0.9)
        b = _mk("entity/organization/p-2", "Ambiguous Name", "Organization", did="2", chunk=0, confidence=0.92)
        _linker, out = _link(a, b)
        self.assertEqual(len(out), 2)
        self.assertEqual({o.entity_type for o in out}, {"Person", "Organization"})


class RelationGroundedChunkParticipationTests(unittest.TestCase):
    """pipeline._relation_grounded_chunk_participation: only GROUNDED
    concrete-chunk relations feed the linker, literals are ignored."""

    def _rel(self, src, tgt, meta, *, literal=False):
        r = Relation(source=src, relation="rel", target=tgt, metadata=dict(meta))
        if literal:
            r.target_is_literal = True
        return r

    def test_grounded_concrete_chunk_relation_feeds_both_endpoints(self):
        r = self._rel(_ETD + "e/a", _ETD + "e/b",
                      {"discourse_id": "12", "chunk_index": 0, "groundingStatus": "grounded"})
        out = EnhancedSemanticPipeline._relation_grounded_chunk_participation([[r]])
        self.assertEqual(out.get(_ETD + "e/a"), {("12", 0)})
        self.assertEqual(out.get(_ETD + "e/b"), {("12", 0)})

    def test_literal_object_is_ignored(self):
        r = self._rel(_ETD + "e/a", "some literal value",
                      {"discourse_id": "12", "chunk_index": 0, "groundingStatus": "grounded"},
                      literal=True)
        out = EnhancedSemanticPipeline._relation_grounded_chunk_participation([[r]])
        self.assertEqual(set(out.keys()), {_ETD + "e/a"})

    def test_unknown_or_ungrounded_relation_contributes_nothing(self):
        r1 = self._rel(_ETD + "e/a", _ETD + "e/b",
                       {"discourse_id": "12", "chunk_index": 0, "chunk_index_unknown": True,
                        "groundingStatus": "grounded"})
        r2 = self._rel(_ETD + "e/a", _ETD + "e/b",
                       {"discourse_id": "12", "chunk_index": -1, "groundingStatus": "ungrounded"})
        out = EnhancedSemanticPipeline._relation_grounded_chunk_participation([[r1, r2]])
        self.assertEqual(out, {})

    def test_missing_grounding_status_still_accepted_when_chunk_concrete(self):
        # a relation grounded during the per-section pass may not carry an
        # explicit status; a concrete, non-unknown chunk index is enough.
        r = self._rel(_ETD + "e/a", _ETD + "e/b", {"discourse_id": "3", "chunk_index": 1})
        out = EnhancedSemanticPipeline._relation_grounded_chunk_participation([[r]])
        self.assertEqual(out.get(_ETD + "e/a"), {("3", 1)})


if __name__ == "__main__":
    unittest.main()
