import unittest

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


def _make_pipeline():
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.wikidata_linker = None
    pipeline._discourse_chunk_spans = {}
    pipeline.semantic_policy = None
    return pipeline


def _entity(uri, label, entity_type, *, discourse_id, start_char, end_char):
    return Entity(
        uri=uri,
        label=label,
        entity_type=entity_type,
        confidence=1.0,
        metadata={
            "discourse_id": discourse_id,
            "section_category": "semantic",
            "start_char": start_char,
            "end_char": end_char,
            "mentions": [
                {
                    "discourse_id": discourse_id,
                    "chunk_index": 0,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            ],
        },
    )


# ---------------------------------------------------------------------------
# Fixture 1 -- Theme 13 pattern: a single, unambiguous "system" actor is
# established early in the field; a later bullet depends on it without
# repeating its name, and has no other locally-available actor.
# ---------------------------------------------------------------------------
POSITIVE_TEXT = (
    "An Automated Triage System is used to prioritize service requests across the platform.\n\n"
    "Operational Issues:\n"
    "* Response Delay: The persistent backlog causes Rural Service Areas to experience significantly longer wait times\n"
    "* Staffing Shortfall: On-call staff report chronic overtime"
)
POSITIVE_DISCOURSE_ID = "T1"
POSITIVE_TARGET_UNIT_TEXT = (
    "Response Delay: The persistent backlog causes Rural Service Areas to experience significantly longer wait times"
)


def _positive_entities():
    return [
        _entity(
            "https://kg/entity/automated_triage_system",
            "Automated Triage System",
            "AISystem",
            discourse_id=POSITIVE_DISCOURSE_ID,
            start_char=3,
            end_char=26,
        ),
        _entity(
            "https://kg/entity/rural_service_areas",
            "Rural Service Areas",
            "Group",
            discourse_id=POSITIVE_DISCOURSE_ID,
            start_char=156,
            end_char=175,
        ),
    ]


# ---------------------------------------------------------------------------
# Fixture 2 -- ambiguity negative control: TWO plausible "system" actors are
# established before the same kind of later bullet. Neither should be carried
# forward automatically.
# ---------------------------------------------------------------------------
AMBIGUOUS_TEXT = (
    "An Automated Triage System routes urgent cases, while a Fallback Dispatch System "
    "handles overflow during peak hours.\n\n"
    "Operational Issues:\n"
    "* Response Delay: Rural Service Areas experience significantly longer wait times"
)
AMBIGUOUS_DISCOURSE_ID = "T2"
AMBIGUOUS_TARGET_UNIT_TEXT = (
    "Response Delay: Rural Service Areas experience significantly longer wait times"
)


def _ambiguous_entities():
    return [
        _entity(
            "https://kg/entity/automated_triage_system",
            "Automated Triage System",
            "AISystem",
            discourse_id=AMBIGUOUS_DISCOURSE_ID,
            start_char=3,
            end_char=26,
        ),
        _entity(
            "https://kg/entity/fallback_dispatch_system",
            "Fallback Dispatch System",
            "AISystem",
            discourse_id=AMBIGUOUS_DISCOURSE_ID,
            start_char=56,
            end_char=80,
        ),
        _entity(
            "https://kg/entity/rural_service_areas",
            "Rural Service Areas",
            "Group",
            discourse_id=AMBIGUOUS_DISCOURSE_ID,
            start_char=156,
            end_char=175,
        ),
    ]


# ---------------------------------------------------------------------------
# Fixture 3 -- passive/agent-elliptical local unit: the same unambiguous prior
# "system" actor is established early, and the later bullet's own local
# subject is a "group"-family (demographic/population-type) entity that is
# the affected party, not an agent -- e.g. "Rural Residents assigned
# significantly longer wait times" (no auxiliary verb, telegraphic bullet
# style, no other locally-available agent-plausible entity).
# ---------------------------------------------------------------------------
SWAP_TEXT = (
    "An Automated Triage System is used to prioritize service requests across the platform.\n\n"
    "Operational Issues:\n"
    "* Response Delay: Rural Residents assigned significantly longer wait times\n"
    "* Staffing Shortfall: On-call staff report chronic overtime"
)
SWAP_DISCOURSE_ID = "T3"
SWAP_TARGET_UNIT_TEXT = "Response Delay: Rural Residents assigned significantly longer wait times"


def _swap_entities():
    return [
        _entity(
            "https://kg/entity/automated_triage_system",
            "Automated Triage System",
            "AISystem",
            discourse_id=SWAP_DISCOURSE_ID,
            start_char=3,
            end_char=26,
        ),
    ]


# ---------------------------------------------------------------------------
# Fixture 4 -- negative control: the local unit names its OWN active agent
# ("Regional Dispatch Coordinators"), so the passive/implicit-agent swap must
# not override it even though a single unambiguous prior actor exists.
# ---------------------------------------------------------------------------
SWAP_NEGATIVE_TEXT = (
    "An Automated Triage System is used to prioritize service requests across the platform.\n\n"
    "Operational Issues:\n"
    "* Response Delay: Regional Dispatch Coordinators assign Rural Residents significantly longer wait times\n"
    "* Staffing Shortfall: On-call staff report chronic overtime"
)
SWAP_NEGATIVE_DISCOURSE_ID = "T4"
SWAP_NEGATIVE_TARGET_UNIT_TEXT = (
    "Response Delay: Regional Dispatch Coordinators assign Rural Residents significantly longer wait times"
)


def _swap_negative_entities():
    return [
        _entity(
            "https://kg/entity/automated_triage_system",
            "Automated Triage System",
            "AISystem",
            discourse_id=SWAP_NEGATIVE_DISCOURSE_ID,
            start_char=3,
            end_char=26,
        ),
        _entity(
            "https://kg/entity/regional_dispatch_coordinators",
            "Regional Dispatch Coordinators",
            "Organisation",
            discourse_id=SWAP_NEGATIVE_DISCOURSE_ID,
            start_char=126,
            end_char=156,
        ),
    ]


class CarryForwardGatingUnitTests(unittest.TestCase):
    """Deterministic tests of _carry_forward_actor_for_unit in isolation."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_single_unambiguous_prior_actor_is_carried_forward(self):
        entities = _positive_entities()
        unit = {"start_offset": 110, "end_offset": 221, "section_category": "semantic"}
        result = self.pipeline._carry_forward_actor_for_unit(
            entities=entities, unit=unit, discourse_id=POSITIVE_DISCOURSE_ID
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.uri, "https://kg/entity/automated_triage_system")

    def test_two_competing_prior_actors_are_not_carried_forward(self):
        entities = _ambiguous_entities()
        unit = {"start_offset": 140, "end_offset": 218, "section_category": "semantic"}
        result = self.pipeline._carry_forward_actor_for_unit(
            entities=entities, unit=unit, discourse_id=AMBIGUOUS_DISCOURSE_ID
        )
        self.assertIsNone(result)

    def test_actor_mentioned_after_the_unit_is_not_carried_forward(self):
        entities = _positive_entities()
        unit = {"start_offset": 0, "end_offset": 2, "section_category": "semantic"}
        result = self.pipeline._carry_forward_actor_for_unit(
            entities=entities, unit=unit, discourse_id=POSITIVE_DISCOURSE_ID
        )
        self.assertIsNone(result)


class CarryForwardPipelineWiringTests(unittest.TestCase):
    """End-to-end tests through _process_relations_batch: the Theme 13 carry-
    forward pattern is recovered with correct provenance/validation, and the
    ambiguity negative control never lets the hint reach the extractor."""

    def test_theme13_pattern_recovers_relation_with_field_context_provenance(self):
        pipeline = _make_pipeline()
        entities = _positive_entities()

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            hinted = next(
                (e for e in local_entities if (e.metadata or {}).get("carried_forward_from_field_context")),
                None,
            )
            if hinted is None:
                return []
            target = next((e for e in local_entities if e.label == "Rural Service Areas"), None)
            if target is None:
                return []
            return [
                Relation(
                    source=hinted.uri,
                    relation="causes",
                    target=target.uri,
                    confidence=0.9,
                    metadata={
                        "extractor": "llm",
                        "context_source": "llm_relation",
                        "evidence_quote": unit_text,
                        "evidence": unit_text,
                    },
                )
            ]

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [POSITIVE_TEXT],
            [entities],
            [POSITIVE_DISCOURSE_ID],
            provenance_list=[{"field": "theme.description", "discourse_id": POSITIVE_DISCOURSE_ID}],
        )
        relations = relations_list[0]

        carried = [
            r for r in relations
            if str(getattr(r, "source", "")) == "https://kg/entity/automated_triage_system"
        ]
        self.assertEqual(len(carried), 1, f"expected exactly one carried-forward relation, got: {relations}")
        relation = carried[0]

        # Subject was resolved from field context, not a literal local mention.
        self.assertTrue(relation.metadata.get("subject_resolved_from_field_context"))
        self.assertTrue(
            str(relation.metadata.get("local_context") or "").startswith("Automated Triage System")
        )

        # Evidence/provenance stay anchored to the target chunk, never widened.
        self.assertEqual(relation.metadata.get("evidence_quote"), POSITIVE_TARGET_UNIT_TEXT)
        self.assertEqual(relation.metadata.get("evidence"), POSITIVE_TARGET_UNIT_TEXT)

        # The widened local_context is real validation input, not a no-op:
        # semantic validation actually accepts the relation because of it.
        entities_by_uri = {e.uri: e for e in entities}
        accepted, reason = pipeline._validate_semantic_relation_candidate(
            relation,
            entities_by_uri=entities_by_uri,
            discourse_text=POSITIVE_TEXT,
        )
        self.assertTrue(accepted, f"expected relation to validate, rejection reason: {reason}")

    def test_ambiguous_prior_actors_never_reach_the_extractor_as_a_hint(self):
        pipeline = _make_pipeline()
        entities = _ambiguous_entities()
        seen_hint = {"value": False}

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            if any((e.metadata or {}).get("carried_forward_from_field_context") for e in local_entities):
                seen_hint["value"] = True
            return []

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        pipeline._process_relations_batch(
            [AMBIGUOUS_TEXT],
            [entities],
            [AMBIGUOUS_DISCOURSE_ID],
            provenance_list=[{"field": "theme.description", "discourse_id": AMBIGUOUS_DISCOURSE_ID}],
        )

        self.assertFalse(
            seen_hint["value"],
            "carry-forward hint must not be offered when two prior actors are ambiguous",
        )


class PassiveAgentEllipticalSwapTests(unittest.TestCase):
    """The extractor may still attribute a passive/agent-elliptical local unit
    to its own local "group"-family (affected-party) entity instead of the
    carried-forward actor. When Text names no other agent-plausible entity,
    _process_relations_batch deterministically swaps: the carried-forward
    actor becomes subject and the local entity becomes the object."""

    def test_passive_local_unit_swaps_to_carried_forward_actor(self):
        pipeline = _make_pipeline()
        entities = _swap_entities()

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            if "Rural Residents" not in unit_text:
                return []
            # Model attributes the relation to its own local surface subject
            # ("Rural Residents") rather than using the field_context_actor
            # hint -- subject_text is unmatched (no local entity pre-exists
            # for it), forcing dynamic resolution during
            # _resolve_semantic_unit_relations, exactly as observed live.
            return [
                Relation(
                    source="Rural Residents",
                    relation="assigned",
                    target="significantly longer wait times",
                    target_is_literal=True,
                    confidence=0.9,
                    metadata={
                        "extractor": "llm",
                        "context_source": "llm_relation",
                        "subject_text": "Rural Residents",
                        "subject_type": "Group",
                        "object_text": "significantly longer wait times",
                        "evidence_quote": unit_text,
                        "evidence": unit_text,
                    },
                )
            ]

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [SWAP_TEXT],
            [entities],
            [SWAP_DISCOURSE_ID],
            provenance_list=[{"field": "theme.description", "discourse_id": SWAP_DISCOURSE_ID}],
        )
        relations = relations_list[0]

        swapped = [
            r for r in relations
            if str(getattr(r, "source", "")) == "https://kg/entity/automated_triage_system"
            and str(getattr(r, "relation", "")) == "assigned"
        ]
        self.assertEqual(len(swapped), 1, f"expected exactly one swapped relation, got: {relations}")
        relation = swapped[0]

        # The local patient entity ("Rural Residents") becomes the object,
        # not a discarded literal.
        self.assertFalse(relation.target_is_literal)
        # The relation's PREVIOUS target was the literal-valued
        # text "significantly longer wait times", describing something the
        # new Entity target does not -- that value must not survive as
        # stale target_literal now that the group entity occupies the
        # target position.
        self.assertIsNone(relation.target_literal)
        target_entity = next(
            (
                e for e in getattr(pipeline.global_registry, "entities", {}).values()
                if str(getattr(e, "uri", "")) == str(relation.target)
            ),
            None,
        )
        self.assertIsNotNone(target_entity)
        self.assertEqual(target_entity.label, "Rural Residents")

        # Field-context resolution is recorded, and evidence stays anchored
        # to the target unit.
        self.assertTrue(relation.metadata.get("subject_resolved_from_field_context"))
        self.assertEqual(relation.metadata.get("evidence_quote"), SWAP_TARGET_UNIT_TEXT)
        self.assertEqual(relation.metadata.get("evidence"), SWAP_TARGET_UNIT_TEXT)
        self.assertTrue(
            str(relation.metadata.get("local_context") or "").startswith("Automated Triage System")
        )

        # Prior to the endpoint-state-invariant fix, this
        # relation was (incorrectly) accepted here only because a stale,
        # pre-swap target_literal ("significantly longer wait times") made
        # _relation_has_literal_object() misclassify it as literal-typed,
        # routing it through semantic_relation_policy's lenient
        # literal-assertion acceptance path instead of genuine entity-role/
        # predicate-entailment validation. Now that target_literal is
        # correctly cleared (asserted above), the relation is evaluated
        # honestly as an Entity-to-Entity assertion: the swap itself,
        # entity role, and domain/range checks all still pass -- only the
        # generic "relatedTo" predicate fallback's strict local-evidence
        # requirement (no explicit association phrase in this fixture's
        # text) is what withholds acceptance, which is a predicate-ontology
        # coverage question unrelated to this test's own subject (the
        # subject/object swap) and unrelated to endpoint-state consistency.
        entities_by_uri = {e.uri: e for e in entities}
        entities_by_uri[str(relation.target)] = target_entity
        accepted, reason = pipeline._validate_semantic_relation_candidate(
            relation,
            entities_by_uri=entities_by_uri,
            discourse_text=SWAP_TEXT,
        )
        self.assertFalse(accepted)
        self.assertEqual(relation.metadata.get("semantic_validation_code"), "predicate_ambiguity:generic_relation_not_justified")
        self.assertEqual(relation.metadata.get("entity_role_check"), "pass")
        self.assertEqual(relation.metadata.get("domain_range_check"), "pass")

    def test_explicit_local_agent_blocks_the_swap(self):
        pipeline = _make_pipeline()
        entities = _swap_negative_entities()

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            coordinators = next(
                (e for e in local_entities if e.label == "Regional Dispatch Coordinators"), None
            )
            if coordinators is None:
                return []
            return [
                Relation(
                    source=coordinators.uri,
                    relation="assign",
                    target="significantly longer wait times",
                    target_is_literal=True,
                    confidence=0.9,
                    metadata={
                        "extractor": "llm",
                        "context_source": "llm_relation",
                        "evidence_quote": unit_text,
                        "evidence": unit_text,
                    },
                )
            ]

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [SWAP_NEGATIVE_TEXT],
            [entities],
            [SWAP_NEGATIVE_DISCOURSE_ID],
            provenance_list=[{"field": "theme.description", "discourse_id": SWAP_NEGATIVE_DISCOURSE_ID}],
        )
        relations = relations_list[0]

        matching = [r for r in relations if str(getattr(r, "relation", "")) == "assign"]
        self.assertEqual(len(matching), 1, f"expected exactly one relation, got: {relations}")
        relation = matching[0]

        # Subject stays the explicit local agent; the swap must not fire.
        self.assertEqual(str(relation.source), "https://kg/entity/regional_dispatch_coordinators")
        self.assertFalse(relation.metadata.get("subject_resolved_from_field_context"))


if __name__ == "__main__":
    unittest.main()
