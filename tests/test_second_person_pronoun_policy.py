"""Safe second-person ("you"/"your") pronoun endpoint policy.

GENERAL RULE: never hardcode "you" -> a concrete Person entity.

  a) if discourse evidence ALREADY grounds "you" to a resolved entity
     (e.g. via an earlier, general-purpose resolution stage), nothing here
     blocks Person-compatible typing for it;
  b) a confidently resolved antecedent/addressee is bound to the existing
     canonical entity it already resolved to -- this mechanism never
     mints a NEW one;
  c) when human reference is clear but identity is not, no falsely
     specific Person identity is minted -- the endpoint stays unresolved,
     exactly like every other unresolved referential/deictic pronoun
     (the unresolved-endpoint fabrication guard);
  d) generic / impersonal / institutional / hypothetical / ambiguous "you"
     is never forced toward Person typing -- there is no "you" -> Person
     special case anywhere in this policy;
  e) possessive "your X" resolves to a clean head-noun literal ("X") and
     NEVER becomes, or is used to mint, a standalone "your ..." entity;
     its possessor is intentionally never guessed by this generic
     mechanism (no textual antecedent, no reflexive-subject fallback --
     see pipeline.py::_SECOND_PERSON_POSSESSIVE_DETERMINERS).

Mirrors the structure/fixtures of test_relation_endpoint_possessive.py and
test_relation_endpoint_unresolved_coreference.py; every discourse/entity
name here is a synthetic fixture.
"""
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


def _unit(unit_id, text, start_offset, *, section_category="semantic", unit_type="sentence"):
    return {
        "unit_id": unit_id,
        "text": text,
        "start_offset": start_offset,
        "end_offset": start_offset + len(text),
        "section_category": section_category,
        "unit_type": unit_type,
    }


def _ent(uri, label, entity_type, *, discourse_id, start_char, end_char, metadata=None):
    meta = {
        "discourse_id": discourse_id,
        "section_category": "semantic",
        "start_char": start_char,
        "end_char": end_char,
        "mentions": [
            {"discourse_id": discourse_id, "chunk_index": 0, "start_char": start_char, "end_char": end_char}
        ],
    }
    meta.update(metadata or {})
    return Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.9, metadata=meta)


class SplitPossessiveEndpointYourTests(unittest.TestCase):
    def test_your_x_splits_to_head_noun(self):
        split = EnhancedSemanticPipeline._split_possessive_endpoint("your data")
        self.assertEqual(split, ("your", "data"))

    def test_your_own_x_strips_own_intensifier(self):
        split = EnhancedSemanticPipeline._split_possessive_endpoint("your own performance")
        self.assertEqual(split, ("your", "performance"))

    def test_bare_your_with_no_head_noun_is_not_a_possessive_phrase(self):
        self.assertIsNone(EnhancedSemanticPipeline._split_possessive_endpoint("your"))

    def test_existing_third_person_possessive_split_unchanged(self):
        self.assertEqual(EnhancedSemanticPipeline._split_possessive_endpoint("their approach"), ("their", "approach"))
        self.assertEqual(EnhancedSemanticPipeline._split_possessive_endpoint("its accuracy"), ("its", "accuracy"))


class PossessiveYourNeverResolvesPossessorTests(unittest.TestCase):
    """(e): "your X" -> clean literal head noun; possessor is NEVER
    resolved by this generic mechanism -- not via antecedent search, not
    via the reflexive-subject fallback other possessives use."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_your_data_becomes_clean_literal_no_possessor(self):
        # A tool subject that WOULD satisfy the reflexive-subject fallback
        # for "its"/"their" (system family) must NOT be used for "your".
        tool = _ent("https://kg/e/tool", "The Tool", "AISystem", discourse_id="dY1", start_char=0, end_char=8)
        relation = Relation(
            source=tool.uri, relation="collects", target="your data", target_is_literal=True,
            metadata={"object_text": "your data", "discourse_id": "dY1"},
        )
        unit = {"start_offset": 100}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[], local_pool_entities=[tool], unit=unit, discourse_id="dY1",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "data")
        self.assertTrue(rel.target_is_literal)
        self.assertNotIn("object_possessor_entity_uri", rel.metadata)
        self.assertNotIn("object_possessor_resolution_method", rel.metadata)
        self.assertEqual(rel.metadata.get("object_possessor_head_noun_phrase"), "data")
        self.assertEqual(rel.metadata.get("object_possessor_original_surface"), "your data")
        # the word "your" must never survive onto the materialised endpoint
        self.assertNotIn("your", str(rel.target).lower())

    def test_your_x_does_not_bind_even_with_unambiguous_nearby_person(self):
        # Item (c)/(d): even a single, unambiguous, nearby Person entity
        # (the shape that WOULD satisfy the occurrence-level general antecedent
        # search for a 3rd-person possessive) must not become "your"'s
        # possessor -- "your" names the discourse ADDRESSEE, not whichever
        # person happens to be textually nearby.
        unit_start = 200
        unit_text = "The reviewer explained the process and then described your rights."
        reviewer_local = unit_text.index("reviewer")
        reviewer = _ent(
            "https://kg/e/reviewer", "Reviewer", "Person",
            discourse_id="dY2", start_char=unit_start + reviewer_local, end_char=unit_start + reviewer_local + len("reviewer"),
        )
        relation = Relation(
            source=reviewer.uri, relation="described", target="your rights", target_is_literal=True,
            metadata={"object_text": "your rights", "discourse_id": "dY2", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[reviewer], local_pool_entities=[reviewer], unit=unit, discourse_id="dY2",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "rights")
        self.assertNotIn("object_possessor_entity_uri", rel.metadata)

    def test_your_organization_x_not_forced_toward_person(self):
        # Institutional/impersonal head noun: still just a clean literal,
        # no possessor guessed, no Person typing implied anywhere.
        subject = _ent("https://kg/e/policy_doc", "The Policy", "Policy", discourse_id="dY3", start_char=0, end_char=11)
        relation = Relation(
            source=subject.uri, relation="requires", target="your organization's consent", target_is_literal=True,
            metadata={"object_text": "your organization's consent", "discourse_id": "dY3"},
        )
        unit = {"start_offset": 0}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[], local_pool_entities=[subject], unit=unit, discourse_id="dY3",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "organization's consent")
        self.assertNotIn("object_possessor_entity_uri", rel.metadata)


class BareSecondPersonPronounDroppedNotFabricatedTests(unittest.TestCase):
    """(c)/(d): bare "you"/"your" with no grounded resolution must be
    left unresolved (dropped from materialisation), exactly like other
    referential/deictic pronouns -- never fabricated into a normal,
    globally-reusable entity, and never silently bound to an unrelated
    nearby entity."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def _run(self, *, source, object_text, discourse_id="dY4"):
        relation = Relation(
            source=source, relation="affects", target=object_text, target_is_literal=False,
            metadata={"subject_text": source, "object_text": object_text},
        )
        unit = _unit("Body:1", f"{source} {object_text}", 0)
        return self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[], discourse_text=unit["text"],
            discourse_id=discourse_id, field="answer.description",
            origin_key=f"answer.description|{discourse_id}", unit=unit, unit_text=unit["text"],
        )

    def test_bare_you_as_subject_is_unresolved_not_fabricated(self):
        resolved = self._run(source="you", object_text="the AI system")
        self.assertEqual(resolved[0].source, "you")
        self.assertTrue(resolved[0].metadata.get("subject_endpoint_unresolved_coreference"))
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any("you" == str(getattr(e, "label", "")).strip().lower() for e in entities))

    def test_bare_you_as_object_is_unresolved_not_fabricated(self):
        resolved = self._run(source="Model Alpha", object_text="you")
        self.assertTrue(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any("you" == str(getattr(e, "label", "")).strip().lower() for e in entities))

    def test_bare_your_alone_is_unresolved_not_fabricated(self):
        resolved = self._run(source="your", object_text="Model Alpha")
        self.assertTrue(resolved[0].metadata.get("subject_endpoint_unresolved_coreference"))
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any("your" == str(getattr(e, "label", "")).strip().lower() for e in entities))

    def test_ambiguous_you_with_two_candidate_persons_never_binds_to_either(self):
        # Two Person entities present locally -- an ordinary 3rd-person
        # pronoun might attempt (and correctly abstain from) an antecedent
        # search here; "you" must not even attempt it.
        a = _ent("https://kg/e/p1", "Reviewer A", "Person", discourse_id="dY5", start_char=0, end_char=1)
        b = _ent("https://kg/e/p2", "Reviewer B", "Person", discourse_id="dY5", start_char=2, end_char=3)
        relation = Relation(
            source="you", relation="affects", target="the process", target_is_literal=False,
            metadata={"subject_text": "you", "object_text": "the process"},
        )
        unit = _unit("Body:1", "you the process", 0)
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[a, b], discourse_text=unit["text"],
            discourse_id="dY5", field="answer.description", origin_key="answer.description|dY5",
            unit=unit, unit_text=unit["text"],
        )
        self.assertNotEqual(resolved[0].source, a.uri)
        self.assertNotEqual(resolved[0].source, b.uri)
        self.assertTrue(resolved[0].metadata.get("subject_endpoint_unresolved_coreference"))


class SecondPersonNominalCorefNotProximityMatchedTests(unittest.TestCase):
    """"you" must never be handed to the general nominal-antecedent
    search (unlike "it"/"this"/etc.) -- confirms the historical
    proximity-mismatch bug this exclusion prevents stays prevented."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_you_is_not_referential_endpoint_text(self):
        self.assertFalse(EnhancedSemanticPipeline._is_referential_endpoint_text("you"))
        self.assertFalse(EnhancedSemanticPipeline._is_referential_endpoint_text("your"))

    def test_resolve_anaphoric_relation_endpoints_leaves_bare_you_untouched(self):
        unit_start = 0
        unit_text = "Biases in the training data can distort outcomes for you."
        biases_local = unit_text.index("Biases")
        biases = _ent(
            "https://kg/e/biases", "Biases", "Concept",
            discourse_id="dY6", start_char=unit_start + biases_local, end_char=unit_start + biases_local + len("Biases"),
        )
        relation = Relation(
            source="training data", relation="distort", target="you", target_is_literal=True,
            metadata={"object_text": "you", "discourse_id": "dY6", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[biases], unit=unit, discourse_id="dY6",
        )
        # unchanged: no nonsense "you" -> "Biases" proximity binding.
        self.assertEqual(resolved[0].target, "you")
        self.assertFalse(resolved[0].metadata.get("object_endpoint_coref_resolved"))
        self.assertEqual(extra, [])


class AlreadyResolvedSecondPersonEndpointTypingIsNotForcedOrBlockedTests(unittest.TestCase):
    """(a)/(b): when a second-person endpoint has ALREADY been grounded to
    a concrete canonical entity by some earlier, general-purpose
    resolution (e.g. the entity extractor's own context-based pronoun
    resolution producing a resolved URI) this policy neither blocks it
    nor forces/changes its type -- Person-compatible typing flows through
    exactly like any other already-resolved endpoint, and the EXISTING
    canonical entity is reused rather than a new one minted."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_already_resolved_you_to_person_entity_passes_through_unblocked(self):
        person = _ent("https://kg/e/addressee", "The Interviewee", "Person", discourse_id="dY7", start_char=0, end_char=10)
        relation = Relation(
            source=person.uri, relation="affects", target="the outcome", target_is_literal=False,
            metadata={"subject_text": "you", "object_text": "the outcome"},
        )
        unit = _unit("Body:1", "you the outcome", 0)
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[person], discourse_text=unit["text"],
            discourse_id="dY7", field="answer.description", origin_key="answer.description|dY7",
            unit=unit, unit_text=unit["text"],
        )
        rel = resolved[0]
        # bound to the EXISTING canonical entity -- no new one minted, no
        # unresolved-coreference flag, type stays whatever it already was.
        self.assertEqual(rel.source, person.uri)
        self.assertFalse(rel.metadata.get("subject_endpoint_unresolved_coreference"))
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any(
            str(getattr(e, "label", "")).strip().lower() in {"you", "your"} for e in entities
        ))


class ExistingFirstAndThirdPersonBehaviourUnchangedTests(unittest.TestCase):
    """Locks that this change touches only second-person handling."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_bare_first_person_pronouns_still_dropped(self):
        for pron in ("i", "we", "my", "our"):
            relation = Relation(
                source=pron, relation="affects", target="Model Alpha", target_is_literal=False,
                metadata={"subject_text": pron, "object_text": "Model Alpha"},
            )
            unit = _unit("Body:1", f"{pron} Model Alpha", 0)
            resolved = self.pipeline._resolve_semantic_unit_relations(
                relations=[relation], local_entities=[], discourse_text=unit["text"],
                discourse_id="dFP", field="answer.description", origin_key="answer.description|dFP",
                unit=unit, unit_text=unit["text"],
            )
            self.assertTrue(resolved[0].metadata.get("subject_endpoint_unresolved_coreference"), pron)

    def test_third_person_reflexive_possessive_still_resolves(self):
        system = _ent("https://kg/e/system_x", "System X", "AISystem", discourse_id="dTP", start_char=0, end_char=8)
        relation = Relation(
            source=system.uri, relation="improves", target="its accuracy", target_is_literal=True,
            metadata={"object_text": "its accuracy", "discourse_id": "dTP"},
        )
        unit = {"start_offset": 100}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[], local_pool_entities=[system], unit=unit, discourse_id="dTP",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "accuracy")
        self.assertEqual(rel.metadata.get("object_possessor_entity_uri"), system.uri)
        self.assertEqual(rel.metadata.get("object_possessor_resolution_method"), "reflexive_subject")

    def test_third_person_bare_it_still_resolves_via_nominal_coref(self):
        unit_start = 100
        unit_text = "The Screening Tool flagged several applicants, and reviewers later overruled it."
        local_start = unit_text.index("Screening Tool")
        tool = _ent(
            "https://kg/e/screening_tool", "Screening Tool", "AISystem",
            discourse_id="dTP2", start_char=unit_start + local_start, end_char=unit_start + local_start + len("Screening Tool"),
        )
        relation = Relation(
            source="reviewers", relation="overruled", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "dTP2", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[tool], unit=unit, discourse_id="dTP2",
        )
        self.assertEqual(resolved[0].target, tool.uri)


class EndToEndYourPossessiveTest(unittest.TestCase):
    """Real per-unit pipeline replay (mirrors
    test_relation_endpoint_possessive.py's "their approach" end-to-end
    test): "your X" through _process_relations_batch must come out as a
    clean literal head noun with no possessor and no "your ..." entity in
    the registry."""

    def test_your_history_end_to_end(self):
        pipeline = _make_pipeline()
        did = "TY1"
        text = (
            "An Ethics Review Framework is used to evaluate new projects before launch.\n\n"
            "Operational Notes:\n"
            "* Transparency: The intake form will review your submission history "
            "before granting access\n"
            "* Staffing: On-call staff report chronic overtime"
        )

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            if "review your" not in unit_text:
                return []
            return [
                Relation(
                    source="the intake form", relation="review", target="your submission history",
                    target_is_literal=True, confidence=0.9,
                    metadata={
                        "extractor": "llm", "context_source": "llm_relation",
                        "subject_text": "the intake form", "object_text": "your submission history",
                        "evidence_quote": unit_text, "evidence": unit_text,
                    },
                )
            ]

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [text], [[]], [did], provenance_list=[{"field": "answer.description", "discourse_id": did}],
        )
        relations = relations_list[0]

        matching = [r for r in relations if str(getattr(r, "relation", "")) == "review"]
        self.assertEqual(len(matching), 1, f"expected exactly one relation, got: {relations}")
        relation = matching[0]

        self.assertEqual(relation.target, "submission history")
        self.assertTrue(relation.target_is_literal)
        self.assertNotIn("object_possessor_entity_uri", relation.metadata)
        self.assertNotIn("your", str(relation.target).lower())

        entities = pipeline.global_registry.get_all_entities()
        self.assertFalse(any("your" in str(getattr(e, "label", "")).strip().lower() for e in entities))


if __name__ == "__main__":
    unittest.main()
