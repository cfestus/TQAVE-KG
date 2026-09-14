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


def _ent(uri, label, entity_type, *, discourse_id, start_char, end_char):
    return Entity(
        uri=uri, label=label, entity_type=entity_type, confidence=0.9,
        metadata={
            "discourse_id": discourse_id,
            "section_category": "semantic",
            "start_char": start_char,
            "end_char": end_char,
            "mentions": [{"discourse_id": discourse_id, "chunk_index": 0, "start_char": start_char, "end_char": end_char}],
        },
    )


def _unit(unit_id, text, start_offset, *, section_category="semantic"):
    return {
        "unit_id": unit_id, "text": text, "start_offset": start_offset,
        "end_offset": start_offset + len(text), "section_category": section_category, "unit_type": "sentence",
    }


class PropositionalDeferPrecedenceTests(unittest.TestCase):
    """A still-unresolved, sentence-initial bare demonstrative
    subject ("This"/"That") must defer Tier 3's nominal fallback so the
    propositional/discourse-deictic Proposition resolver (which runs next,
    unmodified) gets first
    opportunity -- instead of being permanently consumed by a superficially
    compatible nominal candidate that merely happens to precede it."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_answers_11_shape_defers_to_proposition_not_nominal_entity(self):
        # A concept entity ("Transparency") is genuinely mentioned in the
        # immediately preceding unit -- exactly the shape that used to let
        # Tier 3 nominal resolution consume "This" before the propositional
        # Proposition resolver ever got a chance.
        prev_text = "Defendants cannot meaningfully challenge a system that lacks algorithmic transparency."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "This undermines legal principles such as the right to contest evidence."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]

        local = prev_text.index("transparency")
        transparency = _ent(
            "https://kg/e/transparency", "Transparency", "Concept", discourse_id="A11",
            start_char=local, end_char=local + len("transparency"),
        )
        prev_relation = Relation(
            source="https://kg/e/defendants", relation="cannotChallenge", target="a system",
            target_is_literal=True, uri="https://kg/r/rel_prev11",
            metadata={"evidence_quote": prev_text},
        )
        unit_relations_by_id = {"Body:1": [prev_relation]}

        relation = Relation(
            source="This", relation="undermines", target="legal principles",
            target_is_literal=True,
            metadata={"subject_text": "This", "discourse_id": "A11", "evidence_quote": cur_text},
        )

        resolved, anaphora_entities = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[transparency], unit=cur, discourse_id="A11",
            unit_index=1, semantic_units=semantic_units,
        )
        self.assertEqual(
            resolved[0].source, "This",
            "Tier 3 must defer, not consume the endpoint with the nominal 'Transparency' candidate",
        )
        self.assertEqual(anaphora_entities, [])

        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=resolved, unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="A11",
            origin_key="answer.description|A11", field="answer.description",
        )
        self.assertEqual(len(extra), 1)
        proposition = extra[0]
        self.assertEqual(proposition.entity_type, "Proposition")
        self.assertEqual(resolved[0].source, proposition.uri)
        self.assertNotEqual(resolved[0].source, transparency.uri)
        self.assertEqual(
            resolved[0].metadata.get("subject_endpoint_coref_resolution_method"),
            "propositional_preceding_unit_reference",
        )

    def test_ordinary_nominal_it_with_strong_entity_evidence_still_resolves(self):
        prev_text = "The system was deployed across every regional office last year."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "It failed during the first week of production use."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]

        local = prev_text.index("system")
        system = _ent(
            "https://kg/e/system", "System", "AISystem", discourse_id="d_nom",
            start_char=local, end_char=local + len("system"),
        )
        relation = Relation(
            source="It", relation="failed", target="during the first week", target_is_literal=True,
            metadata={"subject_text": "It", "discourse_id": "d_nom", "evidence_quote": cur_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[system], unit=cur, discourse_id="d_nom",
            unit_index=1, semantic_units=semantic_units,
        )
        self.assertEqual(resolved[0].source, system.uri, "'It' with strong nominal evidence must still resolve nominally")
        self.assertEqual(
            resolved[0].metadata.get("subject_endpoint_coref_resolution_method"), "unambiguous_preceding_entity"
        )

    def test_explicit_head_demonstrative_not_routed_to_proposition(self):
        # "This disagreement" is a determiner+noun phrase, not a bare
        # subject pronoun -- it must never even be eligible for the defer
        # path, regardless of sentence position.
        prev_text = "Northpointe contested the finding and argued accuracy parity held."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "This disagreement highlights a critical issue in the field."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]

        self.assertFalse(
            self.pipeline._endpoint_prefers_propositional_defer(
                raw_text="This disagreement", unit_text=cur_text, endpoint_offset=cur["start_offset"],
                unit_start=cur["start_offset"], unit_index=1, semantic_units=semantic_units,
            )
        )
        # And end-to-end: with no compatible entity extracted at all, the
        # phrase simply stays unresolved raw text (the unresolved-endpoint
        # guard's concern, not this resolver's) -- it is not force-routed to
        # a Proposition either.
        relation = Relation(
            source="This disagreement", relation="highlights", target="a critical issue",
            target_is_literal=True,
            metadata={"subject_text": "This disagreement", "discourse_id": "d_head", "evidence_quote": cur_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[], unit=cur, discourse_id="d_head",
            unit_index=1, semantic_units=semantic_units,
        )
        self.assertEqual(resolved[0].source, "This disagreement")
        resolved, prop_extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=resolved, unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id={}, discourse_id="d_head",
            origin_key="answer.description|d_head", field="answer.description",
        )
        self.assertEqual(resolved[0].source, "This disagreement")
        self.assertEqual(prop_extra, [])

    def test_expletive_it_creates_no_proposition_and_is_never_deferred(self):
        prev_text = "The team ran a full compliance review last quarter."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "It is important to document every exception that arises."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]

        self.assertFalse(
            self.pipeline._endpoint_prefers_propositional_defer(
                raw_text="It", unit_text=cur_text, endpoint_offset=cur["start_offset"],
                unit_start=cur["start_offset"], unit_index=1, semantic_units=semantic_units,
            ),
            "'it' is never a defer trigger, expletive or otherwise",
        )

    def test_genuine_clause_local_tier2_reference_remains_entity_resolution(self):
        unit_text = "Some students will struggle, so the lecturer plans to reach out to them proactively."
        unit_start = 500
        unit = _unit("Body:3", unit_text, unit_start)
        local = unit_text.index("students")
        students = _ent(
            "https://kg/e/students", "Students", "Population", discourse_id="d_tier2",
            start_char=unit_start + local, end_char=unit_start + local + len("students"),
        )
        relation = Relation(
            source="the lecturer", relation="reachOutTo", target="them", target_is_literal=True,
            metadata={"object_text": "them", "discourse_id": "d_tier2", "evidence_quote": unit_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[students], unit=unit, discourse_id="d_tier2",
            unit_index=0, semantic_units=[unit],
        )
        self.assertEqual(resolved[0].target, students.uri)
        self.assertEqual(
            resolved[0].metadata.get("object_endpoint_coref_resolution_method"), "occurrence_level_preceding_mention"
        )

    def test_ambiguous_proposition_candidate_stays_unresolved_not_forced_to_entity(self):
        # Only ONE entity is available for Tier 3 (so it would have been an
        # unambiguous nominal candidate pre-fix), but the preceding unit's
        # OWN relations show two distinct subjects -- the propositional
        # resolver's independent ambiguity gate must still decline, and the
        # endpoint must end up unresolved rather than forced onto either
        # the deferred-past nominal candidate or a Proposition.
        prev_text = "Team Alpha missed the deadline. Team Beta shipped early."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "This complicates the retrospective for everyone involved."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]

        local = prev_text.index("Team Alpha")
        team_alpha = _ent(
            "https://kg/e/team_alpha", "Team Alpha", "Organisation", discourse_id="d_amb",
            start_char=local, end_char=local + len("Team Alpha"),
        )
        prev_relations = [
            Relation(source="https://kg/e/team_alpha", relation="missed", target="the deadline",
                     target_is_literal=True, uri="https://kg/r/rel_a2"),
            Relation(source="https://kg/e/team_beta", relation="shipped", target="early",
                     target_is_literal=True, uri="https://kg/r/rel_b2"),
        ]
        unit_relations_by_id = {"Body:1": prev_relations}

        relation = Relation(
            source="This", relation="complicates", target="the retrospective", target_is_literal=True,
            metadata={"subject_text": "This", "discourse_id": "d_amb", "evidence_quote": cur_text},
        )
        resolved, anaphora_entities = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[team_alpha], unit=cur, discourse_id="d_amb",
            unit_index=1, semantic_units=semantic_units,
        )
        self.assertEqual(resolved[0].source, "This", "must defer rather than consuming the sole nominal candidate")
        self.assertEqual(anaphora_entities, [])

        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=resolved, unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="d_amb",
            origin_key="answer.description|d_amb", field="answer.description",
        )
        self.assertEqual(resolved[0].source, "This", "ambiguous preceding unit must leave the endpoint unresolved")
        self.assertEqual(extra, [])

    def test_possessive_reuse_untouched_no_unit_index_supplied(self):
        # The possessive resolver's antecedent search reuses
        # _resolve_anaphoric_endpoint_antecedent WITHOUT unit_index/
        # semantic_units -- the defer gate must stay a complete no-op there,
        # exactly like the 6.2.1/6.2.2 gates it is modelled on.
        unit_text = "if organisations/teams do this, share their approach publically."
        unit = _unit("Body:1", unit_text, 0)
        team = Entity(
            uri="https://kg/e/team2", label="Organisations Teams", entity_type="Concept",
            confidence=0.9, metadata={"discourse_id": "d_poss"},
        )
        relation = Relation(
            source=team.uri, relation="share", target="their approach",
            metadata={"object_text": "their approach", "discourse_id": "d_poss", "evidence_quote": unit_text},
        )
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[team], discourse_text=unit_text, discourse_id="d_poss",
            field="answer.description", origin_key="answer.description|d_poss", unit=unit, unit_text=unit_text,
        )
        final = self.pipeline._apply_possessive_endpoint_semantics(
            relations=resolved, entities=[team], local_pool_entities=[team], unit=unit, discourse_id="d_poss",
        )
        self.assertEqual(final[0].target, "approach")
        self.assertTrue(final[0].target_is_literal)
        self.assertEqual(final[0].metadata.get("object_possessor_entity_uri"), team.uri)


if __name__ == "__main__":
    unittest.main()
