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


def _unit(unit_id, text, start_offset):
    return {
        "unit_id": unit_id, "text": text, "start_offset": start_offset,
        "end_offset": start_offset + len(text), "section_category": "semantic", "unit_type": "sentence",
    }


class UnrepresentedLocalAntecedentTests(unittest.TestCase):
    """Tier 3 must not prefer a distant, broader canonical
    entity over an unrepresented but structurally plausible local NP."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_local_unregistered_np_blocks_farther_compatible_entity(self):
        # Reproduces the real answers/17 shape: "Systems" is family
        # -compatible and in the immediately preceding unit, but "the
        # application" (never extracted) sits much closer to the pronoun,
        # within the same clause, with a first "it" directly adjacent to it.
        u0_text = "I think these systems are really hard to trust because you cannot make it explainable."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = (
            "An example - if you say something relevant to the application it can show you "
            "'here is where they discuss it' for example, but if it is trying to show you "
            "that they do not discuss something you cannot point at nothing."
        )
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        units = [u0, u1]

        sys_local = u0_text.index("systems")
        systems = _ent(
            "https://kg/e/systems", "Systems", "AISystem", discourse_id="d1",
            start_char=sys_local, end_char=sys_local + len("systems"),
        )
        relation = Relation(
            source="it", relation="show", target="something", target_is_literal=True,
            metadata={
                "subject_text": "it", "discourse_id": "d1",
                "evidence_quote": "but if it is trying to show you that they do not discuss something",
            },
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[systems], unit=u1, discourse_id="d1",
            unit_index=1, semantic_units=units,
        )
        self.assertEqual(resolved[0].source, "it")
        self.assertEqual(extra, [])

    def test_no_nearer_plausible_np_immediately_preceding_exact_entity_resolves(self):
        u0_text = "An Automated Triage System is used to prioritize requests."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = "Response Delay: It causes significantly longer wait times."
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        units = [u0, u1]

        local = u0_text.index("Automated Triage System")
        triage = _ent(
            "https://kg/e/triage", "Automated Triage System", "AISystem", discourse_id="d2",
            start_char=local, end_char=local + len("Automated Triage System"),
        )
        relation = Relation(
            source="it", relation="causes", target="significantly longer wait times", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d2", "evidence_quote": u1_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[triage], unit=u1, discourse_id="d2",
            unit_index=1, semantic_units=units,
        )
        self.assertEqual(resolved[0].source, triage.uri)

    def test_nearer_np_inside_unrelated_quote_does_not_block_resolution(self):
        u0_text = "An Automated Triage System is used to prioritize requests."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = "Staff said 'the old backup process' was retired, and it causes longer wait times now."
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        units = [u0, u1]

        local = u0_text.index("Automated Triage System")
        triage = _ent(
            "https://kg/e/triage2", "Automated Triage System", "AISystem", discourse_id="d3",
            start_char=local, end_char=local + len("Automated Triage System"),
        )
        relation = Relation(
            source="it", relation="causes", target="longer wait times now", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d3", "evidence_quote": u1_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[triage], unit=u1, discourse_id="d3",
            unit_index=1, semantic_units=units,
        )
        # "the old backup process" is inside a quotation the endpoint "it"
        # is not part of -- must not block resolution to the genuine,
        # adjacent-unit antecedent.
        self.assertEqual(resolved[0].source, triage.uri)

    def test_plural_pronoun_not_blocked_by_singular_nearer_np(self):
        u0_text = "Automated Triage Systems are used to prioritize many requests each day."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = "An example is the new dashboard, and they cause significantly longer wait times."
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        units = [u0, u1]

        local = u0_text.index("Automated Triage Systems")
        triage_systems = _ent(
            "https://kg/e/triage_systems", "Automated Triage Systems", "AISystem", discourse_id="d4",
            start_char=local, end_char=local + len("Automated Triage Systems"),
        )
        relation = Relation(
            source="they", relation="cause", target="significantly longer wait times", target_is_literal=True,
            metadata={"subject_text": "they", "discourse_id": "d4", "evidence_quote": u1_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[triage_systems], unit=u1, discourse_id="d4",
            unit_index=1, semantic_units=units,
        )
        # "the new dashboard" (singular NP) must not block a PLURAL pronoun
        # from resolving to the genuine, adjacent-unit antecedent.
        self.assertEqual(resolved[0].source, triage_systems.uri)

    def test_repeated_pronoun_chain_with_local_missed_np_abstains(self):
        u0_text = "I think these systems are really hard to trust because you cannot make it clear."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = (
            "An example - if you say something relevant to the application it can show you "
            "results, but if it is trying to show you something else it cannot point at nothing."
        )
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        units = [u0, u1]

        sys_local = u0_text.index("systems")
        systems = _ent(
            "https://kg/e/systems2", "Systems", "AISystem", discourse_id="d5",
            start_char=sys_local, end_char=sys_local + len("systems"),
        )
        relation = Relation(
            source="it", relation="show", target="something else", target_is_literal=True,
            metadata={
                "subject_text": "it", "discourse_id": "d5",
                "evidence_quote": "but if it is trying to show you something else",
            },
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[systems], unit=u1, discourse_id="d5",
            unit_index=1, semantic_units=units,
        )
        self.assertEqual(resolved[0].source, "it")
        self.assertEqual(extra, [])


if __name__ == "__main__":
    unittest.main()
