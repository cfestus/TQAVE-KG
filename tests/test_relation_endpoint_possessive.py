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
            "mentions": [
                {"discourse_id": discourse_id, "chunk_index": 0, "start_char": start_char, "end_char": end_char}
            ],
        },
    )


class PossessiveEndpointSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_reflexive_subject_their_approach(self):
        # "organisations/teams ... share their approach" -- the possessive
        # has no separate nearby entity, so it falls back to the relation's
        # own already-resolved subject.
        orgs = _ent("https://kg/e/orgs", "Organisations", "Organisation", discourse_id="d1", start_char=0, end_char=13)
        relation = Relation(
            source=orgs.uri, relation="share", target="their approach", target_is_literal=True,
            metadata={"object_text": "their approach", "discourse_id": "d1"},
        )
        unit = {"start_offset": 100}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[], local_pool_entities=[orgs], unit=unit, discourse_id="d1",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "approach")
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.metadata.get("object_possessor_entity_uri"), orgs.uri)
        self.assertEqual(rel.metadata.get("object_possessor_resolution_method"), "reflexive_subject")
        self.assertEqual(rel.metadata.get("object_possessor_head_noun_phrase"), "approach")
        self.assertEqual(rel.metadata.get("object_possessor_original_surface"), "their approach")
        # The endpoint must never become the antecedent's URI/label.
        self.assertNotEqual(rel.target, orgs.uri)
        self.assertNotEqual(rel.target, "Organisations")

    def test_general_search_prefers_nearby_entity_over_unrelated_subject(self):
        # "people ... predict their own performance" -- "people" is a
        # distinct, nearby, same-unit entity that is NOT the relation's
        # formal subject (a tool). The contextually correct antecedent must
        # win over the reflexive-subject fallback.
        unit_start = 200
        unit_text = "Some people applying could use this tool to predict their own performance."
        people_local = unit_text.index("people")
        people = _ent(
            "https://kg/e/people", "People", "Population",
            discourse_id="d2", start_char=unit_start + people_local, end_char=unit_start + people_local + len("people"),
        )
        tool = _ent("https://kg/e/tool", "This Tool", "AISystem", discourse_id="d2", start_char=5, end_char=20)
        relation = Relation(
            source=tool.uri, relation="predict", target="their own performance", target_is_literal=True,
            metadata={"object_text": "their own performance", "discourse_id": "d2", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[people], local_pool_entities=[tool], unit=unit, discourse_id="d2",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "performance")
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.metadata.get("object_possessor_entity_uri"), people.uri)
        self.assertEqual(rel.metadata.get("object_possessor_resolution_method"), "occurrence_level_preceding_mention")

    def test_reflexive_subject_its_accuracy(self):
        # "system improves its accuracy" -- "its" only accepts a
        # non-person-family possessor; the system subject qualifies.
        system = _ent("https://kg/e/system_x", "System X", "AISystem", discourse_id="d3", start_char=0, end_char=8)
        relation = Relation(
            source=system.uri, relation="improves", target="its accuracy", target_is_literal=True,
            metadata={"object_text": "its accuracy", "discourse_id": "d3"},
        )
        unit = {"start_offset": 100}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[], local_pool_entities=[system], unit=unit, discourse_id="d3",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "accuracy")
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.metadata.get("object_possessor_entity_uri"), system.uri)
        self.assertEqual(rel.metadata.get("object_possessor_resolution_method"), "reflexive_subject")

    def test_its_does_not_reflexively_resolve_to_a_person_subject(self):
        # Family-compatibility guard: "its" must not fall back to a
        # person-family subject.
        person = _ent("https://kg/e/reviewer", "Reviewer", "Person", discourse_id="d3b", start_char=0, end_char=8)
        relation = Relation(
            source=person.uri, relation="improves", target="its accuracy", target_is_literal=True,
            metadata={"object_text": "its accuracy", "discourse_id": "d3b"},
        )
        unit = {"start_offset": 100}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[], local_pool_entities=[person], unit=unit, discourse_id="d3b",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "accuracy")  # head noun still applied
        self.assertNotIn("object_possessor_entity_uri", rel.metadata)  # but possessor stays unresolved

    def test_ambiguous_possessive_antecedent_remains_unresolved(self):
        unit_start = 100
        unit_text = "Team Alpha and Team Beta worked together, and later published their approach."
        alpha_local = unit_text.index("Team Alpha")
        beta_local = unit_text.index("Team Beta")
        alpha = _ent(
            "https://kg/e/team_alpha", "Team Alpha", "Organisation",
            discourse_id="d4", start_char=unit_start + alpha_local, end_char=unit_start + alpha_local + len("Team Alpha"),
        )
        beta = _ent(
            "https://kg/e/team_beta", "Team Beta", "Organisation",
            discourse_id="d4", start_char=unit_start + beta_local, end_char=unit_start + beta_local + len("Team Beta"),
        )
        unrelated_subject = _ent("https://kg/e/reporter", "Reporter", "Person", discourse_id="d4", start_char=1000, end_char=1008)
        relation = Relation(
            source=unrelated_subject.uri, relation="published", target="their approach", target_is_literal=True,
            metadata={"object_text": "their approach", "discourse_id": "d4", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[alpha, beta], local_pool_entities=[unrelated_subject], unit=unit, discourse_id="d4",
        )
        rel = resolved[0]
        # Head noun still preserved as the endpoint...
        self.assertEqual(rel.target, "approach")
        self.assertTrue(rel.target_is_literal)
        # ...but the possessor is left unresolved rather than guessed.
        self.assertNotIn("object_possessor_entity_uri", rel.metadata)
        self.assertNotIn("object_possessor_resolution_method", rel.metadata)

    def test_no_cross_discourse_possessor_resolution(self):
        unit_start = 200
        unit_text = "Some people applying could use this tool to predict their own performance."
        people_local = unit_text.index("people")
        # Same shape as the successful general-search test, but the
        # candidate entity is tagged to a DIFFERENT discourse.
        people_other_discourse = _ent(
            "https://kg/e/people_other", "People", "Population",
            discourse_id="d_other", start_char=unit_start + people_local, end_char=unit_start + people_local + len("people"),
        )
        tool = _ent("https://kg/e/tool2", "This Tool", "AISystem", discourse_id="d5", start_char=5, end_char=20)
        relation = Relation(
            source=tool.uri, relation="predict", target="their own performance", target_is_literal=True,
            metadata={"object_text": "their own performance", "discourse_id": "d5", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved = self.pipeline._apply_possessive_endpoint_semantics(
            relations=[relation], entities=[people_other_discourse], local_pool_entities=[tool], unit=unit, discourse_id="d5",
        )
        rel = resolved[0]
        self.assertEqual(rel.target, "performance")
        # No cross-discourse candidate, and "its"/"their" reflexive check on
        # `tool` (system family, unrestricted for "their") DOES apply here,
        # since "their" has no family hint -- so it falls back to the
        # relation's own (same-discourse) subject instead of the
        # cross-discourse "People" entity.
        self.assertEqual(rel.metadata.get("object_possessor_entity_uri"), tool.uri)
        self.assertEqual(rel.metadata.get("object_possessor_resolution_method"), "reflexive_subject")


class PossessiveEndToEndPipelineTest(unittest.TestCase):
    """End-to-end replay of the real answers/29 'their approach' shape
    through _process_relations_batch."""

    def test_organisations_share_their_approach_end_to_end(self):
        pipeline = _make_pipeline()
        did = "T9"
        text = (
            "An Ethics Review Framework is used to evaluate new projects before launch.\n\n"
            "Operational Notes:\n"
            "* Transparency: I also think that if organisations/teams do this, share their "
            "approach publically, this process can be supercharged\n"
            "* Staffing: On-call staff report chronic overtime"
        )

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            if "share their" not in unit_text:
                return []
            return [
                Relation(
                    source="organisations/teams", relation="share", target="their approach", target_is_literal=True,
                    confidence=0.9,
                    metadata={
                        "extractor": "llm", "context_source": "llm_relation",
                        "subject_text": "organisations/teams", "object_text": "their approach",
                        "evidence_quote": unit_text, "evidence": unit_text,
                    },
                )
            ]

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [text], [[]], [did], provenance_list=[{"field": "answer.description", "discourse_id": did}],
        )
        relations = relations_list[0]

        matching = [r for r in relations if str(getattr(r, "relation", "")) == "share"]
        self.assertEqual(len(matching), 1, f"expected exactly one relation, got: {relations}")
        relation = matching[0]

        self.assertEqual(relation.target, "approach")
        self.assertTrue(relation.target_is_literal)
        self.assertNotEqual(relation.target, relation.source)
        self.assertEqual(relation.metadata.get("object_possessor_resolution_method"), "reflexive_subject")
        possessor_uri = relation.metadata.get("object_possessor_entity_uri")
        self.assertIsNotNone(possessor_uri)
        self.assertEqual(possessor_uri, relation.source)


if __name__ == "__main__":
    unittest.main()
