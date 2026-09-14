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


class OccurrenceLevelSameUnitResolutionTests(unittest.TestCase):
    """Bare nominal pronouns resolved using the endpoint's own
    occurrence position, not the semantic unit's start_offset."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_same_sentence_antecedent_before_pronoun_resolves(self):
        unit_start = 500
        unit_text = "Some residents will be affected by delays and staff proactively reach out to them."
        local_start = unit_text.index("residents")
        residents = _ent(
            "https://kg/e/residents", "Residents", "Group",
            discourse_id="d1", start_char=unit_start + local_start, end_char=unit_start + local_start + len("residents"),
        )
        relation = Relation(
            source="staff", relation="reachOut", target="them", target_is_literal=True,
            metadata={"object_text": "them", "discourse_id": "d1", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[residents], unit=unit, discourse_id="d1",
        )
        self.assertEqual(resolved[0].target, residents.uri)
        self.assertFalse(resolved[0].target_is_literal)
        self.assertEqual(resolved[0].metadata.get("object_endpoint_coref_resolution_method"), "occurrence_level_preceding_mention")
        self.assertEqual(resolved[0].metadata.get("object_endpoint_original_surface"), "them")
        self.assertEqual(extra, [residents])

    def test_antecedent_earlier_in_the_same_unit_general_case(self):
        unit_start = 200
        unit_text = "The Screening Tool flagged several applicants, and reviewers later overruled it."
        local_start = unit_text.index("Screening Tool")
        tool = _ent(
            "https://kg/e/screening_tool", "Screening Tool", "AISystem",
            discourse_id="d2", start_char=unit_start + local_start, end_char=unit_start + local_start + len("Screening Tool"),
        )
        relation = Relation(
            source="reviewers", relation="overruled", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "d2", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, _ = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[tool], unit=unit, discourse_id="d2",
        )
        self.assertEqual(resolved[0].target, tool.uri)
        self.assertEqual(resolved[0].metadata.get("object_endpoint_coref_resolution_method"), "occurrence_level_preceding_mention")

    def test_antecedent_in_immediately_preceding_unit_still_resolves(self):
        # Preceding-unit resolution (Tier 3) is unchanged by occurrence-level
        # resolution; confirm
        # it still works when the current unit also carries unit text.
        prior_unit_text = "An Automated Triage System is used to prioritize requests."
        current_unit_start = 200
        current_unit_text = "Response Delay: It causes significantly longer wait times."
        system = _ent(
            "https://kg/e/triage_system", "Automated Triage System", "AISystem",
            discourse_id="d3", start_char=3, end_char=3 + len("Automated Triage System"),
        )
        relation = Relation(
            source="it", relation="causes", target="significantly longer wait times", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d3", "evidence_quote": current_unit_text},
        )
        unit = {"start_offset": current_unit_start, "text": current_unit_text}
        resolved, _ = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[system], unit=unit, discourse_id="d3",
        )
        self.assertEqual(resolved[0].source, system.uri)
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_coref_resolution_method"), "unambiguous_preceding_entity")

    def test_later_same_unit_mention_is_not_selected_cataphora(self):
        unit_start = 300
        unit_text = "It struggled to justify a decision, and reviewers flagged Model Z as unreliable."
        local_start = unit_text.index("Model Z")
        model_z = _ent(
            "https://kg/e/model_z", "Model Z", "AISystem",
            discourse_id="d4", start_char=unit_start + local_start, end_char=unit_start + local_start + len("Model Z"),
        )
        relation = Relation(
            source="it", relation="struggled", target="justify a decision", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d4", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[model_z], unit=unit, discourse_id="d4",
        )
        # "Model Z" occurs AFTER "It" in the same unit -- must not be used.
        self.assertEqual(resolved[0].source, "it")
        self.assertEqual(extra, [])

    def test_two_equally_plausible_same_unit_candidates_stay_unresolved(self):
        unit_start = 100
        unit_text = "System Alpha and System Beta were both deployed, and staff later disabled it."
        alpha_start = unit_text.index("System Alpha")
        beta_start = unit_text.index("System Beta")
        alpha = _ent(
            "https://kg/e/sys_alpha", "System Alpha", "AISystem",
            discourse_id="d5", start_char=unit_start + alpha_start, end_char=unit_start + alpha_start + len("System Alpha"),
        )
        beta = _ent(
            "https://kg/e/sys_beta", "System Beta", "AISystem",
            discourse_id="d5", start_char=unit_start + beta_start, end_char=unit_start + beta_start + len("System Beta"),
        )
        relation = Relation(
            source="staff", relation="disabled", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "d5", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[alpha, beta], unit=unit, discourse_id="d5",
        )
        self.assertEqual(resolved[0].target, "it")
        self.assertTrue(resolved[0].target_is_literal)
        self.assertEqual(extra, [])

    def test_no_cross_discourse_resolution_for_same_unit_tier(self):
        unit_start = 500
        unit_text = "Some residents will be affected by delays and staff proactively reach out to them."
        local_start = unit_text.index("residents")
        other_discourse_residents = _ent(
            "https://kg/e/residents_other", "Residents", "Group",
            discourse_id="d_other", start_char=unit_start + local_start, end_char=unit_start + local_start + len("residents"),
        )
        relation = Relation(
            source="staff", relation="reachOut", target="them", target_is_literal=True,
            metadata={"object_text": "them", "discourse_id": "d6", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[other_discourse_residents], unit=unit, discourse_id="d6",
        )
        self.assertEqual(resolved[0].target, "them")
        self.assertEqual(extra, [])


class QuestionsSeventeenLiveShapeReplayTest(unittest.TestCase):
    """End-to-end replay of the real questions/17 shape (generic content,
    not the literal dataset text) through _process_relations_batch."""

    def test_them_resolves_to_students_entity_end_to_end(self):
        pipeline = _make_pipeline()
        did = "T7"
        text = (
            "It's hypothetically possible to use a model to rank candidates.\n\n"
            "Also, I worry about this from the perspective of a mentor - sometimes we might "
            "predict that some students will struggle more with certain topics and "
            "proactively reach out to them."
        )
        local_start = text.index("students")
        students = _ent(
            "https://kg/e/students", "Students", "Group",
            discourse_id=did, start_char=local_start, end_char=local_start + len("students"),
        )
        entities = [students]

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            if "reach out to them" not in unit_text:
                return []
            return [
                Relation(
                    source="a mentor", relation="reachOut", target="them", target_is_literal=True,
                    confidence=0.9,
                    metadata={
                        "extractor": "llm", "context_source": "llm_relation",
                        "subject_text": "a mentor", "object_text": "them",
                        "evidence_quote": unit_text, "evidence": unit_text,
                    },
                )
            ]

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [text], [entities], [did], provenance_list=[{"field": "question.body", "discourse_id": did}],
        )
        relations = relations_list[0]

        matching = [r for r in relations if str(getattr(r, "relation", "")) == "reachOut"]
        self.assertEqual(len(matching), 1, f"expected exactly one relation, got: {relations}")
        relation = matching[0]

        self.assertTrue(relation.metadata.get("object_endpoint_coref_resolved"))
        self.assertEqual(relation.metadata.get("object_endpoint_coref_resolution_method"), "occurrence_level_preceding_mention")
        self.assertEqual(relation.metadata.get("object_endpoint_original_surface"), "them")
        self.assertFalse(relation.target_is_literal)


if __name__ == "__main__":
    unittest.main()
