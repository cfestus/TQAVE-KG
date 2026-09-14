import unittest

from config import Config
from models import Entity
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.semantic_chunker import SemanticChunker
from utils.semantic_relation_policy import SemanticRelationPolicy


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
    pipeline.semantic_policy = SemanticRelationPolicy()
    pipeline.coref_resolver = _EchoResolver()
    pipeline._discourse_chunk_spans = {}
    pipeline._candidate_generation_diagnostics = []
    return pipeline


def _entity(uri_suffix, label, entity_type, discourse_id, start_char, end_char, *, section="Body"):
    return Entity(
        uri=f"{Config.NAMESPACES['etd']}{uri_suffix}",
        label=label,
        entity_type=entity_type,
        confidence=1.0,
        metadata={
            "discourse_id": discourse_id,
            "section": section,
            "section_category": "semantic",
            "start_char": start_char,
            "end_char": end_char,
            "mentions": [
                {
                    "discourse_id": discourse_id,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            ],
        },
    )


class SemanticRecallRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def _normalized_labels(self, entities):
        return {
            self.pipeline.entity_normalizer.normalize_label(entity.label)
            for entity in entities
        }

    def test_surface_recovery_rejects_clause_like_subject_fragment(self):
        text = (
            "Fairness in algorithmic decision-making, especially in high-stakes domains like criminal justice, "
            "goes beyond technical accuracy; it requires accountability, interpretability, and equity."
        )
        section_spans = SemanticChunker.parse_structured_sections(text, field="answer.description")
        chunk_spans = [{"chunk_index": 0, "start_offset": 0, "end_offset": len(text)}]
        entities = []

        self.pipeline._augment_lexical_entities(
            text=text,
            metadata={"discourse_id": "d_clause_subject", "field": "answer.description", "origin_key": "answer.description|d_clause_subject"},
            entities=entities,
            section_spans=section_spans,
            chunk_spans=chunk_spans,
        )

        labels = {entity.label for entity in entities}
        self.assertNotIn("Goes Beyond Technical Accuracy", labels)
        self.assertIn("accountability", labels)

    def test_surface_recovery_rejects_clause_like_object_fragment(self):
        text = (
            "AI recruitment tool trained on historical workforce data. "
            "A UW study found names associated with Black women were disadvantaged."
        )
        section_spans = SemanticChunker.parse_structured_sections(text, field="theme.description")
        chunk_spans = [{"chunk_index": 0, "start_offset": 0, "end_offset": len(text)}]
        entities = []

        self.pipeline._augment_lexical_entities(
            text=text,
            metadata={"discourse_id": "d_clause_object", "field": "theme.description", "origin_key": "theme.description|d_clause_object"},
            entities=entities,
            section_spans=section_spans,
            chunk_spans=chunk_spans,
        )

        labels = {entity.label for entity in entities}
        self.assertNotIn("UW study found names associated with Black women", labels)
        self.assertIn("AI recruitment tool", labels)

    def test_surface_recovery_preserves_compound_system_and_dataset_phrases(self):
        text = "AI recruitment system trained on historical workforce data."
        section_spans = SemanticChunker.parse_structured_sections(text, field="theme.description")
        chunk_spans = [{"chunk_index": 0, "start_offset": 0, "end_offset": len(text)}]
        entities = []

        self.pipeline._augment_lexical_entities(
            text=text,
            metadata={"discourse_id": "d1", "field": "theme.description", "origin_key": "theme.description|d1"},
            entities=entities,
            section_spans=section_spans,
            chunk_spans=chunk_spans,
        )

        labels = {entity.label for entity in entities}
        self.assertIn("AI recruitment system", labels)
        self.assertIn("historical workforce data", labels)

    def test_surface_recovery_splits_distinct_coordinated_targets(self):
        text = (
            "AI-powered insurance underwriting systems show bias in risk assessment and pricing, "
            "particularly affecting minority communities and certain geographic areas."
        )
        section_spans = SemanticChunker.parse_structured_sections(text, field="theme.description")
        chunk_spans = [{"chunk_index": 0, "start_offset": 0, "end_offset": len(text)}]
        entities = []

        self.pipeline._augment_lexical_entities(
            text=text,
            metadata={"discourse_id": "d_split", "field": "theme.description", "origin_key": "theme.description|d_split"},
            entities=entities,
            section_spans=section_spans,
            chunk_spans=chunk_spans,
        )

        labels = self._normalized_labels(entities)
        self.assertIn("Minority Communities And Certain Geographic Areas", labels)
        self.assertIn("Minority Communities", labels)
        self.assertIn("Certain Geographic Areas", labels)

    def test_surface_recovery_does_not_split_single_group_with_coordinated_modifier(self):
        text = (
            "Dynamic pricing algorithms systematically discriminate against neighborhoods with non-white "
            "and low-income populations."
        )
        section_spans = SemanticChunker.parse_structured_sections(text, field="theme.description")
        chunk_spans = [{"chunk_index": 0, "start_offset": 0, "end_offset": len(text)}]
        entities = []

        self.pipeline._augment_lexical_entities(
            text=text,
            metadata={"discourse_id": "d_nonsplit", "field": "theme.description", "origin_key": "theme.description|d_nonsplit"},
            entities=entities,
            section_spans=section_spans,
            chunk_spans=chunk_spans,
        )

        labels = self._normalized_labels(entities)
        self.assertIn("Neighborhoods With Non-White And Low-Income Populations", labels)
        self.assertNotIn("Neighborhoods With Non-White", labels)
        self.assertNotIn("Low-Income Populations", labels)

    def test_local_entity_selection_prefers_specific_surface_entity(self):
        unit = {
            "text": "AI recruitment tool",
            "start_offset": 0,
            "end_offset": 19,
            "section": "Body",
            "section_category": "semantic",
        }
        entities = [
            _entity("ai_generic", "AI", "AISystem", "d1", 0, 2),
            _entity("ai_tool", "AI recruitment tool", "AISystem", "d1", 0, 19),
            _entity("dashboard", "Dashboard", "Dataset", "d1", 30, 39),
        ]

        selected = self.pipeline._local_entities_for_unit(entities, unit, "d1")
        labels = [entity.label for entity in selected]
        self.assertIn("AI recruitment tool", labels)
        self.assertNotIn("AI", labels)

    def test_candidate_pairs_are_cue_guided_not_all_pairs(self):
        sentence = "AI recruitment system trained on historical workforce data."
        entities = [
            _entity("system", "AI recruitment system", "AISystem", "d1", 0, 21),
            _entity("dataset", "historical workforce data", "Dataset", "d1", 33, 59),
            _entity("dashboard", "Dashboard", "Dataset", "d1", 80, 89),
        ]

        pairs = self.pipeline._cue_guided_candidate_pairs_for_sentence(
            sentence=sentence,
            sentence_entities=entities,
        )

        signatures = {
            (pair["source_label"], tuple(pair["predicate_hints"]), pair["target_label"])
            for pair in pairs
        }
        self.assertIn(("AI recruitment system", ("trainedOn",), "historical workforce data"), signatures)
        self.assertNotIn(("AI recruitment system", ("trainedOn",), "Dashboard"), signatures)
        self.assertEqual(len(signatures), 1)

    def test_safe_local_anaphora_is_annotated_and_reused(self):
        text = "AI recruitment systems are increasingly used in hiring. These systems may amplify historical bias."
        units = SemanticChunker.semantic_relation_units(text, field="theme.description")
        self.assertEqual(len(units), 2)
        self.assertIn("anaphora", units[1])

        actor = _entity("actor", "AI recruitment systems", "AISystem", "d1", 0, 22, section="Description")
        bias = _entity("bias", "historical bias", "AlgorithmicBias", "d1", 73, 88, section="Description")

        selected = self.pipeline._local_entities_for_unit([actor, bias], units[1], "d1")
        actor_selected = next(entity for entity in selected if entity.label == "AI recruitment systems")
        aliases = actor_selected.metadata.get("aliases") or []
        self.assertIn("These systems", aliases)

    def test_non_safe_pronoun_does_not_create_anaphora(self):
        text = "AI recruitment systems are increasingly used in hiring. They may amplify historical bias."
        units = SemanticChunker.semantic_relation_units(text, field="theme.description")
        self.assertEqual(len(units), 2)
        self.assertNotIn("anaphora", units[1])

    def test_system_actor_beats_dataset_actor_for_exhibits_risk(self):
        sentence = "A predictive system like COMPAS exhibits systemic racial disparities."
        entities = [
            _entity("system", "predictive system like COMPAS", "AISystem", "d1", 2, 31),
            _entity("dataset", "COMPAS", "Dataset", "d1", 27, 33),
            _entity("risk", "systemic racial disparities", "AlgorithmicBias", "d1", 43, 71),
        ]

        pairs = self.pipeline._cue_guided_candidate_pairs_for_sentence(
            sentence=sentence,
            sentence_entities=entities,
        )

        self.assertTrue(pairs)
        source_labels = {pair["source_label"] for pair in pairs if "exhibitsRisk" in pair["predicate_hints"]}
        self.assertIn("predictive system like COMPAS", source_labels)
        self.assertNotIn("COMPAS", source_labels)

    def test_candidate_pairs_skip_contextual_uses_and_stop_at_trained_on(self):
        sentence = (
            "Predictive System used in Life-Altering Decisions may exhibit Systemic Racial Disparities "
            "even when trained on Real-World Data."
        )
        entities = []
        for uri_suffix, label, entity_type in [
            ("predictive_system", "Predictive System", "AISystem"),
            ("life_altering_decisions", "Life-Altering Decisions", "Impact"),
            ("systemic_racial_disparities", "Systemic Racial Disparities", "AlgorithmicBias"),
            ("real_world_data", "Real-World Data", "Dataset"),
        ]:
            start = sentence.index(label)
            end = start + len(label)
            entities.append(_entity(uri_suffix, label, entity_type, "d2", start, end))

        pairs = self.pipeline._cue_guided_candidate_pairs_for_sentence(
            sentence=sentence,
            sentence_entities=entities,
        )

        signatures = {
            (pair["source_label"], tuple(pair["predicate_hints"]), pair["target_label"])
            for pair in pairs
        }
        self.assertIn(("Predictive System", ("exhibitsRisk",), "Systemic Racial Disparities"), signatures)
        self.assertIn(("Predictive System", ("trainedOn",), "Real-World Data"), signatures)
        self.assertNotIn(("Predictive System", ("uses",), "Life-Altering Decisions"), signatures)
        self.assertNotIn(("Predictive System", ("uses",), "Real-World Data"), signatures)

    def test_candidate_pairs_do_not_cross_into_subordinate_clause_after_affecting(self):
        sentence = (
            "Commercial Health Risk Algorithms affecting 200 million Americans annually, "
            "used to identify patients needing Additional Care Programs."
        )
        entities = []
        for uri_suffix, label, entity_type in [
            ("health_risk_algorithms", "Commercial Health Risk Algorithms", "AISystem"),
            ("additional_care_programs", "Additional Care Programs", "Concept"),
        ]:
            start = sentence.index(label)
            end = start + len(label)
            entities.append(_entity(uri_suffix, label, entity_type, "d3", start, end))

        pairs = self.pipeline._cue_guided_candidate_pairs_for_sentence(
            sentence=sentence,
            sentence_entities=entities,
        )

        signatures = {
            (pair["source_label"], tuple(pair["predicate_hints"]), pair["target_label"])
            for pair in pairs
        }
        self.assertNotIn(
            ("Commercial Health Risk Algorithms", ("affects",), "Additional Care Programs"),
            signatures,
        )

    def test_candidate_pairs_do_not_cross_by_gerund_subordinate_clause(self):
        sentence = "Could the AI system affect human autonomy by generating over-reliance by end-users?"
        entities = []
        for uri_suffix, needle, label, entity_type in [
            ("ai_system", "ai system", "AI system", "AISystem"),
            ("human_autonomy", "human autonomy", "Human Autonomy", "Concept"),
            ("end_users", "end-users", "End-Users", "Population/Group"),
        ]:
            start = sentence.lower().index(needle)
            end = start + len(needle)
            entities.append(_entity(uri_suffix, label, entity_type, "d4", start, end))

        pairs = self.pipeline._cue_guided_candidate_pairs_for_sentence(
            sentence=sentence,
            sentence_entities=entities,
        )

        signatures = {
            (pair["source_label"], tuple(pair["predicate_hints"]), pair["target_label"])
            for pair in pairs
        }
        self.assertIn(("AI system", ("affects",), "Human Autonomy"), signatures)
        self.assertNotIn(("AI system", ("affects",), "End-Users"), signatures)


if __name__ == "__main__":
    unittest.main()
