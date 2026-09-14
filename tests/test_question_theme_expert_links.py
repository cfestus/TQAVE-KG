import unittest
from types import SimpleNamespace

from config import Config
from models import Entity, Relation
from pipeline import EnhancedSemanticPipeline


class QuestionThemeExpertLinkTests(unittest.TestCase):
    def _make_pipeline_stub(self) -> EnhancedSemanticPipeline:
        pipe = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        pipe.config = SimpleNamespace(NAMESPACES=Config.NAMESPACES)
        pipe.ontology_mapper = SimpleNamespace(
            map_problem_category=lambda _: [],
            map_domain=lambda _: None,
        )
        pipe._compose_origin_key = EnhancedSemanticPipeline._compose_origin_key.__get__(pipe, EnhancedSemanticPipeline)
        pipe._process_entities_batch = lambda texts, metadata_list: [[] for _ in texts]
        pipe._process_relations_batch = lambda texts, entities_list, discourse_ids, provenance_list=None: [[] for _ in texts]
        pipe._ensure_relation_chunk_provenance = lambda rel, fallback_discourse_id=None: None
        pipe._classify_question_type = lambda text: "open"

        def _register_structural_node(**kwargs):
            return Entity(
                uri=kwargs["uri"],
                label=kwargs["label"],
                entity_type=kwargs["entity_type"],
                confidence=1.0,
                metadata=kwargs.get("extra_meta", {}),
            )

        pipe._register_structural_node = _register_structural_node
        return pipe

    def test_question_record_with_expert_id_emits_asked_by_relation(self):
        pipe = self._make_pipeline_stub()

        results = pipe.process_questions_batch([
            {
                "id": 14,
                "theme_id": 6,
                "expert_id": 5,
                "body": "How should AI systems be audited?",
            }
        ])

        rels = results[0].relations
        asked_by = [r for r in rels if r.relation == "askedBy"]
        self.assertEqual(len(asked_by), 1)
        self.assertEqual(asked_by[0].source, f"{Config.NAMESPACES['etd']}question_14")
        self.assertEqual(asked_by[0].target, f"{Config.NAMESPACES['etd']}expert_5")
        self.assertEqual(asked_by[0].metadata.get("field"), "question.body")
        self.assertEqual(asked_by[0].metadata.get("source_doc_id"), "14")

    def test_question_semantic_relations_receive_interrogative_metadata(self):
        pipe = self._make_pipeline_stub()
        pipe._process_entities_batch = lambda texts, metadata_list: [[
            Entity(
                uri=f"{Config.NAMESPACES['etd']}ai_model",
                label="AI model",
                entity_type="AISystem",
                confidence=1.0,
                metadata={},
            ),
            Entity(
                uri=f"{Config.NAMESPACES['etd']}dataset",
                label="Dataset",
                entity_type="Dataset",
                confidence=1.0,
                metadata={},
            ),
        ]]
        pipe._process_relations_batch = lambda texts, entities_list, discourse_ids, provenance_list=None: [[
            Relation(
                source=f"{Config.NAMESPACES['etd']}ai_model",
                relation="trainedOn",
                target=f"{Config.NAMESPACES['etd']}dataset",
                confidence=0.9,
                metadata={"relation_kind": "extracted_semantic"},
            )
        ]]

        results = pipe.process_questions_batch([
            {
                "id": 17,
                "theme_id": 10,
                "expert_id": 5,
                "body": "Can an AI model be trained on a dataset of current or past employees?",
            }
        ])

        semantic = [r for r in results[0].relations if r.relation == "trainedOn"]
        self.assertEqual(len(semantic), 1)
        self.assertEqual(semantic[0].metadata.get("discourse_mode"), "interrogative")
        self.assertEqual(semantic[0].metadata.get("epistemic_status"), "questioned")
        self.assertEqual(semantic[0].metadata.get("field"), "question.body")

    def test_theme_record_with_expert_id_emits_created_by_relation(self):
        pipe = self._make_pipeline_stub()

        results = pipe.process_themes_batch([
            {
                "id": 6,
                "expert_id": 5,
                "description": "AI governance in public services.",
                "problem_category": "accountability",
                "domain_category": "public-sector",
            }
        ])

        rels = results[0].relations
        created_by = [r for r in rels if r.relation == "createdBy"]
        self.assertEqual(len(created_by), 1)
        self.assertEqual(created_by[0].source, f"{Config.NAMESPACES['etd']}theme_6")
        self.assertEqual(created_by[0].target, f"{Config.NAMESPACES['etd']}expert_5")
        self.assertEqual(created_by[0].metadata.get("field"), "theme.description")
        self.assertEqual(created_by[0].metadata.get("source_doc_id"), "6")


if __name__ == "__main__":
    unittest.main()
