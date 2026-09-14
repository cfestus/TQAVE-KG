import unittest

from models import Entity, Relation
from utils.predicate_precision import PredicatePrecisionLayer


class PredicatePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.layer = PredicatePrecisionLayer()
        self.entities = [
            Entity(uri="https://alfie-project.eu/etd-hub#expert_1", label="Expert 1", entity_type="Expert", confidence=1.0, metadata={}),
            Entity(uri="https://alfie-project.eu/etd-hub#answer_1", label="Answer 1", entity_type="Answer", confidence=1.0, metadata={}),
            Entity(uri="https://alfie-project.eu/etd-hub#dataset_1", label="COMPAS", entity_type="Dataset", confidence=1.0, metadata={}),
            Entity(uri="https://alfie-project.eu/etd-hub#policy_1", label="EU AI Act", entity_type="PolicyFramework", confidence=1.0, metadata={}),
            Entity(uri="https://alfie-project.eu/etd-hub#system_1", label="Model", entity_type="AISystem", confidence=1.0, metadata={}),
            Entity(uri="https://alfie-project.eu/etd-hub#risk_1", label="Systemic Bias", entity_type="Risk", confidence=1.0, metadata={}),
        ]

    def test_returns_new_relation_instances(self):
        rel = Relation(source=self.entities[0].uri, relation="supports", target=self.entities[3].uri, confidence=0.6, metadata={})
        out = self.layer.refine_relations([rel], self.entities)
        self.assertEqual(len(out), 1)
        self.assertIsNot(out[0], rel)
        self.assertEqual(rel.relation, "supports")

    def test_synonym_normalization(self):
        rels = [
            Relation(source=self.entities[0].uri, relation="criticises", target=self.entities[3].uri, confidence=0.6, metadata={}),
            Relation(source=self.entities[0].uri, relation="accordingTo", target=self.entities[3].uri, confidence=0.6, metadata={}),
            Relation(source=self.entities[0].uri, relation="publishedIn", target=self.entities[3].uri, confidence=0.6, metadata={}),
            Relation(source=self.entities[4].uri, relation="mitigates", target=self.entities[3].uri, confidence=0.6, metadata={}),
            Relation(source=self.entities[2].uri, relation="reflects", target=self.entities[5].uri, confidence=0.6, metadata={}),
            Relation(source=self.entities[4].uri, relation="discriminates against", target=self.entities[5].uri, confidence=0.6, metadata={}),
        ]
        out = self.layer.refine_relations(rels, self.entities)
        self.assertEqual(out[0].relation, "criticizes")
        self.assertEqual(out[1].relation, "relatedTo")
        self.assertEqual(out[2].relation, "relatedTo")
        self.assertEqual(out[3].relation, "relatedTo")
        self.assertEqual(out[4].relation, "reflects")
        self.assertEqual(out[5].relation, "relatedTo")

    def test_whitelist_enforcement(self):
        rel = Relation(source=self.entities[0].uri, relation="randomVerb", target=self.entities[3].uri, confidence=0.7, metadata={})
        out = self.layer.refine_relations([rel], self.entities)
        self.assertEqual(out[0].relation, "relatedTo")

    def test_type_guard_trained_on(self):
        # trainedOn must target dataset/data source/corpus
        rel_bad = Relation(source=self.entities[4].uri, relation="trainedOn", target=self.entities[0].uri, confidence=0.8, metadata={})
        rel_ok = Relation(source=self.entities[4].uri, relation="trainedOn", target=self.entities[2].uri, confidence=0.8, metadata={})
        out = self.layer.refine_relations([rel_bad, rel_ok], self.entities)
        self.assertEqual(out[0].relation, "relatedTo")
        self.assertEqual(out[1].relation, "trainedOn")

    def test_type_guard_compliance_and_requires(self):
        rel1 = Relation(source=self.entities[4].uri, relation="compliesWith", target=self.entities[3].uri, confidence=0.8, metadata={})
        rel2 = Relation(source=self.entities[4].uri, relation="compliesWith", target=self.entities[0].uri, confidence=0.8, metadata={})
        rel3 = Relation(source=self.entities[4].uri, relation="requires", target=self.entities[3].uri, confidence=0.8, metadata={})
        out = self.layer.refine_relations([rel1, rel2, rel3], self.entities)
        self.assertEqual(out[0].relation, "compliesWith")
        self.assertEqual(out[1].relation, "relatedTo")
        self.assertEqual(out[2].relation, "requires")

    def test_evidence_upgrades_five_examples(self):
        generic = [
            Relation(source=self.entities[4].uri, relation="relatedTo", target=self.entities[2].uri, confidence=0.4, metadata={"evidence": "The Model was trained on COMPAS data."}),
            Relation(source=self.entities[4].uri, relation="relatedTo", target=self.entities[3].uri, confidence=0.4, metadata={"evidence": "The Model violates the EU AI Act requirements."}),
            Relation(source=self.entities[4].uri, relation="relatedTo", target=self.entities[3].uri, confidence=0.4, metadata={"evidence": "The Model requires compliance with the EU AI Act."}),
            Relation(source=self.entities[4].uri, relation="relatedTo", target=self.entities[3].uri, confidence=0.4, metadata={"evidence": "The Model mitigation reduces harms through the EU AI Act."}),
            Relation(source=self.entities[4].uri, relation="relatedTo", target=self.entities[3].uri, confidence=0.4, metadata={"evidence": "According to the EU AI Act, the Model is recommended."}),
            Relation(source=self.entities[2].uri, relation="relatedTo", target=self.entities[5].uri, confidence=0.4, metadata={"evidence": "Historical data reflects systemic bias in COMPAS."}),
            Relation(source=self.entities[4].uri, relation="relatedTo", target=self.entities[5].uri, confidence=0.4, metadata={"evidence": "The Model disproportionately affects systemic bias."}),
        ]
        out = self.layer.refine_relations(generic, self.entities)
        preds = [r.relation for r in out]
        self.assertIn("trainedOn", preds)
        self.assertIn("violates", preds)
        self.assertIn("requiresComplianceWith", preds)
        self.assertEqual(preds.count("relatedTo"), 4)

    def test_requires_upgrades_for_non_policy_prerequisite_concepts(self):
        source = Entity(
            uri="https://alfie-project.eu/etd-hub#concept_1",
            label="Algorithmic Decision-Making",
            entity_type="AlgorithmicBias",
            confidence=1.0,
            metadata={},
        )
        target = Entity(
            uri="https://alfie-project.eu/etd-hub#concept_2",
            label="Accountability",
            entity_type="Governance",
            confidence=1.0,
            metadata={},
        )
        rel = Relation(
            source=source.uri,
            relation="relatedTo",
            target=target.uri,
            confidence=0.4,
            metadata={"evidence": "Algorithmic decision-making requires accountability in outcomes."},
        )
        out = self.layer.refine_relations([rel], [source, target])
        self.assertEqual(out[0].relation, "requires")

    def test_generic_fallback_predicates_receive_confidence_penalty(self):
        rels = [
            Relation(source=self.entities[4].uri, relation="relatedTo", target=self.entities[2].uri, confidence=0.6, metadata={}),
            Relation(source=self.entities[4].uri, relation="associatedWith", target=self.entities[2].uri, confidence=0.6, metadata={}),
        ]
        out = self.layer.refine_relations(rels, self.entities)
        self.assertLess(out[0].confidence, 0.6)
        self.assertLess(out[1].confidence, 0.6)

    def test_lexical_only_semantic_relations_receive_small_confidence_penalty(self):
        rel = Relation(
            source=self.entities[4].uri,
            relation="uses",
            target=self.entities[2].uri,
            confidence=0.7,
            metadata={"extractor": "lexical", "llm_semantic_agreement": False},
        )
        out = self.layer.refine_relations([rel], self.entities)
        self.assertLess(out[0].confidence, 0.7)

    def test_backbone_predicates_preserved_with_canonical_uri_pattern(self):
        # Canonical linker remints URIs as .../entity/<type>/<slug>-<hash>.
        # Predicate precision must still keep backbone predicates.
        rels = [
            Relation(
                source="https://alfie-project.eu/etd-hub#entity/answer/answer-abc12345",
                relation="answersQuestion",
                target="https://alfie-project.eu/etd-hub#entity/question/question-def67890",
                confidence=0.98,
                metadata={"extractor": "structural"},
            ),
            Relation(
                source="https://alfie-project.eu/etd-hub#entity/answer/answer-abc12345",
                relation="authoredBy",
                target="https://alfie-project.eu/etd-hub#entity/expert/expert-fedcba09",
                confidence=0.96,
                metadata={"extractor": "structural"},
            ),
            Relation(
                source="https://alfie-project.eu/etd-hub#entity/question/question-def67890",
                relation="hasTheme",
                target="https://alfie-project.eu/etd-hub#entity/theme/theme-11223344",
                confidence=0.98,
                metadata={"extractor": "structural"},
            ),
        ]
        # Intentionally pass empty entity list to exercise URI-based type inference.
        out = self.layer.refine_relations(rels, [])
        self.assertEqual(out[0].relation, "answersQuestion")
        self.assertEqual(out[1].relation, "authoredBy")
        self.assertEqual(out[2].relation, "belongsToTheme")

    def test_question_theme_expert_backbone_predicates_preserved(self):
        rels = [
            Relation(
                source="https://alfie-project.eu/etd-hub#entity/question/question-14-abc12345",
                relation="askedBy",
                target="https://alfie-project.eu/etd-hub#entity/person/expert-5-def67890",
                confidence=0.96,
                metadata={"extractor": "structural"},
            ),
            Relation(
                source="https://alfie-project.eu/etd-hub#entity/theme/theme-6-98765432",
                relation="createdBy",
                target="https://alfie-project.eu/etd-hub#entity/person/expert-5-def67890",
                confidence=0.96,
                metadata={"extractor": "structural"},
            ),
        ]
        out = self.layer.refine_relations(rels, [])
        self.assertEqual(out[0].relation, "askedBy")
        self.assertEqual(out[1].relation, "createdBy")


if __name__ == "__main__":
    unittest.main()

