import unittest

from config import Config
from models import Entity, Relation
from utils.semantic_relation_policy import SemanticRelationPolicy


class SemanticRelationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = SemanticRelationPolicy()

    def _entity(self, local_id, label, entity_type, **metadata):
        return Entity(
            uri=f"{Config.NAMESPACES['etd']}{local_id}",
            label=label,
            entity_type=entity_type,
            confidence=1.0,
            metadata=metadata,
        )

    def _relation(self, source, predicate, target, evidence, field="answer.description"):
        rel = Relation(
            source=source.uri,
            relation=predicate,
            target=target.uri,
            confidence=0.9,
            metadata={
                "field": field,
                "extractor": "llm",
                "context_source": "llm_relation",
                "section_category": "semantic",
                "evidence_quote": evidence,
                "evidence": evidence,
            },
        )
        rel.context = evidence
        return rel

    def test_best_predicate_match_prefers_specific_registry_predicate(self):
        pred, phrase, source = self.policy.best_predicate_match(
            "The Model requires compliance with the EU AI Act."
        )
        self.assertEqual(pred, "requiresComplianceWith")
        self.assertEqual(phrase, "requires compliance with")
        self.assertEqual(source, "explicit_lexical_relation")

    def test_validate_rejects_dataset_actor_when_sentence_subject_is_system(self):
        dataset = self._entity(
            "compas_dataset",
            "COMPAS Recidivism Dataset",
            "Dataset",
            surface_form="COMPAS",
            section_category="semantic",
        )
        risk = self._entity(
            "racial_disparities",
            "Racial Disparities",
            "Racialbiasrisk",
            section_category="semantic",
        )
        text = "A predictive system like COMPAS exhibits systemic racial disparities."
        rel = self._relation(dataset, "exhibitsRisk", risk, text)

        decision = self.policy.validate(
            rel,
            entities_by_uri={dataset.uri: dataset, risk.uri: risk},
            discourse_text=text,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason_code, "entity_role:dataset_actor_mismatch_system_cue")

    def test_validate_rejects_dataset_when_analysis_is_true_actor(self):
        dataset = self._entity(
            "hmda_dataset",
            "HMDA Dataset",
            "Dataset",
            surface_form="HMDA dataset",
            section_category="semantic",
        )
        harm = self._entity(
            "systematic_discrimination",
            "Systematic Discrimination",
            "Biasrisk",
            section_category="semantic",
        )
        text = "Bartlett et al. analysis of HMDA dataset revealed systematic discrimination."
        rel = self._relation(dataset, "reflects", harm, text)

        decision = self.policy.validate(
            rel,
            entities_by_uri={dataset.uri: dataset, harm.uri: harm},
            discourse_text=text,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason_code, "entity_role:analysis_actor_missing")

    def test_validate_preserves_question_and_hypothetical_modality(self):
        model = self._entity("hiring_model", "Hiring Model", "AISystem", section_category="semantic")
        dataset = self._entity("applicant_dataset", "Applicant Dataset", "Dataset", section_category="semantic")
        text = "Could a Hiring Model be trained on the Applicant Dataset?"
        rel = self._relation(model, "trainedOn", dataset, text, field="question.body")

        decision = self.policy.validate(
            rel,
            entities_by_uri={model.uri: model, dataset.uri: dataset},
            discourse_text=text,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.discourse_mode, "interrogative")
        self.assertIn("questioned", decision.modality_tags)
        self.assertIn("hypothetical", decision.modality_tags)

    def test_validate_rejects_unjustified_generic_related_to(self):
        fairness = self._entity("fairness", "Fairness", "Fairness", section_category="semantic")
        accountability = self._entity("accountability", "Accountability", "Governance", section_category="semantic")
        text = "Fairness and accountability are recurring themes in the discussion."
        rel = self._relation(fairness, "relatedTo", accountability, text)

        decision = self.policy.validate(
            rel,
            entities_by_uri={fairness.uri: fairness, accountability.uri: accountability},
            discourse_text=text,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason_code, "predicate_ambiguity:generic_relation_not_justified")

    def test_reflects_direct_object_is_locally_entailed(self):
        self.assertTrue(
            self.policy.predicate_locally_entailed(
                "reflects",
                ["historical data"],
                ["systemic bias"],
                "Historical data reflects systemic bias.",
            )
        )

    def test_reflects_example_tail_does_not_inherit_predicate(self):
        evidence = (
            "Historical data reflects systemic bias, from over-policing in Black communities "
            "to harsher sentencing for similar offenses."
        )
        self.assertFalse(
            self.policy.predicate_locally_entailed(
                "reflects",
                ["historical data"],
                ["over-policing"],
                evidence,
            )
        )
        self.assertFalse(
            self.policy.predicate_locally_entailed(
                "reflects",
                ["historical data"],
                ["harsher sentencing"],
                evidence,
            )
        )
        self.assertTrue(
            self.policy.predicate_locally_entailed(
                "reflects",
                ["historical data"],
                ["systemic bias"],
                evidence,
            )
        )

    def test_show_bias_against_population_is_entailed_as_particularly_affects(self):
        # "show(s)/exhibit(s)/demonstrate(s) ... against" expresses disparate impact
        # the same way "discriminates against" does, but previously had no matching
        # trigger and only satisfied the generic exhibitsRisk cue instead.
        self.assertTrue(
            self.policy.predicate_locally_entailed(
                "particularlyAffects",
                ["automated essay scoring systems"],
                ["students from marginalized groups"],
                "Automated essay scoring systems show systematic bias against students from marginalized groups.",
            )
        )

    def test_show_without_against_is_not_entailed_as_particularly_affects(self):
        # Negative control: "show <noun phrase>" alone (no "against") must not be
        # rescued into particularlyAffects -- this is the genuine exhibitsRisk case
        # (comparing system performance across groups, not a claim of impact on them).
        self.assertFalse(
            self.policy.predicate_locally_entailed(
                "particularlyAffects",
                ["diagnostic algorithms"],
                ["racial and gender groups"],
                "Diagnostic algorithms show varying performance across racial and gender groups.",
            )
        )

    def test_quantitative_disparity_comparison_is_entailed_as_particularly_affects(self):
        # A stated rate/multiplier followed by an explicit comparison marker
        # ("19% ... vs. 12%", "2.6 times ... than") is a domain-agnostic textual
        # signature of disparate/differential impact, independent of any specific
        # bias-vocabulary verb.
        self.assertTrue(
            self.policy.predicate_locally_entailed(
                "particularlyAffects",
                ["gpt-4o"],
                ["black students"],
                "GPT-4o incorrectly predicted academic failure for Black students 19% of the time vs. 12% for white students.",
            )
        )

    def test_unrelated_quantitative_comparison_is_not_entailed_as_particularly_affects(self):
        # Negative control: a quantitative comparison elsewhere in the sentence,
        # not following the target's mention, must not be treated as evidence of
        # a disparate-impact relation between an unrelated pair.
        self.assertFalse(
            self.policy.predicate_locally_entailed(
                "particularlyAffects",
                ["system x"],
                ["community group"],
                "Revenue grew 19% compared to last year. System X was later deployed near Community Group.",
            )
        )

    def test_such_as_tail_does_not_blindly_propagate_exhibits_risk(self):
        self.assertFalse(
            self.policy.predicate_locally_entailed(
                "exhibitsRisk",
                ["the system"],
                ["bias"],
                "The system exhibits risks such as bias and privacy loss.",
            )
        )
        self.assertFalse(
            self.policy.predicate_locally_entailed(
                "exhibitsRisk",
                ["the system"],
                ["privacy loss"],
                "The system exhibits risks such as bias and privacy loss.",
            )
        )

    def test_requires_coordination_remains_supported(self):
        evidence = "Fairness requires accountability, interpretability, and equity."
        for target in ("accountability", "interpretability", "equity"):
            self.assertTrue(
                self.policy.predicate_locally_entailed(
                    "requires",
                    ["fairness"],
                    [target],
                    evidence,
                )
            )

    def test_repeated_predicate_allows_multiple_reflects_relations(self):
        evidence = "Historical data reflects systemic bias and reflects harsher sentencing patterns."
        self.assertTrue(
            self.policy.predicate_locally_entailed(
                "reflects",
                ["historical data"],
                ["systemic bias"],
                evidence,
            )
        )
        self.assertTrue(
            self.policy.predicate_locally_entailed(
                "reflects",
                ["historical data"],
                ["harsher sentencing patterns"],
                evidence,
            )
        )


if __name__ == "__main__":
    unittest.main()
