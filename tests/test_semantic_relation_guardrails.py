import types
import unittest

from config import Config
from models import AnnotationBatch, Entity, ProcessingResult, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from reasoning.reasoner import MultiHopReasoner
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


class _NoopCanonicalLinker:
    @staticmethod
    def rewrite_relation_endpoints(_relations):
        return None


class _RecordingRelationExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, text, entities, discourse_id=None):
        self.calls.append(
            {
                "text": text,
                "labels": [getattr(entity, "label", "") for entity in entities],
                "discourse_id": discourse_id,
            }
        )
        return []


def _make_pipeline():
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.canonical_linker = _NoopCanonicalLinker()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.reasoner = MultiHopReasoner(min_confidence=0.5, max_iterations=2, debug=False)
    pipeline._discourse_chunk_spans = {}
    return pipeline


def _register_entity(
    pipeline,
    *,
    local_id,
    label,
    entity_type,
    discourse_id="d1",
    section="Ethical Challenges",
    section_category="semantic",
    mentions=None,
):
    entity = Entity(
        uri=f"{Config.NAMESPACES['etd']}{local_id}",
        label=label,
        entity_type=entity_type,
        confidence=1.0,
        metadata={
            "discourse_id": discourse_id,
            "section": section,
            "section_category": section_category,
            "mentions": mentions or [],
        },
    )
    pipeline._ensure_entity_ontology_grounding(entity)
    return pipeline.global_registry.get_or_create(entity)


def _make_relation(source, predicate, target, **metadata):
    rel = Relation(
        source=source,
        relation=predicate,
        target=target,
        confidence=float(metadata.pop("confidence", 0.85)),
        is_inferred=bool(metadata.pop("is_inferred", False)),
        metadata=metadata,
    )
    return rel


class SemanticRelationGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def _validate(self, relation, entities, discourse_text):
        entity_map = {entity.uri: entity for entity in entities}
        return self.pipeline._validate_semantic_relation_candidate(
            relation,
            entities_by_uri=entity_map,
            discourse_text=discourse_text,
        )

    def _assert_validation_code(self, relation, expected_code):
        self.assertEqual(
            relation.metadata.get("semantic_validation_code"),
            expected_code,
            getattr(relation, "relation", ""),
        )
        self.assertTrue(str(relation.metadata.get("semantic_validation_reason") or "").strip())

    def _assert_validation_code_in(self, relation, expected_codes):
        self.assertIn(
            relation.metadata.get("semantic_validation_code"),
            set(expected_codes),
            getattr(relation, "relation", ""),
        )
        self.assertTrue(str(relation.metadata.get("semantic_validation_reason") or "").strip())

    def test_known_bad_relations_are_rejected_by_general_rules(self):
        discourse = "The local evidence names both endpoints, but the semantic pairing is invalid."

        dynamic_pricing = _register_entity(
            self.pipeline,
            local_id="dynamic_pricing",
            label="Dynamic Pricing Algorithms",
            entity_type="AISystem",
        )
        dashboard = _register_entity(
            self.pipeline,
            local_id="dashboard",
            label="Dashboard",
            entity_type="Dataset",
            section="Dashboard",
            section_category="resource",
        )
        street_bump = _register_entity(
            self.pipeline,
            local_id="street_bump",
            label="Boston Street Bump Dataset",
            entity_type="Dataset",
        )
        nyc311 = _register_entity(
            self.pipeline,
            local_id="nyc311",
            label="NYC 311",
            entity_type="Dataset",
            section="Public Datasets",
            section_category="resource",
        )
        fitzpatrick = _register_entity(
            self.pipeline,
            local_id="fitzpatrick",
            label="Fitzpatrick Skin Type",
            entity_type="Classification",
        )
        civil_rights = _register_entity(
            self.pipeline,
            local_id="civil_rights",
            label="Civil Rights Impact",
            entity_type="Impact",
        )
        joy = _register_entity(
            self.pipeline,
            local_id="joy",
            label="Joy Buolamwini",
            entity_type="Person",
        )
        mit_media_lab = _register_entity(
            self.pipeline,
            local_id="mit_media_lab",
            label="MIT Media Lab",
            entity_type="Organisation",
        )
        discourse_unit = _register_entity(
            self.pipeline,
            local_id="discourse_9",
            label="DiscourseUnit 9",
            entity_type="DiscourseUnit",
        )
        group = _register_entity(
            self.pipeline,
            local_id="african_american_women",
            label="African American women",
            entity_type="Concept",
        )

        bad_cases = [
            (
                _make_relation(
                    dynamic_pricing.uri,
                    "hasRisk",
                    dashboard.uri,
                    field="theme.description",
                    extractor="llm",
                    context_source="llm_relation",
                    section_category="semantic",
                    evidence_quote="Dynamic Pricing Algorithms and Dashboard are both listed here.",
                ),
                [dynamic_pricing, dashboard],
            ),
            (
                _make_relation(
                    street_bump.uri,
                    "hasRisk",
                    nyc311.uri,
                    field="theme.description",
                    extractor="llm",
                    context_source="llm_relation",
                    section_category="semantic",
                    evidence_quote="Boston Street Bump Dataset and NYC 311 appear in the same list.",
                ),
                [street_bump, nyc311],
            ),
            (
                _make_relation(
                    fitzpatrick.uri,
                    "hasRisk",
                    civil_rights.uri,
                    field="theme.description",
                    extractor="llm",
                    context_source="llm_relation",
                    section_category="semantic",
                    evidence_quote="Fitzpatrick Skin Type is mentioned alongside Civil Rights Impact.",
                ),
                [fitzpatrick, civil_rights],
            ),
            (
                _make_relation(
                    joy.uri,
                    "hasRisk",
                    mit_media_lab.uri,
                    field="answer.description",
                    extractor="llm",
                    context_source="llm_relation",
                    section_category="semantic",
                    evidence_quote="Joy Buolamwini and MIT Media Lab co-occur in the same answer.",
                ),
                [joy, mit_media_lab],
            ),
            (
                _make_relation(
                    discourse_unit.uri,
                    "causes",
                    group.uri,
                    field="theme.description",
                    extractor="llm",
                    context_source="llm_relation",
                    section_category="semantic",
                    evidence_quote="DiscourseUnit 9 mentions African American women.",
                ),
                [discourse_unit, group],
            ),
        ]

        for relation, entities in bad_cases:
            accepted, reason = self._validate(relation, entities, discourse)
            self.assertFalse(accepted, relation.relation)
            self._assert_validation_code_in(
                relation,
                {
                    "domain_range:predicate_domain_range_invalid",
                    "entailment:predicate_not_locally_entailed",
                    "locality:no_shared_local_proposition",
                },
            )

    def test_vote_field_resource_section_and_positive_local_claims(self):
        system = _register_entity(
            self.pipeline,
            local_id="hiring_system",
            label="Automated Hiring System",
            entity_type="AISystem",
        )
        bias_risk = _register_entity(
            self.pipeline,
            local_id="gender_bias",
            label="gender bias risk",
            entity_type="Risk",
        )
        dataset = _register_entity(
            self.pipeline,
            local_id="public_dataset",
            label="Public Benchmark Dataset",
            entity_type="Dataset",
            section="Public Datasets",
            section_category="resource",
        )

        valid_relation = _make_relation(
            system.uri,
            "hasRisk",
            bias_risk.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote="The Automated Hiring System has gender bias risk in screening decisions.",
        )
        accepted, reason = self._validate(
            valid_relation,
            [system, bias_risk],
            "The Automated Hiring System has gender bias risk in screening decisions.",
        )
        self.assertTrue(accepted)
        self._assert_validation_code(valid_relation, "accepted")
        self.assertIn("predicate", reason)

        vote_relation = _make_relation(
            system.uri,
            "endorses",
            bias_risk.uri,
            field="vote.vote_value",
            extractor="vote",
            context_source="dataset_vote",
            section_category="semantic",
            evidence_quote="The Automated Hiring System has gender bias risk in screening decisions.",
        )
        accepted, reason = self._validate(vote_relation, [system, bias_risk], valid_relation.metadata["evidence_quote"])
        self.assertFalse(accepted)
        self._assert_validation_code(vote_relation, "resource_filter:field_not_allowlisted")

        resource_relation = _make_relation(
            system.uri,
            "hasRisk",
            dataset.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="resource",
            evidence_quote="The Automated Hiring System and Public Benchmark Dataset are both listed.",
        )
        accepted, reason = self._validate(resource_relation, [system, dataset], resource_relation.metadata["evidence_quote"])
        self.assertFalse(accepted)
        self._assert_validation_code(resource_relation, "resource_filter:non_semantic_section")

    def test_local_entailment_guard_rejects_directionless_uses_claims(self):
        health_risk_algorithms = _register_entity(
            self.pipeline,
            local_id="health_risk_algorithms",
            label="Health Risk Algorithms",
            entity_type="AISystem",
        )
        racial_bias = _register_entity(
            self.pipeline,
            local_id="racial_bias",
            label="Racial Bias",
            entity_type="Risk",
        )
        model_cards = _register_entity(
            self.pipeline,
            local_id="model_cards",
            label="Model Cards",
            entity_type="Model",
        )
        datasheets = _register_entity(
            self.pipeline,
            local_id="datasheets",
            label="Datasheets/Spec Sheets",
            entity_type="Dataset",
        )
        auditing_framework = _register_entity(
            self.pipeline,
            local_id="auditing_framework",
            label="Auditing Framework",
            entity_type="Concept",
        )

        bad_cases = [
            (
                _make_relation(
                    health_risk_algorithms.uri,
                    "uses",
                    racial_bias.uri,
                    field="theme.description",
                    extractor="llm",
                    context_source="llm_relation",
                    section_category="semantic",
                    evidence_quote="racial bias in commercial health risk algorithms",
                ),
                [health_risk_algorithms, racial_bias],
            ),
            (
                _make_relation(
                    model_cards.uri,
                    "uses",
                    datasheets.uri,
                    field="answer.description",
                    extractor="llm",
                    context_source="llm_relation",
                    section_category="semantic",
                    evidence_quote="You can use Model Cards for model reporting and Datasheets/Spec Sheets for datasets",
                ),
                [model_cards, datasheets],
            ),
        ]

        for relation, entities in bad_cases:
            accepted, reason = self._validate(relation, entities, relation.metadata["evidence_quote"])
            self.assertFalse(accepted)
            self._assert_validation_code_in(
                relation,
                {
                    "entailment:predicate_not_locally_entailed",
                    "domain_range:predicate_domain_range_invalid",
                    "locality:no_shared_local_proposition",
                },
            )

        good_relation = _make_relation(
            auditing_framework.uri,
            "uses",
            model_cards.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote="The Auditing Framework uses Model Cards to document model behavior.",
        )
        accepted, reason = self._validate(
            good_relation,
            [auditing_framework, model_cards],
            good_relation.metadata["evidence_quote"],
        )
        self.assertTrue(accepted)
        self._assert_validation_code(good_relation, "accepted")
        self.assertIn("predicate", reason)

    def test_used_in_sentence_does_not_validate_active_uses_or_cross_cue_object(self):
        system = _register_entity(
            self.pipeline,
            local_id="predictive_system_used_in_test",
            label="Predictive System",
            entity_type="AISystem",
        )
        decisions = _register_entity(
            self.pipeline,
            local_id="life_altering_decisions_used_in_test",
            label="Life-Altering Decisions",
            entity_type="Impact",
        )
        disparities = _register_entity(
            self.pipeline,
            local_id="systemic_racial_disparities_used_in_test",
            label="Systemic Racial Disparities",
            entity_type="AlgorithmicBias",
        )
        dataset = _register_entity(
            self.pipeline,
            local_id="real_world_data_used_in_test",
            label="Real-World Data",
            entity_type="Dataset",
        )
        discourse = (
            "Predictive System used in Life-Altering Decisions may exhibit Systemic Racial Disparities "
            "even when trained on Real-World Data."
        )

        bad_decisions = _make_relation(
            system.uri,
            "uses",
            decisions.uri,
            field="question.body",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        bad_decisions.context = discourse
        accepted, _reason = self._validate(bad_decisions, [system, decisions], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code_in(
            bad_decisions,
            {
                "entailment:predicate_not_locally_entailed",
                "domain_range:predicate_domain_range_invalid",
            },
        )

        bad_dataset = _make_relation(
            system.uri,
            "uses",
            dataset.uri,
            field="question.body",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        bad_dataset.context = discourse
        accepted, _reason = self._validate(bad_dataset, [system, dataset], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(bad_dataset, "entailment:predicate_not_locally_entailed")

        good_dataset = _make_relation(
            system.uri,
            "trainedOn",
            dataset.uri,
            field="question.body",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        good_dataset.context = discourse
        accepted, _reason = self._validate(good_dataset, [system, dataset], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(good_dataset, "accepted")

    def test_affecting_relation_does_not_cross_into_used_to_subordinate_clause(self):
        system = _register_entity(
            self.pipeline,
            local_id="commercial_health_risk_algorithms_affecting_test",
            label="Commercial Health Risk Algorithms",
            entity_type="AISystem",
        )
        programs = _register_entity(
            self.pipeline,
            local_id="additional_care_programs_affecting_test",
            label="Additional Care Programs",
            entity_type="Concept",
        )
        discourse = (
            "Commercial Health Risk Algorithms affecting 200 million Americans annually, "
            "used to identify patients needing Additional Care Programs."
        )
        relation = _make_relation(
            system.uri,
            "affects",
            programs.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse
        accepted, _reason = self._validate(relation, [system, programs], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(relation, "entailment:predicate_not_locally_entailed")

    def test_affect_relation_keeps_matrix_object_and_rejects_by_gerund_object(self):
        system = _register_entity(
            self.pipeline,
            local_id="ai_system_matrix_subject",
            label="AI system",
            entity_type="AISystem",
        )
        autonomy = _register_entity(
            self.pipeline,
            local_id="human_autonomy_matrix_object",
            label="Human Autonomy",
            entity_type="Concept",
        )
        end_users = _register_entity(
            self.pipeline,
            local_id="end_users_subordinate_object",
            label="End-Users",
            entity_type="Population/Group",
        )
        discourse = "Could the AI system affect human autonomy by generating over-reliance by end-users?"

        good_relation = _make_relation(
            system.uri,
            "affects",
            autonomy.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        good_relation.context = discourse
        accepted, _reason = self._validate(good_relation, [system, autonomy], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(good_relation, "accepted")

        bad_relation = _make_relation(
            system.uri,
            "affects",
            end_users.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        bad_relation.context = discourse
        accepted, _reason = self._validate(bad_relation, [system, end_users], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(bad_relation, "entailment:predicate_not_locally_entailed")

    def test_reporting_heading_is_not_treated_as_semantic_subject(self):
        heading = _register_entity(
            self.pipeline,
            local_id="economic_impact_heading",
            label="Economic Impact",
            entity_type="Impact",
        )
        payments = _register_entity(
            self.pipeline,
            local_id="excessive_interest_payments",
            label="$765M+ annually in excessive interest payments",
            entity_type="EconomicImpact",
        )
        discourse = "Economic Impact: Results in $765M+ annually in excessive interest payments"
        relation = _make_relation(
            heading.uri,
            "causes",
            payments.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse
        accepted, _reason = self._validate(relation, [heading, payments], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(relation, "entity_role:reporting_heading_not_semantic_subject")

    def test_valid_elliptical_semantic_heading_is_retained(self):
        bias = _register_entity(
            self.pipeline,
            local_id="intersectional_bias_heading",
            label="Intersectional Bias",
            entity_type="Bias",
        )
        group = _register_entity(
            self.pipeline,
            local_id="african_american_women_heading_target",
            label="African American women",
            entity_type="Population",
        )
        discourse = "Intersectional Bias: Particularly affects African American women"
        relation = _make_relation(
            bias.uri,
            "particularlyAffects",
            group.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse
        accepted, _reason = self._validate(relation, [bias, group], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(relation, "accepted")

    def test_coordinated_object_relation_requires_split(self):
        system = _register_entity(
            self.pipeline,
            local_id="insurance_system_coordinated_object",
            label="AI-powered insurance underwriting systems",
            entity_type="AISystem",
        )
        combined = _register_entity(
            self.pipeline,
            local_id="minority_and_geographic_combined",
            label="Minority Communities And Certain Geographic Areas",
            entity_type="DemographicGroup",
        )
        discourse = (
            "AI-powered insurance underwriting systems show bias in risk assessment and pricing, "
            "particularly affecting minority communities and certain geographic areas."
        )
        relation = _make_relation(
            system.uri,
            "particularlyAffects",
            combined.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse
        accepted, _reason = self._validate(relation, [system, combined], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(relation, "entity_role:coordinated_object_requires_split")

    def test_passive_uses_clause_does_not_create_active_framework_relation(self):
        framework = _register_entity(
            self.pipeline,
            local_id="technical_auditing_framework",
            label="Technical Auditing Framework",
            entity_type="Model",
        )
        fairness = _register_entity(
            self.pipeline,
            local_id="framework_fairness",
            label="Fairness",
            entity_type="Fairness",
        )
        discourse = (
            "Then you can build a simple technical auditing framework by answering the following questions: "
            "what statistical methods will be used to test data/model fairness/quality?"
        )
        relation = _make_relation(
            framework.uri,
            "uses",
            fairness.uri,
            field="answer.description",
            extractor="lexical",
            context_source="lexical_sentence",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [framework, fairness], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(relation, "entailment:predicate_not_locally_entailed")

    def test_pronoun_led_requires_rejects_embedded_phrase_subjects(self):
        fairness = _register_entity(
            self.pipeline,
            local_id="fairness_pronoun_subject",
            label="Fairness",
            entity_type="Fairness",
        )
        decision_making = _register_entity(
            self.pipeline,
            local_id="decision_making_pronoun_subject",
            label="Algorithmic Decision-Making",
            entity_type="AISystem",
        )
        accountability = _register_entity(
            self.pipeline,
            local_id="accountability_pronoun_target",
            label="Accountability",
            entity_type="Governance",
        )
        discourse = (
            "Fairness in algorithmic decision-making, especially in high-stakes domains like criminal justice, "
            "goes beyond technical accuracy; it requires accountability."
        )

        fair_relation = _make_relation(
            fairness.uri,
            "requires",
            accountability.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        fair_relation.context = discourse
        accepted, reason = self._validate(fair_relation, [fairness, accountability], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(fair_relation, "accepted")

        embedded_relation = _make_relation(
            decision_making.uri,
            "requires",
            accountability.uri,
            field="answer.description",
            extractor="lexical",
            context_source="lexical_sentence",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        embedded_relation.context = discourse
        accepted, reason = self._validate(embedded_relation, [decision_making, accountability], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(embedded_relation, "entailment:predicate_not_locally_entailed")

        technical_accuracy = _register_entity(
            self.pipeline,
            local_id="technical_accuracy_pronoun_subject",
            label="Technical Accuracy",
            entity_type="Concept",
        )
        complement_relation = _make_relation(
            technical_accuracy.uri,
            "requires",
            accountability.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        complement_relation.context = discourse
        accepted, reason = self._validate(complement_relation, [technical_accuracy, accountability], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(complement_relation, "entailment:predicate_not_locally_entailed")

    def test_structural_semantic_predicate_is_blocked_before_materialization(self):
        answer = _register_entity(
            self.pipeline,
            local_id="answer_1",
            label="Answer 1",
            entity_type="Answer",
            discourse_id="501",
        )
        vote = _register_entity(
            self.pipeline,
            local_id="vote_1",
            label="Vote 1",
            entity_type="Vote",
            discourse_id="501",
        )
        rel = _make_relation(
            answer.uri,
            "relatedTo",
            vote.uri,
            field="vote.vote_value",
            extractor="structural",
            context_source="dataset_vote",
            section_category="semantic",
            relation_kind="structural",
            evidence_quote="Structural link: answer_1 receivesVote vote_1.",
        )
        annotations = AnnotationBatch(
            votes={
                "501": ProcessingResult(
                    entities=[answer, vote],
                    relations=[rel],
                    metadata={"text": "Structural link: answer_1 receivesVote vote_1."},
                )
            }
        )

        summary = self.pipeline.validate_semantic_relations(annotations)
        self.assertEqual(summary["dropped"], 1)
        self.assertEqual(annotations.votes["501"].relations, [])

    def test_process_relations_batch_includes_resource_units_in_semantic_extraction(self):
        text = (
            "**Ethical Challenges**: Dynamic Pricing Algorithms can cause discrimination risk for riders.\n"
            "**Dashboard**: Public Dashboard tracks service tickets and uploads."
        )
        challenge_start = text.index("Dynamic Pricing Algorithms")
        challenge_end = challenge_start + len("Dynamic Pricing Algorithms")
        risk_start = text.index("discrimination risk")
        risk_end = risk_start + len("discrimination risk")
        dashboard_start = text.index("Public Dashboard")
        dashboard_end = dashboard_start + len("Public Dashboard")
        tickets_start = text.index("service tickets")
        tickets_end = tickets_start + len("service tickets")

        extractor = _RecordingRelationExtractor()
        self.pipeline.llm_relation_extractor = extractor

        entities = [
            _register_entity(
                self.pipeline,
                local_id="pricing_model",
                label="Dynamic Pricing Algorithms",
                entity_type="AISystem",
                mentions=[{"discourse_id": "9", "start_char": challenge_start, "end_char": challenge_end}],
            ),
            _register_entity(
                self.pipeline,
                local_id="discrimination_risk",
                label="discrimination risk",
                entity_type="Risk",
                mentions=[{"discourse_id": "9", "start_char": risk_start, "end_char": risk_end}],
            ),
            _register_entity(
                self.pipeline,
                local_id="public_dashboard",
                label="Public Dashboard",
                entity_type="Dataset",
                section="Dashboard",
                section_category="resource",
                mentions=[{"discourse_id": "9", "start_char": dashboard_start, "end_char": dashboard_end}],
            ),
            _register_entity(
                self.pipeline,
                local_id="service_tickets",
                label="service tickets",
                entity_type="Dataset",
                section="Dashboard",
                section_category="resource",
                mentions=[{"discourse_id": "9", "start_char": tickets_start, "end_char": tickets_end}],
            ),
        ]

        relations = self.pipeline._process_relations_batch(
            [text],
            [entities],
            ["9"],
            provenance_list=[{"field": "theme.description", "origin_key": "theme.description|9"}],
        )

        self.assertEqual(len(relations), 1)
        self.assertEqual(len(extractor.calls), 2)
        combined = "\n".join(call["text"] for call in extractor.calls)
        self.assertIn("Dynamic Pricing Algorithms", combined)
        self.assertIn("Public Dashboard", combined)

    def test_predicate_precision_does_not_upgrade_mentions_or_cooccurrence(self):
        layer = PredicatePrecisionLayer()
        system = Entity(
            uri=f"{Config.NAMESPACES['etd']}system_1",
            label="Loan Scoring Model",
            entity_type="AISystem",
            confidence=1.0,
            metadata={},
        )
        dashboard = Entity(
            uri=f"{Config.NAMESPACES['etd']}dashboard_1",
            label="Operations Dashboard",
            entity_type="Dataset",
            confidence=1.0,
            metadata={},
        )
        risk = Entity(
            uri=f"{Config.NAMESPACES['etd']}risk_1",
            label="bias risk",
            entity_type="Risk",
            confidence=1.0,
            metadata={},
        )

        relations = [
            Relation(
                source=system.uri,
                relation="mentions",
                target=risk.uri,
                confidence=0.4,
                metadata={
                    "context_source": "lightweight_mentions",
                    "evidence": "Loan Scoring Model mentions bias risk in a list.",
                },
            ),
            Relation(
                source=system.uri,
                relation="relatedTo",
                target=dashboard.uri,
                confidence=0.4,
                metadata={
                    "context_source": "cooccurrence_chunk",
                    "evidence": "Loan Scoring Model appears near Operations Dashboard while bias risk is discussed elsewhere.",
                },
            ),
        ]

        refined = layer.refine_relations(relations, [system, dashboard, risk])
        self.assertEqual(refined[0].relation, "mentions")
        self.assertEqual(refined[1].relation, "relatedTo")

    def test_apply_reasoning_rejects_inferred_claims_without_local_grounding(self):
        system = _register_entity(
            self.pipeline,
            local_id="screening_model",
            label="Screening Model",
            entity_type="AISystem",
        )
        risk = _register_entity(
            self.pipeline,
            local_id="racial_bias",
            label="racial bias risk",
            entity_type="Risk",
        )

        base_relation = _make_relation(
            system.uri,
            "relatedTo",
            risk.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote="The Screening Model has racial bias risk in downstream decisions.",
        )
        annotations = AnnotationBatch(
            answers={
                "301": ProcessingResult(
                    entities=[system, risk],
                    relations=[base_relation],
                    metadata={"text": "The Screening Model has racial bias risk in downstream decisions."},
                )
            }
        )

        inferred_relation = Relation(
            source=system.uri,
            relation="hasRisk",
            target=risk.uri,
            confidence=0.8,
            is_inferred=True,
            metadata={
                "inference_rule": "schema_bridge:v1",
                "inferenceRule": "schema_bridge:v1",
            },
        )
        self.pipeline.reasoner = types.SimpleNamespace(infer_relations=lambda _rels: [inferred_relation])

        inferred = self.pipeline.apply_reasoning(annotations)
        self.assertEqual(inferred, [])

    def test_positive_sentence_local_relations_survive_validation(self):
        cases = [
            (
                "Historical Data",
                "Dataset",
                "reflects",
                "Systemic Bias",
                "Risk",
                "Historical Data from the criminal justice system reflects decades of systemic bias.",
            ),
            (
                "Historical Data",
                "Dataset",
                "reflects",
                "Systemic Bias",
                "Risk",
                "Historical Data from the criminal justice system reflects decades of systemic bias, from over-policing in Black communities to harsher sentencing for similar offenses.",
            ),
            (
                "Dynamic Pricing Algorithms",
                "AISystem",
                "particularlyAffects",
                "low-income neighborhoods",
                "Concept",
                "Dynamic Pricing Algorithms systematically discriminate against low-income neighborhoods.",
            ),
            (
                "Facial Recognition Systems",
                "AISystem",
                "exhibitsRisk",
                "Intersectional Bias",
                "Risk",
                "Facial Recognition Systems exhibit intersectional bias in error rates.",
            ),
            (
                "AI Recruitment Systems",
                "AISystem",
                "hasRisk",
                "Gender Bias",
                "Risk",
                "AI Recruitment Systems have gender bias in screening decisions.",
            ),
            (
                "Algorithms",
                "AISystem",
                "trainedOn",
                "Historical Workforce Data",
                "Dataset",
                "Algorithms were trained on historical workforce data collected from prior hiring rounds.",
            ),
            (
                "Healthcare Risk Algorithms",
                "AISystem",
                "uses",
                "Healthcare Spending",
                "Concept",
                "Healthcare Risk Algorithms used healthcare spending as a proxy in allocation decisions.",
            ),
            (
                "Smart City Algorithms",
                "AISystem",
                "affects",
                "Underserved Communities",
                "Concept",
                "Smart City Algorithms affect underserved communities through unequal service allocation.",
            ),
        ]

        for idx, (src_label, src_type, pred, tgt_label, tgt_type, discourse) in enumerate(cases, start=1):
            src = _register_entity(
                self.pipeline,
                local_id=f"positive_src_{idx}",
                label=src_label,
                entity_type=src_type,
            )
            tgt = _register_entity(
                self.pipeline,
                local_id=f"positive_tgt_{idx}",
                label=tgt_label,
                entity_type=tgt_type,
            )
            relation = _make_relation(
                src.uri,
                pred,
                tgt.uri,
                field="answer.description",
                extractor="llm",
                context_source="llm_relation",
                section_category="semantic",
                section="Answer",
                evidence_quote=discourse.split(".")[0].strip() + ".",
                evidence=discourse.split(".")[0].strip() + ".",
            )
            relation.context = discourse
            accepted, reason = self._validate(relation, [src, tgt], discourse)
            self.assertTrue(accepted, f"{pred}: {discourse}")
            self._assert_validation_code(relation, "accepted")
            self.assertIn("predicate", reason)

    def test_reflects_example_tail_targets_are_rejected(self):
        discourse = (
            "Historical data from the criminal justice system reflects decades of systemic bias, "
            "from over-policing in Black communities to harsher sentencing for similar offenses."
        )
        historical_data = _register_entity(
            self.pipeline,
            local_id="historical_data_scope",
            label="Historical Data",
            entity_type="Dataset",
        )
        over_policing = _register_entity(
            self.pipeline,
            local_id="over_policing_scope",
            label="Over-Policing",
            entity_type="Bias",
        )
        harsher_sentencing = _register_entity(
            self.pipeline,
            local_id="harsher_sentencing_scope",
            label="Harsher Sentencing",
            entity_type="Bias",
        )

        for target in (over_policing, harsher_sentencing):
            relation = _make_relation(
                historical_data.uri,
                "reflects",
                target.uri,
                field="answer.description",
                source_field="answer.description",
                extractor="lexical",
                context_source="lexical_sentence",
                section_category="semantic",
                section="Body",
                evidence_quote=discourse,
                evidence=discourse,
                source_doc_id="11",
                origin_key="11|answer.description",
                relation_kind="extracted_semantic",
            )
            relation.context = discourse
            accepted, _ = self._validate(relation, [historical_data, target], discourse)
            self.assertFalse(accepted)
            self._assert_validation_code(relation, "entailment:predicate_not_locally_entailed")

    def test_context_level_validation_accepts_supported_relation_when_quote_is_short(self):
        source = _register_entity(
            self.pipeline,
            local_id="facial_recognition_systems",
            label="Facial Recognition Systems",
            entity_type="AISystem",
        )
        target = _register_entity(
            self.pipeline,
            local_id="dark_skinned_women",
            label="dark-skinned women",
            entity_type="Concept",
        )
        discourse = "Facial Recognition Systems performed worse on dark-skinned women than on lighter-skinned men."
        relation = _make_relation(
            source.uri,
            "particularlyAffects",
            target.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Description",
            evidence_quote="performed worse on dark-skinned women",
            evidence="performed worse on dark-skinned women",
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [source, target], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(relation, "accepted")

    def test_question_style_training_relation_is_locally_entailed(self):
        source = _register_entity(
            self.pipeline,
            local_id="question_ai_model",
            label="AI model",
            entity_type="AISystem",
        )
        target = _register_entity(
            self.pipeline,
            local_id="question_dataset",
            label="Dataset",
            entity_type="Dataset",
        )
        discourse = (
            "It's hypothetically possible to train an AI model to predict the job performance "
            "of a job candidate by comparing some of their characteristics to a dataset of current or past employees."
        )
        relation = _make_relation(
            source.uri,
            "trainedOn",
            target.uri,
            field="question.body",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Body",
            discourse_mode="interrogative",
            epistemic_status="questioned",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [source, target], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(relation, "accepted")

    def test_contaminate_phrase_recovers_affects_relation_in_question_modality(self):
        source = _register_entity(
            self.pipeline,
            local_id="biases_contaminate_test",
            label="Biases",
            entity_type="Bias",
        )
        target = _register_entity(
            self.pipeline,
            local_id="fairness_contaminate_test",
            label="Fairness",
            entity_type="Fairness",
        )
        discourse = "To what extent do a persons own biases contaminate the fairness of a dataset during the labelling process?"
        relation = _make_relation(
            source.uri,
            "affects",
            target.uri,
            field="question.body",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Body",
            discourse_mode="interrogative",
            epistemic_status="questioned",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [source, target], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(relation, "accepted")

    def test_part_of_domain_accepts_canonical_domain_variant_types(self):
        source = _register_entity(
            self.pipeline,
            local_id="algorithmic_decision_making_domain_variant",
            label="Algorithmic Decision-Making",
            entity_type="AlgorithmicBias",
        )
        target = _register_entity(
            self.pipeline,
            local_id="criminal_justice_domain_variant",
            label="Criminal Justice",
            entity_type="Justicedomain",
        )
        discourse = (
            "Fairness in algorithmic decision-making, especially in high-stakes domains like criminal justice, "
            "goes beyond technical accuracy."
        )
        relation = _make_relation(
            source.uri,
            "partOfDomain",
            target.uri,
            field="answer.description",
            extractor="lexical",
            context_source="lexical_sentence",
            section_category="semantic",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [source, target], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(relation, "accepted")

    def test_exhibits_risk_rejects_dataset_actor_when_sentence_subject_is_system(self):
        source = _register_entity(
            self.pipeline,
            local_id="compas_bias_variant",
            label="COMPAS",
            entity_type="Dataset",
        )
        target = _register_entity(
            self.pipeline,
            local_id="racial_disparities_bias_variant",
            label="Racial Disparities",
            entity_type="Racialbiasrisk",
        )
        discourse = "A predictive system like COMPAS exhibits systemic racial disparities."
        relation = _make_relation(
            source.uri,
            "exhibitsRisk",
            target.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            evidence_quote="exhibits systemic racial disparities",
            evidence=discourse,
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [source, target], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(relation, "entity_role:dataset_actor_mismatch_system_cue")

    def test_lexical_group_impact_prefers_actual_system_subject(self):
        discourse = (
            "Joy Buolamwini and Timnit Gebru's Gender Shades project exposed severe intersectional bias "
            "in commercial facial recognition systems, demonstrating these systems performed worse on "
            "dark-skinned women compared to light-skinned men."
        )
        joy = _register_entity(self.pipeline, local_id="joy_test", label="Joy Buolamwini", entity_type="Person")
        timnit = _register_entity(self.pipeline, local_id="timnit_test", label="Timnit Gebru", entity_type="Person")
        project = _register_entity(self.pipeline, local_id="gender_shades_test", label="Gender Shades", entity_type="Project")
        bias = _register_entity(self.pipeline, local_id="intersectional_bias_test", label="Intersectional Bias", entity_type="Risk")
        system = _register_entity(
            self.pipeline,
            local_id="frs_test",
            label="Facial Recognition Systems",
            entity_type="Technology",
        )
        harmed_group = _register_entity(
            self.pipeline,
            local_id="dark_skinned_women_test",
            label="dark-skinned women",
            entity_type="Demographic",
        )
        comparison_group = _register_entity(
            self.pipeline,
            local_id="light_skinned_men_test",
            label="light-skinned men",
            entity_type="Demographic",
        )

        relations = self.pipeline._lexical_semantic_relations_for_unit(
            unit_text=discourse,
            entities=[joy, timnit, project, bias, system, harmed_group, comparison_group],
            discourse_id="901",
            field="theme.description",
            origin_key="theme.description|901",
            unit={"section": "Description", "section_category": "semantic", "unit_id": "u901", "unit_type": "sentence"},
            llm_relations=[],
        )
        impact = {
            (rel.source, rel.relation, rel.target)
            for rel in relations
            if rel.relation in {"affects", "affectedBy", "particularlyAffects"}
        }

        self.assertIn((system.uri, "particularlyAffects", harmed_group.uri), impact)
        self.assertNotIn((project.uri, "particularlyAffects", harmed_group.uri), impact)
        self.assertNotIn((project.uri, "affects", harmed_group.uri), impact)
        self.assertNotIn((joy.uri, "particularlyAffects", harmed_group.uri), impact)
        self.assertNotIn((timnit.uri, "particularlyAffects", harmed_group.uri), impact)
        self.assertNotIn((bias.uri, "particularlyAffects", harmed_group.uri), impact)
        self.assertNotIn((system.uri, "particularlyAffects", comparison_group.uri), impact)

    def test_lexical_affects_direction_is_active(self):
        discourse = "Smart City Algorithms affect underserved communities through unequal service allocation."
        system = _register_entity(
            self.pipeline,
            local_id="smart_city_algorithms_test",
            label="Smart City Algorithms",
            entity_type="AISystem",
        )
        group = _register_entity(
            self.pipeline,
            local_id="underserved_communities_test",
            label="underserved communities",
            entity_type="Demographic",
        )

        relations = self.pipeline._lexical_semantic_relations_for_unit(
            unit_text=discourse,
            entities=[system, group],
            discourse_id="902",
            field="theme.description",
            origin_key="theme.description|902",
            unit={"section": "Description", "section_category": "semantic", "unit_id": "u902", "unit_type": "sentence"},
            llm_relations=[],
        )
        impact = {(rel.source, rel.relation, rel.target) for rel in relations}
        self.assertIn((system.uri, "affects", group.uri), impact)
        self.assertNotIn((group.uri, "affects", system.uri), impact)
        self.assertNotIn((group.uri, "affectedBy", system.uri), impact)

    def test_passive_affected_by_is_canonicalized_to_active_direction(self):
        system = _register_entity(
            self.pipeline,
            local_id="screening_model_test",
            label="Screening Model",
            entity_type="AISystem",
        )
        group = _register_entity(
            self.pipeline,
            local_id="applicants_test",
            label="job applicants",
            entity_type="Demographic",
        )
        discourse = "Job applicants are affected by Screening Model ranking decisions."
        relation = _make_relation(
            group.uri,
            "affectedBy",
            system.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Answer",
            evidence_quote=discourse,
            evidence=discourse,
        )
        relation.context = discourse

        accepted, reason = self._validate(relation, [system, group], discourse)
        self.assertTrue(accepted)
        self._assert_validation_code(relation, "accepted")
        self.assertEqual(relation.relation, "affects")
        self.assertEqual(relation.source, system.uri)
        self.assertEqual(relation.target, group.uri)

    def test_comparison_baseline_group_is_not_treated_as_harmed_target(self):
        system = _register_entity(
            self.pipeline,
            local_id="facial_systems_comparison_test",
            label="Facial Recognition Systems",
            entity_type="Technology",
        )
        comparison_group = _register_entity(
            self.pipeline,
            local_id="light_skinned_men_comparison_test",
            label="light-skinned men",
            entity_type="Demographic",
        )
        discourse = (
            "Facial Recognition Systems performed worse on dark-skinned women compared to light-skinned men."
        )
        relation = _make_relation(
            system.uri,
            "particularlyAffects",
            comparison_group.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Description",
            evidence_quote="performed worse on dark-skinned women compared to light-skinned men",
            evidence="performed worse on dark-skinned women compared to light-skinned men",
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [system, comparison_group], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(relation, "entailment:predicate_not_locally_entailed")

    def test_generic_bias_alias_does_not_substitute_for_specific_impact_subject(self):
        source = _register_entity(
            self.pipeline,
            local_id="resource_allocation_bias_test",
            label="Resource Allocation Bias",
            entity_type="Biasrisk",
            mentions=[{"discourse_id": "905", "start_char": 53, "end_char": 57}],
        )
        target = _register_entity(
            self.pipeline,
            local_id="minority_communities_alias_test",
            label="minority communities",
            entity_type="Demographic",
        )
        discourse = "AI-powered insurance underwriting systems show bias in risk assessment and pricing, particularly affecting minority communities."
        relation = _make_relation(
            source.uri,
            "particularlyAffects",
            target.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Description",
            evidence_quote="particularly affecting minority communities",
            evidence="particularly affecting minority communities",
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [source, target], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(relation, "locality:no_shared_local_proposition")

    def test_generic_risk_label_does_not_override_system_actor(self):
        source = _register_entity(
            self.pipeline,
            local_id="bias_actor_test",
            label="Bias",
            entity_type="Biasrisk",
        )
        target = _register_entity(
            self.pipeline,
            local_id="minority_communities_system_actor_test",
            label="minority communities",
            entity_type="Demographic",
        )
        discourse = (
            "AI-powered insurance underwriting systems show bias in risk assessment and pricing, "
            "particularly affecting minority communities."
        )
        relation = _make_relation(
            source.uri,
            "particularlyAffects",
            target.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Description",
            evidence_quote="particularly affecting minority communities",
            evidence=discourse,
        )
        relation.context = discourse
        accepted, reason = self._validate(relation, [source, target], discourse)
        self.assertFalse(accepted)
        self._assert_validation_code(relation, "entity_role:actor_family_mismatch")

    def test_protected_group_mention_alone_does_not_create_impact_relation(self):
        discourse = "The report discusses bias, discrimination, and low-income neighborhoods in separate examples."
        report = _register_entity(
            self.pipeline,
            local_id="report_test",
            label="Fairness Report",
            entity_type="Project",
        )
        group = _register_entity(
            self.pipeline,
            local_id="low_income_neighborhoods_test",
            label="low-income neighborhoods",
            entity_type="Demographic",
        )
        bias = _register_entity(
            self.pipeline,
            local_id="bias_test",
            label="bias",
            entity_type="Risk",
        )

        relations = self.pipeline._lexical_semantic_relations_for_unit(
            unit_text=discourse,
            entities=[report, group, bias],
            discourse_id="903",
            field="theme.description",
            origin_key="theme.description|903",
            unit={"section": "Description", "section_category": "semantic", "unit_id": "u903", "unit_type": "sentence"},
            llm_relations=[],
        )
        impact = [rel for rel in relations if rel.relation in {"affects", "affectedBy", "particularlyAffects"}]
        self.assertEqual(impact, [])

    def test_discriminate_against_and_particularly_affects_map_to_specific_predicate(self):
        cases = [
            (
                "Dynamic Pricing Algorithms discriminate against low-income neighborhoods.",
                "Dynamic Pricing Algorithms",
                "low-income neighborhoods",
            ),
            (
                "Insurance Scoring Models particularly affect minority communities.",
                "Insurance Scoring Models",
                "minority communities",
            ),
        ]

        for idx, (discourse, source_label, target_label) in enumerate(cases, start=1):
            system = _register_entity(
                self.pipeline,
                local_id=f"specific_system_{idx}",
                label=source_label,
                entity_type="AISystem",
            )
            group = _register_entity(
                self.pipeline,
                local_id=f"specific_group_{idx}",
                label=target_label,
                entity_type="Demographic",
            )
            relations = self.pipeline._lexical_semantic_relations_for_unit(
                unit_text=discourse,
                entities=[system, group],
                discourse_id=f"904_{idx}",
                field="theme.description",
                origin_key=f"theme.description|904_{idx}",
                unit={"section": "Description", "section_category": "semantic", "unit_id": f"u904_{idx}", "unit_type": "sentence"},
                llm_relations=[],
            )
            predicates = {(rel.source, rel.relation, rel.target) for rel in relations}
            self.assertIn((system.uri, "particularlyAffects", group.uri), predicates)

    def test_accepted_semantic_relations_receive_validation_metadata(self):
        system = _register_entity(
            self.pipeline,
            local_id="metadata_system",
            label="Healthcare Risk Algorithms",
            entity_type="AISystem",
            discourse_id="777",
        )
        risk = _register_entity(
            self.pipeline,
            local_id="metadata_risk",
            label="Racial Bias",
            entity_type="Risk",
            discourse_id="777",
        )
        discourse = "Healthcare Risk Algorithms have racial bias in commercial allocation decisions."
        rel = _make_relation(
            system.uri,
            "hasRisk",
            risk.uri,
            field="answer.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Answer",
            evidence_quote=discourse,
            evidence=discourse,
            source_doc_id="777",
            origin_key="777|answer.description",
        )
        rel.context = discourse
        annotations = AnnotationBatch(
            answers={
                "777": ProcessingResult(
                    entities=[system, risk],
                    relations=[rel],
                    metadata={"text": discourse},
                )
            }
        )

        summary = self.pipeline.validate_semantic_relations(annotations)
        self.assertEqual(summary["kept"], 1)
        kept = annotations.answers["777"].relations[0]
        meta = kept.metadata
        self.assertEqual(meta.get("semantic_validation_status"), "accepted")
        self.assertEqual(meta.get("semantic_validation_code"), "accepted")
        self.assertTrue(str(meta.get("semantic_validation_reason") or "").strip())
        self.assertEqual(meta.get("field"), "answer.description")
        self.assertEqual(meta.get("source_field"), "answer.description")
        self.assertEqual(meta.get("section"), "Answer")
        self.assertEqual(meta.get("section_category"), "semantic")
        self.assertEqual(meta.get("evidence"), discourse)
        self.assertIsInstance(getattr(kept, "confidence", None), float)

    def test_validate_semantic_relations_keeps_only_most_specific_predicate_for_same_proposition(self):
        system = _register_entity(
            self.pipeline,
            local_id="insurance_underwriting_specificity_test",
            label="AI-powered insurance underwriting systems",
            entity_type="AISystem",
            discourse_id="915",
        )
        group = _register_entity(
            self.pipeline,
            local_id="minority_communities_specificity_test",
            label="minority communities",
            entity_type="Demographic",
            discourse_id="915",
        )
        discourse = (
            "AI-powered insurance underwriting systems show bias in risk assessment and pricing, "
            "particularly affecting minority communities."
        )

        affects = _make_relation(
            system.uri,
            "affects",
            group.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Description",
            evidence_quote=discourse,
            evidence=discourse,
            source_doc_id="915",
            origin_key="915|theme.description",
        )
        affects.context = discourse
        specific = _make_relation(
            system.uri,
            "particularlyAffects",
            group.uri,
            field="theme.description",
            extractor="llm",
            context_source="llm_relation",
            section_category="semantic",
            section="Description",
            evidence_quote=discourse,
            evidence=discourse,
            source_doc_id="915",
            origin_key="915|theme.description",
        )
        specific.context = discourse

        annotations = AnnotationBatch(
            themes={
                "915": ProcessingResult(
                    entities=[system, group],
                    relations=[affects, specific],
                    metadata={"text": discourse},
                )
            }
        )

        summary = self.pipeline.validate_semantic_relations(annotations)
        self.assertEqual(summary["kept"], 1)
        self.assertEqual(summary["dropped"], 1)
        kept = annotations.themes["915"].relations
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].relation, "particularlyAffects")
        self.assertEqual(kept[0].metadata.get("semantic_validation_code"), "accepted")
        self.assertEqual(
            affects.metadata.get("semantic_validation_code"),
            "specificity:shadowed_by_more_specific_predicate",
        )


if __name__ == "__main__":
    unittest.main()
