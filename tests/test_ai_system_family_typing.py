import tempfile
import unittest

from config import Config
from models import Entity
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.canonical_linker import CanonicalLinker
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer


# Test-isolation fix: CanonicalLinker(output_dir=...) writes
# canonical_linking_log.jsonl into that directory. A literal "output"
# here resolved to the real production output/ directory and silently
# overwrote its live log with this module's synthetic test data.
_TEST_OUTPUT_DIR = tempfile.mkdtemp(prefix="skg_test_ai_system_family_typing_")


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


class _FixedSpanEntityExtractor:
    def __init__(self, entities):
        self._entities = entities

    def extract(self, text, doc_id=None, context=None):
        out = []
        for label, entity_type, start, end in self._entities:
            out.append(
                Entity(
                    uri=f"{Config.NAMESPACES['etd']}placeholder_{start}",
                    label=label, entity_type=entity_type, confidence=0.9,
                    metadata={
                        "source": "llm", "surface_form": label,
                        "start_char": start, "end_char": end,
                        "spans": [(start, end)], "ontology_uri": "",
                    },
                )
            )
        return out


class _FakeAiSystemFamilyClassifier:
    def __init__(self, response=None, responder=None):
        self._response = response
        self._responder = responder
        self.calls = []

    def classify(self, *, surface, sentence, proposed_type=""):
        self.calls.append({"surface": surface, "sentence": sentence, "proposed_type": proposed_type})
        if self._responder is not None:
            return self._responder(surface=surface, sentence=sentence, proposed_type=proposed_type)
        return dict(self._response or {"label": "Ambiguous", "confidence": "ambiguous"})


def _make_pipeline(entity_extractor=None, ai_system_family_classifier=None):
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.entity_extractor = entity_extractor
    pipeline.wikidata_linker = None
    pipeline.dataset_family_classifier = None
    pipeline.population_group_classifier = None
    pipeline.ai_system_family_classifier = ai_system_family_classifier
    pipeline._discourse_chunk_spans = {}
    pipeline.semantic_policy = None
    return pipeline


class DeterministicPreFilterTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_legal_entity_suffix_resolved_as_organization(self):
        new_type, reason = self.pipeline._classify_ai_system_family_deterministic(label="Some Vendor Inc")
        self.assertEqual(new_type, "Organization")
        self.assertEqual(reason, "legal_entity_suffix_present")

    def test_clause_with_auxiliary_verb_resolved_as_process(self):
        new_type, reason = self.pipeline._classify_ai_system_family_deterministic(
            label="was reviewed by an external panel"
        )
        self.assertEqual(new_type, "Process")
        self.assertEqual(reason, "clause_shaped_activity_not_a_system")

    def test_leading_gerund_resolved_as_process(self):
        new_type, reason = self.pipeline._classify_ai_system_family_deterministic(label="Testing Procedures")
        self.assertEqual(new_type, "Process")
        self.assertEqual(reason, "gerund_shaped_activity_label")

    def test_trailing_gerund_resolved_as_process(self):
        new_type, reason = self.pipeline._classify_ai_system_family_deterministic(label="Red Teaming")
        self.assertEqual(new_type, "Process")
        self.assertEqual(reason, "gerund_shaped_activity_label")

    def test_clean_short_label_escalates_to_llm(self):
        new_type, reason = self.pipeline._classify_ai_system_family_deterministic(label="Some Platform")
        self.assertIsNone(new_type)
        self.assertEqual(reason, "needs_llm_classification")

    def test_indefinite_pronoun_ending_in_ing_does_not_misfire_gerund_check(self):
        new_type, reason = self.pipeline._classify_ai_system_family_deterministic(label="Something Unclear")
        self.assertIsNone(new_type)
        self.assertEqual(reason, "needs_llm_classification")

    def test_label_containing_model_word_still_escalates(self):
        # "model" appearing in the label must not deterministically resolve
        # to Model -- only the LLM, using context, may do that.
        new_type, reason = self.pipeline._classify_ai_system_family_deterministic(label="Model Card")
        self.assertIsNone(new_type)
        self.assertEqual(reason, "needs_llm_classification")


class LLMEscalationIntegrationTests(unittest.TestCase):
    def _run(self, text, label, entity_type, classifier):
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, entity_type, start, end)])
        pipeline = _make_pipeline(entity_extractor=extractor, ai_system_family_classifier=classifier)
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])
        matching = [e for e in results[0] if e.label == label]
        return matching[0] if matching else results[0][0]

    def test_deployed_ai_driven_application_confirmed_as_aisystem(self):
        text = "The Screening Platform flagged several applications for review."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "AISystem", "confidence": "confident"})
        entity = self._run(text, "The Screening Platform", "AISystem", classifier)
        self.assertEqual(entity.entity_type, "AISystem")

    def test_underlying_trained_model_confirmed_as_model(self):
        text = "The underlying scoring model was trained on historical outcomes."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "Model", "confidence": "confident"})
        entity = self._run(text, "The underlying scoring model", "AISystem", classifier)
        self.assertEqual(entity.entity_type, "Model")

    def test_ordinary_software_tool_reclassified_as_softwareproduct(self):
        text = "Reviewers logged their notes in a shared spreadsheet application."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "SoftwareProduct", "confidence": "confident"})
        entity = self._run(text, "a shared spreadsheet application", "Model", classifier)
        self.assertEqual(entity.entity_type, "SoftwareProduct")

    def test_workflow_activity_reclassified_as_process(self):
        text = "The team ran a structured review before deployment."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "Process", "confidence": "confident"})
        entity = self._run(text, "a structured review", "System", classifier)
        self.assertEqual(entity.entity_type, "Process")

    def test_audit_workflow_reclassified_as_auditprocess(self):
        text = "An independent compliance review examined the deployment for fairness violations."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "AuditProcess", "confidence": "confident"})
        entity = self._run(text, "An independent compliance review", "AISystem", classifier)
        self.assertEqual(entity.entity_type, "AuditProcess")

    def test_evaluation_activity_reclassified_as_evaluation(self):
        text = "A structured assessment measured accuracy across subgroups."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "Evaluation", "confidence": "confident"})
        entity = self._run(text, "A structured assessment", "Model", classifier)
        self.assertEqual(entity.entity_type, "Evaluation")

    def test_governance_methodology_reclassified_as_aigovernanceframework(self):
        text = "The organisation adopted a structured methodology for approving new deployments."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "AiGovernanceFramework", "confidence": "confident"})
        entity = self._run(text, "a structured methodology", "AISystem", classifier)
        self.assertEqual(entity.entity_type, "AiGovernanceFramework")

    def test_institution_reclassified_as_organization(self):
        text = "The Regional Oversight Authority reviewed the deployment plan."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "Organization", "confidence": "confident"})
        entity = self._run(text, "The Regional Oversight Authority", "AISystem", classifier)
        self.assertEqual(entity.entity_type, "Organization")

    def test_framework_mentioned_in_ai_context_does_not_become_aisystem(self):
        text = "A widely used framework for structuring AI risk reviews was applied here."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "AiGovernanceFramework", "confidence": "confident"})
        entity = self._run(text, "A widely used framework for structuring AI risk reviews", "AISystem", classifier)
        self.assertNotEqual(entity.entity_type, "AISystem")

    def test_process_containing_word_model_does_not_become_model(self):
        text = "The team completed a formal review of the model before release."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "Evaluation", "confidence": "confident"})
        entity = self._run(text, "a formal review of the model", "Model", classifier)
        self.assertNotEqual(entity.entity_type, "Model")

    def test_same_surface_different_context_receives_different_types(self):
        text_a = "The Analyzer flagged the application automatically."
        text_b = "The Analyzer conducted a scheduled compliance check."

        def responder_a(*, surface, sentence, proposed_type=""):
            return {"label": "AISystem", "confidence": "confident"}

        def responder_b(*, surface, sentence, proposed_type=""):
            return {"label": "AuditProcess", "confidence": "confident"}

        entity_a = self._run(text_a, "The Analyzer", "AISystem", _FakeAiSystemFamilyClassifier(responder=responder_a))
        entity_b = self._run(text_b, "The Analyzer", "AISystem", _FakeAiSystemFamilyClassifier(responder=responder_b))
        self.assertEqual(entity_a.entity_type, "AISystem")
        self.assertEqual(entity_b.entity_type, "AuditProcess")

    def test_ambiguous_llm_result_falls_back_to_concept(self):
        text = "Something Unclear was mentioned only in passing."
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "AISystem", "confidence": "ambiguous"})
        entity = self._run(text, "Something Unclear", "System", classifier)
        self.assertEqual(entity.entity_type, "Concept")

    def test_no_classifier_available_falls_back_to_conservative_concept(self):
        text = "Some Unresolved Mention appeared without further detail."
        entity = self._run(text, "Some Unresolved Mention", "AISystem", None)
        self.assertEqual(entity.entity_type, "Concept")
        self.assertEqual(entity.metadata.get("ai_system_family_typing_reason"), "no_confident_signal_abstained_no_classifier")


class OntologyGroundingTests(unittest.TestCase):
    def test_model_no_longer_grounds_identically_to_aisystem(self):
        mapper = OntologyMapper()
        ai_entity = Entity(
            uri="https://alfie-project.eu/etd-hub#a1", label="Some System", entity_type="AISystem",
            confidence=1.0, metadata={},
        )
        model_entity = Entity(
            uri="https://alfie-project.eu/etd-hub#m1", label="Some Model", entity_type="Model",
            confidence=1.0, metadata={},
        )
        ai_mapped = mapper.map_entity(ai_entity)
        model_mapped = mapper.map_entity(model_entity)
        self.assertIn("aipo:AISystem", ai_mapped)
        self.assertNotIn("aipo:AISystem", model_mapped)
        self.assertIn("etd:Model", model_mapped)

    def test_process_and_auditprocess_ground_to_local_classes(self):
        mapper = OntologyMapper()
        process_entity = Entity(
            uri="https://alfie-project.eu/etd-hub#p1", label="Some Activity", entity_type="Process",
            confidence=1.0, metadata={},
        )
        audit_entity = Entity(
            uri="https://alfie-project.eu/etd-hub#p2", label="Some Audit", entity_type="AuditProcess",
            confidence=1.0, metadata={},
        )
        self.assertIn("etd:Process", mapper.map_entity(process_entity))
        self.assertIn("etd:AuditProcess", mapper.map_entity(audit_entity))


class UriTypeOntologyConsistencyTests(unittest.TestCase):
    def test_retyped_entity_has_consistent_uri_bucket_and_ontology(self):
        text = "Some Unresolved Mention appeared without further detail."
        label = "Some Unresolved Mention"
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, "AISystem", start, end)])
        classifier = _FakeAiSystemFamilyClassifier(response={"label": "Process", "confidence": "confident"})
        pipeline = _make_pipeline(entity_extractor=extractor, ai_system_family_classifier=classifier)
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])
        entity = results[0][0]
        self.assertEqual(entity.entity_type, "Process")
        self.assertIn("process", entity.uri.lower())
        self.assertNotIn("aisystem", entity.uri.lower())


class Task941CompatibilityRegressionTests(unittest.TestCase):
    """The incompatible-type-disagreement protection must still
    correctly block a silent merge after AI-system-family retyping introduces
    new type disagreements sharing an external ID."""

    class _Registry:
        def __init__(self):
            self.entities = {}

        def get_all_entities(self):
            return list(self.entities.values())

        def replace_entity(self, old_uri, new_uri):
            if old_uri in self.entities and new_uri not in self.entities:
                self.entities[new_uri] = self.entities[old_uri]

    def test_incompatible_merge_still_blocked_after_ai_system_retyping(self):
        reg = self._Registry()
        external = {"wikidata_uri": "https://www.wikidata.org/entity/Q999999"}
        organization_member = Entity(
            uri="https://alfie-project.eu/etd-hub#o1", label="Some Vendor", entity_type="Organization",
            confidence=0.9, metadata=dict(external),
        )
        aisystem_member = Entity(
            uri="https://alfie-project.eu/etd-hub#o2", label="Some Vendor", entity_type="AISystem",
            confidence=0.9, metadata=dict(external),
        )
        reg.entities = {organization_member.uri: organization_member, aisystem_member.uri: aisystem_member}

        linker = CanonicalLinker(registry=reg, output_dir=_TEST_OUTPUT_DIR)
        out = linker.run_full_registry()

        self.assertEqual(len(out), 2)
        types = {o.entity_type for o in out}
        self.assertEqual(types, {"Organization", "AISystem"})
        self.assertTrue(organization_member.metadata.get("external_link_suspect"))
        self.assertTrue(aisystem_member.metadata.get("external_link_suspect"))


if __name__ == "__main__":
    unittest.main()
