import unittest

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer
from utils.semantic_relation_policy import SemanticRelationPolicy


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


class _FakePredicateRefinementClassifier:
    def __init__(self, response=None, responder=None):
        self._response = response
        self._responder = responder
        self.calls = []

    def classify(self, **kwargs):
        self.calls.append(kwargs)
        if self._responder is not None:
            return self._responder(**kwargs)
        return dict(self._response or {
            "decision": "keep_generic", "final_predicate": kwargs.get("original_predicate", "relatedTo"),
            "reasoning_category": "other", "confidence": "ambiguous",
        })


def _make_pipeline(predicate_refinement_classifier=None):
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.wikidata_linker = None
    pipeline._discourse_chunk_spans = {}
    pipeline.semantic_policy = SemanticRelationPolicy()
    pipeline.predicate_refinement_classifier = predicate_refinement_classifier
    return pipeline


def _entity(uri, label, entity_type):
    return Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.9, metadata={})


def _rel(source, relation, target, *, target_is_literal=False, target_literal=None,
         predicate_text=None, object_type=None, evidence_quote=""):
    return Relation(
        source=source, relation=relation, target=target,
        target_is_literal=target_is_literal, target_literal=target_literal,
        object_type=object_type,
        metadata={
            "predicate_text": predicate_text, "evidence_quote": evidence_quote,
            "evidence": evidence_quote, "field": "theme.description",
        },
    )


class DeterministicPreFilterTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_predicate_text_reresolved_when_domain_range_valid(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        tgt = _entity("https://kg/e/t1", "Some Concept", "Concept")
        rel = _rel(subj.uri, "relatedTo", tgt.uri, predicate_text="includes")
        entities_by_uri = {subj.uri: subj, tgt.uri: tgt}
        new_pred, reason = self.pipeline._refine_generic_predicate_deterministic(rel, entities_by_uri)
        self.assertEqual(new_pred, "includes")
        self.assertEqual(reason, "predicate_text_reresolved_domain_range_valid")

    def test_predicate_text_reresolution_rejected_when_domain_range_invalid(self):
        # "trainedOn" requires a system/model-family subject -- a bare
        # Concept subject must not silently receive it.
        subj = _entity("https://kg/e/s1", "Some Concept", "Concept")
        tgt = _entity("https://kg/e/t1", "Some Dataset", "Dataset")
        rel = _rel(subj.uri, "relatedTo", tgt.uri, predicate_text="trained on")
        entities_by_uri = {subj.uri: subj, tgt.uri: tgt}
        new_pred, reason = self.pipeline._refine_generic_predicate_deterministic(rel, entities_by_uri)
        self.assertIsNone(new_pred)
        self.assertEqual(reason, "needs_llm_classification")

    def test_field_description_bullet_pattern_resolved_as_hasdescription(self):
        subj = _entity("https://kg/e/s1", "Some Resource", "Concept")
        rel = _rel(
            subj.uri, "is", "a general-purpose collection of records used for evaluation",
            target_is_literal=True, target_literal="a general-purpose collection of records used for evaluation",
            predicate_text="is",
            evidence_quote="Some Resource: a general-purpose collection of records used for evaluation",
        )
        new_pred, reason = self.pipeline._refine_generic_predicate_deterministic(rel, {subj.uri: subj})
        self.assertEqual(new_pred, "hasDescription")
        self.assertEqual(reason, "field_description_bullet_pattern")

    def test_field_description_bullet_pattern_applies_to_has_too(self):
        # Same general structural shape, regardless of which generic verb
        # the upstream extractor happened to pick for this run.
        subj = _entity("https://kg/e/s1", "Some Resource", "Concept")
        rel = _rel(
            subj.uri, "has", "a general-purpose collection of records used for evaluation",
            target_is_literal=True, target_literal="a general-purpose collection of records used for evaluation",
            predicate_text="has",
            evidence_quote="Some Resource: a general-purpose collection of records used for evaluation",
        )
        new_pred, reason = self.pipeline._refine_generic_predicate_deterministic(rel, {subj.uri: subj})
        self.assertEqual(new_pred, "hasDescription")
        self.assertEqual(reason, "field_description_bullet_pattern")

    def test_plain_generic_relation_escalates_to_llm(self):
        subj = _entity("https://kg/e/s1", "Some Subject", "Concept")
        rel = _rel(subj.uri, "relatedTo", "some object", target_is_literal=True, target_literal="some object",
                    predicate_text="relatedTo", evidence_quote="Some Subject is generally connected to some object.")
        new_pred, reason = self.pipeline._refine_generic_predicate_deterministic(rel, {subj.uri: subj})
        self.assertIsNone(new_pred)
        self.assertEqual(reason, "needs_llm_classification")


class LLMEscalationIntegrationTests(unittest.TestCase):
    def _run(self, relation, entities_by_uri, classifier, discourse_text=""):
        pipeline = _make_pipeline(predicate_refinement_classifier=classifier)
        pipeline._refine_generic_predicates_for_relations([relation], entities_by_uri, discourse_text)
        return relation

    def test_definitional_relation_upgraded_not_blindly_retained_as_is(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(
            subj.uri, "is", "a description of the dataset contents",
            target_is_literal=True, target_literal="a description of the dataset contents",
            predicate_text="is", evidence_quote="Some Dataset: a description of the dataset contents",
        )
        classifier = _FakePredicateRefinementClassifier(response={
            "decision": "upgrade", "final_predicate": "hasDescription",
            "reasoning_category": "description", "confidence": "confident",
        })
        result = self._run(rel, {subj.uri: subj}, classifier)
        self.assertEqual(result.relation, "hasDescription")
        self.assertEqual(result.metadata.get("generic_predicate_original"), "is")

    def test_generic_has_refined_when_specific_relation_entailed(self):
        subj = _entity("https://kg/e/s1", "Some System", "AISystem")
        tgt = _entity("https://kg/e/t1", "Some Risk", "Risk")
        rel = _rel(subj.uri, "has", tgt.uri, predicate_text="has", evidence_quote="Some System poses a risk.")
        classifier = _FakePredicateRefinementClassifier(response={
            "decision": "upgrade", "final_predicate": "hasRisk",
            "reasoning_category": "characteristic", "confidence": "confident",
        })
        result = self._run(rel, {subj.uri: subj, tgt.uri: tgt}, classifier)
        self.assertEqual(result.relation, "hasRisk")

    def test_genuine_possession_may_remain_has(self):
        subj = _entity("https://kg/e/s1", "Some Subject", "Concept")
        rel = _rel(subj.uri, "has", "some literal value", target_is_literal=True, target_literal="some literal value",
                    predicate_text="has", evidence_quote="Some Subject has some literal value.")
        classifier = _FakePredicateRefinementClassifier(response={
            "decision": "keep_generic", "final_predicate": "has",
            "reasoning_category": "possession", "confidence": "confident",
        })
        result = self._run(rel, {subj.uri: subj}, classifier)
        self.assertEqual(result.relation, "has")

    def test_relatedto_remains_when_only_weak_association_supported(self):
        subj = _entity("https://kg/e/s1", "Some Subject", "Concept")
        rel = _rel(subj.uri, "relatedTo", "some loosely connected topic", target_is_literal=True,
                    target_literal="some loosely connected topic", predicate_text="relatedTo",
                    evidence_quote="Some Subject is broadly connected to some loosely connected topic.")
        classifier = _FakePredicateRefinementClassifier(response={
            "decision": "keep_generic", "final_predicate": "relatedTo",
            "reasoning_category": "weak_association", "confidence": "ambiguous",
        })
        result = self._run(rel, {subj.uri: subj}, classifier)
        self.assertEqual(result.relation, "relatedTo")

    def test_unknown_specific_predicate_remains_local_unmapped(self):
        subj = _entity("https://kg/e/s1", "Some Subject", "Concept")
        tgt = _entity("https://kg/e/t1", "Some Object", "Concept")
        rel = _rel(subj.uri, "relatedTo", tgt.uri, predicate_text="relatedTo",
                    evidence_quote="Some Subject formally supersedes Some Object in this context.")
        classifier = _FakePredicateRefinementClassifier(response={
            "decision": "local_specific", "final_predicate": "supersedes",
            "reasoning_category": "other", "confidence": "confident",
        })
        result = self._run(rel, {subj.uri: subj, tgt.uri: tgt}, classifier)
        self.assertEqual(result.relation, "supersedes")
        self.assertEqual(result.predicate_mapping_status, "local_unmapped")

    def test_causal_relation_rejected_when_context_does_not_entail_causation(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(subj.uri, "relatedTo", "some unrelated outcome", target_is_literal=True,
                    target_literal="some unrelated outcome", predicate_text="relatedTo",
                    evidence_quote="Some Dataset is mentioned near some unrelated outcome.")

        def responder(**kwargs):
            return {"decision": "upgrade", "final_predicate": "causes",
                    "reasoning_category": "causes", "confidence": "confident"}

        classifier = _FakePredicateRefinementClassifier(responder=responder)
        result = self._run(rel, {subj.uri: subj}, classifier)
        # "causes" requires a non-dataset/resource/person/organisation
        # source per the domain/range registry -- a Dataset subject must
        # be rejected even though the LLM confidently proposed it.
        self.assertNotEqual(result.relation, "causes")
        self.assertEqual(result.relation, "relatedTo")

    def test_subject_object_types_used_for_validation_not_invention(self):
        # The deterministic pre-filter must never invent a predicate from
        # type alone -- only validate an already-proposed one.
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        tgt = _entity("https://kg/e/t1", "Some Concept", "Concept")
        rel = _rel(subj.uri, "relatedTo", tgt.uri, predicate_text=None, evidence_quote="")
        new_pred, reason = self.pipeline_for_type_test()._refine_generic_predicate_deterministic(
            rel, {subj.uri: subj, tgt.uri: tgt}
        )
        self.assertIsNone(new_pred)
        self.assertEqual(reason, "needs_llm_classification")

    def pipeline_for_type_test(self):
        return _make_pipeline()

    def test_literal_object_preserves_predicate_semantics(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(
            subj.uri, "is", "83.5", target_is_literal=True, target_literal="83.5", object_type="percentage",
            predicate_text="is", evidence_quote="Some Dataset with 83.5% coverage.",
        )
        classifier = _FakePredicateRefinementClassifier(response={
            "decision": "upgrade", "final_predicate": "hasAttribute",
            "reasoning_category": "attribute_assignment", "confidence": "confident",
        })
        result = self._run(rel, {subj.uri: subj}, classifier)
        self.assertEqual(result.relation, "hasAttribute")
        self.assertTrue(result.target_is_literal)
        self.assertEqual(result.target_literal, "83.5")

    def test_predicate_change_preserves_evidence_and_provenance(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(
            subj.uri, "is", "a general description", target_is_literal=True, target_literal="a general description",
            predicate_text="is", evidence_quote="Some Dataset: a general description",
        )
        rel.metadata["discourse_id"] = "d1"
        rel.metadata["source_doc_id"] = "doc1"
        rel.uri = "https://kg/r/rel1"
        classifier = _FakePredicateRefinementClassifier(response={
            "decision": "upgrade", "final_predicate": "hasDescription",
            "reasoning_category": "description", "confidence": "confident",
        })
        result = self._run(rel, {subj.uri: subj}, classifier)
        self.assertEqual(result.uri, "https://kg/r/rel1")
        self.assertEqual(result.metadata.get("discourse_id"), "d1")
        self.assertEqual(result.metadata.get("source_doc_id"), "doc1")
        self.assertEqual(result.metadata.get("evidence_quote"), "Some Dataset: a general description")
        self.assertEqual(result.source, subj.uri)

    def test_no_relation_uri_or_state_corruption_after_refinement(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(
            subj.uri, "is", "a general description", target_is_literal=True, target_literal="a general description",
            predicate_text="is", evidence_quote="Some Dataset: a general description",
        )
        classifier = _FakePredicateRefinementClassifier(response={
            "decision": "upgrade", "final_predicate": "hasDescription",
            "reasoning_category": "description", "confidence": "confident",
        })
        result = self._run(rel, {subj.uri: subj}, classifier)
        self.assertTrue(result.target_is_literal)
        self.assertEqual(result.target, "a general description")
        self.assertEqual(result.target_literal, "a general description")


if __name__ == "__main__":
    unittest.main()
