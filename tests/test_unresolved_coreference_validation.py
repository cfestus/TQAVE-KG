import unittest

from config import Config
from models import AnnotationBatch, Entity, ProcessingResult, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer
from utils.semantic_relation_policy import SemanticRelationPolicy


def _entity(local_id, label, entity_type, **metadata):
    return Entity(
        uri=f"{Config.NAMESPACES['etd']}{local_id}",
        label=label,
        entity_type=entity_type,
        confidence=1.0,
        metadata=metadata,
    )


def _relation(source_uri, predicate, target, evidence, field="answer.description", *, target_is_literal=False, extra_meta=None):
    meta = {
        "field": field,
        "extractor": "llm",
        "context_source": "llm_relation",
        "section_category": "semantic",
        "evidence_quote": evidence,
        "evidence": evidence,
    }
    if extra_meta:
        meta.update(extra_meta)
    rel = Relation(
        source=source_uri,
        relation=predicate,
        target=target.uri if isinstance(target, Entity) else target,
        target_is_literal=target_is_literal,
        target_literal=target if target_is_literal else None,
        confidence=0.9,
        metadata=meta,
    )
    rel.context = evidence
    return rel


class UnresolvedCoreferenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.policy = SemanticRelationPolicy()

    def test_unresolved_subject_it_rejected(self):
        target = _entity("model_alpha", "Model Alpha", "AISystem", section_category="semantic")
        text = "It affects the outcome for Model Alpha."
        rel = _relation(
            "it", "affects", target, text,
            extra_meta={
                "subject_endpoint_unresolved_coreference": True,
                "subject_endpoint_unresolved_original_surface": "it",
                "subject_endpoint_unresolved_reason": "unresolved_referential_pronoun",
                "discourse_id": "d1",
            },
        )
        decision = self.policy.validate(rel, entities_by_uri={target.uri: target}, discourse_text=text)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason_code, "unresolved_coreference")

    def test_unresolved_object_them_rejected(self):
        source = _entity("mentor", "Mentor", "Person", section_category="semantic")
        text = "A mentor proactively reaches out to them."
        rel = _relation(
            source.uri, "reachesOutTo", "them", text, target_is_literal=True,
            extra_meta={
                "object_endpoint_unresolved_coreference": True,
                "object_endpoint_unresolved_original_surface": "them",
                "object_endpoint_unresolved_reason": "unresolved_referential_pronoun",
                "discourse_id": "d2",
            },
        )
        decision = self.policy.validate(rel, entities_by_uri={source.uri: source}, discourse_text=text)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason_code, "unresolved_coreference")

    def test_unresolved_this_disagreement_rejected(self):
        target = _entity("critical_issue", "A Critical Issue", "Concept", section_category="semantic")
        text = "This disagreement highlights a critical issue in the field."
        rel = _relation(
            "this disagreement", "highlights", target, text,
            extra_meta={
                "subject_endpoint_unresolved_coreference": True,
                "subject_endpoint_unresolved_original_surface": "this disagreement",
                "subject_endpoint_unresolved_reason": "unresolved_generic_demonstrative_phrase",
                "discourse_id": "d3",
            },
        )
        decision = self.policy.validate(rel, entities_by_uri={target.uri: target}, discourse_text=text)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason_code, "unresolved_coreference")

    def test_resolved_nominal_pronoun_validated_normally(self):
        tool = _entity("screening_tool", "Screening Tool", "AISystem", section_category="semantic")
        applicants = _entity("applicants", "Applicants", "Group", section_category="semantic")
        text = "The Screening Tool flagged several Applicants, and reviewers later overruled it."
        rel = _relation(
            tool.uri, "flags", applicants, text,
            extra_meta={
                "object_endpoint_coref_resolved": True,
                "object_endpoint_coref_resolution_method": "occurrence_level_preceding_mention",
                "object_endpoint_original_surface": "it",
            },
        )
        decision = self.policy.validate(
            rel, entities_by_uri={tool.uri: tool, applicants.uri: applicants}, discourse_text=text,
        )
        self.assertNotEqual(decision.reason_code, "unresolved_coreference")

    def test_resolved_proposition_subject_validated_normally(self):
        proposition = _entity(
            "proposition_body8", "Proposition Body:8", "Proposition",
            section_category="semantic", discourse_scoped_identity=True,
        )
        legal = _entity("legal_principles", "Legal Principles", "Concept", section_category="semantic")
        text = "This undermines legal principles such as the right to contest evidence."
        rel = _relation(
            proposition.uri, "undermines", legal, text,
            extra_meta={
                "subject_endpoint_coref_resolved": True,
                "subject_endpoint_coref_resolution_method": "propositional_preceding_unit_reference",
                "subject_endpoint_original_surface": "This",
            },
        )
        decision = self.policy.validate(
            rel, entities_by_uri={proposition.uri: proposition, legal.uri: legal}, discourse_text=text,
        )
        self.assertNotEqual(decision.reason_code, "unresolved_coreference")

    def test_possessive_head_endpoint_validated_normally(self):
        team = _entity("organisations_teams", "Organisations Teams", "Organisation", section_category="semantic")
        text = "If organisations/teams do this, share their approach publically."
        rel = _relation(
            team.uri, "share", "approach", text, target_is_literal=True,
            extra_meta={
                "object_possessor_original_surface": "their approach",
                "object_possessor_head_noun_phrase": "approach",
                "object_possessor_entity_uri": team.uri,
                "object_possessor_resolution_method": "reflexive_subject",
            },
        )
        decision = self.policy.validate(rel, entities_by_uri={team.uri: team}, discourse_text=text)
        self.assertNotEqual(decision.reason_code, "unresolved_coreference")

    def test_explicit_named_entity_unaffected(self):
        model = _entity("hiring_model", "Hiring Model", "AISystem", section_category="semantic")
        dataset = _entity("applicant_dataset", "Applicant Dataset", "Dataset", section_category="semantic")
        text = "Could a Hiring Model be trained on the Applicant Dataset?"
        rel = _relation(model.uri, "trainedOn", dataset, text, field="question.body")
        decision = self.policy.validate(
            rel, entities_by_uri={model.uri: model, dataset.uri: dataset}, discourse_text=text,
        )
        self.assertTrue(decision.accepted)
        self.assertNotEqual(decision.reason_code, "unresolved_coreference")

    def test_rejection_diagnostics_preserve_surface_and_reason(self):
        target = _entity("model_alpha", "Model Alpha", "AISystem", section_category="semantic")
        text = "It affects the outcome for Model Alpha."
        rel = _relation(
            "it", "affects", target, text,
            extra_meta={
                "subject_endpoint_unresolved_coreference": True,
                "subject_endpoint_unresolved_original_surface": "it",
                "subject_endpoint_unresolved_reason": "unresolved_referential_pronoun",
                "discourse_id": "d7",
                "field": "answer.description",
            },
        )
        decision = self.policy.validate(rel, entities_by_uri={target.uri: target}, discourse_text=text)
        self.assertIn("subject", decision.reason_detail)
        self.assertIn("'it'", decision.reason_detail)
        self.assertIn("unresolved_referential_pronoun", decision.reason_detail)
        self.assertIn("d7", decision.reason_detail)
        self.assertIn("answer.description", decision.reason_detail)


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


class UnresolvedCoreferenceDroppedFromMaterializationTests(unittest.TestCase):
    def test_no_unresolved_relation_reaches_materialization(self):
        pipeline = _make_pipeline()
        target = _entity("model_alpha", "Model Alpha", "AISystem", section_category="semantic")
        text = "It affects the outcome for Model Alpha."
        bad_rel = _relation(
            "it", "affects", target, text,
            extra_meta={
                "subject_endpoint_unresolved_coreference": True,
                "subject_endpoint_unresolved_original_surface": "it",
                "subject_endpoint_unresolved_reason": "unresolved_referential_pronoun",
                "discourse_id": "d9",
            },
        )
        good_rel = _relation(
            target.uri, "affects", "the outcome", text, target_is_literal=True,
        )

        pr = ProcessingResult(
            entities=[target], relations=[bad_rel, good_rel], metadata={"text": text},
        )
        annotations = AnnotationBatch(answers={"d9": pr})
        pipeline.global_registry.get_or_create(target)

        summary = pipeline.validate_semantic_relations(annotations)

        self.assertEqual(summary["dropped"], 1)
        self.assertEqual(len(pr.relations), 1)
        self.assertIs(pr.relations[0], good_rel)
        for rel in pr.relations:
            self.assertFalse(rel.metadata.get("subject_endpoint_unresolved_coreference"))
            self.assertFalse(rel.metadata.get("object_endpoint_unresolved_coreference"))

    def test_unresolved_relation_with_unregistered_predicate_still_rejected(self):
        # Real-world gap: a predicate the LLM produced that never made it
        # into the curated SEMANTIC_RELATION_PREDICATES registry (e.g.
        # "highlights") would otherwise skip policy.validate() entirely via
        # the predicate-routing shortcut and be kept unvalidated -- the
        # unresolved-coreference check must fire before that shortcut too.
        pipeline = _make_pipeline()
        target = _entity("critical_issue", "A Critical Issue", "Concept", section_category="semantic")
        text = "This disagreement highlights a critical issue in the field."
        bad_rel = _relation(
            "this disagreement", "highlights", target, text,
            extra_meta={
                "subject_endpoint_unresolved_coreference": True,
                "subject_endpoint_unresolved_original_surface": "this disagreement",
                "subject_endpoint_unresolved_reason": "unresolved_generic_demonstrative_phrase",
                "discourse_id": "d10",
            },
        )
        pr = ProcessingResult(entities=[target], relations=[bad_rel], metadata={"text": text})
        annotations = AnnotationBatch(answers={"d10": pr})
        pipeline.global_registry.get_or_create(target)

        summary = pipeline.validate_semantic_relations(annotations)

        self.assertEqual(summary["dropped"], 1)
        self.assertEqual(len(pr.relations), 0)


if __name__ == "__main__":
    unittest.main()
