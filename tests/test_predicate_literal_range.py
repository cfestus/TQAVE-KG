import unittest

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer
from utils.semantic_relation_policy import (
    SemanticRelationPolicy,
    PREDICATE_REGISTRY,
    literal_datatype_family,
    literal_datatype_permitted,
)


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
    pipeline.semantic_policy = SemanticRelationPolicy()
    pipeline.predicate_refinement_classifier = None
    return pipeline


def _entity(uri, label, entity_type):
    return Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.9, metadata={})


def _rel(source, relation, target, *, target_is_literal=False, target_literal=None,
         object_type=None, target_datatype=None, predicate_text=None, evidence_quote=""):
    return Relation(
        source=source, relation=relation, target=target,
        target_is_literal=target_is_literal, target_literal=target_literal,
        object_type=object_type, target_datatype=target_datatype,
        metadata={"predicate_text": predicate_text, "evidence_quote": evidence_quote, "evidence": evidence_quote},
    )


class RegistryUnitTests(unittest.TestCase):
    def test_predicate_accepting_textual_literal(self):
        entry = PREDICATE_REGISTRY["includes"]
        self.assertTrue(literal_datatype_permitted(entry, "string", None))

    def test_predicate_accepting_numeric_literal(self):
        entry = PREDICATE_REGISTRY["trainedOn"]
        # trainedOn only declares "string" -- a numeric-only literal should
        # NOT be permitted for it (datatype constraint enforced).
        self.assertFalse(literal_datatype_permitted(entry, "count", None))
        self.assertTrue(literal_datatype_permitted(entry, "string", None))

    def test_predicate_accepting_entity_but_not_literal(self):
        entry = PREDICATE_REGISTRY["hasRisk"]
        self.assertFalse(entry.literal_allowed)
        self.assertFalse(literal_datatype_permitted(entry, "string", None))

    def test_predicate_accepting_both_entity_and_literal(self):
        entry = PREDICATE_REGISTRY["includes"]
        self.assertTrue(entry.literal_allowed)
        # entity-side acceptance is a separate, pre-existing range_families
        # concern, unaffected by literal_allowed.
        self.assertIn("concept", entry.range_families)

    def test_invalid_literal_datatype_rejected(self):
        entry = PREDICATE_REGISTRY["referencesSource"]
        self.assertTrue(literal_datatype_permitted(entry, "string", None))
        self.assertFalse(literal_datatype_permitted(entry, "boolean", None))

    def test_literal_datatype_family_classification_is_general(self):
        self.assertEqual(literal_datatype_family("percentage", None), "numeric")
        self.assertEqual(literal_datatype_family(None, "xsd:integer"), "numeric")
        self.assertEqual(literal_datatype_family("date", None), "date")
        self.assertEqual(literal_datatype_family(None, None), "string")


class DomainRangeValidationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_valid_entity_object_still_accepted(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        tgt = _entity("https://kg/e/t1", "Some Concept", "Concept")
        rel = _rel(subj.uri, "includes", tgt.uri)
        self.assertTrue(self.pipeline._predicate_domain_range_plausible("includes", rel, {subj.uri: subj, tgt.uri: tgt}))

    def test_invalid_entity_object_still_rejected(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        tgt = _entity("https://kg/e/t1", "Some Organisation", "Organization")
        rel = _rel(subj.uri, "includes", tgt.uri)
        self.assertFalse(self.pipeline._predicate_domain_range_plausible("includes", rel, {subj.uri: subj, tgt.uri: tgt}))

    def test_valid_literal_object_accepted_when_predicate_permits(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(subj.uri, "includes", "a list of enumerated items", target_is_literal=True,
                    target_literal="a list of enumerated items", object_type="string")
        self.assertTrue(self.pipeline._predicate_domain_range_plausible("includes", rel, {subj.uri: subj}))

    def test_invalid_literal_object_rejected_when_predicate_does_not_permit(self):
        subj = _entity("https://kg/e/s1", "Some System", "AISystem")
        rel = _rel(subj.uri, "hasRisk", "a described risk", target_is_literal=True,
                    target_literal="a described risk", object_type="string")
        self.assertFalse(self.pipeline._predicate_domain_range_plausible("hasRisk", rel, {subj.uri: subj}))

    def test_literal_datatype_mismatch_rejected(self):
        subj = _entity("https://kg/e/s1", "Some System", "AISystem")
        rel = _rel(subj.uri, "trainedOn", "true", target_is_literal=True, target_literal="true", object_type="boolean")
        self.assertFalse(self.pipeline._predicate_domain_range_plausible("trainedOn", rel, {subj.uri: subj}))

    def test_subject_domain_still_enforced_for_literal_object(self):
        # A subject family that does not satisfy the predicate's domain
        # must still be rejected even when the object-side literal
        # datatype would otherwise be permitted -- literal-awareness only
        # changes the OBJECT-side check.
        subj = _entity("https://kg/e/s1", "Some Organisation", "Organization")
        rel = _rel(subj.uri, "trainedOn", "a textual description", target_is_literal=True,
                    target_literal="a textual description", object_type="string")
        self.assertFalse(self.pipeline._predicate_domain_range_plausible("trainedOn", rel, {subj.uri: subj}))

    def test_no_hardcoded_predicate_object_phrase_mapping(self):
        # The regression fix is registry-driven, not a hardcoded
        # if predicate == "includes" special case -- an UNRELATED
        # predicate with no literal_allowed flag must still be rejected
        # for the exact same literal shape "includes" now accepts.
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(subj.uri, "particularlyAffects", "a list of enumerated items", target_is_literal=True,
                    target_literal="a list of enumerated items", object_type="string")
        self.assertFalse(self.pipeline._predicate_domain_range_plausible("particularlyAffects", rel, {subj.uri: subj}))


class RegressionAndProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_includes_with_literal_enumeration_recovers_specific_predicate(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(
            subj.uri, "relatedTo", "Loan applications, approvals, denials, applicant demographics",
            target_is_literal=True, target_literal="Loan applications, approvals, denials, applicant demographics",
            predicate_text="includes",
            evidence_quote="Includes: Loan applications, approvals, denials, applicant demographics",
        )
        new_pred, reason = self.pipeline._refine_generic_predicate_deterministic(rel, {subj.uri: subj})
        self.assertEqual(new_pred, "includes")
        self.assertEqual(reason, "predicate_text_reresolved_domain_range_valid")

    def test_generic_fallback_remains_when_specific_predicate_genuinely_invalid(self):
        subj = _entity("https://kg/e/s1", "Some Organisation", "Organization")
        rel = _rel(
            subj.uri, "relatedTo", "a described risk factor", target_is_literal=True,
            target_literal="a described risk factor", predicate_text="poses risk",
            evidence_quote="Some Organisation poses a described risk factor.",
        )
        new_pred, reason = self.pipeline._refine_generic_predicate_deterministic(rel, {subj.uri: subj})
        # "poses risk" normalizes to hasRisk, which does not permit a
        # literal object -- must not be silently applied.
        self.assertNotEqual(new_pred, "hasRisk")

    def test_literal_conversion_does_not_alter_evidence_or_provenance(self):
        subj = _entity("https://kg/e/s1", "Some Dataset", "Dataset")
        rel = _rel(
            subj.uri, "relatedTo", "Loan applications, approvals, denials",
            target_is_literal=True, target_literal="Loan applications, approvals, denials",
            predicate_text="includes", evidence_quote="Includes: Loan applications, approvals, denials",
        )
        rel.metadata["discourse_id"] = "d1"
        rel.uri = "https://kg/r/rel1"
        new_pred, _ = self.pipeline._refine_generic_predicate_deterministic(rel, {subj.uri: subj})
        self.assertEqual(new_pred, "includes")
        self.assertEqual(rel.uri, "https://kg/r/rel1")
        self.assertEqual(rel.metadata.get("discourse_id"), "d1")
        self.assertEqual(rel.metadata.get("evidence_quote"), "Includes: Loan applications, approvals, denials")
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.target_literal, "Loan applications, approvals, denials")


if __name__ == "__main__":
    unittest.main()
