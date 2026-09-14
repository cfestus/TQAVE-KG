import unittest

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer
from utils.quantity_literal_classifier import classify_quantity_candidate


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
                    label=label,
                    entity_type=entity_type,
                    confidence=0.9,
                    metadata={
                        "source": "llm", "surface_form": label,
                        "start_char": start, "end_char": end,
                        "spans": [(start, end)], "ontology_uri": "",
                    },
                )
            )
        return out


def _make_pipeline(entity_extractor=None):
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
    pipeline._discourse_chunk_spans = {}
    pipeline.semantic_policy = None
    return pipeline


def _unit(unit_id, text, start_offset):
    return {
        "unit_id": unit_id, "text": text, "start_offset": start_offset,
        "end_offset": start_offset + len(text), "section_category": "semantic", "unit_type": "sentence",
    }


def _resolve_unit_relations(pipeline, relation, unit, discourse_id="d1", field="answer.description", local_entities=None):
    return pipeline._resolve_semantic_unit_relations(
        relations=[relation],
        local_entities=local_entities if local_entities is not None else [],
        discourse_text=unit["text"], discourse_id=discourse_id, field=field,
        origin_key=f"{field}|{discourse_id}", unit=unit, unit_text=unit["text"],
    )


# ============================================================
# 1. Classifier unit tests -- pure function, no pipeline required.
# ============================================================
class QuantityLiteralClassifierUnitTests(unittest.TestCase):
    def test_bare_percentage_is_pure_quantity(self):
        d = classify_quantity_candidate("39%")
        self.assertTrue(d.is_pure_quantity)
        self.assertEqual(d.object_type, "percentage")
        self.assertEqual(d.literal_value, "39")

    def test_integer_count_is_pure_quantity_with_integer_datatype(self):
        d = classify_quantity_candidate("1,270")
        self.assertTrue(d.is_pure_quantity)
        self.assertEqual(d.literal_value, "1270")
        self.assertEqual(d.datatype, "xsd:integer")

    def test_decimal_measurement_is_pure_quantity_with_decimal_datatype(self):
        d = classify_quantity_candidate("0.82")
        self.assertTrue(d.is_pure_quantity)
        self.assertEqual(d.literal_value, "0.82")
        self.assertEqual(d.datatype, "xsd:decimal")

    def test_currency_amount_is_pure_quantity_with_currency_qualifier(self):
        d = classify_quantity_candidate("£2.5 million")
        self.assertTrue(d.is_pure_quantity)
        self.assertEqual(d.object_type, "money")
        self.assertEqual(d.literal_value, "2.5")
        self.assertEqual(d.qualifiers.get("currency"), "GBP")

    def test_multiplier_comparison_is_pure_quantity(self):
        d = classify_quantity_candidate("6 times")
        self.assertTrue(d.is_pure_quantity)
        self.assertEqual(d.object_type, "comparison")
        self.assertEqual(d.literal_value, "6")

    def test_count_plus_unit_phrase_decomposed(self):
        d = classify_quantity_candidate("12,800 essays")
        self.assertTrue(d.is_decomposable)
        self.assertEqual(d.residual_label, "essays")
        self.assertEqual(d.literal_value, "12800")

    def test_percentage_plus_demographic_noun_decomposed_not_pure(self):
        # The residual referent ("Women") is preserved as a candidate
        # entity label -- the quantity classifier itself never assigns
        # DemographicGroup or any other type; it only proposes a label
        # split, and entity_type is untouched by design (verified at the
        # pipeline-integration layer below).
        d = classify_quantity_candidate("39% Women")
        self.assertTrue(d.is_decomposable)
        self.assertEqual(d.residual_label, "Women")
        self.assertEqual(d.literal_value, "39")
        self.assertEqual(d.object_type, "percentage")

    def test_model_version_identifier_remains_unclassified(self):
        self.assertIsNone(classify_quantity_candidate("Model 3"))

    def test_standard_identifier_remains_unclassified(self):
        self.assertIsNone(classify_quantity_candidate("ISO 27001"))

    def test_year_containing_proper_name_remains_unclassified(self):
        self.assertIsNone(classify_quantity_candidate("2025 Census"))

    def test_organization_name_with_digits_remains_unclassified(self):
        self.assertIsNone(classify_quantity_candidate("Article 6"))

    def test_dataset_name_with_version_number_remains_unclassified(self):
        self.assertIsNone(classify_quantity_candidate("ASAP 2.0 Dataset"))

    def test_label_with_no_digits_is_not_a_candidate_at_all(self):
        self.assertIsNone(classify_quantity_candidate("COMPAS Recidivism Dataset"))

    def test_average_score_of_number_partitive_decomposed(self):
        d = classify_quantity_candidate("average score of 0.82")
        self.assertTrue(d.is_decomposable)
        self.assertEqual(d.residual_label, "average score")
        self.assertEqual(d.literal_value, "0.82")

    def test_leading_preposition_stripped_from_decomposed_residual(self):
        d = classify_quantity_candidate("12% for white students")
        self.assertTrue(d.is_decomposable)
        self.assertEqual(d.residual_label, "white students")
        self.assertEqual(d.literal_value, "12")


# ============================================================
# 2. Entity-minting integration tests (_process_entities_batch).
# ============================================================
class EntityMintingIntegrationTests(unittest.TestCase):
    def _run(self, text, label, entity_type):
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, entity_type, start, end)])
        pipeline = _make_pipeline(entity_extractor=extractor)
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])
        return results[0]

    def test_pure_percentage_entity_not_minted(self):
        text = "Reports found that 39% of respondents disagreed."
        entities = self._run(text, "39%", "Concept")
        self.assertEqual(entities, [])

    def test_pure_currency_entity_not_minted(self):
        text = "The programme cost £2.5 million to run."
        entities = self._run(text, "£2.5 million", "Concept")
        self.assertEqual(entities, [])

    def test_count_plus_unit_entity_decomposed_to_residual_label(self):
        text = "The corpus consists of 12,800 essays collected over two years."
        entities = self._run(text, "12,800 essays", "Concept")
        self.assertEqual(len(entities), 1)
        entity = entities[0]
        self.assertEqual(entity.label, "Essays")
        # entity_type is never touched by the quantity/literal decision --
        # whatever type family typing already decided stays unchanged.
        self.assertEqual(entity.entity_type, "Concept")
        self.assertEqual(entity.metadata.get("quantity_literal_value"), "12800")
        self.assertEqual(entity.metadata.get("quantity_literal_datatype"), "xsd:integer")

    def test_percentage_plus_group_noun_does_not_retype_via_quantity_decision(self):
        # entity_type here is deliberately outside the
        # PopulationGroup/DemographicGroup family (a separate, later
        # pipeline stage) so this test isolates exactly what the quantity
        # decision itself does: split the label, never touch type.
        text = "The sample scored 39% Accuracy across all sites."
        entities = self._run(text, "39% Accuracy", "Metric")
        matching = [e for e in entities if e.metadata.get("quantity_literal_value")]
        self.assertEqual(len(matching), 1)
        entity = matching[0]
        self.assertEqual(entity.label, "Accuracy")
        self.assertEqual(entity.entity_type, "Metric")
        self.assertEqual(entity.metadata.get("quantity_literal_value"), "39")

    def test_model_version_identifier_remains_full_entity(self):
        # Deliberately avoids the bare word "system" elsewhere in the
        # sentence, which independently triggers this pipeline's own
        # pre-existing lexical-entity augmentation for AISystem-family
        # keywords -- unrelated to the quantity/literal decision under test.
        text = "The team upgraded the deployment to Model 3 last year."
        entities = self._run(text, "Model 3", "AISystem")
        matching = [e for e in entities if e.label == "Model 3"]
        self.assertEqual(len(matching), 1)
        self.assertNotIn("quantity_literal_value", matching[0].metadata)

    def test_standard_identifier_remains_full_entity(self):
        text = "The organisation is certified to ISO 27001."
        entities = self._run(text, "ISO 27001", "Concept")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].label, "ISO 27001")

    def test_year_containing_proper_name_remains_full_entity(self):
        text = "Findings from the 2025 Census were released."
        entities = self._run(text, "2025 Census", "Concept")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].label, "2025 Census")

    def test_dataset_family_quantity_lead_guard_still_fires_before_decomposition(self):
        # Regression: the quantity gate must run AFTER the
        # Dataset-family quantity-lead guard, not before it. That guard's
        # own quantity-lead detection needs the entity's ORIGINAL label --
        # if the quantity gate renamed the label first, the guard would no longer
        # see a digit-leading label and would silently fail to redirect a
        # genuinely quantity-shaped Dataset-typed mention to Concept.
        text = "The portal separately tracks 311 Service Requests across boroughs."
        entities = self._run(text, "311 Service Requests", "Dataset")
        self.assertEqual(len(entities), 1)
        entity = entities[0]
        self.assertEqual(entity.metadata.get("dataset_family_original_type"), "Dataset")
        self.assertEqual(entity.metadata.get("dataset_family_typing_reason"), "quantity_or_statistic_like_label")
        self.assertEqual(entity.entity_type, "Concept")
        self.assertEqual(entity.label, "Service Requests")
        self.assertEqual(entity.metadata.get("quantity_literal_value"), "311")


# ============================================================
# 3. Relation-endpoint literal-conversion integration tests
#    (_resolve_semantic_unit_relations, non-literal-typed branch).
# ============================================================
class RelationEndpointLiteralConversionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_non_literal_typed_pure_quantity_object_converted_to_literal(self):
        subject = Entity(
            uri="https://kg/e/system", label="Screening System", entity_type="AISystem",
            confidence=0.9, metadata={"discourse_id": "d1"},
        )
        unit = _unit("Body:1", "The screening system flagged 39% of applications.", 0)
        relation = Relation(
            source=subject.uri, relation="flagged", target="39%",
            target_is_literal=False,
            metadata={"object_text": "39%", "discourse_id": "d1", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, local_entities=[subject])
        rel = resolved[0]
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.target_literal, "39")
        self.assertEqual(rel.target, "39")
        self.assertEqual(rel.object_type, "percentage")
        self.assertFalse(str(rel.target).startswith("http"))

    def test_non_literal_typed_decomposable_object_resolves_to_residual_entity(self):
        subject = Entity(
            uri="https://kg/e/corpus", label="Training Corpus", entity_type="Dataset",
            confidence=0.9, metadata={"discourse_id": "d2"},
        )
        unit = _unit("Body:1", "The corpus was built from 12,800 essays.", 0)
        relation = Relation(
            source=subject.uri, relation="contains", target="12,800 essays",
            target_is_literal=False,
            metadata={"object_text": "12,800 essays", "discourse_id": "d2", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d2", local_entities=[subject])
        rel = resolved[0]
        self.assertFalse(rel.target_is_literal)
        self.assertTrue(str(rel.target).startswith("http"))
        self.assertEqual(rel.metadata.get("object_quantity_decomposed_value"), "12800")
        self.assertEqual(rel.metadata.get("object_quantity_decomposed_original_surface"), "12,800 essays")

    def test_protected_model_identifier_object_still_mints_as_entity(self):
        subject = Entity(
            uri="https://kg/e/team", label="Engineering Team", entity_type="Organization",
            confidence=0.9, metadata={"discourse_id": "d3"},
        )
        unit = _unit("Body:1", "The team deployed Model 3 into production.", 0)
        relation = Relation(
            source=subject.uri, relation="deployed", target="Model 3",
            target_is_literal=False,
            metadata={"object_text": "Model 3", "discourse_id": "d3", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d3", local_entities=[subject])
        rel = resolved[0]
        self.assertFalse(rel.target_is_literal)
        self.assertTrue(str(rel.target).startswith("http"))

    def test_existing_already_literal_comparison_behaviour_unchanged(self):
        # Already-literal shape: target_is_literal already True from the
        # relation extractor -- must go through the pre-existing literal
        # branch untouched by the non-literal-branch addition.
        subject = Entity(
            uri="https://kg/e/facial_recognition", label="Commercial Facial Recognition Systems",
            entity_type="AISystem", confidence=0.9, metadata={"discourse_id": "d4"},
        )
        unit = _unit(
            "Body:1",
            "These systems performed 32 times worse on dark-skinned women compared to light-skinned men.",
            0,
        )
        relation = Relation(
            source=subject.uri, relation="performed", target="32 times worse on dark-skinned women",
            target_is_literal=True, target_literal="32 times worse on dark-skinned women",
            metadata={
                "object_text": "32 times worse on dark-skinned women", "object_type": "comparison",
                "discourse_id": "d4", "evidence_quote": unit["text"],
            },
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d4", local_entities=[subject])
        rel = resolved[0]
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.target_literal, "32 times worse on dark-skinned women")
        self.assertFalse(rel.metadata.get("object_endpoint_unresolved_coreference"))


# ============================================================
# 4. Relation.set_target_literal atomicity (mirrors the
#    set_target_entity invariant tests, in the reverse direction).
# ============================================================
class SetTargetLiteralInvariantTests(unittest.TestCase):
    def test_entity_state_target_converted_to_literal_state_coherent(self):
        rel = Relation(
            source="https://kg/e/subj", relation="hasValue", target="https://kg/e/some_entity",
            target_is_literal=False, target_literal=None, target_datatype=None,
            object_type=None,
        )
        rel.set_target_literal("39", datatype="xsd:decimal", object_type="percentage")
        self.assertEqual(rel.target, "39")
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.target_literal, "39")
        self.assertEqual(rel.target_datatype, "xsd:decimal")
        self.assertEqual(rel.object_type, "percentage")
        self.assertFalse(str(rel.target).startswith("http"))
        self.assertEqual(rel.metadata.get("object_type"), "percentage")

    def test_qualifiers_merged_not_replaced(self):
        rel = Relation(
            source="https://kg/e/subj", relation="hasValue", target="https://kg/e/some_entity",
            qualifiers={"existing": "kept"},
        )
        rel.set_target_literal("6", datatype="xsd:integer", object_type="comparison", qualifiers={"unit": "times"})
        self.assertEqual(rel.qualifiers.get("existing"), "kept")
        self.assertEqual(rel.qualifiers.get("unit"), "times")

    def test_no_stale_entity_uri_survives_conversion(self):
        rel = Relation(
            source="https://kg/e/subj", relation="hasValue", target="https://kg/e/formerly_an_entity",
            target_is_literal=False,
        )
        rel.set_target_literal("12800", datatype="xsd:integer", object_type="count")
        self.assertNotIn("https://kg/e/", str(rel.target))


if __name__ == "__main__":
    unittest.main()
