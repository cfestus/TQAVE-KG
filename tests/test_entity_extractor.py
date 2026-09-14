import unittest
from unittest.mock import patch

from extractors.entity_extractor import LLMEntityExtractor


def _make_extractor():
    return LLMEntityExtractor(ontology_terms={"Concept": "skos:Concept"})


class EntityExtractorPromptTests(unittest.TestCase):
    """Structural checks on the extraction prompt itself: it must generalize
    (no hard-coded entity names or dataset-specific vocabulary) while
    instructing exhaustive, coordination-aware, short-subject-preserving
    extraction."""

    def setUp(self):
        self.extractor = _make_extractor()
        self.prompt = self.extractor._build_prompt("Example text.")

    def test_prompt_instructs_exhaustive_coordinated_extraction(self):
        lowered = self.prompt.lower()
        self.assertIn("exhaustive", lowered)
        self.assertIn("coordinate", lowered)

    def test_prompt_preserves_short_subject_fallback_without_dropping(self):
        lowered = self.prompt.lower()
        self.assertIn("short or generic", lowered)
        self.assertIn("must never cause an entity to be dropped", lowered)

    def test_prompt_does_not_hardcode_specific_entities_or_dataset_vocabulary(self):
        # The fix must be dataset-agnostic: the requirements text should only
        # describe the general principle, not name any of the specific entities
        # this fix was diagnosed against, nor current-dataset-specific terms.
        lowered = self.prompt.lower()
        for forbidden in ("compas", "mimic-cxr", "hmda", "delayed treatment", "mortgage lending"):
            self.assertNotIn(forbidden, lowered)


class EntityExtractorPostProcessingTests(unittest.TestCase):
    """Deterministic tests (no network calls) confirming nothing downstream of
    the LLM call -- confidence filtering, span matching, deduplication --
    silently drops a short/generic entity or a coordinated-list member once
    the model returns it. Isolates the fix to the prompt layer."""

    def setUp(self):
        self.extractor = _make_extractor()

    def _extract_with_mocked_response(self, text, mocked_entities):
        with patch.object(self.extractor, "_call_model", return_value=mocked_entities):
            return self.extractor.extract(text, doc_id="test")

    def test_short_generic_subject_survives_full_pipeline(self):
        text = "AI trained on historical workforce data predominantly male."
        mocked = [
            {"surface_form": "AI", "entity_type": "AISystem", "ontology_uri": "aipo:AISystem", "confidence": 0.95},
            {"surface_form": "historical workforce data", "entity_type": "Dataset", "ontology_uri": "dcat:Dataset", "confidence": 0.93},
        ]
        entities = self._extract_with_mocked_response(text, mocked)
        labels = {e.label for e in entities}
        self.assertIn("AI", labels)
        ai_entity = next(e for e in entities if e.label == "AI")
        self.assertEqual(ai_entity.metadata.get("start_char"), 0)
        self.assertEqual(ai_entity.metadata.get("end_char"), 2)

    def test_coordinated_object_members_both_survive_full_pipeline(self):
        text = "Biased diagnostic tools can lead to misdiagnosis or delayed treatment."
        mocked = [
            {"surface_form": "biased diagnostic tools", "entity_type": "AISystem", "ontology_uri": "", "confidence": 0.9},
            {"surface_form": "misdiagnosis", "entity_type": "Harm", "ontology_uri": "", "confidence": 0.9},
            {"surface_form": "delayed treatment", "entity_type": "Harm", "ontology_uri": "", "confidence": 0.9},
        ]
        entities = self._extract_with_mocked_response(text, mocked)
        labels = {e.label for e in entities}
        self.assertIn("misdiagnosis", labels)
        self.assertIn("delayed treatment", labels)

    def test_short_entity_not_dropped_by_confidence_filter_when_llm_is_confident(self):
        # Regression guard: min_confidence filtering must key off confidence,
        # never off surface-form length.
        extractor = LLMEntityExtractor(ontology_terms={}, min_confidence=0.5)
        mocked = [{"surface_form": "AI", "entity_type": "AISystem", "ontology_uri": "", "confidence": 0.95}]
        with patch.object(extractor, "_call_model", return_value=mocked):
            entities = extractor.extract("The AI is trained on relevant, validated indicators of performance.", doc_id="test")
        self.assertEqual([e.label for e in entities], ["AI"])

    def test_no_indiscriminate_growth_deduplicates_repeated_mentions(self):
        # The fix must not turn every mention into a distinct duplicate entity.
        text = "The AI system uses the AI system's own prior outputs."
        mocked = [
            {"surface_form": "AI system", "entity_type": "AISystem", "ontology_uri": "", "confidence": 0.9},
            {"surface_form": "AI system", "entity_type": "AISystem", "ontology_uri": "", "confidence": 0.95},
        ]
        entities = self._extract_with_mocked_response(text, mocked)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].confidence, 0.95)


if __name__ == "__main__":
    unittest.main()
