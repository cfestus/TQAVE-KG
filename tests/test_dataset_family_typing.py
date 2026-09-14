import unittest

from config import Config
from models import Entity
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


class _ConfigWithWikidata(Config):
    USE_WIKIDATA_LINKING = True


class _FixedSpanEntityExtractor:
    """Simulates a compliant LLMEntityExtractor: returns entities with the
    given label/type at the given chunk-local span."""

    def __init__(self, entities):
        # entities: list of (label, entity_type, start, end)
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
                        "source": "llm",
                        "surface_form": label,
                        "start_char": start,
                        "end_char": end,
                        "spans": [(start, end)],
                        "ontology_uri": "",
                    },
                )
            )
        return out


class _FakeWikidataLinker:
    def __init__(self, description):
        self._description = description

    def link(self, label, *, entity_type=None, context_text=None, ontology_types=None,
             source_grounded_aliases=None):
        return {"qid": "Q1", "label": label, "description": self._description, "score": 0.9}


class _FakeDatasetFamilyClassifier:
    """Test double for extractors.dataset_family_classifier.DatasetFamilyTypeClassifier.
    Same call signature as the real classifier. Either returns a fixed
    {"label", "confidence"} dict, or -- if constructed with a callable --
    delegates to it so a test can assert on what was actually passed in
    (surface/sentence/linked_context/proposed_type) and vary the response
    per call."""

    def __init__(self, response=None, responder=None):
        self._response = response
        self._responder = responder
        self.calls = []

    def classify(self, *, surface, sentence, linked_context="", proposed_type=""):
        self.calls.append(
            {"surface": surface, "sentence": sentence, "linked_context": linked_context, "proposed_type": proposed_type}
        )
        if self._responder is not None:
            return self._responder(surface=surface, sentence=sentence, linked_context=linked_context, proposed_type=proposed_type)
        return dict(self._response or {"label": "Ambiguous", "confidence": "ambiguous"})


def _make_pipeline(entity_extractor=None, wikidata_linker=None, config=Config, dataset_family_classifier=None):
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.entity_extractor = entity_extractor
    pipeline.wikidata_linker = wikidata_linker
    pipeline.dataset_family_classifier = dataset_family_classifier
    pipeline._discourse_chunk_spans = {}
    return pipeline


class DeterministicPreFilterTests(unittest.TestCase):
    """_classify_dataset_family_deterministic is a cheap
    PRE-filter only -- it must never itself confirm "stays Dataset" from
    bare keyword presence, and must escalate anything it cannot
    confidently resolve to the LLM classifier rather than guessing."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_quantity_lead_label_deferred_as_concept(self):
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="34.2%", context_window="",
        )
        self.assertEqual(new_type, "Concept")
        self.assertEqual(reason, "quantity_or_statistic_like_label")

    def test_research_study_label_lead_word(self):
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="Study of housing approval patterns", context_window="",
        )
        self.assertEqual(new_type, "ResearchStudy")
        self.assertEqual(reason, "label_structurally_denotes_researchstudy")

    def test_analysis_of_dataset_is_analysis_not_dataset(self):
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="Analysis of regional housing records", context_window="",
        )
        self.assertEqual(new_type, "Analysis")

    def test_publication_label_lead_word(self):
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="Report on housing approval disparities", context_window="",
        )
        self.assertEqual(new_type, "Publication")

    def test_software_product_label_lead_word(self):
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="Spreadsheet application used for review", context_window="",
        )
        self.assertEqual(new_type, "SoftwareProduct")

    def test_single_same_line_category_keyword_still_confident(self):
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="Housing Toolkit",
            context_window="the housing toolkit, a software application, was used to clean values",
        )
        self.assertEqual(new_type, "SoftwareProduct")
        self.assertEqual(reason, "local_context_keyword_match")

    def test_no_same_line_signal_escalates_to_llm(self):
        # No bare-keyword auto-confirmation of Dataset -- must
        # escalate instead of returning None/"stays Dataset" directly.
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="Regional Housing Archive",
            context_window="a collection of 50,000 records covering housing applications",
        )
        self.assertIsNone(new_type)
        self.assertEqual(reason, "needs_llm_classification")

    def test_conflicting_same_line_signals_escalate_to_llm_not_concept_directly(self):
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="Housing Records Compilation",
            context_window="researchers published a study analyzing the software toolkit's output",
        )
        self.assertIsNone(new_type)
        self.assertEqual(reason, "needs_llm_classification")

    def test_ambiguous_generic_mention_escalates_to_llm(self):
        new_type, reason = self.pipeline._classify_dataset_family_deterministic(
            label="Applicant Data",
            context_window="the committee discussed applicant data during the meeting",
        )
        self.assertIsNone(new_type)
        self.assertEqual(reason, "needs_llm_classification")


class LLMEscalationIntegrationTests(unittest.TestCase):
    """End-to-end through _process_entities_batch, exercising the LLM
    classifier escalation path with a fake classifier double."""

    def _run(self, text, label, entity_type, classifier):
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, entity_type, start, end)])
        pipeline = _make_pipeline(entity_extractor=extractor, dataset_family_classifier=classifier)
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])
        return results[0][0]

    def test_named_dataset_with_no_numerical_count_confirmed_by_llm(self):
        text = "Regional Housing Archive was referenced throughout the discussion."
        classifier = _FakeDatasetFamilyClassifier(response={"label": "Dataset", "confidence": "confident"})
        entity = self._run(text, "Regional Housing Archive", "Dataset", classifier)
        self.assertEqual(entity.entity_type, "Dataset")
        self.assertNotIn("dataset_family_original_type", entity.metadata)
        self.assertEqual(len(classifier.calls), 1)

    def test_research_study_using_dataset_confirmed_by_llm(self):
        text = "A recent investigation drew on Regional Housing Archive to examine lending bias."
        classifier = _FakeDatasetFamilyClassifier(response={"label": "ResearchStudy", "confidence": "confident"})
        entity = self._run(text, "A recent investigation", "Dataset", classifier)
        self.assertEqual(entity.entity_type, "ResearchStudy")
        self.assertEqual(entity.metadata.get("dataset_family_original_type"), "Dataset")
        self.assertIn("researchstudy", str(entity.uri).lower())

    def test_repository_hosting_datasets_becomes_resource_not_dataset(self):
        text = "Open City Data Portal hosts dozens of civic datasets for public use."
        classifier = _FakeDatasetFamilyClassifier(response={"label": "Resource", "confidence": "confident"})
        entity = self._run(text, "Open City Data Portal", "Dataset", classifier)
        self.assertEqual(entity.entity_type, "Resource")
        self.assertIn("resource", str(entity.uri).lower())
        self.assertIsNotNone(entity.ontology_uri)

    def test_abstract_concept_becomes_concept_via_ambiguous_llm_result(self):
        text = "Improving Training Data Quality remains an open challenge for the team."
        classifier = _FakeDatasetFamilyClassifier(response={"label": "Ambiguous", "confidence": "ambiguous"})
        entity = self._run(text, "Training Data Quality", "Dataset", classifier)
        self.assertEqual(entity.entity_type, "Concept")
        self.assertEqual(entity.metadata.get("dataset_family_typing_reason"), "llm_classification_ambiguous")

    def test_confident_label_matching_confidence_low_still_abstains(self):
        # A returned confident-sounding label with confidence="ambiguous"
        # must still abstain to Concept -- confidence governs, not label.
        text = "Some data was referenced without further detail."
        classifier = _FakeDatasetFamilyClassifier(response={"label": "Dataset", "confidence": "ambiguous"})
        entity = self._run(text, "Some data", "Dataset", classifier)
        self.assertEqual(entity.entity_type, "Concept")

    def test_no_classifier_available_falls_back_to_conservative_abstention(self):
        text = "Regional Housing Archive appears in this record with no further description."
        entity = self._run(text, "Regional Housing Archive", "Dataset", classifier=None)
        self.assertEqual(entity.entity_type, "Concept")
        self.assertEqual(entity.metadata.get("dataset_family_typing_reason"), "no_confident_signal_abstained_no_classifier")

    def test_neighboring_bullet_record_count_does_not_auto_confirm_dataset(self):
        # The unrelated neighbouring bullet's record count must not
        # automatically win -- the (fake) LLM is the one making the call,
        # and here it correctly recognises the mention as a repository.
        text = (
            "* Primary: Open City Data Portal\n"
            "* URL: https://example.org/portal\n"
            "* Content: 48,000 records collected from public housing applications\n"
        )
        classifier = _FakeDatasetFamilyClassifier(response={"label": "Resource", "confidence": "confident"})
        entity = self._run(text, "Open City Data Portal", "Dataset", classifier)
        self.assertEqual(entity.entity_type, "Resource")
        # The block context (containing the neighbouring record count) was
        # still made available to the classifier as informational input...
        self.assertIn("48,000 records", classifier.calls[0]["linked_context"])
        # ...but did not by itself force a Dataset outcome.
        self.assertNotEqual(entity.entity_type, "Dataset")

    def test_structurally_linked_continuation_describing_same_dataset_is_passed_as_context(self):
        text = (
            "* Primary: Regional Housing Archive\n"
            "* URL: https://example.org/archive\n"
            "* Content: 48,000 records collected from public housing applications\n"
        )
        classifier = _FakeDatasetFamilyClassifier(response={"label": "Dataset", "confidence": "confident"})
        entity = self._run(text, "Regional Housing Archive", "Dataset", classifier)
        self.assertEqual(entity.entity_type, "Dataset")
        self.assertIn("48,000 records", classifier.calls[0]["linked_context"])

    def test_same_surface_different_context_can_receive_different_types(self):
        def _responder(*, surface, sentence, linked_context, proposed_type):
            if "executable code" in sentence.lower():
                return {"label": "SoftwareProduct", "confidence": "confident"}
            return {"label": "Dataset", "confidence": "confident"}

        classifier = _FakeDatasetFamilyClassifier(responder=_responder)

        text_a = "Meridian was released as executable code for cleaning values."
        entity_a = self._run(text_a, "Meridian", "Dataset", classifier)
        self.assertEqual(entity_a.entity_type, "SoftwareProduct")

        text_b = "Meridian was released as an open collection of annotated records."
        entity_b = self._run(text_b, "Meridian", "Dataset", classifier)
        self.assertEqual(entity_b.entity_type, "Dataset")


class WikidataConsistencyTests(unittest.TestCase):
    """Contradiction may only redirect an entity still
    literally typed Dataset; corroboration may only restore Dataset from
    an ambiguous Concept state. Neither may override a DIFFERENT
    confident classification."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    @staticmethod
    def _entity(entity_type, original_type=None):
        meta = {}
        if original_type:
            meta["dataset_family_original_type"] = original_type
        return Entity(uri="https://kg/e/x", label="X", entity_type=entity_type, confidence=0.9, metadata=meta)

    def test_contradiction_detected_for_dataset_with_software_description(self):
        result = self.pipeline._dataset_family_wikidata_contradiction(
            entity=self._entity("Dataset"), wikidata_description="a spreadsheet software application",
        )
        self.assertEqual(result, ("SoftwareProduct", "wikidata_description_contradicts_dataset"))

    def test_wrong_wikidata_match_cannot_override_confident_non_dataset_context(self):
        # Entity was confidently classified ResearchStudy by context; an
        # (incorrect) Wikidata match suggesting software must not override it.
        result = self.pipeline._dataset_family_wikidata_contradiction(
            entity=self._entity("ResearchStudy", original_type="Dataset"),
            wikidata_description="a spreadsheet software application",
        )
        self.assertIsNone(result)

    def test_corroboration_restores_dataset_from_ambiguous_concept(self):
        result = self.pipeline._dataset_family_wikidata_contradiction(
            entity=self._entity("Concept", original_type="Dataset"),
            wikidata_description="a public database of housing records",
        )
        self.assertEqual(result, ("Dataset", "wikidata_description_corroborates_dataset"))

    def test_corroboration_strengthens_but_is_not_sole_determinant(self):
        # Corroboration only ever fires from an ambiguous (Concept) state
        # -- it cannot by itself promote a non-ambiguous, non-Dataset type.
        result = self.pipeline._dataset_family_wikidata_contradiction(
            entity=self._entity("SoftwareProduct", original_type="Dataset"),
            wikidata_description="a dataset and software toolkit",
        )
        self.assertIsNone(result)

    def test_non_dataset_family_origin_never_touched(self):
        result = self.pipeline._dataset_family_wikidata_contradiction(
            entity=self._entity("AISystem"), wikidata_description="a spreadsheet software application",
        )
        self.assertIsNone(result)


class EndToEndWikidataIntegrationTests(unittest.TestCase):
    """Full _process_entities_batch runs combining the LLM classifier
    escalation with the later Wikidata consistency pass, confirming
    type/ontology_uri/uri stay atomically consistent throughout."""

    def test_wikidata_contradiction_after_llm_ambiguous_reconciles_uri(self):
        text = "Zephyr Beta was mentioned briefly during the meeting without further detail."
        label = "Zephyr Beta"
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, "Dataset", start, end)])
        classifier = _FakeDatasetFamilyClassifier(response={"label": "Ambiguous", "confidence": "ambiguous"})
        linker = _FakeWikidataLinker(description="a spreadsheet software application")
        pipeline = _make_pipeline(
            entity_extractor=extractor, wikidata_linker=linker, config=_ConfigWithWikidata,
            dataset_family_classifier=classifier,
        )
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d3", "field": "theme.description"}])
        entity = results[0][0]
        # Ambiguous LLM result -> Concept; Wikidata contradiction only
        # applies to entities still literally typed Dataset, so it must
        # NOT fire here (Concept is not "Dataset").
        self.assertEqual(entity.entity_type, "Concept")

    def test_wikidata_contradiction_after_deterministic_dataset_reconciles_uri(self):
        text = "Zephyr Beta contains 40,000 records collected from public housing applications."
        label = "Zephyr Beta"
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, "Dataset", start, end)])
        classifier = _FakeDatasetFamilyClassifier(response={"label": "Dataset", "confidence": "confident"})
        linker = _FakeWikidataLinker(description="a spreadsheet software application")
        pipeline = _make_pipeline(
            entity_extractor=extractor, wikidata_linker=linker, config=_ConfigWithWikidata,
            dataset_family_classifier=classifier,
        )
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d4", "field": "theme.description"}])
        entity = results[0][0]
        self.assertEqual(entity.entity_type, "SoftwareProduct")
        self.assertIn("softwareproduct", str(entity.uri).lower())
        self.assertIsNotNone(entity.ontology_uri)


if __name__ == "__main__":
    unittest.main()
