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


class _FakePopulationGroupClassifier:
    """Test double for extractors.population_group_classifier.
    PopulationGroupTypeClassifier. Same call signature as the real
    classifier. Either returns a fixed {"label", "confidence"} dict, or --
    if constructed with a callable -- delegates to it so a test can assert
    on what was actually passed in and vary the response per call."""

    def __init__(self, response=None, responder=None):
        self._response = response
        self._responder = responder
        self.calls = []

    def classify(self, *, surface, sentence, proposed_type=""):
        self.calls.append({"surface": surface, "sentence": sentence, "proposed_type": proposed_type})
        if self._responder is not None:
            return self._responder(surface=surface, sentence=sentence, proposed_type=proposed_type)
        return dict(self._response or {"label": "Ambiguous", "confidence": "ambiguous"})


def _make_pipeline(entity_extractor=None, population_group_classifier=None):
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
    pipeline.population_group_classifier = population_group_classifier
    pipeline._discourse_chunk_spans = {}
    pipeline.semantic_policy = None
    return pipeline


class DeterministicPreFilterTests(unittest.TestCase):
    """_classify_population_group_deterministic is a cheap
    PRE-filter only -- it must never itself confirm a group from bare
    keyword presence, and must escalate anything it cannot confidently
    resolve as clause/comparison-shaped to the LLM classifier."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_clause_with_auxiliary_verb_resolved_as_outcome(self):
        new_type, reason = self.pipeline._classify_population_group_deterministic(
            label="was disproportionately affected by the decision"
        )
        self.assertEqual(new_type, "Outcome")
        self.assertEqual(reason, "clause_shaped_not_a_group")

    def test_comparison_marker_resolved_as_outcome(self):
        new_type, reason = self.pipeline._classify_population_group_deterministic(
            label="worse outcomes than other applicants"
        )
        self.assertEqual(new_type, "Outcome")
        self.assertEqual(reason, "comparison_or_disparity_marker_present")

    def test_clean_short_noun_phrase_escalates_to_llm(self):
        new_type, reason = self.pipeline._classify_population_group_deterministic(label="Rural Residents")
        self.assertIsNone(new_type)
        self.assertEqual(reason, "needs_llm_classification")

    def test_long_noun_phrase_with_no_clause_signal_escalates_to_llm(self):
        new_type, reason = self.pipeline._classify_population_group_deterministic(
            label="Low Income Single Parent Households With Young Children"
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
        pipeline = _make_pipeline(entity_extractor=extractor, population_group_classifier=classifier)
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])
        matching = [e for e in results[0] if e.label == label or e.metadata.get("quantity_literal_original_label") == label]
        return matching[0] if matching else results[0][0]

    def test_explicit_demographic_group_confirmed(self):
        text = "Rural Residents reported longer wait times for approval."
        classifier = _FakePopulationGroupClassifier(response={"label": "DemographicGroup", "confidence": "confident"})
        entity = self._run(text, "Rural Residents", "PopulationGroup", classifier)
        self.assertEqual(entity.entity_type, "DemographicGroup")
        self.assertEqual(len(classifier.calls), 1)

    def test_generic_study_population_confirmed_as_populationgroup(self):
        text = "The Study Participants completed a follow-up survey after six months."
        classifier = _FakePopulationGroupClassifier(response={"label": "PopulationGroup", "confidence": "confident"})
        entity = self._run(text, "The Study Participants", "PopulationGroup", classifier)
        self.assertEqual(entity.entity_type, "PopulationGroup")

    def test_rate_or_disparity_about_a_group_not_demographicgroup(self):
        text = "Longer Wait Times Were Reported for this cohort throughout the review."
        classifier = _FakePopulationGroupClassifier(response={"label": "Outcome", "confidence": "confident"})
        entity = self._run(text, "Longer Wait Times Were Reported", "DemographicGroup", classifier)
        self.assertNotEqual(entity.entity_type, "DemographicGroup")
        self.assertNotEqual(entity.entity_type, "PopulationGroup")

    def test_clause_containing_group_noun_not_automatically_group(self):
        text = "Applicants Frequently Experience Delays during the review process."
        classifier = _FakePopulationGroupClassifier(response={"label": "Outcome", "confidence": "confident"})
        entity = self._run(text, "Applicants Frequently Experience Delays", "PopulationGroup", classifier)
        self.assertNotIn(entity.entity_type, {"PopulationGroup", "DemographicGroup"})

    def test_abstract_use_of_group_noun_not_population(self):
        text = "Community was the central theme explored throughout the discussion."
        classifier = _FakePopulationGroupClassifier(response={"label": "Concept", "confidence": "confident"})
        entity = self._run(text, "Community", "PopulationGroup", classifier)
        self.assertEqual(entity.entity_type, "Concept")

    def test_participant_population_typed_populationgroup_where_context_supports(self):
        # A plural common noun already proposed as population/group-family
        # by the upstream typing stage, whose local context genuinely
        # supports "collection of members" (attended/enrolled/participated),
        # is confirmed as PopulationGroup rather than downgraded.
        text = "Enrolled Participants attended every session of the programme."
        classifier = _FakePopulationGroupClassifier(response={"label": "PopulationGroup", "confidence": "confident"})
        entity = self._run(text, "Enrolled Participants", "PopulationGroup", classifier)
        self.assertEqual(entity.entity_type, "PopulationGroup")

    def test_ambiguous_llm_result_falls_back_to_concept(self):
        text = "Group A Members were mentioned briefly without further detail."
        classifier = _FakePopulationGroupClassifier(response={"label": "PopulationGroup", "confidence": "ambiguous"})
        entity = self._run(text, "Group A Members", "PopulationGroup", classifier)
        self.assertEqual(entity.entity_type, "Concept")

    def test_no_classifier_available_falls_back_to_conservative_concept(self):
        text = "Cohort Members took part in the follow-up round."
        entity = self._run(text, "Cohort Members", "DemographicGroup", None)
        self.assertEqual(entity.entity_type, "Concept")
        self.assertEqual(entity.metadata.get("population_group_typing_reason"), "no_confident_signal_abstained_no_classifier")

    def test_percentage_prefix_stripped_then_residual_group_classified(self):
        # The quantity gate strips the percentage first,
        # then the residual noun phrase alone reaches this classifier.
        text = "Roughly 42% Rural Residents responded to the outreach campaign."
        classifier = _FakePopulationGroupClassifier(response={"label": "DemographicGroup", "confidence": "confident"})
        entity = self._run(text, "42% Rural Residents", "PopulationGroup", classifier)
        self.assertEqual(entity.label, "Rural Residents")
        self.assertEqual(entity.entity_type, "DemographicGroup")
        self.assertEqual(entity.metadata.get("quantity_literal_value"), "42")
        self.assertEqual(classifier.calls[0]["surface"], "Rural Residents")

    def test_comparison_between_two_groups_both_remain_entities(self):
        text = "Rural Residents waited longer than Urban Residents for a decision."
        start_a = text.index("Rural Residents")
        end_a = start_a + len("Rural Residents")
        start_b = text.index("Urban Residents")
        end_b = start_b + len("Urban Residents")
        extractor = _FixedSpanEntityExtractor([
            ("Rural Residents", "PopulationGroup", start_a, end_a),
            ("Urban Residents", "PopulationGroup", start_b, end_b),
        ])
        classifier = _FakePopulationGroupClassifier(response={"label": "PopulationGroup", "confidence": "confident"})
        pipeline = _make_pipeline(entity_extractor=extractor, population_group_classifier=classifier)
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])
        labels = {e.label: e.entity_type for e in results[0]}
        self.assertEqual(labels.get("Rural Residents"), "PopulationGroup")
        self.assertEqual(labels.get("Urban Residents"), "PopulationGroup")


class TypeVariantNormalizationTests(unittest.TestCase):
    def test_type_variants_normalize_to_canonical_forms(self):
        norm = EntityNormalizer(GlobalEntityRegistry())
        for variant in ("Population", "PopulationGroup", "Population/Group", "Population-Group", "Population Group"):
            self.assertEqual(norm.normalize_type(variant), "PopulationGroup", variant)
        for variant in ("DemographicGroup", "Demographicgroup", "Demographic-Group"):
            self.assertEqual(norm.normalize_type(variant), "DemographicGroup", variant)


class OntologyGroundingTests(unittest.TestCase):
    def test_populationgroup_grounds_to_local_etd_class_not_prov_entity(self):
        from models import Entity as _E
        mapper = OntologyMapper()
        entity = _E(
            uri="https://alfie-project.eu/etd-hub#populationgroup_x", label="Some Cohort",
            entity_type="PopulationGroup", confidence=1.0, metadata={},
        )
        mapped = mapper.map_entity(entity)
        self.assertIn("etd:PopulationGroup", mapped)

    def test_demographicgroup_grounds_to_local_etd_class_not_prov_entity(self):
        from models import Entity as _E
        mapper = OntologyMapper()
        entity = _E(
            uri="https://alfie-project.eu/etd-hub#demographicgroup_x", label="Some Group",
            entity_type="DemographicGroup", confidence=1.0, metadata={},
        )
        mapped = mapper.map_entity(entity)
        self.assertIn("etd:DemographicGroup", mapped)


class UriTypeOntologyConsistencyTests(unittest.TestCase):
    def test_retyped_entity_has_consistent_uri_bucket_and_ontology(self):
        text = "Cohort Members took part in the follow-up round."
        label = "Cohort Members"
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, "DemographicGroup", start, end)])
        classifier = _FakePopulationGroupClassifier(response={"label": "PopulationGroup", "confidence": "confident"})
        pipeline = _make_pipeline(entity_extractor=extractor, population_group_classifier=classifier)
        results = pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])
        entity = results[0][0]
        self.assertEqual(entity.entity_type, "PopulationGroup")
        self.assertIn("populationgroup", entity.uri.lower())
        self.assertNotIn("demographicgroup", entity.uri.lower())


if __name__ == "__main__":
    unittest.main()
