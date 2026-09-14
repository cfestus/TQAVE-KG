import tempfile
import unittest
from pathlib import Path

from config import Config
from models import Entity
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.canonical_linker import CanonicalLinker
from utils.entity_consolidator import EntityConsolidator
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


class _RecordingWikidataLinker:
    def __init__(self):
        self.calls = []

    def link(self, label, *, entity_type=None, context_text=None, ontology_types=None,
             source_grounded_aliases=None):
        self.calls.append(label)
        return None


class _StubEntityExtractor:
    def __init__(self, by_chunk):
        self._by_chunk = by_chunk

    def extract(self, text, doc_id=None, context=None):
        return list(self._by_chunk.get(text, []))


class _TestConfig(Config):
    USE_WIKIDATA_LINKING = True
    USE_ONTOLOGY_MAPPING = False


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


def _unit(unit_id, text, start_offset, *, section_category="semantic", unit_type="sentence"):
    return {
        "unit_id": unit_id,
        "text": text,
        "start_offset": start_offset,
        "end_offset": start_offset + len(text),
        "section_category": section_category,
        "unit_type": unit_type,
    }


class CanonicalUriScopeTests(unittest.TestCase):
    """EntityNormalizer.canonical_uri: discourse_scope must change identity
    only when explicitly supplied -- genuine entities (no scope) are
    byte-for-byte unaffected."""

    def setUp(self):
        self.normalizer = EntityNormalizer(GlobalEntityRegistry())

    def test_same_label_different_scope_yields_different_uri(self):
        uri_a = self.normalizer.canonical_uri(label="it", entity_type="Concept", discourse_scope="answer.description|11")
        uri_b = self.normalizer.canonical_uri(label="it", entity_type="Concept", discourse_scope="answer.description|29")
        self.assertNotEqual(uri_a, uri_b)

    def test_no_scope_is_backward_compatible(self):
        with_none = self.normalizer.canonical_uri(label="Model Alpha", entity_type="AISystem", discourse_scope=None)
        without_param = self.normalizer.canonical_uri(label="Model Alpha", entity_type="AISystem")
        self.assertEqual(with_none, without_param)

    def test_same_scope_same_label_yields_same_uri(self):
        uri_a = self.normalizer.canonical_uri(label="this disagreement", entity_type="Concept", discourse_scope="answer.description|11")
        uri_b = self.normalizer.canonical_uri(label="this disagreement", entity_type="Concept", discourse_scope="answer.description|11")
        self.assertEqual(uri_a, uri_b)


def _scoped_entity(uri, label, discourse_id, origin_key, entity_type="Concept"):
    return Entity(
        uri=uri, label=label, entity_type=entity_type, confidence=0.85,
        metadata={"discourse_id": discourse_id, "origin_key": origin_key, "discourse_scoped_identity": True},
    )


def _genuine_entity(uri, label, discourse_id, origin_key, entity_type="AISystem"):
    return Entity(
        uri=uri, label=label, entity_type=entity_type, confidence=0.85,
        metadata={"discourse_id": discourse_id, "origin_key": origin_key},
    )


class ConsolidatorScopeTests(unittest.TestCase):
    def setUp(self):
        self.registry = GlobalEntityRegistry()
        self.normalizer = EntityNormalizer(self.registry)
        # Test-isolation fix: output_dir="output" resolved to the real
        # production output/ directory (relative to CWD), so
        # consolidate()'s _write_artifacts() overwrote its live
        # alias_map.json/merge_log.jsonl with this test's synthetic data.
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.consolidator = EntityConsolidator(self.registry, self.normalizer, output_dir=Path(self._tmpdir.name))

    def test_unresolved_referential_surface_not_merged_across_discourses(self):
        a = _scoped_entity("https://kg/e/it_d1", "it", "d1", "field|d1")
        b = _scoped_entity("https://kg/e/it_d2", "it", "d2", "field|d2")
        result = self.consolidator.consolidate([a, b])
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0].uri, result[1].uri)

    def test_unresolved_demonstrative_phrase_not_merged_across_discourses(self):
        a = _scoped_entity("https://kg/e/disagree_d1", "this disagreement", "d1", "field|d1")
        b = _scoped_entity("https://kg/e/disagree_d2", "this disagreement", "d2", "field|d2")
        result = self.consolidator.consolidate([a, b])
        self.assertEqual(len(result), 2)

    def test_genuine_named_entities_still_merge_across_discourses(self):
        a = _genuine_entity("https://kg/e/model_alpha_d1", "Model Alpha", "d1", "field|d1")
        b = _genuine_entity("https://kg/e/model_alpha_d2", "Model Alpha", "d2", "field|d2")
        result = self.consolidator.consolidate([a, b])
        self.assertEqual(len(result), 1)


class CanonicalLinkerScopeTests(unittest.TestCase):
    def setUp(self):
        self.registry = GlobalEntityRegistry()
        # Test-isolation fix: see ConsolidatorScopeTests.setUp above --
        # output_dir="output" resolved to the real production output/
        # directory and overwrote its canonical_linking_log.jsonl.
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.linker = CanonicalLinker(self.registry, output_dir=Path(self._tmpdir.name))

    def test_unresolved_referential_surface_not_merged_across_discourses(self):
        a = _scoped_entity("https://kg/e/it_d1", "it", "d1", "field|d1")
        b = _scoped_entity("https://kg/e/it_d2", "it", "d2", "field|d2")
        linked = self.linker.link_entities([a, b])
        self.assertEqual(len(linked), 2)
        self.assertNotEqual(linked[0].uri, linked[1].uri)

    def test_unresolved_demonstrative_phrase_not_merged_across_discourses(self):
        a = _scoped_entity("https://kg/e/tests_d1", "these tests", "d1", "field|d1")
        b = _scoped_entity("https://kg/e/tests_d2", "these tests", "d2", "field|d2")
        linked = self.linker.link_entities([a, b])
        self.assertEqual(len(linked), 2)
        self.assertNotEqual(linked[0].uri, linked[1].uri)

    def test_genuine_named_entities_still_merge_across_discourses(self):
        a = _genuine_entity("https://kg/e/model_alpha_d1", "Model Alpha", "d1", "field|d1")
        b = _genuine_entity("https://kg/e/model_alpha_d2", "Model Alpha", "d2", "field|d2")
        linked = self.linker.link_entities([a, b])
        self.assertEqual(len(linked), 1)

    def test_resolved_pronoun_uses_shared_genuine_entity_uri(self):
        # A pronoun resolved via coreference resolution to a real antecedent never creates
        # a NEW entity at all -- the relation just points at the real
        # entity's own URI, which is unaffected by scope and merges
        # normally, as demonstrated here directly.
        real_d1 = _genuine_entity("https://kg/e/screening_tool_d1", "Screening Tool", "d1", "field|d1")
        real_d2 = _genuine_entity("https://kg/e/screening_tool_d2", "Screening Tool", "d2", "field|d2")
        linked = self.linker.link_entities([real_d1, real_d2])
        self.assertEqual(len(linked), 1)


class WikidataLinkingScopeTests(unittest.TestCase):
    def test_unresolved_referential_expression_not_wikidata_linked(self):
        pipeline = _make_pipeline()
        pipeline.config = _TestConfig
        recorder = _RecordingWikidataLinker()
        pipeline.wikidata_linker = recorder
        pipeline.entity_extractor = _StubEntityExtractor({
            "It affects outcomes broadly.": [
                Entity(uri="https://tmp/it", label="it", entity_type="Concept", confidence=0.8, metadata={}),
            ],
            "Model Alpha was deployed last year.": [
                Entity(uri="https://tmp/model_alpha", label="Model Alpha", entity_type="AISystem", confidence=0.9, metadata={}),
            ],
        })

        texts = ["It affects outcomes broadly.", "Model Alpha was deployed last year."]
        metadata_list = [
            {"discourse_id": "d1", "field": "answer.description", "origin_key": "answer.description|d1"},
            {"discourse_id": "d2", "field": "answer.description", "origin_key": "answer.description|d2"},
        ]
        entities_list = pipeline._process_entities_batch(texts, metadata_list)

        self.assertNotIn("it", [c.lower() for c in recorder.calls])
        self.assertIn("Model Alpha", recorder.calls)

        flat = [e for group in entities_list for e in group]
        it_entities = [e for e in flat if str(getattr(e, "label", "")).strip().lower() == "it"]
        self.assertTrue(it_entities)
        self.assertTrue(it_entities[0].metadata.get("discourse_scoped_identity"))
        model_entities = [e for e in flat if getattr(e, "label", "") == "Model Alpha"]
        self.assertTrue(model_entities)
        self.assertFalse(model_entities[0].metadata.get("discourse_scoped_identity"))


class PropositionCrossDiscourseScopeTests(unittest.TestCase):
    def test_proposition_stable_within_provenance_distinct_across_discourses(self):
        pipeline = _make_pipeline()
        unit = _unit("Body:8", "Defendants cannot meaningfully challenge the system.", 0)

        first = pipeline._get_or_create_unit_proposition_entity(
            unit=unit, discourse_id="d1", origin_key="answer.description|d1",
            field="answer.description", derived_from_relation_uris=[],
        )
        again = pipeline._get_or_create_unit_proposition_entity(
            unit=unit, discourse_id="d1", origin_key="answer.description|d1",
            field="answer.description", derived_from_relation_uris=[],
        )
        self.assertEqual(first.uri, again.uri, "same structural provenance must be stable")

        other_discourse = pipeline._get_or_create_unit_proposition_entity(
            unit=unit, discourse_id="d2", origin_key="answer.description|d2",
            field="answer.description", derived_from_relation_uris=[],
        )
        self.assertNotEqual(first.uri, other_discourse.uri, "same unit_id in a different discourse must not collide")
        self.assertTrue(first.metadata.get("discourse_scoped_identity"))

    def test_proposition_identity_survives_canonical_linking_scoped(self):
        # The two Propositions above share the SAME label ("Proposition
        # Body:8") but different origin_key scope -- CanonicalLinker must
        # keep them distinct rather than merging by label alone.
        registry = GlobalEntityRegistry()
        # Test-isolation fix: see ConsolidatorScopeTests.setUp above.
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        linker = CanonicalLinker(registry, output_dir=Path(tmpdir.name))
        prop_d1 = Entity(
            uri="https://kg/e/prop_d1", label="Proposition Body:8", entity_type="Proposition", confidence=1.0,
            metadata={"discourse_id": "d1", "origin_key": "answer.description|d1", "discourse_scoped_identity": True},
        )
        prop_d2 = Entity(
            uri="https://kg/e/prop_d2", label="Proposition Body:8", entity_type="Proposition", confidence=1.0,
            metadata={"discourse_id": "d2", "origin_key": "answer.description|d2", "discourse_scoped_identity": True},
        )
        linked = linker.link_entities([prop_d1, prop_d2])
        self.assertEqual(len(linked), 2)
        self.assertNotEqual(linked[0].uri, linked[1].uri)


if __name__ == "__main__":
    unittest.main()
