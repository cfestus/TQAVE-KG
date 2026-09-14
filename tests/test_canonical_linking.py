import tempfile
import unittest
from pathlib import Path

from models import Entity
from utils.entity_consolidator import EntityConsolidator
from utils.entity_normalizer import EntityNormalizer


class _Registry:
    def __init__(self):
        self.redirected = {}

    def replace_entity(self, old_uri: str, new_uri: str):
        self.redirected[old_uri] = new_uri


class CanonicalLinkingTests(unittest.TestCase):
    def setUp(self):
        self.norm = EntityNormalizer()
        # Test-isolation fix: EntityConsolidator.consolidate() writes
        # alias_map.json/merge_log.jsonl via self._write_artifacts(). Without
        # an explicit output_dir, it falls back to Config.OUTPUT_DIR and
        # silently overwrites the real production output/ directory's
        # artifacts with this test's synthetic data.
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    def test_alias_expansion(self):
        self.assertEqual(self.norm.normalize_label("HMDA"), "Home Mortgage Disclosure Act")
        self.assertEqual(self.norm.normalize_label("Apple INc"), "Apple Inc")
        self.assertEqual(self.norm.normalize_label("mimic cxr"), "MIMIC-CXR")
        self.assertEqual(self.norm.normalize_label("COMPAS"), "COMPAS")
        self.assertEqual(self.norm.normalize_label("UNESCO"), "UNESCO AI Ethics Recommendation")

    def test_do_not_merge_guard(self):
        self.assertTrue(self.norm.should_avoid_fuzzy_merge("AI"))
        self.assertFalse(self.norm.should_avoid_fuzzy_merge("AI Act"))

    def test_consolidator_merges_hmda_variants(self):
        reg = _Registry()
        cons = EntityConsolidator(registry=reg, normalizer=self.norm, output_dir=self.output_dir)
        e1 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_hmda_a",
            label="HMDA",
            entity_type="RegulatoryReference",
            confidence=0.8,
            metadata={},
        )
        e2 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_hmda_b",
            label="Home Mortgage Disclosure Act",
            entity_type="RegulatoryReference",
            confidence=0.9,
            metadata={},
        )
        out = cons.consolidate([e1, e2])
        self.assertEqual(len(out), 1)
        self.assertTrue(cons.redirects)

    def test_ai_not_merged_with_ai_act(self):
        reg = _Registry()
        cons = EntityConsolidator(registry=reg, normalizer=self.norm, output_dir=self.output_dir)
        e1 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_ai",
            label="AI",
            entity_type="EthicalPrinciple",
            confidence=0.9,
            metadata={},
        )
        e2 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_ai_act",
            label="AI Act",
            entity_type="RegulatoryReference",
            confidence=0.9,
            metadata={},
        )
        out = cons.consolidate([e1, e2])
        self.assertEqual(len(out), 2)

    def test_generic_terms_not_fuzzily_merged_into_specific_entities(self):
        reg = _Registry()
        cons = EntityConsolidator(registry=reg, normalizer=self.norm, output_dir=self.output_dir)
        dataset = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_dataset",
            label="Dataset",
            entity_type="Dataset",
            confidence=0.9,
            metadata={},
        )
        specific_dataset = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_dataset_specific",
            label="NIH Chest X-ray Dataset",
            entity_type="Dataset",
            confidence=0.9,
            metadata={},
        )
        fairness = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_fairness",
            label="Fairness",
            entity_type="Fairness",
            confidence=0.9,
            metadata={},
        )
        fairness_algorithms = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_fairness_algorithms",
            label="Fairness Algorithms",
            entity_type="Fairness",
            confidence=0.9,
            metadata={},
        )

        out = cons.consolidate([dataset, specific_dataset, fairness, fairness_algorithms])

        self.assertEqual(len(out), 4)
        self.assertFalse(cons.redirects)

    def test_consolidator_merges_unesco_variants(self):
        reg = _Registry()
        cons = EntityConsolidator(registry=reg, normalizer=self.norm, output_dir=self.output_dir)
        e1 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_unesco_a",
            label="UNESCO",
            entity_type="PolicyFramework",
            confidence=0.8,
            metadata={},
        )
        e2 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_unesco_b",
            label="UNESCO AI Ethics Recommendation",
            entity_type="PolicyFramework",
            confidence=0.9,
            metadata={},
        )
        out = cons.consolidate([e1, e2])
        self.assertEqual(len(out), 1)

    def test_consolidator_merges_apple_typo(self):
        reg = _Registry()
        cons = EntityConsolidator(registry=reg, normalizer=self.norm, output_dir=self.output_dir)
        e1 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_apple_a",
            label="Apple INc",
            entity_type="Organisation",
            confidence=0.7,
            metadata={},
        )
        e2 = Entity(
            uri="https://alfie-project.eu/etd-hub#entity_apple_b",
            label="Apple Inc",
            entity_type="Organisation",
            confidence=0.9,
            metadata={},
        )
        out = cons.consolidate([e1, e2])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()

