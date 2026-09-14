import unittest

from utils.entity_normalizer import EntityNormalizer


class EntityNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = EntityNormalizer()

    def test_normalize_label_strips_leading_list_markers(self):
        cases = {
            "1) AI systems": "AI system",
            "1. AI systems": "AI system",
            "(1) AI systems": "AI system",
            "A) Fairness Algorithms": "Fairness Algorithms",
            "- Minority communities": "Minority Communities",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.normalizer.normalize_label(raw), expected)

    def test_normalize_label_preserves_legitimate_leading_numerals(self):
        self.assertEqual(self.normalizer.normalize_label("3M Sensor"), "3M Sensor")


if __name__ == "__main__":
    unittest.main()
