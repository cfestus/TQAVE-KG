import unittest
import importlib.util

HAS_REQUESTS = importlib.util.find_spec("requests") is not None

@unittest.skipUnless(HAS_REQUESTS, "requests is required for reasoner module import")
class ConflictDetectorNestedTests(unittest.TestCase):
    def setUp(self):
        from reasoning.reasoner import ConflictDetector
        self.detector = ConflictDetector()

    def test_detects_conflicts_in_nested_annotations(self):
        annotations = {
            "themes": {
                "t1": {
                    "relations": [
                        {
                            "source": "https://alfie-project.eu/etd-hub#entity_a",
                            "relation": "supports",
                            "target": "https://alfie-project.eu/etd-hub#entity_b",
                        },
                        {
                            "source": "https://alfie-project.eu/etd-hub#entity_a",
                            "relation": "opposes",
                            "target": "https://alfie-project.eu/etd-hub#entity_b",
                        },
                    ]
                }
            },
            "questions": {},
            "answers": {},
            "votes": {},
            "metadata": {},
        }

        conflicts = self.detector.detect(annotations)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["source"], "https://alfie-project.eu/etd-hub#entity_a")
        self.assertEqual(conflicts[0]["target"], "https://alfie-project.eu/etd-hub#entity_b")
        self.assertCountEqual(conflicts[0]["relations"], ["supports", "opposes"])

    def test_inferred_relations_are_included_in_conflict_scan(self):
        annotations = {
            "themes": {},
            "questions": {},
            "answers": {},
            "votes": {},
            "metadata": {
                "inferred_relations": [
                    {
                        "source": "https://alfie-project.eu/etd-hub#entity_x",
                        "relation": "compliesWith",
                        "target": "https://alfie-project.eu/etd-hub#entity_y",
                        "is_inferred": True,
                    },
                    {
                        "source": "https://alfie-project.eu/etd-hub#entity_x",
                        "relation": "violates",
                        "target": "https://alfie-project.eu/etd-hub#entity_y",
                        "is_inferred": True,
                    },
                ]
            },
        }

        conflicts = self.detector.detect(annotations)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["source"], "https://alfie-project.eu/etd-hub#entity_x")
        self.assertEqual(conflicts[0]["target"], "https://alfie-project.eu/etd-hub#entity_y")
        self.assertCountEqual(conflicts[0]["relations"], ["compliesWith", "violates"])


if __name__ == "__main__":
    unittest.main()

