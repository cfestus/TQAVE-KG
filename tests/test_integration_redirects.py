import unittest

from models import AnnotationBatch, ProcessingResult, Relation
from pipeline import EnhancedSemanticPipeline


class RedirectIntegrationTests(unittest.TestCase):
    def test_redirects_rewrite_relation_endpoints(self):
        ann = AnnotationBatch(
            themes={
                "t1": ProcessingResult(
                    entities=[],
                    relations=[
                        Relation(
                            source="https://alfie-project.eu/etd-hub#old_entity",
                            relation="mentions",
                            target="https://alfie-project.eu/etd-hub#old_target",
                            confidence=0.8,
                            metadata={},
                        )
                    ],
                    metadata={},
                )
            }
        )

        redirects = {
            "https://alfie-project.eu/etd-hub#old_entity": "https://alfie-project.eu/etd-hub#new_entity",
            "https://alfie-project.eu/etd-hub#old_target": "https://alfie-project.eu/etd-hub#new_target",
        }

        p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        p._apply_entity_redirects_to_annotations(ann, redirects)

        rel = ann.themes["t1"].relations[0]
        self.assertEqual(rel.source, "https://alfie-project.eu/etd-hub#new_entity")
        self.assertEqual(rel.target, "https://alfie-project.eu/etd-hub#new_target")


if __name__ == "__main__":
    unittest.main()


