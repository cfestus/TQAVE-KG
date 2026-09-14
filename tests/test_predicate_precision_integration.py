import unittest

from models import Entity, Relation
from utils.predicate_precision import PredicatePrecisionLayer

try:
    from ontology.mapper import OntologyMapper
    MAPPER_AVAILABLE = True
except Exception:
    OntologyMapper = None
    MAPPER_AVAILABLE = False


@unittest.skipUnless(MAPPER_AVAILABLE, "OntologyMapper import/use unavailable")
class PredicatePrecisionIntegrationTests(unittest.TestCase):
    def test_precision_then_enrichment_preserves_predicate(self):
        layer = PredicatePrecisionLayer()
        mapper = OntologyMapper()

        entities = [
            Entity(
                uri="https://alfie-project.eu/etd-hub#expert_1",
                label="Expert 1",
                entity_type="Expert",
                confidence=1.0,
                metadata={},
            ),
            Entity(
                uri="https://alfie-project.eu/etd-hub#policy_1",
                label="EU AI Act",
                entity_type="PolicyFramework",
                confidence=1.0,
                metadata={},
            ),
        ]

        rel = Relation(
            source=entities[0].uri,
            relation="criticises",
            target=entities[1].uri,
            confidence=0.72,
            metadata={"doc_id": "d1", "evidence": "Experts criticise the policy."},
        )

        refined = layer.refine_relations([rel], entities)
        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].relation, "criticizes")

        enrichment = mapper.enrich_relation(refined[0])
        has_enrichment = isinstance(enrichment, dict) and bool(enrichment)
        self.assertTrue(has_enrichment or bool(refined[0].ontology_uri))

        if has_enrichment:
            refined[0].metadata.update(enrichment)

        # Enrichment must not mutate canonical predicate label.
        self.assertEqual(refined[0].relation, "criticizes")


if __name__ == "__main__":
    unittest.main()

