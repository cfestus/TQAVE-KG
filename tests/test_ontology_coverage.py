import unittest

from models import Entity
from ontology.mapper import OntologyMapper


class OntologyCoverageTests(unittest.TestCase):
    def test_structural_entity_types_have_ontology_mapping(self):
        mapper = OntologyMapper()
        types_to_check = [
            "Theme",
            "Question",
            "Answer",
            "Vote",
            "Expert",
            "Document",
            "Resource",
            "DiscourseUnit",
        ]
        for etype in types_to_check:
            ent = Entity(
                uri=f"https://alfie-project.eu/etd-hub#{etype.lower()}_x",
                label=f"{etype} X",
                entity_type=etype,
                confidence=1.0,
                metadata={},
            )
            mapped = mapper.map_entity(ent)
            self.assertTrue(mapped, f"Expected ontology mapping for entity_type={etype}")


if __name__ == "__main__":
    unittest.main()

