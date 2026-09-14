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


class _RepeatedMentionEntityExtractor:
    """Simulates a compliant LLMEntityExtractor: returns ONE Entity for a
    surface form that occurs twice in the chunk, with both chunk-local
    occurrences recorded in metadata["spans"] (exactly what
    LLMEntityExtractor._convert_to_entities produces)."""

    def __init__(self, label, entity_type, spans):
        self.label = label
        self.entity_type = entity_type
        self.spans = spans

    def extract(self, text, doc_id=None, context=None):
        first_start, first_end = self.spans[0]
        return [
            Entity(
                uri=f"{Config.NAMESPACES['etd']}placeholder",
                label=self.label,
                entity_type=self.entity_type,
                confidence=0.9,
                metadata={
                    "source": "llm",
                    "surface_form": self.label,
                    "start_char": first_start,
                    "end_char": first_end,
                    "spans": list(self.spans),
                    "ontology_uri": "",
                },
            )
        ]


def _make_pipeline(entity_extractor):
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.entity_extractor = entity_extractor
    pipeline.wikidata_linker = None
    pipeline._discourse_chunk_spans = {}
    return pipeline


class EntityMentionTrackingTests(unittest.TestCase):
    def setUp(self):
        self.text = (
            "Black patients face bias in this system. The report also notes "
            "that Black patients wait longer for care."
        )
        # Offsets are computed against the raw text's actual casing ("Black
        # patients"); the entity_normalizer title-cases the label afterward
        # to "Black Patients", which is what the assertions below check.
        self.first_start = self.text.find("Black patients")
        self.first_end = self.first_start + len("Black patients")
        self.second_start = self.text.rfind("Black patients")
        self.second_end = self.second_start + len("Black patients")
        self.assertNotEqual(self.first_start, self.second_start, "fixture must contain two distinct occurrences")

        extractor = _RepeatedMentionEntityExtractor(
            label="Black patients",
            entity_type="Population",
            spans=[(self.first_start, self.first_end), (self.second_start, self.second_end)],
        )
        self.pipeline = _make_pipeline(extractor)
        self.metadata = {"discourse_id": "d1", "field": "theme.description"}

        results = self.pipeline._process_entities_batch([self.text], [self.metadata])
        self.assertEqual(len(results), 1)
        self.entities = results[0]

    def test_no_duplicate_entity_node_is_created(self):
        # Exactly one entity node for "Black Patients", not one per occurrence.
        matching = [e for e in self.entities if e.label == "Black Patients"]
        self.assertEqual(len(matching), 1)

    def test_both_occurrences_are_recorded_in_metadata_mentions(self):
        entity = next(e for e in self.entities if e.label == "Black Patients")
        mentions = entity.metadata.get("mentions")
        self.assertIsInstance(mentions, list)
        self.assertEqual(len(mentions), 2)
        spans_found = sorted((m["start_char"], m["end_char"]) for m in mentions)
        expected = sorted([(self.first_start, self.first_end), (self.second_start, self.second_end)])
        self.assertEqual(spans_found, expected)

    def test_entity_is_locally_available_in_both_containing_units(self):
        entity = next(e for e in self.entities if e.label == "Black Patients")

        unit_one = {"start_offset": 0, "end_offset": self.first_end + 5}
        unit_two = {"start_offset": self.second_start - 5, "end_offset": len(self.text)}

        local_one = self.pipeline._local_entities_for_unit([entity], unit_one, "d1")
        local_two = self.pipeline._local_entities_for_unit([entity], unit_two, "d1")

        self.assertIn(entity.uri, {str(getattr(e, "uri", "")) for e in local_one})
        self.assertIn(entity.uri, {str(getattr(e, "uri", "")) for e in local_two})

    def test_unit_not_overlapping_either_occurrence_does_not_get_the_entity(self):
        entity = next(e for e in self.entities if e.label == "Black Patients")
        # A unit strictly between the two occurrences, overlapping neither.
        gap_unit = {"start_offset": self.first_end + 1, "end_offset": self.second_start - 1}
        local = self.pipeline._local_entities_for_unit([entity], gap_unit, "d1")
        self.assertNotIn(entity.uri, {str(getattr(e, "uri", "")) for e in local})


if __name__ == "__main__":
    unittest.main()
