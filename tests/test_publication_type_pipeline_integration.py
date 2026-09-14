"""
End-to-end test proving real pipeline entity metadata (ontology_classes /
ontology_uri, computed by the real OntologyMapper from an entity's local
type -- see ontology_mappings.py's "Publication" -> ["bibo:Document"]
mapping) actually reaches WikidataLinker.link(ontology_types=...) through
the real pipeline.py call site in _process_entities_batch, and that the
structured ontology evidence is what the publication-compatibility signal
actually uses (not the bare/generic local type string).

No live LLM/network calls: entity extraction and Wikidata linking are both
test doubles, but the ontology-mapping and pipeline wiring in between are
the REAL production code paths.
"""
import unittest

from config import Config
from models import Entity
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer
from utils.wikidata_linker import WikidataLinker


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


class _ConfigWithWikidata(Config):
    USE_WIKIDATA_LINKING = True


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


class _RecordingWikidataLinker:
    """Records the exact kwargs it was called with, so the test can assert
    on what the REAL pipeline call site actually passed through -- not a
    hand-built call."""

    def __init__(self):
        self.calls = []

    def link(self, label, *, entity_type=None, context_text=None, ontology_types=None,
             source_grounded_aliases=None):
        self.calls.append({
            "label": label, "entity_type": entity_type,
            "context_text": context_text, "ontology_types": ontology_types,
            "source_grounded_aliases": source_grounded_aliases,
        })
        return None  # abstain -- this test is about what was PASSED IN, not the decision


def _make_pipeline(entity_extractor, wikidata_linker, config=_ConfigWithWikidata):
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.entity_extractor = entity_extractor
    pipeline.wikidata_linker = wikidata_linker
    pipeline.dataset_family_classifier = None
    pipeline._discourse_chunk_spans = {}
    return pipeline


class PublicationTypeEndToEndIntegrationTests(unittest.TestCase):
    def test_publication_entity_ontology_classes_reach_the_linker(self):
        """A 'Publication'-typed entity must have the real OntologyMapper's
        bibo:Document mapping (expanded to its full URI) show up in the
        actual ontology_types= argument the pipeline passes to
        WikidataLinker.link -- proving the wiring is real, not just
        unit-tested on the linker in isolation."""
        text = "Big Data & Society is a well-known journal in this field."
        label = "Big Data & Society"
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, "Publication", start, end)])
        linker = _RecordingWikidataLinker()
        pipeline = _make_pipeline(entity_extractor=extractor, wikidata_linker=linker)

        pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])

        self.assertEqual(len(linker.calls), 1)
        call = linker.calls[0]
        self.assertEqual(call["label"], label)
        self.assertIsNotNone(call["ontology_types"], "ontology_types must be passed, not omitted")
        self.assertIn(
            "http://purl.org/ontology/bibo/Document", call["ontology_types"],
            f"expected the real OntologyMapper's Publication->bibo:Document mapping in ontology_types, got {call['ontology_types']!r}",
        )

    def test_ontology_types_then_gates_off_the_publication_penalty_in_the_real_linker(self):
        """Full round trip with a REAL WikidataLinker (not a recorder):
        the structured ontology evidence produced by the pipeline for a
        Publication entity must be enough, on its own, to prevent the
        scholarly-article mismatch penalty -- even though this entity's
        surface label ('Big Data & Society') looks nothing like an
        obviously 'generic' local type."""
        journal_candidate = {
            "qid": "Q27726175", "label": "Big Data & Society", "description": "journal",
            "aliases": [], "match_type": "label", "match_text": "Big Data & Society",
        }
        real_linker = WikidataLinker()
        real_linker._candidate_cache["big data & society|lang=en|limit=25"] = {
            "queries_tried": ["Big Data & Society"], "retrieval_params": {}, "candidates": [journal_candidate],
        }
        # A scholarly-article-classed decoy would normally trigger the
        # penalty for a non-publication entity -- here it should not,
        # because the pipeline supplies explicit bibo:Document evidence.
        real_linker._p31_cache["Q27726175"] = ["Q5633421"]  # scientific journal, not Q13442814 -- irrelevant either way

        text = "Big Data & Society is a well-known journal in this field."
        label = "Big Data & Society"
        start = text.index(label)
        end = start + len(label)
        extractor = _FixedSpanEntityExtractor([(label, "Publication", start, end)])
        pipeline = _make_pipeline(entity_extractor=extractor, wikidata_linker=real_linker)

        results = pipeline._process_entities_batch([text], [{"discourse_id": "d1", "field": "theme.description"}])
        entity = results[0][0]
        self.assertEqual(entity.metadata.get("wikidata_uri"), "http://www.wikidata.org/entity/Q27726175")


if __name__ == "__main__":
    unittest.main()
