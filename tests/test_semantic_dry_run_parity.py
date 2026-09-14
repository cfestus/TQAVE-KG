import tempfile
import unittest
from pathlib import Path

from rdflib import Namespace, RDF

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline, SEMANTIC_RELATION_PREDICATES
from scripts.semantic_policy_dry_run import collect_final_accepted_rows, make_entity_index, semantic_signature
from serializers.rdf_serializer import RDFSerializer
from utils.canonical_linker import CanonicalLinker
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


class _NoopCanonicalLinker:
    def __init__(self, registry):
        self.registry = registry
        self.redirects = {}

    def run_full_registry(self):
        get_all = getattr(self.registry, "get_all_entities", None)
        return list(get_all()) if callable(get_all) else []

    def rewrite_relation_endpoints(self, _relations):
        return None

    @staticmethod
    def get_report(pre_link_count, canonical_count):
        return {
            "raw_entities": int(pre_link_count),
            "canonical_entities": int(canonical_count),
            "merges_count": 0,
        }


class _NoopRelationExtractor:
    @staticmethod
    def extract(_text, _entities, discourse_id=None):
        return []


class _NoopEntityExtractor:
    @staticmethod
    def extract(_text, doc_id=None, context=None):
        return []


class _ConfirmingPopulationGroupClassifier:
    """Test double: confirms whatever type the deterministic
    pre-filter/upstream extraction already proposed, confidently. Stands in
    for a real PopulationGroupTypeClassifier call so this file's fixtures
    (which predate that classifier and were not written with it in
    mind) exercise the intended real-world behaviour -- where a genuine LLM
    call resolves a clear, short group-denoting noun phrase confidently --
    rather than the conservative no-classifier-available abstention that
    only exists as a safety net for pipelines built without one."""

    @staticmethod
    def classify(*, surface, sentence, proposed_type=""):
        return {"label": proposed_type, "confidence": "confident"}


def _make_pipeline():
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.canonical_linker = _NoopCanonicalLinker(pipeline.global_registry)
    pipeline.coref_resolver = _EchoResolver()
    pipeline.wikidata_linker = None
    pipeline.llm_relation_extractor = _NoopRelationExtractor()
    pipeline.dataset_family_classifier = None
    pipeline.population_group_classifier = _ConfirmingPopulationGroupClassifier()
    pipeline._discourse_chunk_spans = {}
    pipeline.semantic_policy = None
    return pipeline


def _entity(local_id, label, entity_type, discourse_id, section="Body", section_category="semantic"):
    return Entity(
        uri=f"{Config.NAMESPACES['etd']}{local_id}",
        label=label,
        entity_type=entity_type,
        confidence=1.0,
        metadata={
            "discourse_id": discourse_id,
            "section": section,
            "section_category": section_category,
        },
    )


def _serializer_signatures(graph):
    etd = Namespace(Config.NAMESPACES["etd"])
    rows = []
    for rel_node in graph.subjects(RDF.type, etd.Relation):
        status = str(graph.value(rel_node, etd.semantic_validation_status) or "").strip()
        if status != "accepted":
            continue
        relation_kind = str(graph.value(rel_node, etd.relation_kind) or "").strip()
        if relation_kind in {"structural", "provenance"}:
            continue
        pred_uri = graph.value(rel_node, etd.predicate)
        if pred_uri is None:
            continue
        predicate = str(pred_uri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        if predicate not in SEMANTIC_RELATION_PREDICATES:
            continue
        row = {
            "source_field": str(graph.value(rel_node, etd.field) or "").strip(),
            "source_id": str(graph.value(rel_node, etd.source_doc_id) or graph.value(rel_node, etd.discourse_id) or "").strip(),
            "subject_canonical_entity": str(graph.value(rel_node, etd.subject) or "").strip(),
            "canonical_predicate": predicate,
            "object_canonical_entity": str(graph.value(rel_node, etd.object) or "").strip(),
            "epistemic_status": str(graph.value(rel_node, etd.epistemic_status) or "").strip(),
            "relation_origin": str(graph.value(rel_node, etd.relation_origin) or "").strip(),
        }
        rows.append(semantic_signature(row))
    return set(rows)


class SemanticDryRunParityTests(unittest.TestCase):
    def setUp(self):
        self.prev_output_dir = Config.OUTPUT_DIR
        self.tmpdir = tempfile.TemporaryDirectory()
        Config.OUTPUT_DIR = Path(self.tmpdir.name)

    def tearDown(self):
        Config.OUTPUT_DIR = self.prev_output_dir
        self.tmpdir.cleanup()

    def test_dry_run_collector_matches_production_materialisation_signatures(self):
        pipeline = _make_pipeline()
        discourse_id = "11"
        fixture_entities = [
            _entity("historical_data_fixture", "Historical Data", "Dataset", discourse_id),
            _entity("systemic_bias_fixture", "Systemic Bias", "Bias", discourse_id),
            _entity("over_policing_fixture", "Over-Policing", "Bias", discourse_id),
            _entity("harsher_sentencing_fixture", "Harsher Sentencing", "Bias", discourse_id),
            _entity("fairness_fixture", "Fairness", "Fairness", discourse_id),
            _entity("accountability_fixture", "Accountability", "Governance", discourse_id),
            _entity("interpretability_fixture", "Interpretability", "Transparency", discourse_id),
            _entity("equity_fixture", "Equity", "Fairness", discourse_id),
        ]

        def _process_entities_batch(texts, metadata_list):
            batches = []
            for text, metadata in zip(texts, metadata_list):
                did = str(metadata.get("discourse_id") or discourse_id)
                entities = []
                for base in fixture_entities:
                    meta = dict(base.metadata)
                    meta["discourse_id"] = did
                    start = str(text).lower().find(str(base.label).lower())
                    if start >= 0:
                        end = start + len(str(base.label))
                        meta["start_char"] = start
                        meta["end_char"] = end
                        meta["mentions"] = [
                            {
                                "discourse_id": did,
                                "start_char": start,
                                "end_char": end,
                            }
                        ]
                    ent = Entity(
                        uri=base.uri,
                        label=base.label,
                        entity_type=base.entity_type,
                        confidence=base.confidence,
                        metadata=meta,
                    )
                    pipeline._ensure_entity_ontology_grounding(ent)
                    entities.append(pipeline.global_registry.get_or_create(ent))
                batches.append(entities)
            return batches

        pipeline._process_entities_batch = _process_entities_batch

        data = {
            "themes": [],
            "questions": [],
            "answers": [
                {
                    "id": discourse_id,
                    "question_id": None,
                    "theme_id": None,
                    "expert_id": None,
                    "description": (
                        "Historical Data reflects systemic bias, from over-policing in Black communities "
                        "to harsher sentencing for similar offenses. "
                        "Fairness requires accountability, interpretability, and equity."
                    ),
                }
            ],
            "votes": [],
            "experts": [],
            "documents": [],
        }

        annotations = pipeline.process_batch(data)
        dry_run_rows = collect_final_accepted_rows(annotations, make_entity_index(pipeline))
        dry_run_signatures = {semantic_signature(row) for row in dry_run_rows}

        serializer = RDFSerializer(annotations.to_dict(), pipeline.global_registry, Config.NAMESPACES)
        production_signatures = _serializer_signatures(serializer.graph)

        self.assertEqual(dry_run_signatures, production_signatures)
        self.assertEqual(len(dry_run_signatures), 4)

    def test_full_pipeline_does_not_materialize_candidate_rejected_by_isolated_validator(self):
        pipeline = _make_pipeline()
        discourse_id = "7"
        text = (
            "(2019) study in Science revealed racial bias in commercial health risk algorithms affecting "
            "~200 million Americans annually, used to identify patients needing additional care programs."
        )

        base_entities = [
            _entity("health_risk_algorithms_fixture", "Commercial Health Risk Algorithms", "AISystem", discourse_id, section="Description"),
            _entity("additional_care_programs_fixture", "Additional Care Programs", "ConceptObject", discourse_id, section="Description"),
        ]

        def _process_entities_batch(texts, metadata_list):
            batches = []
            for current_text, metadata in zip(texts, metadata_list):
                did = str(metadata.get("discourse_id") or discourse_id)
                entities = []
                for base in base_entities:
                    meta = dict(base.metadata)
                    meta["discourse_id"] = did
                    start = str(current_text).lower().find(str(base.label).lower())
                    if start >= 0:
                        end = start + len(str(base.label))
                        meta["start_char"] = start
                        meta["end_char"] = end
                        meta["mentions"] = [
                            {
                                "discourse_id": did,
                                "start_char": start,
                                "end_char": end,
                            }
                        ]
                    ent = Entity(
                        uri=base.uri,
                        label=base.label,
                        entity_type=base.entity_type,
                        confidence=base.confidence,
                        metadata=meta,
                    )
                    pipeline._ensure_entity_ontology_grounding(ent)
                    entities.append(pipeline.global_registry.get_or_create(ent))
                batches.append(entities)
            return batches

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            entity_map = {entity.label: entity for entity in local_entities}
            return [
                Relation(
                    source=entity_map["Commercial Health Risk Algorithms"].uri,
                    relation="affects",
                    target=entity_map["Additional Care Programs"].uri,
                    confidence=0.92,
                    metadata={
                        "extractor": "llm",
                        "context_source": "llm_relation",
                        "evidence_quote": unit_text,
                        "evidence": unit_text,
                    },
                )
            ]

        pipeline._process_entities_batch = _process_entities_batch
        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        fixture_entities = _process_entities_batch([text], [{"discourse_id": discourse_id}])[0]
        isolated = Relation(
            source=fixture_entities[0].uri,
            relation="affects",
            target=fixture_entities[1].uri,
            confidence=0.92,
            metadata={
                "field": "theme.description",
                "source_field": "theme.description",
                "extractor": "llm",
                "context_source": "llm_relation",
                "section_category": "semantic",
                "evidence_quote": text,
                "evidence": text,
            },
        )
        isolated.context = text
        accepted, _reason = pipeline._validate_semantic_relation_candidate(
            isolated,
            entities_by_uri={entity.uri: entity for entity in fixture_entities},
            discourse_text=text,
        )
        self.assertFalse(accepted)

        data = {
            "themes": [
                {
                    "id": discourse_id,
                    "title": "Healthcare: Population Health Risk Algorithms",
                    "description": text,
                }
            ],
            "questions": [],
            "answers": [],
            "votes": [],
            "experts": [],
            "documents": [],
        }

        annotations = pipeline.process_batch(data)
        dry_run_rows = collect_final_accepted_rows(annotations, make_entity_index(pipeline))
        signatures = {
            (
                row["source_field"],
                row["source_id"],
                row["subject"],
                row["predicate"],
                row["object"],
            )
            for row in dry_run_rows
        }
        self.assertNotIn(
            (
                "theme.description",
                discourse_id,
                "Commercial Health Risk Algorithms",
                "affects",
                "Additional Care Programs",
            ),
            signatures,
        )

    def test_closure_subset_matches_expected_semantic_signatures(self):
        pipeline = _make_pipeline()
        pipeline.entity_extractor = _NoopEntityExtractor()
        pipeline.canonical_linker = CanonicalLinker(pipeline.global_registry, output_dir=Config.OUTPUT_DIR)

        data = {
            "themes": [
                {
                    "id": "9",
                    "title": "Financial Services: Mortgage Lending Discrimination",
                    "description": (
                        "**Economic Impact**: Results in $765M+ annually in excessive interest payments\n"
                        "**Intersectional Bias**: Particularly affects African American women"
                    ),
                },
                {
                    "id": "15",
                    "title": "Insurance: Algorithmic Underwriting Bias",
                    "description": (
                        "AI-powered insurance underwriting systems show bias in risk assessment and pricing, "
                        "particularly affecting minority communities and certain geographic areas."
                    ),
                },
            ],
            "questions": [
                {
                    "id": "20",
                    "theme_id": "7",
                    "expert_id": "49",
                    "body": "AI could be dangerous because:\n1) AI systems can have real-world impacts.",
                }
            ],
            "answers": [
                {
                    "id": "13",
                    "question_id": "16",
                    "theme_id": "7",
                    "expert_id": "49",
                    "description": "Could the AI system affect human autonomy by generating over-reliance by end-users?",
                },
                {
                    "id": "24",
                    "question_id": "17",
                    "theme_id": "10",
                    "expert_id": "5",
                    "description": "There are fairness algorithms in place to mitigate disparate impact.",
                },
            ],
            "votes": [],
            "experts": [],
            "documents": [],
        }

        annotations = pipeline.process_batch(data)
        rows = collect_final_accepted_rows(annotations, make_entity_index(pipeline))
        signatures = {
            (
                row["source_field"],
                row["source_id"],
                row["subject"],
                row["predicate"],
                row["object"],
            )
            for row in rows
        }

        expected = {
            ("answer.description", "13", "AI system", "affects", "Human Autonomy"),
            ("answer.description", "24", "Fairness Algorithms", "mitigates", "Disparate Impact"),
            ("theme.description", "15", "AI-Powered Insurance Underwriting Systems", "particularlyAffects", "Minority Communities"),
            ("theme.description", "15", "AI-Powered Insurance Underwriting Systems", "particularlyAffects", "Certain Geographic Areas"),
            ("theme.description", "9", "Intersectional Bias", "particularlyAffects", "African American Women"),
        }

        self.assertEqual(signatures, expected)
        self.assertNotIn(
            ("answer.description", "13", "AI system", "affects", "End-Users"),
            signatures,
        )
        self.assertNotIn(
            (
                "theme.description",
                "15",
                "AI-powered insurance underwriting systems",
                "particularlyAffects",
                "Minority Communities And Certain Geographic Areas",
            ),
            signatures,
        )
        self.assertNotIn(
            (
                "theme.description",
                "9",
                "Economic Impact",
                "causes",
                "$765M+ annually in excessive interest payments",
            ),
            signatures,
        )


if __name__ == "__main__":
    unittest.main()
