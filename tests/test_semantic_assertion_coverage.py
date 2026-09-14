import tempfile
import unittest
from pathlib import Path

from rdflib import Literal, Namespace, RDF
from rdflib.namespace import XSD

from config import Config
from models import AnnotationBatch, Entity, ProcessingResult, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from serializers.rdf_serializer import RDFSerializer
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer
from utils.semantic_chunker import SemanticChunker
from utils.semantic_relation_policy import SemanticRelationPolicy


class _EchoResolver:
    @staticmethod
    def resolve(text):
        return text


def _make_pipeline():
    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    pipeline.semantic_policy = SemanticRelationPolicy()
    pipeline.predicate_precision = PredicatePrecisionLayer(pipeline.semantic_policy)
    pipeline.coref_resolver = _EchoResolver()
    pipeline._discourse_chunk_spans = {}
    pipeline._candidate_generation_diagnostics = []
    return pipeline


def _register_entity(pipeline, local_id, label, entity_type, discourse_id="d1"):
    entity = Entity(
        uri=f"{Config.NAMESPACES['etd']}{local_id}",
        label=label,
        entity_type=entity_type,
        confidence=1.0,
        metadata={
            "discourse_id": discourse_id,
            "field": "answer.description",
            "source_field": "answer.description",
            "section": "Body",
            "section_category": "semantic",
            "context_source": "test",
        },
    )
    pipeline._ensure_entity_ontology_grounding(entity)
    return pipeline.global_registry.get_or_create(entity)


class SemanticAssertionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()
        self.prev_output_dir = Config.OUTPUT_DIR
        self.tmpdir = tempfile.TemporaryDirectory()
        Config.OUTPUT_DIR = Path(self.tmpdir.name)

    def tearDown(self):
        Config.OUTPUT_DIR = self.prev_output_dir
        self.tmpdir.cleanup()

    def test_numbered_bullet_units_are_preserved(self):
        text = "1) Primary: MIMIC-CXR Database\n2) Source: PhysioNet\n3) Access: publicly available for research"
        units = SemanticChunker.semantic_relation_units(text, field="answer.description")
        self.assertEqual(len(units), 3)
        self.assertEqual(units[0]["unit_type"], "bullet")
        self.assertIn("Primary: MIMIC-CXR Database", units[0]["text"])
        self.assertIn("Access: publicly available for research", units[2]["text"])

    def test_resource_sections_are_not_excluded_from_semantics(self):
        text = (
            "**Primary**: MIMIC-CXR Database\n"
            "**URL**: https://physionet.org/content/mimic-cxr/2.0.0/\n"
            "**Content**: 377,110 chest X-ray images\n"
            "**Source**: PhysioNet"
        )
        sections = SemanticChunker.parse_structured_sections(text, field="theme.description")
        self.assertTrue(all(section["section_category"] == "semantic" for section in sections))
        units = SemanticChunker.semantic_relation_units(text, field="theme.description")
        self.assertEqual(len(units), 4)
        joined = "\n".join(unit["text"] for unit in units)
        self.assertIn("MIMIC-CXR Database", joined)
        self.assertIn("PhysioNet", joined)

    def test_long_answer_late_paragraph_units_are_retained(self):
        text = (
            "First paragraph explains the setting.\n\n"
            "Second paragraph adds more context.\n\n"
            "Historical data reflects systemic bias.\n\n"
            "Final paragraph states that fairness requires accountability."
        )
        units = SemanticChunker.semantic_relation_units(text, field="answer.description")
        self.assertGreaterEqual(len(units), 4)
        self.assertTrue(any("fairness requires accountability" in unit["text"].lower() for unit in units))

    def test_unknown_predicate_survives_when_textually_supported(self):
        dataset = _register_entity(self.pipeline, "mimic", "MIMIC-CXR Database", "Dataset")
        host = _register_entity(self.pipeline, "physionet", "PhysioNet", "Organisation")
        relation = Relation(
            source=dataset.uri,
            relation="hostedBy",
            target=host.uri,
            confidence=0.93,
            predicate_text="hosted by",
            predicate_mapping_status="local_unmapped",
            metadata={
                "field": "theme.description",
                "source_field": "theme.description",
                "section": "Primary",
                "section_category": "semantic",
                "context_source": "llm_relation",
                "evidence": "MIMIC-CXR Database is hosted by PhysioNet.",
                "evidence_quote": "hosted by PhysioNet",
                "allow_local_predicate": True,
            },
        )
        accepted, _ = self.pipeline._validate_semantic_relation_candidate(
            relation,
            entities_by_uri={dataset.uri: dataset, host.uri: host},
            discourse_text="MIMIC-CXR Database is hosted by PhysioNet.",
        )
        self.assertTrue(accepted)
        self.assertEqual(relation.metadata.get("semantic_validation_code"), "accepted_unmapped_predicate")

    # ------------------------------------------------------------------
    # Full production-order integration tests: refine_relations() THEN
    # validate(). The test above (test_unknown_predicate_survives_when_
    # textually_supported) constructs relation="hostedBy" directly and
    # skips PredicatePrecisionLayer.refine_relations() entirely, so it
    # never exercises the real first-stage gate the live pipeline always
    # runs first (apply_predicate_precision_and_enrichment() before
    # validate_semantic_relations()). These tests close that gap.
    # ------------------------------------------------------------------

    def _run_refine_then_validate(self, relation, entities):
        entities_by_uri = {e.uri: e for e in entities}
        discourse_text = str(relation.metadata.get("evidence") or "")
        refined = self.pipeline.predicate_precision.refine_relations(
            relations=[relation],
            entities=entities,
            discourse_text=discourse_text,
        )
        self.assertEqual(len(refined), 1)
        refined_relation = refined[0]
        accepted, _ = self.pipeline._validate_semantic_relation_candidate(
            refined_relation,
            entities_by_uri=entities_by_uri,
            discourse_text=discourse_text,
        )
        return refined_relation, accepted

    def test_specific_unmapped_predicate_survives_full_chain_not_relatedTo(self):
        # NOTE: this test originally used relation="hostedBy" as its
        # stand-in "unregistered but specific" predicate example -- for
        # exactly this MIMIC-CXR/PhysioNet hosting shape. A later
        # root-cause fix (manual semantic-QA pass, category G) registered
        # hostedBy as a first-class predicate precisely because this shape
        # is common and was previously mishandled (see
        # utils/semantic_relation_policy.py PREDICATE_REGISTRY["hostedBy"]),
        # so hostedBy is no longer a valid example of an *unregistered*
        # predicate. Swapped to a different, still-unregistered stand-in
        # verb ("mirroredBy") to keep testing the same general policy.
        dataset = _register_entity(self.pipeline, "mimic_fc", "MIMIC-CXR Database", "Dataset")
        host = _register_entity(self.pipeline, "physionet_fc", "PhysioNet", "Organisation")
        relation = Relation(
            source=dataset.uri,
            relation="mirroredBy",
            target=host.uri,
            confidence=0.93,
            metadata={
                "field": "answer.description",
                "source_field": "answer.description",
                "section": "Body",
                "section_category": "semantic",
                "context_source": "llm_json",
                "extractor": "llm_relation_extractor",
                "evidence": "MIMIC-CXR Database is mirrored by PhysioNet.",
                "evidence_quote": "mirrored by PhysioNet",
                "predicate_text": "mirrored by",
                "allow_local_predicate": True,
            },
        )
        refined_relation, accepted = self._run_refine_then_validate(relation, [dataset, host])
        self.assertNotEqual(refined_relation.relation, "relatedTo")
        self.assertEqual(refined_relation.relation, "mirroredBy")
        self.assertTrue(accepted)
        self.assertEqual(refined_relation.metadata.get("semantic_validation_code"), "accepted_unmapped_predicate")

    def test_genuine_generic_relation_remains_valid_full_chain(self):
        concept_a = _register_entity(self.pipeline, "concept_a_fc", "Concept A", "EthicalConcept")
        concept_b = _register_entity(self.pipeline, "concept_b_fc", "Concept B", "EthicalConcept")
        relation = Relation(
            source=concept_a.uri,
            relation="relatedTo",
            target=concept_b.uri,
            confidence=0.8,
            metadata={
                "field": "answer.description",
                "source_field": "answer.description",
                "section": "Body",
                "section_category": "semantic",
                "context_source": "llm_json",
                "extractor": "llm_relation_extractor",
                "evidence": "Concept A is related to Concept B.",
                "evidence_quote": "Concept A is related to Concept B.",
                "predicate_text": "related to",
                "allow_local_predicate": True,
            },
        )
        refined_relation, accepted = self._run_refine_then_validate(relation, [concept_a, concept_b])
        self.assertEqual(refined_relation.relation, "relatedTo")
        self.assertTrue(accepted)
        self.assertEqual(refined_relation.metadata.get("semantic_validation_code"), "accepted")

    def test_existing_canonical_predicate_unaffected_full_chain(self):
        system = _register_entity(self.pipeline, "system_a_fc", "System A", "AISystem")
        risk = _register_entity(self.pipeline, "risk_b_fc", "Risk B", "Risk")
        relation = Relation(
            source=system.uri,
            relation="causes",
            target=risk.uri,
            confidence=0.9,
            metadata={
                "field": "answer.description",
                "source_field": "answer.description",
                "section": "Body",
                "section_category": "semantic",
                "context_source": "llm_json",
                "extractor": "llm_relation_extractor",
                "evidence": "System A causes Risk B in deployment.",
                "evidence_quote": "System A causes Risk B in deployment.",
                "predicate_text": "causes",
                "allow_local_predicate": True,
            },
        )
        refined_relation, accepted = self._run_refine_then_validate(relation, [system, risk])
        self.assertEqual(refined_relation.relation, "causes")
        self.assertTrue(accepted)
        self.assertEqual(refined_relation.metadata.get("semantic_validation_code"), "accepted")

    def test_show_bias_against_recovers_as_particularly_affects_full_chain(self):
        # Coverage fix: the LLM/cue-hint generator mis-predicates "show ... bias
        # against <population>" sentences as exhibitsRisk (whose range excludes
        # population/group targets), which used to type-guard-downgrade to
        # relatedTo and then get rejected outright. It must now self-correct to
        # particularlyAffects via the existing evidence-upgrade mechanism.
        system = _register_entity(self.pipeline, "aes_system_fc", "Automated Essay Scoring Systems", "AISystem")
        group = _register_entity(self.pipeline, "marginalized_groups_fc", "Students from Marginalized Groups", "Demographicgroup")
        evidence = "Automated Essay Scoring Systems show systematic bias against Students from Marginalized Groups."
        relation = Relation(
            source=system.uri,
            relation="exhibitsRisk",
            target=group.uri,
            confidence=0.9,
            metadata={
                "field": "theme.description",
                "source_field": "theme.description",
                "section": "Description",
                "section_category": "semantic",
                "context_source": "llm_json",
                "extractor": "llm_relation_extractor",
                "evidence": evidence,
                "evidence_quote": evidence,
                "predicate_text": "show",
                "allow_local_predicate": True,
            },
        )
        refined_relation, accepted = self._run_refine_then_validate(relation, [system, group])
        self.assertEqual(refined_relation.relation, "particularlyAffects")
        self.assertTrue(accepted)

    def test_quantitative_disparity_recovers_as_particularly_affects_full_chain(self):
        # Coverage fix: a stated rate/multiplier with an explicit comparison marker
        # ("19% ... vs. 12%") is disparate-impact evidence even without any of the
        # registered bias-vocabulary trigger verbs.
        model = _register_entity(self.pipeline, "gpt4o_fc", "GPT-4o", "Model")
        students = _register_entity(self.pipeline, "black_students_fc", "Black students", "Population")
        evidence = (
            "GPT-4o incorrectly predicted academic failure for Black students 19% of the time "
            "vs. 12% for white students."
        )
        relation = Relation(
            source=model.uri,
            relation="particularlyAffects",
            target=students.uri,
            confidence=0.92,
            metadata={
                "field": "theme.description",
                "source_field": "theme.description",
                "section": "Description",
                "section_category": "semantic",
                "context_source": "llm_json",
                "extractor": "llm_relation_extractor",
                "evidence": evidence,
                "evidence_quote": evidence,
                "predicate_text": "predicted",
                "allow_local_predicate": True,
            },
        )
        refined_relation, accepted = self._run_refine_then_validate(relation, [model, students])
        self.assertEqual(refined_relation.relation, "particularlyAffects")
        self.assertTrue(accepted)

    def test_unregistered_specific_verbs_are_never_silently_collapsed_to_relatedTo(self):
        # Regression pin for RC-1: an unrecognized-but-specific predicate must
        # never normalize straight to relatedTo/associatedWith. Only a raw
        # predicate that is itself a registered generic phrase may do so.
        # "hostedBy" was removed from this list: a later root-cause fix
        # (manual semantic-QA pass, category G) registered it as a
        # first-class predicate, so it is no longer an example of an
        # *unregistered* one -- see PREDICATE_REGISTRY["hostedBy"] in
        # utils/semantic_relation_policy.py.
        layer = self.pipeline.predicate_precision
        for raw in ("reinforces", "mirroredBy", "usedBy", "proxyFor", "similarTo", "locatedIn", "provides"):
            self.assertEqual(
                layer._normalize_candidate(raw),
                "",
                msg=f"{raw!r} should be unrecognized (not silently mapped to relatedTo/associatedWith)",
            )
        for raw in ("related to", "connected to", "linked to"):
            self.assertEqual(layer._normalize_candidate(raw), "relatedTo")
        self.assertEqual(layer._normalize_candidate("associated with"), "associatedWith")
        self.assertEqual(layer._normalize_candidate("causes"), "causes")

    def test_unknown_predicate_without_local_support_is_rejected(self):
        dataset = _register_entity(self.pipeline, "mimic2", "MIMIC-CXR Database", "Dataset")
        host = _register_entity(self.pipeline, "physionet2", "PhysioNet", "Organisation")
        relation = Relation(
            source=dataset.uri,
            relation="hostedBy",
            target=host.uri,
            confidence=0.93,
            predicate_text="hosted by",
            predicate_mapping_status="local_unmapped",
            metadata={
                "field": "theme.description",
                "source_field": "theme.description",
                "section": "Primary",
                "section_category": "semantic",
                "context_source": "llm_relation",
                "evidence": "MIMIC-CXR Database includes diagnostic images from multiple hospitals.",
                "evidence_quote": "includes diagnostic images",
                "allow_local_predicate": True,
            },
        )
        accepted, _ = self.pipeline._validate_semantic_relation_candidate(
            relation,
            entities_by_uri={dataset.uri: dataset, host.uri: host},
            discourse_text="MIMIC-CXR Database includes diagnostic images from multiple hospitals.",
        )
        self.assertFalse(accepted)
        self.assertEqual(relation.metadata.get("semantic_validation_code"), "rejected_not_entailed")

    def test_quantitative_comparison_preserves_qualifiers(self):
        black = _register_entity(self.pipeline, "black_patients", "Black patients", "DemographicGroup")
        white = _register_entity(self.pipeline, "white_patients", "White patients", "DemographicGroup")
        relation = Relation(
            source=black.uri,
            relation="spendLessThan",
            target=white.uri,
            confidence=0.96,
            predicate_text="spent less than",
            predicate_mapping_status="local_unmapped",
            qualifiers={"amount": 1800, "currency": "USD", "period": "annual", "approximate": True},
            quantitative_information={"amount": 1800, "currency": "USD", "period": "annual"},
            metadata={
                "field": "theme.description",
                "source_field": "theme.description",
                "section": "Description",
                "section_category": "semantic",
                "context_source": "llm_relation",
                "evidence": "Black patients spent approximately $1,800 less annually than white patients with identical conditions.",
                "evidence_quote": "spent approximately $1,800 less annually than white patients",
                "allow_local_predicate": True,
            },
        )
        accepted, _ = self.pipeline._validate_semantic_relation_candidate(
            relation,
            entities_by_uri={black.uri: black, white.uri: white},
            discourse_text="Black patients spent approximately $1,800 less annually than white patients with identical conditions.",
        )
        self.assertTrue(accepted)
        self.assertEqual(relation.metadata.get("semantic_validation_code"), "accepted_unmapped_predicate")
        self.assertEqual(relation.qualifiers.get("currency"), "USD")
        self.assertEqual(relation.quantitative_information.get("amount"), 1800)

    def test_literal_assertion_serializes_as_literal_with_metadata(self):
        dataset = _register_entity(self.pipeline, "mimic_count", "MIMIC-CXR Database", "Dataset")
        relation = Relation(
            source=dataset.uri,
            relation="imageCount",
            target="377110",
            target_literal=377110,
            target_datatype="integer",
            target_is_literal=True,
            object_type="count",
            confidence=0.98,
            predicate_text="contains",
            predicate_mapping_status="local_unmapped",
            qualifiers={"approximate": False},
            quantitative_information={"count": 377110},
            metadata={
                "field": "theme.description",
                "source_field": "theme.description",
                "section": "Content",
                "section_category": "semantic",
                "context_source": "llm_relation",
                "evidence": "MIMIC-CXR Database contains 377,110 chest X-ray images.",
                "evidence_quote": "377,110 chest X-ray images",
                "allow_local_predicate": True,
                "target_is_literal": True,
                "target_datatype": "integer",
                "object_type": "count",
            },
        )
        annotations = AnnotationBatch(
            themes={
                "7": ProcessingResult(
                    entities=[dataset],
                    relations=[relation],
                    metadata={"text": "MIMIC-CXR Database contains 377,110 chest X-ray images."},
                )
            }
        )
        serializer = RDFSerializer(annotations.to_dict(), self.pipeline.global_registry, Config.NAMESPACES)
        graph = serializer.graph
        etd = Namespace(Config.NAMESPACES["etd"])
        self.assertIn((Namespace(Config.NAMESPACES["etd"])["mimic_count"], etd.imageCount, Literal(377110, datatype=XSD.integer)), graph)
        rel_nodes = list(graph.subjects(RDF.type, etd.Relation))
        self.assertTrue(rel_nodes)
        self.assertTrue(any(str(graph.value(node, etd.qualifiers_json) or "").find("\"approximate\": false") >= 0 for node in rel_nodes))

    def test_unseen_domain_supported_local_predicate_is_allowed(self):
        sensor = _register_entity(self.pipeline, "sensor_array", "Satellite sensor array", "AISystem")
        anomaly = _register_entity(self.pipeline, "wind_anomaly", "wind-speed anomalies", "Concept")
        relation = Relation(
            source=sensor.uri,
            relation="records",
            target=anomaly.uri,
            confidence=0.91,
            predicate_text="records",
            predicate_mapping_status="local_unmapped",
            metadata={
                "field": "answer.description",
                "source_field": "answer.description",
                "section": "Body",
                "section_category": "semantic",
                "context_source": "llm_relation",
                "evidence": "A satellite sensor array records wind-speed anomalies during coastal storms.",
                "evidence_quote": "records wind-speed anomalies",
                "allow_local_predicate": True,
            },
        )
        accepted, _ = self.pipeline._validate_semantic_relation_candidate(
            relation,
            entities_by_uri={sensor.uri: sensor, anomaly.uri: anomaly},
            discourse_text="A satellite sensor array records wind-speed anomalies during coastal storms.",
        )
        self.assertTrue(accepted)
        self.assertEqual(relation.metadata.get("semantic_validation_code"), "accepted_unmapped_predicate")

