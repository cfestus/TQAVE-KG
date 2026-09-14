"""Regression tests for the manual semantic-quality evaluation pass
(Evaluator 1 / Evaluator 2 comments).

Tests are grouped by failure class, matching the evaluator report:

  A. Non-asserted / negated relations must not surface as an unqualified
     positive direct SPO triple.
  B. Endpoint surface-form / canonical entity consistency.
  C. Invalid clause-fragment entities.
  D. Pronoun / clausal endpoint resolution.
  E. Cross-discourse over-canonicalisation.
  F. Comparative / coordinated constructions.
  G. Predicate semantics / role confusion.
  H. Entity typing.
  I. Evidence entailment / grounding quality.
  J. Redundant / contradictory malformed relations.

No test calls the live OpenAI API. Where a fix depends on LLM-extractor
output, that output is mocked/hand-built as a deterministic fixture and only
the downstream deterministic logic is exercised.
"""

import importlib.util
import unittest

from config import Config
from models import Entity, Relation

HAS_RDFLIB = importlib.util.find_spec("rdflib") is not None


class _Registry:
    def __init__(self, entities):
        self._entities = entities

    def get_all_entities(self):
        return self._entities


def _base_annotations(relations, entities_by_doc=None):
    """Minimal annotations_dict shape accepted by RDFSerializer."""
    return {
        "themes": {}, "questions": {},
        "answers": {
            "d1": {"relations": relations},
        },
        "votes": {}, "experts": {}, "documents": {},
        "metadata": {},
    }


@unittest.skipUnless(HAS_RDFLIB, "rdflib is required for RDF serialization tests")
class NonAssertedRelationDirectSPOTests(unittest.TestCase):
    """Section A: negated/questioned/hypothetical/uncertain relations must
    keep their reified relation node (with negation/epistemic metadata) but
    must NOT also emit an unqualified positive direct SPO triple."""

    etd_base = Config.NAMESPACES["etd"]

    def _build_graph(self, rel_metadata):
        from rdflib import Namespace, RDF
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        source_uri = f"{self.etd_base}entity_source_deadbeef00"
        target_uri = f"{self.etd_base}entity_target_deadbeef01"

        entities = [
            Entity(uri=source_uri, label="Context Insensitivity", entity_type="Concept", confidence=0.8, metadata={}),
            Entity(uri=target_uri, label="Cultural Context", entity_type="Concept", confidence=0.8, metadata={}),
        ]
        base_meta = {
            "discourse_id": "d1",
            "field": "answer.description",
            "context_source": "llm_json",
            "chunk_index": 0,
            "source": "llm_relation_extractor",
            "subject_text": "Context Insensitivity",
            "object_text": "cultural context",
        }
        base_meta.update(rel_metadata)
        relations = [
            {
                "source": source_uri,
                "relation": "understand",
                "target": target_uri,
                "confidence": 0.9,
                "metadata": base_meta,
            }
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        return serializer.graph, etd_ns, source_uri, target_uri

    def test_negated_relation_has_no_direct_positive_triple(self):
        from rdflib import URIRef
        g, etd_ns, source_uri, target_uri = self._build_graph({"negated": True})

        direct = list(g.triples((URIRef(source_uri), etd_ns.understand, URIRef(target_uri))))
        self.assertEqual(direct, [], "negated relation must not emit an unqualified direct triple")

        # Reified relation + its negation metadata must still be present.
        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(source_uri)))
        self.assertEqual(len(rel_nodes), 1)
        negated_vals = [bool(v) for v in g.objects(rel_nodes[0], etd_ns.negated)]
        self.assertEqual(negated_vals, [True])
        self.assertEqual(list(g.objects(rel_nodes[0], etd_ns.object)), [URIRef(target_uri)])

    def test_hypothetical_relation_has_no_direct_positive_triple(self):
        from rdflib import URIRef
        g, etd_ns, source_uri, target_uri = self._build_graph({"epistemic_status": "hypothetical"})

        direct = list(g.triples((URIRef(source_uri), etd_ns.understand, URIRef(target_uri))))
        self.assertEqual(direct, [])

        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(source_uri)))
        self.assertEqual(len(rel_nodes), 1)
        self.assertEqual([str(v) for v in g.objects(rel_nodes[0], etd_ns.epistemic_status)], ["hypothetical"])

    def test_questioned_relation_has_no_direct_positive_triple(self):
        from rdflib import URIRef
        g, etd_ns, source_uri, target_uri = self._build_graph({"epistemic_status": "questioned"})

        direct = list(g.triples((URIRef(source_uri), etd_ns.understand, URIRef(target_uri))))
        self.assertEqual(direct, [])

        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(source_uri)))
        self.assertEqual(len(rel_nodes), 1)

    def test_uncertain_relation_has_no_direct_positive_triple(self):
        from rdflib import URIRef
        g, etd_ns, source_uri, target_uri = self._build_graph({"epistemic_status": "uncertain"})

        direct = list(g.triples((URIRef(source_uri), etd_ns.understand, URIRef(target_uri))))
        self.assertEqual(direct, [])

        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(source_uri)))
        self.assertEqual(len(rel_nodes), 1)

    def test_ordinary_asserted_relation_still_emits_direct_triple(self):
        from rdflib import URIRef
        g, etd_ns, source_uri, target_uri = self._build_graph({"epistemic_status": "asserted"})

        direct = list(g.triples((URIRef(source_uri), etd_ns.understand, URIRef(target_uri))))
        self.assertEqual(len(direct), 1, "an ordinary asserted-positive relation must still emit its direct triple")

        rel_nodes = list(g.subjects(etd_ns.subject, URIRef(source_uri)))
        self.assertEqual(len(rel_nodes), 1)

    def test_relation_with_no_epistemic_metadata_still_emits_direct_triple(self):
        """Absence of negated/epistemic_status (the vast majority of
        structural/legacy relations) must default to asserted behaviour --
        this fix must not regress ordinary relations that never populate
        these fields at all."""
        from rdflib import URIRef
        g, etd_ns, source_uri, target_uri = self._build_graph({})

        direct = list(g.triples((URIRef(source_uri), etd_ns.understand, URIRef(target_uri))))
        self.assertEqual(len(direct), 1)


class EndpointSurfaceFormConsistencyTests(unittest.TestCase):
    """Section B: a relation endpoint that resolves to a canonical entity
    must be lexically compatible with the surface text actually captured
    for that endpoint."""

    etd_base = Config.NAMESPACES["etd"]

    def _build(self, *, subject_label, subject_surface, object_label, object_surface):
        from rdflib import Namespace
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(self.etd_base)
        source_uri = f"{self.etd_base}entity_source_bfeed00"
        target_uri = f"{self.etd_base}entity_target_bfeed01"
        entities = [
            # Explicit surface_form (as a genuinely-extracted entity would
            # carry) -- keeps these entities OUT of
            # _generic_surface_form_entities, whose members are exempt
            # from the compatibility check below since their label is a
            # synthetic URI-slug placeholder, not a real claim.
            Entity(uri=source_uri, label=subject_label, entity_type="Concept", confidence=0.8, metadata={"surface_form": subject_label}),
            Entity(uri=target_uri, label=object_label, entity_type="Concept", confidence=0.8, metadata={"surface_form": object_label}),
        ]
        relations = [
            {
                "source": source_uri,
                "relation": "relatedTo",
                "target": target_uri,
                "confidence": 0.9,
                "metadata": {
                    "discourse_id": "d1",
                    "field": "answer.description",
                    "chunk_index": 0,
                    "source": "llm_relation_extractor",
                    "subject_surface_form": subject_surface,
                    "object_surface_form": object_surface,
                },
            }
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        return serializer.graph, etd_ns, source_uri, target_uri

    def _relation_exists(self, g, etd_ns, source_uri, target_uri):
        from rdflib import URIRef
        return bool(list(g.subjects(etd_ns.subject, URIRef(source_uri)))) or bool(
            list(g.triples((URIRef(source_uri), None, URIRef(target_uri))))
        )

    def test_compatible_normalized_surface_is_kept(self):
        g, etd_ns, s, t = self._build(
            subject_label="Cultural Context", subject_surface="cultural context",
            object_label="Context Insensitivity", object_surface="context insensitivity",
        )
        self.assertTrue(self._relation_exists(g, etd_ns, s, t))

    def test_legitimate_acronym_alias_is_kept(self):
        g, etd_ns, s, t = self._build(
            subject_label="Artificial Intelligence", subject_surface="AI",
            object_label="General Data Protection Regulation", object_surface="the GDPR",
        )
        self.assertTrue(self._relation_exists(g, etd_ns, s, t))

    def test_explicit_pronoun_coreference_is_exempt(self):
        g, etd_ns, s, t = self._build(
            subject_label="The Credit Scoring Model", subject_surface="it",
            object_label="Loan Applicants", object_surface="applicants",
        )
        self.assertTrue(self._relation_exists(g, etd_ns, s, t))

    def test_incompatible_subject_binding_is_dropped(self):
        g, etd_ns, s, t = self._build(
            subject_label="AI Recruitment Tool", subject_surface="names associated with Black women",
            object_label="Hiring Outcomes", object_surface="hiring outcomes",
        )
        self.assertFalse(self._relation_exists(g, etd_ns, s, t))

    def test_incompatible_object_binding_is_dropped(self):
        g, etd_ns, s, t = self._build(
            subject_label="Facial Recognition System", subject_surface="the system",
            object_label="Black Women", object_surface="white men",
        )
        self.assertFalse(self._relation_exists(g, etd_ns, s, t))


def _make_bare_pipeline():
    """Shared minimal EnhancedSemanticPipeline fixture -- bypasses __init__
    (which would touch real config/API state) and wires only what
    _resolve_semantic_unit_relations and its helpers actually need, per
    the established pattern in test_kg_quality_root_cause_fixes.py."""
    from pipeline import EnhancedSemanticPipeline
    from utils.helpers import GlobalEntityRegistry
    from ontology.mapper import OntologyMapper
    from utils.entity_normalizer import EntityNormalizer

    pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
    pipeline.config = Config
    pipeline.global_registry = GlobalEntityRegistry()
    pipeline.ontology_mapper = OntologyMapper()
    pipeline.entity_normalizer = EntityNormalizer(pipeline.global_registry)
    return pipeline


def _register_entity(pipeline, *, uri, label, entity_type="Concept"):
    ent = Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.9, metadata={"discourse_id": "d1"})
    pipeline._ensure_entity_ontology_grounding(ent)
    return pipeline.global_registry.get_or_create(ent)


class ClauseFragmentEntityTests(unittest.TestCase):
    """Section C: finite clause / speaker-intent fragments must never
    become persistent entities."""

    def test_is_finite_clause_fragment_rejects_known_examples(self):
        from utils.helpers import is_finite_clause_fragment
        for phrase in ("I Think There Needs", "I believe that", "we need to", "You should know"):
            self.assertTrue(is_finite_clause_fragment(phrase), phrase)

    def test_is_finite_clause_fragment_accepts_legitimate_concepts(self):
        from utils.helpers import is_finite_clause_fragment
        for phrase in ("Algorithmic Decision-Making", "AI Team", "Audit Trails", "Automated Content Moderation"):
            self.assertFalse(is_finite_clause_fragment(phrase), phrase)

    def test_pipeline_fabrication_guard_blocks_clause_fragment(self):
        from pipeline import EnhancedSemanticPipeline
        self.assertTrue(EnhancedSemanticPipeline._is_unresolved_coreference_candidate("I Think There Needs"))
        self.assertFalse(EnhancedSemanticPipeline._is_unresolved_coreference_candidate("Algorithmic Decision-Making"))

    def test_entity_extractor_rejects_clause_fragment_entities(self):
        from unittest.mock import patch
        from extractors.entity_extractor import LLMEntityExtractor

        extractor = LLMEntityExtractor(ontology_terms={}, min_confidence=0.5)
        mocked = [
            {"surface_form": "I Think There Needs", "entity_type": "Concept", "ontology_uri": "", "confidence": 0.9},
            {"surface_form": "Audit Trails", "entity_type": "Concept", "ontology_uri": "", "confidence": 0.9},
        ]
        with patch.object(extractor, "_call_model", return_value=mocked):
            entities = extractor.extract(
                "I Think There Needs to be better Audit Trails in this system.", doc_id="test"
            )
        labels = [e.label for e in entities]
        self.assertNotIn("I Think There Needs", labels)
        self.assertIn("Audit Trails", labels)


class PronounClausalEndpointResolutionTests(unittest.TestCase):
    """Section D: SV027/SV028-shaped self-referential relations must be
    dropped, without over-triggering on legitimately distinct same-type
    endpoints."""

    def test_exact_self_loop_relation_is_dropped(self):
        """SV028 shape: 'Audit Trails make Audit Trails'."""
        pipeline = _make_bare_pipeline()
        audit_trails = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}concept_audit_trails_t1", label="Audit Trails")

        relation = Relation(
            source=audit_trails.uri, relation="make", target=audit_trails.uri,
            metadata={"subject_text": "Audit Trails", "object_text": "Audit Trails", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "Audit trails make audit trails auditable."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[audit_trails], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(resolved, [])

    def test_near_duplicate_causal_relation_is_dropped(self):
        """SV027 shape: 'AI Team lead a happier AI team'."""
        pipeline = _make_bare_pipeline()
        ai_team = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}concept_ai_team_t1", label="AI Team")

        relation = Relation(
            source=ai_team.uri, relation="lead", target="a happier AI team",
            metadata={"subject_text": "AI Team", "object_text": "a happier AI team", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "Investing in AI ethics can lead to a happier AI team."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[ai_team], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(resolved, [])

    def test_same_type_different_entities_not_treated_as_reflexive(self):
        """Control: distinct comparator-shaped entities sharing only a
        generic head noun must survive -- the guard is predicate/overlap
        scoped, not a blanket same-type rejection."""
        pipeline = _make_bare_pipeline()
        black_def = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}group_black_defendants_t1", label="Black Defendants", entity_type="PopulationGroup")

        relation = Relation(
            source=black_def.uri, relation="labeled", target="White Defendants",
            metadata={"subject_text": "Black Defendants", "object_text": "White Defendants", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "Black Defendants were labeled high risk more often."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[black_def], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)

    def test_non_causal_predicate_with_overlapping_text_is_not_dropped(self):
        """Control: high textual overlap alone, outside the causal
        predicate family, must not trigger the guard. Exercises the
        classifier directly (rather than the full resolution pipeline) so
        the assertion is not confounded by unrelated endpoint-resolution
        behaviour for the specific surface text chosen."""
        from pipeline import EnhancedSemanticPipeline
        self.assertFalse(
            EnhancedSemanticPipeline._is_self_referential_relation_shape(
                subject_text="AI Team",
                object_text="the AI team's charter",
                predicate_name="mentions",
                source_uri="https://alfie-project.eu/etd-hub#concept_ai_team",
                target_uri="https://alfie-project.eu/etd-hub#concept_ai_team_charter",
                target_is_literal=False,
            )
        )


class CrossDiscourseOverCanonicalizationTests(unittest.TestCase):
    """Section E: a generic/common noun must not merge with a specific
    concept from another discourse on lexical overlap alone."""

    def _linker(self):
        import tempfile
        from utils.canonical_linker import CanonicalLinker
        return CanonicalLinker(registry=None, output_dir=tempfile.mkdtemp())

    def test_generic_noun_does_not_fuzzy_merge_into_specific_concept(self):
        linker = self._linker()
        generic = Entity(uri="urn:e1", label="prediction", entity_type="Concept", confidence=0.8, metadata={})
        specific = Entity(uri="urn:e2", label="Performativity Of Predictions", entity_type="Concept", confidence=0.8, metadata={})
        self.assertFalse(linker._can_fuzzy_merge(generic, specific, "Concept"))

    def test_generic_noun_with_determiner_does_not_fuzzy_merge_into_specific_concept(self):
        linker = self._linker()
        generic = Entity(uri="urn:e1", label="the prediction", entity_type="Concept", confidence=0.8, metadata={})
        specific = Entity(uri="urn:e2", label="Performativity Of Predictions", entity_type="Concept", confidence=0.8, metadata={})
        self.assertFalse(linker._can_fuzzy_merge(generic, specific, "Concept"))

    def test_genuine_repeat_of_specific_concept_across_discourses_still_merges(self):
        linker = self._linker()
        a = Entity(uri="urn:e1", label="COMPAS Risk Score", entity_type="Concept", confidence=0.8, metadata={})
        b = Entity(uri="urn:e2", label="COMPAS Risk Scores", entity_type="Concept", confidence=0.8, metadata={})
        self.assertTrue(linker._can_fuzzy_merge(a, b, "Concept"))

    def test_bare_generic_label_shape_classifier(self):
        from utils.canonical_linker import CanonicalLinker
        import tempfile
        linker = CanonicalLinker(registry=None, output_dir=tempfile.mkdtemp())
        self.assertTrue(linker._is_bare_generic_label("prediction"))
        self.assertTrue(linker._is_bare_generic_label("the prediction"))
        self.assertFalse(linker._is_bare_generic_label("performativity of predictions"))
        self.assertFalse(linker._is_bare_generic_label("compas risk score"))


class ComparativeCoordinatedConstructionTests(unittest.TestCase):
    """Section F: comparative and coordinated 'respectively' constructions
    must not be flattened into a malformed direct relation."""

    def test_comparator_group_as_classification_object_is_dropped(self):
        pipeline = _make_bare_pipeline()
        black_def = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}group_black_defendants_f1", label="Black Defendants", entity_type="PopulationGroup")

        relation = Relation(
            source=black_def.uri, relation="labeled", target="white defendants",
            metadata={
                "subject_text": "Black defendants", "object_text": "white defendants",
                "discourse_id": "d1",
                "evidence_quote": "Black defendants were twice as likely as white defendants to be labelled high risk.",
            },
        )
        unit = {"start_offset": 0, "text": "Black defendants were twice as likely as white defendants to be labelled high risk."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[black_def], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(resolved, [])

    def test_classification_without_comparative_cue_is_kept(self):
        """Control: same predicate/entity shapes, no comparative cue in
        evidence -- must not be dropped."""
        pipeline = _make_bare_pipeline()
        black_def = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}group_black_defendants_f2", label="Black Defendants", entity_type="PopulationGroup")

        relation = Relation(
            source=black_def.uri, relation="labeled", target="high risk individuals",
            metadata={
                "subject_text": "Black defendants", "object_text": "high risk individuals",
                "discourse_id": "d1",
                "evidence_quote": "Black defendants were labelled high risk by the system.",
            },
        )
        unit = {"start_offset": 0, "text": "Black defendants were labelled high risk by the system."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[black_def], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)

    def test_respectively_shared_value_is_split_positionally(self):
        pipeline = _make_bare_pipeline()
        black_b = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}group_black_borrowers_f1", label="Black Borrowers", entity_type="PopulationGroup")
        latino_b = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}group_latino_borrowers_f1", label="Latino Borrowers", entity_type="PopulationGroup")

        evidence = "Black and Latino borrowers pay 7.9 and 3.6 basis points more respectively."
        rel_black = Relation(
            source=black_b.uri, relation="pays", target="7.9 and 3.6 basis points more",
            metadata={"subject_text": "Black borrowers", "object_text": "7.9 and 3.6 basis points more",
                      "discourse_id": "d1", "evidence_quote": evidence},
        )
        rel_latino = Relation(
            source=latino_b.uri, relation="pays", target="7.9 and 3.6 basis points more",
            metadata={"subject_text": "Latino borrowers", "object_text": "7.9 and 3.6 basis points more",
                      "discourse_id": "d1", "evidence_quote": evidence},
        )
        unit = {"start_offset": 0, "text": evidence}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[rel_black, rel_latino], local_entities=[black_b, latino_b], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 2)
        by_subject = {str(r.source): r for r in resolved}
        black_rel = by_subject[black_b.uri]
        latino_rel = by_subject[latino_b.uri]
        self.assertNotEqual(str(black_rel.target), str(latino_rel.target))
        self.assertIn("7.9", black_rel.metadata.get("object_text", "") + str(black_rel.target))
        self.assertIn("3.6", latino_rel.metadata.get("object_text", "") + str(latino_rel.target))


class PredicateSemanticsRoleConfusionTests(unittest.TestCase):
    """Section G: predicate/role errors must be corrected to a
    semantically appropriate existing predicate, not left as identity or
    description relations, and passive-voice role inversions must not be
    serialized as-is."""

    def test_is_predicate_to_repository_becomes_hostedby(self):
        pipeline = _make_bare_pipeline()
        dataset = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}dataset_adult_census_g1", label="Adult Census Income Dataset", entity_type="Dataset")

        relation = Relation(
            source=dataset.uri, relation="is", target="UCI Machine Learning Repository",
            metadata={"subject_text": "Adult/Census Income Dataset", "object_text": "UCI Machine Learning Repository", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "The Adult/Census Income Dataset is the UCI Machine Learning Repository."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[dataset], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].relation, "hostedBy")

    def test_hasdescription_entity_object_becomes_hostedby(self):
        pipeline = _make_bare_pipeline()
        dataset = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}dataset_mimic_cxr_g1", label="MIMIC-CXR Database", entity_type="Dataset")

        relation = Relation(
            source=dataset.uri, relation="hasDescription", target="PhysioNet",
            target_is_literal=False,
            metadata={"subject_text": "MIMIC-CXR Database", "object_text": "PhysioNet", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "The MIMIC-CXR Database hasDescription PhysioNet."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[dataset], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].relation, "hostedBy")

    def test_hasdescription_literal_object_is_unaffected(self):
        """Control: hasDescription's ordinary literal-text usage must be
        completely untouched by the hostedBy correction."""
        pipeline = _make_bare_pipeline()
        dataset = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}dataset_chexpert_g1", label="CheXpert Dataset", entity_type="Dataset")

        relation = Relation(
            source=dataset.uri, relation="hasDescription", target="a large chest X-ray dataset",
            target_is_literal=True, target_literal="a large chest X-ray dataset",
            metadata={"subject_text": "CheXpert Dataset", "object_text": "a large chest X-ray dataset", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "CheXpert is a large chest X-ray dataset."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[dataset], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].relation, "hasDescription")
        self.assertTrue(resolved[0].target_is_literal)

    def test_analytical_dimension_documented_by_bias_is_dropped(self):
        pipeline = _make_bare_pipeline()
        age = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}concept_age_g1", label="Age")

        relation = Relation(
            source=age.uri, relation="documented", target="systematic bias",
            metadata={"subject_text": "Age", "object_text": "systematic bias", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "Outcomes are documented/broken down by race, sex, age, and insurance type, revealing systematic bias."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[age], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(resolved, [])

    def test_non_dimension_subject_documented_relation_is_kept(self):
        """Control: a genuine agent-like subject under 'documented' must
        not be caught by the analytical-dimension guard."""
        pipeline = _make_bare_pipeline()
        report = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}concept_audit_report_g1", label="The Audit Report")

        relation = Relation(
            source=report.uri, relation="documented", target="systematic bias",
            metadata={"subject_text": "The Audit Report", "object_text": "systematic bias", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "The Audit Report documented systematic bias in the system."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[report], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)


class EntityTypingTests(unittest.TestCase):
    """Section H: entity type assignment must be validated against textual
    signal, and unambiguous calendar dates must not become generic
    entities."""

    def _typed_classes(self, *, label, entity_type):
        from rdflib import Namespace, RDF, URIRef
        from serializers.rdf_serializer import RDFSerializer

        etd_ns = Namespace(Config.NAMESPACES["etd"])
        uri = f"{Config.NAMESPACES['etd']}entity_h_{abs(hash((label, entity_type)))}"
        anchor_uri = f"{Config.NAMESPACES['etd']}entity_h_anchor_{abs(hash((label, entity_type)))}"
        entities = [
            Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.8, metadata={}),
            Entity(uri=anchor_uri, label="Anchor", entity_type="Concept", confidence=0.8, metadata={}),
        ]
        # A relation is required so the entity survives prune_isolated_entities()
        # (which runs during RDFSerializer construction and removes any
        # entity with no relation-based connection) -- otherwise even a
        # correctly-typed entity would show an empty class set for reasons
        # unrelated to the typing logic under test.
        relations = [
            {
                "source": uri, "relation": "mentions", "target": anchor_uri, "confidence": 0.9,
                "metadata": {"discourse_id": "d1", "field": "answer.description", "chunk_index": 0, "source": "llm_relation_extractor"},
            }
        ]
        ann = _base_annotations(relations)
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(entities), namespaces=Config.NAMESPACES)
        g = serializer.graph
        classes = {str(t).rsplit("#", 1)[-1] for t in g.objects(URIRef(uri), RDF.type)}
        return classes

    def test_process_labeled_bias_is_retyped_algorithmic_process(self):
        classes = self._typed_classes(label="Algorithmic Decision-Making", entity_type="AlgorithmicBias")
        self.assertIn("AlgorithmicProcess", classes)
        self.assertNotIn("AlgorithmicBias", classes)

    def test_genuine_bias_label_keeps_algorithmicbias(self):
        classes = self._typed_classes(label="Racial Bias in Sentencing", entity_type="AlgorithmicBias")
        self.assertIn("AlgorithmicBias", classes)

    def test_model_cards_retyped_resource(self):
        classes = self._typed_classes(label="Model Cards", entity_type="AISystem")
        self.assertIn("Resource", classes)
        self.assertNotIn("AISystem", classes)

    def test_genuine_ai_system_keeps_aisystem(self):
        classes = self._typed_classes(label="GPT-4 Model", entity_type="AISystem")
        self.assertIn("AISystem", classes)

    def test_classify_date_candidate_recognizes_unambiguous_dates(self):
        from utils.quantity_literal_classifier import classify_date_candidate
        self.assertEqual(classify_date_candidate("2024-03-15"), {"datatype": "xsd:date", "value": "2024-03-15"})
        self.assertEqual(classify_date_candidate("March 15, 2024"), {"datatype": "xsd:date", "value": "2024-03-15"})
        self.assertEqual(classify_date_candidate("March 2024"), {"datatype": "xsd:gYearMonth", "value": "2024-03"})

    def test_classify_date_candidate_leaves_bare_year_and_named_years_alone(self):
        from utils.quantity_literal_classifier import classify_date_candidate
        self.assertIsNone(classify_date_candidate("2024"))
        self.assertIsNone(classify_date_candidate("2025 Census"))

    def test_publication_date_becomes_date_literal_not_generic_entity(self):
        pipeline = _make_bare_pipeline()
        paper = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}publication_paper_h1", label="Fairness In ML Paper")

        relation = Relation(
            source=paper.uri, relation="hasPublicationDate", target="March 15, 2024",
            metadata={"subject_text": "Fairness in ML Paper", "object_text": "March 15, 2024", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "The paper was published on March 15, 2024."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[paper], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)
        out = resolved[0]
        self.assertTrue(out.target_is_literal)
        self.assertEqual(out.target, "2024-03-15")


class EvidenceEntailmentGroundingTests(unittest.TestCase):
    """Section I: a bare URL is not, by itself, sufficient evidence for a
    rich factual claim."""

    def test_url_only_evidence_dropped_without_broader_context(self):
        pipeline = _make_bare_pipeline()
        github = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}resource_github_analysis_i1", label="GitHub Analysis")

        relation = Relation(
            source=github.uri, relation="analyzed", target="100+ million rideshare samples",
            metadata={
                "subject_text": "GitHub Analysis", "object_text": "100+ million rideshare samples",
                "discourse_id": "d1", "evidence_quote": "https://github.com/example/rideshare-analysis",
            },
        )
        unit = {"start_offset": 0, "text": "https://github.com/example/rideshare-analysis"}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[github], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(resolved, [])

    def test_url_only_evidence_upgraded_to_broader_context_when_available(self):
        pipeline = _make_bare_pipeline()
        github = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}resource_github_analysis_i2", label="GitHub Analysis")

        broader = "GitHub Analysis analyzed 100+ million rideshare samples according to the published study."
        relation = Relation(
            source=github.uri, relation="analyzed", target="100+ million rideshare samples",
            metadata={
                "subject_text": "GitHub Analysis", "object_text": "100+ million rideshare samples",
                "discourse_id": "d1", "evidence_quote": "https://github.com/example/rideshare-analysis",
                "source_context": broader,
            },
        )
        unit = {"start_offset": 0, "text": broader}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[github], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].metadata.get("evidence"), broader)
        self.assertTrue(resolved[0].metadata.get("evidence_url_only_upgraded_to_context"))

    def test_url_only_evidence_allowed_for_source_reference_predicate(self):
        """Control: when the URL genuinely IS the claim (a
        source/reference-type predicate), it is not treated as
        insufficient."""
        pipeline = _make_bare_pipeline()
        doc = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}resource_doc_i3", label="Project Homepage")

        relation = Relation(
            source=doc.uri, relation="availableFrom", target="https://example.org/project",
            target_is_literal=True, target_literal="https://example.org/project",
            metadata={
                "subject_text": "Project Homepage", "object_text": "https://example.org/project",
                "discourse_id": "d1", "evidence_quote": "https://example.org/project",
            },
        )
        unit = {"start_offset": 0, "text": "https://example.org/project"}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[doc], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)


class RedundantMalformedRelationTests(unittest.TestCase):
    """Section J: a malformed relation must not be retained merely because
    a better relation from the same evidence also exists."""

    def test_population_group_object_dropped_when_abstract_finding_sibling_exists(self):
        pipeline = _make_bare_pipeline()
        aes = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}system_aes_j1", label="AES Systems", entity_type="AISystem")

        evidence = "AES Systems show systematic bias against students from marginalized groups."
        rel_bias = Relation(
            source=aes.uri, relation="show", target="systematic bias",
            metadata={"subject_text": "AES Systems", "object_text": "systematic bias", "discourse_id": "d1", "evidence_quote": evidence},
        )
        rel_group = Relation(
            source=aes.uri, relation="show", target="students from marginalized groups",
            metadata={"subject_text": "AES Systems", "object_text": "students from marginalized groups", "discourse_id": "d1", "evidence_quote": evidence},
        )
        unit = {"start_offset": 0, "text": evidence}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[rel_bias, rel_group], local_entities=[aes], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].metadata.get("object_text"), "systematic bias")

    def test_population_group_object_kept_without_abstract_sibling(self):
        """Control: a population-group object with no abstract-finding
        sibling relation is a potentially genuine, distinct fact and must
        survive untouched."""
        pipeline = _make_bare_pipeline()
        aes = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}system_aes_j2", label="AES Systems", entity_type="AISystem")

        relation = Relation(
            source=aes.uri, relation="show", target="students from marginalized groups",
            metadata={"subject_text": "AES Systems", "object_text": "students from marginalized groups", "discourse_id": "d1", "evidence_quote": "AES systems are used to grade students from marginalized groups."},
        )
        unit = {"start_offset": 0, "text": "AES systems are used to grade students from marginalized groups."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[aes], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)


def _make_synthetic_domain_profile():
    """A second, deliberately unrelated domain profile (legal-filings
    flavoured) used to prove the CORE mechanisms in pipeline.py /
    rdf_serializer.py are genuinely profile-driven, not AI-ethics-specific.
    Every value here is chosen to share NO vocabulary with the AI-ethics
    profile in domain_profiles.py."""
    import re as _re
    from domain_profiles import DomainProfile, ClassCorrectionRule

    return DomainProfile(
        name="synthetic_legal_test_domain",
        category_signals={
            "population_group": _re.compile(r"\b(litigants|plaintiffs|jurors|witnesses)\b", _re.I),
            "analytical_dimension": _re.compile(r"^(jurisdiction|venue|filing year|case type)$", _re.I),
            "abstract_finding": _re.compile(r"\b(malpractice|negligence|breach of contract)\b", _re.I),
        },
        category_types={
            "population_group": {"PopulationGroup"},
            "abstract_finding": {"LegalFinding"},
        },
        gazetteers={
            "hosting_platform_names": {"courtlistener", "pacer", "westlaw"},
            # This domain's own fuzzy-merge blocklist -- shares no terms
            # with AI_ETHICS_PROFILE's "generic_merge_blocklist".
            "generic_merge_blocklist": {"tort", "liability"},
        },
        predicate_bindings={"hosting_relation": "hostedBy"},
        class_correction_rules=[
            ClassCorrectionRule(
                declared_class="PrecedentCase",
                required_signal=_re.compile(r"\b(ruling|holding|verdict)\b", _re.I),
                fallback_class="Concept",
            ),
        ],
        # Section 1 (surface-entity type inference): a completely
        # different bucket/type scheme, sharing no vocabulary with the
        # AI-ethics profile's "system"/"dataset"/"group"/"risk"/"concept"
        # buckets.
        surface_type_buckets={
            "litigation_party": {"plaintiffs", "plaintiff", "defendants", "defendant", "witnesses", "witness"},
            "case_document": {"filing", "filings", "brief", "briefs", "motion", "motions"},
        },
        surface_type_overrides={"brief": "LegalBrief", "briefs": "LegalBrief"},
        surface_type_bucket_defaults={
            "litigation_party": "Litigant",
            "case_document": "CaseFiling",
        },
        # Section 4 (surface trimming): this domain's own prefix pattern,
        # applied to its own "litigation_party" bucket.
        surface_trim_prefix_patterns={
            "litigation_party": [_re.compile(r"^alleged\s+", _re.I)],
        },
        # Section 5 (lexical entity-recovery typing): this domain's own
        # trigger/type mappings, sharing no vocabulary with AI-ethics's.
        lexical_recovery_terms={
            "due diligence": "Governance",
            "chain of custody": "Governance",
        },
        # Section 3 (shared entity-family classifier): an entirely
        # different family taxonomy -- family NAMES themselves need not
        # match the AI-ethics profile's at all, proving the classifier is
        # genuinely generic.
        entity_family_types={
            "discourse": {"discourseunit", "theme", "question", "answer", "document"},
            "vote": {"vote"},
            "dataset": {"dataset"},
            "case_filing": {"filing", "brief", "motion"},
            "litigant_family": {"plaintiff", "defendant", "witness"},
            "finding_family": {"legalfinding", "malpracticefinding"},
        },
        entity_family_resource_exempt_types={"filing", "brief", "motion"},
    )


class DomainProfileArchitectureTests(unittest.TestCase):
    """Cross-domain architectural-cleanup pass: the domain-profile layer
    (domain_profiles.py) and its consumption by pipeline.py /
    serializers/rdf_serializer.py.

    A. the default ("ai_ethics") profile reproduces every current
       semantic-quality behaviour exactly.
    B. a synthetic second-domain profile with entirely different
       vocabulary can be loaded and drives the SAME generic mechanisms,
       with no change to pipeline.py / rdf_serializer.py.
    C. AI-ethics-specific terms do not fire under the synthetic profile
       unless justified by ontology/entity typing.
    D. generic CORE mechanisms (independent of any profile) keep working
       under both profiles.
    """

    # ---- A: default profile reproduces current behaviour ----------------

    def test_default_active_profile_is_ai_ethics(self):
        from config import Config
        from domain_profiles import get_active_profile
        self.assertEqual(getattr(Config, "DOMAIN_PROFILE", None), "ai_ethics")
        self.assertEqual(get_active_profile().name, "ai_ethics")

    def test_ai_ethics_profile_reproduces_known_vocabulary(self):
        from domain_profiles import get_active_profile
        profile = get_active_profile()
        self.assertTrue(profile.matches_category_signal("population_group", "Black defendants"))
        self.assertTrue(profile.matches_category_signal("analytical_dimension", "insurance type"))
        self.assertTrue(profile.matches_category_signal("abstract_finding", "systematic bias"))
        self.assertTrue(profile.matches_gazetteer("hosting_platform_names", "PhysioNet"))
        self.assertEqual(profile.predicate_for("hosting_relation"), "hostedBy")
        rule_classes = {r.declared_class for r in profile.class_correction_rules}
        self.assertEqual(rule_classes, {"AlgorithmicBias", "AISystem"})

    def test_rdf_serializer_defaults_to_ai_ethics_profile(self):
        from serializers.rdf_serializer import RDFSerializer
        ann = _base_annotations([])
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry([]), namespaces=Config.NAMESPACES)
        self.assertEqual(serializer.domain_profile.name, "ai_ethics")

    # ---- B + C: synthetic profile drives the same generic mechanisms ----

    def test_synthetic_profile_drives_comparator_guard_with_unrelated_vocabulary(self):
        """Section F1's comparator-misattribution guard fires on the
        synthetic profile's own population vocabulary ("litigants"),
        which shares no words with the AI-ethics profile -- proving the
        guard itself (pipeline.py) contains no AI-ethics vocabulary."""
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        group_a = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}group_litigants_a", label="Litigant Group A")

        evidence = "Litigants from group A were twice as likely as litigants from group B to be found at fault."
        relation = Relation(
            source=group_a.uri, relation="labeled", target="litigants from group b",
            metadata={"subject_text": "litigants from group A", "object_text": "litigants from group B",
                      "discourse_id": "d1", "evidence_quote": evidence},
        )
        unit = {"start_offset": 0, "text": evidence}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[group_a], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(resolved, [])

    def test_ai_ethics_terms_do_not_fire_under_synthetic_profile_without_ontology_type(self):
        """The exact AI-ethics-shaped defect ("Black defendants" vs "white
        defendants") must NOT be caught by the comparator guard once the
        synthetic profile is active, because that profile's own
        population-group vocabulary does not include "defendants" and the
        entities are not ontology-typed as a population group either."""
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        black_def = _register_entity(
            pipeline, uri=f"{Config.NAMESPACES['etd']}group_black_defendants_synth1",
            label="Black Defendants", entity_type="Concept",
        )

        evidence = "Black defendants were twice as likely as white defendants to be labelled high risk."
        relation = Relation(
            source=black_def.uri, relation="labeled", target="white defendants",
            metadata={"subject_text": "Black defendants", "object_text": "white defendants",
                      "discourse_id": "d1", "evidence_quote": evidence},
        )
        unit = {"start_offset": 0, "text": evidence}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[black_def], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1, "AI-ethics vocabulary must not leak into an unrelated active profile")

    def test_ontology_type_satisfies_category_ahead_of_profile_lexical_hint(self):
        """Priority order: ontology/entity type > profile lexical hint. An
        entity explicitly typed with the generic ontology class
        "PopulationGroup" must satisfy the category even though the
        synthetic profile's own lexical signal for "population_group"
        does not mention "defendants" at all."""
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        black_def = _register_entity(
            pipeline, uri=f"{Config.NAMESPACES['etd']}group_black_defendants_synth2",
            label="Black Defendants", entity_type="PopulationGroup",
        )
        white_def = _register_entity(
            pipeline, uri=f"{Config.NAMESPACES['etd']}group_white_defendants_synth2",
            label="White Defendants", entity_type="PopulationGroup",
        )

        evidence = "Black defendants were twice as likely as white defendants to be labelled high risk."
        relation = Relation(
            source=black_def.uri, relation="labeled", target=white_def.uri,
            metadata={"subject_text": "Black Defendants", "object_text": "White Defendants",
                      "discourse_id": "d1", "evidence_quote": evidence},
        )
        unit = {"start_offset": 0, "text": evidence}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[black_def, white_def], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(resolved, [], "ontology typing alone must be enough to satisfy the category, regardless of profile vocabulary")

    def test_synthetic_profile_hosting_gazetteer_and_class_correction(self):
        """hostedBy correction and the generic class-correction mechanism
        both operate correctly on a profile with entirely different
        platform names and class-correction rules."""
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        doc = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}doc_ruling_synth1", label="Case Filing 12-345")

        relation = Relation(
            source=doc.uri, relation="is", target="CourtListener",
            metadata={"subject_text": "Case Filing 12-345", "object_text": "CourtListener", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "Case Filing 12-345 is CourtListener."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[doc], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].relation, "hostedBy")

        from serializers.rdf_serializer import RDFSerializer
        synthetic = _make_synthetic_domain_profile()
        # PrecedentCase with no ruling/holding/verdict signal -> falls back
        # to Concept (this profile's rule has no secondary signal).
        no_evidence = RDFSerializer.__new__(RDFSerializer)
        no_evidence.domain_profile = synthetic
        self.assertEqual(no_evidence._corrected_entity_class("PrecedentCase", "Smith v. Jones Case Summary"), "Concept")
        self.assertEqual(no_evidence._corrected_entity_class("PrecedentCase", "The court's holding in this case"), "PrecedentCase")
        # A class this profile has no rule for passes through unchanged.
        self.assertEqual(no_evidence._corrected_entity_class("AlgorithmicBias", "Algorithmic Decision-Making"), "AlgorithmicBias")

    # ---- D: generic CORE mechanisms are unaffected by the active profile

    def test_core_self_loop_guard_unaffected_by_active_profile(self):
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        audit_trails = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}concept_audit_trails_synth1", label="Audit Trails")

        relation = Relation(
            source=audit_trails.uri, relation="make", target=audit_trails.uri,
            metadata={"subject_text": "Audit Trails", "object_text": "Audit Trails", "discourse_id": "d1"},
        )
        unit = {"start_offset": 0, "text": "Audit trails make audit trails auditable."}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[audit_trails], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(resolved, [])

    def test_core_date_and_respectively_handling_unaffected_by_active_profile(self):
        from utils.quantity_literal_classifier import classify_date_candidate
        # Pure function, takes no profile at all -- confirms date handling
        # is CORE regardless of which profile object exists anywhere.
        self.assertEqual(classify_date_candidate("March 15, 2024"), {"datatype": "xsd:date", "value": "2024-03-15"})

        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        litigant_a = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}group_litigants_resp1", label="Group A Litigants")
        litigant_b = _register_entity(pipeline, uri=f"{Config.NAMESPACES['etd']}group_litigants_resp2", label="Group B Litigants")

        evidence = "Group A and Group B litigants received 12.5 and 8.2 months in damages respectively."
        rel_a = Relation(
            source=litigant_a.uri, relation="receives", target="12.5 and 8.2 months in damages",
            metadata={"subject_text": "Group A litigants", "object_text": "12.5 and 8.2 months in damages",
                      "discourse_id": "d1", "evidence_quote": evidence},
        )
        rel_b = Relation(
            source=litigant_b.uri, relation="receives", target="12.5 and 8.2 months in damages",
            metadata={"subject_text": "Group B litigants", "object_text": "12.5 and 8.2 months in damages",
                      "discourse_id": "d1", "evidence_quote": evidence},
        )
        unit = {"start_offset": 0, "text": evidence}
        resolved = pipeline._resolve_semantic_unit_relations(
            relations=[rel_a, rel_b], local_entities=[litigant_a, litigant_b], discourse_text=unit["text"],
            discourse_id="d1", field="answer.description", origin_key="answer.description|d1",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(len(resolved), 2)
        by_subject = {str(r.source): r for r in resolved}
        a_rel = by_subject[litigant_a.uri]
        b_rel = by_subject[litigant_b.uri]
        self.assertIn("12.5", a_rel.metadata.get("object_text", ""))
        self.assertIn("8.2", b_rel.metadata.get("object_text", ""))


class GenericMergeBlocklistProfileSplitTests(unittest.TestCase):
    """DO_NOT_MERGE cleanup: utils/canonical_linker.py::DO_NOT_MERGE keeps
    only CORE_GENERIC terms (data/dataset(s)/model/system/audit(s));
    AI-ethics-specific terms (ai/ml/fair/fairness/transparency/
    accountability/interpretability/equity/bias(es)/discrimination) now
    live in the active domain profile's "generic_merge_blocklist"
    gazetteer, consulted by CanonicalLinker._can_fuzzy_merge alongside the
    core list."""

    def _linker(self, domain_profile=None):
        import tempfile
        from utils.canonical_linker import CanonicalLinker
        return CanonicalLinker(registry=None, output_dir=tempfile.mkdtemp(), domain_profile=domain_profile)

    # NOTE: the two labels in each pair below are deliberately both
    # bare/single-content-token (e.g. "Bias"/"Biases", not "Bias"/"Biased
    # Outcome Report") so that the UNRELATED bare-generic-vs-specific
    # guard (_is_bare_generic_label, category E) never fires here -- these
    # tests isolate the blocklist gate specifically.

    def test_ai_ethics_blocked_term_protected_under_ai_ethics_profile(self):
        """"bias" (moved out of core DO_NOT_MERGE) is still blocked by
        default, because the default active profile is ai_ethics."""
        linker = self._linker()  # defaults to the active (ai_ethics) profile
        a = Entity(uri="urn:e1", label="Bias", entity_type="Concept", confidence=0.8, metadata={})
        b = Entity(uri="urn:e2", label="Biases", entity_type="Concept", confidence=0.8, metadata={})
        self.assertFalse(linker._can_fuzzy_merge(a, b, "Concept"))

    def test_ai_ethics_term_not_protected_under_synthetic_profile(self):
        """The exact same "bias" term must NOT be blocked once an
        unrelated synthetic profile (with no "bias" entry of its own) is
        active -- AI-ethics vocabulary does not leak into another
        domain's configuration."""
        linker = self._linker(domain_profile=_make_synthetic_domain_profile())
        a = Entity(uri="urn:e1", label="Bias", entity_type="Concept", confidence=0.8, metadata={})
        b = Entity(uri="urn:e2", label="Biases", entity_type="Concept", confidence=0.8, metadata={})
        # Not blocked by the (reduced) core list, not blocked by the
        # synthetic profile's blocklist either -- whether it actually
        # merges also depends on the similarity score, which this method
        # does not compute; we only assert the blocklist gate itself does
        # not reject the pair.
        self.assertTrue(linker._can_fuzzy_merge(a, b, "Concept"))

    def test_core_generic_term_protected_under_both_profiles(self):
        """"model" is CORE_GENERIC -- still in canonical_linker.py's own
        DO_NOT_MERGE -- so it stays blocked regardless of which profile is
        active."""
        for profile in (None, _make_synthetic_domain_profile()):
            linker = self._linker(domain_profile=profile)
            a = Entity(uri="urn:e1", label="Model", entity_type="Concept", confidence=0.8, metadata={})
            b = Entity(uri="urn:e2", label="Models", entity_type="Concept", confidence=0.8, metadata={})
            self.assertFalse(linker._can_fuzzy_merge(a, b, "Concept"), f"failed for profile={profile}")

    def test_second_domain_specific_term_blocked_via_configuration_only(self):
        """"tort" is blocked ONLY because the synthetic profile's own
        generic_merge_blocklist says so -- canonical_linker.py was not
        modified to know this word, proving a completely different
        blocklist can be supplied purely through configuration."""
        synthetic = _make_synthetic_domain_profile()
        blocked_linker = self._linker(domain_profile=synthetic)
        a = Entity(uri="urn:e1", label="Tort", entity_type="Concept", confidence=0.8, metadata={})
        b = Entity(uri="urn:e2", label="Torts", entity_type="Concept", confidence=0.8, metadata={})
        self.assertFalse(blocked_linker._can_fuzzy_merge(a, b, "Concept"))

        # The AI-ethics profile has no opinion on "tort" -- not blocked
        # there (core list doesn't mention it, len>=3 so the length guard
        # doesn't catch it either).
        ai_ethics_linker = self._linker()
        self.assertTrue(ai_ethics_linker._can_fuzzy_merge(a, b, "Concept"))

    def test_legitimate_named_entities_still_merge_correctly(self):
        """End-to-end: a genuine cross-mention repeat (acronym + full
        name) must still merge under either profile -- the blocklist
        split does not collaterally break real canonicalization."""
        import tempfile
        from utils.canonical_linker import CanonicalLinker

        class _Registry:
            def __init__(self, entities):
                self.entities = {e.uri: e for e in entities}

            def get_all_entities(self):
                return list(self.entities.values())

            def replace_entity(self, old_uri, new_uri):
                if old_uri in self.entities and new_uri not in self.entities:
                    self.entities[new_uri] = self.entities[old_uri]

        for profile in (None, _make_synthetic_domain_profile()):
            e1 = Entity(uri="https://alfie-project.eu/etd-hub#hmda1", label="HMDA", entity_type="Regulation", confidence=0.8, metadata={})
            e2 = Entity(uri="https://alfie-project.eu/etd-hub#hmda2", label="Home Mortgage Disclosure Act", entity_type="Regulation", confidence=0.9, metadata={})
            reg = _Registry([e1, e2])
            linker = CanonicalLinker(registry=reg, output_dir=tempfile.mkdtemp(), domain_profile=profile)
            out = linker.run_full_registry()
            self.assertEqual(len(out), 1, f"failed for profile={profile}")


class EntityNormalizerMergeBlocklistProfileTests(unittest.TestCase):
    """EntityNormalizer.DO_NOT_MERGE_TERMS cleanup: same policy split as
    CanonicalLinker -- CORE_GENERIC terms stay hardcoded, AI-ethics terms
    come from the active domain profile's "generic_merge_blocklist"
    gazetteer (the SAME gazetteer CanonicalLinker consults)."""

    def _normalizer(self, domain_profile=None):
        from utils.entity_normalizer import EntityNormalizer
        return EntityNormalizer(registry=None, domain_profile=domain_profile)

    def test_bias_protected_under_ai_ethics_profile(self):
        norm = self._normalizer()  # defaults to the active (ai_ethics) profile
        self.assertTrue(norm.should_avoid_fuzzy_merge("Bias"))

    def test_bias_not_automatically_protected_under_synthetic_profile(self):
        norm = self._normalizer(domain_profile=_make_synthetic_domain_profile())
        self.assertFalse(norm.should_avoid_fuzzy_merge("Bias"))

    def test_model_protected_under_both_profiles(self):
        for profile in (None, _make_synthetic_domain_profile()):
            norm = self._normalizer(domain_profile=profile)
            self.assertTrue(norm.should_avoid_fuzzy_merge("Model"), f"failed for profile={profile}")

    def test_tort_protected_only_under_synthetic_profile(self):
        synthetic = _make_synthetic_domain_profile()
        self.assertTrue(self._normalizer(domain_profile=synthetic).should_avoid_fuzzy_merge("Tort"))
        self.assertFalse(self._normalizer().should_avoid_fuzzy_merge("Tort"))

    def test_legitimate_named_entity_normalization_works_under_both_profiles(self):
        for profile in (None, _make_synthetic_domain_profile()):
            norm = self._normalizer(domain_profile=profile)
            self.assertFalse(norm.should_avoid_fuzzy_merge("COMPAS"), f"failed for profile={profile}")
            self.assertFalse(norm.should_avoid_fuzzy_merge("Home Mortgage Disclosure Act"), f"failed for profile={profile}")
            # normalize_label's alias resolution is itself entirely
            # profile-independent -- unaffected either way.
            self.assertEqual(norm.normalize_label("hmda"), "Home Mortgage Disclosure Act")

    def test_canonical_linker_and_entity_normalizer_agree_on_classifications(self):
        """Both components consult exactly the same two sources (their own
        reduced CORE_GENERIC list, plus the active profile's
        generic_merge_blocklist gazetteer) -- they must never disagree on
        whether a given term is blocked."""
        import tempfile
        from utils.canonical_linker import CanonicalLinker

        terms = ["data", "model", "system", "audit", "bias", "fairness", "tort", "liability", "compas", "hmda"]
        for profile in (None, _make_synthetic_domain_profile()):
            normalizer = self._normalizer(domain_profile=profile)
            linker = CanonicalLinker(registry=None, output_dir=tempfile.mkdtemp(), domain_profile=profile)
            for term in terms:
                norm_result = normalizer.should_avoid_fuzzy_merge(term)
                linker_result = term in linker.DO_NOT_MERGE or linker.domain_profile.matches_gazetteer("generic_merge_blocklist", term)
                self.assertEqual(
                    norm_result, linker_result,
                    f"disagreement on {term!r} under profile={profile}: normalizer={norm_result}, linker={linker_result}",
                )


class RelationValidationCategoryUnificationTests(unittest.TestCase):
    """Architectural-cleanup pass: pipeline.py::_is_generic_risk_label/
    _is_generic_abstract_label/_label_has_group_cue and their byte-for-byte
    duplicates in utils/semantic_relation_policy.py both now delegate to
    the ONE shared domain_profiles.matches_semantic_category function --
    no vocabulary is hardcoded in either file."""

    def _pipeline(self, domain_profile=None):
        p = _make_bare_pipeline()
        if domain_profile is not None:
            p.domain_profile = domain_profile
        return p

    def _policy(self, domain_profile=None):
        from utils.semantic_relation_policy import SemanticRelationPolicy
        return SemanticRelationPolicy(domain_profile=domain_profile)

    def test_ai_ethics_risk_term_affects_validation_under_ai_ethics_profile(self):
        pipeline = self._pipeline()  # default active (ai_ethics) profile
        policy = self._policy()
        self.assertTrue(pipeline._is_generic_risk_label("Bias"))
        self.assertTrue(policy._is_generic_risk_label("Bias"))
        self.assertTrue(pipeline._is_generic_abstract_label("Fairness"))
        self.assertTrue(policy._is_generic_abstract_label("Fairness"))
        self.assertTrue(pipeline._label_has_group_cue("Black Defendants"))
        self.assertTrue(policy._label_has_group_cue("Black Defendants"))

    def test_ai_ethics_terms_do_not_automatically_affect_validation_under_synthetic_profile(self):
        synthetic = _make_synthetic_domain_profile()
        pipeline = self._pipeline(domain_profile=synthetic)
        policy = self._policy(domain_profile=synthetic)
        self.assertFalse(pipeline._is_generic_risk_label("Bias"))
        self.assertFalse(policy._is_generic_risk_label("Bias"))
        self.assertFalse(pipeline._is_generic_abstract_label("Fairness"))
        self.assertFalse(policy._is_generic_abstract_label("Fairness"))
        self.assertFalse(pipeline._label_has_group_cue("Black Defendants"))
        self.assertFalse(policy._label_has_group_cue("Black Defendants"))

    def test_synthetic_domain_equivalents_supplied_entirely_through_configuration(self):
        """The synthetic profile's OWN finding/group vocabulary (shares no
        words with AI-ethics) drives the exact same functions, with zero
        changes to pipeline.py or utils/semantic_relation_policy.py."""
        synthetic = _make_synthetic_domain_profile()
        pipeline = self._pipeline(domain_profile=synthetic)
        policy = self._policy(domain_profile=synthetic)
        self.assertTrue(pipeline._is_generic_risk_label("Negligence"))
        self.assertTrue(policy._is_generic_risk_label("Negligence"))
        self.assertTrue(pipeline._label_has_group_cue("Litigants"))
        self.assertTrue(policy._label_has_group_cue("Litigants"))
        # Core_generic finding-context nouns still apply under either profile.
        self.assertTrue(pipeline._is_generic_abstract_label("Dataset"))
        self.assertTrue(policy._is_generic_abstract_label("Dataset"))

    def test_ontology_type_takes_priority_over_lexical_profile_hint(self):
        """An entity whose OWN type is one of the synthetic profile's
        configured "abstract_finding" type names must satisfy the category
        even though its label matches none of that profile's lexical
        vocabulary at all."""
        from domain_profiles import matches_semantic_category
        synthetic = _make_synthetic_domain_profile()
        self.assertFalse(synthetic.matches_category_signal("abstract_finding", "Totally Unrelated Label"))
        self.assertTrue(
            matches_semantic_category(
                synthetic, "abstract_finding", "Totally Unrelated Label", entity_type="LegalFinding",
            )
        )

    def test_pipeline_and_policy_never_disagree_on_category_decisions(self):
        """Both layers consult exactly the same shared function and the
        same active profile -- for any label, under any profile, they
        must return identical answers. This is what prevents a relation
        from being accepted by one layer's domain/range check and
        rejected by the other's because of duplicated, drifting category
        logic."""
        labels = ["Bias", "Intersectional Bias", "Fairness", "Black Defendants",
                  "Negligence", "Litigants", "Dataset", "Xyzzy Unrelated Term"]
        for profile in (None, _make_synthetic_domain_profile()):
            pipeline = self._pipeline(domain_profile=profile)
            policy = self._policy(domain_profile=profile)
            for label in labels:
                self.assertEqual(
                    pipeline._is_generic_risk_label(label), policy._is_generic_risk_label(label),
                    f"_is_generic_risk_label disagreement on {label!r} under profile={profile}",
                )
                self.assertEqual(
                    pipeline._is_generic_abstract_label(label), policy._is_generic_abstract_label(label),
                    f"_is_generic_abstract_label disagreement on {label!r} under profile={profile}",
                )
                self.assertEqual(
                    pipeline._label_has_group_cue(label), policy._label_has_group_cue(label),
                    f"_label_has_group_cue disagreement on {label!r} under profile={profile}",
                )

    def test_generic_core_behavior_unchanged(self):
        """A specific, non-generic finding label ("Intersectional Bias")
        must NOT be treated as a generic risk/abstract label under either
        profile -- this is the exact regression this refactor
        (exact-match vs substring-match) had to avoid."""
        for profile in (None, _make_synthetic_domain_profile()):
            pipeline = self._pipeline(domain_profile=profile)
            self.assertFalse(pipeline._is_generic_risk_label("Intersectional Bias"), f"failed for profile={profile}")
            self.assertFalse(pipeline._is_generic_abstract_label("Intersectional Bias"), f"failed for profile={profile}")
            # An entirely unrelated label must never match anything, under
            # any profile ("no assumption" is always the safe default).
            self.assertFalse(pipeline._is_generic_risk_label("Xyzzy Unrelated Term"), f"failed for profile={profile}")
            self.assertFalse(pipeline._label_has_group_cue("Xyzzy Unrelated Term"), f"failed for profile={profile}")


class SurfaceEntityAndFamilyCrossDomainTests(unittest.TestCase):
    """FINAL bounded cross-domain cleanup: surface-entity type inference
    (_surface_entity_head_buckets/_infer_surface_entity_type/
    _surface_entity_bucket), _generic_surface_label,
    _trim_recovered_surface, the lexical entity-recovery dict, and the
    shared entity-semantic-family classifier."""

    # ---- A: AI-ethics profile preserves current behaviour --------------

    def test_ai_ethics_surface_type_inference_unchanged(self):
        pipeline = _make_bare_pipeline()
        self.assertEqual(pipeline._infer_surface_entity_type("systematic bias"), "AlgorithmicBias")
        self.assertEqual(pipeline._infer_surface_entity_type("predictive system"), "AISystem")
        self.assertEqual(pipeline._infer_surface_entity_type("the underlying model"), "Model")
        self.assertEqual(pipeline._infer_surface_entity_type("affected patients"), "DemographicGroup")
        self.assertEqual(pipeline._infer_surface_entity_type("historical workforce data"), "Dataset")
        self.assertIsNone(pipeline._infer_surface_entity_type("something entirely unrelated"))

    def test_ai_ethics_generic_surface_label_unchanged(self):
        pipeline = _make_bare_pipeline()
        self.assertTrue(pipeline._generic_surface_label("system"))  # CORE
        self.assertTrue(pipeline._generic_surface_label("bias"))  # DOMAIN_PROFILE
        self.assertFalse(pipeline._generic_surface_label("COMPAS"))

    def test_ai_ethics_family_classification_unchanged(self):
        pipeline = _make_bare_pipeline()
        bias_entity = Entity(uri="urn:e1", label="Racial Bias In Sentencing", entity_type="AlgorithmicBias", confidence=0.9, metadata={})
        self.assertEqual(pipeline._entity_semantic_family(bias_entity), "risk")

        from utils.semantic_relation_policy import SemanticRelationPolicy
        policy = SemanticRelationPolicy()
        self.assertEqual(policy.entity_family(bias_entity), "risk_or_harm")

    # ---- B + E: synthetic profile via configuration only ----------------

    def test_synthetic_profile_surface_type_inference_via_configuration_only(self):
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        self.assertEqual(pipeline._infer_surface_entity_type("the named defendant"), "Litigant")
        self.assertEqual(pipeline._infer_surface_entity_type("the filed brief"), "LegalBrief")
        # AI-ethics-shaped surface text has no meaning under this profile.
        self.assertIsNone(pipeline._infer_surface_entity_type("systematic bias"))

    def test_synthetic_profile_surface_trim_via_configuration_only(self):
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        trimmed, _delta = pipeline._trim_recovered_surface(
            "alleged defendant", bucket="litigation_party", head_norm="defendant",
        )
        self.assertEqual(trimmed, "defendant")

    def test_synthetic_profile_family_classification_via_configuration_only(self):
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        defendant = Entity(uri="urn:e1", label="Jane Doe", entity_type="defendant", confidence=0.9, metadata={})
        # Family NAME here ("litigant_family") is entirely this profile's
        # own choice -- pipeline.py hardcodes none of it.
        self.assertEqual(pipeline._entity_semantic_family(defendant), "litigant_family")

    def test_ai_ethics_terms_do_not_influence_synthetic_profile_surface_inference(self):
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        for surface in ("systematic bias", "predictive system", "affected patients", "historical workforce data"):
            self.assertIsNone(pipeline._infer_surface_entity_type(surface), surface)

    # ---- C: ontology typing overrides lexical hints ---------------------

    def test_ontology_type_overrides_lexical_hint_in_family_classification(self):
        """An entity explicitly typed with one of the synthetic profile's
        OWN configured family types must classify accordingly even though
        its label matches no lexical signal in either profile."""
        pipeline = _make_bare_pipeline()
        pipeline.domain_profile = _make_synthetic_domain_profile()
        entity = Entity(uri="urn:e1", label="Totally Unrelated Label Text", entity_type="plaintiff", confidence=0.9, metadata={})
        self.assertEqual(pipeline._entity_semantic_family(entity), "litigant_family")

    # ---- D: pipeline and policy agree (via the one shared decision) -----

    def test_pipeline_and_policy_family_translations_agree_on_the_shared_decision(self):
        """Both _entity_semantic_family (pipeline.py) and entity_family
        (SemanticRelationPolicy) are deterministic translations of the
        SAME underlying domain_profiles.classify_entity_semantic_family
        decision -- verified directly here so the two can never silently
        diverge without this test catching it."""
        from domain_profiles import classify_entity_semantic_family
        from utils.semantic_relation_policy import SemanticRelationPolicy

        entities = [
            Entity(uri="urn:e1", label="Racial Bias In Sentencing", entity_type="AlgorithmicBias", confidence=0.9, metadata={}),
            Entity(uri="urn:e2", label="Black Defendants", entity_type="PopulationGroup", confidence=0.9, metadata={}),
            Entity(uri="urn:e3", label="Some Regulation", entity_type="Law", confidence=0.9, metadata={}),
            Entity(uri="urn:e4", label="A Metric", entity_type="Metric", confidence=0.9, metadata={}),
            Entity(uri="urn:e5", label="Algorithmic Decision-Making", entity_type="AlgorithmicBias", confidence=0.9, metadata={}),
        ]
        for profile in (None, _make_synthetic_domain_profile()):
            pipeline = _make_bare_pipeline()
            if profile is not None:
                pipeline.domain_profile = profile
            policy = SemanticRelationPolicy(domain_profile=profile)
            for entity in entities:
                active_profile = pipeline.domain_profile
                canonical = classify_entity_semantic_family(entity, profile=active_profile)
                self.assertEqual(
                    pipeline._entity_semantic_family(entity),
                    pipeline._CANONICAL_TO_LEGACY_FAMILY.get(canonical, canonical),
                    f"pipeline translation drifted for {entity.label!r}",
                )
                self.assertEqual(
                    policy.entity_family(entity),
                    policy._CANONICAL_TO_LEGACY_FAMILY.get(canonical, canonical),
                    f"policy translation drifted for {entity.label!r}",
                )

    # ---- F: CORE generic behaviour is profile-independent ---------------

    def test_core_dataset_and_discourse_family_classification_unaffected_by_profile(self):
        dataset_entity = Entity(uri="urn:e1", label="Some Dataset", entity_type="Dataset", confidence=0.9, metadata={})
        for profile in (None, _make_synthetic_domain_profile()):
            pipeline = _make_bare_pipeline()
            if profile is not None:
                pipeline.domain_profile = profile
            self.assertEqual(pipeline._entity_semantic_family(dataset_entity), "dataset", f"failed for profile={profile}")

    def test_core_generic_surface_label_structural_nouns_unaffected_by_profile(self):
        for profile in (None, _make_synthetic_domain_profile()):
            pipeline = _make_bare_pipeline()
            if profile is not None:
                pipeline.domain_profile = profile
            self.assertTrue(pipeline._generic_surface_label("model"), f"failed for profile={profile}")
            self.assertTrue(pipeline._generic_surface_label("dataset"), f"failed for profile={profile}")


if __name__ == "__main__":
    unittest.main()
