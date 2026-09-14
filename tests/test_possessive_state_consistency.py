import unittest

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer


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
    pipeline.predicate_precision = PredicatePrecisionLayer()
    pipeline.coref_resolver = _EchoResolver()
    pipeline.wikidata_linker = None
    pipeline._discourse_chunk_spans = {}
    pipeline.semantic_policy = None
    return pipeline


def _unit(unit_id, text, start_offset):
    return {
        "unit_id": unit_id, "text": text, "start_offset": start_offset,
        "end_offset": start_offset + len(text), "section_category": "semantic", "unit_type": "sentence",
    }


class PossessiveStateConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()
        self.precision = PredicatePrecisionLayer()

    def _run_possessive(self, *, subject, unit_text, object_text, discourse_id="d1"):
        unit = _unit("Body:1", unit_text, 0)
        relation = Relation(
            source=subject.uri, relation="share", target=object_text,
            # allow_local_predicate=True matches real production relations
            # (llm_relation_extractor.py always sets it) -- without it,
            # "share" (an unregistered predicate) falls through the
            # unknown-predicate downgrade to relatedTo instead of surviving
            # as a local/unmapped predicate, which is not what this test is
            # about.
            metadata={
                "object_text": object_text, "discourse_id": discourse_id, "evidence_quote": unit_text,
                "allow_local_predicate": True,
            },
        )
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[subject], discourse_text=unit_text, discourse_id=discourse_id,
            field="answer.description", origin_key=f"answer.description|{discourse_id}", unit=unit, unit_text=unit_text,
        )
        final = self.pipeline._apply_possessive_endpoint_semantics(
            relations=resolved, entities=[subject], local_pool_entities=[subject], unit=unit, discourse_id=discourse_id,
        )
        return final[0]

    def test_entity_uri_endpoint_stays_coherent_through_refine_relations(self):
        subject = Entity(uri="https://kg/e/team", label="Team", entity_type="Organisation", confidence=0.9, metadata={})
        target = Entity(uri="https://kg/e/policy", label="Policy", entity_type="Policy", confidence=0.9, metadata={})
        relation = Relation(source=subject.uri, relation="uses", target=target.uri, metadata={"field": "answer.description"})
        refined = self.precision.refine_relations([relation], entities=[subject, target], discourse_text="")
        self.assertEqual(refined[0].target, target.uri)
        self.assertFalse(refined[0].target_is_literal)
        self.assertIsNone(refined[0].target_literal)

    def test_resolved_possessor_metadata_survives_refine_relations(self):
        subject = Entity(
            uri="https://kg/e/organisations_teams", label="Organisations Teams", entity_type="Concept",
            confidence=0.9, metadata={"discourse_id": "d1"},
        )
        rel = self._run_possessive(
            subject=subject,
            unit_text="if organisations/teams do this, share their approach publically.",
            object_text="their approach",
        )
        self.assertEqual(rel.target, "approach")
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.target_literal, "approach")
        self.assertEqual(rel.metadata.get("object_possessor_entity_uri"), subject.uri)
        self.assertEqual(rel.metadata.get("object_possessor_resolution_method"), "reflexive_subject")

        refined = self.precision.refine_relations([rel], entities=[subject], discourse_text=rel.context or "")
        r = refined[0]
        self.assertEqual(r.target, "approach")
        self.assertTrue(r.target_is_literal)
        self.assertEqual(r.target_literal, "approach")
        self.assertEqual(r.metadata.get("object_possessor_entity_uri"), subject.uri)
        self.assertEqual(r.metadata.get("object_possessor_resolution_method"), "reflexive_subject")

    def test_unresolved_possessor_endpoint_state_still_coherent(self):
        subject = Entity(
            uri="https://kg/e/applicants", label="Applicants", entity_type="Population",
            confidence=0.9, metadata={"discourse_id": "d2"},
        )
        rel = self._run_possessive(
            subject=subject,
            unit_text="what if this tool could be used to predict their own performance.",
            object_text="their own performance",
            discourse_id="d2",
        )
        # Whether or not the possessor itself resolves is the possessive
        # resolver's own, unchanged decision (here it happens to resolve
        # reflexively to the subject) -- what this invariant guarantees is
        # that the endpoint STATE
        # (target/target_is_literal/target_literal) stays internally
        # coherent regardless of the possessor-resolution outcome.
        self.assertEqual(rel.target, "performance")
        self.assertTrue(rel.target_is_literal)
        self.assertEqual(rel.target_literal, "performance")
        possessor_before = rel.metadata.get("object_possessor_entity_uri")

        refined = self.precision.refine_relations([rel], entities=[subject], discourse_text=rel.context or "")
        r = refined[0]
        self.assertEqual(r.target, "performance")
        self.assertTrue(r.target_is_literal)
        self.assertEqual(r.target_literal, "performance")
        self.assertEqual(r.metadata.get("object_possessor_entity_uri"), possessor_before)

    def test_no_stale_target_literal_after_refine(self):
        subject = Entity(uri="https://kg/e/subj", label="Subject", entity_type="Concept", confidence=0.9, metadata={})
        rel = Relation(
            source=subject.uri, relation="describes", target="approach", target_is_literal=True,
            target_literal="approach", metadata={"field": "answer.description", "allow_local_predicate": True},
        )
        refined = self.precision.refine_relations([rel], entities=[subject], discourse_text="")
        self.assertEqual(refined[0].target_literal, "approach")
        self.assertTrue(refined[0].target_is_literal)

    def test_object_type_preserved_through_refine_relations(self):
        subject = Entity(
            uri="https://kg/e/organisations_teams2", label="Organisations Teams", entity_type="Concept",
            confidence=0.9, metadata={"discourse_id": "d3"},
        )
        rel = self._run_possessive(
            subject=subject,
            unit_text="if organisations/teams do this, share their approach publically.",
            object_text="their approach",
            discourse_id="d3",
        )
        original_object_type = rel.metadata.get("object_type")
        refined = self.precision.refine_relations([rel], entities=[subject], discourse_text=rel.context or "")
        self.assertEqual(refined[0].metadata.get("object_type"), original_object_type)

    def test_serialized_rdf_object_is_literal_not_dropped(self):
        from rdflib import Literal, RDF, Namespace
        from serializers.rdf_serializer import RDFSerializer

        etd_base = Config.NAMESPACES["etd"]
        subject_uri = f"{etd_base}entity_organisations_teams"

        class _Registry:
            def get_all_entities(self):
                return [Entity(uri=subject_uri, label="Organisations Teams", entity_type="Concept", confidence=0.9, metadata={"discourse_id": "d4"})]

        ann = {
            "themes": {}, "questions": {},
            "answers": {
                "d4": {
                    "relations": [
                        {
                            "source": subject_uri, "relation": "share", "target": "approach",
                            "target_is_literal": True, "target_literal": "approach",
                            "confidence": 0.9,
                            "metadata": {
                                "discourse_id": "d4", "field": "answer.description",
                                "object_possessor_original_surface": "their approach",
                                "object_possessor_head_noun_phrase": "approach",
                                "object_possessor_entity_uri": subject_uri,
                            },
                        }
                    ]
                }
            },
            "votes": {}, "experts": {}, "documents": {}, "metadata": {},
        }
        serializer = RDFSerializer(annotations_dict=ann, registry=_Registry(), namespaces=Config.NAMESPACES)
        g = serializer.graph
        etd_ns = Namespace(etd_base)
        objects = list(g.objects(None, etd_ns.object))
        literal_objects = [o for o in objects if isinstance(o, Literal) and str(o) == "approach"]
        self.assertTrue(literal_objects, f"expected a literal 'approach' object triple, got: {objects}")


class RealPossessiveCasesRemainCorrectTests(unittest.TestCase):
    """answers/29 'their approach' and answers/24 'their own performance'
    negative controls, reproducing the real confirmed shapes."""

    def setUp(self):
        self.pipeline = _make_pipeline()
        self.precision = PredicatePrecisionLayer()

    def test_answers_29_their_approach_remains_correct(self):
        team = Entity(
            uri="https://alfie-project.eu/etd-hub#concept_organisations_teams_56bdf53a37",
            label="Organisations Teams", entity_type="Concept", confidence=0.9, metadata={"discourse_id": "29"},
        )
        unit_text = "I also think that if organisations/teams do this, share their approach publically."
        unit = _unit("Body:28", unit_text, 3691)
        relation = Relation(
            source=team.uri, relation="share", target="their approach",
            metadata={
                "object_text": "their approach", "discourse_id": "29", "evidence_quote": unit_text,
                "allow_local_predicate": True,
            },
        )
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[team], discourse_text=unit_text, discourse_id="29",
            field="answer.description", origin_key="answer.description|29", unit=unit, unit_text=unit_text,
        )
        final = self.pipeline._apply_possessive_endpoint_semantics(
            relations=resolved, entities=[team], local_pool_entities=[team], unit=unit, discourse_id="29",
        )
        refined = self.precision.refine_relations(final, entities=[team], discourse_text=unit_text)
        r = refined[0]
        self.assertEqual(r.target, "approach")
        self.assertTrue(r.target_is_literal)
        self.assertEqual(r.target_literal, "approach")
        self.assertEqual(r.metadata.get("object_possessor_entity_uri"), team.uri)
        self.assertEqual(r.metadata.get("object_possessor_resolution_method"), "reflexive_subject")

    def test_answers_24_their_own_performance_remains_correct(self):
        applicants = Entity(
            uri="https://alfie-project.eu/etd-hub/entity/fairness/fairness-algorithms-d9755ca2",
            label="Fairness Algorithms", entity_type="Algorithm", confidence=0.9, metadata={"discourse_id": "24"},
        )
        unit_text = "what if this tool could be used by people applying for a job to predict their own performance."
        unit = _unit("Body:5", unit_text, 973)
        relation = Relation(
            source=applicants.uri, relation="predict", target="their own performance",
            metadata={
                "object_text": "their own performance", "discourse_id": "24", "evidence_quote": unit_text,
                "allow_local_predicate": True,
            },
        )
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation], local_entities=[applicants], discourse_text=unit_text, discourse_id="24",
            field="answer.description", origin_key="answer.description|24", unit=unit, unit_text=unit_text,
        )
        final = self.pipeline._apply_possessive_endpoint_semantics(
            relations=resolved, entities=[applicants], local_pool_entities=[applicants], unit=unit, discourse_id="24",
        )
        refined = self.precision.refine_relations(final, entities=[applicants], discourse_text=unit_text)
        r = refined[0]
        # The endpoint state must stay coherent regardless of whether the
        # possessor itself resolves (the possessive resolver's own, unchanged decision).
        self.assertEqual(r.target, "performance")
        self.assertTrue(r.target_is_literal)
        self.assertEqual(r.target_literal, "performance")
        self.assertEqual(r.metadata.get("object_possessor_entity_uri"), final[0].metadata.get("object_possessor_entity_uri"))


if __name__ == "__main__":
    unittest.main()
