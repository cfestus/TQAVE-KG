import unittest

from config import Config
from models import Entity, Relation
from ontology.mapper import OntologyMapper
from pipeline import EnhancedSemanticPipeline
from utils.entity_normalizer import EntityNormalizer
from utils.helpers import GlobalEntityRegistry
from utils.predicate_precision import PredicatePrecisionLayer
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


def _resolve_unit_relations(pipeline, relation, unit, discourse_id="d1", field="answer.description", local_entities=None):
    return pipeline._resolve_semantic_unit_relations(
        relations=[relation],
        local_entities=local_entities if local_entities is not None else [],
        discourse_text=unit["text"], discourse_id=discourse_id, field=field,
        origin_key=f"{field}|{discourse_id}", unit=unit, unit_text=unit["text"],
    )


class LiteralReferentialBypassTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_literal_typed_bare_this_not_accepted_as_literal(self):
        subject = Entity(
            uri="https://kg/e/you", label="You", entity_type="Person", confidence=0.9,
            metadata={"discourse_id": "d1"},
        )
        unit = _unit("Body:1", "maybe you could even knowledge graph on top of this?", 0)
        relation = Relation(
            source=subject.uri, relation="couldKnowledgeGraph", target="this",
            target_is_literal=True, target_literal="this",
            metadata={"object_text": "this", "discourse_id": "d1", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, local_entities=[subject])
        self.assertFalse(resolved[0].target_is_literal)
        self.assertIsNone(resolved[0].target_literal)
        self.assertEqual(resolved[0].target, "this")
        self.assertTrue(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))
        self.assertEqual(resolved[0].metadata.get("object_endpoint_unresolved_original_surface"), "this")

    def test_literal_typed_it_routed_through_coreference_handling(self):
        subject = Entity(
            uri="https://kg/e/reviewers", label="Reviewers", entity_type="Person", confidence=0.9,
            metadata={"discourse_id": "d2"},
        )
        unit = _unit("Body:1", "reviewers say it is unclear.", 0)
        relation = Relation(
            source=subject.uri, relation="say", target="it",
            target_is_literal=True, target_literal="it",
            metadata={"object_text": "it", "discourse_id": "d2", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d2", local_entities=[subject])
        self.assertFalse(resolved[0].target_is_literal)
        self.assertIsNone(resolved[0].target_literal)
        self.assertTrue(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))

    def test_unresolved_literal_endpoint_rejected_by_validation(self):
        subject = Entity(
            uri="https://kg/e/you2", label="You", entity_type="Person", confidence=1.0,
            metadata={"section_category": "semantic"},
        )
        unit = _unit("Body:1", "maybe you could even knowledge graph on top of this?", 0)
        relation = Relation(
            source=subject.uri, relation="couldKnowledgeGraph", target="this",
            target_is_literal=True, target_literal="this",
            metadata={"object_text": "this", "discourse_id": "d3", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d3", local_entities=[subject])
        rel = resolved[0]
        rel.metadata["field"] = "answer.description"
        rel.metadata["extractor"] = "llm"
        rel.metadata["context_source"] = "llm_relation"
        rel.metadata["section_category"] = "semantic"
        policy = SemanticRelationPolicy()
        decision = policy.validate(rel, entities_by_uri={subject.uri: subject}, discourse_text=unit["text"])
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason_code, "unresolved_coreference")

    def test_resolvable_literal_pronoun_becomes_entity_uri(self):
        unit_start = 100
        unit_text = "The Screening Tool flagged several applicants, and reviewers later overruled it."
        local_start = unit_text.index("Screening Tool")
        tool = Entity(
            uri="https://kg/e/screening_tool", label="Screening Tool", entity_type="AISystem", confidence=0.9,
            metadata={
                "discourse_id": "d4", "section_category": "semantic",
                "start_char": unit_start + local_start, "end_char": unit_start + local_start + len("Screening Tool"),
                "mentions": [{
                    "discourse_id": "d4", "chunk_index": 0,
                    "start_char": unit_start + local_start, "end_char": unit_start + local_start + len("Screening Tool"),
                }],
            },
        )
        relation = Relation(
            source="reviewers", relation="overruled", target="it",
            target_is_literal=True, target_literal="it",
            metadata={"object_text": "it", "discourse_id": "d4", "evidence_quote": unit_text},
        )
        unit = _unit("Body:1", unit_text, unit_start)
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[tool], unit=unit, discourse_id="d4",
        )
        self.assertEqual(resolved[0].target, tool.uri)
        self.assertFalse(resolved[0].target_is_literal)
        self.assertIsNone(resolved[0].target_literal)
        self.assertEqual(resolved[0].metadata.get("object_endpoint_coref_resolution_method"), "occurrence_level_preceding_mention")

        # Passing back through _resolve_semantic_unit_relations must not
        # re-enter the literal branch at all (target_is_literal/target_literal
        # already cleared, target already an http URI).
        final = _resolve_unit_relations(self.pipeline, resolved[0], unit, discourse_id="d4", local_entities=[tool])
        self.assertEqual(final[0].target, tool.uri)
        self.assertFalse(final[0].target_is_literal)
        self.assertIsNone(final[0].target_literal)

    def test_resolvable_literal_discourse_deixis_becomes_proposition_uri(self):
        prev_text = "The review process omitted several required verification steps."
        prev = _unit("Body:1", prev_text, 0)
        cur_text = "This undermines confidence in the resulting decisions."
        cur = _unit("Body:2", cur_text, prev["end_offset"] + 1)
        semantic_units = [prev, cur]
        prev_relation = Relation(
            source="https://kg/e/review_process", relation="omitted", target="verification steps",
            target_is_literal=True, uri="https://kg/r/rel_prev",
            metadata={"evidence_quote": prev_text},
        )
        unit_relations_by_id = {"Body:1": [prev_relation]}
        # Deliberately literal-typed OBJECT-position "This" -- exercises the
        # object-side literal-endpoint path through propositional
        # resolution (grammatical role is irrelevant to the mechanism).
        relation = Relation(
            source="https://kg/e/legal_principles", relation="undermines", target="This",
            target_is_literal=True, target_literal="This",
            metadata={"object_text": "This", "evidence_quote": cur_text},
        )
        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=[relation], unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="d5",
            origin_key="answer.description|d5", field="answer.description",
        )
        self.assertEqual(len(extra), 1)
        proposition = extra[0]
        self.assertEqual(proposition.entity_type, "Proposition")
        self.assertEqual(resolved[0].target, proposition.uri)
        self.assertFalse(resolved[0].target_is_literal)
        self.assertIsNone(resolved[0].target_literal)
        self.assertEqual(
            resolved[0].metadata.get("object_endpoint_coref_resolution_method"),
            "propositional_preceding_unit_reference",
        )

    def test_stale_target_literal_with_false_flag_cannot_bypass(self):
        # A risky shape: target_is_literal=False but a
        # stale target_literal value is still present.
        subject = Entity(
            uri="https://kg/e/you3", label="You", entity_type="Person", confidence=0.9,
            metadata={"discourse_id": "d6"},
        )
        unit = _unit("Body:1", "maybe you could even knowledge graph on top of this?", 0)
        relation = Relation(
            source=subject.uri, relation="couldKnowledgeGraph", target="something else",
            target_is_literal=False, target_literal="this",
            metadata={"object_text": "this", "discourse_id": "d6", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d6", local_entities=[subject])
        self.assertFalse(resolved[0].target_is_literal)
        self.assertIsNone(resolved[0].target_literal)
        self.assertTrue(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))

    def test_genuine_numeric_literal_remains_accepted(self):
        subject = Entity(
            uri="https://kg/e/dataset1", label="Dataset", entity_type="Dataset", confidence=0.9,
            metadata={"discourse_id": "d7"},
        )
        unit = _unit("Body:1", "The dataset contains 1270 unique individuals.", 0)
        relation = Relation(
            source=subject.uri, relation="containsSubject", target="1270",
            target_is_literal=True, target_literal="1270",
            metadata={"object_text": "1270", "discourse_id": "d7", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d7", local_entities=[subject])
        self.assertTrue(resolved[0].target_is_literal)
        self.assertEqual(resolved[0].target_literal, "1270")
        self.assertFalse(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))

    def test_percentage_literal_remains_accepted(self):
        subject = Entity(
            uri="https://kg/e/dataset2", label="Training Dataset", entity_type="Dataset", confidence=0.9,
            metadata={"discourse_id": "d8"},
        )
        unit = _unit("Body:1", "Training datasets with 83.5% white individuals.", 0)
        relation = Relation(
            source=subject.uri, relation="contains", target="83.5%",
            target_is_literal=True, target_literal="83.5%",
            metadata={"object_text": "83.5%", "object_type": "percentage", "discourse_id": "d8", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d8", local_entities=[subject])
        self.assertTrue(resolved[0].target_is_literal)
        self.assertEqual(resolved[0].target_literal, "83.5%")
        self.assertFalse(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))

    def test_genuine_textual_scalar_literal_remains_accepted(self):
        subject = Entity(
            uri="https://kg/e/audit_process", label="Audit Process", entity_type="Model", confidence=0.9,
            metadata={"discourse_id": "d9"},
        )
        unit = _unit("Body:1", "Building an acceptable audit process is quite easy to do.", 0)
        relation = Relation(
            source=subject.uri, relation="is", target="quite easy to do",
            target_is_literal=True, target_literal="quite easy to do",
            metadata={"object_text": "quite easy to do", "discourse_id": "d9", "evidence_quote": unit["text"]},
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d9", local_entities=[subject])
        self.assertTrue(resolved[0].target_is_literal)
        self.assertEqual(resolved[0].target_literal, "quite easy to do")
        self.assertFalse(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))

    def test_quantitative_comparison_relation_unchanged(self):
        subject = Entity(
            uri="https://kg/e/facial_recognition", label="Commercial Facial Recognition Systems",
            entity_type="AISystem", confidence=0.9, metadata={"discourse_id": "d10"},
        )
        unit = _unit(
            "Body:1",
            "These systems performed 32 times worse on dark-skinned women compared to light-skinned men.",
            0,
        )
        relation = Relation(
            source=subject.uri, relation="performed", target="32 times worse on dark-skinned women",
            target_is_literal=True, target_literal="32 times worse on dark-skinned women",
            metadata={
                "object_text": "32 times worse on dark-skinned women", "object_type": "comparison",
                "discourse_id": "d10", "evidence_quote": unit["text"],
            },
        )
        resolved = _resolve_unit_relations(self.pipeline, relation, unit, discourse_id="d10", local_entities=[subject])
        self.assertTrue(resolved[0].target_is_literal)
        self.assertEqual(resolved[0].target_literal, "32 times worse on dark-skinned women")
        self.assertFalse(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))


if __name__ == "__main__":
    unittest.main()
