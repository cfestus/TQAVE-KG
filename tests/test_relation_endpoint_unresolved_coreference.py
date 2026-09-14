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


def _unit(unit_id, text, start_offset, *, section_category="semantic", unit_type="sentence"):
    return {
        "unit_id": unit_id,
        "text": text,
        "start_offset": start_offset,
        "end_offset": start_offset + len(text),
        "section_category": section_category,
        "unit_type": unit_type,
    }


class UnresolvedCoreferenceBlockingTests(unittest.TestCase):
    """Unresolved referential relation endpoints must not be
    fabricated into normal, globally-reusable Entity objects."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def _run(self, *, source, object_text, discourse_id="d1", local_entities=None):
        relation = Relation(
            source=source, relation="affects", target=object_text, target_is_literal=False,
            metadata={"subject_text": source, "object_text": object_text},
        )
        unit = _unit("Body:1", f"{source} {object_text}", 0)
        resolved = self.pipeline._resolve_semantic_unit_relations(
            relations=[relation],
            local_entities=local_entities if local_entities is not None else [],
            discourse_text=unit["text"],
            discourse_id=discourse_id,
            field="answer.description",
            origin_key=f"answer.description|{discourse_id}",
            unit=unit,
            unit_text=unit["text"],
        )
        return resolved

    def test_unresolved_bare_it_does_not_create_entity(self):
        resolved = self._run(source="it", object_text="Model Alpha")
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].source, "it")
        self.assertTrue(resolved[0].metadata.get("subject_endpoint_unresolved_coreference"))
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_unresolved_original_surface"), "it")
        self.assertIn("unresolved", str(resolved[0].metadata.get("subject_endpoint_unresolved_reason")))
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any("it" == str(getattr(e, "label", "")).strip().lower() for e in entities))

    def test_unresolved_bare_this_does_not_create_entity(self):
        resolved = self._run(source="this", object_text="Model Alpha")
        self.assertEqual(resolved[0].source, "this")
        self.assertTrue(resolved[0].metadata.get("subject_endpoint_unresolved_coreference"))
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any("this" == str(getattr(e, "label", "")).strip().lower() for e in entities))

    def test_unresolved_deictic_noun_phrase_does_not_create_entity(self):
        resolved = self._run(source="Model Alpha", object_text="these frameworks")
        self.assertEqual(resolved[0].target, "these frameworks")
        self.assertFalse(resolved[0].target_is_literal)
        self.assertTrue(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))
        self.assertEqual(resolved[0].metadata.get("object_endpoint_unresolved_original_surface"), "these frameworks")
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any("framework" in str(getattr(e, "label", "")).lower() for e in entities))

    def test_unresolved_this_disagreement_does_not_create_reusable_entity_uri(self):
        resolved = self._run(source="this disagreement", object_text="Model Alpha")
        self.assertEqual(resolved[0].source, "this disagreement")
        self.assertTrue(resolved[0].metadata.get("subject_endpoint_unresolved_coreference"))
        self.assertEqual(
            resolved[0].metadata.get("subject_endpoint_unresolved_reason"),
            "unresolved_generic_demonstrative_phrase",
        )
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any("disagreement" in str(getattr(e, "label", "")).lower() for e in entities))

    def test_same_unresolved_phrase_in_two_discourses_shares_no_entity(self):
        first = self._run(source="these tests", object_text="Model Alpha", discourse_id="dA")
        second = self._run(source="these tests", object_text="Model Beta", discourse_id="dB")
        self.assertEqual(first[0].source, "these tests")
        self.assertEqual(second[0].source, "these tests")
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertFalse(any("test" in str(getattr(e, "label", "")).lower() for e in entities))

    def test_task2_resolved_it_keeps_real_antecedent(self):
        unit_start = 100
        unit_text = "The Screening Tool flagged several applicants, and reviewers later overruled it."
        local_start = unit_text.index("Screening Tool")
        tool = Entity(
            uri="https://kg/e/screening_tool", label="Screening Tool", entity_type="AISystem", confidence=0.9,
            metadata={
                "discourse_id": "d2", "section_category": "semantic",
                "start_char": unit_start + local_start, "end_char": unit_start + local_start + len("Screening Tool"),
                "mentions": [{
                    "discourse_id": "d2", "chunk_index": 0,
                    "start_char": unit_start + local_start, "end_char": unit_start + local_start + len("Screening Tool"),
                }],
            },
        )
        relation = Relation(
            source="reviewers", relation="overruled", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "d2", "evidence_quote": unit_text},
        )
        unit = _unit("Body:1", unit_text, unit_start)
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[tool], unit=unit, discourse_id="d2",
        )
        self.assertEqual(resolved[0].target, tool.uri)

        final = self.pipeline._resolve_semantic_unit_relations(
            relations=resolved, local_entities=[tool] + extra, discourse_text=unit_text,
            discourse_id="d2", field="answer.description", origin_key="answer.description|d2",
            unit=unit, unit_text=unit_text,
        )
        self.assertEqual(final[0].target, tool.uri)
        self.assertFalse(final[0].metadata.get("object_endpoint_unresolved_coreference"))

    def test_task2e2_resolved_this_keeps_proposition_uri(self):
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
        relation = Relation(
            source="This", relation="undermines", target="confidence in the resulting decisions",
            target_is_literal=True,
            metadata={"subject_text": "This", "evidence_quote": cur_text},
        )
        resolved, extra = self.pipeline._resolve_propositional_relation_endpoints(
            relations=[relation], unit=cur, unit_index=1, semantic_units=semantic_units,
            unit_relations_by_id=unit_relations_by_id, discourse_id="d3",
            origin_key="answer.description|d3", field="answer.description",
        )
        proposition_uri = resolved[0].source
        self.assertTrue(proposition_uri.startswith("http"))

        final = self.pipeline._resolve_semantic_unit_relations(
            relations=resolved, local_entities=list(extra), discourse_text=cur_text,
            discourse_id="d3", field="answer.description", origin_key="answer.description|d3",
            unit=cur, unit_text=cur_text,
        )
        self.assertEqual(final[0].source, proposition_uri)
        self.assertFalse(final[0].metadata.get("subject_endpoint_unresolved_coreference"))

    def test_task2d_possessive_head_noun_not_blocked(self):
        # Pre-Task-2D shape: object is still the full possessive phrase.
        resolved = self._run(source="Model Alpha", object_text="their approach")
        self.assertFalse(resolved[0].metadata.get("object_endpoint_unresolved_coreference"))
        self.assertTrue(str(resolved[0].target).startswith("http"), "generic entity fabrication must proceed as before")

        # Post-Task-2D shape: possessive semantics already decomposed the
        # endpoint to its head noun as a literal -- must remain untouched.
        subject = Entity(
            uri="https://kg/e/team", label="Team", entity_type="Organization", confidence=0.9,
            metadata={"discourse_id": "d4"},
        )
        possessive_relation = Relation(
            source=subject.uri, relation="shares", target="approach", target_is_literal=True,
            target_literal="approach",
            metadata={"object_text": "their approach", "object_possessor_entity_uri": subject.uri},
        )
        unit = _unit("Body:1", "They share their approach.", 0)
        final = self.pipeline._resolve_semantic_unit_relations(
            relations=[possessive_relation], local_entities=[subject], discourse_text=unit["text"],
            discourse_id="d4", field="answer.description", origin_key="answer.description|d4",
            unit=unit, unit_text=unit["text"],
        )
        self.assertEqual(final[0].target, "approach")
        self.assertTrue(final[0].target_is_literal)
        self.assertFalse(final[0].metadata.get("object_endpoint_unresolved_coreference"))

    def test_ordinary_explicit_entity_still_resolved_normally(self):
        resolved = self._run(source="Model Alpha", object_text="fairness metrics")
        self.assertTrue(str(resolved[0].source).startswith("http"))
        self.assertFalse(resolved[0].metadata.get("subject_endpoint_unresolved_coreference"))
        entities = self.pipeline.global_registry.get_all_entities()
        self.assertTrue(any("model alpha" in str(getattr(e, "label", "")).lower() for e in entities))


if __name__ == "__main__":
    unittest.main()
