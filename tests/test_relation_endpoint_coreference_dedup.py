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


def _ent(uri, label, entity_type, *, discourse_id, start_char, end_char):
    return Entity(
        uri=uri, label=label, entity_type=entity_type, confidence=0.9,
        metadata={
            "discourse_id": discourse_id,
            "section_category": "semantic",
            "start_char": start_char,
            "end_char": end_char,
            "mentions": [
                {"discourse_id": discourse_id, "chunk_index": 0, "start_char": start_char, "end_char": end_char}
            ],
        },
    )


class CandidateDeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_nested_duplicate_mentions_collapse_to_one_candidate(self):
        # "Students" (736-744) and "Some Students" (731-744) -- the real
        # questions/17 shape: overlapping/nested spans, same family, same
        # head noun.
        students = _ent("https://kg/e/students", "Students", "Group", discourse_id="d1", start_char=736, end_char=744)
        some_students = _ent("https://kg/e/some_students", "Some Students", "Group", discourse_id="d1", start_char=731, end_char=744)
        collapsed = self.pipeline._collapse_duplicate_candidate_entities(
            {students.uri: students, some_students.uri: some_students}, "d1",
        )
        self.assertEqual(len(collapsed), 1)

    def test_more_informative_mention_is_preferred(self):
        students = _ent("https://kg/e/students", "Students", "Group", discourse_id="d1", start_char=736, end_char=744)
        some_students = _ent("https://kg/e/some_students", "Some Students", "Group", discourse_id="d1", start_char=731, end_char=744)
        collapsed = self.pipeline._collapse_duplicate_candidate_entities(
            {students.uri: students, some_students.uri: some_students}, "d1",
        )
        kept = next(iter(collapsed.values()))
        self.assertEqual(kept.label, "Some Students")

    def test_overlapping_mentions_with_incompatible_family_remain_distinct(self):
        students = _ent("https://kg/e/students2", "Students", "Group", discourse_id="d2", start_char=100, end_char=108)
        # Overlapping span, but a different semantic family (a system, not a group).
        program = _ent("https://kg/e/learning_program", "Learning Support Program", "AISystem", discourse_id="d2", start_char=95, end_char=120)
        collapsed = self.pipeline._collapse_duplicate_candidate_entities(
            {students.uri: students, program.uri: program}, "d2",
        )
        self.assertEqual(len(collapsed), 2)

    def test_non_overlapping_same_family_entities_remain_distinct(self):
        sys_a = _ent("https://kg/e/sys_a2", "System A", "AISystem", discourse_id="d3", start_char=5, end_char=13)
        sys_b = _ent("https://kg/e/sys_b2", "System B", "AISystem", discourse_id="d3", start_char=50, end_char=58)
        collapsed = self.pipeline._collapse_duplicate_candidate_entities(
            {sys_a.uri: sys_a, sys_b.uri: sys_b}, "d3",
        )
        self.assertEqual(len(collapsed), 2)

    def test_no_cross_discourse_deduplication(self):
        students_d1 = _ent("https://kg/e/students_d1", "Students", "Group", discourse_id="d4", start_char=736, end_char=744)
        # Same overlapping span/label pattern, but tagged to a DIFFERENT
        # discourse -- must never be treated as the same referent.
        some_students_d2 = _ent("https://kg/e/students_d2", "Some Students", "Group", discourse_id="d5", start_char=731, end_char=744)
        collapsed = self.pipeline._collapse_duplicate_candidate_entities(
            {students_d1.uri: students_d1, some_students_d2.uri: some_students_d2}, "d4",
        )
        self.assertEqual(len(collapsed), 2)

    def test_existing_ambiguity_negative_control_still_unresolved(self):
        # The original two-genuinely-distinct-antecedents case: not
        # overlapping, different labels/head nouns -- must remain ambiguous.
        sys_a = _ent("https://kg/e/sys_a3", "System A", "AISystem", discourse_id="d4b", start_char=5, end_char=13)
        sys_b = _ent("https://kg/e/sys_b3", "System B", "AISystem", discourse_id="d4b", start_char=20, end_char=28)
        relation = Relation(
            source="it", relation="causes", target="harm", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d4b"},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[sys_a, sys_b], unit={"start_offset": 50}, discourse_id="d4b",
        )
        self.assertEqual(resolved[0].source, "it")
        self.assertEqual(extra, [])


class QuestionsSeventeenDedupEndToEndTest(unittest.TestCase):
    """End-to-end replay of the real questions/17 shape, now WITH the
    overlapping duplicate 'Students'/'Some Students' entities that blocked
    resolution before this fix."""

    def test_them_resolves_despite_overlapping_duplicate_student_entities(self):
        pipeline = _make_pipeline()
        did = "T8"
        text = (
            "It's hypothetically possible to use a model to rank candidates.\n\n"
            "Also, I worry about this from the perspective of a mentor - sometimes we might "
            "predict that some students will struggle more with certain topics and "
            "proactively reach out to them."
        )
        local_start_students = text.index("students")
        local_start_some = text.index("some students")
        students = _ent(
            "https://kg/e/students", "Students", "Group",
            discourse_id=did, start_char=local_start_students, end_char=local_start_students + len("students"),
        )
        some_students = _ent(
            "https://kg/e/some_students", "Some Students", "Group",
            discourse_id=did, start_char=local_start_some, end_char=local_start_some + len("some students"),
        )
        entities = [students, some_students]

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            if "reach out to them" not in unit_text:
                return []
            return [
                Relation(
                    source="a mentor", relation="reachOut", target="them", target_is_literal=True,
                    confidence=0.9,
                    metadata={
                        "extractor": "llm", "context_source": "llm_relation",
                        "subject_text": "a mentor", "object_text": "them",
                        "evidence_quote": unit_text, "evidence": unit_text,
                    },
                )
            ]

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [text], [entities], [did], provenance_list=[{"field": "question.body", "discourse_id": did}],
        )
        relations = relations_list[0]

        matching = [r for r in relations if str(getattr(r, "relation", "")) == "reachOut"]
        self.assertEqual(len(matching), 1, f"expected exactly one relation, got: {relations}")
        relation = matching[0]

        self.assertTrue(relation.metadata.get("object_endpoint_coref_resolved"))
        self.assertEqual(relation.metadata.get("object_endpoint_coref_resolution_method"), "occurrence_level_preceding_mention")
        self.assertFalse(relation.target_is_literal)
        self.assertIn(relation.target, {students.uri, some_students.uri})


if __name__ == "__main__":
    unittest.main()
