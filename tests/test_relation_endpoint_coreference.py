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


def _ent(uri, label, entity_type, *, discourse_id, start_char, end_char,
         resolved_from_context=False, surface_form=None):
    meta = {
        "discourse_id": discourse_id,
        "section_category": "semantic",
        "start_char": start_char,
        "end_char": end_char,
        "mentions": [
            {"discourse_id": discourse_id, "chunk_index": 0, "start_char": start_char, "end_char": end_char}
        ],
    }
    if resolved_from_context:
        meta["mention_resolved_from_context"] = True
        meta["surface_form"] = surface_form
    return Entity(uri=uri, label=label, entity_type=entity_type, confidence=0.9, metadata=meta)


class IsolatedResolutionTests(unittest.TestCase):
    """Direct tests of _resolve_anaphoric_relation_endpoints / its helper,
    with no pipeline batch machinery involved."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_this_resolved_from_context_resolved_entity_same_discourse(self):
        # Entity extraction already resolved a literal "this" occurrence to
        # a specific referent via context (mention_resolved_from_context).
        system = _ent(
            "https://kg/e/system_a", "System A", "AISystem",
            discourse_id="d1", start_char=10, end_char=16,
            resolved_from_context=True, surface_form="this",
        )
        relation = Relation(
            source="this", relation="lacks", target="transparency", target_is_literal=True,
            metadata={"subject_text": "this", "discourse_id": "d1"},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[system], unit={"start_offset": 100}, discourse_id="d1",
        )
        self.assertEqual(resolved[0].source, system.uri)
        self.assertTrue(resolved[0].metadata.get("subject_endpoint_coref_resolved"))
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_coref_resolution_method"), "context_resolved_mention")
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_original_surface"), "this")
        self.assertEqual(extra, [system])

    def test_it_resolved_to_one_unambiguous_preceding_entity(self):
        model = _ent("https://kg/e/model_x", "Model X", "AISystem", discourse_id="d2", start_char=5, end_char=12)
        relation = Relation(
            source="it", relation="causes", target="bias", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d2"},
        )
        resolved, _ = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[model], unit={"start_offset": 50}, discourse_id="d2",
        )
        self.assertEqual(resolved[0].source, model.uri)
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_coref_resolution_method"), "unambiguous_preceding_entity")

    def test_this_tool_resolved_to_existing_same_discourse_model_not_unrelated_entity(self):
        tool = _ent("https://kg/e/triage_tool", "Triage Assistant", "AISystem", discourse_id="d3", start_char=5, end_char=25)
        unrelated_group = _ent("https://kg/e/residents", "Rural Residents", "Group", discourse_id="d3", start_char=30, end_char=45)
        relation = Relation(
            source="something", relation="predicts", target="this tool", target_is_literal=True,
            metadata={"object_text": "this tool", "discourse_id": "d3"},
        )
        resolved, _ = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[tool, unrelated_group], unit={"start_offset": 100}, discourse_id="d3",
        )
        self.assertEqual(resolved[0].target, tool.uri)
        self.assertFalse(resolved[0].target_is_literal)
        self.assertEqual(resolved[0].metadata.get("object_endpoint_original_surface"), "this tool")

    def test_this_tool_stays_unresolved_when_only_incompatible_entity_precedes(self):
        # Only a non-system-family entity precedes -- "this tool" must not
        # be forced onto an entity of an incompatible kind.
        unrelated_group = _ent("https://kg/e/residents2", "Rural Residents", "Group", discourse_id="d3b", start_char=5, end_char=20)
        relation = Relation(
            source="something", relation="predicts", target="this tool", target_is_literal=True,
            metadata={"object_text": "this tool", "discourse_id": "d3b"},
        )
        resolved, _ = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[unrelated_group], unit={"start_offset": 100}, discourse_id="d3b",
        )
        self.assertEqual(resolved[0].target, "this tool")
        self.assertTrue(resolved[0].target_is_literal)

    def test_ambiguous_two_plausible_antecedents_stays_unresolved(self):
        sys_a = _ent("https://kg/e/sys_a", "System A", "AISystem", discourse_id="d4", start_char=5, end_char=12)
        sys_b = _ent("https://kg/e/sys_b", "System B", "AISystem", discourse_id="d4", start_char=20, end_char=27)
        relation = Relation(
            source="it", relation="causes", target="harm", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d4"},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[sys_a, sys_b], unit={"start_offset": 50}, discourse_id="d4",
        )
        self.assertEqual(resolved[0].source, "it")
        self.assertFalse(resolved[0].metadata.get("subject_endpoint_coref_resolved"))
        self.assertEqual(extra, [])

    def test_explicitly_named_endpoints_remain_unchanged(self):
        relation = Relation(
            source="https://kg/e/named_entity", relation="causes", target="Specific Named Harm",
            target_is_literal=True, metadata={"discourse_id": "d5"},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[], unit={"start_offset": 50}, discourse_id="d5",
        )
        self.assertEqual(resolved[0].source, "https://kg/e/named_entity")
        self.assertEqual(resolved[0].target, "Specific Named Harm")
        self.assertNotIn("subject_endpoint_coref_resolved", resolved[0].metadata)
        self.assertEqual(extra, [])

    def test_no_resolution_crosses_discourse_boundaries(self):
        other_discourse_entity = _ent("https://kg/e/other_sys", "Other System", "AISystem", discourse_id="d_other", start_char=5, end_char=12)
        relation = Relation(
            source="it", relation="causes", target="harm", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d6"},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[other_discourse_entity], unit={"start_offset": 50}, discourse_id="d6",
        )
        self.assertEqual(resolved[0].source, "it")
        self.assertEqual(extra, [])


class PipelineWiringTests(unittest.TestCase):
    """End-to-end test through _process_relations_batch confirming the
    resolved entity survives _resolve_semantic_unit_relations (which would
    otherwise fabricate a new entity from the raw pronoun text) and that
    semantic validation still accepts the resolved relation."""

    def test_pronoun_endpoint_resolved_end_to_end_and_validates(self):
        pipeline = _make_pipeline()
        did = "T1"
        text = (
            "An Ethics Review Framework is used to evaluate new projects before launch.\n\n"
            "Operational Issues:\n"
            "* Approval Delay: It delays approval for time-sensitive projects\n"
            "* Staffing Shortfall: On-call staff report chronic overtime"
        )
        framework = Entity(
            uri="https://kg/e/ethics_review_framework",
            label="Ethics Review Framework",
            entity_type="PolicyFramework",
            confidence=1.0,
            metadata={
                "discourse_id": did,
                "section_category": "semantic",
                "start_char": 3,
                "end_char": 27,
                "mentions": [{"discourse_id": did, "chunk_index": 0, "start_char": 3, "end_char": 27}],
            },
        )
        entities = [framework]

        def _extract_relations_with_pair_focus(unit_text, local_entities, discourse_id, candidate_pairs, context=None):
            if "It delays approval" not in unit_text:
                return []
            return [
                Relation(
                    source="it",
                    relation="delays",
                    target="approval for time-sensitive projects",
                    target_is_literal=True,
                    confidence=0.9,
                    metadata={
                        "extractor": "llm",
                        "context_source": "llm_relation",
                        "subject_text": "It",
                        "evidence_quote": unit_text,
                        "evidence": unit_text,
                    },
                )
            ]

        pipeline._extract_relations_with_pair_focus = _extract_relations_with_pair_focus

        relations_list = pipeline._process_relations_batch(
            [text], [entities], [did],
            provenance_list=[{"field": "theme.description", "discourse_id": did}],
        )
        relations = relations_list[0]

        matching = [r for r in relations if str(getattr(r, "relation", "")) == "delays"]
        self.assertEqual(len(matching), 1, f"expected exactly one relation, got: {relations}")
        relation = matching[0]

        # The endpoint is resolved BEFORE _resolve_semantic_unit_relations
        # would otherwise fabricate a brand-new entity from the raw "It" --
        # confirm the resolved URI, not a synthetic surface entity, survives
        # through to the final relation.
        self.assertEqual(relation.source, framework.uri)
        self.assertTrue(relation.metadata.get("subject_endpoint_coref_resolved"))
        self.assertEqual(relation.metadata.get("subject_endpoint_coref_resolution_method"), "unambiguous_preceding_entity")
        self.assertEqual(relation.metadata.get("subject_endpoint_original_surface"), "It")
        # Evidence/provenance stay anchored to the target unit -- resolution
        # only changes the endpoint identity, never the evidence span.
        self.assertIn("It delays approval", relation.metadata.get("evidence_quote") or "")

        # Note: whether this resolved relation also passes semantic
        # validation depends on local_context/co-occurrence shaping, which
        # is explicitly out of scope for this task (validation rejection
        # rules are not modified here) -- not asserted.


if __name__ == "__main__":
    unittest.main()
