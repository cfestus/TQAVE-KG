import unittest

from extractors.llm_relation_extractor import LLMRelationExtractor
from models import Entity


def _extractor() -> LLMRelationExtractor:
    # Avoid OpenAI client init in unit tests.
    return LLMRelationExtractor.__new__(LLMRelationExtractor)


class HybridContextFocusPromptTests(unittest.TestCase):
    """Structural checks on the prompt itself: context and target-span text
    must be kept as two distinct sections, with an explicit instruction that
    context resolves references but is never itself an extraction source."""

    def test_context_section_appears_when_context_differs_from_text(self):
        ex = _extractor()
        full_field = (
            "The Commercial Health Risk Algorithm was developed in 2019. "
            "It was used to identify patients needing additional care."
        )
        target_chunk = "It was used to identify patients needing additional care."
        prompt = ex._build_prompt(target_chunk, [], candidate_pairs=None, context=full_field)
        self.assertIn("Context", prompt)
        self.assertIn(full_field, prompt)
        self.assertIn(target_chunk, prompt)
        # Text block must be clearly marked as the extraction/evidence source.
        self.assertIn("evidence_quote MUST come from here", prompt)

    def test_no_context_section_when_context_matches_text_or_is_absent(self):
        ex = _extractor()
        chunk = "GPT-4o was trained on licensed data."
        prompt_no_context = ex._build_prompt(chunk, [], candidate_pairs=None, context=None)
        self.assertNotIn("Context (full source field", prompt_no_context)

        prompt_same_context = ex._build_prompt(chunk, [], candidate_pairs=None, context=chunk)
        self.assertNotIn("Context (full source field", prompt_same_context)

    def test_system_prompt_instructs_context_is_for_resolution_only(self):
        # LLMRelationExtractor.__init__ sets self.system_prompt; construct a
        # real instance minimally (no network call happens at construction
        # time other than building an OpenAI client object) to read it -- or,
        # to stay fully offline, inspect the class-level docstring text by
        # instantiating via __new__ and calling the same assignment path is
        # not available, so we check the source text directly instead.
        import inspect
        source = inspect.getsource(LLMRelationExtractor.__init__)
        self.assertIn("Context", source)
        self.assertIn("resolve", source.lower())
        self.assertIn("never use context to introduce a proposition", source.lower())


class HybridContextFocusGroundingTests(unittest.TestCase):
    """Deterministic (no network) tests on _convert_to_relations proving
    evidence/provenance stays anchored to the target chunk, never to context,
    regardless of what context was supplied upstream."""

    def test_evidence_only_findable_in_context_stays_ungrounded(self):
        # Simulates a hypothetical non-compliant model response whose
        # evidence_quote was drawn from context rather than the target chunk.
        # It must NOT be treated as grounded just because that text exists
        # somewhere in the wider field.
        ex = _extractor()
        target_chunk = "It was used to identify patients needing additional care."
        source_uri = "https://alfie-project.eu/etd-hub#entity_algorithm"
        target_uri = "https://alfie-project.eu/etd-hub#entity_additional_care"
        rels = [
            {
                "source_id": source_uri,
                "relation": "uses",
                "target_id": target_uri,
                "confidence": 0.9,
                # This phrase only exists in the (hypothetical) wider field, not in target_chunk.
                "evidence_quote": "The Commercial Health Risk Algorithm was developed in 2019",
            }
        ]
        out = ex._convert_to_relations(rels, discourse_id="d1", text=target_chunk, entities=[])
        self.assertEqual(len(out), 1)
        meta = out[0].metadata
        self.assertEqual(meta.get("groundingStatus"), "ungrounded")
        self.assertEqual(meta.get("char_start"), -1)

    def test_evidence_found_in_target_chunk_grounds_normally(self):
        # Existing locally-explicit-relation behavior must be unchanged: when
        # the evidence genuinely comes from the target chunk, grounding still
        # succeeds exactly as before.
        ex = _extractor()
        target_chunk = "It was used to identify patients needing additional care."
        source_uri = "https://alfie-project.eu/etd-hub#entity_algorithm"
        target_uri = "https://alfie-project.eu/etd-hub#entity_additional_care"
        rels = [
            {
                "source_id": source_uri,
                "relation": "uses",
                "target_id": target_uri,
                "confidence": 0.9,
                "evidence_quote": "used to identify patients needing additional care",
            }
        ]
        out = ex._convert_to_relations(rels, discourse_id="d1", text=target_chunk, entities=[])
        self.assertEqual(len(out), 1)
        meta = out[0].metadata
        self.assertEqual(meta.get("groundingStatus"), "grounded")
        self.assertEqual(meta.get("groundingMethod"), "evidence_quote_span")
        expected_start = target_chunk.find("used to identify patients needing additional care")
        self.assertEqual(meta.get("char_start"), expected_start)


class HybridContextFocusPipelineWiringTests(unittest.TestCase):
    """Confirms pipeline.py threads the full field through as context while
    keeping the semantic unit as the extraction focus, via the
    EnhancedSemanticPipeline._extract_relations_with_pair_focus wiring."""

    def test_extract_relations_with_pair_focus_forwards_context(self):
        from pipeline import EnhancedSemanticPipeline

        captured = {}

        class _FakeExtractor:
            def extract(self, text, entities, discourse_id=None, candidate_pairs=None, context=None):
                captured["text"] = text
                captured["context"] = context
                return []

        pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        pipeline.llm_relation_extractor = _FakeExtractor()

        pipeline._extract_relations_with_pair_focus(
            unit_text="It was used to identify patients needing additional care.",
            local_entities=[],
            discourse_id="d1",
            candidate_pairs=[],
            context="The Commercial Health Risk Algorithm was developed in 2019. It was used to identify patients needing additional care.",
        )

        self.assertEqual(captured["text"], "It was used to identify patients needing additional care.")
        self.assertIn("Commercial Health Risk Algorithm", captured["context"])

    def test_extract_relations_with_pair_focus_degrades_gracefully_without_context_support(self):
        # Backward compatibility: an extractor stub that doesn't accept
        # `context` (e.g. an older/simplified test double) must still work.
        from pipeline import EnhancedSemanticPipeline

        captured = {}

        class _LegacyExtractor:
            def extract(self, text, entities, discourse_id=None, candidate_pairs=None):
                captured["text"] = text
                return []

        pipeline = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        pipeline.llm_relation_extractor = _LegacyExtractor()

        pipeline._extract_relations_with_pair_focus(
            unit_text="It was used to identify patients needing additional care.",
            local_entities=[],
            discourse_id="d1",
            candidate_pairs=[],
            context="full field text",
        )
        self.assertEqual(captured["text"], "It was used to identify patients needing additional care.")


if __name__ == "__main__":
    unittest.main()
