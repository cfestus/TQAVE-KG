import unittest

from extractors.entity_extractor import LLMEntityExtractor


def _extractor(min_confidence=0.35):
    return LLMEntityExtractor(ontology_terms={}, min_confidence=min_confidence)


class EntityExtractorContextPromptTests(unittest.TestCase):
    """Structural checks on the prompt itself."""

    def test_context_section_appears_when_context_differs_from_text(self):
        ex = _extractor()
        full_field = (
            "The Commercial Health Risk Algorithm was developed in 2019. "
            "It was used to identify patients needing additional care."
        )
        target_chunk = "It was used to identify patients needing additional care."
        prompt = ex._build_prompt(target_chunk, context=full_field)
        self.assertIn("Context", prompt)
        self.assertIn(full_field, prompt)
        self.assertIn(target_chunk, prompt)
        self.assertIn("surface_form MUST come from here", prompt)

    def test_no_context_section_when_context_matches_text_or_is_absent(self):
        ex = _extractor()
        chunk = "GPT-4o was trained on licensed data."
        prompt_no_context = ex._build_prompt(chunk, context=None)
        self.assertNotIn("Context (full source field", prompt_no_context)

        prompt_same_context = ex._build_prompt(chunk, context=chunk)
        self.assertNotIn("Context (full source field", prompt_same_context)

    def test_system_prompt_instructs_context_is_for_resolution_only(self):
        ex = _extractor()
        lowered = ex.system_prompt.lower()
        self.assertIn("resolved_label", lowered)
        self.assertIn("pronoun", lowered)
        self.assertIn("never to extract an entity that appears only in context", lowered)


class EntityExtractorPronounResolutionTests(unittest.TestCase):
    """A pronoun/shorthand mention in the target chunk, resolved via a
    resolved_label the model derived from Context, must produce an Entity
    whose label is the resolved canonical name while the literal mention text
    and its span stay truthfully anchored to the target chunk."""

    def test_pronoun_resolved_via_context_keeps_truthful_mention_span(self):
        ex = _extractor()
        target_chunk = "It was used to identify patients needing additional care."
        mocked = [
            {
                "surface_form": "It",
                "resolved_label": "Commercial Health Risk Algorithm",
                "entity_type": "AISystem",
                "ontology_uri": "aipo:AISystem",
                "confidence": 0.9,
            }
        ]
        entities = ex._convert_to_entities(mocked, target_chunk, doc_id="d1")
        self.assertEqual(len(entities), 1)
        entity = entities[0]

        # Resolved canonical name becomes the entity's display label...
        self.assertEqual(entity.label, "Commercial Health Risk Algorithm")
        # ...but provenance stays truthful: the literal mention was "It", found
        # at its real position in the target chunk, not the resolved name.
        self.assertEqual(entity.metadata.get("surface_form"), "It")
        self.assertTrue(entity.metadata.get("mention_resolved_from_context"))
        expected_start = target_chunk.find("It")
        self.assertEqual(entity.metadata.get("start_char"), expected_start)
        self.assertEqual(entity.metadata.get("end_char"), expected_start + len("It"))

    def test_two_different_mentions_resolving_to_same_referent_share_identity(self):
        # "It" and "the tool" both resolved to the same canonical name within
        # one extraction call must converge on the same entity URI (and dedupe
        # to a single entity), the same way two direct mentions of an
        # already-named entity would.
        ex = _extractor()
        target_chunk = "It was deployed in 2019. Critics say the tool was biased."
        mocked = [
            {
                "surface_form": "It",
                "resolved_label": "Commercial Health Risk Algorithm",
                "entity_type": "AISystem",
                "confidence": 0.85,
            },
            {
                "surface_form": "the tool",
                "resolved_label": "Commercial Health Risk Algorithm",
                "entity_type": "AISystem",
                "confidence": 0.9,
            },
        ]
        entities = ex._deduplicate(ex._convert_to_entities(mocked, target_chunk, doc_id="d1"))
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].label, "Commercial Health Risk Algorithm")
        # Highest-confidence mention wins the dedup, but is still the resolved name.
        self.assertEqual(entities[0].confidence, 0.9)

    def test_no_resolution_needed_behaves_exactly_as_before(self):
        # Regression: when resolved_label is absent/empty or equals surface_form,
        # behavior is unchanged -- label is simply the surface form.
        ex = _extractor()
        target_chunk = "GPT-4o was trained on licensed data."
        mocked = [
            {"surface_form": "GPT-4o", "resolved_label": "", "entity_type": "Model", "confidence": 0.9},
            {"surface_form": "licensed data", "entity_type": "Dataset", "confidence": 0.9},
        ]
        entities = ex._convert_to_entities(mocked, target_chunk, doc_id="d1")
        labels = {e.label for e in entities}
        self.assertEqual(labels, {"GPT-4o", "licensed data"})
        for entity in entities:
            self.assertFalse(entity.metadata.get("mention_resolved_from_context"))
            self.assertEqual(entity.metadata.get("surface_form"), entity.label)


class EntityExtractorContextLeakageTests(unittest.TestCase):
    """Entities must never be grounded to text that only exists in Context;
    span-finding is always scoped to the target chunk."""

    def test_entity_mentioned_only_in_context_gets_no_span_in_target_chunk(self):
        ex = _extractor()
        target_chunk = "It was used to identify patients needing additional care."
        # "Commercial Health Risk Algorithm" is not literally present in the
        # target chunk at all (only in a hypothetical wider context) -- if a
        # non-compliant response tried to use it as surface_form directly, no
        # span should be found within the target chunk.
        mocked = [
            {
                "surface_form": "Commercial Health Risk Algorithm",
                "entity_type": "AISystem",
                "confidence": 0.9,
            }
        ]
        entities = ex._convert_to_entities(mocked, target_chunk, doc_id="d1")
        self.assertEqual(len(entities), 1)
        self.assertIsNone(entities[0].metadata.get("start_char"))
        self.assertEqual(entities[0].metadata.get("spans"), [])

    def test_surface_span_search_never_receives_context_text(self):
        # _surface_spans is only ever called with the target-chunk `text`
        # argument passed into _convert_to_entities, never with `context`.
        ex = _extractor()
        target_chunk = "It was used to identify patients needing additional care."
        context = "The Commercial Health Risk Algorithm was developed in 2019. " + target_chunk

        calls = []
        original = ex._surface_spans

        def _spy(text, surface):
            calls.append(text)
            return original(text, surface)

        ex._surface_spans = _spy
        mocked = [{"surface_form": "It", "resolved_label": "Commercial Health Risk Algorithm", "entity_type": "AISystem", "confidence": 0.9}]
        ex._convert_to_entities(mocked, target_chunk, doc_id="d1")

        self.assertTrue(calls)
        for called_text in calls:
            self.assertEqual(called_text, target_chunk)
            self.assertNotIn("developed in 2019", called_text)


if __name__ == "__main__":
    unittest.main()
