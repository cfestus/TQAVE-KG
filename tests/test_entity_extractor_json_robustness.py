import unittest

from extractors.entity_extractor import EntityJSONParseError, LLMEntityExtractor


def _make_extractor():
    return LLMEntityExtractor(ontology_terms={"Concept": "skos:Concept"})


VALID_ENTITY = (
    '{"surface_form": "System A", "resolved_label": "", '
    '"entity_type": "AISystem", "ontology_uri": "", "confidence": 0.9}'
)


class ValidAndFencedJsonTests(unittest.TestCase):
    """Well-formed output, and output wrapped in a markdown code fence,
    must both parse straight through without hitting any repair path."""

    def setUp(self):
        self.extractor = _make_extractor()

    def test_valid_json_parses_directly(self):
        content = '{"entities": [%s]}' % VALID_ENTITY
        result = self.extractor._parse_entities_json(content)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["surface_form"], "System A")

    def test_fenced_json_is_recovered(self):
        content = '```json\n{"entities": [%s]}\n```' % VALID_ENTITY
        result = self.extractor._parse_entities_json(content)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["surface_form"], "System A")

    def test_leading_and_trailing_prose_is_stripped(self):
        content = 'Here is the JSON:\n{"entities": [%s]}\nHope that helps!' % VALID_ENTITY
        result = self.extractor._parse_entities_json(content)
        self.assertEqual(len(result["entities"]), 1)


class RepairableMalformedJsonTests(unittest.TestCase):
    """Minor formatting defects (trailing commas, a truncated tail cut off
    mid-entity) must be repaired without ever inventing content -- an
    incomplete trailing entity is dropped, not completed."""

    def setUp(self):
        self.extractor = _make_extractor()

    def test_trailing_comma_before_closing_bracket_is_repaired(self):
        content = '{"entities": [%s,]}' % VALID_ENTITY
        result = self.extractor._parse_entities_json(content)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["surface_form"], "System A")

    def test_truncated_tail_drops_incomplete_entity_without_fabricating(self):
        # Simulates a max_tokens cutoff mid-way through a second entity:
        # the first, complete entity must be recovered; the second,
        # incomplete one must be dropped -- not silently completed.
        content = (
            '{"entities": [%s, {"surface_form": "System B", "resolved_la'
            % VALID_ENTITY
        )
        result = self.extractor._parse_entities_json(content)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["surface_form"], "System A")

    def test_truncated_immediately_after_trailing_comma_drops_nothing_extra(self):
        content = '{"entities": [%s,' % VALID_ENTITY
        result = self.extractor._parse_entities_json(content)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["surface_form"], "System A")


class IrreparableMalformedJsonTests(unittest.TestCase):
    """Output with no recoverable JSON (free-text refusal, or truncation
    before any entity is even complete) must raise EntityJSONParseError --
    never silently return fabricated or partially-guessed content."""

    def setUp(self):
        self.extractor = _make_extractor()

    def test_freeform_non_json_text_raises(self):
        with self.assertRaises(EntityJSONParseError):
            self.extractor._parse_entities_json(
                "I'm sorry, I can't process that request."
            )

    def test_truncated_before_any_entity_completes_raises_rather_than_fabricates(self):
        # Cut off mid-string, before the first entity object ever closes --
        # there is nothing complete to salvage, so this must fail rather
        # than invent a partial entity.
        content = '{"entities": [{"surface_form": "System A", "resolved_label": "'
        with self.assertRaises(EntityJSONParseError):
            self.extractor._parse_entities_json(content)

    def test_mismatched_brackets_raise(self):
        content = '{"entities": [{"surface_form": "System A"]}'
        with self.assertRaises(EntityJSONParseError):
            self.extractor._parse_entities_json(content)

    def test_empty_content_raises(self):
        with self.assertRaises(EntityJSONParseError):
            self.extractor._parse_entities_json("")

    def test_parse_error_is_classified_as_retryable(self):
        # The whole point of raising a dedicated exception type is that the
        # existing retry loop treats a malformed/truncated completion as a
        # one-off, worth retrying -- unlike a genuine non-transient error.
        try:
            self.extractor._parse_entities_json("not json at all")
        except EntityJSONParseError as exc:
            self.assertTrue(LLMEntityExtractor._is_transient_error(exc))
        else:
            self.fail("expected EntityJSONParseError")


class StructuredOutputResponseFormatTests(unittest.TestCase):
    """The extractor must prefer OpenAI Structured Outputs (a JSON-schema
    response_format) over plain free-text/json_object mode when available,
    per the fix's design requirement."""

    def test_response_format_uses_json_schema_mode(self):
        fmt = LLMEntityExtractor._RESPONSE_FORMAT
        self.assertEqual(fmt.get("type"), "json_schema")
        schema = fmt.get("json_schema", {})
        self.assertTrue(schema.get("strict"))
        entity_schema = schema["schema"]["properties"]["entities"]["items"]
        # Every declared field is required (strict mode constraint) and no
        # field semantics changed -- optional fields still default to "".
        self.assertEqual(
            set(entity_schema["required"]),
            {"surface_form", "resolved_label", "entity_type", "ontology_uri", "confidence"},
        )
        self.assertFalse(entity_schema.get("additionalProperties", True))


if __name__ == "__main__":
    unittest.main()
