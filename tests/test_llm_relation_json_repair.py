import unittest

from extractors.llm_relation_extractor import LLMRelationExtractor


class LLMRelationJsonRepairTests(unittest.TestCase):
    def _extractor(self) -> LLMRelationExtractor:
        return LLMRelationExtractor.__new__(LLMRelationExtractor)

    def test_repair_missing_closing_brace(self):
        ex = self._extractor()
        raw = '{"relations":[{"source_id":"s","target_id":"t","relation":"mentions","confidence":0.9}]'
        parsed = ex._parse_relations_json(raw)
        self.assertIn("relations", parsed)
        self.assertEqual(len(parsed["relations"]), 1)

    def test_repair_missing_array_close_and_brace(self):
        ex = self._extractor()
        raw = '{"relations":[{"source_id":"s","target_id":"t","relation":"mentions","confidence":0.9}'
        parsed = ex._parse_relations_json(raw)
        self.assertIn("relations", parsed)
        self.assertEqual(len(parsed["relations"]), 1)

    def test_invalid_payload_returns_empty_relations(self):
        ex = self._extractor()
        parsed = ex._parse_relations_json("this is not valid json at all")
        self.assertEqual(parsed, {"relations": []})


if __name__ == "__main__":
    unittest.main()

