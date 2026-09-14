import unittest
from unittest.mock import patch

from pipeline import EnhancedSemanticPipeline
from utils.semantic_chunker import SemanticChunker


class SemanticChunkerOffsetsTests(unittest.TestCase):
    def test_chunk_with_offsets_text_matches_span_slice(self):
        text = (
            "Ethical challenge: transparency and fairness matter. "
            "Dataset context includes benchmark details and governance notes. "
            "Risk section explains mitigations."
        )
        chunks = SemanticChunker.chunk_with_offsets(text)
        self.assertGreaterEqual(len(chunks), 1)
        for idx, ch in enumerate(chunks):
            self.assertEqual(int(ch.get("chunk_index", -1)), idx)
            s = int(ch["start_offset"])
            e = int(ch["end_offset"])
            self.assertGreaterEqual(s, 0)
            self.assertGreater(e, s)
            self.assertEqual(ch["text"], text[s:e])

    def test_pipeline_build_chunk_spans_uses_chunker_native_offsets(self):
        p = EnhancedSemanticPipeline.__new__(EnhancedSemanticPipeline)
        p._discourse_chunk_spans = {}
        text = "Alpha. Beta."

        with patch.object(
            SemanticChunker,
            "chunk_with_offsets",
            return_value=[
                {"text": "Alpha.", "start_offset": 0, "end_offset": 6, "chunk_index": 0},
                {"text": "Beta.", "start_offset": 7, "end_offset": 12, "chunk_index": 1},
            ],
        ):
            spans = p._build_chunk_spans(text, ["ignored"], "d1")

        self.assertEqual(
            spans,
            [
                {"discourse_id": "d1", "chunk_index": 0, "start_offset": 0, "end_offset": 6},
                {"discourse_id": "d1", "chunk_index": 1, "start_offset": 7, "end_offset": 12},
            ],
        )


if __name__ == "__main__":
    unittest.main()
