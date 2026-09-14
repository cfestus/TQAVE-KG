"""
Tests for validation/span_integrity.py, including reproductions of the two
still-open truncation cases (African American women, minority communities)
against their REAL source text -- proving the validator correctly flags
them as defects rather than silently passing, and documenting that no
repair was applied (see the 2026-09 release-preflight report for why: no
single deterministic, general root cause could be pinned down for either
within the bounded scope of that pass).
"""
import json
import unittest

from validation.span_integrity import check_entity_span_integrity, check_span_against_text


class CheckSpanAgainstTextTests(unittest.TestCase):
    def test_ok_exact_match(self):
        result = check_span_against_text("The Agile process works well.", 4, 9, "Agile")
        self.assertEqual(result["status"], "OK")

    def test_ok_case_and_whitespace_insensitive(self):
        result = check_span_against_text("the   AGILE   process", 6, 11, "Agile")
        self.assertEqual(result["status"], "OK")

    def test_mismatch_reports_actual_excerpt(self):
        result = check_span_against_text("Ethical Challenges: things happen.", 0, 13, "Cultural Bias")
        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(result["excerpt"], "Ethical Chall")

    def test_out_of_bounds(self):
        result = check_span_against_text("short text", 5, 500, "text")
        self.assertEqual(result["status"], "OUT_OF_BOUNDS")

    def test_missing_source_text_never_silently_passes(self):
        result = check_span_against_text(None, 0, 5, "Agile")
        self.assertEqual(result["status"], "MISSING_SOURCE_TEXT")
        result2 = check_span_against_text("", 0, 5, "Agile")
        self.assertEqual(result2["status"], "MISSING_SOURCE_TEXT")

    def test_missing_span(self):
        result = check_span_against_text("some text", None, None, "text")
        self.assertEqual(result["status"], "MISSING_SPAN")

    def test_normalization_transformation_checks_preserved_original(self):
        """EntityNormalizer.normalize_entity rewrites .label/surface_form
        to a canonical form while preserving the true pre-normalization
        original mention (e.g. in metadata["aliases"]) -- the validator
        must accept a span that matches THAT preserved original even
        though it no longer matches the (transformed) surface_form
        supplied as the primary check target."""
        text = "the vulnerable communities were affected"
        # surface_form is the CANONICAL/resolved form, genuinely different
        # text from what's literally in the source -- the preserved
        # original ("vulnerable communities") is what actually occurs.
        result = check_span_against_text(
            text, 4, 26, "Marginalized Communities",
            preserved_original_mentions=["vulnerable communities"],
        )
        self.assertEqual(result["status"], "OK_VIA_NORMALIZATION")
        self.assertEqual(result["matched_against"], "vulnerable communities")

    def test_normalization_does_not_paper_over_a_genuine_mismatch(self):
        """If the excerpt matches NEITHER the surface_form NOR any
        preserved original, it is still a real MISMATCH -- normalization
        tolerance is not a blanket excuse."""
        text = "completely unrelated sentence here"
        result = check_span_against_text(
            text, 0, 10, "Cultural Bias", preserved_original_mentions=["cultural bias"],
        )
        self.assertEqual(result["status"], "MISMATCH")


class RealTruncationCaseReproductionTests(unittest.TestCase):
    """Reproduces the two still-open cases against the REAL source dataset
    -- proves the validator flags them, confirming they were investigated
    and left unrepaired rather than silently passed."""

    @classmethod
    def setUpClass(cls):
        with open("data/ai_ethics_datasets.json", encoding="utf-8") as f:
            raw = json.load(f)
        cls.theme_by_id = {str(t["id"]): t.get("description", "") for t in raw["Theme"]}

    def test_african_american_women_flagged_as_mismatch(self):
        # Recorded in the current preserved TTL: discourse_id "9",
        # field theme.description, start_char=153, end_char=158.
        text = self.theme_by_id["9"]
        result = check_span_against_text(text, 153, 158, "African American women")
        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(result["excerpt"].strip().lower(), "g, wi".strip().lower())

    def test_minority_communities_flagged_as_mismatch(self):
        # Recorded: discourse_id "15", field theme.description,
        # start_char=481, end_char=553 (72 chars for a 20-char phrase).
        text = self.theme_by_id["15"]
        result = check_span_against_text(text, 481, 553, "minority communities")
        self.assertEqual(result["status"], "MISMATCH")
        self.assertIn("minority communities", result["excerpt"].lower())
        self.assertNotEqual(result["excerpt"].strip().lower(), "minority communities")

    def test_minority_communities_true_occurrence_would_pass(self):
        """Confirms the validator itself is correct (not overly strict):
        the genuinely correct span for this same entity passes cleanly."""
        text = self.theme_by_id["15"]
        result = check_span_against_text(text, 533, 553, "minority communities")
        self.assertEqual(result["status"], "OK")

    def test_an_employer_reported_as_missing_not_repaired(self):
        """'an employer': the surface form does not occur verbatim
        anywhere in its tagged source text at all -- reported as
        unresolved provenance, not papered over with an invented span."""
        with open("data/ai_ethics_datasets.json", encoding="utf-8") as f:
            raw = json.load(f)
        answer_by_id = {str(a["id"]): a.get("description", "") for a in raw["Answer"]}
        text = answer_by_id.get("24", "")
        result = check_span_against_text(text, 27, 38, "an employer")
        self.assertEqual(result["status"], "MISMATCH")
        self.assertNotIn("employer", result["excerpt"].lower())


class CheckEntitySpanIntegrityTests(unittest.TestCase):
    def test_uses_primary_mention_when_supplied(self):
        sources = {("theme.description", "14"): "xxxx Cultural Bias is discussed here."}

        def resolve(field, discourse_id):
            return sources.get((field, discourse_id))

        metadata = {
            "surface_form": "Cultural Bias", "field": "theme.description",
            "discourse_id": "9999", "start_char": 1, "end_char": 2,  # stale scalar, must be ignored
        }
        primary_mention = {"field": "theme.description", "discourse_id": "14", "start_char": 5, "end_char": 18}
        result = check_entity_span_integrity(metadata, resolve, primary_mention=primary_mention)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["discourse_id"], "14")

    def test_falls_back_to_scalar_metadata_when_no_primary_mention(self):
        sources = {("answer.description", "5"): "The Agile process works well."}

        def resolve(field, discourse_id):
            return sources.get((field, discourse_id))

        metadata = {"surface_form": "Agile", "field": "answer.description", "discourse_id": "5", "start_char": 4, "end_char": 9}
        result = check_entity_span_integrity(metadata, resolve)
        self.assertEqual(result["status"], "OK")


if __name__ == "__main__":
    unittest.main()
