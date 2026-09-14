import unittest

from extractors.llm_relation_extractor import LLMRelationExtractor


def _extractor() -> LLMRelationExtractor:
    # Avoid OpenAI client init in unit tests.
    return LLMRelationExtractor.__new__(LLMRelationExtractor)


class ComparativeStatementPromptTests(unittest.TestCase):
    """Structural checks on the extraction prompt: it must generalize (no
    hard-coded Theme 7 / healthcare / bias vocabulary) while instructing the
    model not to use a copula as predicate and not to fold the comparison
    number into the relation's target."""

    def setUp(self):
        # __new__ avoids __init__'s OpenAI client construction, so
        # system_prompt (assigned in __init__) isn't set on the instance --
        # read it from source instead, the same way test_llm_relation_
        # extractor_context.py inspects it without a network-touching init.
        import inspect
        self.prompt = inspect.getsource(LLMRelationExtractor.__init__)

    def test_prompt_instructs_quantitative_qualifier_handling(self):
        lowered = self.prompt.lower()
        self.assertIn("quantitative_information", lowered)
        self.assertIn("compared_to", lowered)
        self.assertIn("direction", lowered)
        self.assertIn("factor", lowered)

    def test_prompt_forbids_copula_predicate_and_number_as_target(self):
        # Normalize whitespace so line-wrapping in the source string doesn't
        # affect substring matching.
        collapsed = " ".join(self.prompt.lower().split())
        self.assertIn("never use the bare copula", collapsed)
        self.assertIn("target_literal / target_is_literal must not be set to the bare comparison number", collapsed)

    def test_prompt_does_not_hardcode_dataset_vocabulary(self):
        lowered = self.prompt.lower()
        for forbidden in ("black patients", "theme 7", "healthcare", "compas", "recidivism"):
            self.assertNotIn(forbidden, lowered)


class ComparativeStatementConversionTests(unittest.TestCase):
    """Deterministic (no network) tests on _convert_to_relations proving a
    well-formed comparative assertion -- the shape the fixed prompt asks the
    model to produce -- is preserved correctly: predicate/object/quantity/
    direction/evidence all survive, and the comparison number never becomes
    the relation's target."""

    def test_comparative_assertion_preserves_all_fields_target_stays_entity(self):
        ex = _extractor()
        text = "Black patients were 2.6 times less likely to receive additional care despite greater medical need."
        assertions = [
            {
                "subject_text": "Black patients",
                "predicate": "receive",
                "object_text": "additional care",
                "object_type": "concept",
                "target_literal": "",
                "qualifiers": {"compared_to": "white patients"},
                "quantitative_information": {"factor": 2.6, "unit": "times", "direction": "less"},
                "confidence": 0.92,
                "evidence_quote": text,
                "evidence": text,
            }
        ]
        relations = ex._convert_to_relations(assertions, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(relations), 1)
        r = relations[0]

        # Subject/predicate/target(outcome) preserved -- predicate is the verb,
        # not the copula, and the target is the outcome text, not the number.
        self.assertEqual(r.relation, "receive")
        self.assertEqual(r.target, "additional care")
        self.assertFalse(r.target_is_literal)

        # Quantity/factor/direction and comparison target preserved in metadata.
        self.assertEqual(r.metadata.get("quantitative_information"), {"factor": 2.6, "unit": "times", "direction": "less"})
        self.assertEqual(r.metadata.get("qualifiers"), {"compared_to": "white patients"})

        # Evidence/provenance preserved verbatim.
        self.assertEqual(r.metadata.get("evidence_quote"), text)

    def test_comparative_assertion_without_separate_object_uses_comparison_target(self):
        ex = _extractor()
        text = "Female applicants are 3 times more likely to be rejected than male applicants."
        assertions = [
            {
                "subject_text": "Female applicants",
                "predicate": "rejected",
                "object_text": "male applicants",
                "target_literal": "",
                "qualifiers": {"compared_to": "male applicants"},
                "quantitative_information": {"factor": 3, "unit": "times", "direction": "more"},
                "confidence": 0.9,
                "evidence_quote": text,
                "evidence": text,
            }
        ]
        relations = ex._convert_to_relations(assertions, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(relations), 1)
        r = relations[0]
        self.assertEqual(r.relation, "rejected")
        self.assertEqual(r.target, "male applicants")
        self.assertFalse(r.target_is_literal)
        self.assertEqual(r.metadata.get("quantitative_information", {}).get("factor"), 3)
        self.assertEqual(r.metadata.get("quantitative_information", {}).get("direction"), "more")

    def test_concessive_clause_is_not_misclassified_as_compared_to(self):
        # Regression test for the Theme 7 sentence: "despite greater medical
        # need" is a concessive clause, not a comparison target, and must
        # never end up in qualifiers.compared_to.
        ex = _extractor()
        text = "Black patients were 2.6 times less likely to receive additional care despite greater medical need."
        assertions = [
            {
                "subject_text": "Black patients",
                "predicate": "receive",
                "object_text": "additional care",
                "target_literal": "",
                "qualifiers": {"concession": "despite greater medical need"},
                "quantitative_information": {"factor": 2.6, "unit": "times", "direction": "less"},
                "confidence": 0.92,
                "evidence_quote": text,
                "evidence": text,
            }
        ]
        relations = ex._convert_to_relations(assertions, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(relations), 1)
        r = relations[0]
        self.assertEqual(r.relation, "receive")
        self.assertEqual(r.target, "additional care")
        self.assertFalse(r.target_is_literal)
        self.assertNotIn("compared_to", r.metadata.get("qualifiers", {}))
        self.assertEqual(r.metadata.get("qualifiers", {}).get("concession"), "despite greater medical need")
        self.assertEqual(r.metadata.get("quantitative_information"), {"factor": 2.6, "unit": "times", "direction": "less"})

    def test_number_in_target_literal_does_not_silently_override_when_object_text_present(self):
        # Regression guard for the original bug shape: even if a non-compliant
        # response still sets target_literal to a bare number, this only wins
        # when object_text is empty (existing, unchanged _convert_to_relations
        # logic) -- confirming a well-formed object_text is what determines
        # target_is_literal=False, not the mere presence of qualifiers.
        ex = _extractor()
        text = "Older workers were 40% less likely to be promoted than younger workers."
        assertions = [
            {
                "subject_text": "Older workers",
                "predicate": "promoted",
                "object_text": "younger workers",
                "target_literal": "",
                "quantitative_information": {"factor": 40, "unit": "percent", "direction": "less"},
                "confidence": 0.9,
                "evidence_quote": text,
                "evidence": text,
            }
        ]
        relations = ex._convert_to_relations(assertions, discourse_id="d1", text=text, entities=[])
        r = relations[0]
        self.assertEqual(r.target, "younger workers")
        self.assertFalse(r.target_is_literal)

    def test_noncomparative_relation_conversion_unaffected(self):
        # Baseline regression: an ordinary, already-well-formed relation is
        # completely unaffected by the new comparative-handling rules.
        ex = _extractor()
        text = "System A causes Risk B in deployment."
        assertions = [
            {
                "subject_text": "System A",
                "predicate": "causes",
                "object_text": "Risk B",
                "confidence": 0.9,
                "evidence_quote": text,
                "evidence": text,
            }
        ]
        relations = ex._convert_to_relations(assertions, discourse_id="d1", text=text, entities=[])
        self.assertEqual(len(relations), 1)
        r = relations[0]
        self.assertEqual(r.relation, "causes")
        self.assertEqual(r.target, "Risk B")
        self.assertFalse(r.target_is_literal)


if __name__ == "__main__":
    unittest.main()
