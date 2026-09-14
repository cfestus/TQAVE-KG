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


def _ent(uri, label, entity_type, *, discourse_id, start_char, end_char, extra_mentions=None):
    mentions = [{"discourse_id": discourse_id, "chunk_index": 0, "start_char": start_char, "end_char": end_char}]
    if extra_mentions:
        mentions.extend(extra_mentions)
    return Entity(
        uri=uri, label=label, entity_type=entity_type, confidence=0.9,
        metadata={
            "discourse_id": discourse_id,
            "section_category": "semantic",
            "start_char": start_char,
            "end_char": end_char,
            "mentions": mentions,
        },
    )


class ClauseLocalOccurrenceResolutionTests(unittest.TestCase):
    """Mention-window handling + clause-local (not whole-unit)
    Tier-2 candidate window + quoted/parenthetical aside boundaries."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_entity_with_old_mention_and_valid_current_clause_mention_is_considered(self):
        # Reproduces the exact real bug shape: an entity's GLOBALLY
        # earliest mention is in an EARLIER unit (would have been wrongly
        # excluded by the old earliest-mention-only check), but it also
        # has a genuine, valid mention inside the CURRENT clause, before
        # the endpoint -- the fix must use that in-window mention.
        prev_unit_text = "The Widget shipped last quarter."
        prev_start = 0
        cur_unit_start = 100
        cur_unit_text = "Reviewers praised the Widget and said it worked well."
        prev_local = prev_unit_text.index("Widget")
        cur_local = cur_unit_text.index("Widget")
        widget = _ent(
            "https://kg/e/widget", "Widget", "AISystem", discourse_id="d1",
            start_char=cur_unit_start + cur_local, end_char=cur_unit_start + cur_local + len("Widget"),
            extra_mentions=[{
                "discourse_id": "d1", "chunk_index": 0,
                "start_char": prev_start + prev_local, "end_char": prev_start + prev_local + len("Widget"),
            }],
        )
        relation = Relation(
            source="reviewers", relation="say", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "d1", "evidence_quote": cur_unit_text},
        )
        unit = {"start_offset": cur_unit_start, "text": cur_unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[widget], unit=unit, discourse_id="d1",
        )
        self.assertEqual(resolved[0].target, widget.uri)
        self.assertEqual(resolved[0].metadata.get("object_endpoint_coref_resolution_method"), "occurrence_level_preceding_mention")
        self.assertEqual(extra, [widget])

    def test_coordinated_fragment_ambiguity_no_false_unique_resolution(self):
        # questions/18 shape: a coordinated NP ("perspectives and biases")
        # was fragmented into two entities by extraction; one fragment's
        # ONLY globally-recorded mention used to look like it was outside
        # the window (old bug), letting the other fragment win uncontested.
        # With the fix, both fragments are correctly seen as in-window
        # candidates -> genuine ambiguity -> abstain.
        unit_start = 200
        unit_text = "is it ever possible to separate a persons own perspectives and biases from this process?"
        persp_local = unit_text.index("perspectives")
        biases_local = unit_text.index("biases")
        perspectives = _ent(
            "https://kg/e/perspectives", "Perspectives", "Bias", discourse_id="d2",
            start_char=unit_start + persp_local, end_char=unit_start + persp_local + len("perspectives"),
        )
        biases = _ent(
            "https://kg/e/biases", "Biases", "Bias", discourse_id="d2",
            start_char=unit_start + biases_local, end_char=unit_start + biases_local + len("biases"),
            extra_mentions=[{
                "discourse_id": "d2", "chunk_index": 0,
                "start_char": 5, "end_char": 11,  # an earlier, unrelated mention far before this unit
            }],
        )
        relation = Relation(
            source="you", relation="separate", target="this process", target_is_literal=True,
            metadata={"object_text": "this process", "discourse_id": "d2", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[perspectives, biases], unit=unit, discourse_id="d2",
        )
        self.assertEqual(resolved[0].target, "this process")
        self.assertEqual(extra, [])

    def test_pronoun_after_nested_quoted_example_not_resolved_to_quoted_entity(self):
        unit_start = 300
        unit_text = "if you say something relevant to the application it can show you 'here is where they mention the widget' but if it is trying to show something you cannot point at nothing"
        widget_local = unit_text.index("widget")
        widget = _ent(
            "https://kg/e/widget2", "Widget", "Dataset", discourse_id="d3",
            start_char=unit_start + widget_local, end_char=unit_start + widget_local + len("widget"),
        )
        relation = Relation(
            source="it", relation="show", target="something",
            target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d3", "evidence_quote": "if it is trying to show something"},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[widget], unit=unit, discourse_id="d3",
        )
        # Widget's only mention is inside the nested single-quoted example;
        # the endpoint "it" is outside that quote -- must not resolve to it.
        self.assertEqual(resolved[0].source, "it")
        self.assertEqual(extra, [])

    def test_oversized_multi_sentence_unit_distant_candidate_not_tier2_local(self):
        prev_unit_text = "An AI System was described earlier."
        prev_start = 0
        cur_unit_start = 100
        cur_unit_text = (
            "There are fairness algorithms in place to help.\n\n"
            "I think when you consider all of this, it is fair by the book, but who knows if it predicts well.\n\n"
            "Another thing to consider is what if this tool could be used by people to predict things."
        )
        fa_local = cur_unit_text.index("fairness algorithms")
        fairness_algorithms = _ent(
            "https://kg/e/fa", "Fairness Algorithms", "Algorithm", discourse_id="d4",
            start_char=cur_unit_start + fa_local, end_char=cur_unit_start + fa_local + len("fairness algorithms"),
        )
        ai_local = prev_unit_text.index("AI System")
        ai_system = _ent(
            "https://kg/e/ai", "AI System", "AISystem", discourse_id="d4",
            start_char=prev_start + ai_local, end_char=prev_start + ai_local + len("AI System"),
        )
        relation = Relation(
            source="this tool", relation="predict", target="things", target_is_literal=True,
            metadata={"subject_text": "this tool", "discourse_id": "d4", "evidence_quote": cur_unit_text},
        )
        unit = {"start_offset": cur_unit_start, "text": cur_unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[fairness_algorithms, ai_system], unit=unit, discourse_id="d4",
        )
        # Must NOT resolve to Fairness Algorithms (several sentences/
        # paragraphs earlier in the same oversized unit) -- Tier 2 finds
        # nothing clause-local, so Tier 3 (unchanged) reaches the
        # preceding-unit system entity instead.
        self.assertEqual(resolved[0].source, ai_system.uri)
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_coref_resolution_method"), "unambiguous_preceding_entity")

    def test_missing_true_antecedent_with_weak_aside_candidate_abstains(self):
        unit_start = 400
        unit_text = "if you ask the reviewer (like the excel example) it will not respond"
        excel_local = unit_text.index("excel example")
        excel_example = _ent(
            "https://kg/e/excel_ex", "Excel Example", "Dataset", discourse_id="d5",
            start_char=unit_start + excel_local, end_char=unit_start + excel_local + len("excel example"),
        )
        relation = Relation(
            source="it", relation="respond", target="query", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d5", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[excel_example], unit=unit, discourse_id="d5",
        )
        self.assertEqual(resolved[0].source, "it")
        self.assertEqual(extra, [])


class ClauseLocalNegativeControlTests(unittest.TestCase):
    """Confirm genuine local antecedents are NOT blocked by the
    new clause/aside logic -- no blanket bans on coordination, existential
    constructions, quotation, or long-but-single-clause sentences."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_genuine_same_clause_plural_antecedent_still_resolves(self):
        unit_start = 500
        unit_text = "Requiring audit trails during your audit and development processes can also make them more robust."
        local = unit_text.index("audit trails")
        audit_trails = _ent(
            "https://kg/e/audit_trails", "Audit Trails", "Process", discourse_id="d6",
            start_char=unit_start + local, end_char=unit_start + local + len("audit trails"),
        )
        relation = Relation(
            source="requiring", relation="make", target="them", target_is_literal=True,
            metadata={"object_text": "them", "discourse_id": "d6", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[audit_trails], unit=unit, discourse_id="d6",
        )
        self.assertEqual(resolved[0].target, audit_trails.uri)
        self.assertEqual(resolved[0].metadata.get("object_endpoint_coref_resolution_method"), "occurrence_level_preceding_mention")

    def test_valid_coordinated_antecedent_not_blocked(self):
        unit_start = 600
        unit_text = "The tests and metrics were flawed, and they needed rework."
        local = unit_text.index("tests and metrics")
        tests_and_metrics = _ent(
            "https://kg/e/tam", "Tests And Metrics", "Concept", discourse_id="d7",
            start_char=unit_start + local, end_char=unit_start + local + len("tests and metrics"),
        )
        relation = Relation(
            source="they", relation="need", target="rework", target_is_literal=True,
            metadata={"subject_text": "they", "discourse_id": "d7", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[tests_and_metrics], unit=unit, discourse_id="d7",
        )
        self.assertEqual(resolved[0].source, tests_and_metrics.uri)

    def test_valid_existential_antecedent_resolves_when_structurally_appropriate(self):
        unit_start = 700
        unit_text = "There is a tool called Model X, and it performs Y."
        local = unit_text.index("Model X")
        model_x = _ent(
            "https://kg/e/model_x", "Model X", "AISystem", discourse_id="d8",
            start_char=unit_start + local, end_char=unit_start + local + len("Model X"),
        )
        relation = Relation(
            source="it", relation="performs", target="Y", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d8", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[model_x], unit=unit, discourse_id="d8",
        )
        self.assertEqual(resolved[0].source, model_x.uri)

    def test_valid_quoted_antecedent_same_quote_not_blocked(self):
        unit_start = 800
        unit_text = "The report says 'the engine failed and it exploded violently'."
        local = unit_text.index("engine")
        engine = _ent(
            "https://kg/e/engine", "Engine", "AISystem", discourse_id="d9",
            start_char=unit_start + local, end_char=unit_start + local + len("engine"),
        )
        relation = Relation(
            source="it", relation="exploded", target="violently", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d9", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[engine], unit=unit, discourse_id="d9",
        )
        self.assertEqual(resolved[0].source, engine.uri)

    def test_long_single_clause_sentence_not_fragmented_by_length_alone(self):
        unit_start = 900
        filler = "very " * 40
        unit_text = f"The Model discusses {filler}many aspects of performance and stakeholders eventually rely on it"
        local = unit_text.index("Model")
        model = _ent(
            "https://kg/e/model_long", "Model", "AISystem", discourse_id="d10",
            start_char=unit_start + local, end_char=unit_start + local + len("Model"),
        )
        relation = Relation(
            source="stakeholders", relation="relyOn", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "d10", "evidence_quote": unit_text},
        )
        unit = {"start_offset": unit_start, "text": unit_text}
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[model], unit=unit, discourse_id="d10",
        )
        self.assertEqual(resolved[0].target, model.uri)


if __name__ == "__main__":
    unittest.main()
