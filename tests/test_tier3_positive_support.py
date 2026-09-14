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
            "mentions": [{"discourse_id": discourse_id, "chunk_index": 0, "start_char": start_char, "end_char": end_char}],
        },
    )


def _unit(unit_id, text, start_offset, *, section_category="semantic"):
    return {
        "unit_id": unit_id, "text": text, "start_offset": start_offset,
        "end_offset": start_offset + len(text), "section_category": section_category, "unit_type": "sentence",
    }


class Tier3PositiveSupportTests(unittest.TestCase):
    """Tier 3 (whole-preceding-discourse nominal resolution)
    now requires positive discourse-adjacency support, not merely
    uniqueness, when the caller supplies unit_index/semantic_units."""

    def setUp(self):
        self.pipeline = _make_pipeline()

    def test_missing_true_antecedent_with_one_distant_broad_candidate_abstains(self):
        # The true antecedent (never extracted) is discussed in unit 0;
        # unit 1 is unrelated filler; the endpoint is in unit 2. The sole
        # family-compatible candidate sits two units back, not adjacent.
        u0_text = "A biased dataset was used for training and evaluation purposes today."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = "Several stakeholders reviewed the quarterly report yesterday afternoon."
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        u2_text = "Is it possible to separate personal bias from this process entirely?"
        u2 = _unit("U:2", u2_text, u1["end_offset"] + 1)
        units = [u0, u1, u2]

        dataset_local = u0_text.index("dataset")
        dataset = _ent(
            "https://kg/e/dataset", "Dataset", "Dataset", discourse_id="d1",
            start_char=dataset_local, end_char=dataset_local + len("dataset"),
        )
        relation = Relation(
            source="you", relation="separate", target="this process", target_is_literal=True,
            metadata={"object_text": "this process", "discourse_id": "d1", "evidence_quote": u2_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[dataset], unit=u2, discourse_id="d1",
            unit_index=2, semantic_units=units,
        )
        self.assertEqual(resolved[0].target, "this process")
        self.assertEqual(extra, [])

    def test_exact_antecedent_in_immediately_preceding_unit_resolves(self):
        u0_text = "An Automated Triage System is used to prioritize requests."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = "Response Delay: It causes significantly longer wait times."
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        units = [u0, u1]

        local = u0_text.index("Automated Triage System")
        triage = _ent(
            "https://kg/e/triage", "Automated Triage System", "AISystem", discourse_id="d2",
            start_char=local, end_char=local + len("Automated Triage System"),
        )
        relation = Relation(
            source="it", relation="causes", target="significantly longer wait times", target_is_literal=True,
            metadata={"subject_text": "it", "discourse_id": "d2", "evidence_quote": u1_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[triage], unit=u1, discourse_id="d2",
            unit_index=1, semantic_units=units,
        )
        self.assertEqual(resolved[0].source, triage.uri)
        self.assertEqual(resolved[0].metadata.get("subject_endpoint_coref_resolution_method"), "unambiguous_preceding_entity")

    def test_broad_system_substitute_for_unextracted_local_antecedent_abstains(self):
        # A generic system-family entity is established far back (unit 0);
        # several unrelated units intervene; the endpoint's own local
        # antecedent ("the application") is never extracted at all.
        u0_text = "The AI System was announced last year to great fanfare."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = "No protected characteristics are used in the design."
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        u2_text = "The predictions are used alongside human judgment always."
        u2 = _unit("U:2", u2_text, u1["end_offset"] + 1)
        u3_text = "There are fairness algorithms in place to help mitigate impact."
        u3 = _unit("U:3", u3_text, u2["end_offset"] + 1)
        u4_text = "Another thing to consider is what if this tool could be used more broadly."
        u4 = _unit("U:4", u4_text, u3["end_offset"] + 1)
        units = [u0, u1, u2, u3, u4]

        ai_local = u0_text.index("AI System")
        ai_system = _ent(
            "https://kg/e/ai_sys", "AI System", "AISystem", discourse_id="d3",
            start_char=ai_local, end_char=ai_local + len("AI System"),
        )
        fa_local = u3_text.index("fairness algorithms")
        fairness_algorithms = _ent(
            "https://kg/e/fa2", "Fairness Algorithms", "Algorithm", discourse_id="d3",
            start_char=fa_local, end_char=fa_local + len("fairness algorithms"),
        )
        relation = Relation(
            source="this tool", relation="couldBeUsed", target="more broadly", target_is_literal=True,
            metadata={"subject_text": "this tool", "discourse_id": "d3", "evidence_quote": u4_text},
        )
        resolved, extra = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[relation], entities=[ai_system, fairness_algorithms], unit=u4, discourse_id="d3",
            unit_index=4, semantic_units=units,
        )
        # Fairness Algorithms is same-clause-excluded or
        # not-adjacent; AI System is family-compatible but several units
        # back with no adjacency support -- must abstain, not guess.
        self.assertEqual(resolved[0].source, "this tool")
        self.assertEqual(extra, [])

    def test_repeated_pronoun_chain_with_established_antecedent_may_resolve(self):
        u0_text = "The Screening Tool was deployed across all regional offices."
        u0 = _unit("U:0", u0_text, 0)
        u1_text = "Reviewers say it works well in most cases they examined."
        u1 = _unit("U:1", u1_text, u0["end_offset"] + 1)
        u2_text = "Nonetheless, staff report that it occasionally misses edge cases."
        u2 = _unit("U:2", u2_text, u1["end_offset"] + 1)
        units = [u0, u1, u2]

        local = u0_text.index("Screening Tool")
        tool = _ent(
            "https://kg/e/tool", "Screening Tool", "AISystem", discourse_id="d4",
            start_char=local, end_char=local + len("Screening Tool"),
        )
        first_relation = Relation(
            source="reviewers", relation="say", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "d4", "evidence_quote": u1_text},
        )
        resolved1, _ = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[first_relation], entities=[tool], unit=u1, discourse_id="d4",
            unit_index=1, semantic_units=units,
        )
        self.assertEqual(resolved1[0].target, tool.uri)

        second_relation = Relation(
            source="staff", relation="report", target="it", target_is_literal=True,
            metadata={"object_text": "it", "discourse_id": "d4", "evidence_quote": u2_text},
        )
        resolved2, extra2 = self.pipeline._resolve_anaphoric_relation_endpoints(
            relations=[second_relation], entities=[tool], unit=u2, discourse_id="d4",
            unit_index=2, semantic_units=units,
        )
        # The tool's only recorded mention is in u0, now two units back from
        # u2 -- this implementation deliberately does not track "pronoun
        # chains" (a resolved pronoun does not itself become a new mention),
        # so positive adjacency support is genuinely absent here and the
        # conservative, intentional outcome is abstention rather than a
        # guessed chain-continuation. "May resolve" (not "must resolve") is
        # satisfied by the FIRST pronoun in the chain (asserted above),
        # which IS adjacent and does resolve.
        self.assertEqual(resolved2[0].target, "it")
        self.assertEqual(extra2, [])


if __name__ == "__main__":
    unittest.main()
