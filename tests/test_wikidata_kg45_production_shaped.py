"""Production-shaped Wikidata homonym regression.

The frozen fixtures in test_wikidata_false_identity.py did not reproduce
a real production run's acceptances of COMPAS / Gold Standard / Algorithms
because:
  * their candidate pools differ in size/composition from the real
    retrieval (so a different candidate ranks top, or the winner has a
    zero score margin);
  * they never pass source_grounded_aliases, so they never exercise the
    circular exact-phrase source-grounded match the real link() path hits
    (link() always prepends the entity surface_form).

These fixtures rebuild that ACTUAL candidate-pool STRUCTURE (rich
generic winner description; a terse correct candidate; several strong
lexical homonym rivals) and pass source_grounded_aliases exactly as
pipeline.py builds them ([surface_form] + verified allow-list). Named
QIDs/labels are regression fixtures only.

Expected: all three -> NIL, because none carries a genuinely independent
identity signal (confirmed type / authoritative id / a source-grounded
alias independently known to be a name of the candidate / a verbatim
shared bigram or strict-majority context overlap).
"""
import unittest
from unittest.mock import patch

from utils.wikidata_linker import WikidataLinker


def _cand(qid, label, description="", aliases=None):
    return {"qid": qid, "label": label, "description": description,
            "aliases": list(aliases or []), "match_type": "label", "match_text": label}


def _link(catalog, query, **kw):
    def _impl(self, label, limit=25, _c=catalog):
        return list(_c.get(label.strip(), []))
    with patch.object(WikidataLinker, "_search_api", _impl):
        return WikidataLinker().link(query, **kw)


def _decide(catalog, query, *, entity_type, context_text, source_grounded_aliases):
    def _impl(self, label, limit=25, _c=catalog):
        return list(_c.get(label.strip(), []))
    with patch.object(WikidataLinker, "_search_api", _impl):
        L = WikidataLinker()
        entry = L._fetch_candidates(query)
        ranked = L._rank_candidates(query, entry["candidates"], entity_type=entity_type,
                                    context_text=context_text,
                                    source_grounded_aliases=source_grounded_aliases)
        return L, ranked, L._decide(ranked, has_context=bool(context_text and context_text.strip()))


# ---------------------------------------------------------------------------
# COMPAS -> Q136407730  (observed: candidates_considered=5, lex_alias=1.0,
# lex_label=0.25, fit_shared_tokens=3, had_strong_homonym_rivals=true,
# runner_up=Q37213748 @ 0.675, score 0.75, source_grounded_alias (circular))
# ---------------------------------------------------------------------------
COMPAS_POOL = [
    _cand("Q136407730", "Institute of Computational Ancient Studies",
          "The CompAS is an institute for Archaeoinformatics and Computational "
          "Philology and Data Science of the Languages of the Ancient World.",
          aliases=["CompAS"]),
    _cand("Q37213748", "Compas", "family name"),
    _cand("Q56294534", "Correctional Offender Management Profiling for Alternative Sanctions",
          "algorithm that calculates a probability of offenders reoffending",
          aliases=["COMPAS"]),
    _cand("Q48811072", "COMPAS", "software system"),
    _cand("Q1129804", "Compas", "genre of Haitian dance music"),
]
COMPAS_CTX = (
    "A predictive tool like COMPAS raises fairness concerns in criminal justice; "
    "even models trained on real-world data can encode systematic bias, and "
    "rigorous science alone cannot fix the computational unfairness in algorithmic "
    "decision-making about defendants and their communities."
)
COMPAS_SGA = ["COMPAS", "Correctional Offender Management Profiling for Alternative Sanctions"]

# ---------------------------------------------------------------------------
# Gold Standard -> Q173509  (observed: candidates_considered=5, lex_label=1.0,
# fit_shared_tokens=1, had_strong_homonym_rivals=true, runner_up @ 0.675,
# score 0.725, source_grounded_alias == surface phrase "gold standard")
# ---------------------------------------------------------------------------
GOLD_POOL = [
    _cand("Q173509", "gold standard", "monetary system based on the value of gold"),
    _cand("Q1572133", "gold standard",
          "diagnostic test or benchmark regarded as definitive under reasonable conditions"),
    _cand("Q49414975", "Gold Standard", "1934 racehorse"),
    _cand("Q5578286", "Gold Standard", "2006 studio album"),
    _cand("Q97355148", "gold standard", "wrestling championship"),
]
GOLD_CTX = (
    "this is a really important consideration when you do not have a gold standard "
    "label, like death, but something more objective to evaluate the model against."
)
GOLD_SGA = ["gold standard"]

# ---------------------------------------------------------------------------
# Algorithms -> Q13636890  (observed: candidates_considered=9, queries
# ["algorithms","algorithm"], lex_label=0.9091, lex_alias=1.0,
# fit_shared_tokens=2, had_strong_homonym_rivals=true, runner_up=Q8366 @
# 0.675, score 0.80 -> combined-score gap 0.125)
# ---------------------------------------------------------------------------
ALGO_POOL = [
    _cand("Q13636890", "algorithmics", "study of algorithms and data structures",
          aliases=["algorithms"]),
    _cand("Q8366", "algorithm",
          "sequence of well-defined instructions to solve a class of problems",
          aliases=["algorithms"]),
    _cand("Q207434", "Algorithm", "2013 studio album by Bass Communion"),
    _cand("Q30587177", "Algorithms", "2014 Indian film"),
    _cand("Q1799600", "algorithmic composition", "technique of using algorithms to create music"),
    _cand("Q380274", "sorting algorithm", "algorithm that arranges lists in order"),
    _cand("Q11367", "greedy algorithm", "algorithmic strategy"),
    _cand("Q1412106", "Euclidean algorithm", "method for computing the greatest common divisor"),
    _cand("Q49773", "social movement", "loosely organized effort by a large group of people"),
]
ALGO_CTX = (
    "Algorithms can perpetuate or even amplify existing biases when they are "
    "trained on biased historical data drawn from unequal social systems and "
    "unaccountable institutional practices."
)
ALGO_SGA = ["Algorithms", "algorithms"]


class KG45ProductionShapedHomonymTests(unittest.TestCase):
    def test_compas_is_nil(self):
        r = _link({"COMPAS": COMPAS_POOL}, "COMPAS", entity_type="AISystem",
                  context_text=COMPAS_CTX, source_grounded_aliases=COMPAS_SGA)
        self.assertIsNone(r)

    def test_gold_standard_is_nil(self):
        r = _link({"gold standard": GOLD_POOL}, "gold standard", entity_type="Concept",
                  context_text=GOLD_CTX, source_grounded_aliases=GOLD_SGA)
        self.assertIsNone(r)

    def test_algorithms_is_nil(self):
        for q in ("algorithms", "algorithm", "Algorithms"):
            r = _link({q: ALGO_POOL}, q, entity_type="Concept",
                      context_text=ALGO_CTX, source_grounded_aliases=ALGO_SGA)
            self.assertIsNone(r, q)

    def test_compas_signals_show_no_independent_corroboration(self):
        L, ranked, (qid, reason) = _decide(
            {"COMPAS": COMPAS_POOL}, "COMPAS", entity_type="AISystem",
            context_text=COMPAS_CTX, source_grounded_aliases=COMPAS_SGA)
        top = ranked[0]
        # the OLD rule would have corroborated (raw source_grounded_label_match
        # and/or a score gap and/or >=2 scattered tokens); the new rule does not
        self.assertFalse(top.get("source_grounded_alias_independent"))
        self.assertFalse(top.get("context_shared_bigram"))
        self.assertIsNone(qid)
        self.assertTrue(reason.startswith("ambiguous_lexical_homonyms") or reason.startswith("ambiguous("))

    def test_gold_standard_circular_source_grounded_alias_is_not_independent(self):
        L, ranked, (qid, reason) = _decide(
            {"gold standard": GOLD_POOL}, "gold standard", entity_type="Concept",
            context_text=GOLD_CTX, source_grounded_aliases=GOLD_SGA)
        top = ranked[0]
        self.assertTrue(top.get("source_grounded_label_match"))          # lexical, kept
        self.assertFalse(top.get("source_grounded_alias_independent"))   # but circular
        self.assertIsNone(qid)

    def test_algorithms_score_gap_alone_does_not_accept(self):
        L, ranked, (qid, reason) = _decide(
            {"algorithms": ALGO_POOL}, "algorithms", entity_type="Concept",
            context_text=ALGO_CTX, source_grounded_aliases=ALGO_SGA)
        top = ranked[0]
        second = ranked[1]
        # a real combined-score gap exists between winner and runner-up ...
        self.assertGreater(top["combined"] - second["combined"], 0.05)
        # ... but it no longer authorises acceptance
        self.assertFalse(top.get("source_grounded_alias_independent"))
        self.assertIsNone(qid)


if __name__ == "__main__":
    unittest.main()
