"""
B -- Wikidata false-identity acceptance.

Confirmed failure class: a lexically exact or near-exact homonym is
ACCEPTED because weak, generic context-token overlap contributes enough
score to cross the acceptance threshold (COMPAS, Dashboard, Gold Standard,
Training On "Real-World" Data, Agile).

Fix (utils/wikidata_linker.py::_decide):
  * Gate 1 -- when >= 2 retrieved candidates match the query (near-)exactly
    on label OR alias, an exact match no longer establishes identity: the
    winner must be the ONLY primary-label match, or be corroborated by
    substantive non-generic evidence (P31 type / source-grounded alias /
    >= 2 discriminating context tokens AND more than the rival / a
    decisive combined-score gap), else NIL.
  * Gate 2 -- when the winner is not a real match to ANY of its own names
    (fuzzy on both label and alias), require the same substantive
    corroboration, else NIL.
  * lex is split into lex_label / lex_alias; alias-only matches are weaker
    identity evidence; a caller-verified source_grounded_alias is trusted.
  * link() exposes identity_evidence / authoritative / lex_* /
    fit_shared_tokens / had_strong_homonym_rivals so a downstream consumer
    can tell a heuristic link from an asserted identifier.

All candidate sets are FROZEN (static Wikidata facts, no network). Real
production contexts are used verbatim. No test label is referenced by
production logic.
"""
import unittest
from unittest.mock import patch

from utils.wikidata_linker import WikidataLinker


def _cand(qid, label, description="", aliases=None):
    return {"qid": qid, "label": label, "description": description,
            "aliases": list(aliases or []), "match_type": "label", "match_text": label}


def _catalog_search(catalog):
    def _impl(self, label, limit=25):
        return list(catalog.get(label.strip(), []))
    return _impl


def _link(catalog, query, **kw):
    with patch.object(WikidataLinker, "_search_api", _catalog_search(catalog)):
        return WikidataLinker().link(query, **kw)


# ---------------------------------------------------------------------------
# Frozen candidate pools for the confirmed false links (trimmed to the
# retrieved items that matter). Real Wikidata content.
# ---------------------------------------------------------------------------

COMPAS = [
    _cand("Q37213748", "Compas", "family name"),
    _cand("Q136407730", "Institute of Computational Ancient Studies",
          "institute for Archaeoinformatics and Computational Philology & Data Science "
          "of the Languages of the Ancient World", aliases=["CompAS"]),
    _cand("Q56294534", "Correctional Offender Management Profiling for Alternative Sanctions",
          "algorithm that calculates a probability of offenders reoffending", aliases=["COMPAS"]),
    _cand("Q48811072", "COMPAS", "software system"),
]
COMPAS_CTX = (
    "A predictive system like COMPAS cannot truly be considered fair if it lacks transparency "
    "and exhibits systemic racial disparities; even if trained on 'real-world' data. Fairness in "
    "algorithmic decision-making, especially in high-stakes domains like criminal justice, goes "
    "beyond technical accuracy; it requires accountability"
)

DASHBOARD = [
    _cand("Q1171872", "Dashboard", "widget engine and application for macOS"),
    _cand("Q1373414", "dashboard", "user interface that organizes and presents information"),
    _cand("Q190752", "dashboard", "control panel of a vehicle"),
]
DASHBOARD_CTX = (
    "Source: City of Chicago Open Data Portal   GitHub Analysis: "
    "https://github.com/toddwschneider/chicago-taxi-data   Dashboard: "
    "https://toddwschneider.com/dashboards/chicago-taxi-ridehailing-data/   Content: 100+ million "
    "rideshare samples analyzed (2018-present)   Includes: Uber, Lyft, Via data"
)

GOLD_STANDARD = [
    _cand("Q173509", "gold standard",
          "monetary system in which the standard economic unit of account is based on a fixed quantity of gold"),
    _cand("Q1572133", "gold standard",
          "diagnostic test or benchmark regarded as definitive under reasonable conditions"),
]
GOLD_STANDARD_CTX = (
    "this is a really important consideration when you don't have a 'gold standard' label, like "
    "'death' but something more objective."
)

REAL_WORLD_DATA = [
    _cand("Q55631277", "Real world data",
          "data relating to patient health status routinely collected from a variety of sources",
          aliases=["RWD"]),
]
REAL_WORLD_DATA_CTX = (
    "This undermines legal principles such as the right to contest evidence and equal protection "
    "under the law.  Training on 'real-world' data is no guarantee of fairness either. Historical "
    "data from the criminal justice system reflects decades of systemic bias"
)

AGILE = [
    _cand("Q30232", "agile software development",
          "group of iterative and incremental software development methods"),
    _cand("Q122795507", "Agile", "ship built in 1982"),
    _cand("Q4692535", "Agile", "race horse"),
]
AGILE_CTX = (
    "a great process can only be built in an augmented capital-a Agile or SCRUM kind of way "
    "(i.e. cycling but retaining quite comprehensive documentation so learning gets embedded)"
)

# ---------------------------------------------------------------------------
# Controls that MUST stay linked / MUST behave as before.
# ---------------------------------------------------------------------------

# distinctive named entity: exactly one candidate whose PRIMARY label is
# the query; the other is a longer, more specific name (not a homonym) ->
# still accepted.
OPENAI = [
    _cand("Q21708200", "OpenAI", "American artificial intelligence research organization"),
    _cand("Q96237091", "OpenAI API", "set of application programming interfaces by OpenAI"),
]
OPENAI_CTX = "who knows what may happen to this data if an organisation uses the OpenAI API for this process"

# ordinary multi-word named entity: single exact match, no rival.
AJL = [_cand("Q104880166", "Algorithmic Justice League",
             "organization founded by Joy Buolamwini to challenge bias in decision-making software")]
AJL_CTX = "the Algorithmic Justice League raises awareness of algorithmic bias"

# short / acronym query with a dominant referent.
NASA = [
    _cand("Q23548", "NASA", "American space and aeronautics agency"),
    _cand("Q6952000", "NASA", "Namibian football club"),
]
NASA_CTX = "the agency has run space missions for decades; NASA's aeronautics research also feeds industry"

# common noun with GENUINE contextual support (>= 2 discriminating tokens).
MODEL = [
    _cand("Q4610556", "model", "person employed to display, advertise and promote products"),
    _cand("Q1122506", "computational model", "mathematical model of a complex system, e.g. weather",
          aliases=["model"]),
]
MODEL_CTX = "bias can occur in the mathematical model of a complex system used for hiring decisions"

# ambiguous common noun with NO real support -> must become NIL.
PLATFORM = [
    _cand("Q241317", "computing platform", "system on which application programs can run"),
    _cand("Q7205", "platform", "raised structure"),
    _cand("Q8514264", "Platform", "2000 film by Jia Zhangke"),
]
PLATFORM_CTX = "the group met on a platform to discuss next steps"


class FalseIdentityRegressionTests(unittest.TestCase):
    """Each confirmed false link: classify + assert the recorded wrong QID
    is no longer accepted."""

    def test_compas_not_the_institute_of_computational_ancient_studies(self):
        # classification: acceptance (correct item Q56294534 IS retrieved;
        # an alias-exact homonym won on incidental "data"/"decision"
        # overlap and cleared the floor+margin).
        r = _link({"COMPAS": COMPAS}, "COMPAS", entity_type="AISystem", context_text=COMPAS_CTX)
        self.assertNotEqual((r or {}).get("qid"), "Q136407730")

    def test_dashboard_not_the_macos_widget_engine(self):
        # classification: acceptance (three exact 'dashboard' homonyms;
        # winner decided by generic overlap, not identity evidence).
        r = _link({"Dashboard": DASHBOARD}, "Dashboard", entity_type="Concept", context_text=DASHBOARD_CTX)
        self.assertNotEqual((r or {}).get("qid"), "Q1171872")

    def test_gold_standard_not_the_monetary_system(self):
        # classification: acceptance (two exact 'gold standard' homonyms).
        r = _link({"Gold Standard": GOLD_STANDARD}, "Gold Standard", entity_type="Concept",
                  context_text=GOLD_STANDARD_CTX)
        self.assertNotEqual((r or {}).get("qid"), "Q173509")

    def test_training_on_real_world_data_not_medical_rwd(self):
        # classification: acceptance (near-exact fuzzy match to a narrower
        # medical concept, accepted on one generic 'data' token).
        r = _link({"Training on 'real-world' data": REAL_WORLD_DATA},
                  "Training on 'real-world' data", entity_type="Concept", context_text=REAL_WORLD_DATA_CTX)
        self.assertNotEqual((r or {}).get("qid"), "Q55631277")

    def test_agile_ambiguous_homonyms_become_nil(self):
        # classification: ranking+acceptance (correct Q30232 retrieved but
        # loses lexically to exact 'Agile' homonyms; none corroborated).
        r = _link({"Agile": AGILE}, "Agile", entity_type="Model", context_text=AGILE_CTX)
        self.assertIsNone(r)

    def test_all_confirmed_false_links_are_rejected(self):
        rejected = {
            "COMPAS": ("Q136407730", _link({"COMPAS": COMPAS}, "COMPAS", entity_type="AISystem", context_text=COMPAS_CTX)),
            "Dashboard": ("Q1171872", _link({"Dashboard": DASHBOARD}, "Dashboard", entity_type="Concept", context_text=DASHBOARD_CTX)),
            "Gold Standard": ("Q173509", _link({"Gold Standard": GOLD_STANDARD}, "Gold Standard", entity_type="Concept", context_text=GOLD_STANDARD_CTX)),
            "Real-World Data": ("Q55631277", _link({"Training on 'real-world' data": REAL_WORLD_DATA}, "Training on 'real-world' data", entity_type="Concept", context_text=REAL_WORLD_DATA_CTX)),
            "Agile": ("Q30232-or-homonym", _link({"Agile": AGILE}, "Agile", entity_type="Model", context_text=AGILE_CTX)),
        }
        for name, (bad_qid, result) in rejected.items():
            got = (result or {}).get("qid")
            self.assertNotEqual(got, bad_qid, f"{name}: still resolves to the confirmed-wrong {bad_qid}")


class CorrectControlRetentionTests(unittest.TestCase):
    def test_openai_distinctive_primary_label_still_links(self):
        r = _link({"OpenAI": OPENAI}, "OpenAI", entity_type="Organisation", context_text=OPENAI_CTX)
        self.assertIsNotNone(r)
        self.assertEqual(r["qid"], "Q21708200")
        self.assertEqual(r["identity_evidence"].split("+")[0], "exact_label")
        self.assertFalse(r["authoritative"])
        self.assertFalse(r["had_strong_homonym_rivals"])

    def test_ordinary_multiword_named_entity_still_links(self):
        r = _link({"Algorithmic Justice League": AJL}, "Algorithmic Justice League",
                  entity_type="Organisation", context_text=AJL_CTX)
        self.assertIsNotNone(r)
        self.assertEqual(r["qid"], "Q104880166")

    def test_short_acronym_query_with_dominant_referent_still_links(self):
        r = _link({"NASA": NASA}, "NASA", entity_type="Organisation", context_text=NASA_CTX)
        self.assertIsNotNone(r)
        self.assertEqual(r["qid"], "Q23548")

    def test_common_noun_with_genuine_contextual_support_still_links(self):
        r = _link({"Model": MODEL}, "Model", entity_type="Model", context_text=MODEL_CTX)
        self.assertIsNotNone(r)
        self.assertEqual(r["qid"], "Q1122506")
        self.assertGreaterEqual(r["fit_shared_tokens"], 2)


class AmbiguousBecomesNilTests(unittest.TestCase):
    def test_ambiguous_common_noun_without_support_becomes_nil(self):
        r = _link({"Platform": PLATFORM}, "Platform", entity_type="Concept", context_text=PLATFORM_CTX)
        self.assertIsNone(r)

    def test_reason_string_names_the_gate(self):
        # exercised via a direct _decide call for a clean assertion on the
        # abstention reason.
        linker = WikidataLinker()
        with patch.object(WikidataLinker, "_search_api", _catalog_search({"Gold Standard": GOLD_STANDARD})):
            entry = linker._fetch_candidates("Gold Standard")
        ranked = linker._rank_candidates("Gold Standard", entry["candidates"],
                                         entity_type="Concept", context_text=GOLD_STANDARD_CTX)
        qid, reason = linker._decide(ranked, has_context=True)
        self.assertIsNone(qid)
        self.assertTrue(reason.startswith("ambiguous_lexical_homonyms") or reason.startswith("ambiguous("))


class AliasAndExternalIdSafetyTests(unittest.TestCase):
    def test_wikidata_alias_only_match_is_labelled_not_exact_label(self):
        cands = [_cand("Q1", "Some Long Canonical Name", "a thing", aliases=["Widget"]),
                 _cand("Q2", "Widget", "a different thing")]
        r = _link({"Widget": cands}, "Widget", entity_type="Concept",
                  context_text="the Widget device is a different thing entirely")
        # whichever it picks, the identity_evidence must reflect the basis
        if r is not None:
            self.assertIn(r["identity_evidence"].split("+")[0],
                          {"exact_label", "wikidata_alias", "fuzzy_label"})
            self.assertIn("lex_label", r)
            self.assertIn("lex_alias", r)
            self.assertFalse(r["authoritative"])

    def test_source_grounded_alias_lifts_an_alias_match_to_trusted(self):
        # 'GPT4' query; the correct item's primary label is 'GPT-4', and
        # the caller has VERIFIED 'GPT-4' appears in the entity's own
        # source text -> treated as a source-grounded label match.
        cands = [
            _cand("Q_RIGHT", "GPT-4", "large language model by OpenAI"),
            _cand("Q_WRONG", "GPT4", "unrelated exact-label item", aliases=["GPT4"]),
        ]
        r = _link({"GPT4": cands}, "GPT4", entity_type="AISystem",
                  context_text="the model is a large language model",
                  source_grounded_aliases=["GPT-4"])
        self.assertIsNotNone(r)
        self.assertEqual(r["qid"], "Q_RIGHT")
        self.assertEqual(r["identity_evidence"].split("+")[0], "source_grounded_alias")

    def test_llm_style_alias_is_not_accepted_as_identity_evidence(self):
        # nothing is passed as source_grounded_aliases; a bare exact-label
        # homonym with only generic overlap must still NIL.
        cands = [
            _cand("Q_A", "Vision", "faculty of sight"),
            _cand("Q_B", "Vision", "2015 Marvel film character"),
        ]
        r = _link({"Vision": cands}, "Vision", entity_type="Concept",
                  context_text="the team shared a vision for the project going forward")
        self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()
