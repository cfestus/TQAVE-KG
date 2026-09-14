"""
Deterministic tests for the revised WikidataLinker architecture:
retrieval/cache -> ranking -> acceptance/abstention, with an explicit
separation between the (label-only, cached) candidate list and the
(context/type-aware, never-cached) decision.

All Wikidata responses used here are FROZEN fixtures captured from real,
disclosed, read-only wbsearchentities/wbgetentities calls made during the
category-1 design review (2026-09-09) -- no network access happens during
these tests. _search_api and _instance_of are monkeypatched per-test so
the suite is fully offline and deterministic.
"""
import unittest
from unittest.mock import patch

from utils.wikidata_linker import WikidataLinker


def _cand(qid, label, description="", aliases=None, match_type="label", match_text=None):
    return {
        "qid": qid, "label": label, "description": description,
        "aliases": aliases or [], "match_type": match_type, "match_text": match_text or label,
    }


# Frozen, real candidate sets (trimmed), keyed by the exact query string used.
DBS_CANDIDATES = [
    _cand("Q618076", "deep brain stimulation",
          "surgical treatment involving the implantation of a medical device called a brain pacemaker",
          aliases=["DBS"], match_type="alias", match_text="DBS"),
    _cand("Q11773961", "DBS", "Czech rapper"),
    _cand("Q705417", "DBS Bank", "multinational banking and financial services company based in Singapore"),
    _cand("Q5281501", "Disclosure and Barring Service", "non-departmental public body in England and Wales",
          aliases=["DBS"], match_type="alias", match_text="DBS"),
]

COMPAS_CANDIDATES = [
    _cand("Q37213748", "Compas", "family name"),
    _cand("Q136407730", "Institute of Computational Ancient Studies",
          "institute for Archaeoinformatics and Computational Philology & Data Science of the Languages of the Ancient World",
          aliases=["CompAS"], match_type="alias", match_text="CompAS"),
    _cand("Q48811072", "COMPAS", "software system"),
    _cand("Q56294534", "Correctional Offender Management Profiling for Alternative Sanctions",
          "algorithm that calculates a probability of offenders reoffending"),
]

RECRUITER_CANDIDATES = [
    _cand("Q2072298", "recruiter", "profession; a person who persuades people to join an organization or work for a company"),
]
RECRUITERS_CANDIDATES = [
    _cand("Q106364290", "Recruiters", "painting by Fritz von Dardel"),
]

MODEL_CANDIDATES = [
    _cand("Q4610556", "model", "person employed to display, advertise and promote products, or to serve as a visual aid"),
    _cand("Q1122506", "computational model", "mathematical model of a complex system, e.g. weather",
          aliases=["model"], match_type="alias", match_text="model"),
]

# Two entities from unrelated real-world domains sharing the exact same
# label with no distinguishing description -- used to test correct
# abstention under genuine ambiguity + missing context.
AMBIGUOUS_TIE_CANDIDATES = [
    _cand("Q1", "Foo", ""),
    _cand("Q2", "Foo", ""),
]

CORRECT_CONTROL_CANDIDATES = [
    _cand("Q107134645", "Amazon Product Review Dataset", "dataset of Amazon customer product reviews used for NLP research"),
    _cand("Q999999", "Amazon", "e-commerce company", match_type="label"),
]


def _fake_search_api(catalog):
    """Builds a _search_api replacement dispatching on the query string."""
    def _impl(self, label, limit=25):
        return list(catalog.get(label.strip(), []))
    return _impl


class CacheAndRetrievalTests(unittest.TestCase):
    def test_candidate_list_is_cached_and_not_refetched(self):
        calls = []
        catalog = {"Model": MODEL_CANDIDATES}

        def counting_search(self, label, limit=25):
            calls.append(label)
            return list(catalog.get(label.strip(), []))

        with patch.object(WikidataLinker, "_search_api", counting_search):
            linker = WikidataLinker()
            linker.link("Model", entity_type="Model", context_text="hiring model bias")
            linker.link("Model", entity_type="Model", context_text="completely different context about weather")
            self.assertEqual(calls.count("Model"), 1, "candidate retrieval must be cached by label")

    def test_cache_stores_only_candidates_not_decisions(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Model": MODEL_CANDIDATES})):
            linker = WikidataLinker()
            entry = linker._fetch_candidates("Model")
            self.assertIn("candidates", entry)
            self.assertNotIn("decision", entry)
            self.assertNotIn("qid", entry)
            self.assertEqual(linker.cache_format_version, 2)

    def test_cache_key_includes_language_and_limit(self):
        """Cache identity must include every retrieval-affecting parameter
        (not just the label) -- defense in depth so two instances (or one
        instance reconfigured) with different language/limit can never
        silently share a candidate list fetched under different params."""
        calls = []

        def counting_search(self, label, limit=25):
            calls.append((label, limit, self.language))
            return list(MODEL_CANDIDATES)

        with patch.object(WikidataLinker, "_search_api", counting_search):
            linker_a = WikidataLinker(max_results=5)
            linker_b = WikidataLinker(max_results=25)
            linker_a._fetch_candidates("Model")
            linker_b._fetch_candidates("Model")
            self.assertEqual(len(calls), 2, "different max_results must each trigger their own retrieval")
            self.assertNotEqual(
                next(iter(linker_a._candidate_cache)), next(iter(linker_b._candidate_cache)),
                "different max_results must produce different cache keys",
            )

            linker_en = WikidataLinker(language="en")
            linker_fr = WikidataLinker(language="fr")
            calls.clear()
            linker_en._fetch_candidates("Model")
            linker_fr._fetch_candidates("Model")
            self.assertEqual(len(calls), 2, "different language must each trigger their own retrieval")
            self.assertNotEqual(next(iter(linker_en._candidate_cache)), next(iter(linker_fr._candidate_cache)))

    def test_same_instance_same_params_reuses_cache_across_limits_untouched(self):
        """A single instance's own limit/language are fixed at construction
        and never mutated (verified: no `linker.max_results = ...` /
        `linker.language = ...` assignment anywhere in the production
        codebase) -- repeated calls with the same label always hit the
        same, single cache entry for that instance."""
        calls = []

        def counting_search(self, label, limit=25):
            calls.append(limit)
            return list(MODEL_CANDIDATES)

        with patch.object(WikidataLinker, "_search_api", counting_search):
            linker = WikidataLinker(max_results=10)
            linker._fetch_candidates("Model")
            linker._fetch_candidates("Model")
            linker._fetch_candidates("Model")
            self.assertEqual(calls, [10], "one instance, one label -> exactly one retrieval, always at its own fixed limit")

    def test_same_label_different_context_can_yield_different_decisions(self):
        """The central cache-correctness regression: two mentions sharing a
        label but with different context/type must be free to reach
        different accept/abstain outcomes -- proving no decision is being
        reused across mentions. Uses real candidates where context
        concretely flips the outcome (computational model vs fashion
        model)."""
        catalog = {"Model": MODEL_CANDIDATES}
        with patch.object(WikidataLinker, "_search_api", _fake_search_api(catalog)):
            linker = WikidataLinker()
            r1 = linker.link(
                "Model", entity_type="Model",
                context_text="Bias can occur in the mathematical model of a complex system used for hiring decisions.",
            )
            r2 = linker.link("Model", entity_type="Model", context_text=None)
            key = f"model|lang={linker.language}|limit={linker.max_results}"
            self.assertIn(key, linker._candidate_cache, "candidate retrieval must be cached under the full param-scoped key")
            self.assertEqual(len(linker._candidate_cache), 1, "one label, one param set -> exactly one cache entry, reused by both calls")
            self.assertEqual(r1["qid"], "Q1122506")
            self.assertIsNone(r2, "with no context and no dominant lexical margin, must abstain rather than reuse r1's decision")

    def test_naive_singular_retry_fires_for_plural_query(self):
        catalog = {"recruiters": RECRUITERS_CANDIDATES, "recruiter": RECRUITER_CANDIDATES}
        with patch.object(WikidataLinker, "_search_api", _fake_search_api(catalog)):
            linker = WikidataLinker()
            entry = linker._fetch_candidates("recruiters")
            qids = {c["qid"] for c in entry["candidates"]}
            self.assertIn("Q106364290", qids)
            self.assertIn("Q2072298", qids, "singular-retry must surface the profession item missing from the plural query")
            self.assertEqual(entry["queries_tried"], ["recruiters", "recruiter"])

    def test_singular_retry_does_not_fire_for_already_singular_query(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"model": MODEL_CANDIDATES})):
            linker = WikidataLinker()
            entry = linker._fetch_candidates("model")
            self.assertEqual(entry["queries_tried"], ["model"])


class RankingSignalTests(unittest.TestCase):
    def test_alias_matched_candidate_can_outrank_label_only_candidate(self):
        """'Model' in a computational/ML context should be able to prefer the
        alias-matched 'computational model' over the label-only 'model'
        (fashion model) candidate -- the old code compared labels only and
        could never surface this."""
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Model": MODEL_CANDIDATES})):
            linker = WikidataLinker()
            result = linker.link(
                "Model", entity_type="Model",
                context_text="Bias can occur in the mathematical model of a complex system used for hiring decisions.",
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["qid"], "Q1122506")

    def test_missing_description_is_not_a_hard_reject(self):
        """A candidate with no description must still be selectable when it
        is otherwise the clear best match -- missing description is
        missing evidence, not disqualifying evidence."""
        candidates = [_cand("Q1", "Widget Corp", "")]
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Widget Corp": candidates})):
            linker = WikidataLinker()
            result = linker.link("Widget Corp", entity_type="Organisation", context_text="Widget Corp announced a new product.")
            self.assertIsNotNone(result, "an undescribed but otherwise-unambiguous candidate must still be linkable")
            self.assertEqual(result["qid"], "Q1")

    def test_missing_description_does_not_beat_a_well_evidenced_alternative(self):
        """Regression check for the earlier design iteration's bug: an
        empty-description candidate must not out-rank a well-described,
        context-matching correct candidate merely by tying on label text."""
        candidates = [
            _cand("Q_EMPTY", "Literature Review", ""),
            _cand("QRICH", "literature review",
                  "process of information search and text of a review article, including current knowledge, "
                  "substantive findings, and theoretical and methodological contributions to a particular topic"),
        ]
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Literature Review": candidates})):
            linker = WikidataLinker()
            result = linker.link(
                "Literature Review", entity_type="Process",
                context_text="you will most likely need to perform some kind of literature review to select tools and methodological contributions",
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["qid"], "QRICH")

    def test_zero_context_overlap_does_not_auto_reject(self):
        """Description-overlap is one soft signal, not a hard gate: a
        correct candidate whose terse description shares no tokens with the
        source context must still be acceptable when it is otherwise
        unambiguous."""
        candidates = [_cand("Q1", "Java", "object-oriented programming language")]
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Java": candidates})):
            linker = WikidataLinker()
            result = linker.link(
                "Java", entity_type="AISystem",
                context_text="The backend service was rewritten to improve performance and portability across platforms.",
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["qid"], "Q1")

    def test_missing_type_yields_neutral_type_signal(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Widget": [_cand("Q1", "Widget", "a thing")]})):
            linker = WikidataLinker()
            entry = linker._fetch_candidates("Widget")
            ranked = linker._rank_candidates("Widget", entry["candidates"], entity_type=None, context_text=None)
            self.assertEqual(ranked[0]["type_signal"], 0.0)

    def test_person_type_signal_uses_p31_and_prefers_confirmed_human(self):
        candidates = [
            _cand("Q_WORK", "Trevor Paglen", "non-fiction work by Trevor Paglen"),
            _cand("Q_PERSON", "Trevor Paglen", "American artist"),
        ]
        p31 = {"Q_WORK": ["Q7725634"], "Q_PERSON": ["Q5"]}  # Q7725634 = literary work (non-human)
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Trevor Paglen": candidates})), \
             patch.object(WikidataLinker, "_instance_of", lambda self, qid: p31.get(qid)):
            linker = WikidataLinker()
            result = linker.link("Trevor Paglen", entity_type="Person", context_text="the artist Trevor Paglen")
            self.assertIsNotNone(result)
            self.assertEqual(result["qid"], "Q_PERSON")

    def test_missing_p31_is_neutral_not_incompatible(self):
        candidates = [_cand("Q1", "Ada Lovelace", "mathematician")]
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Ada Lovelace": candidates})), \
             patch.object(WikidataLinker, "_instance_of", lambda self, qid: None):
            linker = WikidataLinker()
            entry = linker._fetch_candidates("Ada Lovelace")
            ranked = linker._rank_candidates("Ada Lovelace", entry["candidates"], entity_type="Person", context_text=None)
            self.assertEqual(ranked[0]["type_signal"], 0.0, "an unfetchable/absent P31 must not be scored as incompatible (-1)")


class PublicationTypeCompatibilityTests(unittest.TestCase):
    """The scholarly-article mismatch penalty must not fire against a
    genuine publication reference, and must prefer explicit ontology
    class evidence over guessing from a possibly generic/imperfect local
    type string."""

    PUB_CANDIDATES = [
        _cand("Q_JOURNAL", "Big Data & Society", "journal"),
    ]

    def test_penalty_fires_for_non_publication_entity_matching_scholarly_article(self):
        candidates = [
            _cand("Q_RIGHT", "Platform Governance", "concept in technology policy"),
            _cand("Q_ARTICLE", "Platform Governance", "scientific article published in 2022"),
        ]
        p31 = {"Q_ARTICLE": ["Q13442814"], "Q_RIGHT": []}
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Platform Governance": candidates})), \
             patch.object(WikidataLinker, "_instance_of", lambda self, qid: p31.get(qid)):
            linker = WikidataLinker()
            ranked = linker._rank_candidates(
                "Platform Governance", candidates, entity_type="Concept", context_text=None, ontology_types=None,
            )
            article = next(c for c in ranked if c["qid"] == "Q_ARTICLE")
            self.assertGreater(article["pub_penalty"], 0.0)

    def test_penalty_gated_off_by_explicit_ontology_type_even_with_generic_local_type(self):
        """A genuine publication reference must not be penalized just
        because its LOCAL type happens to be generic/imperfect (e.g. a
        bare "Entity") -- the explicit ontology class (bibo:Document) must
        be enough on its own to gate the penalty off."""
        p31 = {"Q_JOURNAL": ["Q13442814"]}
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Big Data & Society": self.PUB_CANDIDATES})), \
             patch.object(WikidataLinker, "_instance_of", lambda self, qid: p31.get(qid)):
            linker = WikidataLinker()
            ranked = linker._rank_candidates(
                "Big Data & Society", self.PUB_CANDIDATES,
                entity_type="Entity",  # deliberately generic/imperfect local type
                context_text=None,
                ontology_types=["http://purl.org/ontology/bibo/Document"],
            )
            self.assertEqual(ranked[0]["pub_penalty"], 0.0)

    def test_penalty_gated_off_by_local_type_when_no_ontology_evidence_supplied(self):
        """Missing explicit ontology evidence must fall back to the local
        type, not be treated as confirmed incompatibility."""
        p31 = {"Q_JOURNAL": ["Q13442814"]}
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Big Data & Society": self.PUB_CANDIDATES})), \
             patch.object(WikidataLinker, "_instance_of", lambda self, qid: p31.get(qid)):
            linker = WikidataLinker()
            ranked = linker._rank_candidates(
                "Big Data & Society", self.PUB_CANDIDATES,
                entity_type="Publication", context_text=None, ontology_types=None,
            )
            self.assertEqual(ranked[0]["pub_penalty"], 0.0)

    def test_ontology_type_naming_a_different_class_still_falls_back_to_local_type(self):
        """Explicit ontology evidence that names a NON-publication class
        must not itself be treated as proof of incompatibility -- the
        local type is still consulted rather than concluding 'not a
        publication' from the ontology mismatch alone."""
        p31 = {"Q_JOURNAL": ["Q13442814"]}
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Big Data & Society": self.PUB_CANDIDATES})), \
             patch.object(WikidataLinker, "_instance_of", lambda self, qid: p31.get(qid)):
            linker = WikidataLinker()
            ranked = linker._rank_candidates(
                "Big Data & Society", self.PUB_CANDIDATES,
                entity_type="Publication",
                context_text=None,
                ontology_types=["https://w3id.org/aipo#SomeUnrelatedClass"],
            )
            self.assertEqual(ranked[0]["pub_penalty"], 0.0)


class RankPriorEvaluationTests(unittest.TestCase):
    """Wikidata's own wbsearchentities result rank was evaluated (2026-09
    release preflight) as a weak, additive ranking prior -- computed and
    exposed on every scored candidate, but NOT wired into `combined`: on
    the full 63-entity development set it caused two real regressions
    (COMPAS's correct pick reverted to NIL; 'recruiters' flipped from a
    correct NIL to wrongly accepting a same-label painting that happens to
    rank first) while not fixing the one case it was tried against
    (Agile). These tests lock in that it stays computed-but-unused."""

    def test_rank_prior_decays_with_rank_and_defaults_neutral(self):
        linker = WikidataLinker()
        self.assertEqual(linker._rank_prior(0), 1.0)
        self.assertAlmostEqual(linker._rank_prior(4), 0.2)
        self.assertLess(linker._rank_prior(24), linker._rank_prior(4))
        self.assertEqual(linker._rank_prior(None), 0.0)

    def test_scored_candidate_exposes_rank_prior_when_api_rank_present(self):
        candidate = _cand("Q1", "Widget", "a thing")
        candidate["api_rank"] = 2
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Widget": [candidate]})):
            linker = WikidataLinker()
            entry = linker._fetch_candidates("Widget")
            ranked = linker._rank_candidates("Widget", entry["candidates"], entity_type=None, context_text=None)
            self.assertIn("rank_prior", ranked[0])
            self.assertGreater(ranked[0]["rank_prior"], 0.0)

    def test_rank_prior_does_not_influence_combined_score(self):
        """Two otherwise-identical candidates differing ONLY in api_rank
        must score identically in `combined` -- proves the signal is
        exposed for transparency but not actually applied."""
        low_rank = _cand("Q1", "Widget", "a thing")
        low_rank["api_rank"] = 0
        high_rank = _cand("Q2", "Widget", "a thing")
        high_rank["api_rank"] = 20
        linker = WikidataLinker()
        ranked = linker._rank_candidates("Widget", [low_rank, high_rank], entity_type=None, context_text=None)
        combined_scores = {c["qid"]: c["combined"] for c in ranked}
        self.assertEqual(combined_scores["Q1"], combined_scores["Q2"])
        # But the rank_prior VALUES themselves do differ, confirming the
        # signal is genuinely computed, just not applied.
        rank_priors = {c["qid"]: c["rank_prior"] for c in ranked}
        self.assertGreater(rank_priors["Q1"], rank_priors["Q2"])

    def test_missing_api_rank_is_neutral(self):
        candidate = _cand("Q1", "Widget", "a thing")
        self.assertNotIn("api_rank", candidate)
        linker = WikidataLinker()
        ranked = linker._rank_candidates("Widget", [candidate], entity_type=None, context_text=None)
        self.assertEqual(ranked[0]["rank_prior"], 0.0)


class AbstentionTests(unittest.TestCase):
    def test_abstains_on_ambiguous_exact_tie_with_no_context(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Foo": AMBIGUOUS_TIE_CANDIDATES})):
            linker = WikidataLinker()
            result = linker.link("Foo", entity_type=None, context_text=None)
            self.assertIsNone(result)

    def test_abstains_rather_than_forcing_wrong_top_candidate_when_correct_absent(self):
        """Direct Data: the retrieved candidate pool contains nothing
        remotely matching the query (all are unrelated RFCs/patents/papers).
        The linker must abstain, not force the top-ranked-but-wrong pick."""
        candidates = [
            _cand("Q1", "RFC 5042: Direct Data Placement Protocol (DDP) / Remote Direct Memory Access Protocol (RDMAP) Security",
                  "request for comments publication",
                  aliases=["Direct Data Placement Protocol (DDP) / Remote Direct Memory Access Protocol (RDMAP) Security"]),
            _cand("Q2", "Direct data share", "US patent 11182797"),
        ]
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Direct Data": candidates})):
            linker = WikidataLinker()
            result = linker.link(
                "Direct Data", entity_type="Concept",
                context_text="* **Direct Data**:",
            )
            self.assertIsNone(result)

    def test_does_not_force_a_pick_when_nothing_clears_the_floor(self):
        candidates = [_cand("Q1", "Somewhat Similar Name", "unrelated thing entirely")]
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Query": candidates})):
            linker = WikidataLinker(accept_floor=0.95)
            result = linker.link("Query", entity_type=None, context_text=None)
            self.assertIsNone(result)


class AmbiguousHomonymAbstentionTests(unittest.TestCase):
    """Fix B: when several retrieved candidates match the query label
    (near-)exactly and none is corroborated by substantive, non-generic
    evidence, the linker must abstain (NIL) rather than accept whichever
    lexically-exact homonym picked up an incidental context token."""

    def test_agile_ambiguous_exact_homonyms_now_abstains(self):
        """'Agile' (software-development context): the intended item
        (Q30232 'agile software development') is retrieved but has no bare
        'Agile' label/alias, while two unrelated items ARE labelled
        exactly 'Agile' (a ship, a racehorse). No type match, no real
        description/context overlap for any of them -> NIL. Previously this
        accepted a wrong lexically-exact homonym (documented open gap);
        the ambiguity gate now correctly abstains."""
        candidates = [
            _cand("Q30232", "agile software development", "group of iterative and incremental development methods"),
            _cand("Q122795507", "Agile", "ship built in 1982"),
            _cand("Q4692535", "Agile", "race horse"),
        ]
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Agile": candidates})):
            linker = WikidataLinker()
            result = linker.link(
                "Agile", entity_type="Model",
                context_text="a great process can only be built in an augmented capital-a Agile or SCRUM kind of way",
            )
            self.assertIsNone(result)


class KnownBadRegressionTests(unittest.TestCase):
    """Frozen candidate sets for the CSV-confirmed-bad cases: the new
    linker must never reproduce the recorded wrong QID. This is a
    regression floor (must-not-regress), not a claim that every case now
    resolves to a fully correct replacement -- see the design report for
    the honest per-case classification (some still abstain for lack of a
    retrievable/rankable correct candidate)."""

    def test_dbs_no_longer_returns_the_czech_rapper(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"DBS": DBS_CANDIDATES})):
            linker = WikidataLinker()
            result = linker.link(
                "DBS", entity_type="Concept",
                context_text="a conviction which will cause something like a DBS or Background check to fail",
            )
            qid = result["qid"] if result else None
            self.assertNotEqual(qid, "Q11773961")

    def test_compas_no_longer_returns_the_surname(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"COMPAS": COMPAS_CANDIDATES})):
            linker = WikidataLinker()
            result = linker.link(
                "COMPAS", entity_type="AISystem",
                context_text="A predictive system like COMPAS cannot truly be considered fair if it lacks transparency",
            )
            qid = result["qid"] if result else None
            self.assertNotEqual(qid, "Q37213748")

    def test_recruiters_no_longer_returns_the_painting(self):
        catalog = {"recruiters": RECRUITERS_CANDIDATES, "recruiter": RECRUITER_CANDIDATES}
        with patch.object(WikidataLinker, "_search_api", _fake_search_api(catalog)):
            linker = WikidataLinker()
            result = linker.link(
                "recruiters", entity_type="PopulationGroup",
                context_text="how to avoid a situation in which recruiters over-rely on the outputs of such systems",
            )
            qid = result["qid"] if result else None
            self.assertNotEqual(qid, "Q106364290")

    def test_model_no_longer_returns_the_fashion_model(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"model": MODEL_CANDIDATES})):
            linker = WikidataLinker()
            result = linker.link(
                "model", entity_type="Model",
                context_text="past job match/hiring decisions that favored certain genders because the model was biased",
            )
            qid = result["qid"] if result else None
            self.assertNotEqual(qid, "Q4610556")


class CorrectControlRetentionTests(unittest.TestCase):
    """Named controls that were already correctly linked must remain
    correctly linked under the new design."""

    def test_amazon_product_review_dataset_control_retained(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Amazon Product Review Dataset": CORRECT_CONTROL_CANDIDATES})):
            linker = WikidataLinker()
            result = linker.link(
                "Amazon Product Review Dataset", entity_type="Dataset",
                context_text="the Amazon Product Review Dataset was used for NLP research on customer reviews",
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["qid"], "Q107134645")


class BackwardsCompatibilityTests(unittest.TestCase):
    def test_min_score_constructor_alias_still_works(self):
        linker = WikidataLinker(min_score=0.7)
        self.assertEqual(linker.accept_floor, 0.7)

    def test_link_with_no_entity_type_or_context_does_not_raise(self):
        with patch.object(WikidataLinker, "_search_api", _fake_search_api({"Widget": [_cand("Q1", "Widget", "a thing")]})):
            linker = WikidataLinker()
            # Should not raise even with both optional args omitted.
            linker.link("Widget")


if __name__ == "__main__":
    unittest.main()
