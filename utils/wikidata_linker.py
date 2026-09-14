"""
utils/wikidata_linker.py
------------------------
Real Wikidata linker using the Wikidata wbsearchentities API.

Architecture (revised -- see Fix_instructions design review, category 1):

    retrieval  -> _fetch_candidates()   cached by normalized query label
    enrichment -> _search_api() + _instance_of()   (aliases/match info kept;
                                                     P31 fetched on demand)
    ranking    -> _rank_candidates() / _score_candidate()   pure, uncached,
                                                             context/type-aware
    acceptance -> _decide()             explicit accept / abstain (NIL)

Only the raw candidate list is cached (it is a property of Wikidata itself,
independent of any one mention). The accept/reject/abstain decision is
NEVER cached -- it is recomputed per call from that call's own context and
type, so the same label used in two different contexts can never leak one
mention's decision onto the other. See CACHE_FORMAT_VERSION below.

    from utils.wikidata_linker import WikidataLinker

    linker = WikidataLinker()
    result = linker.link(
        "CheXpert Dataset",
        entity_type="Dataset",
        context_text="... surrounding source sentence ...",
    )
    # -> {"qid": "Qxxxx", "label": "...", "description": "...", "score": 0.83,
    #     "decision_reason": "accepted", "candidates_considered": 12, ...}
    # or None if no candidate is sufficiently supported (explicit NIL/abstain).
"""

from __future__ import annotations

import logging
import re
import time
import random
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Cache format versioning.
#
# v1 (deprecated, never persisted to disk in this codebase but described
# here so a future on-disk cache cannot silently reuse it): keyed
# "normalized label -> accepted decision dict". That shape fuses a
# context-INDEPENDENT fact (which candidates Wikidata has for a search
# string) with a context-DEPENDENT judgement (which one is correct for a
# *particular* mention), so a v1 entry can leak a decision made for one
# mention's context onto an unrelated mention that happens to share a
# label. v2 caches ONLY the raw candidate list; the decision is always
# recomputed per call and is never stored. Any future on-disk persistence
# MUST tag its file with CACHE_FORMAT_VERSION and refuse to load an
# unversioned or v1-shaped file.
# ---------------------------------------------------------------------
CACHE_FORMAT_VERSION = 2

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with", "by", "is", "are",
    "was", "were", "this", "that", "these", "those", "it", "its", "as", "at", "from", "be",
    "being", "been", "has", "have", "had", "not", "no", "which", "who", "whom", "what", "when",
    "where", "how", "also", "other", "one", "used", "use",
}


def _tokens(text: Any) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if t not in _STOPWORDS and len(t) > 1}


class WikidataLinker:
    """
    Link entity mentions to Wikidata items using the public Wikidata API.

    Public method:
        link(label, *, entity_type=None, context_text=None) -> Optional[Dict[str, Any]]
    """

    API_URL = "https://www.wikidata.org/w/api.php"

    # Wikidata's P31="human" (Q5) is the one stable, unfragmented top-level
    # class covering "this item denotes a specific named person" across the
    # whole graph. It is used ONLY to compare against this pipeline's own
    # Person-family local types -- a general, domain-agnostic mechanism
    # (one class family, not a per-entity or per-QID lookup), not an
    # enumeration of specific entities or predicates.
    #
    # A broader Organisation/AISystem/Dataset P31 whitelist was evaluated
    # and NOT adopted for v1: real P31 values for those families are too
    # fragmented (e.g. a well-known AI system's own P31 values are things
    # like "generative pre-trained transformer" / "foundation model", not
    # any short, generic "software"-style class), so a short hand list
    # would be unreliable and risks exactly the kind of narrow, poorly-
    # generalizing rule this design is meant to avoid. Left as a disclosed,
    # tracked gap rather than silently worked around.
    HUMAN_QID = "Q5"
    PERSON_LOCAL_TYPE_MARKERS = ("person", "foaf:person", "expert")

    # Wikidata's stable class for auto-imported academic-paper metadata
    # (see _publication_mismatch_penalty for the empirical justification).
    SCHOLARLY_ARTICLE_QID = "Q13442814"

    # Explicit external-ontology classes this project's own OntologyMapper
    # (ontology_mappings.py) maps local "Publication"/"Document" types to
    # (bibo:Document, foaf:Document) -- preferred over local-type string
    # matching wherever the caller can supply them, since these are
    # unambiguous, externally-grounded class URIs rather than a guess from
    # a local type string that could be spelled/renamed differently.
    PUBLICATION_ONTOLOGY_URIS = frozenset({
        "http://purl.org/ontology/bibo/Document",
        "http://xmlns.com/foaf/0.1/Document",
    })
    # Fallback ONLY when no explicit ontology type is available at all
    # (missing type evidence, not confirmed incompatibility) -- a local
    # type string match, kept narrow and only used as a secondary signal.
    PUBLICATION_LOCAL_TYPE_MARKERS = ("publication", "document", "creativework", "article", "dataset")

    def __init__(
        self,
        endpoint: str = "https://query.wikidata.org/sparql",  # kept for backwards compat; unused
        language: str = "en",
        max_results: int = 25,
        # Empirically evidenced (2026-09-09 design review): the previous
        # default of 5 was the dominant cause of two confirmed "correct
        # candidate not retrieved" failures (COMPAS's real Wikidata item
        # ranked 16th/33rd of 50; "DBS" -> Disclosure and Barring Service
        # ranked 19th of 50, for bare-acronym/short queries where Wikidata's
        # own relevance ranking does not put the intended item first).
        # 25 is a bounded, general widening (not tuned to any single
        # example) that recovers the DBS case and one of the two COMPAS
        # candidates at negligible extra cost; recovering the more specific
        # COMPAS item (ranked 33rd) would need an even wider fetch, which is
        # a disclosed, tunable trade-off rather than silently accepted.
        # Calibrated 2026-09-09 against a 63-entity held-out set drawn from
        # the current production TTL (27 adjudicated-correct-before
        # controls not in the CSV, plus the 17 CSV-flagged cases as a
        # must-not-regress regression floor -- see
        # Fix_instructions/ design-review notes). A small sweep
        # {(0.55,0.045),(0.55,0.02),(0.58,0.02),(0.60,0.02),(0.60,0.03),
        # (0.62,0.02),(0.65,0.02),(0.60,0.045)} was compared on: how many
        # of the 17 known-bad cases still reproduce their old wrong QID,
        # how many previously-correct controls are retained vs falsely
        # abstained, and how many of a broader adjudicated-wrong set get
        # fixed. (0.60, 0.045) dominated every other point tried (fewest
        # still-wrong CSV cases, most wrong-before-fixed, no change to
        # retention/false-abstention counts) -- chosen for that reason,
        # not because it eliminates all 17 (it does not: 2 of the 17 are
        # still wrong under this exact setting, see design report).
        accept_floor: float = 0.60,
        min_margin: float = 0.045,
        user_agent: str = "ETD-Hub-KG-Pipeline/1.0 (mailto:your-email@example.com)",
        timeout: int = 15,
        max_retries: int = 3,
        rate_limit_delay: float = 0.25,
        min_score: Optional[float] = None,  # backwards-compat alias for accept_floor
    ):
        """
        :param language: Preferred language for labels/descriptions.
        :param max_results: Candidates fetched per search query (see rationale above).
        :param accept_floor: Minimum combined score required to accept the top candidate.
        :param min_margin: Minimum score gap required between the top two candidates;
            below this the ranking is treated as genuinely ambiguous and the linker
            abstains (returns NIL) rather than guessing.
        :param user_agent: User-Agent header (Wikidata requires a descriptive UA).
        :param timeout: HTTP timeout in seconds.
        :param max_retries: Number of retries on transient HTTP errors.
        :param rate_limit_delay: Delay (in seconds) between requests to avoid throttling.
        :param min_score: Deprecated alias for accept_floor, kept so existing callers
            that pass min_score=... keep working unchanged.
        """
        self.language = language
        self.max_results = max_results
        self.accept_floor = accept_floor if min_score is None else min_score
        self.min_margin = min_margin
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay

        self.cache_format_version = CACHE_FORMAT_VERSION
        # normalized label -> {"queries_tried": [...], "retrieval_params": {...}, "candidates": [...]}
        self._candidate_cache: Dict[str, Dict[str, Any]] = {}
        # qid -> list[str] P31 ids, or None if not yet fetched, or [] if confirmed no P31 statements
        self._p31_cache: Dict[str, Optional[List[str]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def link(
        self,
        label: str,
        *,
        entity_type: Optional[str] = None,
        context_text: Optional[str] = None,
        ontology_types: Optional[Any] = None,
        source_grounded_aliases: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Link a mention to the best-supported Wikidata item, or return None
        (explicit NIL/no-link) when no candidate is sufficiently supported.
        Never forces a pick merely because a candidate exists.

        :param source_grounded_aliases: Alternative surface forms the CALLER
            has verified appear in this entity's own source text (e.g.
            entity.metadata["aliases"], filled by EntityNormalizer with the
            genuine pre-normalization mention). These are trusted identity
            evidence. LLM-generated / inferred aliases must NOT be passed
            here -- they can widen retrieval upstream but are not identity
            evidence.

        :param label: The entity's mention text. Callers should pass the
            genuine source surface form (e.g. entity.metadata["surface_form"])
            rather than a normalized/canonicalized label where available --
            normalization can rewrite a mention into a form Wikidata's
            search index will not recognize.
        :param entity_type: The pipeline's own (best-available, post-
            correction) local type for this entity, used for the narrow
            Person/Q5 type-compatibility signal, and as a FALLBACK
            publication-type signal only when ontology_types is not
            supplied (missing explicit type evidence is not treated as
            confirmed incompatibility).
        :param context_text: A window of the entity's own source text
            (NOT a rewritten canonical label, NOT unrelated discourse),
            used as soft, non-gating evidence against candidate descriptions.
        :param ontology_types: The entity's explicit external-ontology
            class URI(s) (e.g. entity.ontology_uri / ontology_classes_curie
            expanded), when available -- str or iterable of str. Preferred
            over entity_type's local-type substring match for the
            publication-type-compatibility signal, since these are
            unambiguous, externally-grounded classes rather than a guess
            from a local type string.
        """
        if not isinstance(label, str):
            return None

        query_label = label.strip()
        if not query_label:
            return None

        entry = self._fetch_candidates(query_label)
        candidates = entry["candidates"]
        if not candidates:
            return None

        ranked = self._rank_candidates(
            query_label, candidates, entity_type=entity_type, context_text=context_text,
            ontology_types=ontology_types, source_grounded_aliases=source_grounded_aliases,
        )
        qid, reason = self._decide(ranked, has_context=bool(context_text and context_text.strip()))
        if qid is None:
            logger.info(f"[WikidataLinker] abstained for '{label}': {reason}")
            return None

        top = next(c for c in ranked if c["qid"] == qid)
        rivals = [c for c in ranked if c["qid"] != qid]
        strong_rivals = [
            c for c in rivals
            if max(float(c.get("lex_label", 0.0)), float(c.get("lex_alias", 0.0))) >= self.LEX_STRONG
        ]

        # External-ID safety (design instruction 4): a downstream consumer
        # (e.g. entity consolidation using a Wikidata QID as same-referent
        # evidence) must be able to tell an authoritative identifier from a
        # heuristic best-guess link. This metadata is exposed; consolidation
        # logic is NOT changed in this pass.
        if bool(top.get("source_grounded_label_match")):
            identity_evidence = "source_grounded_alias"
        elif float(top.get("lex_label", 0.0)) >= self.LEX_STRONG:
            identity_evidence = "exact_label"
        elif float(top.get("lex_alias", 0.0)) >= self.LEX_STRONG:
            identity_evidence = "wikidata_alias"
        else:
            identity_evidence = "fuzzy_label"
        if top.get("type_signal", 0.0) > 0 or int(top.get("fit_shared_n", 0)) >= self.FIT_SHARED_MIN:
            identity_evidence += "+corroborated"

        return {
            "qid": qid,
            "label": top["label"],
            "description": top["description"],
            "score": top["combined"],
            "source": "wikidata-api",
            "decision_reason": reason,
            "candidates_considered": len(candidates),
            "queries_tried": entry["queries_tried"],
            # --- identity provenance / confidence ---
            "identity_evidence": identity_evidence,
            "authoritative": False,  # a heuristic API link, never an asserted identifier
            "lex_label": top.get("lex_label"),
            "lex_alias": top.get("lex_alias"),
            "fit_shared_tokens": int(top.get("fit_shared_n", 0)),
            "type_confirmed": bool(top.get("type_signal", 0.0) > 0),
            "had_strong_homonym_rivals": bool(strong_rivals),
            "runner_up_qid": (rivals[0]["qid"] if rivals else None),
            "runner_up_score": (rivals[0]["combined"] if rivals else None),
        }

    # ------------------------------------------------------------------
    # Retrieval + candidate cache (label-keyed only; never caches a decision)
    # ------------------------------------------------------------------
    def _fetch_candidates(self, label: str) -> Dict[str, Any]:
        # The cache key includes every retrieval-affecting parameter
        # (language, max_results/limit), not just the label -- defense in
        # depth so the cache stays correct even if a future change makes
        # these mutable per-instance (today they are set once at __init__
        # and never reassigned; verified, but not relied upon). Two
        # WikidataLinker instances (or one instance reconfigured over
        # time) with different retrieval parameters can never silently
        # share a candidate list fetched under different parameters.
        key = f"{self._sanitize_search_label(label).lower()}|lang={self.language}|limit={self.max_results}"
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached

        time.sleep(self.rate_limit_delay)
        queries_tried = [label]
        try:
            raw = self._search_api(label, limit=self.max_results)
        except Exception as e:
            logger.warning(f"[WikidataLinker] search error for '{label}': {e}")
            raw = []

        # Generic, domain-agnostic retrieval widening: Wikidata's search API
        # does no stemming/lemmatization, so a plural surface form can miss
        # an item registered only under its singular label (confirmed:
        # "recruiters" returns zero results overlapping with "recruiter",
        # which IS the correct item). This is a deterministic English
        # suffix-stripping heuristic applied to ANY query, never a lookup
        # table of specific words.
        singular = self._naive_singular(label)
        if singular and singular.strip().lower() != label.strip().lower():
            time.sleep(self.rate_limit_delay)
            try:
                queries_tried.append(singular)
                raw = raw + self._search_api(singular, limit=self.max_results)
            except Exception as e:
                logger.warning(f"[WikidataLinker] singular-retry search error for '{singular}': {e}")

        seen = set()
        candidates: List[Dict[str, Any]] = []
        for c in raw:
            qid = c.get("qid")
            if not qid or qid in seen:
                continue
            seen.add(qid)
            candidates.append(c)

        entry = {
            "queries_tried": queries_tried,
            "retrieval_params": {"language": self.language, "type": "item", "limit": self.max_results},
            "candidates": candidates,
        }
        self._candidate_cache[key] = entry
        return entry

    @staticmethod
    def _naive_singular(label: str) -> Optional[str]:
        """Deterministic, general English suffix-stripping -- not a word list.
        Skipped for single-word, all-uppercase queries (e.g. "COMPAS",
        "NEWS", "AIDS"): a trailing 'S' on an acronym is not an English
        plural, and stripping it produces a meaningless query (confirmed:
        "COMPAS" -> "COMPA" during evaluation) -- a general convention
        check (any all-caps token), not a specific-word exception."""
        q = label.strip()
        if q.isupper() and " " not in q:
            return None
        low = q.lower()
        if len(q) > 4 and low.endswith("ies"):
            return q[:-3] + "y"
        if len(q) > 4 and low.endswith("es") and not low.endswith("ses"):
            return q[:-2]
        if len(q) > 3 and low.endswith("s") and not low.endswith("ss"):
            return q[:-1]
        return None

    # ------------------------------------------------------------------
    # Internal: call Wikidata search API (keeps aliases + match info, which
    # the previous version silently discarded)
    # ------------------------------------------------------------------
    def _search_api(self, label: str, limit: int = 25) -> List[Dict[str, Any]]:
        sanitized = self._sanitize_search_label(label)

        params = {
            "action": "wbsearchentities",
            "search": sanitized,
            "language": self.language,
            "uselang": self.language,
            "type": "item",
            "format": "json",
            "limit": str(int(limit)),
        }
        headers = {"User-Agent": self.user_agent}
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(self.API_URL, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    search_results = data.get("search", [])
                    candidates: List[Dict[str, Any]] = []
                    for idx, item in enumerate(search_results):
                        qid = item.get("id")
                        if not qid:
                            continue
                        match = item.get("match") or {}
                        candidates.append({
                            "qid": qid,
                            "label": item.get("label", "") or "",
                            "description": item.get("description", "") or "",
                            "aliases": list(item.get("aliases") or []),
                            "match_type": match.get("type"),
                            "match_text": match.get("text"),
                            # Position in THIS query's own result order (0 =
                            # Wikidata's own top relevance rank for this
                            # exact query string). Used as a weak, additive
                            # ranking prior -- see _rank_prior.
                            "api_rank": idx,
                        })
                    return candidates

                if self._is_transient_status(resp.status_code):
                    delay = self._retry_delay(attempt, resp.headers.get("Retry-After"))
                    logger.warning(
                        f"[WikidataLinker] transient HTTP {resp.status_code} for '{label}' "
                        f"(attempt {attempt}/{self.max_retries}); retrying in {delay:.2f}s"
                    )
                    last_exc = Exception(f"HTTP {resp.status_code}")
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        continue
                else:
                    logger.error(
                        f"[WikidataLinker] non-transient HTTP {resp.status_code} for '{label}'; not retrying"
                    )
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
            except requests.RequestException as e:
                last_exc = e
                delay = self._retry_delay(attempt, None)
                logger.warning(
                    f"[WikidataLinker] request error for '{label}' "
                    f"(attempt {attempt}/{self.max_retries}); retrying in {delay:.2f}s: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    continue

        if last_exc:
            logger.error(
                f"[WikidataLinker] final failure for '{label}' after {self.max_retries} attempts: {last_exc}"
            )
            raise last_exc
        return []

    def _instance_of(self, qid: str) -> Optional[List[str]]:
        """P31 (instance-of) values for a QID, cached by QID. Returns None
        only when the fetch itself failed (unknown, not evidence either
        way); returns [] when Wikidata confirms the item has no P31
        statements at all (also not evidence of incompatibility -- just
        absence of a signal, per the 'missing != incompatible' rule)."""
        if qid in self._p31_cache:
            return self._p31_cache[qid]
        try:
            time.sleep(self.rate_limit_delay)
            resp = requests.get(
                self.API_URL,
                params={"action": "wbgetentities", "ids": qid, "props": "claims", "format": "json"},
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                self._p31_cache[qid] = None
                return None
            ent = (resp.json().get("entities") or {}).get(qid) or {}
            claims = (ent.get("claims") or {}).get("P31") or []
            vals = []
            for c in claims:
                try:
                    vals.append(c["mainsnak"]["datavalue"]["value"]["id"])
                except Exception:
                    continue
            self._p31_cache[qid] = vals
            return vals
        except Exception as e:
            logger.warning(f"[WikidataLinker] P31 fetch error for '{qid}': {e}")
            self._p31_cache[qid] = None
            return None

    @staticmethod
    def _is_transient_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= int(status_code) <= 599

    def _retry_delay(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return float(retry_after)
            except Exception:
                pass
        base = min(4.0, 0.5 * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, min(0.25, base * 0.25))
        return max(self.rate_limit_delay, base + jitter)

    # ------------------------------------------------------------------
    # Ranking -- pure, never cached, always computed from THIS call's
    # own context/type
    # ------------------------------------------------------------------
    def _rank_candidates(
        self,
        query_label: str,
        candidates: List[Dict[str, Any]],
        *,
        entity_type: Optional[str],
        context_text: Optional[str],
        ontology_types: Optional[Any] = None,
        source_grounded_aliases: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        sg = self._normalize_source_grounded_aliases(source_grounded_aliases)
        scored = [
            self._score_candidate(query_label, c, entity_type, context_text, ontology_types, sg)
            for c in candidates
        ]
        scored.sort(key=lambda x: -x["combined"])
        return scored

    @staticmethod
    def _normalize_source_grounded_aliases(value: Any) -> List[str]:
        """Aliases the CALLER has verified appear in this entity's own
        source text (e.g. entity.metadata['aliases'], which
        EntityNormalizer fills with the genuine pre-normalization surface
        form). These are trusted identity evidence. An LLM-generated or
        otherwise-inferred alias must NOT be passed here -- it may be used
        for retrieval widening upstream, but not as identity evidence."""
        if value is None:
            return []
        items = value if isinstance(value, (list, tuple, set)) else [value]
        out: List[str] = []
        for a in items:
            s = str(a or "").strip()
            if s and s.lower() not in {x.lower() for x in out}:
                out.append(s)
        return out

    def _score_candidate(
        self,
        query_label: str,
        candidate: Dict[str, Any],
        entity_type: Optional[str],
        context_text: Optional[str],
        ontology_types: Optional[Any] = None,
        source_grounded_aliases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        lex = self._lex_score(query_label, candidate)
        lex_label = self._lex_label_score(query_label, candidate)
        lex_alias = self._lex_alias_score(query_label, candidate)
        # A candidate whose primary label matches one of the entity's OWN
        # source-grounded surface forms is identity evidence as strong as a
        # direct label match -- lift lex_label to that best source-grounded
        # ratio. (Only source-grounded forms; Wikidata aliases stay in
        # lex_alias.)
        sg_label_ratio = 0.0
        for a in source_grounded_aliases or []:
            sg_label_ratio = max(sg_label_ratio, self._best_ratio(a, [candidate.get("label", "")]))
        source_grounded_label_match = sg_label_ratio >= 0.92
        if sg_label_ratio > lex_label:
            lex_label = sg_label_ratio

        # INDEPENDENT source-grounded evidence: a caller-verified
        # source-text name that is NOT merely the entity's own query phrase
        # copied into source_grounded_aliases, and that matches a name THIS
        # candidate is independently known by (its label OR a Wikidata
        # alias). An exact phrase match where the source-grounded alias IS
        # the query phrase (e.g. a common noun equal to a homonym's label)
        # is circular and does not prove identity.
        cand_names = [str(candidate.get("label", ""))] + [
            str(x) for x in (candidate.get("aliases") or [])
        ]
        source_grounded_alias_independent = False
        for a in source_grounded_aliases or []:
            a = str(a).strip()
            if not a:
                continue
            if self._best_ratio(a, [query_label]) >= self.NEAR_EXACT:
                continue  # this alias is (essentially) the query phrase itself
            if self._best_ratio(a, cand_names) >= 0.92:
                source_grounded_alias_independent = True
                break

        has_context = bool(context_text and context_text.strip())
        has_desc = bool(str(candidate.get("description") or "").strip())
        fit = self._context_fit(candidate.get("description", ""), context_text, query_label=query_label)
        fit_shared_n = self._context_shared_count(
            candidate.get("description", ""), context_text, query_label=query_label,
        )
        # The ACTUAL discriminating tokens shared by this candidate's
        # description and the source context (query-own name removed for
        # proper-name queries -- same plumbing as fit_shared_n). _decide
        # uses this set, not just its size, so it can subtract tokens that
        # are generic across the retrieved candidate pool or also present
        # in a homonym rival's description.
        _ov = self._context_overlap_tokens(
            candidate.get("description", ""), context_text, query_label=query_label,
        )
        fit_shared_tokens = sorted(_ov[0] & _ov[1]) if (_ov and _ov[0] and _ov[1]) else []
        # A shared VERBATIM content bigram (two adjacent non-stopword
        # tokens common to the description and the context) is a genuine
        # phrase-level match; scattered single tokens common to a long,
        # generic candidate description are not.
        context_shared_bigram = self._context_shared_bigram(
            candidate.get("description", ""), context_text, query_label=query_label,
        )

        if fit is not None:
            fit_component = fit
        elif not has_context:
            # Context is unavailable for EVERY candidate alike -- a caller-
            # side limitation, not this candidate's fault. True neutral
            # midpoint: cannot discriminate, so don't try to.
            fit_component = 0.5
        else:
            # Context IS available but THIS candidate has no description to
            # compare it against -- a per-candidate evidence gap. Real
            # overlap-coefficient scores for genuinely-correct matches
            # against terse Wikidata descriptions typically fall well below
            # the 0.5 midpoint (confirmed empirically), so treating "no
            # description" as equal to "true neutral" would perversely let
            # an undescribed candidate outscore a described candidate that
            # DOES show real (if modest) supporting overlap. Missing
            # evidence must not score better than present-but-modest
            # evidence -- set below the typical positive range, but still
            # far above zero (not a rejection).
            fit_component = 0.15

        # Missing description is missing EVIDENCE, not evidence of being
        # wrong -- a mild down-weight (not the harsh gate an earlier
        # iteration of this design used), reflecting that we have less to
        # go on, not that the candidate is disfavored.
        desc_richness = 1.0 if has_desc else 0.5

        type_signal = self._type_compatibility(entity_type, candidate.get("qid"))
        pub_penalty = self._publication_mismatch_penalty(
            entity_type, candidate.get("qid"), candidate.get("description"), ontology_types,
        )
        rank_prior = self._rank_prior(candidate.get("api_rank"))

        # Evaluated (2026-09 release preflight) as a candidate signal --
        # Wikidata's own wbsearchentities relevance rank, on the theory
        # that it already encodes notability/popularity information this
        # linker otherwise has no access to. Computed and exposed here for
        # transparency/reproducibility, but NOT included in `combined`:
        # even at a small 0.03 weight it caused real regressions on the
        # full 63-entity development set (COMPAS flipped from its correct
        # replacement to NIL; "recruiters" flipped from a correct NIL to
        # wrongly accepting the "Recruiters" painting, because that
        # painting happens to rank first in Wikidata's own results for the
        # bare query) while NOT fixing Agile (the lexical gap between the
        # wrong exact-label homonym and the correct multi-word-labeled
        # item is too large for a weak prior to close). Per the explicit
        # instruction not to sacrifice known-correct links to repair one
        # known-wrong one, this signal is disclosed but not wired into the
        # decision -- Agile remains a confirmed, un-fixed FALSE_ACCEPTANCE.
        combined = (
            0.50 * lex
            + 0.25 * fit_component
            + 0.10 * desc_richness
            + 0.15 * (0.5 + 0.5 * type_signal)
            - pub_penalty
        )
        return {
            "qid": candidate["qid"],
            "label": candidate.get("label"),
            "description": candidate.get("description"),
            "lex": round(lex, 4),
            "lex_label": round(lex_label, 4),
            "lex_alias": round(lex_alias, 4),
            "source_grounded_label_match": bool(source_grounded_label_match),
            "source_grounded_alias_independent": bool(source_grounded_alias_independent),
            "fit": (round(fit, 4) if fit is not None else None),
            "fit_shared_n": int(fit_shared_n),
            "fit_shared_token_set": list(fit_shared_tokens),
            "context_shared_bigram": bool(context_shared_bigram),
            "desc_richness": desc_richness,
            "type_signal": type_signal,
            "pub_penalty": pub_penalty,
            "rank_prior": round(rank_prior, 4),
            "combined": round(combined, 4),
        }

    @staticmethod
    def _rank_prior(api_rank: Optional[int]) -> float:
        """Wikidata's own wbsearchentities relevance rank for this
        candidate, expressed as a weak, decaying prior (1 / (1 + rank):
        rank 0 -> 1.0, rank 4 -> 0.2, rank 24 -> 0.04). Computed and
        exposed on every scored candidate for transparency, but NOT
        currently used in `combined` -- see _score_candidate's comment for
        why (evaluated, found to cause regressions on known-correct
        controls without fixing the one case it was tried against).
        Missing/unknown rank is neutral (0.0), never penalized."""
        if api_rank is None:
            return 0.0
        try:
            r = int(api_rank)
        except (TypeError, ValueError):
            return 0.0
        if r < 0:
            return 0.0
        return 1.0 / (1.0 + r)

    @staticmethod
    def _best_ratio(query: str, strings: Any) -> float:
        q = str(query or "").strip().lower()
        if not q:
            return 0.0
        best = 0.0
        for s in strings or []:
            s_norm = str(s or "").strip().lower()
            if not s_norm:
                continue
            r = SequenceMatcher(None, q, s_norm).ratio()
            if r > best:
                best = r
        return best

    def _lex_label_score(self, query_label: str, candidate: Dict[str, Any]) -> float:
        """Fuzzy match against the candidate's PRIMARY label only. This is
        the distinctive identity signal: a candidate whose own registered
        name IS (near) the query string is a much stronger identity claim
        than one that merely carries the query as one of many aliases."""
        return self._best_ratio(query_label, [candidate.get("label", "")])

    def _lex_alias_score(self, query_label: str, candidate: Dict[str, Any]) -> float:
        """Fuzzy match against the candidate's Wikidata-registered aliases
        only. A useful RETRIEVAL/recall signal, but weaker identity
        evidence than a primary-label match -- an alias is one of possibly
        dozens of strings pointing loosely at an item, and (for
        LLM/pipeline-supplied aliases, were any ever threaded in here) not
        independently verified at all. See _decide's ambiguity gate."""
        return self._best_ratio(query_label, list(candidate.get("aliases") or []))

    def _lex_score(self, query_label: str, candidate: Dict[str, Any]) -> float:
        """Best fuzzy match against the candidate's label OR any of its
        aliases -- keeps an alias-matched candidate (e.g. 'computational
        model' matched via the alias 'model') competitive in RANKING so it
        can still be surfaced. Identity-strength distinctions between a
        label match and an alias-only match are made in _decide, not here,
        so this stays a pure recall-preserving ranking signal."""
        return max(
            self._lex_label_score(query_label, candidate),
            self._lex_alias_score(query_label, candidate),
        )

    @staticmethod
    def _context_shared_bigram(
        description: str, context_text: Optional[str], *, query_label: str = "",
    ) -> bool:
        """True when the candidate description and the source context share
        a contiguous CONTENT bigram (two adjacent non-stopword tokens),
        with the query's own name removed for proper-name queries (same
        heuristic as _context_overlap_tokens). A verbatim phrase overlap is
        genuine corroboration; a handful of single tokens scattered through
        a long, generic candidate description is not. Purely structural --
        no word list, no domain terms."""
        if not description or not context_text:
            return False

        def _seq(text: Any) -> List[str]:
            return [
                t for t in re.findall(r"[a-z0-9]+", str(text or "").lower())
                if t not in _STOPWORDS and len(t) > 1
            ]

        words = [w for w in str(query_label or "").strip().split() if w]
        drop = (
            _tokens(query_label)
            if (len(words) >= 2 and all(w[:1].isupper() for w in words))
            else set()
        )
        d = [w for w in _seq(description) if w not in drop]
        c = [w for w in _seq(context_text) if w not in drop]
        if len(d) < 2 or len(c) < 2:
            return False
        d_bi = {(d[i], d[i + 1]) for i in range(len(d) - 1)}
        c_bi = {(c[i], c[i + 1]) for i in range(len(c) - 1)}
        return bool(d_bi & c_bi)

    @staticmethod
    def _context_overlap_tokens(
        description: str, context_text: Optional[str], *, query_label: str = "",
    ) -> Optional[Tuple[set, set]]:
        """Shared plumbing for _context_fit and the discriminating-token
        count used by _decide. Returns the (description_tokens,
        context_tokens) sets AFTER query-own-token removal, or None when
        either side has no usable text at all (the 'no signal' case). An
        empty set on one side after removal is returned as an empty set
        (weak/no evidence), NOT None."""
        if not context_text or not description:
            return None
        d_raw = _tokens(description)
        c_raw = _tokens(context_text)
        if not d_raw or not c_raw:
            return None
        words = [w for w in str(query_label or "").strip().split() if w]
        looks_like_proper_name = len(words) >= 2 and all(w[:1].isupper() for w in words)
        if looks_like_proper_name:
            query_tokens = _tokens(query_label)
            return d_raw - query_tokens, c_raw - query_tokens
        return d_raw, c_raw

    def _context_shared_count(
        self, description: str, context_text: Optional[str], *, query_label: str = "",
    ) -> int:
        """Number of DISCRIMINATING tokens (query's own name removed for
        proper-name queries) shared by the candidate description and the
        source context. 0 when there is no context/description or no
        overlap. This is the raw evidence count _decide uses to tell a
        single incidental generic token apart from real corroboration --
        an overlap COEFFICIENT of 0.5 can come from |d∩c|=1,|d|=2, which is
        not itself identity evidence."""
        pair = self._context_overlap_tokens(description, context_text, query_label=query_label)
        if pair is None:
            return 0
        d, c = pair
        if not d or not c:
            return 0
        return len(d & c)

    @staticmethod
    def _context_fit(description: str, context_text: Optional[str], *, query_label: str = "") -> Optional[float]:
        """Overlap-coefficient token overlap between the candidate
        description and the entity's own source-context window: |d ∩ c| /
        min(|d|, |c|), NOT divided by |d| alone. A Wikidata description is
        typically much longer (in distinct tokens) than a short context
        window, so dividing by |d| mechanically suppresses genuine matches
        (confirmed empirically: a manifestly correct description/context
        pair still scored under 0.2 that way) -- the overlap coefficient
        instead asks "how much of the SMALLER side's vocabulary is
        confirmed," which stays meaningful for both long and short sides.

        The query label's OWN tokens are removed from both sides first:
        almost every candidate's description for a query contains the
        query's own name/words (e.g. every "Trevor Paglen" candidate's
        description mentions "Trevor Paglen"), and the context necessarily
        does too (it is the window AROUND that same mention) -- so raw
        self-referential overlap is present for essentially every
        candidate regardless of correctness and must not count as
        evidence. What's left after removing it is the part that actually
        discriminates between candidates.

        The proper-name query-token removal (a multi-word Title-Case
        capitalization heuristic, no word list) is applied inside
        _context_overlap_tokens, shared with _context_shared_count so the
        coefficient and the raw discriminating-token count always agree.

        Returns None (not 0) when either side is unavailable -- callers
        must treat that as 'no signal', never as 'zero overlap =
        incompatible'; zero IS returned when both sides have tokens but
        share none (real negative-ish signal, still soft/additive, not a
        gate)."""
        pair = WikidataLinker._context_overlap_tokens(description, context_text, query_label=query_label)
        if pair is None:
            return None
        d, c = pair
        # Real description/context text WAS present -- if removing the
        # query's own name empties either side out, that side carried NO
        # informative content beyond restating the query itself. Weak/no
        # evidence, same as a genuine zero-overlap match, NOT "no signal".
        if not d or not c:
            return 0.0
        denom = min(len(d), len(c))
        if denom == 0:
            return 0.0
        return len(d & c) / denom

    def _type_compatibility(self, entity_type: Optional[str], qid: Optional[str]) -> float:
        """+1 candidate confirmed compatible, -1 candidate positively
        confirmed incompatible, 0 no local-type signal or no P31 evidence
        either way (missing P31 is missing evidence, never treated as
        incompatibility)."""
        if not qid or not self._is_person_type(entity_type):
            return 0.0
        p31 = self._instance_of(qid)
        if p31 is None or len(p31) == 0:
            return 0.0
        if self.HUMAN_QID in p31:
            return 1.0
        return -1.0

    def _is_person_type(self, entity_type: Optional[str]) -> bool:
        et = str(entity_type or "").lower()
        return any(marker in et for marker in self.PERSON_LOCAL_TYPE_MARKERS)

    # Cheap, general TEXT PRE-FILTER (not the decision itself -- only used
    # to decide whether spending a live P31 lookup is worth it at all).
    # Matches the generic phrasing Wikidata itself uses for auto-imported
    # citation metadata, regardless of subject matter/domain.
    _PUBLICATION_DESCRIPTION_HINTS = (
        "scientific article", "scholarly article", "journal article", "academic journal",
        "conference paper", "published in", "published on", "research paper",
    )

    def _publication_mismatch_penalty(
        self,
        entity_type: Optional[str],
        qid: Optional[str],
        description: Optional[str] = None,
        ontology_types: Optional[Any] = None,
    ) -> float:
        """General, evidence-driven penalty (not a per-label/per-QID fix):
        Wikidata's SCHOLARLY_ARTICLE_QID (Q13442814) is its single, highly
        stable, near-universal class for auto-imported academic-paper
        metadata -- millions of items, most with no real-world referent
        beyond "a paper happened to be titled this". Empirically (2026-09
        preflight, live P31 lookups): 6 of a 10-item sample of confirmed
        CSV-flagged wrong QIDs carry exactly this P31 (Platform Governance,
        Predictive systems, Filter bubbles and fake news, Algorithmic
        Decision-making and the Law, Minority Communities Research Grant,
        Business opportunities abound in Israel) -- an exact-label-matching
        paper title winning over the intended general concept/organisation/
        system referent. None of the checked legitimate controls (Amazon
        Product Review Dataset, Big Data & Society, computational model,
        COMPAS/software system, algorithmics) carry this P31, so the
        penalty is safe to apply broadly.
        Gated off entirely for entities that plausibly DO refer to a
        publication/document, checked via _is_publication_type (explicit
        ontology class preferred, local-type substring only as fallback --
        see that method)."""
        if not qid or self._is_publication_type(entity_type, ontology_types):
            return 0.0
        desc_lower = str(description or "").lower()
        if not any(hint in desc_lower for hint in self._PUBLICATION_DESCRIPTION_HINTS):
            # Cheap text pre-filter only -- saves a network call for the
            # (large majority of) candidates that plainly aren't citation
            # metadata; does NOT itself decide the penalty.
            return 0.0
        p31 = self._instance_of(qid)
        if not p31:
            return 0.0
        if self.SCHOLARLY_ARTICLE_QID in p31:
            return 0.12
        return 0.0

    def _is_publication_type(self, entity_type: Optional[str], ontology_types: Optional[Any] = None) -> bool:
        """Prefers explicit, externally-grounded ontology class URIs
        (bibo:Document / foaf:Document, as this project's own
        OntologyMapper maps local Publication/Document types to -- see
        ontology_mappings.py) over guessing from the local type string.
        Falls back to the local-type substring check ONLY when no
        ontology_types evidence is supplied at all -- missing explicit
        type evidence is treated as "unknown", not as confirmed
        incompatibility, so the (narrower, still-general) local-type
        fallback still applies rather than defaulting to "not a
        publication"."""
        if ontology_types:
            values = ontology_types if isinstance(ontology_types, (list, tuple, set)) else [ontology_types]
            uris = {str(v).strip() for v in values if str(v or "").strip()}
            if uris & self.PUBLICATION_ONTOLOGY_URIS:
                return True
            # Explicit ontology evidence WAS supplied but named a
            # different, non-publication class -- still fall through to
            # the local-type check below rather than conclude
            # incompatibility from ontology evidence alone, since a local
            # type string can carry information the ontology mapping
            # missed (e.g. "Publication" locally with no ontology_uri
            # populated yet for this particular entity).
        et = str(entity_type or "").lower()
        return any(marker in et for marker in self.PUBLICATION_LOCAL_TYPE_MARKERS)

    # ------------------------------------------------------------------
    # Acceptance / abstention
    # ------------------------------------------------------------------
    # A candidate whose label OR alias matches the query at/above this
    # ratio is a "strong lexical match" -- close enough that lexical
    # similarity alone carries no discriminating power against another
    # candidate that also matches this strongly.
    LEX_STRONG = 0.92
    # Below this, the winner is not even a (near-)exact match to any of its
    # own names -- a fuzzy/partial match that needs real corroboration.
    NEAR_EXACT = 0.90
    # Discriminating shared description/context tokens that count as
    # SUBSTANTIVE corroboration (more than one incidental generic token).
    FIT_SHARED_MIN = 2
    # When context-token overlap is the ONLY corroboration for an ambiguous
    # homonym, the discriminating tokens must ALSO be either a verbatim
    # shared bigram OR a strict majority of the smaller side's content
    # vocabulary (overlap coefficient > this) -- otherwise a few tokens
    # scattered through one long, generic candidate description would count.
    # Not a global floor/margin: it only gates whether
    # token overlap authorises corroboration in the ambiguous-homonym case.
    CORROBORATION_FIT_MIN = 0.5

    def _discriminating_shared_tokens(
        self,
        top: Dict[str, Any],
        rival: Optional[Dict[str, Any]],
        pool: Optional[List[Dict[str, Any]]],
    ) -> set:
        """The winner's description<->context shared tokens that actually
        DISCRIMINATE it from its homonyms -- i.e. minus:
          * tokens that also occur in the strongest lexical rival's own
            description (they cannot separate the two), and
          * tokens that are generic domain vocabulary in THIS retrieval:
            present in more than a third of the OTHER retrieved
            candidates' descriptions.
        Both filters are derived only from the candidate set already in
        hand -- no word list, no domain terms, no threshold change."""
        top_shared = set(top.get("fit_shared_token_set") or [])
        if not top_shared:
            return set()
        generic: set = set()
        if rival is not None:
            generic |= top_shared & _tokens(rival.get("description") or "")
        others = [c for c in (pool or []) if c is not top]
        if others:
            df: Dict[str, int] = {}
            for c in others:
                for tok in _tokens(c.get("description") or ""):
                    df[tok] = df.get(tok, 0) + 1
            cap = max(1, len(others) // 3)
            generic |= {t for t in top_shared if df.get(t, 0) > cap}
        return top_shared - generic

    def _winner_is_substantively_corroborated(
        self,
        top: Dict[str, Any],
        rival: Optional[Dict[str, Any]],
        pool: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """At least one GENUINELY INDEPENDENT, non-lexical identity signal
        that `top` -- rather than an equally-well-lexically-matched homonym
        rival -- is the intended referent.

        Deliberately NOT sufficient on their own for an ambiguous homonym
        with no authoritative evidence:
          * an exact source-grounded PHRASE match where the alias is just
            the entity's own query phrase (does not prove identity between
            homonymous common expressions) -- ``source_grounded_label_match``
            is kept only as lexical evidence, upstream in ``lex_label``;
          * a bare combined-score gap between winner and rival (ranks
            candidates, is not identity evidence).
        """
        # 1. Confirmed / compatible type evidence.
        if float(top.get("type_signal", 0.0) or 0.0) > 0:
            return True
        # 2. Authoritative external identifier asserted for this candidate.
        if bool(top.get("authoritative")):
            return True
        # 3. A source-grounded alias independently known to be a name of
        #    THIS candidate (not merely the entity surface phrase).
        if bool(top.get("source_grounded_alias_independent")):
            return True
        # 4. Substantive context corroboration: >= FIT_SHARED_MIN
        #    discriminating shared tokens AND either a verbatim shared
        #    content bigram or a strict-majority overlap coefficient --
        #    guards against scattered incidental overlap inside one long,
        #    generic candidate description.
        if len(self._discriminating_shared_tokens(top, rival, pool)) >= self.FIT_SHARED_MIN:
            if bool(top.get("context_shared_bigram")):
                return True
            fit = top.get("fit")
            if fit is not None and float(fit) > self.CORROBORATION_FIT_MIN:
                return True
        return False

    def _decide(self, ranked: List[Dict[str, Any]], *, has_context: bool) -> Tuple[Optional[str], str]:
        if not ranked:
            return None, "no_candidates"

        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        margin = (top["combined"] - second["combined"]) if second else 1.0

        if top["combined"] < self.accept_floor:
            return None, f"below_floor({top['combined']:.3f}<{self.accept_floor})"

        if second is not None and margin < self.min_margin:
            if not has_context:
                return None, f"no_context_and_ambiguous(margin={margin:.3f})"
            return None, f"ambiguous(margin={margin:.3f})"

        if not has_context:
            # No context signal at all: an exact label match alone must not
            # be treated as establishing identity. Require the winner's
            # LEXICAL margin specifically to be decisive, not merely the
            # combined score (which, with no context, already collapses to
            # a lexically-dominated comparison via the neutral fit
            # component on both sides -- this is a belt-and-suspenders
            # explicit re-check).
            lex_margin = top["lex"] - (second["lex"] if second else 0.0)
            if lex_margin < self.min_margin:
                return None, f"no_context_lex_ambiguous(lex_margin={lex_margin:.3f})"

        # --------------------------------------------------------------
        # B-fix: an ambiguous/near-exact lexical match must not be turned
        # into an accepted identity by weak, generic context overlap.
        # --------------------------------------------------------------
        top_best_lex = max(float(top.get("lex_label", 0.0)), float(top.get("lex_alias", 0.0)))

        # Gate 1 -- MULTIPLE REAL HOMONYMS. Another retrieved candidate
        # matches the query (near-)exactly on label OR alias. An exact
        # label/alias match no longer establishes identity on its own:
        # unless the winner is the ONLY primary-label match (the rivals
        # match only via alias -- the ordinary "distinctive named entity"
        # case), it must be distinguished by substantive, non-generic
        # evidence, else NIL.
        contenders = [
            c for c in ranked
            if c is not top
            and max(float(c.get("lex_label", 0.0)), float(c.get("lex_alias", 0.0))) >= self.LEX_STRONG
        ]
        if contenders:
            winner_is_only_primary_label_match = (
                float(top.get("lex_label", 0.0)) >= self.LEX_STRONG
                and not any(float(c.get("lex_label", 0.0)) >= self.LEX_STRONG for c in contenders)
            )
            if not winner_is_only_primary_label_match:
                # The retrieved pool holds two or more genuine homonyms of
                # this query. A structural rule cannot tell "bare name = the
                # entity itself" from "bare name = a qualified concept",
                # so when the winner carries no substantive, non-generic
                # corroboration over its strongest lexical rival, abstain
                # rather than guess either way.
                if not self._winner_is_substantively_corroborated(top, contenders[0], pool=ranked):
                    return None, (
                        f"ambiguous_lexical_homonyms(contenders={len(contenders) + 1},"
                        f"top_lex_label={top.get('lex_label')},"
                        f"discriminating={sorted(self._discriminating_shared_tokens(top, contenders[0], ranked))})"
                    )

        # Gate 2 -- WINNER IS NOT A REAL MATCH TO ANY OF ITS OWN NAMES.
        # top matched only fuzzily/partially (on both label and alias);
        # accept only with real corroboration (type / source-grounded /
        # >= 2 discriminating context tokens), never on lexical proximity
        # plus a single generic shared word.
        if top_best_lex < self.NEAR_EXACT:
            if not self._winner_is_substantively_corroborated(top, None, pool=ranked):
                return None, (
                    f"weak_lexical_match_uncorroborated(best_lex={top_best_lex:.3f},"
                    f"fit_shared_n={top.get('fit_shared_n')})"
                )

        return top["qid"], "accepted"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_search_label(label: str) -> str:
        s = label.strip()
        s = s.replace('"', " ")
        s = re.sub(r"\s+", " ", s)
        return s
