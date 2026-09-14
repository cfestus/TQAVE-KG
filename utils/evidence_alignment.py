"""
General, domain-agnostic evidence-span alignment.

An LLM-returned evidence_quote is treated as a *locator hint*, not
authoritative source text. This module resolves it deterministically
against the actual Source_Context (the full authoritative source text
already established elsewhere in the pipeline), using a tiered strategy:

  Tier 1 -- exact substring match in the raw source.
  Tier 2 -- reversible-normalization match (markdown emphasis markers,
            backslash-escaped punctuation, Unicode normalization, repeated
            whitespace/line-break collapsing, typographic quote/dash
            normalization) with an index mapping back to raw offsets.
  Tier 3 -- conservative fuzzy alignment (bounded similarity, unique match
            required) that still recovers and returns a verbatim raw-source
            span rather than the paraphrased candidate.
  Tier 4 -- unresolved: no unique/reliable source span found.

Nothing here ever fabricates or rewrites text -- the returned evidence text
is always either the original candidate (Tier 1) or a verbatim substring of
the supplied source_context (Tiers 2-3), never a paraphrase.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

STATUS_EXACT = "exact"
STATUS_NORMALIZED_EXACT = "normalized_exact"
STATUS_FUZZY_RECOVERED = "fuzzy_recovered"
STATUS_UNRESOLVED = "unresolved"

_FUZZY_ACCEPT_RATIO = 0.85
_FUZZY_AMBIGUITY_MARGIN = 0.03

_TYPOGRAPHIC_MAP = {
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "−": "-",
}

_ESCAPABLE_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


@dataclass
class AlignmentResult:
    text: str
    status: str
    method: str
    context_char_start: Optional[int]
    context_char_end: Optional[int]
    original_candidate: Optional[str]


def _normalize_with_mapping(raw: str) -> Tuple[str, List[int]]:
    """
    Reversible, general-purpose formatting normalization.

    Returns (normalized_string, raw_index_of_each_normalized_char) so any
    span found in the normalized string can be mapped back to an exact
    substring of `raw`.
    """
    out_chars: List[str] = []
    raw_starts: List[int] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]

        # Markdown emphasis: "**bold**" -- drop the literal marker only.
        if raw.startswith("**", i):
            i += 2
            continue

        # Backslash-escaped punctuation (Markdown convention): "\-" -> "-".
        if ch == "\\" and i + 1 < n and raw[i + 1] in _ESCAPABLE_PUNCT:
            out_chars.append(raw[i + 1])
            raw_starts.append(i)
            i += 2
            continue

        # Typographic quote/dash normalization.
        if ch in _TYPOGRAPHIC_MAP:
            out_chars.append(_TYPOGRAPHIC_MAP[ch])
            raw_starts.append(i)
            i += 1
            continue

        # Any run of whitespace (including line breaks) collapses to one space.
        if ch.isspace():
            j = i
            while j < n and raw[j].isspace():
                j += 1
            out_chars.append(" ")
            raw_starts.append(i)
            i = j
            continue

        # Unicode normalization for everything else.
        nfkc = unicodedata.normalize("NFKC", ch)
        for c in nfkc:
            out_chars.append(c)
            raw_starts.append(i)
        i += 1

    return "".join(out_chars), raw_starts


def _map_norm_span_to_raw(
    norm_start: int, norm_len: int, raw_starts: List[int], raw_text: str
) -> Tuple[int, int]:
    norm_end_idx = norm_start + norm_len - 1
    raw_start = raw_starts[norm_start]
    if norm_end_idx + 1 < len(raw_starts):
        raw_end = raw_starts[norm_end_idx + 1]
    else:
        raw_end = len(raw_text)
    return raw_start, raw_end


def _find_occurrences(needle: str, haystack: str) -> List[int]:
    if not needle:
        return []
    positions = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _pick_disambiguated(
    positions: List[int],
    raw_starts_for_positions: List[int],
    hint_raw_offset: Optional[int],
) -> Optional[int]:
    """
    Disambiguate among multiple equally-valid textual occurrences using an
    available proximity hint (original extraction offset / chunk position).
    Returns the chosen position, or None if ambiguity cannot be resolved.
    """
    if len(positions) == 1:
        return positions[0]
    if hint_raw_offset is None:
        return None
    dists = sorted(
        (abs(raw_off - hint_raw_offset), pos)
        for pos, raw_off in zip(positions, raw_starts_for_positions)
    )
    if len(dists) >= 2 and dists[0][0] == dists[1][0]:
        return None
    return dists[0][1]


def _snap_to_word_boundaries(context_norm: str, ws: int, we: int) -> Tuple[int, int]:
    """
    Extend a window outward to the nearest whitespace/string boundary so a
    recovered span never starts or ends mid-word. Character-level similarity
    search can otherwise pick a technically-high-scoring window that clips a
    word in half; a verbatim evidence span must read as whole words.
    """
    n = len(context_norm)
    while ws > 0 and not context_norm[ws - 1].isspace():
        ws -= 1
    while we < n and not context_norm[we].isspace():
        we += 1
    return ws, we


def _best_window_at_start(candidate_norm: str, context_norm: str, ws: int) -> Tuple[float, int, int]:
    """
    For an anchor start position, snap it to a word boundary, then search a
    bounded range of end offsets (source phrasing may be longer or shorter
    than the candidate, e.g. an abbreviation expanded in the raw source),
    snapping each candidate end to a word boundary too, and return the best
    (ratio, ws, we). Bounded and cheap: candidate lengths in practice are
    short evidence spans, not full documents.
    """
    ws, _ = _snap_to_word_boundaries(context_norm, ws, ws)
    cand_len = len(candidate_norm)
    lo = ws + max(1, int(cand_len * 0.6))
    hi = min(len(context_norm), ws + int(cand_len * 1.6) + 1)
    best_ratio, best_ws, best_we = 0.0, ws, min(len(context_norm), ws + cand_len)
    seen: set = set()
    for we_raw in range(lo, hi + 1):
        if we_raw <= ws:
            continue
        _, we = _snap_to_word_boundaries(context_norm, ws, we_raw)
        if we in seen:
            continue
        seen.add(we)
        window = context_norm[ws:we]
        ratio = difflib.SequenceMatcher(None, candidate_norm, window, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio, best_ws, best_we = ratio, ws, we
    return best_ratio, best_ws, best_we


def _fuzzy_recover(
    candidate_norm: str, context_norm: str, raw_starts: List[int], raw_text: str
) -> Optional[Tuple[int, int, float, bool]]:
    """
    Conservative fuzzy alignment: anchor on the longest common block(s),
    search a bounded end-offset range from each anchor start (so a source
    span that is somewhat longer/shorter than the candidate -- e.g. an
    abbreviation expanded in the source -- is not truncated), and accept
    only a single, sufficiently-similar, unambiguous best match.
    Returns (raw_start, raw_end, ratio, is_ambiguous) or None if nothing
    reaches the acceptance threshold at all.
    """
    if not candidate_norm or not context_norm:
        return None

    sm = difflib.SequenceMatcher(None, candidate_norm, context_norm, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None

    cand_len = len(candidate_norm)
    anchor_starts = sorted({max(0, block.b - block.a) for block in blocks})
    # Cluster anchors that are close together (within one candidate length)
    # so near-duplicate anchors from the same real occurrence do not get
    # scored/compared as if they were distinct candidate matches.
    clustered: List[int] = []
    for a in anchor_starts:
        if not clustered or a - clustered[-1] > max(1, cand_len):
            clustered.append(a)

    scored: List[Tuple[float, int, int]] = []
    for anchor in clustered:
        ratio, ws, we = _best_window_at_start(candidate_norm, context_norm, anchor)
        if ratio > 0:
            scored.append((ratio, ws, we))

    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best_ratio, best_ws, best_we = scored[0]
    if best_ratio < _FUZZY_ACCEPT_RATIO:
        return None

    is_ambiguous = False
    for ratio, ws, we in scored[1:]:
        if (we <= best_ws or ws >= best_we) and (best_ratio - ratio) <= _FUZZY_AMBIGUITY_MARGIN:
            is_ambiguous = True
            break

    raw_start, raw_end = _map_norm_span_to_raw(best_ws, best_we - best_ws, raw_starts, raw_text)
    return raw_start, raw_end, best_ratio, is_ambiguous


def align_evidence_span(
    candidate: str,
    source_context: str,
    *,
    hint_char_offset: Optional[int] = None,
) -> AlignmentResult:
    """
    Resolve `candidate` (an LLM-produced evidence_quote / locator hint)
    against `source_context` (the authoritative Source_Context text).

    `hint_char_offset` is an optional, best-effort original extraction
    offset used only to disambiguate between multiple otherwise-equal
    textual occurrences -- never used to fabricate a match.
    """
    candidate = (candidate or "").strip()
    source_context = source_context or ""

    if not candidate or not source_context:
        return AlignmentResult(
            text=candidate, status=STATUS_UNRESOLVED, method="no_candidate_or_context",
            context_char_start=None, context_char_end=None, original_candidate=None,
        )

    # Tier 1: exact substring match.
    exact_positions = _find_occurrences(candidate, source_context)
    if exact_positions:
        chosen = _pick_disambiguated(exact_positions, exact_positions, hint_char_offset)
        if chosen is not None:
            return AlignmentResult(
                text=candidate, status=STATUS_EXACT, method="verbatim",
                context_char_start=chosen, context_char_end=chosen + len(candidate),
                original_candidate=None,
            )
        return AlignmentResult(
            text=candidate, status=STATUS_UNRESOLVED,
            method="ambiguous_exact_occurrences_no_disambiguating_signal",
            context_char_start=None, context_char_end=None, original_candidate=candidate,
        )

    # Tier 2: reversible normalization match.
    cand_norm, cand_raw_starts = _normalize_with_mapping(candidate)
    ctx_norm, ctx_raw_starts = _normalize_with_mapping(source_context)
    cand_norm_stripped = cand_norm.strip()
    if cand_norm_stripped:
        norm_positions = _find_occurrences(cand_norm_stripped, ctx_norm)
        if norm_positions:
            raw_offsets = [ctx_raw_starts[p] for p in norm_positions]
            chosen = _pick_disambiguated(norm_positions, raw_offsets, hint_char_offset)
            if chosen is not None:
                raw_start, raw_end = _map_norm_span_to_raw(
                    chosen, len(cand_norm_stripped), ctx_raw_starts, source_context
                )
                recovered = source_context[raw_start:raw_end]
                return AlignmentResult(
                    text=recovered, status=STATUS_NORMALIZED_EXACT,
                    method="markdown/escape/whitespace/unicode normalization",
                    context_char_start=raw_start, context_char_end=raw_end,
                    original_candidate=candidate,
                )
            return AlignmentResult(
                text=candidate, status=STATUS_UNRESOLVED,
                method="ambiguous_normalized_occurrences_no_disambiguating_signal",
                context_char_start=None, context_char_end=None, original_candidate=candidate,
            )

    # Tier 3: conservative fuzzy alignment.
    fuzzy = _fuzzy_recover(cand_norm_stripped, ctx_norm, ctx_raw_starts, source_context)
    if fuzzy is not None:
        raw_start, raw_end, ratio, is_ambiguous = fuzzy
        if is_ambiguous:
            return AlignmentResult(
                text=candidate, status=STATUS_UNRESOLVED,
                method=f"ambiguous_fuzzy_candidates (best_ratio={ratio:.3f})",
                context_char_start=None, context_char_end=None, original_candidate=candidate,
            )
        recovered = source_context[raw_start:raw_end]
        return AlignmentResult(
            text=recovered, status=STATUS_FUZZY_RECOVERED,
            method=f"fuzzy sequence alignment (ratio={ratio:.3f})",
            context_char_start=raw_start, context_char_end=raw_end,
            original_candidate=candidate,
        )

    # Tier 4: unresolved.
    return AlignmentResult(
        text=candidate, status=STATUS_UNRESOLVED, method="no_unique_source_span_found",
        context_char_start=None, context_char_end=None, original_candidate=candidate,
    )
