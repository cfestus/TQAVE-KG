"""General, portable entity-vs-literal decision for quantity/
statistic/scalar-value candidate labels.

Structural, syntactic, and morphological signals only -- no dataset IDs,
record IDs, entity names, specific values, corpus phrases, or domain
vocabulary. The patterns below describe general English quantity-expression
shapes (numeric morphology, currency/percent/multiplier markers, a small
closed set of irregular English plurals, partitive "X of N" constructions)
that apply identically to any future dataset or domain.

Decision priority (most to least conservative):
  1. No digit anywhere in the label -> not a quantity candidate at all.
  2. Whole label is a bare numeric/percent/currency/range/multiplier
     expression -> pure quantity (no residual referent at all).
  3. "<phrase> of <number>" partitive/measure construction, phrase itself
     digit-free -> decomposable (residual = phrase).
  4. "<word> <number>" (word precedes number, e.g. "Model 3", "ISO 27001")
     -> always protected; a proper name/identifier is far more often
     suffixed by a trailing number than prefixed, so this ordering alone
     is treated as a strong entity-identity signal.
  5. "<number> <word>" (number precedes a single trailing word):
     - number is a plausible 4-digit year and no currency/percent marker
       is present -> protected (e.g. "2025 Census").
     - trailing word is acronym-shaped -> protected.
     - a currency/percent/multiplier marker anchors the number -> the
       residual word is treated as decomposable regardless of its own
       morphology (the marker itself is the strong signal here).
     - no marker, but the trailing word is a known irregular English
       plural or ends in -s/-es and is long enough not to be a stray
       token -> decomposable (general plural-noun-as-quantified-category
       shape, e.g. "12,800 <plural noun>").
     - otherwise -> protected (default; ambiguous cases are left alone).
  6. Anything else (multi-token labels, no 2-chunk number/word shape) ->
     protected by default, consistent with "must not be classified as
     quantities merely because they contain numbers."
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from utils.helpers import is_acronym

_NUM = r"(?:\d[\d,]*(?:\.\d+)?)"
_CURRENCY_SYM = r"[£$€]"
_CURRENCY_SYMBOL_TO_CODE = {"£": "GBP", "$": "USD", "€": "EUR"}
_CURRENCY_SCALE = r"(?:million|billion|thousand|bn|mn)"
_PERCENT_MARK = r"(?:%|percent|per\s?cent|percentage\s?points?)"
_MULTIPLIER_WORD = r"(?:times|x|-?fold)"

_ANY_DIGIT_RE = re.compile(r"\d")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_LEADING_NUMBER_RE = re.compile(rf"^{_CURRENCY_SYM}?\s?{_NUM}")

_PURE_NUMBER_RE = re.compile(
    rf"^{_CURRENCY_SYM}?\s?{_NUM}(?:\s*[-–—]\s*{_NUM})?\s*\+?"
    rf"\s*(?:{_CURRENCY_SCALE})?\s*{_PERCENT_MARK}?\s*$",
    re.IGNORECASE,
)
_TRAILING_MULTIPLIER_RE = re.compile(rf"^{_NUM}\s*{_MULTIPLIER_WORD}$", re.IGNORECASE)
_OF_NUMBER_RE = re.compile(
    rf"^(.+?)\s+of\s+{_CURRENCY_SYM}?\s?{_NUM}\s*{_PERCENT_MARK}?$", re.IGNORECASE
)
_WORD_THEN_NUMBER_RE = re.compile(rf"^([A-Za-z][A-Za-z\-]*)\s+{_CURRENCY_SYM}?{_NUM}\+?$")
# Trailing residual may be a short noun phrase (up to 3 tokens, one
# optional leading currency-scale word) rather than a single word --
# e.g. "100+ Million Rideshare Samples", "24,000 Argumentative Essays".
# Decomposability of the *phrase* is still decided on its FINAL token's
# morphology/markers only, in classify_quantity_candidate below.
_NUMBER_THEN_WORD_RE = re.compile(
    rf"^{_CURRENCY_SYM}?\s?{_NUM}\+?\s*{_PERCENT_MARK}?\s+"
    rf"((?:{_CURRENCY_SCALE}\s+)?[A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){{0,2}})$",
    re.IGNORECASE,
)
_HAS_MARKER_RE = re.compile(rf"{_PERCENT_MARK}|{_CURRENCY_SYM}|{_CURRENCY_SCALE}", re.IGNORECASE)

_IRREGULAR_PLURALS = {"women", "men", "children", "people"}


@dataclass
class QuantityLiteralDecision:
    is_pure_quantity: bool = False
    is_decomposable: bool = False
    residual_label: Optional[str] = None
    literal_value: Optional[str] = None
    numeric_value: Optional[float] = None
    datatype: Optional[str] = None
    object_type: Optional[str] = None
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def _clean_numeric_token(raw: str) -> Optional[str]:
    s = re.sub(r"[£$€,]", "", str(raw or "")).strip()
    s = s.rstrip("+").strip()
    if not s:
        return None
    try:
        float(s)
    except ValueError:
        return None
    return s


def _numeric_and_datatype(clean: str) -> (float, str):
    numeric_value = float(clean)
    datatype = "xsd:integer" if "." not in clean else "xsd:decimal"
    return numeric_value, datatype


def _currency_code(text: str) -> Optional[str]:
    for sym, code in _CURRENCY_SYMBOL_TO_CODE.items():
        if sym in text:
            return code
    return None


def _comparator_qualifier(text: str) -> Optional[str]:
    lead = _LEADING_NUMBER_RE.match(text)
    if not lead:
        return None
    tail = text[lead.end():].lstrip()
    if tail.startswith("+"):
        return ">="
    return None


def _pure_decision(text: str, *, reason: str) -> Optional[QuantityLiteralDecision]:
    lead = _LEADING_NUMBER_RE.match(text)
    if not lead:
        return None
    clean = _clean_numeric_token(lead.group(0))
    if clean is None:
        return None
    numeric_value, datatype = _numeric_and_datatype(clean)
    qualifiers: Dict[str, Any] = {}
    object_type = "quantity"
    if re.search(_PERCENT_MARK, text, re.IGNORECASE):
        object_type = "percentage"
        datatype = "xsd:decimal"
        qualifiers["unit"] = "percent"
    currency = _currency_code(text)
    if currency:
        object_type = "money"
        datatype = "xsd:decimal"
        qualifiers["currency"] = currency
        scale_m = re.search(_CURRENCY_SCALE, text, re.IGNORECASE)
        if scale_m:
            qualifiers["scale"] = scale_m.group(0).lower()
    if re.search(_MULTIPLIER_WORD, text, re.IGNORECASE) and object_type == "quantity":
        object_type = "comparison"
        qualifiers["unit"] = "times"
    comparator = _comparator_qualifier(text)
    if comparator:
        qualifiers["comparator"] = comparator
    return QuantityLiteralDecision(
        is_pure_quantity=True,
        literal_value=clean,
        numeric_value=numeric_value,
        datatype=datatype,
        object_type=object_type,
        qualifiers=qualifiers,
        reason=reason,
    )


def _decompose_decision(
    text: str, residual: str, *, reason: str
) -> Optional[QuantityLiteralDecision]:
    lead = _LEADING_NUMBER_RE.match(text)
    number_token = lead.group(0) if lead else None
    if number_token is None:
        num_m = re.search(_NUM, text)
        number_token = num_m.group(0) if num_m else None
    clean = _clean_numeric_token(number_token) if number_token else None
    numeric_value = datatype = None
    qualifiers: Dict[str, Any] = {}
    object_type = "quantity"
    if clean is not None:
        numeric_value, datatype = _numeric_and_datatype(clean)
    if re.search(_PERCENT_MARK, text, re.IGNORECASE):
        object_type = "percentage"
        datatype = "xsd:decimal"
        qualifiers["unit"] = "percent"
    currency = _currency_code(text)
    if currency:
        object_type = "money"
        datatype = "xsd:decimal"
        qualifiers["currency"] = currency
    comparator = _comparator_qualifier(text)
    if comparator:
        qualifiers["comparator"] = comparator
    residual_clean = re.sub(r"\s+", " ", residual).strip(" .,-")
    # A leading preposition ("for", "of", "among", ...) is part of the
    # quantity construction's grammar, not the referent's own name -- e.g.
    # "12% for white students" should decompose to "White Students", not
    # "For White Students". General closed English preposition class, not
    # domain vocabulary.
    residual_clean = re.sub(
        r"^(?:for|of|among|in|within|across|on)\s+", "", residual_clean, flags=re.IGNORECASE
    ).strip()
    if not residual_clean:
        return None
    return QuantityLiteralDecision(
        is_decomposable=True,
        residual_label=residual_clean,
        literal_value=clean,
        numeric_value=numeric_value,
        datatype=datatype,
        object_type=object_type,
        qualifiers=qualifiers,
        reason=reason,
    )


def classify_quantity_candidate(label: str) -> Optional[QuantityLiteralDecision]:
    """Classify a candidate entity/relation-object label as a pure quantity
    literal, a decomposable quantity+referent phrase, or (returning None)
    a protected identifier/name that must stay untouched.

    Returns None whenever the label is not a digit-bearing string at all,
    or whenever it does not confidently match one of the general quantity
    shapes below -- absence of a match is always the conservative outcome."""
    text = str(label or "").strip()
    if not text or not _ANY_DIGIT_RE.search(text):
        return None

    if _PURE_NUMBER_RE.match(text):
        return _pure_decision(text, reason="whole_label_is_numeric_expression")
    if _TRAILING_MULTIPLIER_RE.match(text):
        return _pure_decision(text, reason="whole_label_is_multiplier_expression")

    m = _OF_NUMBER_RE.match(text)
    if m and not _ANY_DIGIT_RE.search(m.group(1)):
        residual = m.group(1).strip()
        if residual and len(residual.split()) <= 4:
            decision = _decompose_decision(text, residual, reason="of_number_partitive_construction")
            if decision is not None:
                return decision

    if _WORD_THEN_NUMBER_RE.match(text):
        # A proper name/identifier is far more often number-suffixed
        # ("Model 3", "ISO 27001") than number-prefixed -- protect.
        return None

    m = _NUMBER_THEN_WORD_RE.match(text)
    if m:
        phrase = m.group(1)
        words = phrase.split()
        # A leading currency-scale word ("million"/"billion"/...) belongs
        # to the quantity expression, not the referent category -- drop it
        # from the residual (but keep evaluating the true final word).
        if words and re.fullmatch(_CURRENCY_SCALE, words[0], re.IGNORECASE):
            words = words[1:]
        if not words:
            return None
        residual_phrase = " ".join(words)
        last_word = words[-1]
        lead = _LEADING_NUMBER_RE.match(text)
        digits_only = re.sub(r"[^\d]", "", lead.group(0)) if lead else ""
        has_marker = bool(_HAS_MARKER_RE.search(text))
        if not has_marker and _YEAR_RE.match(digits_only) and len(words) == 1:
            return None  # plausible year + proper noun, e.g. "2025 Census"
        if any(is_acronym(w) for w in words):
            return None
        if has_marker:
            return _decompose_decision(text, residual_phrase, reason="marker_anchored_quantity_plus_noun")
        wl = last_word.lower()
        if wl in _IRREGULAR_PLURALS or (wl.endswith(("s", "es")) and len(wl) >= 4):
            return _decompose_decision(text, residual_phrase, reason="plural_common_noun_quantity_complement")
        return None

    return None


# ---------------------------------------------------------------------
# Root-cause fix (manual semantic-QA pass, category H): a relation-object
# candidate that is unambiguously a full calendar date (never a bare year
# alone -- e.g. a standalone "1996" is far too often part of a proper name
# or an event label to protect-by-default, so it is deliberately left to
# ordinary entity resolution) should become an xsd:date/xsd:gYearMonth
# literal, not a generic Entity/Concept node. Narrow and conservative by
# design: only fully-specified date shapes match; anything else falls
# through to the existing quantity/entity-resolution path unchanged.
# ---------------------------------------------------------------------
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
_MONTH_NAME_ALT = "|".join(_MONTH_NAMES)
_MONTH_DAY_YEAR_RE = re.compile(
    rf"^({_MONTH_NAME_ALT})\s+(\d{{1,2}}),?\s+(\d{{4}})$", re.IGNORECASE
)
_MONTH_YEAR_RE = re.compile(rf"^({_MONTH_NAME_ALT})\s+(\d{{4}})$", re.IGNORECASE)


def classify_date_candidate(label: str) -> Optional[Dict[str, str]]:
    """Classify a candidate relation-object label as a calendar date.

    Returns {"datatype": "xsd:date"|"xsd:gYearMonth", "value": "..."} when
    `label` unambiguously matches ISO "YYYY-MM-DD", "Month DD, YYYY", or
    "Month YYYY". Returns None for anything else, including a bare year
    alone -- absence of a match is always the conservative outcome, so the
    caller falls through to its existing entity-resolution path."""
    text = str(label or "").strip()
    if not text:
        return None

    m = _ISO_DATE_RE.match(text)
    if m:
        year, month, day = m.groups()
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return {"datatype": "xsd:date", "value": text}
        return None

    m = _MONTH_DAY_YEAR_RE.match(text)
    if m:
        month_name, day, year = m.groups()
        month_num = _MONTH_NAMES.index(month_name.lower()) + 1
        if 1 <= int(day) <= 31:
            return {"datatype": "xsd:date", "value": f"{year}-{month_num:02d}-{int(day):02d}"}
        return None

    m = _MONTH_YEAR_RE.match(text)
    if m:
        month_name, year = m.groups()
        month_num = _MONTH_NAMES.index(month_name.lower()) + 1
        return {"datatype": "xsd:gYearMonth", "value": f"{year}-{month_num:02d}"}

    return None
