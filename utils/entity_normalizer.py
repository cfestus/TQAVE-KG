"""Entity normalization and deterministic canonical linking primitives."""

import hashlib
import re
import unicodedata
from typing import Any, Optional

from config import Config
from ontology.add_schema import canonicalize_class_local
from utils.helpers import is_acronym
from utils.predicate_map import CANONICAL_REL_PREDICATES


def canonicalize_predicate(pred: Any) -> str:
    if pred is None:
        return "relatedTo"
    key = str(pred).strip().lower()
    if not key:
        return "relatedTo"
    return CANONICAL_REL_PREDICATES.get(key, "relatedTo")


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\*\*", "", s)
    while True:
        stripped = re.sub(r"^\s*(?:[-*•]+|\([0-9A-Za-z]+\)|[0-9A-Za-z]+[.)])\s+", "", s)
        if stripped == s:
            break
        s = stripped
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_match_key(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# C.1 -- generic identity primitives for containment/fuzzy merge decisions.
# Domain-neutral English function words only (determiners, prepositions,
# conjunctions, possessive marker). NOT a domain dictionary and NOT a
# label list -- these are the words that never, on their own, distinguish
# one referent from another.
# ---------------------------------------------------------------------------
_ENGLISH_FUNCTION_TOKENS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "such",
    "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "into", "onto", "about", "as", "and", "or", "s",
})


def _strip_possessive(tok: str) -> str:
    t = str(tok or "").strip().lower()
    t = re.sub(r"(?:'s|s')$", "", t)
    return t.rstrip("'")


def inflection_equivalent(tok_a: str, tok_b: str) -> bool:
    """True when two tokens differ only by a regular English plural or a
    possessive -- 'prediction'/'predictions', \"model's\"/'model',
    'policy'/'policies', 'bias'/'biases'. A RELATIONAL check (never
    absolute stemming), so it does not mis-fold a singular noun that
    merely ends in 's' ('bias', 'analysis', 'lens')."""
    a = _strip_possessive(tok_a)
    b = _strip_possessive(tok_b)
    if not a or not b:
        return a == b
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if long == short + "s":
        return True
    if long == short + "es":
        return True
    if short.endswith("y") and long == short[:-1] + "ies":
        return True
    return False


def containment_adds_material_token(short_key: str, long_key: str) -> bool:
    """`short_key` is a substring of `long_key` (both already normalized).
    Returns True when `long_key` carries a CONTENT token that `short_key`
    does not -- a qualifier / modifier / extra head noun that materially
    changes the referent (broader vs narrower, population vs subgroup,
    'non-'/opposite modifier, generic vs qualified noun). Returns False
    when every extra token is an English function word or a
    plural/possessive inflection of a token already present (word-order
    and spelling variants -- a legitimate merge)."""
    short_toks = [t for t in str(short_key or "").split() if t]
    long_toks = [t for t in str(long_key or "").split() if t]
    short_content = [t for t in short_toks if t not in _ENGLISH_FUNCTION_TOKENS]
    for t in long_toks:
        if t in _ENGLISH_FUNCTION_TOKENS:
            continue
        if any(inflection_equivalent(t, s) for s in short_content):
            continue
        return True
    return False


def is_initialism_pair(label_a: str, label_b: str) -> bool:
    """True when one label is a genuine initialism of the other: a single
    all-letter uppercase-ish token whose letters are exactly the leading
    letters of the other label's content words (>= 2 words). General
    convention, no acronym list."""
    a = str(label_a or "").strip()
    b = str(label_b or "").strip()
    if not a or not b:
        return False

    def _try(short: str, long: str) -> bool:
        s = re.sub(r"[^A-Za-z]", "", short)
        if len(s) < 2 or " " in short.strip():
            return False
        if not (short.isupper() or is_acronym(short)):
            return False
        words = [w for w in re.split(r"[^A-Za-z0-9]+", long) if w]
        content = [w for w in words if _norm_match_key(w) not in _ENGLISH_FUNCTION_TOKENS]
        if len(content) < 2:
            return False
        initials = "".join(w[0] for w in content).lower()
        return initials == s.lower()

    return _try(a, b) or _try(b, a)


KNOWN_ACRONYMS = {
    "AI", "ML", "LLM", "NLP", "CV", "RL",
    "GDPR", "OECD", "ISO", "NIST", "IEEE", "IEC",
    "API", "UK", "US", "EU", "UNESCO", "HMDA", "COMPAS",
}

ALIAS_CANONICAL_LABELS = {
    "hmda": "Home Mortgage Disclosure Act",
    "home mortgage disclosure act": "Home Mortgage Disclosure Act",
    "unesco": "UNESCO AI Ethics Recommendation",
    "unesco ai recommendation": "UNESCO AI Ethics Recommendation",
    "unesco ai ethics recommendation": "UNESCO AI Ethics Recommendation",
    "unesco recommendation": "UNESCO AI Ethics Recommendation",
    "oecd ai principles": "OECD AI Principles",
    "oecd ai principle": "OECD AI Principles",
    "oecd principles": "OECD AI Principles",
    "apple inc": "Apple Inc",
    "apple inc.": "Apple Inc",
    "apple inc typo": "Apple Inc",
    "apple incs": "Apple Inc",
    "compas": "COMPAS",
    "ai system": "AI system",
    "ai systems": "AI system",
    "correctional offender management profiling for alternative sanctions": "COMPAS",
    "mimic cxr": "MIMIC-CXR",
    "mimic-cxr": "MIMIC-CXR",
    "mimic cxr dataset": "MIMIC-CXR",
}

# CORE_GENERIC: bare structural/resource/process nouns whose genericity
# makes fuzzy merging unsafe across essentially any subject-matter domain
# (see the same split -- and the same reasoning -- applied to
# utils/canonical_linker.py::DO_NOT_MERGE in the architectural-cleanup
# pass). Domain-specific blocked terms (AI-ethics vocabulary such as
# "bias"/"fairness") no longer live here -- they are supplied by the
# active domain profile's "generic_merge_blocklist" gazetteer instead
# (see domain_profiles.py::AI_ETHICS_PROFILE and
# EntityNormalizer.should_avoid_fuzzy_merge below, which consults BOTH
# this list and the profile's, exactly like
# CanonicalLinker._can_fuzzy_merge).
DO_NOT_MERGE_TERMS = {
    "data",
    "dataset",
    "datasets",
    "model",
    "system",
    "audit",
    "audits",
}


TYPE_ALIASES = {
    "AiSystem": "AISystem",
    "Aisystem": "AISystem",
    "Algorithmicbias": "AlgorithmicBias",
    "Personaldata": "PersonalData",
    "Ai": "AI",
    "Ml": "ML",
    "Organization": "Organisation",
}


def _preserve_acronyms_in_phrase(original: str, titled: str) -> str:
    orig_tokens = re.split(r"(\W+)", original)
    titled_tokens = re.split(r"(\W+)", titled)
    out = []
    for o_tok, t_tok in zip(orig_tokens, titled_tokens):
        if not o_tok or re.fullmatch(r"\W+", o_tok):
            out.append(t_tok)
            continue
        if o_tok.isupper() or o_tok.upper() in KNOWN_ACRONYMS:
            out.append(o_tok.upper())
        else:
            out.append(t_tok)
    return "".join(out) if len(out) == len(titled_tokens) else titled


def _normalize_casing(surface: str) -> str:
    s = surface.strip()
    if is_acronym(s):
        return s.upper()
    tokens = re.split(r"(\W+)", s)
    out = []
    for token in tokens:
        if not token or re.fullmatch(r"\W+", token):
            out.append(token)
            continue
        if token.isupper() or token.upper() in KNOWN_ACRONYMS:
            out.append(token.upper())
            continue
        if re.search(r"[a-z].*[A-Z]|[A-Z].*[a-z].*[A-Z]", token):
            out.append(token)
            continue
        out.append(token[:1].upper() + token[1:].lower())
    normalized = "".join(out)
    return _preserve_acronyms_in_phrase(s, normalized)


def _normalize_entity_type(entity_type: Any) -> str:
    if entity_type is None:
        return "Entity"
    et = str(entity_type).strip()
    if not et:
        return "Entity"
    if ":" in et and " " not in et:
        _, local = et.split(":", 1)
        et = local or et
    et = et.replace("_", " ").replace("-", " ")
    et = " ".join(et.split()).strip()
    norm = et.title().replace(" ", "")
    canonical = canonicalize_class_local(norm)
    return TYPE_ALIASES.get(canonical, canonical)


def _canonical_name(label: str) -> str:
    label = label.strip()
    label = re.sub(r"[^\w]+", "_", label)
    label = re.sub(r"_+", "_", label)
    return label.lower().strip("_")


class EntityNormalizer:
    def __init__(self, registry=None, domain_profile=None):
        self.registry = registry
        # Active domain profile (see domain_profiles.py) -- same
        # resolution pattern as EnhancedSemanticPipeline, RDFSerializer,
        # and CanonicalLinker: an explicit domain_profile= overrides (used
        # by tests exercising a synthetic profile), otherwise it is
        # resolved from Config.DOMAIN_PROFILE, reproducing current
        # AI-ethics behaviour by default.
        if domain_profile is not None:
            self.domain_profile = domain_profile
        else:
            from domain_profiles import get_active_profile

            self.domain_profile = get_active_profile()

    def match_key(self, label: str) -> str:
        return _norm_match_key(label or "")

    def should_avoid_fuzzy_merge(self, label: str) -> bool:
        key = self.match_key(label)
        if key in DO_NOT_MERGE_TERMS:
            return True
        # Domain-specific blocked terms (e.g. AI-ethics vocabulary) come
        # from the active domain profile, not a hardcoded list here --
        # the SAME gazetteer CanonicalLinker._can_fuzzy_merge consults, so
        # the two components always agree on what is blocked and why.
        return self.domain_profile.matches_gazetteer("generic_merge_blocklist", key)

    def normalize_label(self, surface_form: Any) -> str:
        cleaned = _clean_text(surface_form)
        key = self.match_key(cleaned)
        if key in ALIAS_CANONICAL_LABELS:
            return ALIAS_CANONICAL_LABELS[key]
        return _normalize_casing(cleaned)

    def normalize_type(self, entity_type: str) -> str:
        return _normalize_entity_type(entity_type)

    def normalize_predicate(self, pred: str) -> str:
        return canonicalize_predicate(pred)

    def normalize_metadata(self, metadata: dict) -> dict:
        if not metadata:
            return metadata
        cleaned = {}
        for key, val in metadata.items():
            if isinstance(val, str):
                cleaned[key] = _clean_text(val)
            elif isinstance(val, list):
                cleaned[key] = [_clean_text(v) if isinstance(v, str) else v for v in val]
            else:
                cleaned[key] = val
        return cleaned

    def canonical_name(self, label: str) -> str:
        return _canonical_name(label)

    def canonical_uri(
        self,
        *,
        label: str,
        entity_type: str,
        ontology_uri: Optional[str] = None,
        discourse_scope: Optional[str] = None,
    ) -> str:
        """`discourse_scope` (optional): when provided, folds a discourse
        /origin-key scope into both the hash and the slug so the resulting
        URI is unique to that scope rather than shared purely by label+type.
        Used for entities whose identity is NOT yet a resolved, genuine
        antecedent (unresolved referential mentions, unit-scoped
        Propositions) -- omitted entirely for genuine named entities, which
        keeps their URIs byte-for-byte identical to before this parameter
        existed."""
        norm_label = self.normalize_label(label)
        norm_type = self.normalize_type(entity_type or "Entity")
        slug = self.canonical_name(norm_label) or "entity"
        ont = (ontology_uri or "").strip().lower()
        key = f"{self.match_key(norm_label)}|{norm_type}|{ont}"
        scope = str(discourse_scope or "").strip()
        if scope:
            key = f"{key}|scope:{scope.lower()}"
            scope_slug = self.canonical_name(scope) or "scope"
            slug = f"{slug}_{scope_slug}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        return f"{Config.NAMESPACES['etd']}{norm_type.lower()}_{slug}_{h}"

    def normalize_entity(self, entity) -> None:
        if entity is None:
            return

        original_label = getattr(entity, "label", "")
        entity.label = self.normalize_label(original_label)

        raw_type = getattr(entity, "entity_type", None)
        entity.metadata = entity.metadata or {}
        if isinstance(raw_type, str) and ":" in raw_type and " " not in raw_type:
            entity.metadata.setdefault("external_entity_type", raw_type)

        entity.entity_type = self.normalize_type(raw_type)
        entity.name = self.canonical_name(entity.label)

        aliases = entity.metadata.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        orig = str(original_label or "").strip()
        if orig and orig != entity.label and orig not in aliases:
            aliases.append(orig)
        if aliases:
            entity.metadata["aliases"] = aliases

        # Identity-evidence safety: `aliases` is a mixed bag (this
        # pre-normalization label, plus -- added by later stages --
        # coreference/consolidation/linker-derived forms). Maintain a
        # SEPARATE allow-list of aliases that are genuinely observed source
        # text, so a downstream consumer that wants trusted identity
        # evidence (e.g. WikidataLinker source_grounded_aliases) does not
        # have to trust the whole collection. The pre-normalization label
        # qualifies ONLY when it was the real extracted mention, not an
        # LLM/context-inferred `resolved_label`.
        md = entity.metadata
        is_inferred_label = bool(md.get("mention_resolved_from_context")) and (
            orig.strip().lower() == str(md.get("resolved_label") or "").strip().lower()
        )
        if orig and orig != entity.label and not is_inferred_label:
            sga = md.get("source_grounded_aliases")
            if not isinstance(sga, list):
                sga = []
            if orig not in sga:
                sga.append(orig)
            md["source_grounded_aliases"] = sga

        entity.metadata = self.normalize_metadata(entity.metadata)
