"""
Small domain-profile / configuration layer.

This exists so that generic pipeline/serializer code (pipeline.py,
serializers/rdf_serializer.py) can ask for domain-specific vocabulary
through one small, uniform API instead of hardcoding a particular subject
domain's ontology class names or lexical word-lists directly. It is
deliberately NOT a plugin framework: one dataclass, one plain-dict
registry, one file.

A DomainProfile supplies, by GENERIC category name (never a domain-specific
name -- generic code never mentions "AlgorithmicBias" or "defendants"):

- category_signals:       regex hints for a semantic category, e.g.
                           "population_group" -> a compiled regex.
- category_types:         ontology/entity-type names that satisfy a
                           category outright -- checked BEFORE any lexical
                           hint (see priority order below).
- gazetteers:             closed name lists (e.g. known hosting platforms).
- predicate_bindings:     concrete predicate names for a generic relation
                           concept, e.g. "hosting_relation" -> "hostedBy".
- class_correction_rules: data for the generic "a loaded/specialized
                           ontology class requires its own textual
                           evidence; otherwise fall back to a configured,
                           less-specific class" mechanism -- see
                           serializers/rdf_serializer.py::_corrected_entity_class,
                           which is a pure rule-interpreter and contains no
                           domain class names itself.

Priority order used by every category lookup (see pipeline.py::
_matches_semantic_category): ontology/entity type > profile category
signal/gazetteer hint > no assumption. A category with no ontology-type
mapping and no lexical signal simply never matches -- "no assumption" is
always the safe default, never a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Set, Tuple


@dataclass(frozen=True)
class ClassCorrectionRule:
    """One rule for the generic "loaded/specialized class requires
    semantic evidence; otherwise use a configured fallback" mechanism.

    Exactly one of `required_signal` / `disqualifying_signal` is normally
    set on a given rule:

    - `required_signal` present but NOT matched in the label -> the
      declared class is unsupported by its own text; use
      `secondary_fallback_class` if `secondary_signal` matches, else
      `fallback_class`.
    - `disqualifying_signal` present AND matched in the label -> the
      declared class is wrong regardless of anything else; use
      `fallback_class` directly.
    """

    declared_class: str
    fallback_class: str
    required_signal: Optional[Pattern] = None
    disqualifying_signal: Optional[Pattern] = None
    secondary_signal: Optional[Pattern] = None
    secondary_fallback_class: Optional[str] = None


@dataclass(frozen=True)
class TypeAssignmentRule:
    """One rule for the generic POSITIVE keyword-triggered typing
    mechanism used by ontology/semantic_typing.py::SemanticTyper.classify
    -- when `trigger_signal` matches a candidate entity's label, every
    class named in `classes_to_add` is proposed (each still subject to
    SemanticTyper's own, unchanged ontology-compatibility gate). Set
    `exact=True` when the label must equal one of the trigger's
    alternatives ENTIRELY (e.g. "Students" -> Stakeholder/PopulationGroup)
    rather than merely contain one as a substring (e.g. any label
    mentioning "bias"/"harm"/"risk" -> Bias/Risk/Harm/Impact)."""

    trigger_signal: Pattern
    classes_to_add: Tuple[str, ...]
    exact: bool = False


@dataclass(frozen=True)
class DomainProfile:
    name: str
    category_signals: Dict[str, Pattern] = field(default_factory=dict)
    category_types: Dict[str, Set[str]] = field(default_factory=dict)
    gazetteers: Dict[str, Set[str]] = field(default_factory=dict)
    predicate_bindings: Dict[str, str] = field(default_factory=dict)
    class_correction_rules: List[ClassCorrectionRule] = field(default_factory=list)
    type_assignment_rules: List[TypeAssignmentRule] = field(default_factory=list)

    # ---- Surface-entity type inference (pipeline.py::_infer_surface_entity_type
    # / _surface_entity_bucket) ----
    # bucket name -> head words that belong to it (the LAST token of a
    # candidate surface phrase). Iteration order is priority order.
    surface_type_buckets: Dict[str, Set[str]] = field(default_factory=dict)
    # a specific head word -> the PRECISE type it should produce, overriding
    # its bucket's default (e.g. bucket "system" defaults to one type, but
    # the head word "model" within it produces a different, more specific one).
    surface_type_overrides: Dict[str, str] = field(default_factory=dict)
    # bucket name -> the type produced when no head-word override applies.
    surface_type_bucket_defaults: Dict[str, str] = field(default_factory=dict)
    # bucket name -> an additional whole-phrase pattern that also belongs to
    # that bucket, independent of the head word (e.g. a specific known
    # compound phrase).
    surface_type_phrase_signals: Dict[str, Pattern] = field(default_factory=dict)

    # ---- Surface-trim prefix patterns (pipeline.py::_trim_recovered_surface) ----
    # bucket name -> prefix patterns to strip from a recovered surface phrase
    # before the generic (CORE) trimming logic runs.
    surface_trim_prefix_patterns: Dict[str, List[Pattern]] = field(default_factory=dict)

    # ---- Lexical entity-recovery typing (semantic surface-entity recovery pass) ----
    # trigger phrase (lowercase) -> the entity type to mint when that phrase
    # is found in raw discourse text.
    lexical_recovery_terms: Dict[str, str] = field(default_factory=dict)

    # ---- Shared entity-semantic-family classifier ----
    # family name -> entity_type strings (lowercase) that belong to it.
    # Iteration order is priority order; entity_type strings must be
    # disjoint across families.
    entity_family_types: Dict[str, Set[str]] = field(default_factory=dict)
    # entity_type strings exempt from the "section_category == resource"
    # structural rule (typically the union of the dataset and
    # policy/framework families' own entity_type strings).
    entity_family_resource_exempt_types: Set[str] = field(default_factory=set)

    def matches_category_signal(self, category: str, text: str) -> bool:
        pattern = self.category_signals.get(category)
        if pattern is None:
            return False
        return bool(pattern.search(str(text or "")))

    def matches_category_signal_exact(self, category: str, text: str) -> bool:
        """Whole-label variant of matches_category_signal: true only when
        the ENTIRE (whitespace-normalized) text is one of the category's
        vocabulary alternatives, not merely a substring of a longer, more
        specific label (e.g. "bias" alone matches the "abstract_finding"
        category exactly; "Intersectional Bias" does not, even though it
        CONTAINS "bias" and would match matches_category_signal). Reuses
        the exact same compiled pattern as the substring form -- there is
        only one copy of each category's vocabulary -- via `fullmatch`
        instead of `search`."""
        pattern = self.category_signals.get(category)
        if pattern is None:
            return False
        normalized = " ".join(str(text or "").strip().split())
        if not normalized:
            return False
        return bool(pattern.fullmatch(normalized))

    def matches_gazetteer(self, gazetteer: str, value: str) -> bool:
        names = self.gazetteers.get(gazetteer)
        if not names:
            return False
        return str(value or "").strip().lower() in names

    def gazetteer(self, name: str) -> Set[str]:
        """Bulk accessor (as opposed to matches_gazetteer's single-value
        membership check) for callers that need to union a named
        gazetteer into a larger CORE set, e.g. a CORE stopword list that
        also needs the active profile's own domain-specific additions."""
        return self.gazetteers.get(name, set())

    def category_type_names(self, category: str) -> Set[str]:
        return self.category_types.get(category, set())

    def predicate_for(self, key: str, default: str = "") -> str:
        return self.predicate_bindings.get(key, default)


def matches_semantic_category(
    profile: "DomainProfile",
    category: str,
    text: str,
    *,
    entity_type: Optional[str] = None,
    exact: bool = False,
) -> bool:
    """The ONE shared, domain-agnostic category-membership check used by
    every CORE mechanism that needs to know "is this a population group /
    an analytical dimension / an abstract finding / etc." -- e.g.
    pipeline.py's comparative-construction, redundant-demonstration, and
    predicate domain/range-validity guards, and
    utils/semantic_relation_policy.py's parallel domain/range-validity
    guard. Callers must not maintain their own copy of this logic or of
    any category's vocabulary.

    Priority order:
      1. ontology/entity type -- `entity_type` is one of the active
         profile's configured type names for `category` (generic
         upper-ontology-level names, e.g. PopulationGroup/Person -- never
         a domain-specific class name coded by a caller);
      2. the profile's lexical/gazetteer signal for `category`;
      3. no assumption (False).

    `category` is always a generic string ("population_group",
    "analytical_dimension", "abstract_finding", ...); the concrete
    vocabulary that satisfies it lives entirely in the active
    DomainProfile.

    `exact=True` requires the WHOLE text to be one of the category's
    vocabulary alternatives (see matches_category_signal_exact) rather
    than merely containing one as a substring -- needed by callers asking
    "is this bare label itself too generic" (e.g. a domain/range validity
    check rejecting a target labeled literally "Bias") as opposed to
    "does this phrase mention the category at all" (e.g. scanning
    "systematic bias" for a finding word). Both modes read the exact same
    per-category vocabulary; only the match strategy differs.
    """
    if entity_type:
        type_names = profile.category_type_names(category)
        if type_names and entity_type in type_names:
            return True
    if exact:
        return profile.matches_category_signal_exact(category, text)
    return profile.matches_category_signal(category, text)


def classify_entity_semantic_family(entity: object, *, profile: "DomainProfile") -> str:
    """The ONE shared entity-semantic-family classifier used by both
    pipeline.py's and utils/semantic_relation_policy.py's predicate
    domain/range-validity guards -- both call this exact function so they
    cannot disagree about what family an entity belongs to. Contains no
    domain-specific type or class names itself.

    Deliberately preserves the exact interleaved priority order the two
    formerly-independent copies of this function already agreed on,
    rather than a strict "all ontology types, then all lexical signals"
    ordering: a few label-based fallbacks (decision-making phrasing,
    population-group cue) are checked BETWEEN certain type-based buckets,
    not only after all of them. This matters in practice -- e.g. an
    entity labeled "Algorithmic Decision-Making" but mistyped
    "AlgorithmicBias" by the extractor must still classify by its label
    as a process/concept, not as a risk finding, for domain/range
    purposes; a strict type-first ordering silently breaks this
    (confirmed by an existing regression test). Ontology/entity type
    still outranks the two lowest-priority, purely-lexical checks
    (population-group cue as a last resort before falling to "risk", and
    the generic risk/finding wording check as the very last resort before
    the universal "concept" fallback) -- see the ordering below.

    `category`/vocabulary lookups (population-group cue, generic
    risk/finding wording) are entirely profile-driven via
    matches_semantic_category; only the "does the label mention
    decision-making phrasing" check is a CORE literal (generic English
    business/governance vocabulary, not a subject-matter category -- see
    the architectural-cleanup report for why this is not treated as
    domain-specific hardcoding).
    """
    if entity is None:
        return "unknown"
    entity_type = str(getattr(entity, "entity_type", "") or "").strip().lower()
    label = str(getattr(entity, "label", "") or "").strip().lower()
    section_category = str((getattr(entity, "metadata", None) or {}).get("section_category") or "").strip().lower()
    types = profile.entity_family_types

    if entity_type in types.get("discourse", set()):
        return "discourse"
    if entity_type in types.get("vote", set()):
        return "vote"
    if section_category == "resource" and entity_type not in profile.entity_family_resource_exempt_types:
        return "resource"
    if entity_type in types.get("dataset", set()):
        return "dataset"
    if entity_type in types.get("domain", set()):
        return "domain"
    if "decision-making" in label or "decision making" in label:
        return "concept"

    # Generic pass over every OTHER family the active profile configures,
    # in the profile's own declared order -- unlike the four families
    # above (which need a fixed relative position for the AI-ethics
    # profile's own historical behaviour, confirmed by regression tests),
    # every other family name is entirely up to the profile: a synthetic
    # profile can declare families with names never seen here (e.g.
    # "litigant_family") and they participate correctly with no change to
    # this function.
    for family_name, type_set in types.items():
        if family_name in ("discourse", "vote", "dataset", "domain"):
            continue
        if entity_type in type_set:
            return family_name

    # Lexical fallbacks -- lowest priority, after every configured type
    # has had a chance to match.
    if matches_semantic_category(profile, "population_group", label):
        return "population_or_group"
    if matches_semantic_category(profile, "risk_or_harm_family_fallback", label):
        return "risk_or_harm"
    return "concept"


# CORE_GENERIC, domain-independent: bare structural/resource-category
# nouns with no specific referent of their own in ANY subject-matter
# domain -- "a dataset" or "an audit" named without qualification is too
# generic to be a meaningful domain-subject or finding-context label
# regardless of whether the deployment is AI-ethics, legal, or medical.
# Consulted alongside (not instead of) a DomainProfile's "abstract_finding"
# category signal by pipeline.py/utils/semantic_relation_policy.py's
# generic-label guards. This is the exact CORE_GENERIC remainder of what
# used to be pipeline.py::_is_generic_abstract_label's own hardcoded word
# list before this cleanup pass -- kept here, not in a DomainProfile,
# because it is not domain-specific data.
CORE_GENERIC_FINDING_CONTEXT_NOUNS = frozenset({"dataset", "datasets", "audit", "audits"})


_PROFILE_REGISTRY: Dict[str, "DomainProfile"] = {}


def register_profile(profile: "DomainProfile") -> None:
    """Registers (or replaces) a profile under its own `.name`."""
    _PROFILE_REGISTRY[profile.name] = profile


def get_profile(name: str) -> "DomainProfile":
    try:
        return _PROFILE_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown domain profile: {name!r}. Registered profiles: {sorted(_PROFILE_REGISTRY)}"
        ) from None


def get_active_profile() -> "DomainProfile":
    """Resolves Config.DOMAIN_PROFILE lazily -- avoids a hard import-time
    dependency on config.py for callers (e.g. tests) that only need
    register_profile()/get_profile() with an explicit profile object."""
    from config import Config

    return get_profile(getattr(Config, "DOMAIN_PROFILE", "ai_ethics"))


def registered_profile_names() -> List[str]:
    return sorted(_PROFILE_REGISTRY)


# =============================================================================
# AI-ethics domain profile.
#
# Reproduces this project's current semantic-quality behaviour exactly --
# every value below was, before this architectural-cleanup pass, a
# hardcoded constant directly inside pipeline.py or
# serializers/rdf_serializer.py. Nothing here changes what the pipeline
# does for the current AI-ethics dataset; it only relocates the data so
# generic code no longer needs to know it.
# =============================================================================

_AI_ETHICS_POPULATION_GROUP_SIGNAL = re.compile(
    r"\b(students|women|men|people|individuals|group|groups|community|communities|"
    r"patients|defendants|borrowers|applicants|users|customers|children|"
    r"minorit\w*|marginali[sz]ed|"
    # Merged in from the former utils/semantic_relation_policy.py::
    # _label_has_group_cue duplicate (architectural-cleanup pass) --
    # racial/ethnic identity terms and additional group nouns that
    # function had but this signal previously did not.
    r"black|latino|latina|latinx|white|asian|hispanic|indigenous|"
    r"neighborhoods?|neighbourhoods?|workers|residents|businesses|"
    r"populations|families|migrants)\b",
    re.I,
)
_AI_ETHICS_ANALYTICAL_DIMENSION_SIGNAL = re.compile(
    r"^(race|sex|gender|age|ethnicity|income|disability|region|"
    r"insurance type|insurance status|zip code|marital status)$",
    re.I,
)
_AI_ETHICS_ABSTRACT_FINDING_SIGNAL = re.compile(
    r"\b(bias|biases|disparity|disparities|discrimination|inequity|inequities|unfairness|prejudice|"
    # Merged in from the former pipeline.py::_is_generic_risk_label and
    # pipeline.py/utils/semantic_relation_policy.py::_is_generic_abstract_label
    # duplicates (architectural-cleanup pass): risk/harm/impact vocabulary
    # and the AI-ethics EthicalPrinciple-branch terms those two functions
    # each separately hardcoded.
    r"risk|risks|harm|harms|impact|impacts|"
    r"fair|fairness|transparency|accountability|interpretability|equity)\b",
    re.I,
)
# Distinctly narrower than "abstract_finding" above -- this is
# specifically pipeline.py::_entity_semantic_family / utils/
# semantic_relation_policy.py::entity_family's own former inline
# risk-word fallback (final cross-domain cleanup pass), which never
# included the fairness/transparency/accountability cluster (an entity
# labeled bare "Accountability" must classify as "concept", not
# "risk_or_harm" -- confirmed by an existing regression test). Kept as
# its own category rather than reusing "abstract_finding" because the two
# were always different, independently-scoped vocabularies before this
# cleanup pass, even though they overlap on bias/harm/impact/discrimination.
_AI_ETHICS_RISK_OR_HARM_FAMILY_FALLBACK_SIGNAL = re.compile(
    r"\b(risk|bias|harm|impact|discrimination|redlining|vulnerability)\b",
    re.I,
)
_AI_ETHICS_BIAS_TYPE_SIGNAL = re.compile(
    r"\b(bias|biases|biased|discriminat\w*|disparit\w*|disparate|prejudic\w*|unfair\w*|inequit\w*|inequal\w*)\b",
    re.I,
)
_AI_ETHICS_PROCESS_TYPE_SIGNAL = re.compile(
    r"\b(decision|decisions|decision-making|process|processes|pipeline|procedure|workflow|"
    r"algorithm|algorithms|method|methodology|methodologies)\b",
    re.I,
)
_AI_ETHICS_DOCUMENTATION_ARTIFACT_SIGNAL = re.compile(
    r"\b(model\s*card|model\s*cards|datasheet|data\s*sheet)\b",
    re.I,
)

# ---- ontology/semantic_typing.py::SemanticTyper.classify() rules -------
# Relocated from that module's own hardcoded keyword lists
# (architectural-cleanup pass, item 4). The generic dispatch mechanism
# (iterate type_assignment_rules, apply each via the existing
# ontology-compatibility gate) stays in semantic_typing.py; only the
# AI-ethics vocabulary and which classes it triggers move here.
_AI_ETHICS_RISK_HARM_IMPACT_TRIGGER = re.compile(
    r"bias|harm|disparate|inequ|inequal|unfair|risk|discrimination|justice|impact",
    re.I,
)
_AI_ETHICS_SYSTEM_COMPONENT_TRIGGER = re.compile(
    r"model|algorithm|system|classifier|framework|pipeline|\bai\b|\bml\b",
    re.I,
)
_AI_ETHICS_ETHICAL_PRINCIPLE_TRIGGER = re.compile(
    r"transparency|explainability|interpretability|fairness|equity|equal|non-discrimination|"
    r"accountability|responsibility|liability|privacy|confidentiality|data protection|"
    r"security|robustness|reliability|safety|trust|trustworthy|human oversight|"
    r"autonomy|dignity|human rights|justice|inclusion|inclusiveness|"
    r"well-being|beneficence|non-maleficence|sustainability|environmental|green|"
    r"bias mitigation|harm mitigation|risk mitigation|ethical|ethics|principle|value",
    re.I,
)
# Whole-label-only (see TypeAssignmentRule.exact) -- "the entire label
# equals a stakeholder term", preserving the original guard against false
# positives like "racial injustice against black people".
_AI_ETHICS_GENERAL_POPULATION_GROUP_EXACT = re.compile(
    r"\b(group|community|population|students|learners|children|boys|girls|"
    r"immigrants|migrants|patients|borrowers|defendants|applicants|"
    r"customers|users|consumers|minority|minorities)\b",
    re.I,
)
_AI_ETHICS_PROTECTED_GROUP_EXACT = re.compile(
    r"\b(black|blacks|white|whites|asian|asians|latino|latinos|hispanic|hispanics|"
    r"indigenous|indigenous peoples|women|men|lgbt|lgbtq|lgbtq\+|queer|"
    r"trans|transgender|disabled|persons with disabilities|"
    r"muslim|christian|jewish)\b",
    re.I,
)

# ---- pipeline.py::_surface_entity_head_buckets / _infer_surface_entity_type
# / _surface_entity_bucket (final cross-domain cleanup pass) -------------
# Bucket order is priority order (first containing the head word wins).
_AI_ETHICS_SURFACE_TYPE_BUCKETS: Dict[str, Set[str]] = {
    "group": {
        "women", "men", "patients", "patient", "borrowers", "borrower",
        "students", "student", "communities", "community", "populations",
        "population", "businesses", "business", "people", "groups", "group",
        "residents", "resident", "neighborhoods", "neighbourhoods",
        "neighborhood", "neighbourhood", "graduates", "areas",
    },
    "dataset": {
        "data", "dataset", "datasets", "indicators", "indicator", "metrics", "metric",
    },
    "risk": {
        "bias", "biases", "disparities", "disparity", "discrimination",
        "impact", "harm", "harms", "risk", "risks", "misdiagnosis",
    },
    "concept": {
        "spending", "needs", "accuracy", "treatment", "care", "autonomy",
        "opportunities", "performance", "outcomes",
    },
    "system": {
        "ai", "system", "systems", "algorithm", "algorithms", "tool", "tools",
        "model", "models", "prediction", "predictions",
    },
}
_AI_ETHICS_SURFACE_TYPE_OVERRIDES: Dict[str, str] = {
    # Within the "risk" bucket, these specific head words are precise
    # enough to warrant AlgorithmicBias rather than the bucket's generic
    # "Impact" default.
    "bias": "AlgorithmicBias", "biases": "AlgorithmicBias",
    "discrimination": "AlgorithmicBias", "disparities": "AlgorithmicBias",
    "disparity": "AlgorithmicBias",
    # Within the "system" bucket, "model"/"models" is the underlying
    # computational artefact rather than the deployed system.
    "model": "Model", "models": "Model",
}
_AI_ETHICS_SURFACE_TYPE_BUCKET_DEFAULTS: Dict[str, str] = {
    "group": "DemographicGroup",
    "dataset": "Dataset",
    "risk": "Impact",
    "concept": "Concept",
    "system": "AISystem",
}
_AI_ETHICS_SURFACE_TYPE_PHRASE_SIGNALS: Dict[str, Pattern] = {
    "dataset": re.compile(r"historical data|workforce data", re.I),
}

# ---- pipeline.py::_trim_recovered_surface (final cross-domain cleanup pass) ----
_AI_ETHICS_SURFACE_TRIM_PREFIX_PATTERNS: Dict[str, List[Pattern]] = {
    "system": [
        re.compile(
            r"^(?:racial|systematic|systemic|intersectional|demographic|measurement|clinical|proxy)\s+"
            r"(?:bias|discrimination|disparities)\s+in\s+",
            re.I,
        ),
    ],
}

# ---- Lexical entity-recovery typing (final cross-domain cleanup pass) --
_AI_ETHICS_LEXICAL_RECOVERY_TERMS: Dict[str, str] = {
    "accountability": "Governance",
    "interpretability": "Transparency",
    "equity": "Fairness",
    "due process": "Governance",
    "historical workforce data": "Dataset",
    "systematic discrimination": "Bias",
    "harsher sentencing": "Bias",
}

# ---- Shared entity-semantic-family classifier (final cross-domain
# cleanup pass) -- reconciles the two independently-evolved taxonomies
# formerly hardcoded in pipeline.py::_entity_semantic_family and
# utils/semantic_relation_policy.py::entity_family. Every distinction
# either function used to make is preserved as its own family here (no
# family is a lossy merge of two formerly-distinct ones); the two
# families that come strictly from this reconciliation --
# "paper_or_analysis" split out of pipeline's old coarser "project"
# bucket, and "classification" kept distinct rather than folded into
# policy's old "concept" bucket -- are noted in the cleanup report.
_AI_ETHICS_ENTITY_FAMILY_TYPES: Dict[str, Set[str]] = {
    "discourse": {"discourseunit", "theme", "question", "answer", "document"},
    "vote": {"vote"},
    "dataset": {"dataset", "datasource", "corpus"},
    "domain": {
        "domain", "criminaljustice", "justicedomain", "employment",
        "healthcare", "education", "highrisk", "highriskai",
    },
    "person": {"person", "expert"},
    "organisation": {"organisation", "organization", "company", "institution"},
    "project": {"project", "researchproject"},
    "paper_or_analysis": {"study", "paper", "researchpaper", "publication", "source", "analysis"},
    "classification": {"classification", "metric"},
    "policy_or_framework": {
        "policyframework", "law", "regulation", "standard",
        "legalrequirement", "regulatoryreference",
    },
    "population_or_group": {
        "demographic", "demographicgroup", "populationgroup", "protectedgroup", "community",
    },
    "risk_or_harm": {
        "impact", "harm", "biasrisk", "bias", "racialbias",
        "racialbiasrisk", "genderbias", "algorithmicbias", "risk",
    },
    "system_or_model": {
        "aisystem", "systemcomponent", "model", "tool", "application", "technology",
    },
    "algorithm_or_process": {"algorithm", "process"},
}
_AI_ETHICS_ENTITY_FAMILY_RESOURCE_EXEMPT_TYPES: Set[str] = (
    _AI_ETHICS_ENTITY_FAMILY_TYPES["dataset"] | _AI_ETHICS_ENTITY_FAMILY_TYPES["policy_or_framework"]
)

AI_ETHICS_PROFILE = DomainProfile(
    name="ai_ethics",
    category_signals={
        "population_group": _AI_ETHICS_POPULATION_GROUP_SIGNAL,
        "analytical_dimension": _AI_ETHICS_ANALYTICAL_DIMENSION_SIGNAL,
        "abstract_finding": _AI_ETHICS_ABSTRACT_FINDING_SIGNAL,
        "risk_or_harm_family_fallback": _AI_ETHICS_RISK_OR_HARM_FAMILY_FALLBACK_SIGNAL,
    },
    category_types={
        # Generic upper-ontology-level type names (already declared in
        # ontology/add_schema.py) that satisfy "population_group" outright,
        # ahead of any lexical hint -- see the priority order above.
        "population_group": {"PopulationGroup", "DemographicGroup", "Person"},
    },
    gazetteers={
        "hosting_platform_names": {
            "physionet", "github", "gitlab", "kaggle", "huggingface",
            "hugging face", "zenodo", "arxiv", "openml", "papers with code",
            "data.gov", "figshare", "dryad",
        },
        # DOMAIN_PROFILE half of utils/canonical_linker.py's former
        # DO_NOT_MERGE list (see the architectural-cleanup pass that split
        # it): AI-ethics-specific terms whose fuzzy-merge protection is
        # specific to this subject-matter domain, not cross-domain generic.
        # "ai"/"ml" are AI-domain acronyms (also already blocked by the
        # separate short-token length check, kept here as defense in
        # depth); "fair"/"fairness"/"transparency"/"accountability"/
        # "equity" mirror this project's own EthicalPrinciple ontology
        # classes; "interpretability" is an ML-specific technical term;
        # "bias"/"biases"/"discrimination" mirror the Bias ontology
        # branch. The CORE_GENERIC remainder (data/dataset(s)/model/
        # system/audit(s)) stays in canonical_linker.py::DO_NOT_MERGE
        # because those nouns are overloaded in essentially any domain.
        "generic_merge_blocklist": {
            "ai", "ml",
            "fair", "fairness", "transparency", "accountability",
            "interpretability", "equity",
            "bias", "biases", "discrimination",
        },
        # DOMAIN_PROFILE half of two independent, formerly-hardcoded word
        # lists sharing this exact vocabulary (final cross-domain cleanup
        # pass): pipeline.py::_generic_surface_label ("this bare label is
        # too generic to prefer as a canonical surface entity") and both
        # pipeline.py::_entity_surface_variants and
        # utils/semantic_relation_policy.py::entity_surface_variants
        # ("this derived alias variant is too generic to treat as
        # distinctive for canonicalization/evidence-matching purposes").
        # AI-domain vocabulary ("ai"/"algorithm(s)") and AI-ethics finding
        # vocabulary ("bias"/"risk"/"harm"/"impact"/"discrimination")
        # whose genericity is domain-dependent; the CORE_GENERIC remainder
        # (system(s)/model(s)/tool(s)/data/dataset(s)/project/study/
        # paper/report/analysis) stays hardcoded in both files.
        "generic_surface_terms": {
            "ai", "algorithm", "algorithms",
            "bias", "risk", "harm", "impact", "discrimination",
        },
        # Entity types exempt from the generic-surface-label rejection in
        # _surface_entity_candidates_for_sentence -- these are exactly the
        # types this profile's own surface-type-inference "system" bucket
        # can produce (see surface_type_bucket_defaults/surface_type_overrides
        # below), for which even a single generic word ("AI") is still a
        # legitimate standalone entity.
        "generic_label_exempt_types": {"AISystem", "Model"},
        # DOMAIN_PROFILE half of pipeline.py's former
        # _surface_entity_boundary_stops word list (final cross-domain
        # cleanup pass): the vast majority of that list is generic English
        # function/connector words (articles, pronouns, auxiliary verbs,
        # discourse connectors) and stays hardcoded as CORE; only these
        # AI-ethics finding words -- used to stop a recovered surface span
        # from bleeding across a bias/impact-word boundary -- are
        # domain-specific.
        "surface_boundary_stop_terms": {
            "bias", "biases", "discrimination", "disparity", "disparities", "impact",
        },
    },
    predicate_bindings={
        "hosting_relation": "hostedBy",
    },
    surface_type_buckets=_AI_ETHICS_SURFACE_TYPE_BUCKETS,
    surface_type_overrides=_AI_ETHICS_SURFACE_TYPE_OVERRIDES,
    surface_type_bucket_defaults=_AI_ETHICS_SURFACE_TYPE_BUCKET_DEFAULTS,
    surface_type_phrase_signals=_AI_ETHICS_SURFACE_TYPE_PHRASE_SIGNALS,
    surface_trim_prefix_patterns=_AI_ETHICS_SURFACE_TRIM_PREFIX_PATTERNS,
    lexical_recovery_terms=_AI_ETHICS_LEXICAL_RECOVERY_TERMS,
    entity_family_types=_AI_ETHICS_ENTITY_FAMILY_TYPES,
    entity_family_resource_exempt_types=_AI_ETHICS_ENTITY_FAMILY_RESOURCE_EXEMPT_TYPES,
    class_correction_rules=[
        ClassCorrectionRule(
            declared_class="AlgorithmicBias",
            required_signal=_AI_ETHICS_BIAS_TYPE_SIGNAL,
            secondary_signal=_AI_ETHICS_PROCESS_TYPE_SIGNAL,
            secondary_fallback_class="AlgorithmicProcess",
            fallback_class="Concept",
        ),
        ClassCorrectionRule(
            declared_class="AISystem",
            disqualifying_signal=_AI_ETHICS_DOCUMENTATION_ARTIFACT_SIGNAL,
            fallback_class="Resource",
        ),
    ],
    type_assignment_rules=[
        TypeAssignmentRule(trigger_signal=_AI_ETHICS_RISK_HARM_IMPACT_TRIGGER, classes_to_add=("Bias", "Risk", "Harm", "Impact")),
        # Must run BEFORE the System/Component rule below: asserting
        # Resource first lets the ontology-compatibility gate in
        # SemanticTyper._add_type block the later, incompatible AISystem
        # proposal for the same "model card"/"datasheet" label -- this is
        # the SAME compiled pattern (not a duplicate copy) used by the
        # AISystem ClassCorrectionRule's disqualifying_signal above; this
        # rule is the label-based (declared-type-independent) equivalent
        # of that URI-provenance-based protection.
        TypeAssignmentRule(trigger_signal=_AI_ETHICS_DOCUMENTATION_ARTIFACT_SIGNAL, classes_to_add=("Resource",)),
        TypeAssignmentRule(trigger_signal=_AI_ETHICS_SYSTEM_COMPONENT_TRIGGER, classes_to_add=("AISystem", "SystemComponent")),
        TypeAssignmentRule(trigger_signal=_AI_ETHICS_ETHICAL_PRINCIPLE_TRIGGER, classes_to_add=("EthicalPrinciple",)),
        TypeAssignmentRule(trigger_signal=_AI_ETHICS_GENERAL_POPULATION_GROUP_EXACT, classes_to_add=("Stakeholder", "PopulationGroup"), exact=True),
        TypeAssignmentRule(trigger_signal=_AI_ETHICS_PROTECTED_GROUP_EXACT, classes_to_add=("Stakeholder", "PopulationGroup", "ProtectedGroup"), exact=True),
    ],
)

register_profile(AI_ETHICS_PROFILE)
