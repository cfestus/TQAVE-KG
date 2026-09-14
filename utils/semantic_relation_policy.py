from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config import Config
from models import Entity, Relation
from utils.helpers import normalize_text
from utils.predicate_map import normalize_predicate
from domain_profiles import matches_semantic_category, CORE_GENERIC_FINDING_CONTEXT_NOUNS, classify_entity_semantic_family


SEMANTIC_RELATION_FIELDS: Set[str] = set(getattr(Config, "SEMANTIC_RELATION_FIELDS", set()) or set())


@dataclass(frozen=True)
class PredicateRegistryEntry:
    predicate: str
    predicate_uri: str
    canonical_label: str
    aliases: Tuple[str, ...]
    lexical_triggers: Tuple[str, ...]
    inverse: Optional[str]
    domain_families: Tuple[str, ...]
    range_families: Tuple[str, ...]
    specificity_rank: int
    generic_fallback: bool = False
    structural_only: bool = False
    # range_families alone only ever describes ENTITY-valued
    # objects (an RDF-literal target has no entity_family beyond the
    # catch-all "unknown"/None, so it can never satisfy a specific
    # range_families list without an "any" wildcard). These two fields
    # let a predicate ALSO -- independently -- declare that it accepts a
    # literal-valued object, and optionally constrain which general
    # literal-datatype family that literal must belong to. A predicate
    # with literal_allowed=False (the default) is entity-object-only,
    # exactly the pre-existing behaviour for every predicate that does not
    # explicitly opt in below.
    literal_allowed: bool = False
    # One or more of: "any", "string", "numeric", "boolean", "date", "uri".
    literal_datatypes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "predicate_uri": self.predicate_uri,
            "canonical_label": self.canonical_label,
            "aliases": list(self.aliases),
            "lexical_triggers": list(self.lexical_triggers),
            "inverse": self.inverse or "",
            "domain_family": list(self.domain_families),
            "range_family": list(self.range_families),
            "specificity_rank": int(self.specificity_rank),
            "generic_fallback": bool(self.generic_fallback),
            "structural_only": bool(self.structural_only),
            "literal_allowed": bool(self.literal_allowed),
            "literal_datatypes": list(self.literal_datatypes),
        }


# General, reusable RDF/XSD-datatype-family classifier -- maps
# the datatype/object_type vocabulary already established elsewhere in this
# pipeline (the quantity/literal classification work: xsd:integer, xsd:decimal,
# percentage, money, count, date, etc., plus the LLM relation extractor's
# own object_type values) into a small set of GENERAL literal families. Not
# corpus-specific -- the same mapping applies to any future dataset/domain.
_LITERAL_DATATYPE_FAMILY: Dict[str, str] = {
    "string": "string", "xsd:string": "string", "text": "string",
    "integer": "numeric", "xsd:integer": "numeric", "count": "numeric",
    "decimal": "numeric", "xsd:decimal": "numeric", "float": "numeric",
    "xsd:float": "numeric", "double": "numeric", "number": "numeric",
    "quantity": "numeric", "percentage": "numeric", "money": "numeric",
    "comparison": "numeric",
    "boolean": "boolean", "xsd:boolean": "boolean",
    "date": "date", "xsd:date": "date", "datetime": "date",
    "xsd:datetime": "date", "temporal_range": "date",
    "uri": "uri", "url": "uri", "xsd:anyuri": "uri",
}


def literal_datatype_family(object_type: Optional[str], target_datatype: Optional[str]) -> str:
    """Classify a literal relation-object into a general datatype family
    ("string"/"numeric"/"boolean"/"date"/"uri"), defaulting to "string"
    when neither signal is informative (a bare descriptive literal is the
    most common, least-constrained case)."""
    for raw in (target_datatype, object_type):
        key = str(raw or "").strip().lower()
        if key in _LITERAL_DATATYPE_FAMILY:
            return _LITERAL_DATATYPE_FAMILY[key]
    return "string"


def literal_datatype_permitted(entry: "PredicateRegistryEntry", object_type: Optional[str], target_datatype: Optional[str]) -> bool:
    """True if this predicate accepts a literal object at all, AND (when
    it constrains which literal families it accepts) the relation's actual
    literal datatype/object_type falls within that allowed set."""
    if not entry.literal_allowed:
        return False
    if not entry.literal_datatypes or "any" in entry.literal_datatypes:
        return True
    return literal_datatype_family(object_type, target_datatype) in entry.literal_datatypes


@dataclass
class SemanticValidationDecision:
    accepted: bool
    reason_code: str
    predicate: str
    predicate_surface_phrase: str = ""
    predicate_source: str = ""
    locality_scope: str = ""
    direction: str = "forward"
    discourse_mode: str = ""
    epistemic_status: str = ""
    modality_tags: Tuple[str, ...] = field(default_factory=tuple)
    relation_origin: str = ""
    subject_family: str = ""
    object_family: str = ""
    entity_role_check: str = ""
    domain_range_check: str = ""
    entailment_check: str = ""
    modality_check: str = ""
    reason_detail: str = ""

    def reason_string(self) -> str:
        if self.reason_detail:
            return self.reason_detail
        parts = [
            f"policy={self.reason_code}",
            f"predicate={self.predicate or 'unknown'}",
            f"predicate_source={self.predicate_source or 'unknown'}",
            f"locality={self.locality_scope or 'unknown'}",
            f"entity_role_check={self.entity_role_check or 'unknown'}",
            f"domain_range={self.domain_range_check or 'unknown'}",
            f"entailment={self.entailment_check or 'unknown'}",
            f"modality={self.modality_check or 'unknown'}",
        ]
        return "; ".join(parts)


def _entry(
    predicate: str,
    *,
    label: str,
    aliases: Sequence[str],
    triggers: Sequence[str],
    inverse: Optional[str],
    domain: Sequence[str],
    range_: Sequence[str],
    rank: int,
    generic: bool = False,
    structural_only: bool = False,
    literal_allowed: bool = False,
    literal_datatypes: Sequence[str] = (),
) -> PredicateRegistryEntry:
    return PredicateRegistryEntry(
        predicate=predicate,
        predicate_uri=f"{Config.NAMESPACES['etd']}{predicate}",
        canonical_label=label,
        aliases=tuple(aliases),
        lexical_triggers=tuple(triggers),
        inverse=inverse,
        domain_families=tuple(domain),
        range_families=tuple(range_),
        specificity_rank=int(rank),
        generic_fallback=bool(generic),
        structural_only=bool(structural_only),
        literal_allowed=bool(literal_allowed),
        literal_datatypes=tuple(literal_datatypes),
    )


PREDICATE_REGISTRY: Dict[str, PredicateRegistryEntry] = {
    "relatedTo": _entry(
        "relatedTo",
        label="related to",
        aliases=("relatedto", "related to"),
        triggers=(r"\b(related to|in relation to|connected to)\b",),
        inverse="relatedTo",
        domain=("any",),
        range_=("any",),
        rank=0,
        generic=True,
    ),
    "associatedWith": _entry(
        "associatedWith",
        label="associated with",
        aliases=("associatedwith", "associated with", "linkedto", "linked to"),
        triggers=(r"\b(associated with|linked to)\b",),
        inverse="associatedWith",
        domain=("any",),
        range_=("any",),
        rank=1,
        generic=True,
    ),
    "uses": _entry(
        "uses",
        label="uses",
        aliases=("uses", "use", "used", "using", "utilizes", "utilises", "applies"),
        triggers=(r"\b(use|uses|used|using|utili[sz]e[sd]?|applying|appl(?:y|ies|ied))\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "policy_or_framework", "concept", "dataset"),
        range_=("dataset", "concept", "policy_or_framework", "system_or_model", "resource"),
        rank=3,
    ),
    "trainedOn": _entry(
        "trainedOn",
        label="trained on",
        aliases=("trainedon", "trained on", "fine tuned on", "fine-tuned on"),
        triggers=(r"\b(trained on|training data|fine tuned on|fine-tuned on|train|trains|training)\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process"),
        range_=("dataset",),
        rank=3,
        # "trained on <description of the training data>" is a
        # common general textual pattern (the training data is described
        # inline rather than named as a separately-typed Dataset entity) --
        # general semantic property of this predicate, not corpus-specific.
        literal_allowed=True,
        literal_datatypes=("string",),
    ),
    "reflects": _entry(
        "reflects",
        label="reflects",
        aliases=("reflects", "reflect", "reflected", "mirrors", "reveals", "revealed"),
        triggers=(r"\b(reflect|reflects|reflected|reflecting|mirror|mirrors|mirrored|mirroring|reveal|reveals|revealed|revealing)\b",),
        inverse=None,
        domain=("dataset", "system_or_model", "concept", "policy_or_framework", "paper_or_analysis"),
        range_=("risk_or_harm", "concept", "system_or_model"),
        rank=3,
    ),
    "hasRisk": _entry(
        "hasRisk",
        label="has risk",
        aliases=("hasrisk", "has risk"),
        triggers=(r"\b(has|have|poses|pose|presents|present|introduces|creates)\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "dataset", "policy_or_framework", "organisation", "phenomenon"),
        range_=("risk_or_harm",),
        rank=3,
    ),
    "exhibitsRisk": _entry(
        "exhibitsRisk",
        label="exhibits risk",
        aliases=("exhibitsrisk", "exhibits risk", "shows bias", "reveals bias"),
        triggers=(r"\b(exhibit|exhibits|show|shows|demonstrate|demonstrates|reveals|reveal|exposes|expose)\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "dataset", "policy_or_framework", "organisation", "phenomenon"),
        range_=("risk_or_harm",),
        rank=3,
    ),
    "causes": _entry(
        "causes",
        label="causes",
        aliases=("causes", "cause", "caused", "leads to", "results in"),
        triggers=(r"\b(cause|causes|caused|causing|lead to|leads to|result in|results in|trigger|triggers|triggered)\b",),
        inverse="causedBy",
        domain=("system_or_model", "algorithm_or_process", "policy_or_framework", "concept", "phenomenon"),
        range_=("risk_or_harm", "population_or_group", "concept", "negative_consequence"),
        rank=3,
    ),
    "causedBy": _entry(
        "causedBy",
        label="caused by",
        aliases=("causedby", "caused by", "driven by"),
        triggers=(r"\b(caused by|driven by|resulting from|results from)\b",),
        inverse="causes",
        domain=("risk_or_harm", "concept", "negative_consequence"),
        range_=("system_or_model", "algorithm_or_process", "policy_or_framework", "phenomenon"),
        rank=3,
    ),
    "affects": _entry(
        "affects",
        label="affects",
        aliases=("affects", "affect", "impacts", "influences", "harms", "contaminate", "contaminates", "contaminated"),
        triggers=(r"\b(affect|affects|affected(?!\s+by)|affecting|impact|impacts|impacted(?!\s+by)|impacting|influence|influences|influenced(?!\s+by)|influencing|harm|harms|harmed|harming|contaminat(?:e|es|ed|ing))\b",),
        inverse="affectedBy",
        domain=("system_or_model", "algorithm_or_process", "policy_or_framework", "dataset", "phenomenon"),
        range_=("population_or_group", "concept", "risk_or_harm", "negative_consequence"),
        rank=3,
    ),
    "affectedBy": _entry(
        "affectedBy",
        label="affected by",
        aliases=("affectedby", "affected by", "impacted by", "influenced by"),
        triggers=(r"\b(affected by|impacted by|influenced by|driven by|targeted by|harmed by|burdened by|disadvantaged by)\b",),
        inverse="affects",
        domain=("population_or_group", "concept", "risk_or_harm", "negative_consequence"),
        range_=("system_or_model", "algorithm_or_process", "policy_or_framework", "dataset", "phenomenon"),
        rank=3,
    ),
    "particularlyAffects": _entry(
        "particularlyAffects",
        label="particularly affects",
        aliases=("particularlyaffects", "particularly affects", "disproportionately affects", "discriminates against"),
        triggers=(
            r"\b(particularly affect|particularly affects|particularly affecting|disproportionately affect|disproportionately affects|disproportionately affecting|discriminat(?:e|es|ed|ing) against|performed worse on|performs worse on|worse on|worse for)\b",
            # "show(s)/exhibit(s)/demonstrate(s) <noun phrase> against" (e.g. "shows systematic bias
            # against X") expresses the same disparate-impact relation as "discriminates against" but
            # without that exact verb; without this, such sentences only match the generic exhibitsRisk
            # cue and get mis-hinted toward a predicate whose range excludes population/group targets.
            r"\b(?:show|shows|exhibit|exhibits|demonstrate|demonstrates)\b[^.!?]{0,60}?\bagainst\b",
        ),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "policy_or_framework", "dataset", "phenomenon"),
        range_=("population_or_group",),
        rank=4,
    ),
    "partOfDomain": _entry(
        "partOfDomain",
        label="part of domain",
        aliases=("partofdomain", "part of domain"),
        triggers=(r"\b(domain(?:s)? like|domains? such as|domain of|domains of|in the domain of|within the domain of|high[- ]stakes domains? like)\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "dataset", "concept", "policy_or_framework"),
        range_=("domain",),
        rank=2,
    ),
    "mitigates": _entry(
        "mitigates",
        label="mitigates",
        aliases=("mitigates", "mitigate", "reduces", "addresses"),
        triggers=(r"\b(mitigate|mitigates|mitigated|reduce|reduces|reduced|address|addresses|addressed)\b",),
        inverse="mitigatedBy",
        domain=("system_or_model", "algorithm_or_process", "policy_or_framework", "concept"),
        range_=("risk_or_harm", "concept"),
        rank=3,
    ),
    "mitigatedBy": _entry(
        "mitigatedBy",
        label="mitigated by",
        aliases=("mitigatedby", "mitigated by", "reduced by", "addressed by"),
        triggers=(r"\b(mitigated by|reduced by|addressed by)\b",),
        inverse="mitigates",
        domain=("risk_or_harm", "concept"),
        range_=("system_or_model", "algorithm_or_process", "policy_or_framework", "concept"),
        rank=3,
    ),
    "supports": _entry(
        "supports",
        label="supports",
        aliases=("supports", "support", "recommended"),
        triggers=(r"\b(support|supports|supported|recommend|recommends|recommended)\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "policy_or_framework", "paper_or_analysis", "concept"),
        range_=("concept", "policy_or_framework", "system_or_model", "dataset"),
        rank=3,
    ),
    "criticizes": _entry(
        "criticizes",
        label="criticizes",
        aliases=("criticizes", "criticises", "criticized", "raises concerns"),
        triggers=(r"\b(criticize|criticizes|criticized|raise concerns|raises concerns)\b",),
        inverse=None,
        domain=("paper_or_analysis", "person", "organisation", "project", "policy_or_framework"),
        range_=("system_or_model", "policy_or_framework", "concept", "dataset"),
        rank=3,
    ),
    "endorses": _entry(
        "endorses",
        label="endorses",
        aliases=("endorses", "endorse", "approved"),
        triggers=(r"\b(endorse|endorses|endorsed|approve|approves|approved)\b",),
        inverse=None,
        domain=("paper_or_analysis", "person", "organisation", "project"),
        range_=("policy_or_framework", "concept", "system_or_model"),
        rank=3,
    ),
    "disagreesWith": _entry(
        "disagreesWith",
        label="disagrees with",
        aliases=("disagreeswith", "disagrees with", "disputed"),
        triggers=(r"\b(disagree|disagrees|disagreed|dispute|disputes|disputed)\b",),
        inverse="disagreesWith",
        domain=("person", "organisation", "paper_or_analysis", "project"),
        range_=("policy_or_framework", "concept", "paper_or_analysis", "system_or_model"),
        rank=3,
    ),
    "opposes": _entry(
        "opposes",
        label="opposes",
        aliases=("opposes", "oppose", "opposed"),
        triggers=(r"\b(oppose|opposes|opposed)\b",),
        inverse=None,
        domain=("person", "organisation", "paper_or_analysis", "project"),
        range_=("policy_or_framework", "concept", "system_or_model"),
        rank=3,
    ),
    "compliesWith": _entry(
        "compliesWith",
        label="complies with",
        aliases=("complieswith", "complies with", "comply with"),
        triggers=(r"\b(comply with|complies with|complied with)\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "organisation", "policy_or_framework", "project"),
        range_=("policy_or_framework",),
        rank=3,
    ),
    "requiresComplianceWith": _entry(
        "requiresComplianceWith",
        label="requires compliance with",
        aliases=("requirescompliancewith", "requires compliance with", "must comply with"),
        triggers=(r"\b(requires compliance with|must comply with)\b",),
        inverse=None,
        domain=("policy_or_framework", "system_or_model", "organisation", "project", "concept"),
        range_=("policy_or_framework",),
        rank=3,
    ),
    "requires": _entry(
        "requires",
        label="requires",
        aliases=("requires", "require", "required", "must"),
        triggers=(r"\b(require|requires|required|must)\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "policy_or_framework", "concept", "domain"),
        range_=("concept", "policy_or_framework", "dataset", "system_or_model"),
        rank=2,
    ),
    "violates": _entry(
        "violates",
        label="violates",
        aliases=("violates", "violate", "breach", "non compliant", "non-compliant"),
        triggers=(r"\b(violate|violates|violated|breach|breaches|breached|non compliant|non-compliant)\b",),
        inverse=None,
        domain=("system_or_model", "algorithm_or_process", "organisation", "project"),
        range_=("policy_or_framework",),
        rank=3,
    ),
    "referencesInstrument": _entry(
        "referencesInstrument",
        label="references governance instrument",
        aliases=("referencesinstrument", "subject to", "under"),
        triggers=(r"\b(under|as required by|as defined by|subject to)\b",),
        inverse=None,
        domain=("paper_or_analysis", "project", "policy_or_framework", "concept"),
        range_=("policy_or_framework",),
        rank=2,
    ),
    "referencesSource": _entry(
        "referencesSource",
        label="references source document",
        aliases=("referencessource", "published in", "reported in"),
        triggers=(r"\b(published in|reported in|study|paper|report)\b",),
        inverse=None,
        domain=("paper_or_analysis", "project", "concept", "policy_or_framework"),
        range_=("paper_or_analysis", "project", "resource"),
        rank=2,
        # A "source" is very commonly expressed as a literal
        # URL or citation string rather than a separately-typed entity --
        # general semantic property of a reference/citation predicate.
        literal_allowed=True,
        literal_datatypes=("string", "uri"),
    ),
    "cites": _entry(
        "cites",
        label="cites source",
        aliases=("cites", "according to", "as per"),
        triggers=(r"\b(according to|as per|cites|cited)\b",),
        inverse=None,
        domain=("paper_or_analysis", "project", "concept", "policy_or_framework"),
        range_=("paper_or_analysis", "project", "resource"),
        rank=2,
        literal_allowed=True,
        literal_datatypes=("string", "uri"),
    ),
    "includes": _entry(
        "includes",
        label="includes",
        aliases=("includes", "include", "included in"),
        triggers=(r"\b(include|includes|included)\b",),
        inverse=None,
        domain=("dataset", "paper_or_analysis", "project", "system_or_model", "concept"),
        range_=("concept", "dataset", "system_or_model", "policy_or_framework"),
        rank=2,
        # An inclusion/enumeration
        # predicate's object is very often a literal list/enumeration of
        # constituent items in free text, not a separately-typed entity per
        # item -- general semantic property of "includes", not specific to
        # any one dataset's phrasing.
        literal_allowed=True,
        literal_datatypes=("any",),
    ),
    # A predicate justified by repeated audit
    # evidence for the "is" fallback -- a subject described/defined by a
    # descriptive text value (overwhelmingly the "Label: descriptive text"
    # structural pattern). Broad domain/range ("any") since the object is
    # almost always a literal description rather than another typed
    # entity, matching relatedTo's own breadth for the same reason.
    "hasDescription": _entry(
        "hasDescription",
        label="has description",
        aliases=("hasdescription", "has description", "described as", "definedas", "defined as"),
        triggers=(),
        inverse=None,
        domain=("any",),
        range_=("any",),
        rank=2,
        # Made explicit -- range_=("any",) already permitted a
        # literal object via the family wildcard, but this predicate's own
        # semantics (a descriptive text value) are literal-shaped by
        # definition, so it is recorded explicitly here too.
        literal_allowed=True,
        literal_datatypes=("any",),
    ),
    # A predicate justified by repeated audit
    # evidence for the "has" fallback -- a subject holding a
    # characteristic, property/value, or state (as distinct from a
    # structural component/membership relation, which should prefer a
    # more specific existing predicate when the evidence supports one).
    "hasAttribute": _entry(
        "hasAttribute",
        label="has attribute",
        aliases=("hasattribute", "has attribute", "hascharacteristic", "has characteristic"),
        triggers=(),
        inverse=None,
        domain=("any",),
        range_=("any",),
        rank=2,
        literal_allowed=True,
        literal_datatypes=("any",),
    ),
    # Defect-2 root-cause fix: the raw LLM phrases "contains"/"has
    # subject"/"contains subject" (utils/predicate_map.py) used to map onto
    # containsSubject, which collides with that name's own distinct,
    # pre-existing structural declaration (DiscourseUnit -> Entity, see
    # ontology/add_schema.py). Real usage is descriptive/quantitative
    # content-of-a-resource text (e.g. a Dataset's row count or a subject-
    # matter summary of an Analysis/Organisation resource), not an entity
    # reference -- a genuinely different relation, so it gets its own name
    # here and in the ontology (matching hasDescription/hasAttribute's
    # established literal-carrying pattern) rather than reusing
    # containsSubject's name or blanket-granting literal_allowed on it.
    "hasSubjectDescription": _entry(
        "hasSubjectDescription",
        label="has subject-matter description",
        aliases=("contains", "containssubject", "contains subject", "hassubject", "has subject"),
        triggers=(),
        inverse=None,
        domain=("dataset", "paper_or_analysis", "organisation"),
        range_=("any",),
        rank=2,
        literal_allowed=True,
        literal_datatypes=("any",),
    ),
    # Category G (manual semantic-QA pass): "Adult/Census Income Dataset
    # is UCI Machine Learning Repository" and "MIMIC-CXR Database
    # hasDescription PhysioNet" both misuse an identity/description
    # predicate for what the source text actually means: the dataset is
    # HOSTED BY/available at a repository/platform, a distinct entity, not
    # something the dataset "is" or is textually "described" as.
    # literal_allowed=False -- the object is always a real hosting entity.
    "hostedBy": _entry(
        "hostedBy",
        label="hosted by",
        aliases=("hosted by", "hosted on", "hosted in", "available at", "found on", "found in", "listed on"),
        triggers=(r"\b(hosted (by|on|in)|available (at|on|via)|found (on|in))\b",),
        inverse=None,
        domain=("dataset", "resource", "any"),
        range_=("organisation", "resource", "any"),
        rank=2,
        literal_allowed=False,
        literal_datatypes=(),
    ),
    # Defect-2 root-cause fix: containsSubject is deliberately NOT
    # registered here. Its compact lowercased local name ("containssubject")
    # is identical to the alias phrase "contains subject" that
    # utils/predicate_map.py now maps to hasSubjectDescription -- giving
    # containsSubject its own self-referential CANONICAL_REL_PREDICATES
    # entry would collide with that same key. containsSubject remains
    # declared in the ontology (owl:ObjectProperty, domain DiscourseUnit,
    # range Entity) as a reserved, currently-unreachable structural
    # predicate: no live code path emits it any more (all its former
    # LLM-alias routes now target hasSubjectDescription instead), so the
    # absence of a runtime literal_allowed=False guard here has no
    # practical effect on the current defect.
    "mentions": _entry(
        "mentions",
        label="mentions",
        aliases=("mentions", "mention"),
        triggers=(r"\b(mention|mentions|mentioned)\b",),
        inverse=None,
        domain=("discourse", "resource"),
        range_=("any",),
        rank=-1,
        structural_only=True,
    ),
}


STRUCTURAL_RELATION_PREDICATES: Set[str] = {
    "mentions",
    "answersQuestion",
    "authoredBy",
    "askedBy",
    "createdBy",
    "belongsToTheme",
    "castBy",
    "receivesVote",
    "castVote",
}

SEMANTIC_RELATION_PREDICATES: Set[str] = {
    pred for pred, entry in PREDICATE_REGISTRY.items() if not entry.structural_only
}


class SemanticRelationPolicy:
    def __init__(self, domain_profile=None) -> None:
        # Active domain profile (see domain_profiles.py) -- same
        # resolution pattern as EnhancedSemanticPipeline, RDFSerializer,
        # CanonicalLinker, and EntityNormalizer: an explicit
        # domain_profile= overrides (used by tests exercising a synthetic
        # profile); otherwise resolved from Config.DOMAIN_PROFILE,
        # reproducing current AI-ethics behaviour by default.
        if domain_profile is not None:
            self.domain_profile = domain_profile
        else:
            from domain_profiles import get_active_profile

            self.domain_profile = get_active_profile()
        self.registry = PREDICATE_REGISTRY
        self.structural_predicates = set(STRUCTURAL_RELATION_PREDICATES)
        self.semantic_predicates = set(SEMANTIC_RELATION_PREDICATES)
        self.allowed_fields = set(SEMANTIC_RELATION_FIELDS)
        self._alias_to_predicate: Dict[str, str] = {}
        self._trigger_patterns: Dict[str, List[re.Pattern[str]]] = {}
        for predicate, entry in self.registry.items():
            for alias in entry.aliases:
                self._alias_to_predicate[normalize_text(alias)] = predicate
                alias_predicate = normalize_predicate(alias)
                if alias_predicate != "relatedTo" or predicate == "relatedTo":
                    self._alias_to_predicate[alias_predicate] = predicate
            self._alias_to_predicate[normalize_predicate(predicate)] = predicate
            self._trigger_patterns[predicate] = [re.compile(pattern, re.I) for pattern in entry.lexical_triggers]

    def predicate_registry_rows(self) -> List[Dict[str, Any]]:
        return [self.registry[p].to_dict() for p in sorted(self.registry.keys())]

    def registry_json(self) -> str:
        return json.dumps(self.predicate_registry_rows(), ensure_ascii=False, indent=2)

    def predicate_specificity(self, predicate: str) -> int:
        entry = self.registry.get(self.canonicalize_predicate(predicate))
        return int(entry.specificity_rank) if entry else -1

    def canonicalize_predicate(self, raw_predicate: str) -> str:
        raw = str(raw_predicate or "").strip()
        if not raw:
            return ""
        lowered = normalize_text(raw)
        if lowered in self._alias_to_predicate:
            return self._alias_to_predicate[lowered]
        normalized = normalize_predicate(raw)
        if normalized in self._alias_to_predicate:
            return self._alias_to_predicate[normalized]
        return normalized

    def infer_relation_origin(self, relation: Relation) -> str:
        meta = getattr(relation, "metadata", None) or {}
        extractor = str(meta.get("extractor") or meta.get("source") or "").strip().lower()
        if getattr(relation, "is_inferred", False) or extractor == "reasoner":
            return "reasoner"
        if extractor == "lexical":
            return "lexical_recovery"
        if extractor in {"llm", "llm_relation_extractor"}:
            return "LLM"
        if meta.get("predicate_policy_source") == "canonical_ontology_equivalent":
            return "canonical_mapping"
        if meta.get("predicate_policy_source") == "strongly_entailed_semantic_mapping":
            return "entailed_mapping"
        return "LLM"

    def infer_modality(self, *, text: str, field: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        meta = dict(metadata or {})
        raw_text = str(text or "")
        norm = normalize_text(raw_text)
        tags: List[str] = []

        discourse_mode = str(meta.get("discourse_mode") or "").strip()
        if not discourse_mode:
            if str(field or "").strip() == "question.body":
                discourse_mode = "interrogative"
            elif raw_text.strip().endswith("?"):
                discourse_mode = "interrogative"
            else:
                discourse_mode = "declarative"

        if discourse_mode == "interrogative":
            tags.append("questioned")
        if re.search(r"\b(hypothetic(?:al|ally)|possible to|possibly|could|might|may)\b", norm, flags=re.I):
            tags.append("hypothetical")
        if re.search(r"\bif\b", norm, flags=re.I):
            tags.append("conditional")
        if re.search(r"\b(according to|reported|study|studies|analysis|paper|found)\b", norm, flags=re.I):
            tags.append("reported")
        if re.search(r"\b(should|ought|must)\b", norm, flags=re.I):
            tags.append("normative")
        if re.search(r"\b(maybe|perhaps|uncertain|unclear|debate|debatable)\b", norm, flags=re.I):
            tags.append("uncertain")
        if not tags:
            tags.append("asserted")

        epistemic_status = str(meta.get("epistemic_status") or "").strip()
        if not epistemic_status:
            priority = ["hypothetical", "conditional", "questioned", "reported", "uncertain", "normative", "asserted"]
            for candidate in priority:
                if candidate in tags:
                    epistemic_status = candidate
                    break
        return {
            "discourse_mode": discourse_mode,
            "epistemic_status": epistemic_status or "asserted",
            "modality_tags": "|".join(dict.fromkeys(tags)),
        }

    @staticmethod
    def _surface_variant_pattern(variant: str) -> str:
        escaped = re.escape(str(variant or "").strip())
        return escaped.replace(r"\ ", r"\s+")

    # CORE_GENERIC remainder of the former "generic_aliases" word list --
    # academic/structural nouns overloaded in essentially any domain, not
    # AI-ethics-specific. Domain vocabulary comes from the active
    # profile's "generic_surface_terms" gazetteer instead -- the SAME
    # gazetteer pipeline.py::_generic_surface_label/
    # _entity_surface_variants consult -- see domain_profiles.py.
    _CORE_GENERIC_ALIAS_TERMS: Set[str] = {
        "system", "systems", "model", "models", "dataset", "data",
        "project", "study", "paper", "report", "analysis",
    }

    def entity_surface_variants(self, entity: Optional[Entity]) -> List[str]:
        if entity is None:
            return []
        meta = getattr(entity, "metadata", None) or {}
        raw_variants: List[Tuple[str, str]] = [
            (str(getattr(entity, "label", "") or ""), "label"),
            (str(meta.get("surface_form") or ""), "surface_form"),
            (str(meta.get("wikidataLabel") or ""), "wikidata"),
        ]
        aliases = meta.get("aliases")
        if isinstance(aliases, list):
            raw_variants.extend((str(alias or ""), "alias") for alias in aliases)

        label = str(getattr(entity, "label", "") or "").strip()
        if label:
            deparen = re.sub(r"\s*\([^)]*\)\s*", " ", label).strip()
            if deparen and deparen != label:
                raw_variants.append((deparen, "derived"))
            slash_parts = [part.strip() for part in re.split(r"\s*/\s*", label) if part.strip()]
            if len(slash_parts) > 1:
                raw_variants.extend((part, "derived") for part in slash_parts)

        normalized: List[str] = []
        seen: Set[str] = set()
        label_norm = normalize_text(label)
        for raw, source_kind in raw_variants:
            norm = normalize_text(raw)
            is_generic_alias = norm in self._CORE_GENERIC_ALIAS_TERMS or self.domain_profile.matches_gazetteer("generic_surface_terms", norm)
            if source_kind != "label" and norm != label_norm and is_generic_alias:
                continue
            if len(norm) < 2 or norm in seen:
                continue
            seen.add(norm)
            normalized.append(norm)
        normalized.sort(key=len, reverse=True)
        return normalized

    def entity_supported_in_text(self, entity: Optional[Entity], evidence: str) -> bool:
        norm_evidence = normalize_text(str(evidence or ""))
        if entity is None or not norm_evidence:
            return False
        for variant in self.entity_surface_variants(entity):
            if re.search(rf"(?<!\w){self._surface_variant_pattern(variant)}(?!\w)", norm_evidence, flags=re.IGNORECASE):
                return True
        return False

    def entity_variant_hits(self, entity: Optional[Entity], text: str) -> List[Tuple[int, int, str]]:
        raw_text = str(text or "")
        if entity is None or not raw_text:
            return []
        hits: List[Tuple[int, int, str]] = []
        seen: Set[Tuple[int, int]] = set()
        for variant in self.entity_surface_variants(entity):
            pattern = rf"(?<!\w){self._surface_variant_pattern(variant)}(?!\w)"
            for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE):
                key = (int(match.start()), int(match.end()))
                if key in seen:
                    continue
                seen.add(key)
                hits.append((key[0], key[1], variant))
        return sorted(hits, key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

    @staticmethod
    def _split_local_sentences(evidence: str) -> List[str]:
        norm = normalize_text(str(evidence or ""))
        if not norm:
            return []
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", norm)
        out = [part.strip() for part in parts if str(part).strip()]
        return out or [norm]

    @classmethod
    def candidate_local_sentences(
        cls,
        evidence: str,
        source_variants: Sequence[str],
        target_variants: Sequence[str],
    ) -> List[str]:
        sentences = cls._split_local_sentences(evidence)
        matched: List[str] = []
        for sentence in sentences:
            has_source = any(
                re.search(rf"(?<!\w){cls._surface_variant_pattern(variant)}(?!\w)", sentence, flags=re.IGNORECASE)
                for variant in source_variants
            )
            has_target = any(
                re.search(rf"(?<!\w){cls._surface_variant_pattern(variant)}(?!\w)", sentence, flags=re.IGNORECASE)
                for variant in target_variants
            )
            if has_source and has_target:
                matched.append(sentence)
        return matched

    @staticmethod
    def _find_all_occurrences(text: str, needle: str) -> List[Tuple[int, int]]:
        if not text or not needle:
            return []
        out: List[Tuple[int, int]] = []
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx < 0:
                break
            out.append((idx, idx + len(needle)))
            start = idx + 1
        return out

    @staticmethod
    def _left_context_token(text: str, start: int) -> str:
        left = str(text or "")[: max(0, int(start))].rstrip()
        if not left:
            return ""
        match = re.search(r"(\b[\w-]+\b)\W*$", left)
        return normalize_text(match.group(1)) if match else ""

    @staticmethod
    def _comparison_boundary_after(text: str, start_offset: int) -> int:
        raw_text = str(text or "")
        if not raw_text:
            return 0
        tail = raw_text[int(start_offset):]
        match = re.search(r"\b(compared to|than|versus|vs\.?)\b", tail, flags=re.IGNORECASE)
        if not match:
            return len(raw_text)
        return int(start_offset) + int(match.start())

    @staticmethod
    def _subordinate_clause_boundary_after(text: str, start_offset: int) -> int:
        raw_text = str(text or "")
        if not raw_text:
            return 0
        tail = raw_text[int(start_offset):]
        match = re.search(
            r"(?:"
            r"(?:,\s*|;\s*)(?:used(?:\s+to)?|using|which|that|leading|resulting)\b"
            r"|"
            r"\s+(?:by\s+[A-Za-z][\w-]*ing|because|while|when|although|which|that|so\s+that|in\s+order\s+to)\b"
            r")",
            tail,
            flags=re.IGNORECASE,
        )
        if not match:
            return len(raw_text)
        return int(start_offset) + int(match.start())

    @staticmethod
    def _example_tail_boundary_after(text: str, start_offset: int) -> Optional[int]:
        raw_text = str(text or "")
        if not raw_text:
            return None
        tail = raw_text[int(start_offset):]
        match = re.search(
            r"(?:,\s*|\s+)(such as|including|for example|e\.g\.|eg\.?|from)\b",
            tail,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        marker = normalize_text(match.group(1))
        if marker == "from" and not re.search(r"\bto\b", tail[int(match.end()):], flags=re.IGNORECASE):
            return None
        return int(start_offset) + int(match.start())

    @staticmethod
    def _uses_contextual_cue(sentence: str, cue_start: int, cue_end: int) -> bool:
        cue_text = str(sentence or "")[int(cue_start):int(cue_end)].lower()
        if cue_text not in {"use", "uses", "used", "using"}:
            return False
        tail = str(sentence or "")[int(cue_end):]
        return bool(re.match(r"^\s+(?:in|for|as|by)\b", tail, flags=re.IGNORECASE))

    @staticmethod
    def _uses_stronger_cue_before_target(sentence: str, cue_end: int, target_start: int) -> bool:
        if int(target_start) <= int(cue_end):
            return False
        between = str(sentence or "")[int(cue_end):int(target_start)]
        if not between:
            return False
        return bool(
            re.search(
                r"\b(trained on|affected by|require(?:s|d)?|reflect(?:s|ed|ing)?|"
                r"discriminat(?:e|es|ed|ing) against|particularly affect(?:s|ed|ing)?|"
                r"disproportionately affect(?:s|ed|ing)?|perform(?:s|ed)? worse on|worse (?:on|for))\b",
                between,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _has_substantive_text(text: str) -> bool:
        return bool(re.search(r"\b[\w-]+\b", str(text or ""), flags=re.IGNORECASE))

    @staticmethod
    def _is_reporting_heading_label(label: str) -> bool:
        norm = normalize_text(str(label or ""))
        if not norm:
            return False
        if norm in {
            "economic impact",
            "clinical decision impact",
            "data representation",
            "platform governance",
            "scale of impact",
        }:
            return True
        if norm.endswith(" governance") or norm.endswith(" representation"):
            return True
        if norm.endswith(" impact") and not re.search(
            r"\b(?:bias|discrimination|disparit(?:y|ies)|risk|harm|misdiagnosis)\b",
            norm,
            flags=re.IGNORECASE,
        ):
            return True
        return False

    @classmethod
    def _subject_is_reporting_heading_prefix(
        cls,
        sentence: str,
        source_variants: Sequence[str],
    ) -> bool:
        raw = normalize_text(str(sentence or ""))
        if ":" not in raw:
            return False
        prefix = raw.split(":", 1)[0].strip(" -*")
        if not prefix or len(prefix.split()) > 6:
            return False
        if not any(normalize_text(variant) == prefix for variant in source_variants):
            return False
        return cls._is_reporting_heading_label(prefix)

    @staticmethod
    def _coordinated_object_conjuncts(label: str) -> Tuple[str, ...]:
        raw = str(label or "").strip()
        if not raw or not re.search(r"\b(?:and|or)\b", raw, flags=re.IGNORECASE):
            return tuple()
        match = re.match(r"(.+?)\s+\b(?:and|or)\b\s+(.+)$", raw, flags=re.IGNORECASE)
        if not match:
            return tuple()
        left = match.group(1).strip(" \t\n\r,.;:")
        right = match.group(2).strip(" \t\n\r,.;:")
        if not left or not right:
            return tuple()
        left_tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", left)
        right_tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", right)
        if not left_tokens or not right_tokens:
            return tuple()
        semantic_heads = {
            "women",
            "men",
            "communities",
            "community",
            "populations",
            "population",
            "businesses",
            "business",
            "people",
            "groups",
            "group",
            "residents",
            "resident",
            "neighborhoods",
            "neighbourhoods",
            "neighborhood",
            "neighbourhood",
            "areas",
            "area",
            "borrowers",
            "borrower",
            "patients",
            "patient",
            "students",
            "student",
            "algorithms",
            "algorithm",
            "systems",
            "system",
            "datasets",
            "dataset",
        }
        left_head = normalize_text(left_tokens[-1])
        right_head = normalize_text(right_tokens[-1])
        if left_head not in semantic_heads or right_head not in semantic_heads:
            return tuple()
        if left_head == right_head:
            return tuple()
        return (left, right)

    @staticmethod
    def _is_disparate_impact_text(text: str) -> bool:
        return bool(
            re.search(
                r"\b(discriminat(?:e|es|ed|ing) against|disproportionately affect(?:s|ed|ing)?|"
                r"particularly affect(?:s|ed|ing)?|perform(?:s|ed)? worse on|worse (?:on|for)|"
                r"disproportionately affected by|particularly affected by)\b",
                str(text or ""),
                flags=re.IGNORECASE,
            )
        )

    _QUANTITATIVE_DISPARITY_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|times\b)[^.!?]{0,120}?\b(?:vs\.?|versus|compared to|than)\b"
        r"|"
        r"\b(?:vs\.?|versus|compared to|than)\b[^.!?]{0,120}?\b\d+(?:\.\d+)?\s*(?:%|percent\b|times\b)",
        re.IGNORECASE,
    )

    @classmethod
    def _quantitative_disparity_marker_near(cls, text: str, target_hits: Sequence[Tuple[int, int]]) -> bool:
        """True when the text states a numeric rate/multiplier alongside an explicit
        comparison marker (e.g. "19% ... vs. 12%", "2.6 times less likely ... than")
        positioned at or after one of the target entity's mentions. This is the
        generic textual signature of a differential/disparate outcome claim about
        the target, independent of domain vocabulary (works for medical, financial,
        educational, etc. comparisons, not just AI-ethics-specific bias phrasing).
        Requiring the marker to follow a target mention (rather than merely
        co-occurring anywhere in the sentence) avoids matching an unrelated
        quantitative comparison elsewhere in a long sentence.
        """
        raw = str(text or "")
        if not target_hits:
            return False
        for match in cls._QUANTITATIVE_DISPARITY_PATTERN.finditer(raw):
            marker_start = int(match.start())
            if any(marker_start >= (target_end - 1) for _, target_end in target_hits):
                return True
        return False

    def detect_predicate_phrase(self, predicate: str, evidence: str) -> Tuple[str, str]:
        pred = self.canonicalize_predicate(predicate)
        sentence = normalize_text(str(evidence or ""))
        entry = self.registry.get(pred)
        if not entry or not sentence:
            return "", ""
        for pattern in self._trigger_patterns.get(pred, []):
            match = pattern.search(sentence)
            if match:
                phrase = sentence[int(match.start()):int(match.end())]
                if phrase.strip() == entry.canonical_label.lower():
                    return phrase, "explicit_lexical_relation"
                if pred in {"relatedTo", "associatedWith"}:
                    return phrase, "generic_meaningful_association"
                return phrase, "canonical_ontology_equivalent"
        return "", ""

    def best_predicate_match(
        self,
        evidence: str,
        *,
        include_generic: bool = False,
    ) -> Tuple[str, str, str]:
        sentence = normalize_text(str(evidence or ""))
        if not sentence:
            return "", "", ""

        candidates: List[Tuple[int, int, str, str, str]] = []
        source_priority = {
            "explicit_lexical_relation": 3,
            "canonical_ontology_equivalent": 3,
            "strongly_entailed_semantic_mapping": 2,
            "generic_meaningful_association": 1,
        }
        for predicate, entry in self.registry.items():
            if entry.structural_only:
                continue
            if entry.generic_fallback and not include_generic:
                continue
            phrase, source = self.detect_predicate_phrase(predicate, sentence)
            if not phrase or not source:
                continue
            candidates.append(
                (
                    int(entry.specificity_rank),
                    int(source_priority.get(source, 0)),
                    predicate,
                    phrase,
                    source,
                )
            )

        if not candidates:
            return "", "", ""

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        best = candidates[0]
        ambiguous = [
            item for item in candidates
            if item[0] == best[0] and item[1] == best[1] and item[2] != best[2]
        ]
        if ambiguous:
            return "", "", ""
        return best[2], best[3], best[4]

    @staticmethod
    def predicate_competition_family(predicate: str) -> str:
        pred = normalize_predicate(predicate)
        if pred in {"affects", "affectedBy", "particularlyAffects"}:
            return "impact"
        return ""

    def predicate_specificity_rank(self, predicate: str) -> int:
        pred = self.canonicalize_predicate(predicate)
        overrides = {
            "particularlyAffects": 40,
            "affects": 30,
            "affectedBy": 30,
        }
        if pred in overrides:
            return overrides[pred]
        entry = self.registry.get(pred)
        if entry is None:
            return 0
        return int(entry.specificity_rank) * 10

    def predicate_locally_entailed(
        self,
        predicate: str,
        source_variants: Sequence[str],
        target_variants: Sequence[str],
        evidence: str,
    ) -> bool:
        pred = self.canonicalize_predicate(predicate)
        norm_evidence = normalize_text(str(evidence or ""))
        if not pred or not source_variants or not target_variants or not norm_evidence:
            return False
        if pred in {"relatedTo", "associatedWith"}:
            for sentence in self.candidate_local_sentences(norm_evidence, source_variants, target_variants):
                if self.detect_predicate_phrase(pred, sentence)[1] in {
                    "generic_meaningful_association",
                    "explicit_lexical_relation",
                }:
                    return True
            return False

        cues = self._trigger_patterns.get(pred, [])
        candidate_sentences = self.candidate_local_sentences(norm_evidence, source_variants, target_variants)
        if not candidate_sentences and any(
            re.search(rf"(?<!\w){self._surface_variant_pattern(variant)}(?!\w)", norm_evidence, flags=re.IGNORECASE)
            for variant in source_variants
        ) and any(
            re.search(rf"(?<!\w){self._surface_variant_pattern(variant)}(?!\w)", norm_evidence, flags=re.IGNORECASE)
            for variant in target_variants
        ):
            candidate_sentences = [norm_evidence]
        if not candidate_sentences:
            return False

        if pred in {"affects", "affectedBy", "particularlyAffects"}:
            impact_cues = list(cues)
            if pred in {"affects", "particularlyAffects"}:
                impact_cues.extend(self._trigger_patterns.get("affectedBy", []))
            for sentence in candidate_sentences:
                source_hits: List[Tuple[int, int]] = []
                target_hits: List[Tuple[int, int]] = []
                for source_variant in source_variants:
                    source_hits.extend(
                        (int(match.start()), int(match.end()))
                        for match in re.finditer(
                            rf"(?<!\w){self._surface_variant_pattern(source_variant)}(?!\w)",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    )
                for target_variant in target_variants:
                    target_hits.extend(
                        (int(match.start()), int(match.end()))
                        for match in re.finditer(
                            rf"(?<!\w){self._surface_variant_pattern(target_variant)}(?!\w)",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    )
                if not source_hits or not target_hits:
                    continue
                if pred == "particularlyAffects" and self._quantitative_disparity_marker_near(sentence, target_hits):
                    return True
                for cue in impact_cues:
                    for cue_match in cue.finditer(sentence):
                        cue_start = int(cue_match.start())
                        cue_end = int(cue_match.end())
                        right_limit = len(sentence)
                        if pred == "particularlyAffects":
                            right_limit = min(
                                right_limit,
                                self._comparison_boundary_after(sentence, cue_end),
                                self._subordinate_clause_boundary_after(sentence, cue_end),
                            )
                        elif pred == "affects":
                            right_limit = min(right_limit, self._subordinate_clause_boundary_after(sentence, cue_end))
                        for _, source_end in source_hits:
                            for target_start, _ in target_hits:
                                if pred == "affectedBy":
                                    if source_end <= cue_start and cue_end <= target_start:
                                        return True
                                    continue
                                if source_end <= cue_start and cue_end <= target_start and target_start < right_limit:
                                    return True
                                if target_start < cue_start and cue_end <= source_end:
                                    return True
            return False

        if pred == "trainedOn":
            for sentence in candidate_sentences:
                source_hits: List[Tuple[int, int]] = []
                target_hits: List[Tuple[int, int]] = []
                for source_variant in source_variants:
                    source_hits.extend(
                        (int(match.start()), int(match.end()))
                        for match in re.finditer(
                            rf"(?<!\w){self._surface_variant_pattern(source_variant)}(?!\w)",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    )
                for target_variant in target_variants:
                    target_hits.extend(
                        (int(match.start()), int(match.end()))
                        for match in re.finditer(
                            rf"(?<!\w){self._surface_variant_pattern(target_variant)}(?!\w)",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    )
                if not source_hits or not target_hits:
                    continue
                for cue in cues:
                    for cue_match in cue.finditer(sentence):
                        cue_start = int(cue_match.start())
                        cue_end = int(cue_match.end())
                        for source_start, source_end in source_hits:
                            for target_start, _ in target_hits:
                                if source_end <= cue_start and cue_end <= target_start:
                                    return True
                                if cue_end <= source_start and source_end <= target_start:
                                    between = sentence[cue_end:target_start]
                                    if re.search(r"\b(on|using|with|from|to)\b", between, flags=re.IGNORECASE):
                                        return True
            return False

        if pred == "requires":
            for sentence in candidate_sentences:
                source_hits: List[Tuple[int, int]] = []
                target_hits: List[Tuple[int, int]] = []
                for source_variant in source_variants:
                    source_hits.extend(
                        (int(match.start()), int(match.end()))
                        for match in re.finditer(
                            rf"(?<!\w){self._surface_variant_pattern(source_variant)}(?!\w)",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    )
                for target_variant in target_variants:
                    target_hits.extend(
                        (int(match.start()), int(match.end()))
                        for match in re.finditer(
                            rf"(?<!\w){self._surface_variant_pattern(target_variant)}(?!\w)",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    )
                if not source_hits or not target_hits:
                    continue
                pronoun_led = bool(re.search(r"\b(it|this|that)\s+requires\b", sentence, flags=re.IGNORECASE))
                blocked_left_tokens = {
                    "in",
                    "of",
                    "for",
                    "like",
                    "with",
                    "to",
                    "on",
                    "about",
                    "within",
                    "among",
                    "across",
                    "through",
                    "via",
                    "by",
                    "from",
                    "as",
                    "beyond",
                }
                for cue in cues:
                    for cue_match in cue.finditer(sentence):
                        cue_start = int(cue_match.start())
                        cue_end = int(cue_match.end())
                        for source_start, source_end in source_hits:
                            if pronoun_led and self._left_context_token(sentence, source_start) in blocked_left_tokens:
                                continue
                            for target_start, _ in target_hits:
                                if source_end <= cue_start and cue_end <= target_start:
                                    return True
            return False

        if pred == "uses":
            for sentence in candidate_sentences:
                source_hits: List[Tuple[int, int]] = []
                target_hits: List[Tuple[int, int]] = []
                for source_variant in source_variants:
                    source_hits.extend(
                        (int(match.start()), int(match.end()))
                        for match in re.finditer(
                            rf"(?<!\w){self._surface_variant_pattern(source_variant)}(?!\w)",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    )
                for target_variant in target_variants:
                    target_hits.extend(
                        (int(match.start()), int(match.end()))
                        for match in re.finditer(
                            rf"(?<!\w){self._surface_variant_pattern(target_variant)}(?!\w)",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    )
                if not source_hits or not target_hits:
                    continue
                for cue in cues:
                    for cue_match in cue.finditer(sentence):
                        cue_start = int(cue_match.start())
                        cue_end = int(cue_match.end())
                        if self._uses_contextual_cue(sentence, cue_start, cue_end):
                            continue
                        cue_prefix = sentence[max(0, cue_start - 24):cue_end]
                        if re.search(
                            r"\b(am|is|are|was|were|be|been|being|will|would|can|could|may|might|must|should)\s+used\b",
                            cue_prefix,
                            flags=re.IGNORECASE,
                        ):
                            continue
                        for _, source_end in source_hits:
                            if source_end > cue_start:
                                continue
                            if cue_start - source_end > 32:
                                continue
                            for target_start, _ in target_hits:
                                if self._uses_stronger_cue_before_target(sentence, cue_end, target_start):
                                    continue
                                if cue_end <= target_start:
                                    return True
            return False

        directional = pred not in {"causedBy", "mitigatedBy"}
        for sentence in candidate_sentences:
            source_hits: List[Tuple[int, int]] = []
            target_hits: List[Tuple[int, int]] = []
            for source_variant in source_variants:
                source_hits.extend(
                    (int(match.start()), int(match.end()))
                    for match in re.finditer(
                        rf"(?<!\w){self._surface_variant_pattern(source_variant)}(?!\w)",
                        sentence,
                        flags=re.IGNORECASE,
                    )
                )
            for target_variant in target_variants:
                target_hits.extend(
                    (int(match.start()), int(match.end()))
                    for match in re.finditer(
                        rf"(?<!\w){self._surface_variant_pattern(target_variant)}(?!\w)",
                        sentence,
                        flags=re.IGNORECASE,
                    )
                )
            if not source_hits or not target_hits:
                continue

            for cue in cues:
                for cue_match in cue.finditer(sentence):
                    cue_start = int(cue_match.start())
                    cue_end = int(cue_match.end())
                    right_limit = len(sentence)
                    if pred == "particularlyAffects":
                        right_limit = min(
                            right_limit,
                            self._comparison_boundary_after(sentence, cue_end),
                            self._subordinate_clause_boundary_after(sentence, cue_end),
                        )
                    elif pred == "affects":
                        right_limit = min(right_limit, self._subordinate_clause_boundary_after(sentence, cue_end))
                    else:
                        example_boundary = self._example_tail_boundary_after(sentence, cue_end)
                        if example_boundary is not None and self._has_substantive_text(sentence[cue_end:example_boundary]):
                            right_limit = int(example_boundary)

                    for _, source_end in source_hits:
                        for target_start, _ in target_hits:
                            if source_end <= cue_start and cue_end <= target_start and target_start < right_limit:
                                return True

                    if not directional:
                        for _, target_end in target_hits:
                            for source_start, _ in source_hits:
                                if target_end <= cue_start and cue_end <= source_start and source_start < right_limit:
                                    return True

            for source_variant in source_variants:
                source_pattern = self._surface_variant_pattern(source_variant)
                for target_variant in target_variants:
                    if source_variant == target_variant:
                        continue
                    target_pattern = self._surface_variant_pattern(target_variant)
                    directional_patterns: List[str] = []
                    if pred == "hasRisk":
                        directional_patterns.extend(
                            [
                                rf"(?<!\w){target_pattern}(?!\w).{{0,80}}\b(in|within|of)\b.{{0,80}}(?<!\w){source_pattern}(?!\w)",
                                rf"(?<!\w){source_pattern}(?!\w).{{0,80}}\b(with|showing|displaying|featuring)\b.{{0,80}}(?<!\w){target_pattern}(?!\w)",
                            ]
                        )
                    elif pred == "exhibitsRisk":
                        directional_patterns.append(
                            rf"(?<!\w){source_pattern}(?!\w).{{0,100}}\b(with|showing|displaying|revealing)\b.{{0,60}}(?<!\w){target_pattern}(?!\w)"
                        )
                    if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in directional_patterns):
                        return True
        return False

    # Root-cause fix (architectural-cleanup pass, relation-validation
    # duplication): these three used to each hardcode their own AI-ethics
    # word list, byte-for-byte duplicated in pipeline.py. All three now
    # delegate to the ONE shared, profile-driven category mechanism
    # (domain_profiles.matches_semantic_category) -- pipeline.py's
    # equivalent methods delegate to the exact same function, so the two
    # layers cannot disagree on a category decision.
    def _is_generic_risk_label(self, label: str) -> bool:
        # exact=True: "Bias" alone is generic; "Intersectional Bias" is a
        # specific finding that merely contains the word "bias" and must
        # NOT be rejected as generic (see matches_semantic_category).
        return matches_semantic_category(self.domain_profile, "abstract_finding", label, exact=True)

    def _is_generic_abstract_label(self, label: str) -> bool:
        # CORE_GENERIC_FINDING_CONTEXT_NOUNS ("dataset"/"audit" and their
        # plurals) is domain-independent and checked regardless of active
        # profile; everything else (fairness/bias/etc.) is the active
        # profile's "abstract_finding" category, whole-label match.
        if normalize_text(str(label or "")) in CORE_GENERIC_FINDING_CONTEXT_NOUNS:
            return True
        return matches_semantic_category(self.domain_profile, "abstract_finding", label, exact=True)

    def _label_has_group_cue(self, label: str) -> bool:
        return matches_semantic_category(self.domain_profile, "population_group", label)

    # Translates the shared classifier's canonical family taxonomy back to
    # this file's own historical naming -- identical to canonical except
    # "classification" (kept distinct by the shared classifier so
    # pipeline.py's own causes/causedBy exclusion rule can still tell it
    # apart) folds back into "concept" here, exactly as this file treated
    # it before. See pipeline.py::_entity_semantic_family for the
    # equivalent (larger) translation it performs.
    _CANONICAL_TO_LEGACY_FAMILY: Dict[str, str] = {"classification": "concept"}

    def entity_family(self, entity: Optional[Entity]) -> str:
        """Delegates the actual classification decision to the ONE shared
        classifier, domain_profiles.classify_entity_semantic_family -- the
        same function pipeline.py::_entity_semantic_family calls. Neither
        file hardcodes any type-name/label-vocabulary mapping of its own
        any more."""
        canonical = classify_entity_semantic_family(entity, profile=self.domain_profile)
        return self._CANONICAL_TO_LEGACY_FAMILY.get(canonical, canonical)

    def _family_matches(self, actual: str, allowed: Sequence[str]) -> bool:
        if "any" in allowed:
            return True
        return actual in set(allowed)

    def domain_range_valid(self, relation: Relation, entities_by_uri: Dict[str, Entity]) -> bool:
        pred = self.canonicalize_predicate(str(getattr(relation, "relation", "") or "").strip())
        src = entities_by_uri.get(str(getattr(relation, "source", "") or ""))
        tgt = entities_by_uri.get(str(getattr(relation, "target", "") or ""))
        src_family = self.entity_family(src)
        tgt_family = self.entity_family(tgt)

        if src_family in {"discourse", "vote", "resource"} or tgt_family in {"discourse", "vote", "resource"}:
            return False

        if pred in {"hasRisk", "exhibitsRisk"}:
            return (
                src_family in {"system_or_model", "algorithm_or_process", "dataset", "policy_or_framework", "organisation", "concept"}
                and tgt_family == "risk_or_harm"
                and not self._is_generic_risk_label(str(getattr(tgt, "label", "") or ""))
            )

        if pred == "trainedOn":
            return src_family in {"system_or_model", "algorithm_or_process"} and tgt_family == "dataset"

        if pred == "reflects":
            return src_family in {"system_or_model", "algorithm_or_process", "dataset", "concept", "policy_or_framework", "paper_or_analysis"} and tgt_family in {"risk_or_harm", "concept", "system_or_model"}

        if pred == "causes":
            return src_family not in {"dataset", "resource", "person", "organisation"} and tgt_family not in {"resource", "organisation"}

        if pred == "causedBy":
            return tgt_family not in {"dataset", "resource", "person", "organisation"} and src_family not in {"resource", "organisation"}

        if pred == "affects":
            return src_family in {"system_or_model", "algorithm_or_process", "policy_or_framework", "dataset", "concept", "risk_or_harm"} and tgt_family in {"population_or_group", "concept", "risk_or_harm", "system_or_model"}

        if pred == "affectedBy":
            return src_family in {"population_or_group", "concept", "risk_or_harm", "system_or_model"} and tgt_family in {"system_or_model", "algorithm_or_process", "policy_or_framework", "dataset", "concept", "risk_or_harm"}

        if pred == "particularlyAffects":
            return src_family in {"system_or_model", "algorithm_or_process", "policy_or_framework", "dataset", "concept", "risk_or_harm"} and tgt_family == "population_or_group"

        if pred == "mitigates":
            return src_family in {"system_or_model", "algorithm_or_process", "organisation", "policy_or_framework", "concept"} and tgt_family in {"risk_or_harm", "concept"}

        if pred in {"relatedTo", "associatedWith"}:
            return src_family not in {"discourse", "vote"} and tgt_family not in {"discourse", "vote"}

        if pred == "partOfDomain":
            src_type = str(getattr(src, "entity_type", "") or "").strip().lower()
            tgt_type = str(getattr(tgt, "entity_type", "") or "").strip().lower()
            domainish_targets = {"domain", "criminaljustice", "justicedomain", "employment", "healthcare", "highrisk", "highriskai"}
            return (
                src_family in {"system_or_model", "algorithm_or_process", "dataset", "concept", "policy_or_framework"}
                and not self._is_generic_abstract_label(str(getattr(src, "label", "") or ""))
                and tgt_family == "domain"
                and (tgt_type in domainish_targets or "domain" in str(getattr(tgt, "label", "") or "").lower())
            )

        entry = self.registry.get(pred)
        if not entry:
            return False
        return self._family_matches(src_family, entry.domain_families) and self._family_matches(tgt_family, entry.range_families)

    def normalize_direction(self, relation: Relation, entities_by_uri: Dict[str, Entity], evidence_text: str) -> str:
        pred = self.canonicalize_predicate(str(getattr(relation, "relation", "") or "").strip())
        if pred != "affectedBy":
            relation.relation = pred
            return "forward"
        src = entities_by_uri.get(str(getattr(relation, "source", "") or ""))
        tgt = entities_by_uri.get(str(getattr(relation, "target", "") or ""))
        source_variants = self.entity_surface_variants(src)
        target_variants = self.entity_surface_variants(tgt)
        if self.predicate_locally_entailed("affectedBy", source_variants, target_variants, evidence_text):
            # Normalize the new target's literal-state fields
            # explicitly instead of carrying raw string values across the
            # swap, so any state that was valid for the old target does
            # not survive stale onto the new one.
            new_target_uri = str(getattr(relation, "source", "") or "")
            relation.source = str(getattr(relation, "target", "") or "")
            if hasattr(relation, "set_target_entity"):
                relation.set_target_entity(new_target_uri)
            else:
                relation.target = new_target_uri
            relation.relation = "particularlyAffects" if self._is_disparate_impact_text(evidence_text) else "affects"
            return "normalized_inverse"
        relation.relation = pred
        return "forward"

    def _entity_role_valid(
        self,
        relation: Relation,
        source_entity: Entity,
        target_entity: Entity,
        evidence_text: str,
    ) -> Tuple[bool, str]:
        source_family = self.entity_family(source_entity)
        target_family = self.entity_family(target_entity)
        source_variants = self.entity_surface_variants(source_entity)
        target_variants = self.entity_surface_variants(target_entity)
        sentence_candidates = self.candidate_local_sentences(evidence_text, source_variants, target_variants)
        sentence = sentence_candidates[0] if sentence_candidates else normalize_text(evidence_text)

        if not sentence:
            return False, "locality:no_local_sentence"

        if source_family == "dataset":
            if any(re.search(rf"\b(predictive system|system|model|algorithm)\s+like\s+{self._surface_variant_pattern(variant)}\b", sentence, flags=re.IGNORECASE) for variant in source_variants):
                return False, "entity_role:dataset_actor_mismatch_system_cue"
            if any(re.search(rf"\banalysis of\s+{self._surface_variant_pattern(variant)}\b", sentence, flags=re.IGNORECASE) for variant in source_variants):
                if str(getattr(relation, "relation", "") or "").strip() in {"reflects", "referencesSource", "cites"}:
                    return False, "entity_role:analysis_actor_missing"

        if source_family in {"person", "project", "paper_or_analysis", "risk_or_harm"} and str(getattr(relation, "relation", "") or "").strip() in {
            "hasRisk",
            "exhibitsRisk",
            "causes",
            "affects",
            "particularlyAffects",
        }:
            if re.search(r"\b(system|systems|model|models|algorithm|algorithms|technology|technologies|tool|tools)\b", sentence, flags=re.IGNORECASE):
                return False, "entity_role:actor_family_mismatch"

        if source_family == "paper_or_analysis" and target_family == "dataset" and str(getattr(relation, "relation", "") or "").strip() == "reflects":
            return False, "entity_role:analysis_to_dataset_mismatch"

        return True, "pass"

    def apply_relation_metadata(self, relation: Relation, *, field: str, evidence_text: str) -> None:
        meta = getattr(relation, "metadata", None) or {}
        modality = self.infer_modality(text=evidence_text, field=field, metadata=meta)
        for key, value in modality.items():
            if value:
                meta[key] = value
        meta["relation_origin"] = self.infer_relation_origin(relation)
        relation.metadata = meta

    @staticmethod
    def _relation_has_literal_object(relation: Relation) -> bool:
        return bool(
            getattr(relation, "target_is_literal", False)
            or getattr(relation, "target_literal", None) is not None
            or str((getattr(relation, "metadata", None) or {}).get("object_type") or "").strip().lower()
            in {"literal", "quantity", "money", "date", "percentage", "count", "temporal_range", "comparison"}
        )

    def _literal_object_supported(self, relation: Relation, validation_text: str) -> bool:
        meta = getattr(relation, "metadata", None) or {}
        candidates: List[str] = []
        if getattr(relation, "target_literal", None) is not None:
            candidates.append(str(getattr(relation, "target_literal", "")).strip())
        if str(meta.get("object_text") or "").strip():
            candidates.append(str(meta.get("object_text") or "").strip())
        quant = getattr(relation, "quantitative_information", None) or meta.get("quantitative_information") or {}
        temp = getattr(relation, "temporal_information", None) or meta.get("temporal_information") or {}
        if isinstance(quant, dict):
            for value in quant.values():
                if value is not None and str(value).strip():
                    candidates.append(str(value).strip())
        if isinstance(temp, dict):
            for value in temp.values():
                if value is not None and str(value).strip():
                    candidates.append(str(value).strip())
        norm_text = normalize_text(validation_text)
        compact_text = re.sub(r"(?<=\d)[,\s](?=\d)", "", norm_text)
        for candidate in candidates:
            if not candidate:
                continue
            norm_candidate = normalize_text(candidate)
            if norm_candidate and norm_candidate in norm_text:
                return True
            compact_candidate = re.sub(r"(?<=\d)[,\s](?=\d)", "", norm_candidate)
            if compact_candidate and compact_candidate in compact_text:
                return True
        return False

    def _local_predicate_surface_supported(self, relation: Relation, validation_text: str) -> Tuple[bool, str]:
        meta = getattr(relation, "metadata", None) or {}
        raw = str(getattr(relation, "predicate_text", None) or meta.get("predicate_text") or getattr(relation, "relation", "") or "").strip()
        if not raw:
            return False, ""
        norm_raw = normalize_text(raw)
        norm_validation = normalize_text(validation_text)
        if norm_raw and norm_raw in norm_validation:
            return True, raw
        raw_tokens = [token for token in re.findall(r"[A-Za-z0-9]+", norm_raw) if token]
        validation_tokens = [token for token in re.findall(r"[A-Za-z0-9]+", norm_validation) if token]
        if len(raw_tokens) >= 2 and validation_tokens:
            idx = 0
            for token in validation_tokens:
                if token == raw_tokens[idx]:
                    idx += 1
                    if idx == len(raw_tokens):
                        return True, raw
        phrase, source = self.detect_predicate_phrase(str(getattr(relation, "relation", "") or ""), validation_text)
        if phrase and source:
            return True, phrase
        return False, raw

    def validate(
        self,
        relation: Relation,
        *,
        entities_by_uri: Dict[str, Entity],
        discourse_text: str,
    ) -> SemanticValidationDecision:
        meta = getattr(relation, "metadata", None) or {}

        # An endpoint explicitly left unresolved by coreference resolution
        # (raw referential surface text with no resolved antecedent) must
        # never reach semantic graph materialization -- checked FIRST,
        # before any other acceptance path (including the literal-object
        # path, which could otherwise silently accept unresolved raw text
        # as if it were a genuine literal value). This is a validation-time
        # gate only: no further antecedent resolution is attempted here
        # (that is coreference resolution's job) and no entity is minted
        # here (that is blocked upstream by the unresolved-endpoint guard).
        unresolved_role = None
        if bool(meta.get("subject_endpoint_unresolved_coreference")):
            unresolved_role = "subject"
        elif bool(meta.get("object_endpoint_unresolved_coreference")):
            unresolved_role = "object"
        if unresolved_role is not None:
            surface = str(meta.get(f"{unresolved_role}_endpoint_unresolved_original_surface") or "").strip()
            category = str(meta.get(f"{unresolved_role}_endpoint_unresolved_reason") or "").strip()
            discourse_id = str(meta.get("discourse_id") or "").strip()
            source_field = str(meta.get("field") or meta.get("source_field") or "").strip()
            return SemanticValidationDecision(
                False,
                "unresolved_coreference",
                str(getattr(relation, "relation", "") or ""),
                entity_role_check="fail",
                reason_detail=(
                    f"unresolved coreference at {unresolved_role} endpoint "
                    f"(surface={surface!r}, reason={category or 'unknown'}, "
                    f"discourse_id={discourse_id or 'unknown'}, field={source_field or 'unknown'})"
                ),
            )

        field = str(meta.get("field") or meta.get("source_field") or "").strip()
        if not field or field not in self.allowed_fields:
            return SemanticValidationDecision(False, "resource_filter:field_not_allowlisted", str(getattr(relation, "relation", "") or ""), reason_detail="field not allowlisted for semantic extraction")

        if str(meta.get("relation_kind") or "").strip() in {"structural", "provenance"}:
            return SemanticValidationDecision(False, "resource_filter:non_semantic_relation_kind", str(getattr(relation, "relation", "") or ""), reason_detail="structural/provenance relations are not semantic policy candidates")

        if str(meta.get("context_source") or "").strip() in {"cooccurrence_chunk", "lightweight_mentions", "dataset_vote"}:
            return SemanticValidationDecision(False, "resource_filter:weak_structural_context", str(getattr(relation, "relation", "") or ""), reason_detail="weak structural context cannot materialize semantic relation")

        if str(meta.get("extractor") or "").strip() == "lightweight":
            return SemanticValidationDecision(False, "resource_filter:lightweight_relation_rejected", str(getattr(relation, "relation", "") or ""), reason_detail="lightweight relations cannot materialize semantic relation")

        if str(meta.get("section_category") or "").strip() != "semantic":
            return SemanticValidationDecision(False, "resource_filter:non_semantic_section", str(getattr(relation, "relation", "") or ""), reason_detail="relation does not originate from semantic section")

        source_entity = entities_by_uri.get(str(getattr(relation, "source", "") or ""))
        target_entity = entities_by_uri.get(str(getattr(relation, "target", "") or ""))
        target_is_literal = self._relation_has_literal_object(relation)
        if source_entity is None or (target_entity is None and not target_is_literal):
            return SemanticValidationDecision(False, "entity_role:missing_relation_endpoint", str(getattr(relation, "relation", "") or ""), reason_detail="relation endpoint missing after canonical linking")

        evidence_quote = str(meta.get("evidence_quote") or "").strip()
        evidence_text = str(meta.get("evidence") or "").strip()
        local_context = str(getattr(relation, "context", None) or meta.get("local_context") or "").strip()
        validation_parts: List[str] = []
        seen_validation_parts: Set[str] = set()
        for candidate in (local_context, evidence_text, evidence_quote):
            norm_candidate = normalize_text(candidate)
            if not norm_candidate or norm_candidate in seen_validation_parts:
                continue
            seen_validation_parts.add(norm_candidate)
            validation_parts.append(candidate)
        validation_text = "\n".join(validation_parts).strip()
        if not validation_text:
            return SemanticValidationDecision(False, "locality:missing_local_evidence", str(getattr(relation, "relation", "") or ""), reason_detail="missing local proposition evidence")

        raw_predicate = str(getattr(relation, "predicate_text", None) or meta.get("predicate_text") or getattr(relation, "relation", "") or "").strip()
        local_unmapped = str(meta.get("predicate_mapping_status") or getattr(relation, "predicate_mapping_status", "") or "").strip() == "local_unmapped"
        direction = self.normalize_direction(relation, entities_by_uri, validation_text) if not local_unmapped else "forward"
        predicate = self.canonicalize_predicate(str(getattr(relation, "relation", "") or "")) if not local_unmapped else str(getattr(relation, "relation", "") or "").strip()
        relation.relation = predicate
        source_entity = entities_by_uri.get(str(getattr(relation, "source", "") or ""))
        target_entity = entities_by_uri.get(str(getattr(relation, "target", "") or ""))
        if source_entity is None or (target_entity is None and not target_is_literal):
            return SemanticValidationDecision(False, "entity_role:missing_relation_endpoint", predicate, reason_detail="relation endpoint missing after direction normalization")

        modality = self.infer_modality(text=validation_text, field=field, metadata=meta)
        meta.update({k: v for k, v in modality.items() if v})

        source_family = self.entity_family(source_entity)
        target_family = self.entity_family(target_entity) if target_entity is not None else "literal"
        source_variants = self.entity_surface_variants(source_entity)
        target_variants = self.entity_surface_variants(target_entity) if target_entity is not None else []
        candidate_sentences = self.candidate_local_sentences(validation_text, source_variants, target_variants) if target_entity is not None else [validation_text]
        locality_scope = "same_sentence" if candidate_sentences else ""
        if target_entity is not None and not candidate_sentences:
            if local_unmapped:
                return SemanticValidationDecision(
                    False,
                    "rejected_not_entailed",
                    predicate,
                    locality_scope="none",
                    discourse_mode=str(meta.get("discourse_mode") or ""),
                    epistemic_status=str(meta.get("epistemic_status") or ""),
                    modality_tags=tuple(str(meta.get("modality_tags") or "").split("|")) if meta.get("modality_tags") else tuple(),
                    relation_origin=self.infer_relation_origin(relation),
                    subject_family=source_family,
                    object_family=target_family,
                    entity_role_check="fail",
                    domain_range_check="not_applicable",
                    entailment_check="fail",
                    modality_check="pass",
                    reason_detail="unmapped predicate is not supportable from a shared local proposition span",
                )
            return SemanticValidationDecision(
                False,
                "locality:no_shared_local_proposition",
                predicate,
                locality_scope="none",
                discourse_mode=str(meta.get("discourse_mode") or ""),
                epistemic_status=str(meta.get("epistemic_status") or ""),
                modality_tags=tuple(str(meta.get("modality_tags") or "").split("|")) if meta.get("modality_tags") else tuple(),
                relation_origin=self.infer_relation_origin(relation),
                subject_family=source_family,
                object_family=target_family,
                entity_role_check="fail",
                domain_range_check="unknown",
                entailment_check="unknown",
                modality_check="pass",
                reason_detail="subject and object do not co-occur in the same local proposition span",
            )

        if not self.entity_supported_in_text(source_entity, validation_text):
            return SemanticValidationDecision(False, "entity_role:source_not_in_local_evidence", predicate, locality_scope=locality_scope, reason_detail="source entity not supportable from local evidence")
        if target_entity is not None and not self.entity_supported_in_text(target_entity, validation_text):
            return SemanticValidationDecision(False, "entity_role:target_not_in_local_evidence", predicate, locality_scope=locality_scope, reason_detail="target entity not supportable from local evidence")
        if target_is_literal and not self._literal_object_supported(relation, validation_text):
            return SemanticValidationDecision(False, "entity_role:literal_not_in_local_evidence", predicate, locality_scope=locality_scope or "same_sentence", reason_detail="literal object not supportable from local evidence")

        if local_unmapped:
            predicate_supported, predicate_surface = self._local_predicate_surface_supported(relation, validation_text)
            if not predicate_supported:
                return SemanticValidationDecision(False, "rejected_not_entailed", predicate, locality_scope=locality_scope or "same_sentence", reason_detail="unmapped predicate is not supportable from local evidence")
            meta["canonical_predicate"] = predicate
            meta["predicate_surface_phrase"] = predicate_surface or raw_predicate or predicate
            meta["predicate_policy_source"] = "local_unmapped_predicate"
            meta["relation_origin"] = self.infer_relation_origin(relation)
            meta["direction_normalized"] = direction
            meta["locality_scope"] = locality_scope or "same_sentence"
            meta["subject_surface_form"] = str(source_variants[0] if source_variants else getattr(source_entity, "label", ""))
            meta["object_surface_form"] = str(target_variants[0] if target_variants else (getattr(relation, "target_literal", None) or meta.get("object_text") or ""))
            meta["subject_canonical_entity"] = str(getattr(source_entity, "uri", "") or "")
            meta["object_canonical_entity"] = str(getattr(target_entity, "uri", "") or "") if target_entity is not None else ""
            meta["subject_family"] = source_family
            meta["object_family"] = target_family
            meta["entity_role_check"] = "pass"
            meta["domain_range_check"] = "not_applicable"
            meta["entailment_check"] = "pass"
            meta["modality_check"] = "pass"
            relation.metadata = meta
            return SemanticValidationDecision(
                True,
                "accepted_unmapped_predicate",
                predicate,
                predicate_surface_phrase=predicate_surface or raw_predicate or predicate,
                predicate_source="local_unmapped_predicate",
                locality_scope=locality_scope or "same_sentence",
                direction=direction,
                discourse_mode=str(meta.get("discourse_mode") or ""),
                epistemic_status=str(meta.get("epistemic_status") or ""),
                modality_tags=tuple(tag for tag in str(meta.get("modality_tags") or "").split("|") if tag),
                relation_origin=str(meta.get("relation_origin") or ""),
                subject_family=source_family,
                object_family=target_family,
                entity_role_check="pass",
                domain_range_check="not_applicable",
                entailment_check="pass",
                modality_check="pass",
                reason_detail="supported local predicate retained without ontology mapping",
            )

        if target_is_literal:
            predicate_surface_phrase, predicate_source = self.detect_predicate_phrase(predicate, validation_text)
            if not predicate_source:
                predicate_surface_phrase = raw_predicate or predicate
                predicate_source = "literal_assertion"
            meta["canonical_predicate"] = predicate
            meta["predicate_surface_phrase"] = predicate_surface_phrase or predicate
            meta["predicate_policy_source"] = predicate_source
            meta["relation_origin"] = self.infer_relation_origin(relation)
            meta["direction_normalized"] = direction
            meta["locality_scope"] = locality_scope or "same_sentence"
            meta["subject_surface_form"] = str(source_variants[0] if source_variants else getattr(source_entity, "label", ""))
            meta["object_surface_form"] = str(getattr(relation, "target_literal", None) or meta.get("object_text") or "")
            meta["subject_canonical_entity"] = str(getattr(source_entity, "uri", "") or "")
            meta["object_canonical_entity"] = ""
            meta["subject_family"] = source_family
            meta["object_family"] = target_family
            meta["entity_role_check"] = "pass"
            meta["domain_range_check"] = "not_applicable"
            meta["entailment_check"] = "pass"
            meta["modality_check"] = "pass"
            relation.metadata = meta
            return SemanticValidationDecision(
                True,
                "accepted_literal_assertion",
                predicate,
                predicate_surface_phrase=predicate_surface_phrase or predicate,
                predicate_source=predicate_source,
                locality_scope=locality_scope or "same_sentence",
                direction=direction,
                discourse_mode=str(meta.get("discourse_mode") or ""),
                epistemic_status=str(meta.get("epistemic_status") or ""),
                modality_tags=tuple(tag for tag in str(meta.get("modality_tags") or "").split("|") if tag),
                relation_origin=str(meta.get("relation_origin") or ""),
                subject_family=source_family,
                object_family=target_family,
                entity_role_check="pass",
                domain_range_check="not_applicable",
                entailment_check="pass",
                modality_check="pass",
                reason_detail="supported literal assertion retained",
            )

        if any(self._subject_is_reporting_heading_prefix(sentence, source_variants) for sentence in candidate_sentences):
            return SemanticValidationDecision(
                False,
                "entity_role:reporting_heading_not_semantic_subject",
                predicate,
                locality_scope=locality_scope,
                subject_family=source_family,
                object_family=target_family,
                reason_detail="reporting/category heading before colon is not a semantic actor",
            )

        if predicate in {
            "affects",
            "particularlyAffects",
            "causes",
            "mitigates",
            "uses",
            "trainedOn",
            "reflects",
            "hasRisk",
            "exhibitsRisk",
        }:
            for variant in target_variants:
                if self._coordinated_object_conjuncts(variant):
                    return SemanticValidationDecision(
                        False,
                        "entity_role:coordinated_object_requires_split",
                        predicate,
                        locality_scope=locality_scope,
                        subject_family=source_family,
                        object_family=target_family,
                        reason_detail="coordinated object must be split into distinct semantic targets",
                    )

        roles_ok, role_reason = self._entity_role_valid(relation, source_entity, target_entity, validation_text)
        if not roles_ok:
            return SemanticValidationDecision(False, role_reason, predicate, locality_scope=locality_scope, subject_family=source_family, object_family=target_family, reason_detail=role_reason.replace(":", " "))

        if not self.domain_range_valid(relation, entities_by_uri):
            return SemanticValidationDecision(False, "domain_range:predicate_domain_range_invalid", predicate, locality_scope=locality_scope, subject_family=source_family, object_family=target_family, entity_role_check="pass", domain_range_check="fail", reason_detail="predicate domain/range incompatible with validated endpoint semantic families")

        predicate_surface_phrase, predicate_source = self.detect_predicate_phrase(predicate, validation_text)
        entailment_ok = self.predicate_locally_entailed(predicate, source_variants, target_variants, validation_text)
        if not predicate_source and entailment_ok:
            predicate_source = "strongly_entailed_semantic_mapping"
        if predicate in {"relatedTo", "associatedWith"} and predicate_source not in {
            "generic_meaningful_association",
            "explicit_lexical_relation",
        }:
            return SemanticValidationDecision(False, "predicate_ambiguity:generic_relation_not_justified", predicate, locality_scope=locality_scope, subject_family=source_family, object_family=target_family, entity_role_check="pass", domain_range_check="pass", entailment_check="fail", reason_detail="generic predicate not justified by explicit local proposition")
        if not entailment_ok:
            return SemanticValidationDecision(False, "entailment:predicate_not_locally_entailed", predicate, locality_scope=locality_scope, subject_family=source_family, object_family=target_family, entity_role_check="pass", domain_range_check="pass", entailment_check="fail", reason_detail="predicate not supportable from sentence-local proposition")

        if not predicate_source:
            return SemanticValidationDecision(False, "predicate_ambiguity:predicate_source_unresolved", predicate, locality_scope=locality_scope, subject_family=source_family, object_family=target_family, entity_role_check="pass", domain_range_check="pass", entailment_check="pass", reason_detail="predicate support exists but policy could not classify it into explicit/canonical/entailed/generic precedence")

        if predicate_source == "generic_meaningful_association":
            meta["generic_predicate_fallback"] = True

        meta["canonical_predicate"] = predicate
        meta["predicate_surface_phrase"] = predicate_surface_phrase or predicate
        meta["predicate_policy_source"] = predicate_source
        meta["relation_origin"] = self.infer_relation_origin(relation)
        meta["direction_normalized"] = direction
        meta["locality_scope"] = locality_scope
        meta["subject_surface_form"] = str(source_variants[0] if source_variants else getattr(source_entity, "label", ""))
        meta["object_surface_form"] = str(target_variants[0] if target_variants else getattr(target_entity, "label", ""))
        meta["subject_canonical_entity"] = str(getattr(source_entity, "uri", "") or "")
        meta["object_canonical_entity"] = str(getattr(target_entity, "uri", "") or "")
        meta["subject_family"] = source_family
        meta["object_family"] = target_family
        meta["entity_role_check"] = "pass"
        meta["domain_range_check"] = "pass"
        meta["entailment_check"] = "pass"
        meta["modality_check"] = "pass"
        relation.metadata = meta

        detail = (
            f"{predicate_source} predicate '{predicate_surface_phrase or predicate}'"
            f" -> {predicate}; locality={locality_scope}; direction={direction};"
            f" entity_role_check=pass; domain_range=pass; entailment=pass;"
            f" modality={meta.get('epistemic_status') or 'asserted'}"
        )
        return SemanticValidationDecision(
            True,
            "accepted",
            predicate,
            predicate_surface_phrase=predicate_surface_phrase or predicate,
            predicate_source=predicate_source,
            locality_scope=locality_scope,
            direction=direction,
            discourse_mode=str(meta.get("discourse_mode") or ""),
            epistemic_status=str(meta.get("epistemic_status") or ""),
            modality_tags=tuple(tag for tag in str(meta.get("modality_tags") or "").split("|") if tag),
            relation_origin=str(meta.get("relation_origin") or ""),
            subject_family=source_family,
            object_family=target_family,
            entity_role_check="pass",
            domain_range_check="pass",
            entailment_check="pass",
            modality_check="pass",
            reason_detail=detail,
        )
