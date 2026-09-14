import re
from typing import Any, Optional

# ---------------------------------------------------------
# Canonical predicate mapping (relation normalization)
# Keys: messy variants (LLM output, spacing, case, synonyms)
# Values: MUST be one of the object property names in add_schema.py
# ---------------------------------------------------------
CANONICAL_REL_PREDICATES = {
    # -------------------------
    # Generic / fallback
    # -------------------------
    "relatedto": "relatedTo",
    "related_to": "relatedTo",
    "related to": "relatedTo",
    "relatesto": "relatedTo",
    "relates_to": "relatedTo",
    "relates to": "relatedTo",
    "isrelatedto": "relatedTo",
    "is related to": "relatedTo",
    "linkedto": "relatedTo",
    "linked to": "relatedTo",
    "connectsto": "relatedTo",
    "connects to": "relatedTo",
    "connectedto": "relatedTo",
    "connected to": "relatedTo",
    "inrelationto": "relatedTo",
    "in relation to": "relatedTo",

    "associatedwith": "associatedWith",
    "associated with": "associatedWith",
    "associated_to": "associatedWith",
    "associatedto": "associatedWith",
    "reflects": "reflects",
    "reflect": "reflects",
    "reflected": "reflects",
    "mirrors": "reflects",
    "mirror": "reflects",

    # -------------------------
    # Discourse / structure
    # -------------------------
    "mentions": "mentions",
    "mention": "mentions",

    "answersquestion": "answersQuestion",
    "answers question": "answersQuestion",

    "authoredby": "authoredBy",
    "authored by": "authoredBy",
    "writtenby": "authoredBy",
    "written by": "authoredBy",

    "askedby": "askedBy",
    "asked by": "askedBy",

    "createdby": "createdBy",
    "created by": "createdBy",

    "repliesto": "repliesTo",
    "replies to": "repliesTo",
    "replyof": "repliesTo",
    "reply of": "repliesTo",

    "extractedfrom": "extractedFrom",
    "extracted from": "extractedFrom",
    "derivedfrom": "extractedFrom",
    "derived from": "extractedFrom",

    "belongstotheme": "belongsToTheme",
    "belongs to theme": "belongsToTheme",
    "belongsto": "belongsToTheme",
    "hastheme": "belongsToTheme",
    "has theme": "belongsToTheme",

    # Votes / interaction
    "hasvote": "hasVote",
    "has vote": "hasVote",
    "receivesvote": "receivesVote",
    "receives vote": "receivesVote",
    "votesforanswer": "receivesVote",
    "votes for answer": "receivesVote",
    "castby": "castBy",
    "cast by": "castBy",
    "castvote": "castVote",
    "cast vote": "castVote",

    # -------------------------
    # Stance / opinion
    # -------------------------
    "supports": "supports",
    "support": "supports",
    "endorses": "endorses",
    "endorse": "endorses",
    "approves": "endorses",
    "approve": "endorses",
    "criticizes": "criticizes",
    "criticise": "criticizes",
    "critiques": "criticizes",
    "disagreeswith": "disagreesWith",
    "disagrees with": "disagreesWith",
    #"opposes": "disagreesWith",
    "oppose": "disagreesWith",

    # LLM “free text” stance-ish verbs -> collapse to schema
    "discriminatesagainst": "particularlyAffects",
    "discriminates against": "particularlyAffects",
    "biasedagainst": "particularlyAffects",
    "biased against": "particularlyAffects",
    "performsworseon": "particularlyAffects",
    "performs worse on": "particularlyAffects",
    "performedworseon": "particularlyAffects",
    "performed worse on": "particularlyAffects",
    "worseon": "particularlyAffects",
    "worse on": "particularlyAffects",
    "worsefor": "particularlyAffects",
    "worse for": "particularlyAffects",
    # NOTE: "reinforces" intentionally has no entry. It is a specific verb,
    # not a generic relation phrase, so it must fall through to
    # normalize_predicate_strict() -> None and be preserved via the
    # local-unmapped-predicate path rather than collapsed to relatedTo.

    # -------------------------
    # Use / training / containment / domain
    # -------------------------
    "uses": "uses",
    "use": "uses",
    "utilises": "uses",
    "utilizes": "uses",
    "employs": "uses",
    "applies": "uses",

    "trainedon": "trainedOn",
    "trained on": "trainedOn",
    "trains on": "trainedOn",
    "trainingdata": "trainedOn",
    "training data": "trainedOn",
    "usedasproxyfor": "uses",
    "used as proxy for": "uses",

    # Defect-2 root-cause fix: these phrases used to map onto
    # "containsSubject", colliding with that name's own distinct,
    # pre-existing structural declaration (DiscourseUnit -> Entity). Actual
    # usage is descriptive/quantitative resource-content text, not an
    # entity reference -- mapped to its own name, registered
    # literal_allowed=True in utils/semantic_relation_policy.py, instead.
    "contains": "hasSubjectDescription",
    "containssubject": "hasSubjectDescription",
    "contains subject": "hasSubjectDescription",
    "hassubject": "hasSubjectDescription",
    "has subject": "hasSubjectDescription",
    "hosted by": "hostedBy",
    "hosted on": "hostedBy",
    "hosted in": "hostedBy",
    "available at": "hostedBy",
    "found on": "hostedBy",
    "found in": "hostedBy",
    "listed on": "hostedBy",

    "partofdomain": "partOfDomain",
    "part of domain": "partOfDomain",
    "indomain": "partOfDomain",
    "in domain": "partOfDomain",
    "partof": "partOfDomain",
    "part of": "partOfDomain",
    "occursin": "partOfDomain",
    "occurs in": "partOfDomain",
    "usedin": "partOfDomain",
    "used in": "partOfDomain",

    # NOTE: "provides" and "developed" intentionally have no entries here.
    # They are specific verbs, not generic relation phrases; leaving them
    # unmapped routes them through the local-unmapped-predicate path
    # instead of collapsing them to relatedTo.

    # -------------------------
    # Risk relations
    # -------------------------
    "hasrisk": "hasRisk",
    "has risk": "hasRisk",
    "posesrisk": "hasRisk",
    "poses risk": "hasRisk",
    "introducesrisk": "hasRisk",
    "introduces risk": "hasRisk",

    "exhibitsrisk": "exhibitsRisk",
    "exhibits risk": "exhibitsRisk",
    "showsrisk": "exhibitsRisk",
    "shows risk": "exhibitsRisk",

    "particularlyaffects": "particularlyAffects",
    "particularly affects": "particularlyAffects",
    "disproportionatelyaffects": "particularlyAffects",
    "disproportionately affects": "particularlyAffects",

    # -------------------------
    # Causality / impact
    # -------------------------
    "causes": "causes",
    "cause": "causes",
    "leads to": "causes",
    "leadsto": "causes",
    "results in": "causes",
    "resultsin": "causes",
    "brings about": "causes",
    "creates": "causes",
    "triggers": "causes",
    "induces": "causes",

    "affects": "affects",
    "affect": "affects",
    "impacts": "affects",
    "impact": "affects",
    "influences": "affects",
    "influence": "affects",
    "harms": "affects",
    "harm": "affects",
    "contaminate": "affects",
    "contaminates": "affects",
    "contaminated": "affects",
    "contaminating": "affects",

    "mitigatedby": "mitigatedBy",
    "mitigated by": "mitigatedBy",
    "mitigatedBy": "mitigatedBy",
    "mitigate": "mitigatedBy",
    "reducedby": "mitigatedBy",
    "reduced by": "mitigatedBy",
    "addressedby": "mitigatedBy",
    "addressed by": "mitigatedBy",

    # Backward-compat
    "causedby": "causedBy",
    "caused by": "causedBy",
    "affectedby": "affectedBy",
    "affected by": "affectedBy",
    "chargedhigherfares": "affectedBy",
    "charged higher fares": "affectedBy",
    "paysmorethan": "affectedBy",
    "pays more than": "affectedBy",

    # -------------------------
    # Risk assessment bundle
    # -------------------------
    "assessesrisk": "assessesRisk",
    "assesses risk": "assessesRisk",
    "assessedby": "assessedBy",
    "assessed by": "assessedBy",
    "usesevidence": "usesEvidence",
    "uses evidence": "usesEvidence",
    "hasseverity": "hasSeverity",
    "has severity": "hasSeverity",
    "haslikelihood": "hasLikelihood",
    "has likelihood": "hasLikelihood",
    "hasrisklevel": "hasRiskLevel",
    "has risk level": "hasRiskLevel",

    # Evidence-ish free text -> collapse to schema
    "documentedby": "usesEvidence",
    "documented by": "usesEvidence",
    "revealed": "usesEvidence",
    "found": "usesEvidence",

    # -------------------------
    # Governance instrument references
    # -------------------------
    "hasregulatoryreference": "hasRegulatoryReference",
    "has regulatory reference": "hasRegulatoryReference",
    "referencesinstrument": "referencesInstrument",
    "references instrument": "referencesInstrument",
    "references governance instrument": "referencesInstrument",
    "references law": "referencesInstrument",
    "references standard": "referencesInstrument",
    "referencessource": "referencesSource",
    "references source": "referencesSource",
    "cites": "cites",
    "cite": "cites",
    "accordingto": "cites",
    "according to": "cites",
    "publishedin": "referencesSource",
    "published in": "referencesSource",

    # ---------------------------------------------------------
    # Map non-schema / noisy LLM predicates to canonical schema predicates
    # ---------------------------------------------------------
    # NOTE: "similar to" / "overlaps with" intentionally have no entries.
    # They are not semantically equivalent to relatedTo/associatedWith;
    # leaving them unmapped routes them through the local-unmapped-predicate
    # path instead of collapsing them to relatedTo.

    "agreeswith": "supports",
    "agrees with": "supports",

    "indirectlycontradicts": "disagreesWith",
    "indirectly contradicts": "disagreesWith",

    # NOTE: "located in" intentionally has no entry. It is not semantically
    # equivalent to relatedTo; leaving it unmapped routes it through the
    # local-unmapped-predicate path instead of collapsing it to relatedTo.

    "isdatasetfor": "trainedOn",
    "is dataset for": "trainedOn",

    # Governance reasoning (new schema predicates)
    "complieswith": "compliesWith",
    "complies with": "compliesWith",
    "requirescompliancewith": "requiresComplianceWith",
    "requires compliance with": "requiresComplianceWith",
    "requires": "requires",
    "violates": "violates",
    "mitigates": "mitigates",
    "opposes": "opposes",

    # ---------------------------------------------------------
    # Structural / containment relations (schema-aligned)
    # ---------------------------------------------------------
    "includes": "includes",
    "include": "includes",
    "includedin": "includes",
    "included in": "includes",

    # Two new predicates justified by repeated is/has
    # fallback audit evidence -- see utils/semantic_relation_policy.py.
    "hasdescription": "hasDescription",
    "has description": "hasDescription",
    "definedas": "hasDescription",
    "defined as": "hasDescription",
    "described as": "hasDescription",
    "hasattribute": "hasAttribute",
    "has attribute": "hasAttribute",
    "hascharacteristic": "hasAttribute",
    "has characteristic": "hasAttribute",

    # Defect-2 root-cause fix: self-referential compact-key entry for
    # hasSubjectDescription itself, matching the pattern above for
    # hasDescription/hasAttribute. Without this, normalize_predicate()
    # given the predicate's OWN camelCase name (not a natural-language
    # phrase) falls through to the "relatedTo" default, which then made
    # SemanticRelationPolicy's alias-table construction (which calls
    # normalize_predicate() on every registered predicate's own name to
    # seed its self-mapping) clobber "relatedTo"'s self-mapping with
    # "hasSubjectDescription" instead -- silently rewriting every
    # unrelated unmapped predicate to hasSubjectDescription.
    "hassubjectdescription": "hasSubjectDescription",

    # Same self-reference requirement as hasSubjectDescription above --
    # without this, SemanticRelationPolicy.__init__'s alias-seeding loop
    # would clobber the shared _alias_to_predicate["relatedTo"] mapping
    # the first time hostedBy is registered.
    "hostedby": "hostedBy",

}


# ---------------------------------------------------------------------------
# E: local (unmapped) predicate cleanup -- deterministic, domain-agnostic.
#
# Applies only to predicates that have NO canonical schema mapping and are
# being kept as local_unmapped. It never touches a mapped predicate, never
# lowers a mapping threshold, and never invents a mapping.
# ---------------------------------------------------------------------------

# Closed, domain-neutral set of English function words: articles,
# demonstratives, prepositions, conjunctions, pronouns, auxiliaries/modals,
# and bare negators/degree words. A predicate whose every token is in this
# set carries no relational content of its own. NOT a domain lexicon.
_PREDICATE_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "such", "same", "other",
    "of", "for", "to", "in", "on", "at", "by", "with", "from", "into", "onto",
    "about", "over", "under", "above", "below", "between", "among", "through",
    "during", "before", "after", "within", "without", "against", "toward",
    "towards", "per", "via", "up", "down", "out", "off", "as",
    "and", "or", "nor", "but", "so", "yet", "than", "then", "because", "if",
    "it", "its", "they", "them", "their", "he", "she", "him", "her", "his",
    "we", "us", "our", "you", "your", "i", "me", "my", "who", "whom", "whose",
    "which", "what", "there", "here",
    "be", "is", "are", "was", "were", "been", "being", "am",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "ought",
    "not", "no", "nor", "yes", "very", "too", "also", "just", "only", "more",
    "most", "less", "least", "much", "many", "each", "every", "some", "any",
    "all", "both", "few", "several",
})


def _split_predicate_words(name: Any) -> list:
    """Split a raw predicate string (camelCase / snake_case / kebab-case /
    spaced / punctuated) into an ordered list of lowercase word tokens."""
    raw = str(name or "").strip()
    if not raw:
        return []
    # split camelCase / PascalCase boundaries
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    tokens = re.split(r"[^A-Za-z0-9]+", spaced)
    return [t.lower() for t in tokens if t]


def _fold_regular_inflection(word: str) -> str:
    """Conservative, non-stemming fold of a SINGLE regular English
    plural / 3rd-person inflection ('-s' / '-es' / '-ies').

    Deliberately does NOT touch '-ed' (past tense/participle cannot be
    reversed without POS knowledge -- e.g. 'used' -> 'us', 'analyzed' ->
    'analyz') nor '-ing' (the silent-'e' restoration -- 'making' -> 'make'
    vs 'showing' -> 'show' -- is not decidable deterministically). Never
    shortens a stem below 3 characters."""
    w = str(word or "").lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    # regular "-es" added after a sibilant ('touches' -> 'touch')
    if w.endswith(("sses", "shes", "ches", "xes", "zes")) and len(w) > 4:
        return w[:-2]
    # regular "-es" added after an "-o" stem ('goes' -> 'go', 'echoes' -> 'echo')
    if w.endswith("oes") and len(w) >= 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "as", "os")) and len(w) > 3:
        return w[:-1]
    return w


def normalize_local_predicate(name: Any) -> str:
    """Deterministic canonical form for a LOCAL (unmapped) predicate:
    casing + punctuation normalization, and a single conservative
    regular-inflection fold on the final word only. Produces a stable
    camelCase local name so that e.g. 'hasfeatures'/'hasFeatures',
    'ispartof'/'isPartOf', 'show'/'shows'/'showing', 'leadTo'/'leadingTo'
    collapse to one predicate. Never merges predicates that differ by a
    content word, and never applies aggressive stemming."""
    words = [_fold_regular_inflection(w) for w in _split_predicate_words(name)]
    words = [w for w in words if w]
    if not words:
        return ""
    return words[0] + "".join(w[:1].upper() + w[1:] for w in words[1:])


def is_malformed_local_predicate(name: Any) -> bool:
    """True only when a predicate is DETERMINISTICALLY not a valid relation
    predicate: it is empty, or every one of its tokens is an English
    function word (article / preposition / conjunction / pronoun /
    auxiliary / bare negator). Anything with at least one content token --
    including a nominalisation or a clause-shaped phrase -- is left
    UNRESOLVED rather than judged malformed here."""
    words = _split_predicate_words(name)
    if not words:
        return True
    return all(w in _PREDICATE_FUNCTION_WORDS for w in words)


def predicate_label_from_local_name(name: Any) -> str:
    """Human-readable rdfs:label for a local predicate: its tokens,
    space-joined, lower-case (matches the style of the fixed schema's
    property labels such as 'has description', 'hosted by')."""
    words = _split_predicate_words(name)
    return " ".join(words)


def normalize_predicate_strict(pred: str) -> Optional[str]:
    """
    Look up a raw predicate phrase in CANONICAL_REL_PREDICATES without ever
    defaulting to "relatedTo". Returns None when the phrase has no explicit
    entry in the table, so callers can distinguish "recognized" (including
    a genuine generic-relation match) from "simply unrecognized" instead of
    having both collapse to the same value.
    """
    if not pred:
        return None

    key = pred.strip().lower()

    # normalize separators (handles "chargedHigherFares", "charged_higher_fares", etc.)
    key = re.sub(r"[\s\-]+", " ", key)   # unify whitespace/dashes
    key = key.replace("_", " ")          # unify underscores
    key = key.strip()

    # try exact
    if key in CANONICAL_REL_PREDICATES:
        return CANONICAL_REL_PREDICATES[key]

    # try compact key (remove spaces)
    compact = key.replace(" ", "")
    if compact in CANONICAL_REL_PREDICATES:
        return CANONICAL_REL_PREDICATES[compact]

    return None


def normalize_predicate(pred: str) -> str:
    """
    Production rule:
    - return ONLY schema-approved property names (values of CANONICAL_REL_PREDICATES)
    - never return novel predicates (prevents ontology explosion in streaming)
    - unrecognized predicates default to "relatedTo" here ONLY. Callers that
      need to preserve unrecognized-but-specific predicates for
      local-unmapped handling (rather than losing them to a generic
      fallback) should use normalize_predicate_strict() instead.
    """
    return normalize_predicate_strict(pred) or "relatedTo"
