"""
pipeline.py
-----------
Main pipeline orchestrator for the Semantic Knowledge Graph.
Coordinates all modules and handles batch processing.
"""

import gc
import time
import re
import json
import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Union, Set, Tuple
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal, Namespace, RDF

from models import Entity, Relation, ProcessingResult, AnnotationBatch
from config import Config
from extractors.entity_extractor import LLMEntityExtractor
from extractors.llm_relation_extractor import LLMRelationExtractor
from extractors.dataset_family_classifier import DatasetFamilyTypeClassifier
from extractors.population_group_classifier import PopulationGroupTypeClassifier
from extractors.ai_system_family_classifier import AiSystemFamilyTypeClassifier
from extractors.predicate_refinement_classifier import PredicateRefinementClassifier
from ontology.mapper import OntologyMapper
from utils.helpers import (
    GlobalEntityRegistry,
    dedup_mentions,
    CoreferenceResolver,
    safe_process_with_retry,
    batch_iterator,
    cleanup_memory,
    save_json,
    load_json,
    normalize_text,
    make_uri_safe,
    expand_curie,
    make_relation,
    is_finite_clause_fragment
)
from reasoning.reasoner import MultiHopReasoner, ConflictDetector
from metrics.quality_metrics import QualityMetricsComputer
from utils.graphdb_client import GraphDBClient
from serializers.rdf_serializer import RDFSerializer
from ontology.add_schema import canonicalize_class_local
from utils.semantic_chunker import SemanticChunker
from validation.shacl_validator import SHACLValidator
from utils.entity_normalizer import EntityNormalizer
from utils.wikidata_linker import WikidataLinker
from utils.entity_consolidator import EntityConsolidator
from utils.canonical_linker import CanonicalLinker
from utils.quantity_literal_classifier import classify_quantity_candidate, classify_date_candidate
from utils.semantic_quality_events import record_event
from domain_profiles import matches_semantic_category, CORE_GENERIC_FINDING_CONTEXT_NOUNS, classify_entity_semantic_family
from utils.predicate_precision import PredicatePrecisionLayer
from utils.semantic_relation_policy import (
    SemanticRelationPolicy,
    SEMANTIC_RELATION_PREDICATES,
    STRUCTURAL_RELATION_PREDICATES,
    PREDICATE_REGISTRY,
    literal_datatype_permitted,
)
from utils.predicate_map import normalize_predicate_strict


ETD = Namespace(Config.ETD_NS)
logger = logging.getLogger(__name__)

SEMANTIC_RELATION_FIELDS = set(getattr(Config, "SEMANTIC_RELATION_FIELDS", set()) or set())

def _make_rel_uri(suffix: str) -> str:
    """Helper to create a deterministic relation URI."""
    return str(ETD[f"rel_{suffix}"])


def make_uri(local_id: str) -> str:
    """Ensure all KG IDs become absolute ETD URIs."""
    return str(ETD[local_id])


def normalize_relation_label(label: str) -> str:
    if not label:
        return ""

    # clean whitespace
    label = label.strip()

    # Convert internal CamelCase â†’ spaced ("causedBy" â†’ "caused By")
    label = re.sub(r'(?<!^)(?=[A-Z])', ' ', label)

    # Split on spaces/underscores
    parts = re.split(r"[ _]+", label)
    parts = [p.lower() for p in parts if p]

    if not parts:
        return ""

    return parts[0] + "".join(x.capitalize() for x in parts[1:])

def _mk_evidence_text(*, text: Optional[str], fallback: str, max_len: int = 500) -> str:
    """
    Always return a non-empty evidence string (traceability safe).
    """
    if text is not None:
        s = str(text).strip()
        if s:
            return s[:max_len]
    return fallback[:max_len]


def make_light_relation(
    source: str,
    rel: str,
    target: str,
    confidence: float = 0.60,
    *,
    discourse_id: Optional[str] = None,
    evidence: Optional[str] = None,
):

    """
    Backwards-compatible wrapper.
    Production polish: deterministic URI via make_relation() (no UUIDs).
    """
    # Ensure URIs are absolute
    if not str(source).startswith("http"):
        source = make_uri(str(source))
    if not str(target).startswith("http"):
        target = make_uri(str(target))

    return make_relation(
        source,
        rel,
        target,
        discourse_id=discourse_id,
        extractor="lightweight",
        context_source="light_relation",
        confidence=float(confidence or 0.0),
        evidence=(evidence or ""),
        is_inferred=False,
    )


class EnhancedSemanticPipeline:
    """Main pipeline for processing documents into knowledge graph."""

    @property
    def domain_profile(self):
        """The active DomainProfile (see domain_profiles.py). Lazily
        resolved from Config.DOMAIN_PROFILE on first access so that both
        normal construction (via __init__) and test fixtures that bypass
        __init__ (via EnhancedSemanticPipeline.__new__) get the same
        backward-compatible default (the "ai_ethics" profile) without
        either needing to set it explicitly."""
        profile = getattr(self, "_domain_profile", None)
        if profile is None:
            from domain_profiles import get_active_profile
            profile = get_active_profile()
            self._domain_profile = profile
        return profile

    @domain_profile.setter
    def domain_profile(self, value):
        self._domain_profile = value

    @property
    def quality_events(self):
        """Semantic-quality instrumentation event sink (see
        utils/semantic_quality_events.py). Lazily created on first access
        -- same rationale as domain_profile above -- so every guard below
        can unconditionally call record_event(self.quality_events, ...)
        regardless of whether this instance went through __init__ or was
        built via EnhancedSemanticPipeline.__new__ (the pattern used
        throughout tests/)."""
        events = getattr(self, "_quality_events", None)
        if events is None:
            events = []
            self._quality_events = events
        return events

    @quality_events.setter
    def quality_events(self, value):
        self._quality_events = value

    @property
    def _chunk_entity_identity_index(self):
        """(discourse_id, normalized_label) -> Entity, used by
        _resolve_surface_entity to reuse an already-minted entity for an
        identically-labeled mention within the same discourse instead of
        minting a second, differently-typed duplicate. Deliberately
        scoped to one discourse -- never broader -- so two unrelated
        documents using the same generic label are never merged by this
        mechanism. Lazily created, same rationale as quality_events."""
        index = getattr(self, "_chunk_entity_identity_index_store", None)
        if index is None:
            index = {}
            self._chunk_entity_identity_index_store = index
        return index

    def _matches_semantic_category(
        self,
        *,
        category: str,
        text: str,
        uri: Optional[str] = None,
        local_entities: Optional[List[Entity]] = None,
    ) -> bool:
        """Thin wrapper around the ONE shared category-membership check
        (domain_profiles.matches_semantic_category) -- this method's only
        job is resolving `entity_type` from `uri`/`local_entities` (the
        plumbing specific to this class), then delegating. See that
        function's docstring for the full priority order and rationale.
        Every other consumer of this mechanism (e.g.
        utils/semantic_relation_policy.py) calls the same shared function
        directly -- there is exactly one copy of the category-matching
        logic and of every category's vocabulary.
        """
        entity_type = None
        if uri and local_entities:
            uri_str = str(uri).strip()
            if uri_str.startswith("http"):
                ent = next(
                    (e for e in local_entities if str(getattr(e, "uri", "") or "").strip() == uri_str),
                    None,
                )
                if ent is not None:
                    entity_type = str(getattr(ent, "entity_type", "") or "") or None
        return matches_semantic_category(self.domain_profile, category, text, entity_type=entity_type)

    def __init__(self, config=None):
        self.config = config if config else Config
        if not getattr(self.config, "USE_LLM_EXTRACTION", True) or getattr(self.config, "USE_HYBRID_EXTRACTION", False):
            raise RuntimeError(
                "LLM-only extraction enforced: USE_LLM_EXTRACTION must be true and "
                "USE_HYBRID_EXTRACTION must be false."
            )
        
        print("\nðŸ”§ Initializing pipeline components...")
        self.global_registry = GlobalEntityRegistry()
        # Explicit here for readability; both are lazily created by their
        # own @property below if some caller (e.g. a test fixture) builds
        # an instance via __new__ and never runs this __init__ at all.
        self._quality_events = []
        self._chunk_entity_identity_index_store = {}

        # -------------------------------------------------------
        #  Wikidata linker (REAL SPARQL version)
        # -------------------------------------------------------
        # Use configured SPARQL endpoint if available, else default
        endpoint = getattr(
            self.config,
            "WIKIDATA_SPARQL_ENDPOINT",
            "https://query.wikidata.org/sparql",
        )

        # Initialise Wikidata linker only if enabled
        self.wikidata_linker = None
        if getattr(self.config, "USE_WIKIDATA_LINKING", False):
            self.wikidata_linker = WikidataLinker(
                endpoint=endpoint,
                language="en",
                max_results=5,
                min_score=0.65,
                user_agent="ETD-Hub-KG-Pipeline/1.0 (mailto:your-email@example.com)",
                timeout=15,
            )


        # ðŸ”§ Entity normalizer for canonical labels
        self.entity_normalizer = EntityNormalizer(self.global_registry)
        self.canonical_linker = CanonicalLinker(self.global_registry, output_dir=self.config.OUTPUT_DIR)
        self.semantic_policy = SemanticRelationPolicy()
        self.predicate_precision = PredicatePrecisionLayer(self.semantic_policy)

        # New LLM-based entity extractor
        self.entity_extractor = LLMEntityExtractor(
            ontology_terms=self.config.ONTOLOGY_TERMS,
            model=self.config.LLM_MODEL
        )

        self.llm_relation_extractor = LLMRelationExtractor(
            #api_key=self.config.load_llm_api_key(),
            #model=config.LLM_MODEL,
            model=self.config.LLM_MODEL,
            max_retries=3,
            relation_confidence_threshold=0.35
            #max_retries=self.config.LLM_MAX_RETRIES
        )   
        
        # Narrowly-scoped LLM classifier for Dataset-family
        # ambiguity resolution (Dataset vs ResearchStudy/Analysis/
        # Publication/SoftwareProduct/Resource/Concept/Ambiguous). Optional
        # by construction -- callers must use getattr(..., None) so a
        # pipeline built without __init__ (the established test-harness
        # pattern in this codebase) degrades gracefully to a conservative
        # deterministic-only abstention instead of raising.
        self.dataset_family_classifier = DatasetFamilyTypeClassifier(model=self.config.LLM_MODEL)
        self.population_group_classifier = PopulationGroupTypeClassifier(model=self.config.LLM_MODEL)
        self.ai_system_family_classifier = AiSystemFamilyTypeClassifier(model=self.config.LLM_MODEL)
        self.predicate_refinement_classifier = PredicateRefinementClassifier(model=self.config.LLM_MODEL)

        self.ontology_mapper = OntologyMapper()
        self.coref_resolver = CoreferenceResolver()
        self.reasoner = MultiHopReasoner(min_confidence=0.5)
        self.conflict_detector = ConflictDetector()
        self._discourse_chunk_spans: Dict[str, List[Dict[str, int]]] = {}
        self._provenance_diagnostics: Dict[str, Any] = {
            "missing_span_offsets": 0,
            "derived_span_from_entities": 0,
            "span_out_of_range": 0,
            "cross_chunk_evidence": 0,
            "unknown_chunk_index_count": 0,
            "inferred_relations_excluded_from_gate": 0,
            "non_textual_relations_excluded_from_gate": 0,
            "extracted_relations_checked": 0,
            "total_relations_checked": 0,
            "unknown_chunk_index_rate": 0.0,
            "unknown_chunk_index_gate_threshold": float(getattr(self.config, "MAX_UNKNOWN_CHUNK_INDEX_PCT", 0.02)),
            "strict_mode": bool(getattr(self.config, "STRICT_PROVENANCE_GROUNDING", False)),
            "gate_passed": True,
        }
        self._candidate_generation_diagnostics: List[Dict[str, Any]] = []
        
        print("âœ“ Pipeline initialized")

    def _get_semantic_policy(self) -> SemanticRelationPolicy:
        policy = getattr(self, "semantic_policy", None)
        if policy is None:
            policy = SemanticRelationPolicy()
            self.semantic_policy = policy
        return policy

    def _compose_origin_key(self, discourse_id: Any, field: Any = None) -> str:
        """
        Deterministic provenance identity:
        disambiguate reused discourse_id values across sections/types.
        """
        did = str(discourse_id).strip() if discourse_id is not None else ""
        fld = str(field).strip() if field is not None else ""
        if did and fld:
            return f"{fld}|{did}"
        return did

    def _field_allows_semantic_extraction(self, field: Any) -> bool:
        fld = str(field or "").strip()
        return bool(fld) and fld in SEMANTIC_RELATION_FIELDS

    def _section_span_for_offset(
        self,
        section_spans: List[Dict[str, Any]],
        start_char: Optional[int],
        end_char: Optional[int],
    ) -> Dict[str, Any]:
        if not section_spans:
            return {"section": "Body", "section_category": "semantic"}
        start = self._coerce_nonneg_int(start_char)
        end = self._coerce_nonneg_int(end_char)
        if start is None:
            return dict(section_spans[0])
        for span in section_spans:
            span_start = self._coerce_nonneg_int(span.get("start_offset"))
            span_end = self._coerce_nonneg_int(span.get("end_offset"))
            if span_start is None or span_end is None:
                continue
            if int(span_start) <= int(start) < int(span_end):
                return dict(span)
            if end is not None and int(span_start) < int(end) <= int(span_end):
                return dict(span)
        return dict(section_spans[0])

    def _annotate_entity_section_metadata(
        self,
        entity: Entity,
        *,
        section_spans: List[Dict[str, Any]],
    ) -> None:
        meta = getattr(entity, "metadata", None) or {}
        span = self._section_span_for_offset(section_spans, meta.get("start_char"), meta.get("end_char"))
        section = str(span.get("section") or "").strip()
        section_category = str(span.get("section_category") or "").strip()
        if section:
            meta["section"] = section
        if section_category:
            meta["section_category"] = section_category
        entity.metadata = meta

    def _local_entities_for_unit(
        self,
        entities: List[Entity],
        unit: Dict[str, Any],
        discourse_id: Optional[str],
    ) -> List[Entity]:
        unit_start = self._coerce_nonneg_int(unit.get("start_offset"))
        unit_end = self._coerce_nonneg_int(unit.get("end_offset"))
        if unit_start is None or unit_end is None or unit_end <= unit_start:
            return []

        target_section = str(unit.get("section") or "").strip()
        target_category = str(unit.get("section_category") or "").strip()
        selected: List[Tuple[Entity, bool]] = []
        seen: Set[str] = set()

        for entity in entities:
            meta = getattr(entity, "metadata", None) or {}
            if target_section and str(meta.get("section") or "").strip() and str(meta.get("section") or "").strip() != target_section:
                continue
            if target_category and str(meta.get("section_category") or "").strip() and str(meta.get("section_category") or "").strip() != target_category:
                continue
            entity_uri = str(getattr(entity, "uri", "") or "")
            if not entity_uri or entity_uri in seen:
                continue
            if self._entity_overlaps_span(entity, discourse_id, unit_start, unit_end):
                selected.append((entity, True))
                seen.add(entity_uri)

        anaphora = unit.get("anaphora")
        if isinstance(anaphora, dict):
            alias_surface = str(anaphora.get("surface_form") or "").strip()
            source_start = self._coerce_nonneg_int(anaphora.get("source_start_offset"))
            source_end = self._coerce_nonneg_int(anaphora.get("source_end_offset"))
            for entity in entities:
                entity_uri = str(getattr(entity, "uri", "") or "")
                if not entity_uri or entity_uri in seen:
                    continue
                meta = getattr(entity, "metadata", None) or {}
                if target_section and str(meta.get("section") or "").strip() and str(meta.get("section") or "").strip() != target_section:
                    continue
                if target_category and str(meta.get("section_category") or "").strip() and str(meta.get("section_category") or "").strip() != target_category:
                    continue
                if source_start is None or source_end is None:
                    continue
                if not self._entity_overlaps_span(entity, discourse_id, source_start, source_end):
                    continue
                self._merge_entity_alias(entity, alias_surface)
                meta["coreference_resolved"] = True
                meta["coreference_source"] = str(anaphora.get("source") or "preceding_sentence")
                entity.metadata = meta
                selected.append((entity, False))
                seen.add(entity_uri)

        ranked = self._prioritize_local_entities(selected, unit_text=str(unit.get("text") or ""))
        return ranked

    def _carry_forward_actor_for_unit(
        self,
        *,
        entities: List[Entity],
        unit: Dict[str, Any],
        discourse_id: Optional[str],
    ) -> Optional[Entity]:
        """Identify a single, unambiguous system/process actor established
        earlier in the SAME field that a later semantic unit may depend on
        without repeating it literally (e.g. an actor named in a field's
        opening description, referenced only implicitly in a later bullet).

        Deliberately conservative: only returns a candidate when EXACTLY ONE
        system-family entity was established strictly before this unit in the
        same discourse and semantic-section-category; two or more candidates
        (genuine ambiguity) or zero candidates both return None.
        """
        unit_start = self._coerce_nonneg_int(unit.get("start_offset"))
        if unit_start is None:
            return None
        target_category = str(unit.get("section_category") or "").strip()
        did = str(discourse_id or "").strip()

        candidates: List[Entity] = []
        seen_uris: Set[str] = set()
        for entity in entities:
            if self._entity_semantic_family(entity) != "system":
                continue
            meta = getattr(entity, "metadata", None) or {}
            if did and str(meta.get("discourse_id") or "").strip() not in {did, ""}:
                continue
            if target_category and str(meta.get("section_category") or "").strip() not in {target_category, ""}:
                continue

            earliest = None
            mentions = meta.get("mentions")
            if isinstance(mentions, list):
                starts = [
                    self._coerce_nonneg_int(m.get("start_char"))
                    for m in mentions
                    if isinstance(m, dict)
                ]
                starts = [s for s in starts if s is not None]
                if starts:
                    earliest = min(starts)
            if earliest is None:
                earliest = self._coerce_nonneg_int(meta.get("start_char"))
            if earliest is None or earliest >= unit_start:
                continue

            uri = str(getattr(entity, "uri", "") or "")
            if not uri or uri in seen_uris:
                continue
            seen_uris.add(uri)
            candidates.append(entity)

        if len(candidates) != 1:
            return None
        return candidates[0]

    # -----------------------------------------------------------------
    # Deterministic relation-endpoint coreference resolution
    #
    # Purely structural/grammatical detection (closed English pronoun
    # class + generic "light noun" phrases such as "this tool"/"these
    # frameworks") -- no dataset IDs, entity names, or domain vocabulary.
    # The light-noun set and the family hints below are linguistic
    # categories (the same ones apply to "this drug" in a medical corpus
    # or "this recipe" in a cooking corpus), not AI-ethics-specific.
    # -----------------------------------------------------------------
    _REFERENTIAL_PRONOUNS: Set[str] = {
        "this", "that", "these", "those", "it", "they", "them",
        "its", "their", "itself", "themselves", "he", "she", "him", "her",
    }
    # 1st/2nd-person pronouns: deliberately kept OUT of _REFERENTIAL_PRONOUNS
    # (unlike 3rd-person/demonstrative pronouns, "you"/"we"/"I" have no
    # textual antecedent to find within the discourse -- they name the
    # discourse participants themselves) -- so _is_referential_endpoint_text
    # / the antecedent-search path below is never asked to resolve one to a
    # nearby entity mention (which produced nonsense matches, e.g. "you"
    # latching onto an unrelated "Biases" entity purely by window
    # proximity). They still must not be silently fabricated into a
    # persistent entity when nothing else claims them, so they feed ONLY
    # _is_unresolved_coreference_candidate (the unresolved-endpoint
    # fabrication guard) below, exactly like _is_generic_demonstrative_phrase
    # already does.
    _PERSON_DEICTIC_PRONOUNS: Set[str] = {
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
    }
    _REFERENTIAL_LIGHT_NOUNS: Set[str] = {
        "tool", "tools", "model", "models", "system", "systems",
        "technology", "technologies", "algorithm", "algorithms",
        "approach", "approaches", "method", "methods", "methodology",
        "methodologies", "framework", "frameworks", "process", "processes",
        "practice", "practices", "thing", "things", "one", "ones",
        "type", "types", "kind", "kinds", "item", "items", "way", "ways",
        "case", "cases", "issue", "issues", "aspect", "aspects",
    }
    _LIGHT_NOUN_FAMILY_HINTS: Dict[str, Set[str]] = {
        "tool": {"system"}, "tools": {"system"},
        "model": {"system"}, "models": {"system"},
        "system": {"system"}, "systems": {"system"},
        "technology": {"system"}, "technologies": {"system"},
        "algorithm": {"system"}, "algorithms": {"system"},
        "approach": {"system", "policy", "concept"}, "approaches": {"system", "policy", "concept"},
        "method": {"system", "policy", "concept"}, "methods": {"system", "policy", "concept"},
        "methodology": {"system", "policy", "concept"}, "methodologies": {"system", "policy", "concept"},
        "framework": {"policy", "system", "concept"}, "frameworks": {"policy", "system", "concept"},
        "process": {"system", "policy", "concept"}, "processes": {"system", "policy", "concept"},
        "practice": {"policy", "concept"}, "practices": {"policy", "concept"},
    }

    @classmethod
    def _referential_endpoint_light_noun(cls, text: str) -> Optional[str]:
        """Return the light noun if `text` is a deictic-determiner + generic
        -noun phrase (e.g. "this tool" -> "tool"), else None."""
        norm = " ".join(str(text or "").strip().lower().split())
        tokens = norm.split()
        if len(tokens) == 2 and tokens[0] in {"this", "that", "these", "those"} and tokens[1] in cls._REFERENTIAL_LIGHT_NOUNS:
            return tokens[1]
        return None

    @classmethod
    def _is_referential_endpoint_text(cls, text: str) -> bool:
        """True when `text` is a bare pronoun or a deictic light-noun phrase
        -- referential/anaphoric rather than a standalone named entity."""
        norm = " ".join(str(text or "").strip().lower().split())
        if not norm:
            return False
        if norm in cls._REFERENTIAL_PRONOUNS:
            return True
        return cls._referential_endpoint_light_noun(norm) is not None

    # A demonstrative-determiner-led phrase with NO internal
    # proper-noun capitalization (e.g. "this disagreement", "these tests",
    # "this policy basis") is structurally referential-shaped -- standing
    # in for previously-established discourse content -- even when its
    # head noun isn't one of the curated light nouns below. This is a
    # purely orthographic/grammatical signal (no noun vocabulary, no
    # dataset/domain terms) used ONLY by the unresolved-endpoint fabrication
    # guard below; the antecedent-search algorithm
    # (_resolve_anaphoric_endpoint_antecedent) is unchanged and continues
    # to use _is_referential_endpoint_text alone.
    _DEMONSTRATIVE_DETERMINERS: Set[str] = {"this", "that", "these", "those"}

    @classmethod
    def _is_generic_demonstrative_phrase(cls, text: str) -> bool:
        tokens = str(text or "").strip().split()
        if len(tokens) < 2:
            return False
        if tokens[0].strip(".,;:").lower() not in cls._DEMONSTRATIVE_DETERMINERS:
            return False
        for word in tokens[1:]:
            core = word.strip(".,;:()[]\"'")
            if core and core[0].isupper():
                return False  # a capitalized remainder reads as a proper-noun name
        return True

    @classmethod
    def _is_person_deictic_pronoun(cls, text: str) -> bool:
        norm = " ".join(str(text or "").strip().lower().split())
        return bool(norm) and norm in cls._PERSON_DEICTIC_PRONOUNS

    # Root-cause fix (bounded correction pass, item 5): "there" as a BARE,
    # single-token captured surface form (e.g. "There" standing in for
    # "there's a practical answer") is the expletive/dummy-subject sense,
    # structurally identical in role to expletive "it" (see
    # _is_expletive_it_construction) -- it has no antecedent to search
    # for, unlike genuine 3rd-person/demonstrative pronouns in
    # _REFERENTIAL_PRONOUNS. Kept as its own tiny set (not folded into
    # _PERSON_DEICTIC_PRONOUNS, which is a different grammatical category)
    # so this stays a narrow, explicit, single-purpose check rather than
    # widening an existing set's meaning. A locative "there" ("living
    # there", "moved there") is captured as part of a longer phrase by
    # the extractor, not as this exact bare single-word surface form, so
    # this check does not need positional/context disambiguation the way
    # expletive "it" does.
    _EXPLETIVE_ONLY_TOKENS: Set[str] = {"there"}

    @classmethod
    def _is_expletive_only_token(cls, text: str) -> bool:
        norm = " ".join(str(text or "").strip().lower().split())
        return norm in cls._EXPLETIVE_ONLY_TOKENS

    @classmethod
    def _is_unresolved_coreference_candidate(cls, text: str) -> bool:
        """Combined structural signal for the unresolved-endpoint
        fabrication guard only: true when `text` is shaped like an unresolved discourse
        reference, OR a finite clause/speaker-intent fragment (manual
        semantic-QA pass, category C -- e.g. "I Think There Needs"), rather
        than a genuine standalone entity. Includes 1st/2nd-person pronouns
        (see _PERSON_DEICTIC_PRONOUNS) even though they are excluded from
        _is_referential_endpoint_text -- they must still never be
        fabricated into a persistent entity, they are just never
        antecedent-search-eligible."""
        return (
            cls._is_referential_endpoint_text(text)
            or cls._is_generic_demonstrative_phrase(text)
            or cls._is_person_deictic_pronoun(text)
            or cls._is_expletive_only_token(text)
            or is_finite_clause_fragment(text)
        )

    @classmethod
    def _unresolved_referential_reason(cls, text: str) -> str:
        """Coarse, general category for an endpoint the unresolved-endpoint
        guard blocked because no earlier coreference/possessive/propositional
        resolution step could resolve it -- reuses existing detectors,
        adds no new lexical/blacklist rules."""
        norm = " ".join(str(text or "").strip().lower().split())
        if cls._referential_endpoint_light_noun(norm) is not None:
            return "unresolved_deictic_noun_phrase"
        if norm in getattr(cls, "_PROPOSITIONAL_CANDIDATE_PRONOUNS", set()):
            return "unresolved_propositional_or_nominal_pronoun"
        if cls._is_generic_demonstrative_phrase(text):
            return "unresolved_generic_demonstrative_phrase"
        if cls._is_expletive_only_token(text):
            return "unresolved_expletive_token"
        return "unresolved_referential_pronoun"

    @staticmethod
    def _earliest_entity_mention_start(entity: Entity) -> Optional[int]:
        meta = getattr(entity, "metadata", None) or {}
        starts: List[int] = []
        mentions = meta.get("mentions")
        if isinstance(mentions, list):
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                s = m.get("start_char")
                try:
                    iv = int(s)
                except (TypeError, ValueError):
                    continue
                if iv >= 0:
                    starts.append(iv)
        if starts:
            return min(starts)
        raw = meta.get("start_char")
        try:
            iv = int(raw)
        except (TypeError, ValueError):
            return None
        return iv if iv >= 0 else None

    def _entity_compatible_with_light_noun(self, entity: Entity, light_noun: str) -> bool:
        hints = self._LIGHT_NOUN_FAMILY_HINTS.get(light_noun)
        if not hints:
            return True
        return self._entity_semantic_family(entity) in hints

    @staticmethod
    def _entity_mention_spans_for_discourse(entity: Entity, discourse_id: str) -> List[Tuple[int, int]]:
        """All (start,end) mention spans this entity has within one
        discourse -- general span data already tracked on every entity,
        not tied to any dataset/domain."""
        meta = getattr(entity, "metadata", None) or {}
        spans: List[Tuple[int, int]] = []
        mentions = meta.get("mentions")
        if isinstance(mentions, list):
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                if discourse_id and str(m.get("discourse_id") or "").strip() not in {discourse_id, ""}:
                    continue
                try:
                    s = int(m.get("start_char"))
                    e = int(m.get("end_char"))
                except (TypeError, ValueError):
                    continue
                if e > s >= 0:
                    spans.append((s, e))
        if not spans:
            # Fallback to the entity's single primary span, but only when
            # the entity itself belongs to this discourse -- otherwise this
            # would silently leak a cross-discourse span into the overlap
            # check whenever its `mentions` list happens to be empty/absent.
            if discourse_id and str(meta.get("discourse_id") or "").strip() not in {discourse_id, ""}:
                return spans
            try:
                s = int(meta.get("start_char"))
                e = int(meta.get("end_char"))
                if e > s >= 0:
                    spans.append((s, e))
            except (TypeError, ValueError):
                pass
        return spans

    def _candidate_mentions_overlap(self, a: Entity, b: Entity, discourse_id: str) -> bool:
        spans_a = self._entity_mention_spans_for_discourse(a, discourse_id)
        spans_b = self._entity_mention_spans_for_discourse(b, discourse_id)
        for sa, ea in spans_a:
            for sb, eb in spans_b:
                if sa < eb and sb < ea:
                    return True
        return False

    def _are_duplicate_candidate_mentions(self, a: Entity, b: Entity, discourse_id: str) -> bool:
        """True when `a` and `b` are clearly overlapping/nested mentions of
        the SAME referent (e.g. "Students" and "Some Students"), not two
        genuinely different entities. Requires: same semantic family, the
        same discourse, overlapping mention spans, AND a normalized lexical
        containment or shared head-noun match -- overlap alone is not
        sufficient, since two different entities can legitimately overlap
        (e.g. a system and a risk it causes mentioned in the same clause)."""
        if self._entity_semantic_family(a) != self._entity_semantic_family(b):
            return False
        if not self._candidate_mentions_overlap(a, b, discourse_id):
            return False
        label_a = " ".join(str(getattr(a, "label", "") or "").strip().lower().split())
        label_b = " ".join(str(getattr(b, "label", "") or "").strip().lower().split())
        if not label_a or not label_b:
            return False
        if label_a == label_b or label_a in label_b or label_b in label_a:
            return True
        tokens_a = label_a.split()
        tokens_b = label_b.split()
        if tokens_a and tokens_b and tokens_a[-1] == tokens_b[-1]:
            return True  # shared head noun, e.g. "students" / "some students"
        return False

    def _collapse_duplicate_candidate_entities(
        self, candidates: Dict[str, Entity], discourse_id: str
    ) -> Dict[str, Entity]:
        """Collapse coreference candidates that are clearly duplicate/near
        -duplicate mentions of one referent, BEFORE ambiguity is counted.
        Local to candidate ranking only -- does not touch the global entity
        registry or canonical-linking merge behaviour. When two mentions
        collapse, the more informative/specific one (longer label) is kept."""
        items = list(candidates.items())
        if len(items) < 2:
            return candidates
        parent = {uri: uri for uri, _ in items}

        def find(x: str) -> str:
            while parent[x] != x:
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(items)):
            uri_a, ent_a = items[i]
            for j in range(i + 1, len(items)):
                uri_b, ent_b = items[j]
                if self._are_duplicate_candidate_mentions(ent_a, ent_b, discourse_id):
                    union(uri_a, uri_b)

        groups: Dict[str, List[Tuple[str, Entity]]] = {}
        for uri, ent in items:
            groups.setdefault(find(uri), []).append((uri, ent))

        collapsed: Dict[str, Entity] = {}
        for members in groups.values():
            best_uri, best_ent = max(members, key=lambda pair: len(str(getattr(pair[1], "label", "") or "")))
            collapsed[best_uri] = best_ent
        return collapsed

    @staticmethod
    def _endpoint_occurrence_offset(
        *,
        raw_text: str,
        unit: Dict[str, Any],
        evidence_text: Optional[str],
        unit_start: Optional[int],
    ) -> Optional[int]:
        """Absolute discourse offset of the endpoint phrase's OWN occurrence
        -- derived from the unit's offset plus the phrase's position inside
        the evidence text (preferred) or the unit's own text -- rather than
        the unit's start_offset. Returns None when it cannot be determined
        (missing unit text, or the phrase isn't literally found); callers
        must then fall back to the coarser unit-start cutoff rather than
        risk treating a later, cataphoric mention as preceding."""
        if unit_start is None:
            return None
        unit_text = str(unit.get("text") or "")
        norm_raw = str(raw_text or "").strip()
        if not unit_text or not norm_raw:
            return None
        pattern = re.compile(rf"(?<!\w){re.escape(norm_raw)}(?!\w)", re.IGNORECASE)

        ev = str(evidence_text or "").strip()
        if ev:
            local_base = unit_text.find(ev)
            if local_base != -1:
                match = pattern.search(ev)
                if match is not None:
                    return int(unit_start) + int(local_base) + int(match.start())

        match = pattern.search(unit_text)
        if match is None:
            return None
        return int(unit_start) + int(match.start())

    # -----------------------------------------------------------------
    # Clause-local candidate window for Tier 2 (occurrence-level,
    # same-unit) nominal anaphora resolution -- general orthographic/
    # structural signals only (sentence-final punctuation, paragraph/
    # newline breaks, quoted or parenthetical spans). No dataset/domain
    # vocabulary anywhere in this block.
    # -----------------------------------------------------------------
    _CLAUSE_BOUNDARY_RE = re.compile(r"[.!?]+(?=\s|$)|\r?\n+")
    _DOUBLE_QUOTE_SPAN_RE = re.compile(r'"[^"\n]*"')
    _SINGLE_QUOTE_SPAN_RE = re.compile(r"(?:^|(?<=\s))'[^'\n]{2,}?'(?=[\s,.;:!?]|$)")
    _PAREN_SPAN_RE = re.compile(r"\([^()\n]*\)")
    _BRACKET_SPAN_RE = re.compile(r"\[[^\[\]\n]*\]")
    # Nothing but a coordinating conjunction (and an optional comma) between
    # two entity mentions -- a general, vocabulary-free signal that both are
    # fragments of one coordinated NP ("perspectives and biases"), not two
    # independent, complete antecedent phrases.
    _COORDINATION_BETWEEN_RE = re.compile(r"\s*,?\s*(?:and|or)\s*", re.IGNORECASE)
    # Determiner-led noun phrase -- purely grammatical (closed-class
    # articles), never a domain/vocabulary term. Deliberately short (article
    # + 1-2 words) so it isolates a single NP rather than swallowing into a
    # following clause on a comma-less run-on sentence.
    _DETERMINER_NP_RE = re.compile(r"\b(?:the|a|an)\s+\w+(?:\s+\w+)?\b", re.IGNORECASE)
    # Bare pronouns for which a nearer, unrepresented local antecedent can
    # plausibly block a distant Tier-3 substitute. Deliberately
    # limited to singular pronouns: a singular "the/a/an + NP" is not a
    # plausible antecedent for a plural pronoun, so plural endpoints are
    # never subject to this check at all -- no morphological plural
    # detector needed.
    _SINGULAR_BLOCKABLE_PRONOUNS: Set[str] = {"it", "this", "that", "he", "she", "him", "her"}

    @classmethod
    def _clause_local_start(cls, unit_text: str, local_endpoint: int) -> int:
        """Local offset (within `unit_text`) where the clause containing
        `local_endpoint` begins -- the position right after the nearest
        preceding sentence-final punctuation or newline run, or 0 when
        none precedes it (the endpoint's own unit-relative clause is the
        whole unit up to that point)."""
        start = 0
        for m in cls._CLAUSE_BOUNDARY_RE.finditer(unit_text):
            if m.end() <= local_endpoint:
                start = m.end()
            else:
                break
        return start

    @classmethod
    def _embedded_aside_spans(cls, unit_text: str) -> List[Tuple[int, int]]:
        """Local (start, end) spans of quoted or parenthetical/bracketed
        asides within `unit_text` -- purely orthographic structure, not
        tied to any vocabulary. A candidate mention inside one of these
        spans is only trusted when the endpoint itself is inside the SAME
        span (both belong to the same embedded discourse); otherwise the
        span boundary is treated as a locality break, not a blanket ban."""
        spans: List[Tuple[int, int]] = []
        for pattern in (cls._DOUBLE_QUOTE_SPAN_RE, cls._SINGLE_QUOTE_SPAN_RE, cls._PAREN_SPAN_RE, cls._BRACKET_SPAN_RE):
            for m in pattern.finditer(unit_text):
                spans.append((m.start(), m.end()))
        return spans

    @staticmethod
    def _span_containing(spans: List[Tuple[int, int]], pos: int) -> Optional[Tuple[int, int]]:
        for s, e in spans:
            if s <= pos < e:
                return (s, e)
        return None

    def _entity_in_window_mention(
        self,
        entity: Entity,
        discourse_id: str,
        window_start: int,
        window_end: int,
    ) -> Optional[Tuple[int, int]]:
        """Occurrence-aware eligibility check:
        does this entity have an ACTUAL mention span inside
        [window_start, window_end)? Replaces any check based on the
        entity's globally earliest mention -- an earlier mention of the
        same canonical entity in a previous unit must never exclude a
        genuine, later, in-window mention. Returns the in-window mention
        nearest the window end (most relevant for locality/diagnostics),
        or None when no mention qualifies."""
        spans = self._entity_mention_spans_for_discourse(entity, discourse_id)
        in_window = [(s, e) for (s, e) in spans if window_start <= s < window_end]
        if not in_window:
            return None
        return max(in_window, key=lambda se: se[0])

    def _has_unrepresented_local_antecedent(
        self,
        *,
        raw_text: str,
        unit_text: str,
        unit_start: int,
        endpoint_offset: int,
        entities: List[Entity],
        discourse_id: str,
    ) -> bool:
        """True when a structurally plausible, nearer nominal
        antecedent (a determiner-led "the/a/an + NP" span) exists in the
        current clause before the endpoint but has no corresponding
        registered Entity mention -- general orthographic/grammatical
        signal only, no vocabulary. Only ever consulted for singular
        pronoun endpoints (a singular NP is not a plausible blocker for a
        plural pronoun); quoted/parenthetical NPs the endpoint itself is
        not part of are not considered plausible (reuses the same aside
        logic as Tier 2)."""
        norm = " ".join(str(raw_text or "").strip().lower().split())
        if norm not in self._SINGULAR_BLOCKABLE_PRONOUNS:
            return False
        local_endpoint = endpoint_offset - unit_start
        if not (0 <= local_endpoint <= len(unit_text)):
            return False
        clause_start_local = self._clause_local_start(unit_text, local_endpoint)
        aside_spans = self._embedded_aside_spans(unit_text)
        endpoint_aside = self._span_containing(aside_spans, local_endpoint)

        registered_spans: List[Tuple[int, int]] = []
        for entity in entities:
            meta = getattr(entity, "metadata", None) or {}
            if discourse_id and str(meta.get("discourse_id") or "").strip() not in {discourse_id, ""}:
                continue
            registered_spans.extend(self._entity_mention_spans_for_discourse(entity, discourse_id))

        for m in self._DETERMINER_NP_RE.finditer(unit_text, clause_start_local, local_endpoint):
            np_aside = self._span_containing(aside_spans, m.start())
            if np_aside is not None and np_aside != endpoint_aside:
                continue  # embedded in a quoted/parenthetical aside the endpoint is not part of
            np_abs_start, np_abs_end = unit_start + m.start(), unit_start + m.end()
            if any(rs < np_abs_end and np_abs_start < re_ for rs, re_ in registered_spans):
                continue  # already covered by a registered entity mention
            return True
        return False

    def _resolve_anaphoric_endpoint_antecedent(
        self,
        *,
        raw_text: str,
        entities: List[Entity],
        unit_start: Optional[int],
        discourse_id: Optional[str],
        endpoint_offset: Optional[int] = None,
        unit_text: Optional[str] = None,
        unit_index: Optional[int] = None,
        semantic_units: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Entity], Optional[str]]:
        """Try to resolve a referential endpoint to exactly one unambiguous
        same-discourse antecedent. Returns (entity, method) or (None, None)
        if nothing resolves -- callers must not guess when this is empty.

        `unit_index`/`semantic_units` (opt-in): when both are
        supplied, Tier 3 additionally requires the candidate to have a
        genuine mention inside the IMMEDIATELY PRECEDING semantic unit
        (positive discourse-adjacency support), not merely "somewhere
        earlier in the discourse" -- matching the one confirmed-correct
        historical Tier-3 pattern. When omitted (e.g. the possessive
        resolver's reuse of this same search), Tier 3's original, unrestricted
        behaviour is preserved exactly."""
        norm_text = " ".join(str(raw_text or "").strip().lower().split())
        if not norm_text:
            return None, None
        did = str(discourse_id or "").strip()
        light_noun = self._referential_endpoint_light_noun(norm_text)

        # Tier 1: an entity whose metadata already records THIS EXACT literal
        # occurrence as resolved via context by entity extraction -- the
        # strongest possible signal, since that work is already done.
        tier1: List[Entity] = []
        for entity in entities:
            meta = getattr(entity, "metadata", None) or {}
            if did and str(meta.get("discourse_id") or "").strip() not in {did, ""}:
                continue
            if not meta.get("mention_resolved_from_context"):
                continue
            surface = str(meta.get("surface_form") or "").strip().lower()
            if surface == norm_text:
                tier1.append(entity)
        uniq_tier1 = {str(getattr(e, "uri", "") or ""): e for e in tier1 if getattr(e, "uri", None)}
        if len(uniq_tier1) == 1:
            return next(iter(uniq_tier1.values())), "context_resolved_mention"
        if len(uniq_tier1) > 1:
            return None, None  # ambiguous even among already-context-resolved mentions

        # Tier 2: occurrence-level, CLAUSE-LOCAL antecedents -- a
        # mention that occurs strictly before the endpoint's own position
        # AND inside the same clause as the endpoint (e.g. "...some
        # students will struggle...reach out to them." -- "students"
        # precedes "them" within one clause). A candidate several
        # sentences/paragraphs earlier no longer qualifies merely because
        # it shares the same (possibly oversized) semantic unit; nor does a
        # candidate whose only in-window mention sits inside a quoted or
        # parenthetical aside the endpoint itself is not part of. A later
        # same-clause mention (cataphora) is never a candidate here, since
        # the cutoff is the endpoint's own offset. Eligibility is based on
        # the entity's ACTUAL mention spans inside the window, not its
        # earliest mention overall -- an earlier mention of the same entity
        # in a previous unit never excludes a genuine later, in-window one.
        # Only engages when the occurrence offset and unit text were
        # actually determined; a tie among clause-local candidates refuses
        # outright; zero clause-local candidates fall through to Tier 3
        # below rather than silently widening back to the whole unit.
        if endpoint_offset is not None and unit_start is not None and unit_text:
            local_endpoint = endpoint_offset - unit_start
            if 0 <= local_endpoint <= len(unit_text):
                clause_start = unit_start + self._clause_local_start(unit_text, local_endpoint)
                aside_spans = self._embedded_aside_spans(unit_text)
                endpoint_aside = self._span_containing(aside_spans, local_endpoint)

                # Pass 1: every entity's in-window mention, raw (no family/
                # aside filtering yet) -- needed so the coordination-pair
                # check below can see BOTH sides of an "X and/or Y" split
                # even when one side would otherwise fail the family filter.
                raw_in_window: Dict[str, Tuple[Entity, int, int]] = {}
                for entity in entities:
                    meta = getattr(entity, "metadata", None) or {}
                    if did and str(meta.get("discourse_id") or "").strip() not in {did, ""}:
                        continue
                    in_window = self._entity_in_window_mention(entity, did, clause_start, endpoint_offset)
                    if in_window is None:
                        continue
                    uri = str(getattr(entity, "uri", "") or "")
                    if uri:
                        raw_in_window[uri] = (entity, in_window[0] - unit_start, in_window[1] - unit_start)

                # Two entity mentions directly joined by "and"/"or" (nothing
                # else between them) are fragments of ONE coordinated NP that
                # extraction split apart -- neither fragment alone is a
                # complete, trustworthy antecedent (general syntactic signal,
                # requires an actual sibling entity mention on the other
                # side, not merely the word "and" appearing nearby).
                coordination_fragment_uris: Set[str] = set()
                ordered = sorted(raw_in_window.items(), key=lambda kv: kv[1][1])
                for i in range(len(ordered) - 1):
                    a_uri, (_a_ent, _a_start, a_end) = ordered[i]
                    b_uri, (_b_ent, b_start, _b_end) = ordered[i + 1]
                    if b_start < a_end:
                        continue
                    between = unit_text[a_end:b_start]
                    if self._COORDINATION_BETWEEN_RE.fullmatch(between):
                        coordination_fragment_uris.add(a_uri)
                        coordination_fragment_uris.add(b_uri)

                local_candidates: Dict[str, Entity] = {}
                for uri, (entity, m_local, _m_end_local) in raw_in_window.items():
                    if uri in coordination_fragment_uris:
                        continue
                    candidate_aside = (
                        self._span_containing(aside_spans, m_local) if 0 <= m_local < len(unit_text) else None
                    )
                    if candidate_aside is not None and candidate_aside != endpoint_aside:
                        continue  # candidate sits inside a quoted/parenthetical aside the endpoint is not part of
                    if light_noun is not None and not self._entity_compatible_with_light_noun(entity, light_noun):
                        continue
                    local_candidates[uri] = entity
                # Collapse overlapping/nested duplicate mentions of the same
                # referent (e.g. "Students" and "Some Students") BEFORE
                # counting ambiguity -- distinct real referents are untouched.
                local_candidates = self._collapse_duplicate_candidate_entities(local_candidates, did)
                if len(local_candidates) == 1:
                    return next(iter(local_candidates.values())), "occurrence_level_preceding_mention"
                if len(local_candidates) > 1:
                    return None, None  # two or more clause-local candidates -- effectively tied, do not guess

        # Tier 3: exactly one unambiguous same-discourse entity mentioned
        # strictly before this endpoint's semantic unit (i.e. in an earlier
        # unit), optionally narrowed to a compatible family when the phrase
        # names a light noun (e.g. "this tool" only considers system/model
        # -family entities, not any preceding entity at all).
        if unit_start is None:
            return None, None
        candidates: Dict[str, Entity] = {}
        for entity in entities:
            meta = getattr(entity, "metadata", None) or {}
            if did and str(meta.get("discourse_id") or "").strip() not in {did, ""}:
                continue
            earliest = self._earliest_entity_mention_start(entity)
            if earliest is None or earliest >= unit_start:
                continue
            if light_noun is not None and not self._entity_compatible_with_light_noun(entity, light_noun):
                continue
            uri = str(getattr(entity, "uri", "") or "")
            if uri:
                candidates[uri] = entity
        candidates = self._collapse_duplicate_candidate_entities(candidates, did)

        # Opt-in positive-support requirement -- "exactly one
        # preceding compatible entity" alone is not sufficient evidence of a
        # genuine antecedent (it may simply be the sole survivor of family
        # filtering, unrelated to the endpoint, or a broad topic-level
        # substitute for a true antecedent that was never extracted). When
        # unit adjacency information is available, additionally require the
        # surviving candidate to have an actual mention inside the
        # IMMEDIATELY PRECEDING semantic unit -- the one structural signal
        # shared by every confirmed-correct historical Tier-3 resolution.
        # Weaker (more distant) support means abstaining rather than
        # accepting an approximate substitute; the unresolved-endpoint guard
        # then preserves/rejects the endpoint as unresolved coreference,
        # exactly as designed.
        if candidates and unit_index is not None and semantic_units is not None and 0 <= unit_index < len(semantic_units):
            # "Immediately preceding" means: nothing else recognised as its
            # own semantic unit sits between the candidate's mention and the
            # current unit. That window's lower bound is the END of the
            # SECOND-TO-LAST preceding unit when one exists; otherwise (the
            # current unit is the first or second in the discourse) the
            # window reaches back to the very start of the discourse, so
            # un-chunked preamble text immediately before the first real
            # unit (a common chunker artefact) is not unfairly excluded.
            window_start = 0
            if unit_index >= 2:
                prior_prior = semantic_units[unit_index - 2]
                prior_prior_end = self._coerce_nonneg_int(prior_prior.get("end_offset"))
                if prior_prior_end is not None:
                    window_start = prior_prior_end
            candidates = {
                uri: entity
                for uri, entity in candidates.items()
                if any(
                    window_start <= s < unit_start
                    for s, _e in self._entity_mention_spans_for_discourse(entity, did)
                )
            }

        # Even a candidate with positive unit-adjacency support
        # (above) must not be preferred over a nearer, structurally
        # plausible local antecedent the entity extractor simply missed --
        # otherwise Tier 3 quietly substitutes a broader discourse-topic
        # entity for an unrepresented local NP. Opt-in (same gate as the
        # check above) so the possessive resolver's reuse of this search is
        # untouched.
        if (
            len(candidates) == 1
            and unit_index is not None
            and semantic_units is not None
            and unit_text
            and endpoint_offset is not None
            and self._has_unrepresented_local_antecedent(
                raw_text=raw_text, unit_text=unit_text, unit_start=unit_start,
                endpoint_offset=endpoint_offset, entities=entities, discourse_id=did,
            )
        ):
            return None, None

        # A still-unresolved endpoint that is structurally shaped
        # like discourse deixis over the immediately preceding semantic
        # unit (bare sentence-initial "this"/"that"/non-expletive "it",
        # not a determiner+noun phrase) must not be consumed here merely
        # because exactly one compatible nominal candidate happens to
        # precede it -- that would permanently prevent the propositional
        # Proposition resolver (which runs next) from ever getting a
        # chance. Deferring only abstains at THIS tier; if the preceding
        # unit turns out not to be a genuine single-assertion antecedent,
        # the propositional resolver itself declines too and the endpoint
        # reaches the unresolved-endpoint guard as unresolved coreference,
        # exactly as designed.
        if len(candidates) == 1 and self._endpoint_prefers_propositional_defer(
            raw_text=raw_text, unit_text=unit_text, endpoint_offset=endpoint_offset,
            unit_start=unit_start, unit_index=unit_index, semantic_units=semantic_units,
        ):
            return None, None

        if len(candidates) == 1:
            return next(iter(candidates.values())), "unambiguous_preceding_entity"
        return None, None

    def _resolve_anaphoric_relation_endpoints(
        self,
        *,
        relations: List[Relation],
        entities: List[Entity],
        unit: Dict[str, Any],
        discourse_id: Optional[str],
        unit_index: Optional[int] = None,
        semantic_units: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Relation], List[Entity]]:
        """Deterministic endpoint-resolution step, run before semantic
        relation validation (and before _resolve_semantic_unit_relations,
        which would otherwise fabricate a brand-new entity from unresolved
        pronoun/deictic text). Only acts on endpoints detected as
        referential; ambiguous or unresolved endpoints are left completely
        untouched -- no guessing, no synthetic canonicalisation here.

        `unit_index`/`semantic_units` (optional): forwarded to
        Tier 3's positive-support check when supplied by the caller."""
        unit_start = self._coerce_nonneg_int(unit.get("start_offset"))
        resolved_entities_for_lookup: Dict[str, Entity] = {}

        for relation in relations:
            meta = getattr(relation, "metadata", None) or {}
            changed = False
            evidence_text = meta.get("evidence_quote") or meta.get("evidence")

            # --- subject / source ---
            source = str(getattr(relation, "source", "") or "")
            if not source.startswith("http"):
                raw_subject = str(meta.get("subject_text") or "").strip() or source
                if self._is_referential_endpoint_text(raw_subject):
                    subject_offset = self._endpoint_occurrence_offset(
                        raw_text=raw_subject, unit=unit, evidence_text=evidence_text, unit_start=unit_start,
                    )
                    entity, method = self._resolve_anaphoric_endpoint_antecedent(
                        raw_text=raw_subject, entities=entities, unit_start=unit_start, discourse_id=discourse_id,
                        endpoint_offset=subject_offset, unit_text=unit.get("text"),
                        unit_index=unit_index, semantic_units=semantic_units,
                    )
                    if entity is not None:
                        uri = str(getattr(entity, "uri", "") or "")
                        if uri:
                            relation.source = uri
                            meta["subject_endpoint_original_surface"] = raw_subject
                            meta["subject_endpoint_coref_resolved"] = True
                            meta["subject_endpoint_coref_resolution_method"] = method
                            resolved_entities_for_lookup[uri] = entity
                            changed = True

            # --- object / target ---
            target_is_literal = bool(getattr(relation, "target_is_literal", False))
            target = str(getattr(relation, "target", "") or "")
            if target_is_literal:
                raw_object = str(meta.get("object_text") or "").strip() or str(getattr(relation, "target_literal", "") or "").strip() or target
            elif not target.startswith("http"):
                raw_object = str(meta.get("object_text") or "").strip() or target
            else:
                raw_object = None  # already a resolved entity URI

            if raw_object is not None and self._is_referential_endpoint_text(raw_object):
                object_offset = self._endpoint_occurrence_offset(
                    raw_text=raw_object, unit=unit, evidence_text=evidence_text, unit_start=unit_start,
                )
                entity, method = self._resolve_anaphoric_endpoint_antecedent(
                    raw_text=raw_object, entities=entities, unit_start=unit_start, discourse_id=discourse_id,
                    endpoint_offset=object_offset, unit_text=unit.get("text"),
                    unit_index=unit_index, semantic_units=semantic_units,
                )
                if entity is not None:
                    uri = str(getattr(entity, "uri", "") or "")
                    if uri:
                        relation.set_target_entity(uri)
                        meta["object_endpoint_original_surface"] = raw_object
                        meta["object_endpoint_coref_resolved"] = True
                        meta["object_endpoint_coref_resolution_method"] = method
                        resolved_entities_for_lookup[uri] = entity
                        changed = True

            if changed:
                relation.metadata = meta

        return relations, list(resolved_entities_for_lookup.values())

    # -----------------------------------------------------------------
    # Possessive-reference endpoint semantics
    #
    # Closed English possessive-determiner class + a generic "own"
    # intensifier -- purely grammatical, no dataset/domain vocabulary.
    # -----------------------------------------------------------------
    _POSSESSIVE_DETERMINERS: Set[str] = {"its", "his", "her", "their"}
    _POSSESSIVE_FAMILY_HINTS: Dict[str, Set[str]] = {
        "its": {"system", "concept", "dataset", "policy", "risk", "project", "classification"},
        "his": {"person"},
        "her": {"person"},
        # "their" is deliberately unrestricted: it is used both as a plural
        # possessive and as a gender-neutral singular one, so no family
        # narrows the search for it.
    }
    # Second-person possessive ("your X", e.g. "your data", "your
    # organization's tool"): kept as its OWN closed set, deliberately
    # separate from _POSSESSIVE_DETERMINERS above. It participates in the
    # SAME structural head-noun split (_split_possessive_endpoint) -- so
    # "your data" still becomes the clean literal head noun "data" instead
    # of leaking the word "your" into a literal value or, worse, being
    # minted as its own standalone "your data" entity by the generic
    # surface-entity resolver -- but it must NEVER go through
    # _resolve_possessive_antecedent's general antecedent search or its
    # reflexive-subject fallback (see the guard at the top of that method).
    # "your" is deictic to the discourse ADDRESSEE, not to a preceding
    # textual antecedent (exactly like bare "you" is deliberately excluded
    # from _REFERENTIAL_PRONOUNS above, for the identical reason: a
    # proximity-based antecedent search previously let "you" latch onto an
    # unrelated nearby entity) and not to the sentence's own grammatical
    # subject (unlike "its own"/"their own", "your X" does not mean "the
    # subject's X" -- "The tool collects your data" is not "the tool's
    # data"). Resolving its possessor is therefore always left to whatever
    # already-resolved, non-fabricating signal the endpoint may carry
    # (e.g. an entity extraction call that already grounded "you" in
    # Context) -- never guessed here.
    _SECOND_PERSON_POSSESSIVE_DETERMINERS: Set[str] = {"your"}

    @classmethod
    def _split_possessive_endpoint(cls, text: str) -> Optional[Tuple[str, str]]:
        """If `text` is a possessive-pronoun-led noun phrase (e.g. "their
        approach", "their own performance", "its accuracy"), return
        (possessive_word, head_noun_phrase) with a leading "own" intensifier
        stripped from the head. Returns None otherwise (including for a
        bare possessive with no head noun, which is not this pattern)."""
        norm = " ".join(str(text or "").strip().lower().split())
        tokens = norm.split()
        if len(tokens) < 2 or (
            tokens[0] not in cls._POSSESSIVE_DETERMINERS
            and tokens[0] not in cls._SECOND_PERSON_POSSESSIVE_DETERMINERS
        ):
            return None
        head_tokens = tokens[1:]
        if head_tokens and head_tokens[0] == "own":
            head_tokens = head_tokens[1:]
        if not head_tokens:
            return None
        return tokens[0], " ".join(head_tokens)

    def _resolve_possessive_antecedent(
        self,
        *,
        possessive_word: str,
        relation: Relation,
        entities: List[Entity],
        local_pool_entities: List[Entity],
        unit: Dict[str, Any],
        unit_start: Optional[int],
        discourse_id: Optional[str],
        evidence_text: Optional[str],
    ) -> Tuple[Optional[Entity], Optional[str]]:
        """Resolve the possessor of a possessive determiner using the same
        same-discourse, occurrence-level search the occurrence-level
        pronoun resolver already implements
        (reused, not duplicated), then -- only if that finds nothing --
        fall back to the relation's own already-resolved subject when
        grammar/family make that interpretation unambiguous."""
        if possessive_word in self._SECOND_PERSON_POSSESSIVE_DETERMINERS:
            # "your" has no textual antecedent to search for (it names the
            # discourse addressee, not a preceding NP -- running the
            # general antecedent search here would risk the same
            # proximity-based mismatch already documented for bare "you")
            # and the reflexive-subject fallback below is a 3rd-person
            # clause-internal heuristic that does not hold for 2nd person.
            # Abstain deterministically: the caller still strips the
            # possessive down to its clean head noun regardless of this
            # return value, but no possessor identity -- correct or false
            # -- is ever guessed for "your" by this generic mechanism.
            return None, None
        endpoint_offset = self._endpoint_occurrence_offset(
            raw_text=possessive_word, unit=unit, evidence_text=evidence_text, unit_start=unit_start,
        )
        entity, method = self._resolve_anaphoric_endpoint_antecedent(
            raw_text=possessive_word,
            entities=entities,
            unit_start=unit_start,
            discourse_id=discourse_id,
            endpoint_offset=endpoint_offset,
            unit_text=unit.get("text"),
        )
        if entity is not None:
            return entity, method

        # The general search returning None is ambiguous by itself -- it
        # covers BOTH "nothing precedes at all" and "two or more compatible
        # candidates precede" (the occurrence-level resolver's own contract
        # doesn't distinguish them, and is not modified here). The
        # reflexive-subject fallback below must only apply to the former.
        # Scoped to SAME-UNIT candidates only (mirroring that resolver's own
        # local tier) -- an
        # unrelated entity mentioned far earlier in a long discourse is not
        # a genuine competing interpretation for a local possessive and
        # must not block the reflexive fallback; only a nearby, same-unit
        # candidate is a real local ambiguity signal.
        cutoff = endpoint_offset if endpoint_offset is not None else None
        if cutoff is not None and unit_start is not None:
            did = str(discourse_id or "").strip()
            for candidate in entities:
                cmeta = getattr(candidate, "metadata", None) or {}
                if did and str(cmeta.get("discourse_id") or "").strip() not in {did, ""}:
                    continue
                earliest = self._earliest_entity_mention_start(candidate)
                if earliest is not None and unit_start <= earliest < cutoff:
                    return None, None  # a same-unit contextual candidate exists but was ambiguous/incompatible

        # Reflexive fallback: the relation's own subject, already resolved
        # to a URI by _resolve_semantic_unit_relations by the time this
        # runs. Only used when the general search above found nothing at
        # all to consider, and only when the possessive word's family hint
        # (if any) is satisfied -- "its" never resolves to a person,
        # "his"/"her" only resolve to a person, "their" is unrestricted.
        source_uri = str(getattr(relation, "source", "") or "")
        if not source_uri.startswith("http"):
            return None, None
        subject_entity = next(
            (e for e in local_pool_entities if str(getattr(e, "uri", "") or "") == source_uri), None
        )
        if subject_entity is None:
            return None, None
        hints = self._POSSESSIVE_FAMILY_HINTS.get(possessive_word)
        if hints is not None and self._entity_semantic_family(subject_entity) not in hints:
            return None, None
        return subject_entity, "reflexive_subject"

    def _apply_possessive_endpoint_semantics(
        self,
        *,
        relations: List[Relation],
        entities: List[Entity],
        local_pool_entities: List[Entity],
        unit: Dict[str, Any],
        discourse_id: Optional[str],
    ) -> List[Relation]:
        """Detect possessive-pronoun-led object endpoints (e.g. "their
        approach") and correct their semantic representation: the relation
        endpoint becomes the HEAD NOUN PHRASE only ("approach"), never the
        antecedent itself, with the resolved possessor (if any) recorded
        separately in metadata. Scoped to the object endpoint only -- every
        confirmed case and every required test case is object-position;
        the relation's `source` field has no literal/non-literal duality to
        safely rewrite the way `target` does."""
        unit_start = self._coerce_nonneg_int(unit.get("start_offset"))

        for relation in relations:
            meta = getattr(relation, "metadata", None) or {}
            raw_object = str(meta.get("object_text") or "").strip()
            split = self._split_possessive_endpoint(raw_object)
            if split is None:
                continue
            possessive_word, head_noun = split
            evidence_text = meta.get("evidence_quote") or meta.get("evidence")

            possessor_entity, method = self._resolve_possessive_antecedent(
                possessive_word=possessive_word,
                relation=relation,
                entities=entities,
                local_pool_entities=local_pool_entities,
                unit=unit,
                unit_start=unit_start,
                discourse_id=discourse_id,
                evidence_text=evidence_text,
            )

            relation.target = head_noun
            relation.target_is_literal = True
            relation.target_literal = head_noun
            meta["object_possessor_original_surface"] = raw_object
            meta["object_possessor_head_noun_phrase"] = head_noun
            if possessor_entity is not None:
                meta["object_possessor_entity_uri"] = str(getattr(possessor_entity, "uri", "") or "")
                meta["object_possessor_resolution_method"] = method
            relation.metadata = meta

        return relations

    # -----------------------------------------------------------------
    # Propositional / discourse-deictic endpoint resolution
    #
    # Distinct from nominal coreference and possessive
    # semantics: a bare sentence-initial "This"/"That"/"It" can
    # refer to the STATE OR CONTENT expressed by the immediately preceding
    # semantic unit, not to any single named entity. Only engages after
    # ordinary entity-coreference resolution has already failed, only for
    # the closed grammatical class of bare singular deictic pronouns, only
    # when the pronoun is the unit's own first word (a general syntactic
    # signal for discourse deixis, not a dataset/domain-specific rule),
    # never for expletive/dummy "it" constructions, and only when the
    # immediately preceding semantic unit is itself an unambiguous single
    # -assertion antecedent. No dataset IDs, entity names, or domain
    # vocabulary anywhere in this logic.
    # -----------------------------------------------------------------
    _PROPOSITIONAL_CANDIDATE_PRONOUNS: Set[str] = {"this", "that", "it"}

    # Expletive/dummy "it" -- a purely syntactic (not lexical) construction:
    # dummy subject + linking/raising verb + a "that"/"to" complement, with
    # no real-world antecedent of any kind (e.g. "It is important that...",
    # "It seems clear to..."). This does NOT exclude "it" globally -- "It
    # undermines the process." does not match this pattern and remains
    # eligible for propositional resolution below.
    _EXPLETIVE_IT_PATTERN = re.compile(
        r"^it\s+(?:is|was|seems|appears|follows|remains|becomes|has\s+been|turns\s+out)"
        r"\s+\S[\s\S]{0,80}?\b(?:that|to)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_propositional_candidate_text(cls, text: str) -> bool:
        norm = " ".join(str(text or "").strip().lower().split())
        return norm in cls._PROPOSITIONAL_CANDIDATE_PRONOUNS

    @staticmethod
    def _endpoint_is_unit_initial(*, unit_text: str, local_offset: int) -> bool:
        if local_offset < 0:
            return False
        prefix = str(unit_text or "")[:local_offset]
        return prefix.strip() == ""

    @classmethod
    def _is_expletive_it_construction(cls, *, unit_text: str, local_offset: int) -> bool:
        segment = str(unit_text or "")[local_offset:]
        return bool(cls._EXPLETIVE_IT_PATTERN.match(segment))

    @classmethod
    def _endpoint_prefers_propositional_defer(
        cls,
        *,
        raw_text: str,
        unit_text: Optional[str],
        endpoint_offset: Optional[int],
        unit_start: Optional[int],
        unit_index: Optional[int],
        semantic_units: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """General, vocabulary-free eligibility pre-check for
        deferring Tier 3's nominal fallback to the propositional Proposition
        resolver, instead of letting Tier 3 consume a still-unresolved
        endpoint that is structurally shaped like discourse deixis over the
        immediately preceding semantic unit.

        Deliberately reuses the SAME structural checks
        _resolve_propositional_endpoint_antecedent itself applies -- the
        closed bare-pronoun class (_is_propositional_candidate_text, which
        excludes determiner+noun phrases like "this disagreement" by
        construction), unit-initial subject position, and non-expletive
        "it" -- plus a lightweight existence check for an immediately
        preceding semantic-category unit with actual text. This is only an
        *eligibility* pre-check: the real propositional resolver still
        independently re-validates everything (including its own
        single-distinct-subject ambiguity gate) before ever minting a
        Proposition, so an endpoint that defers here but fails that
        stricter check simply stays unresolved -- it is never forced onto
        an entity.

        Opt-in via the same `unit_index`/`semantic_units` gate the Tier-3
        positive-support check already uses: when either is None (e.g. the
        possessive resolver's reuse of this search), this returns False and
        Tier 3's original,
        unrestricted nominal-fallback behaviour is preserved exactly.

        Restricted to bare "this"/"that" -- deliberately NOT "it" -- as the
        defer trigger. This is a general English-grammar distinction, not a
        dataset/domain rule: a bare demonstrative subject ("This
        undermines...", "That explains...") is the canonical
        proposition/event-reference construction, whereas a bare personal
        subject pronoun ("It failed.") overwhelmingly takes a concrete
        nominal antecedent when one has just been introduced -- exactly the
        "strong entity-level evidence" case that must keep resolving
        nominally. "it" remains fully eligible for propositional resolution
        via the unchanged, broader _is_propositional_candidate_text/
        _resolve_propositional_endpoint_antecedent path in its own right
        (e.g. when no nominal candidate exists at all); it is only excluded
        from pre-emptively deferring AWAY a Tier 3 nominal candidate that
        already exists."""
        norm = " ".join(str(raw_text or "").strip().lower().split())
        if norm not in {"this", "that"}:
            return False
        if endpoint_offset is None or unit_start is None or not unit_text:
            return False
        local_offset = endpoint_offset - unit_start
        if local_offset < 0 or local_offset > len(unit_text):
            return False
        if not cls._endpoint_is_unit_initial(unit_text=unit_text, local_offset=local_offset):
            return False
        if unit_index is None or semantic_units is None:
            return False
        if unit_index < 1 or unit_index > len(semantic_units):
            return False
        prev_unit = semantic_units[unit_index - 1]
        if str(prev_unit.get("section_category") or "").strip() != "semantic":
            return False
        if not str(prev_unit.get("text") or "").strip():
            return False
        return True

    def _get_or_create_unit_proposition_entity(
        self,
        *,
        unit: Dict[str, Any],
        discourse_id: str,
        origin_key: str,
        field: str,
        derived_from_relation_uris: List[str],
    ) -> Optional[Entity]:
        """Create (or, deterministically, re-fetch) the ONE Proposition
        entity representing the semantic content of `unit`. Identity is
        derived purely from structural provenance -- origin key (source
        field + discourse id), the semantic unit's own id, and its
        offsets -- never from LLM-generated wording, so repeated
        processing of the same source text always yields the same URI."""
        unit_id = str(unit.get("unit_id") or "").strip()
        start = self._coerce_nonneg_int(unit.get("start_offset"))
        end = self._coerce_nonneg_int(unit.get("end_offset"))
        unit_text = str(unit.get("text") or "").strip()
        if not unit_id or start is None or end is None or not unit_text:
            return None

        token = f"{origin_key or discourse_id}_{unit_id}_{start}_{end}"
        prop_uri = make_uri(f"proposition_{make_uri_safe(token)}")

        prop_entity = Entity(
            uri=prop_uri,
            label=f"Proposition {unit_id}",
            entity_type="Proposition",
            confidence=1.0,
            metadata={
                "discourse_id": discourse_id,
                "origin_key": origin_key,
                "field": field,
                # A Proposition's identity is scoped to its own
                # structural provenance (origin_key + unit_id, already baked
                # into prop_uri above) -- it must never be globally merged
                # with a same-label Proposition from an unrelated discourse
                # purely because two different units happen to share a
                # generic unit_id like "Body:8". Consumed generically by
                # EntityConsolidator/CanonicalLinker (metadata-driven, no
                # dependency on this module).
                "discourse_scoped_identity": True,
                # A Proposition is a CONSTRUCTED endpoint standing for the
                # semantic content of a clause -- it is not a verbatim source
                # mention, so it must not carry a `surface_form` (that key is
                # read downstream as "this exact text appeared in the source"
                # and would emit a bogus skos:hiddenLabel "proposition").
                # Its source grounding is the clause offsets + clause text
                # below; the marker keeps it explicitly distinct from
                # ordinary mention-grounded entities.
                "mention_grounding": "constructed",
                "constructed_from": "semantic_clause",
                "context_source": "propositional_coreference",
                "source_doc_id": str(discourse_id),
                "text": unit_text,
                "start_char": start,
                "end_char": end,
                "semantic_unit_id": unit_id,
                "semantic_unit_type": str(unit.get("unit_type") or ""),
            },
        )
        if derived_from_relation_uris:
            prop_entity.metadata["derived_from_relation_uris"] = list(derived_from_relation_uris)

        if not hasattr(prop_entity, "name") or prop_entity.name is None:
            prop_entity.name = prop_entity.label
        if not hasattr(prop_entity, "type") or prop_entity.type is None:
            prop_entity.type = prop_entity.entity_type
        self._ensure_entity_ontology_grounding(prop_entity)

        if hasattr(self.global_registry, "get_or_create"):
            prop_entity = self.global_registry.get_or_create(prop_entity)
            self._ensure_entity_ontology_grounding(prop_entity)
        elif hasattr(self.global_registry, "add_entity"):
            self.global_registry.add_entity(prop_entity)
        elif hasattr(self.global_registry, "add"):
            self.global_registry.add(prop_entity)
        else:
            if hasattr(self.global_registry, "entities"):
                self.global_registry.entities[prop_uri] = prop_entity
        return prop_entity

    def _resolve_propositional_endpoint_antecedent(
        self,
        *,
        raw_text: str,
        endpoint_offset: Optional[int],
        unit: Dict[str, Any],
        unit_start: Optional[int],
        unit_index: int,
        semantic_units: List[Dict[str, Any]],
        unit_relations_by_id: Dict[str, List[Relation]],
        discourse_id: str,
        origin_key: str,
        field: str,
    ) -> Tuple[Optional[Entity], Optional[str]]:
        """Resolve a bare sentence-initial deictic endpoint to the
        Proposition representing the immediately preceding semantic unit
        -- ONLY when that unit is itself an unambiguous single-assertion
        antecedent. Returns (None, None) whenever any condition is not
        met; callers must not guess or fall back further."""
        if not self._is_propositional_candidate_text(raw_text):
            return None, None
        if endpoint_offset is None or unit_start is None:
            return None, None
        unit_text = str(unit.get("text") or "")
        local_offset = endpoint_offset - unit_start
        if local_offset < 0 or local_offset > len(unit_text):
            return None, None
        if not self._endpoint_is_unit_initial(unit_text=unit_text, local_offset=local_offset):
            return None, None  # mid-sentence deixis is not treated as propositional here

        norm = " ".join(str(raw_text or "").strip().lower().split())
        if norm == "it" and self._is_expletive_it_construction(unit_text=unit_text, local_offset=local_offset):
            return None, None  # dummy/expletive subject -- no antecedent of any kind

        if unit_index <= 0:
            return None, None  # no preceding unit at all
        prev_unit = semantic_units[unit_index - 1]
        if str(prev_unit.get("section_category") or "").strip() != "semantic":
            return None, None
        prev_text = str(prev_unit.get("text") or "").strip()
        if not prev_text:
            return None, None

        prev_unit_id = str(prev_unit.get("unit_id") or "")
        prev_relations = unit_relations_by_id.get(prev_unit_id) or []
        distinct_subjects = {
            str(getattr(r, "source", "") or "")
            for r in prev_relations
            if str(getattr(r, "source", "") or "")
        }
        if len(distinct_subjects) > 1:
            return None, None  # preceding unit expresses multiple distinct assertions -- structurally ambiguous

        derived_from_relation_uris = sorted(
            {str(getattr(r, "uri", "") or "") for r in prev_relations if str(getattr(r, "uri", "") or "")}
        )
        proposition_entity = self._get_or_create_unit_proposition_entity(
            unit=prev_unit,
            discourse_id=discourse_id,
            origin_key=origin_key,
            field=field,
            derived_from_relation_uris=derived_from_relation_uris,
        )
        if proposition_entity is None:
            return None, None
        return proposition_entity, "propositional_preceding_unit_reference"

    def _resolve_propositional_relation_endpoints(
        self,
        *,
        relations: List[Relation],
        unit: Dict[str, Any],
        unit_index: int,
        semantic_units: List[Dict[str, Any]],
        unit_relations_by_id: Dict[str, List[Relation]],
        discourse_id: str,
        origin_key: str,
        field: str,
    ) -> Tuple[List[Relation], List[Entity]]:
        """Deterministic propositional/discourse-deictic endpoint
        resolution, run AFTER ordinary entity-coreference resolution
        has already had its chance and BEFORE
        _resolve_semantic_unit_relations would otherwise fabricate a
        brand-new entity from the unresolved pronoun text. Only acts on
        endpoints still unresolved (not already an entity URI)."""
        unit_start = self._coerce_nonneg_int(unit.get("start_offset"))
        resolved_entities: Dict[str, Entity] = {}

        for relation in relations:
            meta = getattr(relation, "metadata", None) or {}
            changed = False
            evidence_text = meta.get("evidence_quote") or meta.get("evidence")

            source = str(getattr(relation, "source", "") or "")
            if not source.startswith("http"):
                raw_subject = str(meta.get("subject_text") or "").strip() or source
                subject_offset = self._endpoint_occurrence_offset(
                    raw_text=raw_subject, unit=unit, evidence_text=evidence_text, unit_start=unit_start,
                )
                entity, method = self._resolve_propositional_endpoint_antecedent(
                    raw_text=raw_subject, endpoint_offset=subject_offset, unit=unit, unit_start=unit_start,
                    unit_index=unit_index, semantic_units=semantic_units,
                    unit_relations_by_id=unit_relations_by_id, discourse_id=discourse_id,
                    origin_key=origin_key, field=field,
                )
                if entity is not None:
                    uri = str(getattr(entity, "uri", "") or "")
                    if uri:
                        relation.source = uri
                        meta["subject_endpoint_original_surface"] = raw_subject
                        meta["subject_endpoint_coref_resolved"] = True
                        meta["subject_endpoint_coref_resolution_method"] = method
                        meta["subject_endpoint_antecedent_unit_id"] = str(semantic_units[unit_index - 1].get("unit_id") or "")
                        resolved_entities[uri] = entity
                        changed = True

            target_is_literal = bool(getattr(relation, "target_is_literal", False))
            target = str(getattr(relation, "target", "") or "")
            if target_is_literal:
                raw_object = str(meta.get("object_text") or "").strip() or str(getattr(relation, "target_literal", "") or "").strip() or target
            elif not target.startswith("http"):
                raw_object = str(meta.get("object_text") or "").strip() or target
            else:
                raw_object = None

            if raw_object is not None:
                object_offset = self._endpoint_occurrence_offset(
                    raw_text=raw_object, unit=unit, evidence_text=evidence_text, unit_start=unit_start,
                )
                entity, method = self._resolve_propositional_endpoint_antecedent(
                    raw_text=raw_object, endpoint_offset=object_offset, unit=unit, unit_start=unit_start,
                    unit_index=unit_index, semantic_units=semantic_units,
                    unit_relations_by_id=unit_relations_by_id, discourse_id=discourse_id,
                    origin_key=origin_key, field=field,
                )
                if entity is not None:
                    uri = str(getattr(entity, "uri", "") or "")
                    if uri:
                        relation.set_target_entity(uri)
                        meta["object_endpoint_original_surface"] = raw_object
                        meta["object_endpoint_coref_resolved"] = True
                        meta["object_endpoint_coref_resolution_method"] = method
                        meta["object_endpoint_antecedent_unit_id"] = str(semantic_units[unit_index - 1].get("unit_id") or "")
                        resolved_entities[uri] = entity
                        changed = True

            if changed:
                relation.metadata = meta

        return relations, list(resolved_entities.values())

    def _entity_overlaps_span(
        self,
        entity: Entity,
        discourse_id: Optional[str],
        span_start: int,
        span_end: int,
    ) -> bool:
        meta = getattr(entity, "metadata", None) or {}
        mentions = meta.get("mentions")
        if isinstance(mentions, list):
            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                m_did = str(mention.get("discourse_id") or "").strip()
                if discourse_id and m_did and m_did != str(discourse_id).strip():
                    continue
                m_start = self._coerce_nonneg_int(mention.get("start_char"))
                m_end = self._coerce_nonneg_int(mention.get("end_char"))
                if m_start is None or m_end is None:
                    continue
                if int(m_start) < int(span_end) and int(m_end) > int(span_start):
                    return True
            return False

        e_start = self._coerce_nonneg_int(meta.get("start_char"))
        e_end = self._coerce_nonneg_int(meta.get("end_char"))
        return (
            e_start is not None
            and e_end is not None
            and int(e_start) < int(span_end)
            and int(e_end) > int(span_start)
        )

    @staticmethod
    def _merge_entity_alias(entity: Entity, alias: str) -> None:
        alias_text = str(alias or "").strip()
        if not alias_text:
            return
        meta = getattr(entity, "metadata", None) or {}
        aliases = meta.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        if alias_text not in aliases and alias_text != str(getattr(entity, "label", "") or ""):
            aliases.append(alias_text)
        if aliases:
            meta["aliases"] = aliases
            entity.metadata = meta

    # CORE_GENERIC: bare structural/resource-category nouns overloaded in
    # essentially any subject-matter domain -- see the identical
    # reasoning already applied to utils/canonical_linker.py::DO_NOT_MERGE.
    # Domain-specific terms (AI-domain vocabulary, AI-ethics finding
    # vocabulary) come from the active profile's "generic_surface_terms"
    # gazetteer instead -- see domain_profiles.py::AI_ETHICS_PROFILE.
    _CORE_GENERIC_SURFACE_LABELS: Set[str] = {
        "system", "systems", "model", "models", "tool", "tools",
        "data", "dataset", "datasets",
    }

    def _generic_surface_label(self, label: str) -> bool:
        norm = normalize_text(str(label or ""))
        if norm in self._CORE_GENERIC_SURFACE_LABELS:
            return True
        return self.domain_profile.matches_gazetteer("generic_surface_terms", norm)

    @staticmethod
    def _entity_head_token(label: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", str(label or ""))
        return normalize_text(tokens[-1]) if tokens else ""

    def _entity_specificity_score(self, entity: Entity, exact_in_unit: bool) -> Tuple[int, int, int]:
        label = str(getattr(entity, "label", "") or "")
        tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", label)
        return (
            0 if exact_in_unit else 1,
            0 if self._generic_surface_label(label) else -1,
            -len(tokens),
        )

    def _prefer_specific_surface_entity(self, left: Entity, right: Entity) -> bool:
        left_label = normalize_text(str(getattr(left, "label", "") or ""))
        right_label = normalize_text(str(getattr(right, "label", "") or ""))
        if not left_label or not right_label or left_label == right_label:
            return False
        if re.search(r"\b(?:and|or)\b", right_label):
            parts = [
                normalize_text(part.strip(" \t\n\r,.;:"))
                for part in re.split(r"\s+(?:and|or)\s+", right_label)
                if part.strip()
            ]
            if left_label in parts:
                return False
        if left_label not in right_label and right_label not in left_label:
            return False
        if self._generic_surface_label(left_label) and self._entity_semantic_family(left) == self._entity_semantic_family(right):
            return len(right_label.split()) > len(left_label.split())
        left_head = self._entity_head_token(left_label)
        right_head = self._entity_head_token(right_label)
        if left_head and right_head and left_head != right_head:
            return False
        return len(right_label.split()) > len(left_label.split())

    def _prioritize_local_entities(
        self,
        selected: List[Tuple[Entity, bool]],
        *,
        unit_text: str,
    ) -> List[Entity]:
        if not selected:
            return []
        ordered = sorted(
            selected,
            key=lambda item: self._entity_specificity_score(item[0], item[1]),
        )
        filtered: List[Entity] = []
        for entity, exact_in_unit in ordered:
            _ = exact_in_unit
            if any(self._prefer_specific_surface_entity(entity, existing) for existing in filtered):
                continue
            filtered.append(entity)
        return filtered

    @staticmethod
    def _evidence_mentions_entity(label: str, evidence: str) -> bool:
        norm_label = normalize_text(str(label or ""))
        norm_evidence = normalize_text(str(evidence or ""))
        if not norm_label or not norm_evidence:
            return False
        return norm_label in norm_evidence

    @staticmethod
    def _surface_variant_pattern(variant: str) -> str:
        escaped = re.escape(str(variant or "").strip())
        return escaped.replace(r"\ ", r"\s+")

    # CORE_GENERIC remainder of the former "generic_aliases" word list --
    # academic/structural nouns overloaded in essentially any domain, not
    # AI-ethics-specific. Domain vocabulary comes from the active
    # profile's "generic_surface_terms" gazetteer instead (the SAME
    # gazetteer _generic_surface_label consults) -- see domain_profiles.py.
    _CORE_GENERIC_ALIAS_TERMS: Set[str] = {
        "system", "systems", "model", "models", "dataset", "data",
        "project", "study", "paper", "report",
    }

    def _entity_surface_variants(self, entity: Optional[Entity]) -> List[str]:
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
            # Root-cause fix: slash-splitting is meant for legitimate
            # "X/Y" alias labels (e.g. "Adult/Census Income Dataset"), but
            # a URL-shaped label ("https://kg/e/org2") also contains "/"
            # and splits into garbage fragments -- "//" alone produces an
            # empty part, and "https:" survives as a spurious short
            # variant that then substring-matches almost any other URL via
            # _entity_matches_surface, silently merging unrelated
            # entities. Skip splitting entirely for URL-shaped labels.
            if "://" not in label:
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

    def _entity_supported_in_text(self, entity: Optional[Entity], evidence: str) -> bool:
        norm_evidence = normalize_text(str(evidence or ""))
        if entity is None or not norm_evidence:
            return False
        for variant in self._entity_surface_variants(entity):
            if re.search(rf"(?<!\w){self._surface_variant_pattern(variant)}(?!\w)", norm_evidence, flags=re.IGNORECASE):
                return True
        return False

    def _entity_variant_hits(self, entity: Optional[Entity], text: str) -> List[Tuple[int, int, str]]:
        raw_text = str(text or "")
        if entity is None or not raw_text:
            return []
        hits: List[Tuple[int, int, str]] = []
        seen: Set[Tuple[int, int]] = set()
        for variant in self._entity_surface_variants(entity):
            pattern = rf"(?<!\w){self._surface_variant_pattern(variant)}(?!\w)"
            for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE):
                key = (int(match.start()), int(match.end()))
                if key in seen:
                    continue
                seen.add(key)
                hits.append((key[0], key[1], variant))
        return sorted(hits, key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

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

    @staticmethod
    def _surface_entity_clause_verbs() -> Set[str]:
        return {
            "goes",
            "found",
            "shows",
            "show",
            "showed",
            "revealed",
            "reveal",
            "reveals",
            "results",
            "result",
            "leads",
            "lead",
            "leading",
            "associated",
            "associate",
            "associates",
            "demonstrates",
            "demonstrate",
            "demonstrated",
        }

    @staticmethod
    def _surface_entity_clause_starters() -> Set[str]:
        return {
            "beyond",
            "especially",
            "even",
            "like",
            "than",
            "versus",
            "vs",
        }

    @classmethod
    def _surface_entity_quality_ok(
        cls,
        surface: str,
        *,
        bucket: Optional[str],
    ) -> bool:
        tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", str(surface or ""))
        if not tokens:
            return False
        norm_tokens = [normalize_text(tok) for tok in tokens]
        if norm_tokens[0] in cls._surface_entity_clause_starters():
            return False
        if any(tok in cls._surface_entity_clause_verbs() for tok in norm_tokens):
            return False
        norm_surface = normalize_text(surface)
        if re.search(r"\b(?:that|which|who)\b", norm_surface, flags=re.IGNORECASE):
            return False
        if bucket in {"concept", "risk", "system", "dataset", "group"} and re.search(
            r"\b(?:goes|found|shows?|revealed?|results?|leads?|leading|associated|demonstrates?)\b",
            norm_surface,
            flags=re.IGNORECASE,
        ):
            return False
        return True

    def _coordinated_surface_entity_splits(
        self,
        surface: str,
    ) -> List[Tuple[str, str, int]]:
        raw = str(surface or "").strip()
        if not raw or not re.search(r"\b(?:and|or)\b", raw, flags=re.IGNORECASE):
            return []
        match = re.match(r"(.+?)\s+\b(?:and|or)\b\s+(.+)$", raw, flags=re.IGNORECASE)
        if not match:
            return []
        parts = [match.group(1).strip(" \t\n\r,.;:"), match.group(2).strip(" \t\n\r,.;:")]
        out: List[Tuple[str, str, int]] = []
        seen: Set[str] = set()
        raw_lower = raw.lower()
        for part in parts:
            inferred_type = self._infer_surface_entity_type(part)
            if inferred_type is None:
                return []
            bucket = self._surface_entity_bucket(part)
            if not self._surface_entity_quality_ok(part, bucket=bucket):
                return []
            norm_part = normalize_text(part)
            if not norm_part or norm_part == normalize_text(raw) or norm_part in seen:
                return []
            offset = raw_lower.find(part.lower())
            if offset < 0:
                return []
            seen.add(norm_part)
            out.append((part, inferred_type, int(offset)))
        return out if len(out) >= 2 else []

    # Root-cause fix (architectural-cleanup pass, relation-validation
    # duplication): these three used to each hardcode their own AI-ethics
    # word list (byte-for-byte duplicated in
    # utils/semantic_relation_policy.py). All three now delegate to the
    # ONE shared, profile-driven category mechanism
    # (domain_profiles.matches_semantic_category) -- no vocabulary is
    # hardcoded here, and utils/semantic_relation_policy.py's equivalent
    # methods delegate to the exact same function, so the two layers
    # cannot disagree.
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

    @staticmethod
    def _split_local_sentences(evidence: str) -> List[str]:
        norm = normalize_text(str(evidence or ""))
        if not norm:
            return []
        parts = re.split(r"(?<=[\.\!\?\;\:])\s+|\n+", norm)
        out = [part.strip() for part in parts if str(part).strip()]
        return out or [norm]

    @staticmethod
    def _split_sentence_spans(text: str) -> List[Tuple[int, int, str]]:
        raw = str(text or "")
        if not raw.strip():
            return []
        spans: List[Tuple[int, int, str]] = []
        raw_spans: List[Tuple[int, int, str]] = []
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", raw, flags=re.S):
            start = int(match.start())
            end = int(match.end())
            sentence = raw[start:end].strip()
            if not sentence:
                continue
            raw_spans.append((start, end, sentence))
        for start, end, sentence in raw_spans:
            if spans and (re.fullmatch(r"[A-Z]\.", sentence) or re.search(r"\b[A-Z]\.$", spans[-1][2])):
                prev_start, _prev_end, prev_sentence = spans[-1]
                merged_text = f"{prev_sentence}{raw[spans[-1][1]:end]}".strip()
                spans[-1] = (prev_start, end, merged_text)
                continue
            spans.append((start, end, sentence))
        return spans or [(0, len(raw), raw.strip())]

    @classmethod
    def _candidate_local_sentences(
        cls,
        evidence: str,
        source_variants: List[str],
        target_variants: List[str],
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

    @classmethod
    def _cue_between_mentions(
        cls,
        text: str,
        left_label: str,
        right_label: str,
        cue_pattern: str,
    ) -> bool:
        left_hits = cls._find_all_occurrences(text, left_label)
        right_hits = cls._find_all_occurrences(text, right_label)
        if not left_hits or not right_hits:
            return False
        for _, left_end in left_hits:
            for right_start, _ in right_hits:
                if left_end > right_start:
                    continue
                between = text[left_end:right_start]
                if re.search(cue_pattern, between, flags=re.IGNORECASE):
                    return True
        return False

    def _predicate_locally_entailed(
        self,
        predicate: str,
        source_variants: List[str],
        target_variants: List[str],
        evidence: str,
    ) -> bool:
        return self._get_semantic_policy().predicate_locally_entailed(
            predicate,
            source_variants,
            target_variants,
            evidence,
        )

    # Translates the shared classifier's canonical (finer-grained) family
    # taxonomy back to this file's own historical, coarser naming
    # convention -- preserving the ~20 other call sites in this file that
    # already expect these exact strings (e.g. "system", "risk", "group",
    # "policy"), without requiring each of them to be individually
    # audited and rewritten. utils/semantic_relation_policy.py performs
    # the equivalent (much smaller) translation to ITS OWN historical
    # convention -- see entity_family there. Both translations start from
    # the SAME underlying classify_entity_semantic_family() decision, so
    # the two files cannot disagree about what an entity fundamentally
    # is; they only differ in which cosmetic name they print for it.
    _CANONICAL_TO_LEGACY_FAMILY: Dict[str, str] = {
        "discourse": "discourse", "vote": "vote", "resource": "resource",
        "dataset": "dataset", "domain": "concept", "person": "person",
        "organisation": "organisation", "project": "project",
        "paper_or_analysis": "project", "classification": "classification",
        "policy_or_framework": "policy", "population_or_group": "group",
        "risk_or_harm": "risk", "system_or_model": "system",
        "algorithm_or_process": "system", "concept": "concept",
        "unknown": "unknown",
    }

    def _entity_semantic_family(self, entity: Optional[Entity]) -> str:
        """Delegates the actual classification decision (ontology/entity
        type against the active domain profile, then a small set of
        profile-driven lexical fallbacks) to the ONE shared classifier,
        domain_profiles.classify_entity_semantic_family -- the same
        function utils/semantic_relation_policy.py::entity_family calls.
        Neither file hardcodes any type-name/label-vocabulary mapping of
        its own any more. The result is translated to this file's
        historical bucket names (see _CANONICAL_TO_LEGACY_FAMILY) purely
        so the many pre-existing call sites elsewhere in this file that
        already branch on those exact strings keep working unchanged."""
        canonical = classify_entity_semantic_family(entity, profile=self.domain_profile)
        return self._CANONICAL_TO_LEGACY_FAMILY.get(canonical, canonical)

    def _is_impact_actor_family(self, family: str) -> bool:
        return family in {"system", "policy", "dataset", "risk", "concept"}

    def _is_impact_actor_entity(self, entity: Optional[Entity]) -> bool:
        family = self._entity_semantic_family(entity)
        if family == "risk" and self._is_generic_risk_label(str(getattr(entity, "label", "") or "")):
            return False
        return self._is_impact_actor_family(family)

    def _is_impact_target_entity(self, entity: Optional[Entity], predicate: str) -> bool:
        family = self._entity_semantic_family(entity)
        if family in {"discourse", "vote", "resource", "person", "organisation", "project", "classification"}:
            return False
        if predicate == "particularlyAffects":
            return family == "group"
        return family in {"group", "concept", "risk", "system", "policy"}

    def _impact_actor_priority(self, entity: Optional[Entity]) -> int:
        family = self._entity_semantic_family(entity)
        return {
            "system": 0,
            "policy": 1,
            "dataset": 2,
            "risk": 3,
            "concept": 4,
        }.get(family, 99)

    def _impact_target_priority(self, entity: Optional[Entity]) -> int:
        family = self._entity_semantic_family(entity)
        return {
            "group": 0,
            "concept": 1,
            "risk": 2,
            "system": 3,
            "policy": 4,
        }.get(family, 99)

    def _predicate_domain_range_valid(self, relation: Relation, entities_by_uri: Dict[str, Entity]) -> bool:
        pred = str(getattr(relation, "relation", "") or "").strip()
        src = entities_by_uri.get(str(getattr(relation, "source", "") or ""))
        tgt = entities_by_uri.get(str(getattr(relation, "target", "") or ""))
        src_family = self._entity_semantic_family(src)
        tgt_family = self._entity_semantic_family(tgt)

        if src_family in {"discourse", "vote"} or tgt_family in {"discourse", "vote"}:
            return False

        if pred == "hasRisk":
            return (
                src_family in {"system", "organisation", "dataset", "policy"}
                and tgt_family == "risk"
                and not self._is_generic_risk_label(str(getattr(tgt, "label", "") or ""))
            )

        if pred == "exhibitsRisk":
            return (
                src_family in {"system", "organisation", "dataset", "policy"}
                and tgt_family == "risk"
                and not self._is_generic_risk_label(str(getattr(tgt, "label", "") or ""))
            )

        if pred == "trainedOn":
            return src_family == "system" and tgt_family == "dataset"

        if pred == "reflects":
            return src_family in {"system", "dataset", "concept", "policy"} and tgt_family in {"risk", "concept", "system"}

        if pred == "causes":
            return src_family not in {"dataset", "resource", "classification", "person"} and tgt_family not in {"resource", "organisation"}

        if pred == "causedBy":
            return tgt_family not in {"dataset", "resource", "classification", "person"} and src_family not in {"resource", "organisation"}

        if pred == "affects":
            return self._is_impact_actor_entity(src) and self._is_impact_target_entity(tgt, pred)

        if pred == "affectedBy":
            return self._is_impact_target_entity(src, "affects") and self._is_impact_actor_entity(tgt)

        if pred == "particularlyAffects":
            return self._is_impact_actor_entity(src) and self._is_impact_target_entity(tgt, pred)

        if pred == "mitigates":
            return src_family in {"system", "organisation", "policy", "concept"} and tgt_family in {"risk", "concept"}

        if pred in {"relatedTo", "associatedWith"}:
            return src_family not in {"discourse", "vote"} and tgt_family not in {"discourse", "vote"}

        if pred == "partOfDomain":
            src_type = str(getattr(src, "entity_type", "") or "").strip().lower()
            tgt_type = str(getattr(tgt, "entity_type", "") or "").strip().lower()
            domainish_targets = {"domain", "criminaljustice", "justicedomain", "employment", "healthcare", "highrisk", "highriskai"}
            return (
                src_family in {"system", "dataset", "concept", "policy"}
                and not self._is_generic_abstract_label(str(getattr(src, "label", "") or ""))
                and tgt_family not in {"discourse", "vote", "resource", "person", "organisation", "risk"}
                and (tgt_type in domainish_targets or "domain" in str(getattr(tgt, "label", "") or "").lower())
            )

        return True

    def _make_lexical_relation(
        self,
        *,
        source_uri: str,
        predicate: str,
        target_uri: str,
        discourse_id: str,
        evidence: str,
        confidence: float,
        field: str,
        origin_key: str,
        unit: Dict[str, Any],
        sent_start: int,
        sent_end: int,
        unit_text: str,
        llm_agreement: bool,
    ) -> Relation:
        relation = make_relation(
            source_uri,
            predicate,
            target_uri,
            discourse_id=discourse_id,
            evidence=evidence,
            extractor="lexical",
            confidence=confidence,
            context_source="lexical_sentence",
        )
        relation.context = unit_text
        meta = getattr(relation, "metadata", None) or {}
        meta["evidence_quote"] = evidence
        meta["field"] = str(field or "").strip()
        meta["source_field"] = str(field or "").strip()
        meta["origin_key"] = origin_key
        meta["source_doc_id"] = str(discourse_id or "")
        meta["section"] = str(unit.get("section") or "")
        meta["section_category"] = str(unit.get("section_category") or "")
        meta["semantic_unit_id"] = str(unit.get("unit_id") or "")
        meta["semantic_unit_type"] = str(unit.get("unit_type") or "")
        meta["relation_kind"] = "extracted_semantic"
        meta["char_start"] = int(sent_start)
        meta["char_end"] = int(sent_end)
        meta["start_offset"] = int(sent_start)
        meta["end_offset"] = int(sent_end)
        meta["llm_semantic_agreement"] = bool(llm_agreement)
        relation.metadata = meta
        self._get_semantic_policy().apply_relation_metadata(
            relation,
            field=str(field or "").strip(),
            evidence_text=unit_text or evidence,
        )
        return relation

    def _relation_source_priority(self, predicate: str, entity: Optional[Entity]) -> int:
        family = self._entity_semantic_family(entity)
        if predicate in {"affects", "particularlyAffects", "trainedOn", "uses", "causes", "mitigates"}:
            order = {"system": 0, "policy": 1, "concept": 2, "dataset": 3, "risk": 4}
            return order.get(family, 99)
        if predicate in {"exhibitsRisk", "reflects"}:
            order = {"system": 0, "dataset": 1, "concept": 2, "policy": 3, "risk": 4}
            return order.get(family, 99)
        if predicate == "partOfDomain":
            order = {"system": 0, "concept": 1, "policy": 2, "dataset": 3}
            return order.get(family, 99)
        if predicate == "requires":
            order = {"concept": 0, "policy": 1, "system": 2}
            return order.get(family, 99)
        return 99

    def _relation_target_priority(self, predicate: str, entity: Optional[Entity]) -> int:
        family = self._entity_semantic_family(entity)
        if predicate == "trainedOn":
            return {"dataset": 0}.get(family, 99)
        if predicate == "uses":
            return {"dataset": 0, "concept": 1, "system": 2}.get(family, 99)
        if predicate in {"affects", "particularlyAffects"}:
            if predicate == "particularlyAffects":
                return {"group": 0, "concept": 1, "risk": 2, "system": 3}.get(family, 99)
            return {"concept": 0, "group": 1, "risk": 2, "system": 3}.get(family, 99)
        if predicate in {"exhibitsRisk", "reflects", "mitigates", "causes"}:
            return {"risk": 0, "concept": 1, "group": 2}.get(family, 99)
        if predicate == "partOfDomain":
            return {"concept": 0}.get(family, 99)
        if predicate == "requires":
            return {"concept": 0, "policy": 1, "risk": 2}.get(family, 99)
        return 99

    def _best_left_entity_for_predicate(
        self,
        *,
        sentence_entities: List[Entity],
        entity_hits: Dict[str, List[Tuple[int, int, str]]],
        cue_start: int,
        predicate: str,
    ) -> Optional[Entity]:
        choices: List[Tuple[int, int, int, Entity]] = []
        for entity in sentence_entities:
            uri = str(getattr(entity, "uri", "") or "").strip()
            if not uri:
                continue
            left_hits = [hit for hit in entity_hits.get(uri, []) if int(hit[1]) <= cue_start]
            if not left_hits:
                continue
            best_hit = max(left_hits, key=lambda hit: hit[1])
            prio = self._relation_source_priority(predicate, entity)
            if prio >= 99:
                continue
            choices.append((prio, -int(best_hit[1]), len(str(getattr(entity, "label", "") or "").split()) * -1, entity))
        if not choices:
            return None
        return sorted(choices, key=lambda item: (item[0], item[1], item[2], str(getattr(item[3], "label", ""))))[0][3]

    def _right_entities_for_predicate(
        self,
        *,
        sentence_entities: List[Entity],
        entity_hits: Dict[str, List[Tuple[int, int, str]]],
        cue_end: int,
        predicate: str,
        source_uri: str,
        right_limit: int,
    ) -> List[Entity]:
        choices: List[Tuple[int, int, int, Entity]] = []
        for entity in sentence_entities:
            uri = str(getattr(entity, "uri", "") or "").strip()
            if not uri or uri == source_uri:
                continue
            right_hits = [
                hit for hit in entity_hits.get(uri, [])
                if int(hit[0]) >= cue_end and int(hit[0]) < int(right_limit)
            ]
            if not right_hits:
                continue
            prio = self._relation_target_priority(predicate, entity)
            if prio >= 99:
                continue
            best_hit = min(right_hits, key=lambda hit: hit[0])
            choices.append((prio, int(best_hit[0]), len(str(getattr(entity, "label", "") or "").split()) * -1, entity))
        return [item[3] for item in sorted(choices, key=lambda item: (item[0], item[1], item[2], str(getattr(item[3], "label", ""))))]

    @staticmethod
    def _is_contextual_uses_cue(sentence: str, cue_start: int, cue_end: int) -> bool:
        cue_text = str(sentence or "")[int(cue_start):int(cue_end)].lower()
        if cue_text not in {"use", "uses", "used", "using"}:
            return False
        tail = str(sentence or "")[int(cue_end):]
        return bool(re.match(r"^\s+(?:in|for|as|by)\b", tail, flags=re.IGNORECASE))

    @staticmethod
    def _later_stronger_predicate_boundary(sentence: str, cue_end: int, predicate: str) -> int:
        raw_sentence = str(sentence or "")
        if not raw_sentence:
            return 0
        if predicate != "uses":
            return len(raw_sentence)
        tail = raw_sentence[int(cue_end):]
        if not tail:
            return len(raw_sentence)
        patterns = [
            r"\btrained on\b",
            r"\baffected by\b",
            r"\brequire(?:s|d)?\b",
            r"\breflect(?:s|ed|ing)?\b",
            r"\bdiscriminat(?:e|es|ed|ing) against\b",
            r"\bparticularly affect(?:s|ed|ing)?\b",
            r"\bdisproportionately affect(?:s|ed|ing)?\b",
            r"\bperform(?:s|ed)? worse on\b",
            r"\bworse (?:on|for)\b",
        ]
        boundary = len(raw_sentence)
        for pattern in patterns:
            match = re.search(pattern, tail, flags=re.IGNORECASE)
            if match:
                boundary = min(boundary, int(cue_end) + int(match.start()))
        return boundary

    def _cue_guided_candidate_pairs_for_sentence(
        self,
        *,
        sentence: str,
        sentence_entities: List[Entity],
    ) -> List[Dict[str, Any]]:
        if len(sentence_entities) < 2:
            return []

        entity_hits: Dict[str, List[Tuple[int, int, str]]] = {}
        for entity in sentence_entities:
            uri = str(getattr(entity, "uri", "") or "").strip()
            if not uri:
                continue
            hits = self._entity_variant_hits(entity, sentence)
            if hits:
                entity_hits[uri] = hits
        if len(entity_hits) < 2:
            return []

        specs = [
            ("particularlyAffects", re.compile(r"\b(discriminat(?:e|es|ed|ing) against|disproportionately affect(?:s|ed|ing)?|particularly affect(?:s|ed|ing)?|perform(?:s|ed)? worse on|worse (?:on|for))\b", re.I)),
            ("affects", re.compile(r"\b(affect(?:s|ed|ing)?(?!\s+by)|impact(?:s|ed|ing)?(?!\s+by)|influence(?:s|d|ing)?(?!\s+by)|harm(?:s|ed|ing)?(?!\s+by))\b", re.I)),
            ("trainedOn", re.compile(r"\b(trained on|training data|train(?:ed|ing|s)? on)\b", re.I)),
            ("uses", re.compile(r"\b(use|uses|used|using|as proxy for)\b", re.I)),
            ("reflects", re.compile(r"\b(reflect|reflects|reflected|reveals?|mirrors?)\b", re.I)),
            ("exhibitsRisk", re.compile(r"\b(exhibit|exhibits|show|shows|demonstrate|demonstrates)\b", re.I)),
            ("causes", re.compile(r"\b(cause|causes|caused|lead to|leads to|result in|results in)\b", re.I)),
            ("mitigates", re.compile(r"\b(mitigate|mitigates|mitigated|reduce|reduces|address|addresses)\b", re.I)),
            ("requires", re.compile(r"\b(require|requires|required)\b", re.I)),
            ("partOfDomain", re.compile(r"\b(domain(?:s)? like|domains? such as|domain of|domains of|in the domain of|within the domain of|high[- ]stakes domains? like)\b", re.I)),
        ]

        out: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for predicate, pattern in specs:
            for match in pattern.finditer(sentence):
                cue_start = int(match.start())
                cue_end = int(match.end())
                if predicate == "uses" and self._is_contextual_uses_cue(sentence, cue_start, cue_end):
                    continue
                source_entity = self._best_left_entity_for_predicate(
                    sentence_entities=sentence_entities,
                    entity_hits=entity_hits,
                    cue_start=cue_start,
                    predicate=predicate,
                )
                if source_entity is None:
                    continue
                source_uri = str(getattr(source_entity, "uri", "") or "").strip()
                if predicate == "particularlyAffects":
                    right_limit = min(
                        self._comparison_boundary_after(sentence, cue_end),
                        self._subordinate_clause_boundary_after(sentence, cue_end),
                    )
                elif predicate == "affects":
                    right_limit = self._subordinate_clause_boundary_after(sentence, cue_end)
                elif predicate == "exhibitsRisk":
                    right_limit = self._comparison_boundary_after(sentence, cue_end)
                else:
                    right_limit = self._later_stronger_predicate_boundary(sentence, cue_end, predicate)
                targets = self._right_entities_for_predicate(
                    sentence_entities=sentence_entities,
                    entity_hits=entity_hits,
                    cue_end=cue_end,
                    predicate=predicate,
                    source_uri=source_uri,
                    right_limit=right_limit,
                )
                if not targets:
                    continue
                for target_entity in targets:
                    target_uri = str(getattr(target_entity, "uri", "") or "").strip()
                    if not source_uri or not target_uri:
                        continue
                    key = (source_uri, predicate, target_uri)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        {
                            "source_id": source_uri,
                            "source_label": str(getattr(source_entity, "label", "") or ""),
                            "target_id": target_uri,
                            "target_label": str(getattr(target_entity, "label", "") or ""),
                            "predicate_hints": [predicate],
                            "cue": match.group(0),
                            "selection_reason": "cue_guided_local_pair",
                        }
                    )
        return out

    def _lexical_impact_relations_for_sentence(
        self,
        *,
        sentence: str,
        sent_start: int,
        sent_end: int,
        sentence_entities: List[Entity],
        discourse_id: str,
        field: str,
        origin_key: str,
        unit: Dict[str, Any],
        unit_text: str,
        entity_index: Dict[str, Entity],
        llm_relation_keys: Set[Tuple[str, str, str]],
    ) -> List[Relation]:
        if len(sentence_entities) < 2:
            return []

        entity_hits: Dict[str, List[Tuple[int, int, str]]] = {}
        for entity in sentence_entities:
            uri = str(getattr(entity, "uri", "") or "").strip()
            if not uri:
                continue
            hits = self._entity_variant_hits(entity, sentence)
            if hits:
                entity_hits[uri] = hits
        if len(entity_hits) < 2:
            return []

        candidates: Dict[Tuple[str, str, str], Relation] = {}
        active_specs = [
            (
                "particularlyAffects",
                re.compile(
                    r"\b(discriminat(?:e|es|ed|ing) against|disproportionately affect(?:s|ed|ing)?|"
                    r"particularly affect(?:s|ed|ing)?|perform(?:s|ed)? worse on|worse (?:on|for))\b",
                    re.I,
                ),
                0.89,
            ),
            (
                "affects",
                re.compile(
                    r"\b(affect(?:s|ed|ing)?(?!\s+by)|impact(?:s|ed|ing)?(?!\s+by)|"
                    r"influence(?:s|d|ing)?(?!\s+by)|harm(?:s|ed|ing)?(?!\s+by))\b",
                    re.I,
                ),
                0.86,
            ),
        ]
        passive_specs = [
            (
                "particularlyAffects",
                re.compile(r"\b(disproportionately affected by|particularly affected by)\b", re.I),
                0.89,
            ),
            (
                "affects",
                re.compile(
                    r"\b(affected by|impacted by|influenced by|targeted by|harmed by|burdened by|disadvantaged by)\b",
                    re.I,
                ),
                0.86,
            ),
        ]

        def _active_source() -> Optional[Tuple[Entity, Tuple[int, int, str]]]:
            choices: List[Tuple[int, int, Entity, Tuple[int, int, str]]] = []
            for entity in sentence_entities:
                uri = str(getattr(entity, "uri", "") or "").strip()
                if not uri or not self._is_impact_actor_entity(entity):
                    continue
                left_hits = [hit for hit in entity_hits.get(uri, []) if int(hit[1]) <= cue_start]
                if not left_hits:
                    continue
                best_hit = max(left_hits, key=lambda hit: hit[1])
                choices.append((self._impact_actor_priority(entity), -int(best_hit[1]), entity, best_hit))
            if not choices:
                return None
            _, _, entity, hit = sorted(choices, key=lambda item: (item[0], item[1]))[0]
            return entity, hit

        def _active_targets(predicate_name: str, source_uri: str, right_limit: int) -> List[Entity]:
            choices: List[Tuple[int, int, Entity]] = []
            for entity in sentence_entities:
                uri = str(getattr(entity, "uri", "") or "").strip()
                if not uri or uri == source_uri or not self._is_impact_target_entity(entity, predicate_name):
                    continue
                right_hits = [
                    hit
                    for hit in entity_hits.get(uri, [])
                    if int(hit[0]) >= cue_end and int(hit[0]) < int(right_limit)
                ]
                if not right_hits:
                    continue
                best_hit = min(right_hits, key=lambda hit: hit[0])
                choices.append((self._impact_target_priority(entity), int(best_hit[0]), entity))
            return [item[2] for item in sorted(choices, key=lambda item: (item[0], item[1]))]

        def _passive_source() -> Optional[Entity]:
            choices: List[Tuple[int, int, Entity]] = []
            for entity in sentence_entities:
                uri = str(getattr(entity, "uri", "") or "").strip()
                if not uri or not self._is_impact_actor_entity(entity):
                    continue
                right_hits = [hit for hit in entity_hits.get(uri, []) if int(hit[0]) >= cue_end]
                if not right_hits:
                    continue
                best_hit = min(right_hits, key=lambda hit: hit[0])
                choices.append((self._impact_actor_priority(entity), int(best_hit[0]), entity))
            if not choices:
                return None
            return sorted(choices, key=lambda item: (item[0], item[1]))[0][2]

        def _passive_target(predicate_name: str, source_uri: str) -> Optional[Entity]:
            choices: List[Tuple[int, int, Entity]] = []
            for entity in sentence_entities:
                uri = str(getattr(entity, "uri", "") or "").strip()
                if not uri or uri == source_uri or not self._is_impact_target_entity(entity, predicate_name):
                    continue
                left_hits = [hit for hit in entity_hits.get(uri, []) if int(hit[1]) <= cue_start]
                if not left_hits:
                    continue
                best_hit = max(left_hits, key=lambda hit: hit[1])
                choices.append((self._impact_target_priority(entity), -int(best_hit[1]), entity))
            if not choices:
                return None
            return sorted(choices, key=lambda item: (item[0], item[1]))[0][2]

        for predicate, pattern, confidence in active_specs:
            for match in pattern.finditer(sentence):
                cue_start, cue_end = int(match.start()), int(match.end())
                if predicate == "affects":
                    cue_window = sentence[max(0, cue_start - 24):min(len(sentence), cue_end + 24)]
                    if self._is_disparate_impact_text(cue_window):
                        continue
                source_match = _active_source()
                if source_match is None:
                    continue
                source_entity, _ = source_match
                source_uri = str(getattr(source_entity, "uri", "") or "").strip()
                if predicate == "particularlyAffects":
                    right_limit = min(
                        self._comparison_boundary_after(sentence, cue_end),
                        self._subordinate_clause_boundary_after(sentence, cue_end),
                    )
                elif predicate == "affects":
                    right_limit = self._subordinate_clause_boundary_after(sentence, cue_end)
                else:
                    right_limit = len(sentence)
                targets = _active_targets(predicate, source_uri, right_limit)
                if not targets:
                    continue
                for target_entity in targets:
                    target_uri = str(getattr(target_entity, "uri", "") or "").strip()
                    if not target_uri:
                        continue
                    key = (source_uri, predicate, target_uri)
                    if key in llm_relation_keys:
                        continue
                    relation = self._make_lexical_relation(
                        source_uri=source_uri,
                        predicate=predicate,
                        target_uri=target_uri,
                        discourse_id=discourse_id,
                        evidence=sentence,
                        confidence=confidence,
                        field=field,
                        origin_key=origin_key,
                        unit=unit,
                        sent_start=sent_start,
                        sent_end=sent_end,
                        unit_text=unit_text,
                        llm_agreement=key in llm_relation_keys,
                    )
                    if not self._predicate_domain_range_valid(relation, entity_index):
                        continue
                    existing = candidates.get(key)
                    if existing is None or float(getattr(relation, "confidence", 0.0) or 0.0) > float(getattr(existing, "confidence", 0.0) or 0.0):
                        candidates[key] = relation

        for predicate, pattern, confidence in passive_specs:
            for match in pattern.finditer(sentence):
                cue_start, cue_end = int(match.start()), int(match.end())
                source_entity = _passive_source()
                if source_entity is None:
                    continue
                source_uri = str(getattr(source_entity, "uri", "") or "").strip()
                target_entity = _passive_target(predicate, source_uri)
                if target_entity is None:
                    continue
                target_uri = str(getattr(target_entity, "uri", "") or "").strip()
                if not source_uri or not target_uri:
                    continue
                key = (source_uri, predicate, target_uri)
                if key in llm_relation_keys:
                    continue
                relation = self._make_lexical_relation(
                    source_uri=source_uri,
                    predicate=predicate,
                    target_uri=target_uri,
                    discourse_id=discourse_id,
                    evidence=sentence,
                    confidence=confidence,
                    field=field,
                    origin_key=origin_key,
                    unit=unit,
                    sent_start=sent_start,
                    sent_end=sent_end,
                    unit_text=unit_text,
                    llm_agreement=key in llm_relation_keys,
                )
                if not self._predicate_domain_range_valid(relation, entity_index):
                    continue
                existing = candidates.get(key)
                if existing is None or float(getattr(relation, "confidence", 0.0) or 0.0) > float(getattr(existing, "confidence", 0.0) or 0.0):
                    candidates[key] = relation

        return list(candidates.values())

    def _heading_led_impact_relations_for_unit(
        self,
        *,
        unit_text: str,
        entities: List[Entity],
        discourse_id: str,
        field: str,
        origin_key: str,
        unit: Dict[str, Any],
        llm_relation_keys: Set[Tuple[str, str, str]],
    ) -> List[Relation]:
        section_name = str(unit.get("section") or "").strip()
        if not section_name or section_name in {"Body", "Description"}:
            return []
        source_surface = section_name
        if self._is_generic_abstract_label(source_surface) and not re.search(r"\bbias|harm|impact|discrimination\b", source_surface, flags=re.I):
            return []

        raw_unit_text = str(unit.get("text") or unit_text or "").strip()
        source_text = raw_unit_text or str(unit_text or "").strip()
        source_entity = self._resolve_surface_entity(
            surface=source_surface,
            preferred_type=self._infer_surface_entity_type(source_surface) or "Concept",
            local_entities=entities,
            discourse_id=discourse_id,
            field=field,
            origin_key=origin_key,
            unit=unit,
            unit_text=source_text,
        )
        if source_entity is None or not self._is_impact_actor_entity(source_entity):
            return []

        relations: Dict[Tuple[str, str, str], Relation] = {}
        sentence_spans = self._split_sentence_spans(str(unit_text or ""))
        for sent_start, sent_end, sentence in sentence_spans:
            prefix_match = re.match(
                rf"^\s*{re.escape(section_name)}\s*:\s*(?P<body>.+)$",
                sentence,
                flags=re.IGNORECASE,
            )
            body = str(prefix_match.group("body") if prefix_match else sentence).strip()
            if not body:
                continue
            predicate_specs = [
                (
                    "particularlyAffects",
                    re.compile(
                        r"\b(discriminat(?:e|es|ed|ing) against|disproportionately affect(?:s|ed|ing)?|"
                        r"particularly affect(?:s|ed|ing)?)\b",
                        re.I,
                    ),
                    0.84,
                ),
                (
                    "affects",
                    re.compile(
                        r"\b(affect(?:s|ed|ing)?(?!\s+by)|impact(?:s|ed|ing)?(?!\s+by)|"
                        r"influence(?:s|d|ing)?(?!\s+by)|harm(?:s|ed|ing)?(?!\s+by))\b",
                        re.I,
                    ),
                    0.81,
                ),
            ]
            for predicate, pattern, confidence in predicate_specs:
                match = pattern.search(body)
                if not match:
                    continue
                tail = body[match.end():].strip(" \t\r\n:,-")
                if not tail:
                    continue
                tail = tail[: self._subordinate_clause_boundary_after(tail, 0)]
                candidate_targets = self._coordinated_surface_entity_splits(tail)
                if not candidate_targets:
                    inferred_type = self._infer_surface_entity_type(tail)
                    bucket = self._surface_entity_bucket(tail)
                    if inferred_type is None or not self._surface_entity_quality_ok(tail, bucket=bucket):
                        continue
                    candidate_targets = [(tail, inferred_type, 0)]
                for target_surface, target_type, _target_delta in candidate_targets:
                    target_entity = self._resolve_surface_entity(
                        surface=target_surface,
                        preferred_type=target_type,
                        local_entities=entities,
                        discourse_id=discourse_id,
                        field=field,
                        origin_key=origin_key,
                        unit=unit,
                        unit_text=source_text,
                    )
                    if target_entity is None or not self._is_impact_target_entity(target_entity, predicate):
                        continue
                    source_uri = str(getattr(source_entity, "uri", "") or "").strip()
                    target_uri = str(getattr(target_entity, "uri", "") or "").strip()
                    if not source_uri or not target_uri:
                        continue
                    key = (source_uri, predicate, target_uri)
                    if key in llm_relation_keys:
                        continue
                    relation = self._make_lexical_relation(
                        source_uri=source_uri,
                        predicate=predicate,
                        target_uri=target_uri,
                        discourse_id=discourse_id,
                        evidence=sentence,
                        confidence=confidence,
                        field=field,
                        origin_key=origin_key,
                        unit=unit,
                        sent_start=sent_start,
                        sent_end=sent_end,
                        unit_text=str(unit_text or ""),
                        llm_agreement=key in llm_relation_keys,
                    )
                    entity_index = {
                        str(getattr(entity, "uri", "") or "").strip(): entity
                        for entity in entities
                        if str(getattr(entity, "uri", "") or "").strip()
                    }
                    if not self._predicate_domain_range_valid(relation, entity_index):
                        continue
                    existing = relations.get(key)
                    if existing is None or float(getattr(relation, "confidence", 0.0) or 0.0) > float(getattr(existing, "confidence", 0.0) or 0.0):
                        relations[key] = relation
        return list(relations.values())

    def _lexical_semantic_relations_for_unit(
        self,
        *,
        unit_text: str,
        entities: List[Entity],
        discourse_id: str,
        field: str,
        origin_key: str,
        unit: Dict[str, Any],
        llm_relations: Optional[List[Relation]] = None,
    ) -> List[Relation]:
        if not unit_text:
            return []

        entity_index = {
            str(getattr(entity, "uri", "") or ""): entity
            for entity in entities
            if str(getattr(entity, "uri", "") or "").strip()
        }
        predicate_inventory = [
            "trainedOn",
            "reflects",
            "hasRisk",
            "exhibitsRisk",
            "uses",
            "causes",
            "mitigates",
            "supports",
            "criticizes",
            "endorses",
            "disagreesWith",
            "opposes",
            "compliesWith",
            "requiresComplianceWith",
            "requires",
            "violates",
            "referencesInstrument",
            "referencesSource",
            "cites",
            "partOfDomain",
        ]
        confidence_by_predicate = {
            "trainedOn": 0.90,
            "reflects": 0.90,
            "hasRisk": 0.88,
            "exhibitsRisk": 0.88,
            "uses": 0.87,
            "causes": 0.88,
            "mitigates": 0.88,
            "supports": 0.85,
            "criticizes": 0.85,
            "endorses": 0.85,
            "disagreesWith": 0.84,
            "opposes": 0.84,
            "compliesWith": 0.86,
            "requiresComplianceWith": 0.87,
            "requires": 0.85,
            "violates": 0.87,
            "referencesInstrument": 0.83,
            "referencesSource": 0.83,
            "cites": 0.83,
            "partOfDomain": 0.82,
        }
        llm_relation_keys: Set[Tuple[str, str, str]] = set()
        for relation in llm_relations or []:
            llm_relation_keys.add(
                (
                    str(getattr(relation, "source", "") or "").strip(),
                    str(getattr(relation, "relation", "") or "").strip(),
                    str(getattr(relation, "target", "") or "").strip(),
                )
            )

        candidates: Dict[Tuple[str, str, str], Relation] = {}
        if len(entity_index) < 2:
            return self._heading_led_impact_relations_for_unit(
                unit_text=unit_text,
                entities=entities,
                discourse_id=discourse_id,
                field=field,
                origin_key=origin_key,
                unit=unit,
                llm_relation_keys=llm_relation_keys,
            )
        for sent_start, sent_end, sentence in self._split_sentence_spans(unit_text):
            sentence_entities = [
                entity for entity in entities
                if self._entity_supported_in_text(entity, sentence)
            ]
            if len(sentence_entities) < 2:
                continue

            for relation in self._lexical_impact_relations_for_sentence(
                sentence=sentence,
                sent_start=sent_start,
                sent_end=sent_end,
                sentence_entities=sentence_entities,
                discourse_id=discourse_id,
                field=str(field or "").strip(),
                origin_key=origin_key,
                unit=unit,
                unit_text=unit_text,
                entity_index=entity_index,
                llm_relation_keys=llm_relation_keys,
            ):
                key = (
                    str(getattr(relation, "source", "") or "").strip(),
                    str(getattr(relation, "relation", "") or "").strip(),
                    str(getattr(relation, "target", "") or "").strip(),
                )
                existing = candidates.get(key)
                if existing is None or float(getattr(relation, "confidence", 0.0) or 0.0) > float(getattr(existing, "confidence", 0.0) or 0.0):
                    candidates[key] = relation

            for pair in self._cue_guided_candidate_pairs_for_sentence(
                sentence=sentence,
                sentence_entities=sentence_entities,
            ):
                source_uri = str(pair.get("source_id") or "").strip()
                target_uri = str(pair.get("target_id") or "").strip()
                if not source_uri or not target_uri or source_uri == target_uri:
                    continue
                source_entity = entity_index.get(source_uri)
                target_entity = entity_index.get(target_uri)
                if source_entity is None or target_entity is None:
                    continue
                source_variants = self._entity_surface_variants(source_entity)
                target_variants = self._entity_surface_variants(target_entity)
                if not source_variants or not target_variants:
                    continue
                for predicate in pair.get("predicate_hints") or predicate_inventory:
                    if predicate not in predicate_inventory:
                        continue
                    if not self._predicate_locally_entailed(predicate, source_variants, target_variants, sentence):
                        continue
                    key = (source_uri, predicate, target_uri)
                    if key in llm_relation_keys:
                        continue
                    relation = self._make_lexical_relation(
                        source_uri=source_uri,
                        predicate=predicate,
                        target_uri=target_uri,
                        discourse_id=discourse_id,
                        evidence=sentence,
                        confidence=confidence_by_predicate.get(predicate, 0.84),
                        field=str(field or "").strip(),
                        origin_key=origin_key,
                        unit=unit,
                        sent_start=sent_start,
                        sent_end=sent_end,
                        unit_text=unit_text,
                        llm_agreement=key in llm_relation_keys,
                    )
                    if not self._predicate_domain_range_valid(relation, entity_index):
                        continue

                    existing = candidates.get(key)
                    if existing is None or float(getattr(relation, "confidence", 0.0) or 0.0) > float(getattr(existing, "confidence", 0.0) or 0.0):
                        candidates[key] = relation

        return list(candidates.values())

    def _surface_entity_head_buckets(self) -> Dict[str, Set[str]]:
        """CORE mechanism, thin accessor: the head-word bucket vocabulary
        lives entirely in the active domain profile
        (DomainProfile.surface_type_buckets) -- this method contains none
        itself. See _infer_surface_entity_type/_surface_entity_bucket."""
        return self.domain_profile.surface_type_buckets

    def _surface_entity_boundary_stops(self) -> Set[str]:
        # CORE English function/connector words (unchanged) unioned with
        # the active profile's own domain-specific boundary-stop terms
        # (see domain_profiles.py::AI_ETHICS_PROFILE's
        # "surface_boundary_stop_terms" gazetteer) -- this method no
        # longer hardcodes any AI-ethics vocabulary itself.
        return self._CORE_SURFACE_ENTITY_BOUNDARY_STOPS | self.domain_profile.gazetteer("surface_boundary_stop_terms")

    _CORE_SURFACE_ENTITY_BOUNDARY_STOPS: Set[str] = {
            "a",
            "an",
            "the",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "am",
            "do",
            "does",
            "did",
            "if",
            "when",
            "while",
            "because",
            "despite",
            "even",
            "can",
            "could",
            "may",
            "might",
            "should",
            "would",
            "will",
            "shall",
            "not",
            "why",
            "what",
            "how",
            "show",
            "shows",
            "showed",
            "showing",
            "exhibit",
            "exhibits",
            "revealed",
            "reveal",
            "reveals",
            "demonstrate",
            "demonstrates",
            "demonstrated",
            "found",
            "results",
            "result",
            "leads",
            "lead",
            "leading",
            "associated",
            "associate",
            "associates",
            "goes",
            "used",
            "uses",
            "use",
            "using",
            "train",
            "trained",
            "trains",
            "training",
            "mitigate",
            "mitigates",
            "mitigated",
            "mitigating",
            "affecting",
            "affected",
            "affects",
            "affect",
            "particularly",
            "disproportionately",
            "systematically",
            "beyond",
            "against",
            "by",
    }

    @staticmethod
    def _surface_entity_connector_tokens() -> Set[str]:
        return {"of", "for", "in", "on", "with", "to", "from", "and", "or", "like"}

    def _infer_surface_entity_type(self, surface: str) -> Optional[str]:
        """CORE mechanism: walk the active profile's surface-type buckets,
        in priority (declaration) order, and return the precise type for
        the first bucket whose head words (or whole-phrase signal)
        matches. Every concrete type name and word list is profile data
        (see domain_profiles.py::DomainProfile.surface_type_buckets/
        surface_type_overrides/surface_type_bucket_defaults/
        surface_type_phrase_signals) -- this method does not know
        "AlgorithmicBias", "DemographicGroup", or any other domain class
        by name."""
        tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", str(surface or ""))
        if not tokens:
            return None
        norm_tokens = [normalize_text(tok) for tok in tokens]
        head = norm_tokens[-1]
        surface_norm = normalize_text(surface)
        profile = self.domain_profile
        for bucket_name, head_words in profile.surface_type_buckets.items():
            phrase_signal = profile.surface_type_phrase_signals.get(bucket_name)
            if head in head_words or (phrase_signal is not None and phrase_signal.search(surface_norm)):
                override = profile.surface_type_overrides.get(head)
                if override:
                    return override
                return profile.surface_type_bucket_defaults.get(bucket_name)
        return None

    def _surface_entity_bucket(self, surface: str) -> Optional[str]:
        """CORE mechanism: return the (profile-configured) bucket name
        whose head words contain this surface phrase's head token, or
        None. See _infer_surface_entity_type for the precise-type lookup
        built on top of this."""
        tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", str(surface or ""))
        if not tokens:
            return None
        head = normalize_text(tokens[-1])
        for bucket_name, head_words in self.domain_profile.surface_type_buckets.items():
            if head in head_words:
                return bucket_name
        return None

    def _trim_recovered_surface(
        self,
        phrase: str,
        *,
        bucket: Optional[str],
        head_norm: str,
    ) -> Tuple[str, int]:
        raw = str(phrase or "")
        trimmed = raw.strip()
        offset_delta = raw.find(trimmed) if trimmed else 0

        # CORE mechanism: strip every profile-configured prefix pattern
        # for THIS bucket, whatever its name (e.g. AI-ethics's "racial/
        # systematic/... bias/discrimination/disparities in " for its
        # "system" bucket -- see
        # domain_profiles.py::AI_ETHICS_PROFILE.surface_trim_prefix_patterns);
        # this method does not hardcode any such phrase itself, nor which
        # bucket name it applies to.
        if bucket is not None:
            for prefix_pattern in self.domain_profile.surface_trim_prefix_patterns.get(bucket, []):
                trimmed = prefix_pattern.sub("", trimmed).strip()

        if bucket == "system":
            trimmed = re.sub(r"^(?:in|of)\s+", "", trimmed, flags=re.IGNORECASE).strip()
            token_matches = list(re.finditer(r"[A-Za-z0-9][\w'/-]*", trimmed))
            if token_matches:
                normalized_tokens = [normalize_text(tok.group(0)) for tok in token_matches]
                try:
                    head_index = normalized_tokens.index(head_norm)
                except ValueError:
                    head_index = -1
                if 0 <= head_index < len(token_matches):
                    end = int(token_matches[head_index].end())
                    tail = trimmed[end:]
                    if tail and re.match(
                        r"^\s+(?:in\s+place\s+to|to|used(?:\s+to)?|using|which|that|by\s+[A-Za-z][\w-]*ing)\b",
                        tail,
                        flags=re.IGNORECASE,
                    ):
                        trimmed = trimmed[:end].strip()
        elif bucket == "concept":
            head_match = None
            for match in re.finditer(rf"\b{re.escape(head_norm)}\b", normalize_text(trimmed), flags=re.IGNORECASE):
                head_match = match
            if head_match is not None:
                token_matches = list(re.finditer(r"[A-Za-z0-9][\w'/-]*", trimmed))
                if token_matches:
                    normalized_tokens = [normalize_text(tok.group(0)) for tok in token_matches]
                    try:
                        head_index = normalized_tokens.index(head_norm)
                    except ValueError:
                        head_index = -1
                    if head_index >= 0:
                        end = int(token_matches[head_index].end())
                        tail = trimmed[end:]
                        if not re.match(r"^\s+of\b", tail, flags=re.IGNORECASE):
                            trimmed = trimmed[:end].strip()
        elif bucket in {"risk", "group"}:
            tail_match = re.search(
                r"(?:^|.*\b(?:to|that|which)\s+[A-Za-z][\w-]*\s+)(.+)$",
                trimmed,
                flags=re.IGNORECASE,
            )
            if tail_match:
                candidate = tail_match.group(1).strip()
                if candidate:
                    candidate_tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", candidate)
                    if candidate_tokens and normalize_text(candidate_tokens[-1]) == head_norm:
                        trimmed = candidate
                        offset_delta = raw.lower().find(candidate.lower())
        elif bucket == "group" and head_norm in {"population", "populations", "community", "communities"}:
            with_match = re.search(r"\bwith\s+(.+)$", trimmed, flags=re.IGNORECASE)
            if with_match and re.search(r"\b(population|populations|community|communities)\b", with_match.group(1), flags=re.IGNORECASE):
                leading = trimmed.find(with_match.group(1))
                if leading >= 0:
                    offset_delta += leading
                    trimmed = with_match.group(1).strip()

        final_delta = raw.find(trimmed) if trimmed and trimmed in raw else offset_delta
        return trimmed, max(0, final_delta)

    def _expand_surface_entity_span(
        self,
        sentence: str,
        token_matches: List[re.Match],
        index: int,
    ) -> Optional[Tuple[int, int]]:
        if index < 0 or index >= len(token_matches):
            return None
        stops = self._surface_entity_boundary_stops()
        connectors = self._surface_entity_connector_tokens()

        start_idx = index
        while start_idx > 0:
            prev = token_matches[start_idx - 1]
            current = token_matches[start_idx]
            prev_norm = normalize_text(prev.group(0))
            gap = sentence[int(prev.end()):int(current.start())]
            if any(mark in gap for mark in ",;:!?"):
                break
            if prev_norm in stops:
                break
            if prev_norm in connectors or re.match(r"[A-Za-z0-9]", prev.group(0)):
                start_idx -= 1
                continue
            break

        while start_idx < index and normalize_text(token_matches[start_idx].group(0)) in connectors:
            start_idx += 1

        end_idx = index
        while end_idx + 1 < len(token_matches):
            current = token_matches[end_idx]
            nxt = token_matches[end_idx + 1]
            next_norm = normalize_text(nxt.group(0))
            gap = sentence[int(current.end()):int(nxt.start())]
            if any(mark in gap for mark in ",;:!?"):
                break
            if next_norm in stops:
                break
            if next_norm in connectors or re.match(r"[A-Za-z0-9]", nxt.group(0)):
                end_idx += 1
                continue
            break

        start = int(token_matches[start_idx].start())
        end = int(token_matches[end_idx].end())
        phrase = sentence[start:end].strip()
        if len(re.findall(r"[A-Za-z0-9][\w'/-]*", phrase)) > 8:
            return None
        return start, end

    def _surface_entity_candidates_for_sentence(
        self,
        sentence: str,
        sentence_start: int,
    ) -> List[Dict[str, Any]]:
        token_matches = list(re.finditer(r"[A-Za-z0-9][\w'/-]*", str(sentence or "")))
        if not token_matches:
            return []
        buckets = self._surface_entity_head_buckets()
        all_heads = set().union(*buckets.values())
        seen: Set[Tuple[int, int]] = set()
        out: List[Dict[str, Any]] = []
        for idx, token in enumerate(token_matches):
            norm = normalize_text(token.group(0))
            if norm not in all_heads:
                continue
            span = self._expand_surface_entity_span(sentence, token_matches, idx)
            if span is None:
                continue
            start, end = span
            if (start, end) in seen:
                continue
            surface = sentence[start:end].strip(" \t\n\r\"'")
            bucket = self._surface_entity_bucket(surface)
            if bucket is None:
                for bucket_name, heads in buckets.items():
                    if norm in heads:
                        bucket = bucket_name
                        break
            surface, delta = self._trim_recovered_surface(surface, bucket=bucket, head_norm=norm)
            if delta:
                start += int(delta)
            entity_type = self._infer_surface_entity_type(surface)
            if not surface or entity_type is None:
                continue
            if (
                len(surface.split()) == 1
                and self._generic_surface_label(surface)
                and entity_type not in self.domain_profile.gazetteer("generic_label_exempt_types")
            ):
                continue
            if not self._surface_entity_quality_ok(surface, bucket=bucket):
                continue
            seen.add((start, end))
            out.append(
                {
                    "surface": surface,
                    "start_char": int(sentence_start + start),
                    "end_char": int(sentence_start + end),
                    "entity_type": entity_type,
                    "source": "surface_recovery",
                }
            )
            for split_surface, split_type, split_delta in self._coordinated_surface_entity_splits(surface):
                split_start = int(start + split_delta)
                split_end = int(split_start + len(split_surface))
                if (split_start, split_end) in seen:
                    continue
                seen.add((split_start, split_end))
                out.append(
                    {
                        "surface": split_surface,
                        "start_char": int(sentence_start + split_start),
                        "end_char": int(sentence_start + split_end),
                        "entity_type": split_type,
                        "source": "surface_recovery",
                    }
                )
        return out

    def _semantic_surface_entity_candidates(
        self,
        *,
        text: str,
        field: str,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for unit in SemanticChunker.semantic_relation_units(text, field=field):
            unit_text = str(unit.get("text") or "")
            unit_start = self._coerce_nonneg_int(unit.get("start_offset")) or 0
            for sent_start, _sent_end, sentence in self._split_sentence_spans(unit_text):
                candidates.extend(
                    self._surface_entity_candidates_for_sentence(
                        sentence,
                        unit_start + int(sent_start),
                    )
                )
        return candidates

    def _augment_lexical_entities(
        self,
        *,
        text: str,
        metadata: Dict[str, Any],
        entities: List[Entity],
        section_spans: List[Dict[str, Any]],
        chunk_spans: List[Dict[str, Any]],
    ) -> None:
        raw_text = str(text or "")
        if not raw_text:
            return

        existing_labels = {normalize_text(str(getattr(entity, "label", "") or "")) for entity in entities}
        discourse_id_raw = metadata.get("discourse_id")
        discourse_id = str(discourse_id_raw).strip() if discourse_id_raw is not None else ""
        origin_key = metadata.get("origin_key") or self._compose_origin_key(discourse_id, metadata.get("field"))
        field = str(metadata.get("field") or "").strip()
        # CORE mechanism: the lexical trigger -> entity-type mappings are
        # entirely profile data (see
        # domain_profiles.py::AI_ETHICS_PROFILE.lexical_recovery_terms) --
        # this method scans configured triggers and requests a configured
        # type; it does not hardcode any AI-ethics trigger/type mapping
        # itself.
        lexical_terms = dict(self.domain_profile.lexical_recovery_terms)
        for candidate in self._semantic_surface_entity_candidates(text=raw_text, field=field):
            surface = str(candidate.get("surface") or "").strip()
            entity_type = str(candidate.get("entity_type") or "").strip()
            if not surface or not entity_type:
                continue
            lexical_terms.setdefault(surface.lower(), entity_type)

        for surface, entity_type in lexical_terms.items():
            pattern = rf"(?<!\w){self._surface_variant_pattern(surface)}(?!\w)"
            for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE):
                norm_surface = normalize_text(match.group(0))
                if not norm_surface or norm_surface in existing_labels:
                    continue
                start_char = int(match.start())
                end_char = int(match.end())
                section_span = self._section_span_for_offset(section_spans, start_char, end_char)
                if str(section_span.get("section_category") or "").strip() != "semantic":
                    continue

                entity = Entity(
                    uri=make_uri(f"lexical_{discourse_id}_{entity_type.lower()}_{start_char}_{end_char}"),
                    label=match.group(0),
                    entity_type=entity_type,
                    confidence=0.82,
                    metadata={
                        "doc_id": f"{discourse_id}|{metadata.get('field', 'unknown')}",
                        "source": "lexical",
                        "surface_form": match.group(0),
                        "start_char": start_char,
                        "end_char": end_char,
                        "ontology_uri": "",
                        "discourse_id": discourse_id,
                        "origin_key": str(origin_key) if origin_key is not None else "",
                        "field": field,
                    },
                )

                for span in chunk_spans:
                    chunk_start = self._coerce_nonneg_int(span.get("start_offset"))
                    chunk_end = self._coerce_nonneg_int(span.get("end_offset"))
                    if chunk_start is None or chunk_end is None:
                        continue
                    if int(chunk_start) <= start_char < int(chunk_end):
                        entity.metadata["chunk_index"] = int(span.get("chunk_index", -1))
                        entity.metadata["chunk_total"] = len(chunk_spans)
                        entity.metadata["start_offset"] = int(chunk_start)
                        entity.metadata["end_offset"] = int(chunk_end)
                        break

                self._append_entity_mention(
                    entity,
                    discourse_id=discourse_id,
                    chunk_index=entity.metadata.get("chunk_index", -1),
                    start_char=start_char,
                    end_char=end_char,
                    field=field,
                )
                self._annotate_entity_section_metadata(
                    entity,
                    section_spans=section_spans,
                )
                entities.append(entity)
                existing_labels.add(norm_surface)

    def _attach_local_surface_aliases(
        self,
        *,
        entities: List[Entity],
        discourse_text: str,
        discourse_id: str,
        unit_start: int,
        unit_end: int,
    ) -> None:
        text = str(discourse_text or "")
        did = str(discourse_id or "").strip()
        if not text or unit_end <= unit_start:
            return
        for entity in entities:
            meta = getattr(entity, "metadata", None) or {}
            aliases = meta.get("aliases")
            if not isinstance(aliases, list):
                aliases = []
            # These surfaces are sliced verbatim from the discourse text at
            # the entity's OWN mention spans -- genuinely observed source
            # text, so they are also recorded as trusted identity evidence.
            sga = meta.get("source_grounded_aliases")
            if not isinstance(sga, list):
                sga = []
            mentions = meta.get("mentions")
            if not isinstance(mentions, list):
                mentions = []
            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                mention_did = str(mention.get("discourse_id") or "").strip()
                if did and mention_did and mention_did != did:
                    continue
                start = self._coerce_nonneg_int(mention.get("start_char"))
                end = self._coerce_nonneg_int(mention.get("end_char"))
                if start is None or end is None or end <= start:
                    continue
                if int(start) >= int(unit_end) or int(end) <= int(unit_start):
                    continue
                surface = text[int(start):int(end)].strip()
                if surface and surface != getattr(entity, "label", ""):
                    if surface not in aliases:
                        aliases.append(surface)
                    if surface not in sga:
                        sga.append(surface)
            if aliases:
                meta["aliases"] = aliases
            if sga:
                meta["source_grounded_aliases"] = sga
            entity.metadata = meta

    def _canonicalize_impact_relation(
        self,
        relation: Relation,
        *,
        entities_by_uri: Dict[str, Entity],
        evidence_text: str,
    ) -> bool:
        pred = str(getattr(relation, "relation", "") or "").strip()
        if pred not in {"affectedBy", "affects", "particularlyAffects"}:
            return False

        source_entity = entities_by_uri.get(str(getattr(relation, "source", "") or ""))
        target_entity = entities_by_uri.get(str(getattr(relation, "target", "") or ""))
        if source_entity is None or target_entity is None:
            return False

        meta = getattr(relation, "metadata", None) or {}
        source_variants = self._entity_surface_variants(source_entity)
        target_variants = self._entity_surface_variants(target_entity)
        should_swap = False

        if pred == "affectedBy":
            should_swap = self._predicate_locally_entailed(
                "affectedBy",
                source_variants,
                target_variants,
                evidence_text,
            )
        elif self._entity_semantic_family(source_entity) == "group" and self._is_impact_actor_entity(target_entity):
            should_swap = self._predicate_locally_entailed(
                "affectedBy",
                source_variants,
                target_variants,
                evidence_text,
            )

        if not should_swap:
            return False

        # Both endpoints are already-resolved entities here
        # (source_entity/target_entity above), but the swap must not
        # simply exchange raw string values -- the new target's own
        # literal-state fields (in case any stale value survived onto it
        # from an earlier stage) are explicitly normalized via
        # set_target_entity rather than carried across implicitly.
        new_target_uri = str(getattr(relation, "source", "") or "")
        relation.source = str(getattr(relation, "target", "") or "")
        relation.set_target_entity(new_target_uri)
        relation.relation = "particularlyAffects" if self._is_disparate_impact_text(evidence_text) else "affects"
        meta["impact_direction_normalized"] = "active"
        meta["normalized_from_predicate"] = pred
        relation.metadata = meta
        return True

    def _validate_semantic_relation_candidate(
        self,
        relation: Relation,
        *,
        entities_by_uri: Dict[str, Entity],
        discourse_text: str,
    ) -> Tuple[bool, str]:
        decision = self._get_semantic_policy().validate(
            relation,
            entities_by_uri=entities_by_uri,
            discourse_text=discourse_text,
        )
        meta = getattr(relation, "metadata", None) or {}
        meta.setdefault("source_field", str(meta.get("field") or "").strip())
        meta["semantic_validation_code"] = decision.reason_code
        meta["relation_origin"] = decision.relation_origin or meta.get("relation_origin") or self._get_semantic_policy().infer_relation_origin(relation)
        if decision.predicate_surface_phrase:
            meta["predicate_surface_phrase"] = decision.predicate_surface_phrase
        if decision.predicate_source:
            meta["predicate_policy_source"] = decision.predicate_source
        if decision.locality_scope:
            meta["locality_scope"] = decision.locality_scope
        if decision.direction:
            meta["direction_normalized"] = decision.direction
        if decision.discourse_mode:
            meta["discourse_mode"] = decision.discourse_mode
        if decision.epistemic_status:
            meta["epistemic_status"] = decision.epistemic_status
        if decision.modality_tags:
            meta["modality_tags"] = "|".join(decision.modality_tags)
        if decision.subject_family:
            meta["subject_family"] = decision.subject_family
        if decision.object_family:
            meta["object_family"] = decision.object_family
        if decision.entity_role_check:
            meta["entity_role_check"] = decision.entity_role_check
        if decision.domain_range_check:
            meta["domain_range_check"] = decision.domain_range_check
        if decision.entailment_check:
            meta["entailment_check"] = decision.entailment_check
        if decision.modality_check:
            meta["modality_check"] = decision.modality_check
        meta["semantic_validation_status"] = "accepted" if decision.accepted else "rejected"
        meta["semantic_validation_reason"] = decision.reason_string()
        relation.metadata = meta
        return decision.accepted, decision.reason_string()

    def _ensure_entity_ontology_grounding(self, entity: Entity) -> None:
        """Ensure a single ontology_uri is always present for an entity."""
        if not isinstance(entity, Entity):
            return

        metadata = getattr(entity, "metadata", None) or {}
        entity.metadata = metadata

        # Canonicalize ETD class local-name early so every pipeline branch converges.
        raw_type = str(getattr(entity, "entity_type", "") or "").strip()
        if raw_type:
            canonical_type = canonicalize_class_local(raw_type)
            if canonical_type != raw_type:
                metadata.setdefault("external_entity_type", raw_type)
                entity.entity_type = canonical_type
                if hasattr(entity, "type"):
                    entity.type = canonical_type

        onto_uri = getattr(entity, "ontology_uri", None) or metadata.get("ontology_uri")
        if isinstance(onto_uri, str):
            onto_uri = onto_uri.strip()
        else:
            onto_uri = None

        if not onto_uri and self.config.USE_ONTOLOGY_MAPPING:
            mapped = self.ontology_mapper.map_entity(entity)
            chosen = None
            if isinstance(mapped, str) and mapped.strip():
                chosen = mapped.strip()
            elif isinstance(mapped, list):
                for candidate in mapped:
                    if isinstance(candidate, str) and candidate.strip():
                        chosen = candidate.strip()
                        break
            if chosen:
                onto_uri = expand_curie(chosen)
                metadata.setdefault("ontology_mapping_source", "entity_type_mapping")

        if not onto_uri:
            onto_uri = expand_curie("prov:Entity")
            metadata.setdefault("ontology_mapping_source", "fallback:prov:Entity")

        entity.ontology_uri = onto_uri
        metadata["ontology_uri"] = onto_uri
        metadata["ontology_classes"] = [onto_uri]

    def _coerce_nonneg_int(self, value: Any) -> Optional[int]:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        return iv if iv >= 0 else None

    def _coerce_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_entity_mention(
        self,
        mention: Any,
        discourse_id: Optional[str] = None,
        field: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(mention, dict):
            return None
        did_raw = discourse_id if discourse_id is not None else mention.get("discourse_id")
        did = str(did_raw).strip() if did_raw is not None else ""
        if not did:
            return None
        start = self._coerce_nonneg_int(mention.get("start_char"))
        end = self._coerce_nonneg_int(mention.get("end_char"))
        if start is None or end is None or end <= start:
            return None
        out: Dict[str, Any] = {
            "discourse_id": did,
            "start_char": int(start),
            "end_char": int(end),
        }
        # C.3: (field, discourse_id) travels together on the mention record.
        fld = str(field or mention.get("field") or mention.get("source_field") or "").strip()
        if fld:
            out["field"] = fld
        ci = self._coerce_nonneg_int(mention.get("chunk_index"))
        if ci is not None:
            out["chunk_index"] = int(ci)
        return out

    def _append_entity_mention(
        self,
        entity: Entity,
        *,
        discourse_id: Any,
        chunk_index: Any,
        start_char: Any,
        end_char: Any,
        field: Optional[str] = None,
    ) -> None:
        if not isinstance(getattr(entity, "metadata", None), dict):
            entity.metadata = {}
        did = str(discourse_id).strip() if discourse_id is not None else ""
        mention_field = str(
            field
            or (entity.metadata.get("field") if isinstance(entity.metadata, dict) else "")
            or (entity.metadata.get("source_field") if isinstance(entity.metadata, dict) else "")
            or ""
        ).strip()
        mention = self._normalize_entity_mention(
            {
                "discourse_id": did,
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
                "field": mention_field,
            },
            discourse_id=did,
            field=mention_field,
        )
        if mention is None:
            return
        mentions = entity.metadata.get("mentions")
        if not isinstance(mentions, list):
            mentions = []
        mentions.append(mention)
        normed = [n for n in (self._normalize_entity_mention(item) for item in mentions) if n is not None]
        entity.metadata["mentions"] = dedup_mentions(normed)

    def _candidate_pairs_for_unit(
        self,
        *,
        unit_text: str,
        entities: List[Entity],
    ) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for _sent_start, _sent_end, sentence in self._split_sentence_spans(unit_text):
            sentence_entities = [
                entity for entity in entities
                if self._entity_supported_in_text(entity, sentence)
            ]
            for pair in self._cue_guided_candidate_pairs_for_sentence(
                sentence=sentence,
                sentence_entities=sentence_entities,
            ):
                source_id = str(pair.get("source_id") or "").strip()
                target_id = str(pair.get("target_id") or "").strip()
                if not source_id or not target_id:
                    continue
                hints = list(pair.get("predicate_hints") or [])
                if hints:
                    unique_hints: List[str] = []
                    for predicate in hints:
                        pred = str(predicate or "").strip()
                        if not pred:
                            continue
                        key = (source_id, pred, target_id)
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_hints.append(pred)
                    if not unique_hints:
                        continue
                    pair = dict(pair)
                    pair["predicate_hints"] = unique_hints
                pairs.append(pair)
        return pairs

    def _extract_relations_with_pair_focus(
        self,
        *,
        unit_text: str,
        local_entities: List[Entity],
        discourse_id: str,
        candidate_pairs: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> List[Relation]:
        extractor = getattr(self, "llm_relation_extractor", None)
        if extractor is None:
            return []
        try:
            return extractor.extract(
                unit_text,
                local_entities,
                discourse_id=discourse_id,
                candidate_pairs=candidate_pairs,
                context=context,
            )
        except TypeError:
            pass
        try:
            return extractor.extract(
                unit_text,
                local_entities,
                discourse_id=discourse_id,
                candidate_pairs=candidate_pairs,
            )
        except TypeError:
            return extractor.extract(
                unit_text,
                local_entities,
                discourse_id=discourse_id,
            )

    def _entity_debug_row(self, entity: Entity) -> Dict[str, Any]:
        meta = getattr(entity, "metadata", None) or {}
        mentions = meta.get("mentions")
        spans = []
        if isinstance(mentions, list):
            for mention in mentions[:5]:
                if not isinstance(mention, dict):
                    continue
                start = self._coerce_nonneg_int(mention.get("start_char"))
                end = self._coerce_nonneg_int(mention.get("end_char"))
                if start is None or end is None:
                    continue
                spans.append([int(start), int(end)])
        if not spans:
            start = self._coerce_nonneg_int(meta.get("start_char"))
            end = self._coerce_nonneg_int(meta.get("end_char"))
            if start is not None and end is not None:
                spans.append([int(start), int(end)])
        return {
            "label": str(getattr(entity, "label", "") or ""),
            "uri": str(getattr(entity, "uri", "") or ""),
            "entity_type": str(getattr(entity, "entity_type", "") or ""),
            "semantic_family": self._entity_semantic_family(entity),
            "spans": spans,
            "aliases": list((meta.get("aliases") or [])[:5]) if isinstance(meta.get("aliases"), list) else [],
        }

    def _relation_debug_row(self, relation: Relation, entity_index: Dict[str, Entity]) -> Dict[str, Any]:
        source = entity_index.get(str(getattr(relation, "source", "") or ""))
        target = entity_index.get(str(getattr(relation, "target", "") or ""))
        meta = getattr(relation, "metadata", None) or {}
        return {
            "subject": str(getattr(source, "label", "") or ""),
            "predicate": str(getattr(relation, "relation", "") or ""),
            "object": str(getattr(target, "label", "") or ""),
            "cue": str(meta.get("predicate_surface_phrase") or meta.get("evidence_quote") or ""),
            "extractor": str(meta.get("extractor") or ""),
        }

    def _record_candidate_generation_diag(self, row: Dict[str, Any]) -> None:
        diagnostics = getattr(self, "_candidate_generation_diagnostics", None)
        if not isinstance(diagnostics, list):
            diagnostics = []
            self._candidate_generation_diagnostics = diagnostics
        diagnostics.append(row)

    def _compose_semantic_support_context(
        self,
        *,
        unit_text: str,
        unit: Dict[str, Any],
        section_text: str,
        discourse_text: str,
    ) -> str:
        snippets: List[str] = []
        section = str(unit.get("section") or "").strip()
        base = str(unit_text or "").strip()
        if section and section not in {"Body", "Description"} and not normalize_text(base).startswith(normalize_text(section)):
            snippets.append(f"{section}: {base}")
        elif base:
            snippets.append(base)
        if len(base.split()) < 5 and section_text:
            section_clean = str(section_text).strip()
            if section_clean and normalize_text(section_clean) != normalize_text(base):
                snippets.append(section_clean)
        if not snippets and discourse_text:
            snippets.append(str(discourse_text or "").strip())
        seen: Set[str] = set()
        ordered: List[str] = []
        for snippet in snippets:
            norm = normalize_text(snippet)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            ordered.append(snippet)
        return "\n".join(ordered[:2]).strip()

    def _normalize_surface_key(self, value: Any) -> str:
        return normalize_text(str(value or "").strip())

    def _entity_matches_surface(self, entity: Entity, surface: str) -> bool:
        target = self._normalize_surface_key(surface)
        if not target:
            return False
        for variant in self._entity_surface_variants(entity):
            norm = self._normalize_surface_key(variant)
            if not norm:
                continue
            if norm == target or target in norm or norm in target:
                return True
        return False

    def _infer_assertion_entity_type(self, preferred_type: str, surface: str, bucket_hint: Optional[str] = None) -> str:
        """CORE mechanism: the coarse bucket_hint -> type fallback (used
        only when _infer_surface_entity_type itself found nothing more
        precise) is read from the same profile data
        (surface_type_bucket_defaults) as that function -- no bucket-name
        or type-name is hardcoded here. "Concept" is the one universal,
        cross-domain fallback (see architectural-cleanup guidance: Concept
        is a generic type, not domain vocabulary)."""
        raw = str(preferred_type or "").strip()
        if raw:
            return raw
        inferred = self._infer_surface_entity_type(surface)
        if inferred:
            return inferred
        if bucket_hint:
            default_type = self.domain_profile.surface_type_bucket_defaults.get(bucket_hint)
            if default_type:
                return default_type
        return "Concept"

    def _find_surface_span_in_unit(
        self,
        *,
        surface: str,
        unit_text: str,
        unit_start: int,
    ) -> Tuple[int, int]:
        local_text = str(unit_text or "")
        target = str(surface or "").strip()
        if not target:
            return -1, -1
        idx = local_text.lower().find(target.lower())
        if idx < 0:
            return -1, -1
        return unit_start + idx, unit_start + idx + len(target)

    def _register_assertion_surface_entity(
        self,
        *,
        surface: str,
        entity_type: str,
        discourse_id: str,
        field: str,
        origin_key: str,
        unit: Dict[str, Any],
        unit_text: str,
    ) -> Entity:
        unit_start = self._coerce_nonneg_int(unit.get("start_offset")) or 0
        start_char, end_char = self._find_surface_span_in_unit(
            surface=surface,
            unit_text=unit_text,
            unit_start=unit_start,
        )
        metadata = {
            "surface_form": surface,
            "field": field,
            "source_field": field,
            "context_source": "semantic_assertion_resolution",
            "source": "semantic_assertion_resolution",
            "discourse_id": discourse_id,
            "origin_key": origin_key,
            "source_doc_id": str(discourse_id),
            "section": str(unit.get("section") or ""),
            "section_category": str(unit.get("section_category") or ""),
            "semantic_unit_id": str(unit.get("unit_id") or ""),
            "semantic_unit_type": str(unit.get("unit_type") or ""),
            "chunk_index": -1,
            "chunk_index_unknown": True,
        }
        if start_char >= 0 and end_char > start_char:
            metadata["start_char"] = int(start_char)
            metadata["end_char"] = int(end_char)
            # C.3: carry the coherent (field, discourse_id, span) tuple as a
            # first-class mention record so a later cross-record merge keeps
            # this occurrence instead of overwriting the scalars piecemeal.
            metadata["mentions"] = [
                {
                    "field": field,
                    "discourse_id": str(discourse_id),
                    "start_char": int(start_char),
                    "end_char": int(end_char),
                }
            ]
        ent = Entity(
            uri=make_uri(f"surface_{make_uri_safe(surface)}_{make_uri_safe(entity_type)}"),
            label=surface,
            entity_type=entity_type,
            confidence=0.86,
            metadata=metadata,
        )
        self.entity_normalizer.normalize_entity(ent)
        ent.uri = self.entity_normalizer.canonical_uri(
            label=ent.label,
            entity_type=ent.entity_type,
            ontology_uri=getattr(ent, "ontology_uri", None),
        )
        ent = self.global_registry.get_or_create(ent)
        self._ensure_entity_ontology_grounding(ent)
        return ent

    def _resolve_surface_entity(
        self,
        *,
        surface: str,
        preferred_type: str,
        local_entities: List[Entity],
        discourse_id: str,
        field: str,
        origin_key: str,
        unit: Dict[str, Any],
        unit_text: str,
    ) -> Optional[Entity]:
        surface = str(surface or "").strip()
        if not surface:
            return None
        for entity in local_entities:
            if self._entity_matches_surface(entity, surface):
                return entity

        # Root-cause fix (bounded correction pass, item 3 -- same-chunk
        # identical-referent duplicates): `local_entities` above is
        # scoped to ONE semantic unit's own extraction call, so it
        # cannot see an entity already minted for the identical mention
        # by a DIFFERENT unit/extraction pass over the same discourse
        # (e.g. the same "$765M+ annually..." figure appearing in two
        # sections of one discourse, independently typed "Money" by one
        # pass and "EconomicImpact" by another). Reuse across units is
        # keyed strictly on (discourse_id, normalized label) -- the
        # SAME scope kg_quality_gate's same-chunk-collision check treats
        # as "one referent" -- so it can never merge same-labeled
        # mentions from two different discourses/documents, and it does
        # not touch the separate, deliberately type-sensitive global
        # CanonicalLinker/EntityConsolidator merge passes.
        normalized_label = normalize_text(surface).strip().lower()
        identity_key = (str(discourse_id), normalized_label) if normalized_label else None
        bucket = self._surface_entity_bucket(surface)
        entity_type = self._infer_assertion_entity_type(preferred_type, surface, bucket_hint=bucket)
        if identity_key is not None:
            existing = self._chunk_entity_identity_index.get(identity_key)
            if existing is not None:
                # Root-cause fix (bounded preflight, item 1): reuse must
                # not fire across two structurally-different KINDS of
                # referent that merely happen to share a bare label
                # within one discourse (e.g. a Person and a Dataset both
                # called "Model" by coincidence) -- that is conflicting
                # identity evidence, not a same-referent duplicate. Only
                # a type-bucket disagreement (the actual target shape:
                # "Money" vs "EconomicImpact" for the identical $765M
                # figure) is expected and intentionally overridden by
                # this mechanism; a FAMILY disagreement (via the same
                # shared classifier every other domain/range guard in
                # this codebase already uses) is a much stronger signal
                # that these are genuinely different things, so reuse is
                # refused and a fresh entity is minted instead.
                existing_family = self._entity_semantic_family(existing)
                new_family = classify_entity_semantic_family(
                    SimpleNamespace(entity_type=entity_type, label=surface, metadata={}),
                    profile=self.domain_profile,
                )
                families_conflict = (
                    existing_family not in ("concept", "unknown")
                    and new_family not in ("concept", "unknown")
                    and existing_family != new_family
                )
                if not families_conflict:
                    if existing not in local_entities:
                        local_entities.append(existing)
                    record_event(
                        self.quality_events,
                        category="same_chunk_entity_reused",
                        action="reused",
                        discourse_id=str(discourse_id),
                        subject_surface=surface,
                        object_surface=str(getattr(existing, "label", "") or "") or None,
                        reason="identical normalized label already minted an entity within this discourse; reused instead of minting a differently-typed duplicate",
                    )
                    return existing

        created = self._register_assertion_surface_entity(
            surface=surface,
            entity_type=entity_type,
            discourse_id=discourse_id,
            field=field,
            origin_key=origin_key,
            unit=unit,
            unit_text=unit_text,
        )
        local_entities.append(created)
        if identity_key is not None:
            self._chunk_entity_identity_index[identity_key] = created
        return created

    def _resolve_semantic_unit_relations(
        self,
        *,
        relations: List[Relation],
        local_entities: List[Entity],
        discourse_text: str,
        discourse_id: str,
        field: str,
        origin_key: str,
        unit: Dict[str, Any],
        unit_text: str,
    ) -> List[Relation]:
        # Root-cause fix (manual semantic-QA pass, category F --
        # "respectively" alignment): must run BEFORE per-relation
        # resolution below, while the raw shared object text is still
        # untouched, so the ordinary quantity-literal classification each
        # relation goes through naturally picks up its own now-corrected
        # single value.
        self._split_shared_respectively_values(relations, local_entities)

        # Defect A (stale shared target identity after an upstream
        # respectively/value split): sibling relations whose per-subject
        # object_text was ALREADY split to distinct single values upstream,
        # but whose `target` still points at ONE shared compound value
        # entity because the "respectively" cue was stripped from their own
        # evidence before _split_shared_respectively_values (above) could
        # see it. Runs AFTER that pass -- it only ever acts on a target
        # that is still a live compound entity URI (the split pass resets
        # matched targets to plain text, so anything it already corrected
        # is skipped here).
        #
        # This PRE-resolution call catches only the shape where the
        # extractor already supplied a resolved compound-entity URI as
        # `target`. The raw-object-text shape (target still plain text
        # here) is caught by the mirrored POST-resolution call at the end
        # of this method, once endpoint resolution has produced resource
        # targets -- see the comment there.
        self._rebind_stale_shared_respectively_targets(
            relations,
            local_entities,
            discourse_text=discourse_text,
            discourse_id=discourse_id,
            field=field,
            origin_key=origin_key,
            unit=unit,
            unit_text=unit_text,
        )

        resolved: List[Relation] = []
        for relation in relations:
            meta = getattr(relation, "metadata", None) or {}
            source = str(getattr(relation, "source", "") or "").strip()
            target = str(getattr(relation, "target", "") or "").strip()

            # --- subject / source ---
            raw_subject = str(meta.get("subject_text") or source).strip()
            source_entity = None
            source_blocked = False
            if source.startswith("http"):
                source_entity = next((entity for entity in local_entities if str(getattr(entity, "uri", "") or "").strip() == source), None)
            if source_entity is None and not source.startswith("http") and self._is_unresolved_coreference_candidate(raw_subject):
                # Coreference/possessive/propositional resolution already had
                # their chance to resolve this endpoint (this method always
                # runs after them) and did
                # not. Fabricating a normal Entity from raw referential text
                # here would mint a reusable, cross-discourse URI for a
                # pronoun/deictic phrase with no antecedent -- block that
                # instead of guessing one.
                source_blocked = True
                meta["subject_endpoint_unresolved_coreference"] = True
                meta["subject_endpoint_unresolved_original_surface"] = raw_subject
                meta["subject_endpoint_unresolved_reason"] = self._unresolved_referential_reason(raw_subject)
                relation.source = raw_subject
            elif source_entity is None:
                source_entity = self._resolve_surface_entity(
                    surface=raw_subject,
                    preferred_type=str(meta.get("subject_type") or ""),
                    local_entities=local_entities,
                    discourse_id=discourse_id,
                    field=field,
                    origin_key=origin_key,
                    unit=unit,
                    unit_text=unit_text,
                )
            if not source_blocked:
                if source_entity is None:
                    continue
                relation.source = str(source_entity.uri)

            # --- object / target ---
            # Root-cause fix (Defect 2, predicate-kind mismatches): the
            # extractor's target_is_literal flag records what the LLM
            # PROPOSED, not what the predicate's declared kind actually
            # permits. hasRisk/mitigatedBy/reflects/relatedTo/requires are
            # all registered literal_allowed=False (entity-range) in
            # PREDICATE_REGISTRY -- when the LLM nonetheless proposes a
            # literal object for one of these (e.g. "requires" -> "credentials"),
            # trusting that flag verbatim used to accept an ungoverned literal
            # on a predicate the ontology declares object-only. Route those
            # cases through the SAME entity-resolution branch used for
            # non-literal targets instead (which still keeps a genuine pure
            # scalar/quantity as a literal via its own quantity carve-out) --
            # do not just relax the predicate to literal_allowed=True, which
            # would silently accept free text on a predicate meant to link
            # to a real entity. Predicates registered literal_allowed=True
            # (hasDescription, hasAttribute, includes, referencesSource,
            # trainedOn, ...) are unaffected -- their proposed literal is
            # still trusted as before.
            predicate_name = str(getattr(relation, "relation", "") or "").strip()
            predicate_entry = PREDICATE_REGISTRY.get(predicate_name)
            declared_entity_only = predicate_entry is not None and not predicate_entry.literal_allowed
            propose_literal = bool(getattr(relation, "target_is_literal", False)) or getattr(relation, "target_literal", None) is not None
            if propose_literal and not declared_entity_only:
                literal_value = getattr(relation, "target_literal", None)
                if literal_value is None or str(literal_value).strip() == "":
                    literal_value = meta.get("object_text") or target
                raw_literal_text = str(literal_value).strip()
                # Coreference/possessive/propositional resolution already
                # had their chance to resolve this endpoint even though the extractor marked it
                # target_is_literal=True (both resolvers already compute
                # raw_object and attempt resolution for literal-typed
                # objects too). If it is STILL a bare pronoun or generic
                # demonstrative phrase at this point, it is not a genuine
                # literal value -- block it exactly like the non-literal
                # branch below does, rather than silently accepting a
                # referential surface form as accepted data. Reuses the
                # same classifier the unresolved-endpoint guard uses; no
                # separate string list.
                if self._is_unresolved_coreference_candidate(raw_literal_text):
                    meta["object_endpoint_unresolved_coreference"] = True
                    meta["object_endpoint_unresolved_original_surface"] = raw_literal_text
                    meta["object_endpoint_unresolved_reason"] = self._unresolved_referential_reason(raw_literal_text)
                    relation.target = raw_literal_text
                    relation.target_is_literal = False
                    relation.target_literal = None
                    # Stale-state safeguard: metadata["object_type"] is an
                    # INDEPENDENT signal _relation_has_literal_object() also
                    # trusts -- clear it too so no combination of
                    # target_is_literal/target_literal/object_type can still
                    # be read as "genuine literal" after this reclassification.
                    meta.pop("object_type", None)
                    relation.object_type = None
                else:
                    relation.target_literal = literal_value
                    relation.target = str(literal_value)
                    relation.target_is_literal = True
            else:
                raw_object = str(meta.get("object_text") or target).strip()
                target_entity = None
                target_blocked = False
                target_literalized = False
                if target.startswith("http"):
                    target_entity = next((entity for entity in local_entities if str(getattr(entity, "uri", "") or "").strip() == target), None)
                if target_entity is None and not target.startswith("http") and self._is_unresolved_coreference_candidate(raw_object):
                    target_blocked = True
                    meta["object_endpoint_unresolved_coreference"] = True
                    meta["object_endpoint_unresolved_original_surface"] = raw_object
                    meta["object_endpoint_unresolved_reason"] = self._unresolved_referential_reason(raw_object)
                    relation.target = raw_object
                    relation.target_is_literal = False
                elif target_entity is None:
                    # Root-cause fix (manual semantic-QA pass, category H):
                    # before minting a brand-new "assertion surface entity"
                    # for this object text, check whether it is itself an
                    # unambiguous calendar date (ISO date, "Month DD,
                    # YYYY", "Month YYYY") -- such a phrase should become an
                    # xsd:date/xsd:gYearMonth literal, not a generic Entity
                    # node. A bare year alone is deliberately NOT handled
                    # here (see classify_date_candidate) and falls through
                    # to ordinary entity resolution below, unchanged.
                    date_decision = classify_date_candidate(raw_object)
                    if date_decision is not None:
                        relation.set_target_literal(
                            date_decision["value"],
                            datatype=date_decision["datatype"],
                            object_type="date",
                        )
                        meta["object_date_literal_original_surface"] = raw_object
                        target_literalized = True

                    # Before minting a brand-new "assertion
                    # surface entity" for this object text (the fallback a
                    # few lines below), check whether the object text is
                    # itself a scalar value/quantity expression -- such a
                    # phrase should become a literal on this relation's
                    # object, not a fabricated graph node. A decomposable
                    # quantity+referent phrase resolves to an entity for the
                    # residual referent only, with the quantity preserved on
                    # the relation's metadata instead of baked into the
                    # entity's own label. Skipped entirely when the date
                    # check above already literalized the target -- no
                    # further entity-resolution attempt should run.
                    quantity_decision = None if target_literalized else classify_quantity_candidate(raw_object)
                    if target_literalized:
                        pass
                    elif quantity_decision is not None and quantity_decision.is_pure_quantity:
                        relation.set_target_literal(
                            quantity_decision.literal_value,
                            datatype=quantity_decision.datatype,
                            object_type=quantity_decision.object_type,
                            qualifiers=quantity_decision.qualifiers,
                        )
                        meta["object_quantity_literal_reason"] = quantity_decision.reason
                        target_literalized = True
                    elif quantity_decision is not None and quantity_decision.is_decomposable and quantity_decision.residual_label:
                        target_entity = self._resolve_surface_entity(
                            surface=quantity_decision.residual_label,
                            preferred_type=str(meta.get("object_type") or ""),
                            local_entities=local_entities,
                            discourse_id=discourse_id,
                            field=field,
                            origin_key=origin_key,
                            unit=unit,
                            unit_text=unit_text,
                        )
                        if target_entity is not None:
                            meta["object_quantity_decomposed_value"] = quantity_decision.literal_value
                            meta["object_quantity_decomposed_datatype"] = quantity_decision.datatype
                            meta["object_quantity_decomposed_original_surface"] = raw_object
                            if quantity_decision.qualifiers:
                                meta["object_quantity_decomposed_qualifiers"] = quantity_decision.qualifiers
                    else:
                        target_entity = self._resolve_surface_entity(
                            surface=raw_object,
                            preferred_type=str(meta.get("object_type") or ""),
                            local_entities=local_entities,
                            discourse_id=discourse_id,
                            field=field,
                            origin_key=origin_key,
                            unit=unit,
                            unit_text=unit_text,
                        )
                if not target_blocked and not target_literalized:
                    if target_entity is None:
                        continue
                    relation.set_target_entity(target_entity.uri)

            # Root-cause fix (manual semantic-QA pass, category G): a bare
            # copula ("is"/"are") or the literal-carrying hasDescription
            # predicate ending up bound to a real entity endpoint whose
            # label names a hosting repository/platform/database/portal
            # (e.g. "Adult/Census Income Dataset IS UCI Machine Learning
            # Repository"; "MIMIC-CXR Database hasDescription PhysioNet")
            # means the source text actually asserted a hosting
            # relationship, not identity or a text description -- reroute
            # to the existing hostedBy predicate.
            self._correct_hosting_platform_predicate(
                relation, meta, local_entities, predicate_name, discourse_id=discourse_id,
                field=field, origin_key=origin_key, unit=unit, unit_text=unit_text,
            )
            predicate_name = str(getattr(relation, "relation", "") or "").strip()

            meta.setdefault("field", field)
            meta.setdefault("source_field", field)
            meta.setdefault("origin_key", origin_key)
            meta.setdefault("source_doc_id", str(discourse_id))
            meta.setdefault("discourse_id", discourse_id)
            # The larger sentence/semantic-unit/field text needed
            # to interpret the full assertion, distinct from the minimal
            # evidence_quote span. Keyed at the SAME granularity as the
            # DiscourseUnit node this relation's Evidence already links to
            # (field+discourse_id, via origin_key) -- not the finer
            # semantic unit -- so relations sharing one DiscourseUnit
            # always agree on one context text rather than one relation's
            # narrower unit silently overwriting another's on the same
            # shared node.
            if discourse_text:
                meta.setdefault("source_context", discourse_text)
            meta.setdefault("predicate_mapping_status", getattr(relation, "predicate_mapping_status", "mapped"))
            meta.setdefault("object_type", getattr(relation, "object_type", None) or meta.get("object_type") or ("literal" if getattr(relation, "target_is_literal", False) else "entity"))
            meta.setdefault("qualifiers", getattr(relation, "qualifiers", {}) or {})
            meta.setdefault("negated", bool(getattr(relation, "negated", False)))
            meta.setdefault("temporal_information", getattr(relation, "temporal_information", {}) or {})
            meta.setdefault("quantitative_information", getattr(relation, "quantitative_information", {}) or {})

            # Root-cause fix (manual semantic-QA pass, category D --
            # SV027/SV028 shapes): whatever upstream stage produced this
            # relation's endpoints (extractor misattribution, or anaphora
            # resolution binding a pronoun back onto the wrong/its own
            # antecedent), a relation whose subject and object denote the
            # same thing is a structurally invalid result, not a fact. Drop
            # it rather than serialize a self-referential edge -- do not
            # invent a corrected endpoint, since we do not know what the
            # true referent was.
            if self._is_self_referential_relation_shape(
                subject_text=meta.get("subject_text"),
                object_text=meta.get("object_text"),
                predicate_name=predicate_name,
                source_uri=relation.source,
                target_uri=relation.target,
                target_is_literal=bool(getattr(relation, "target_is_literal", False)),
            ):
                record_event(
                    self.quality_events,
                    category="self_reference_drop",
                    action="dropped",
                    discourse_id=discourse_id,
                    source_relation_id=str(getattr(relation, "uri", "") or "") or None,
                    subject_surface=str(meta.get("subject_text") or "").strip() or None,
                    predicate_surface=predicate_name,
                    object_surface=str(meta.get("object_text") or "").strip() or None,
                    evidence_text=self._relation_evidence_text(relation) or None,
                    reason="subject and object endpoints denote the same referent",
                )
                continue

            # Root-cause fix (manual semantic-QA pass, category G): "is
            # documented/broken down by race, sex, age, insurance type..."
            # is a passive-voice construction where the analytical
            # dimension (age, insurance type, ...) is the INSTRUMENT the
            # finding is broken down BY, not an agent that "documents" the
            # finding. A bare-active predicate "documented"/"brokenDownBy"/
            # "disaggregatedBy" with an analytical-dimension noun as
            # SUBJECT and an abstract finding (e.g. "bias") as object is
            # exactly this inversion. No confident deterministic
            # reconstruction of the correct subject is available (the true
            # subject was the finding-bearing entity earlier in the
            # sentence, not captured on this relation), so drop rather
            # than serialize the inverted role assignment.
            _g3_evidence_text = self._relation_evidence_text(relation) or ""
            if (
                predicate_name.strip().lower() in {"documented", "documentedby", "brokendownby", "disaggregatedby"}
                and self._matches_semantic_category(
                    category="analytical_dimension",
                    text=str(meta.get("subject_text") or "").strip(),
                    uri=relation.source,
                    local_entities=local_entities,
                )
                and (
                    self._matches_semantic_category(
                        category="abstract_finding",
                        text=str(meta.get("object_text") or ""),
                        uri=relation.target,
                        local_entities=local_entities,
                    )
                    # Root-cause fix (bounded correction pass, item 5): the
                    # object text captured by the extractor can be
                    # truncated to just the artifact being broken down
                    # (e.g. "medical image classifiers"), dropping the
                    # actual finding head noun ("bias") that was present
                    # earlier in the same clause ("documented bias in
                    # medical image classifiers by ... age, and insurance
                    # type"). The object_text-only check then never
                    # matches "abstract_finding" and this passive-
                    # dimension inversion goes uncaught. Falling back to
                    # the relation's own evidence text recovers the
                    # SAME signal the object_text check already looks
                    # for -- it does not broaden the category vocabulary
                    # or invent anything not present in the source text.
                    or self._matches_semantic_category(
                        category="abstract_finding",
                        text=_g3_evidence_text,
                        uri=None,
                        local_entities=None,
                    )
                )
            ):
                record_event(
                    self.quality_events,
                    category="passive_dimension_agent_drop",
                    action="dropped",
                    discourse_id=discourse_id,
                    source_relation_id=str(getattr(relation, "uri", "") or "") or None,
                    subject_surface=str(meta.get("subject_text") or "").strip() or None,
                    predicate_surface=predicate_name,
                    object_surface=str(meta.get("object_text") or "").strip() or None,
                    evidence_text=_g3_evidence_text or None,
                    reason="analytical-dimension noun bound as grammatical agent of a passive finding-breakdown predicate",
                )
                continue

            # Root-cause fix (manual semantic-QA pass, category F --
            # comparative outcome misattribution): "Black defendants were
            # twice as likely as white defendants to be labelled high
            # risk" must not decompose into (Black Defendants) labeled
            # (White Defendants) -- the comparator group is not the
            # classification OUTCOME. Detected as: a classification-style
            # predicate ("labeled"/"classified"/"rated"/"scored"/
            # "predicted"/"assessed") whose subject AND object are BOTH
            # population/demographic-group-shaped text, with a comparative
            # cue ("twice as likely as", "compared to", "more likely
            # than", ...) present in the evidence. Reconstructing the
            # correct focal-group/comparator/outcome roles needs real
            # semantic-role labelling that is not available here, so this
            # drops the malformed edge rather than emit it.
            evidence_for_comparative_check = self._relation_evidence_text(relation) or str(meta.get("source_context") or "")
            if (
                predicate_name.strip().lower() in self._CLASSIFICATION_PREDICATES
                and self._COMPARATIVE_CUE_SIGNAL.search(evidence_for_comparative_check)
                and self._matches_semantic_category(
                    category="population_group",
                    text=str(meta.get("subject_text") or ""),
                    uri=relation.source,
                    local_entities=local_entities,
                )
                and self._matches_semantic_category(
                    category="population_group",
                    text=str(meta.get("object_text") or ""),
                    uri=relation.target,
                    local_entities=local_entities,
                )
            ):
                record_event(
                    self.quality_events,
                    category="comparative_misattribution_drop",
                    action="dropped",
                    discourse_id=discourse_id,
                    source_relation_id=str(getattr(relation, "uri", "") or "") or None,
                    subject_surface=str(meta.get("subject_text") or "").strip() or None,
                    predicate_surface=predicate_name,
                    object_surface=str(meta.get("object_text") or "").strip() or None,
                    evidence_text=evidence_for_comparative_check or None,
                    reason="classification predicate with two population-group endpoints under a comparative cue",
                )
                continue

            # Root-cause fix (manual semantic-QA pass, category I): a bare
            # URL is not, by itself, evidence for a rich factual claim
            # (e.g. "GitHub Analysis analyzed 100+ million rideshare
            # samples" evidenced only by a GitHub link) -- align_evidence_span
            # downstream only verifies the URL is a genuine verbatim span
            # of the source document, which proves source OCCURRENCE, not
            # that it entails this specific subject-predicate-object
            # claim. When the captured evidence is nothing but a URL and
            # the predicate is not itself a source/reference predicate
            # (where the URL genuinely IS the claim), prefer the fuller
            # source_context text already captured on this relation's
            # metadata if that text is not itself just the same
            # bare URL; otherwise there is nothing here that actually
            # grounds the claim beyond co-occurrence, so drop it rather
            # than serialize an ungrounded relation as if it were
            # evidenced.
            relation_kind_for_evidence = str(meta.get("relation_kind") or "").strip()
            if relation_kind_for_evidence not in {"structural", "provenance"}:
                evidence_text = str(meta.get("evidence") or meta.get("evidence_quote") or "").strip()
                if self._is_bare_url_text(evidence_text) and predicate_name.strip().lower() not in self._SOURCE_REFERENCE_PREDICATES:
                    broader_context = str(meta.get("source_context") or "").strip()
                    if broader_context and not self._is_bare_url_text(broader_context):
                        meta["evidence"] = broader_context
                        meta["evidence_url_only_upgraded_to_context"] = True
                        meta["evidence_original_url_only_value"] = evidence_text
                        record_event(
                            self.quality_events,
                            category="url_only_evidence_upgrade",
                            action="corrected",
                            discourse_id=discourse_id,
                            source_relation_id=str(getattr(relation, "uri", "") or "") or None,
                            subject_surface=str(meta.get("subject_text") or "").strip() or None,
                            predicate_surface=predicate_name,
                            object_surface=str(meta.get("object_text") or "").strip() or None,
                            evidence_text=evidence_text or None,
                            reason="bare-URL evidence replaced with broader source_context text",
                        )
                    else:
                        record_event(
                            self.quality_events,
                            category="url_only_evidence_drop",
                            action="dropped",
                            discourse_id=discourse_id,
                            source_relation_id=str(getattr(relation, "uri", "") or "") or None,
                            subject_surface=str(meta.get("subject_text") or "").strip() or None,
                            predicate_surface=predicate_name,
                            object_surface=str(meta.get("object_text") or "").strip() or None,
                            evidence_text=evidence_text or None,
                            reason="evidence is a bare URL with no broader grounding context, predicate is not a source-reference predicate",
                        )
                        continue

            relation.metadata = meta
            resolved.append(relation)

        # Defect A (production-order fix): the pre-resolution call above only
        # catches the shape where the extractor ALREADY handed us a resolved
        # compound-entity URI as `target`. When the extractor instead emits
        # raw object text, `target` is still plain text at that point, so the
        # detector's "target is a resource URI" precondition fails and it
        # skips -- and the per-relation endpoint resolution just completed
        # then binds every already-split sibling value back onto the ONE
        # pre-existing compound value entity via permissive surface matching
        # (each single value's tokens are a subset of the compound label).
        # Re-run the SAME generic detector now that every endpoint is a
        # resolved resource: `target` is a live compound-entity URI, so the
        # group forms and every remaining structural/source-text safeguard
        # applies unchanged. Idempotent by construction -- a group the
        # earlier call already rebound now has distinct per-sibling targets,
        # so it can no longer form a >=2 shared-target group here (no double
        # correction, no duplicate value entities, no duplicate events).
        self._rebind_stale_shared_respectively_targets(
            resolved,
            local_entities,
            discourse_text=discourse_text,
            discourse_id=discourse_id,
            field=field,
            origin_key=origin_key,
            unit=unit,
            unit_text=unit_text,
        )

        return self._drop_redundant_demonstration_relations(resolved, local_entities)

    # Category J (manual semantic-QA pass): perception/demonstration verbs
    # whose object should be the abstract finding itself (e.g. "systematic
    # bias"), not the population/person group the finding is ABOUT. This
    # verb family is CORE/domain-independent (generic perception verbs);
    # what counts as an "abstract finding" or "population group" object is
    # entirely profile-driven -- see _matches_semantic_category and
    # domain_profiles.py.
    _DEMONSTRATION_VERBS: Set[str] = {
        "show", "shows", "demonstrate", "demonstrates", "exhibit",
        "exhibits", "reveal", "reveals", "indicate", "indicates",
        "reflect", "reflects",
    }

    def _drop_redundant_demonstration_relations(
        self, relations: List[Relation], local_entities: List[Entity],
    ) -> List[Relation]:
        """
        Root-cause fix (manual semantic-QA pass, category J): when the SAME
        subject and predicate (drawn from a small perception/demonstration
        verb family -- "show", "demonstrate", "exhibit", "reveal",
        "indicate", "reflect") produce two sibling relations from the same
        evidence, one whose object is an abstract finding (e.g. "systematic
        bias") and one whose object is a population/person group (e.g.
        "students from marginalized groups"), the population-group version
        is (subject) verb (the group the finding is ABOUT) -- a malformed
        decomposition of source text that actually asserts (subject) verb
        (finding) against/toward (population), not two independent facts.
        Drop the population-group sibling; the abstract-finding relation
        already carries the genuine assertion.

        Conservative by construction: only fires when BOTH shapes are
        present for the same (subject, predicate, evidence) group -- a
        population-group object with no abstract-finding sibling is a
        potentially genuine, wholly different fact and is left untouched.

        "Abstract finding" / "population group" are resolved via
        _matches_semantic_category (ontology type first, then the active
        domain profile's lexical signal) -- this method contains no
        AI-ethics-specific vocabulary itself.
        """
        by_key: Dict[Tuple[str, str, str], List[Relation]] = {}
        for rel in relations:
            pred = str(getattr(rel, "relation", "") or "").strip().lower()
            if pred not in self._DEMONSTRATION_VERBS:
                continue
            subj = str(getattr(rel, "source", "") or "").strip().lower()
            evidence = normalize_text(self._relation_evidence_text(rel))[:120]
            by_key.setdefault((subj, pred, evidence), []).append(rel)

        to_drop: Set[int] = set()
        for group in by_key.values():
            if len(group) < 2:
                continue
            abstract, population = [], []
            for rel in group:
                meta = getattr(rel, "metadata", None) or {}
                obj_text = str(meta.get("object_text") or getattr(rel, "target", "") or "")
                obj_uri = None if bool(getattr(rel, "target_is_literal", False)) else str(getattr(rel, "target", "") or "")
                if self._matches_semantic_category(category="abstract_finding", text=obj_text, uri=obj_uri, local_entities=local_entities):
                    abstract.append(rel)
                elif self._matches_semantic_category(category="population_group", text=obj_text, uri=obj_uri, local_entities=local_entities):
                    population.append(rel)
            if abstract and population:
                to_drop.update(id(rel) for rel in population)
                abstract_ids = [str(getattr(r, "uri", "") or "") for r in abstract if getattr(r, "uri", None)]
                for rel in population:
                    rmeta = getattr(rel, "metadata", None) or {}
                    record_event(
                        self.quality_events,
                        category="redundant_demonstration_drop",
                        action="dropped",
                        discourse_id=rmeta.get("discourse_id"),
                        source_relation_id=str(getattr(rel, "uri", "") or "") or None,
                        subject_surface=str(rmeta.get("subject_text") or "").strip() or None,
                        predicate_surface=str(getattr(rel, "relation", "") or "").strip() or None,
                        object_surface=str(rmeta.get("object_text") or "").strip() or None,
                        evidence_text=self._relation_evidence_text(rel) or None,
                        reason="population-group sibling of a demonstration-verb relation already covered by an abstract-finding sibling",
                        replacement_relation_ids=abstract_ids or None,
                    )

        if not to_drop:
            return relations
        return [rel for rel in relations if id(rel) not in to_drop]

    # Category F (manual semantic-QA pass): comparator-misattribution guard.
    _CLASSIFICATION_PREDICATES = {
        "labeled", "labels", "classified", "classifies", "rated", "rates",
        "scored", "scores", "predicted", "predicts", "assessed", "assesses",
    }
    _COMPARATIVE_CUE_SIGNAL = re.compile(
        r"\b(twice as likely as|times as likely as|times more likely than|compared to|compared with|relative to|more likely than)\b",
        re.I,
    )
    # Category F: "respectively" value-splitting.
    _RESPECTIVELY_CUE = re.compile(r"\brespectively\b", re.I)
    _RESPECTIVELY_TWO_VALUE_RE = re.compile(r"^\s*([\d][\d.,]*\s*%?)\s+and\s+([\d][\d.,]*\s*%?)\s+(.+?)\s*$", re.I)

    def _split_shared_respectively_values(
        self, relations: List[Relation], local_entities: Optional[List[Entity]] = None,
    ) -> None:
        """
        Root-cause fix (manual semantic-QA pass, category F --
        "respectively" alignment; entity-binding gap closed in the
        targeted correction pass, item 2). In-place pre-pass, run BEFORE
        per-relation endpoint resolution: when a sentence like "Black and
        Latino borrowers pay 7.9 and 3.6 basis points more respectively"
        is decomposed into two per-subject relations, the extractor can
        attach the SAME combined "7.9 and 3.6 basis points more" object
        text to BOTH subjects instead of splitting it positionally --
        sometimes as raw shared text, sometimes (confirmed via the real
        annotations.json record for this exact sentence, reproduced in
        scratchpad/repro_respectively2.py) as BOTH relations' `target`
        already pointing at ONE standalone entity the primary entity-
        extraction pass minted for the unsplit compound phrase.
        Detected as exactly two relations sharing (predicate, evidence
        text containing "respectively", identical raw object text
        matching an "N1 and N2 <unit>" shape) with two DIFFERENT subjects.

        Corrected by assigning the first value to whichever subject's own
        distinguishing token appears FIRST in the shared evidence text --
        verified against source mention order, not assumed from
        relation-list/extraction order -- and the second value to the
        other. If either subject's token cannot be located in the
        evidence, the group is left uncorrected rather than guessing an
        order.

        Every relation in a corrected group has its target UNCONDITIONALLY
        reset to the new split text (with target_literal/target_is_literal/
        object_type cleared), regardless of what it currently holds,
        because by construction of `groups` every member's existing
        target denotes the same stale, unsplit compound value -- forcing
        the ordinary per-relation object-resolution logic that runs
        immediately after this pre-pass (in the same caller,
        _resolve_semantic_unit_relations) to resolve the corrected text
        fresh. The stale shared compound entity (if any) is additionally
        removed from `local_entities` -- but ONLY when no relation
        outside this correction group still targets it -- so that
        fresh resolution cannot immediately re-match it again via
        substring surface-matching (confirmed necessary: "N2 <unit>",
        e.g. "3.6 basis points more", is a literal trailing substring of
        the original "N1 and N2 <unit>" label, so ordinary permissive
        surface matching would otherwise silently re-select the stale
        entity for the SECOND split value even after its target was
        reset).
        """
        groups: Dict[Tuple[str, str, str], List[Relation]] = {}
        for rel in relations:
            meta = getattr(rel, "metadata", None) or {}
            evidence = self._relation_evidence_text(rel)
            if not self._RESPECTIVELY_CUE.search(evidence):
                continue
            obj_text = str(meta.get("object_text") or getattr(rel, "target", "") or "").strip()
            if not self._RESPECTIVELY_TWO_VALUE_RE.match(obj_text):
                continue
            pred = str(getattr(rel, "relation", "") or "").strip().lower()
            key = (pred, normalize_text(evidence)[:160], obj_text.lower())
            groups.setdefault(key, []).append(rel)

        for group in groups.values():
            if len(group) != 2:
                continue
            meta_a = getattr(group[0], "metadata", None) or {}
            meta_b = getattr(group[1], "metadata", None) or {}
            subj_a = str(meta_a.get("subject_text") or "").strip().lower()
            subj_b = str(meta_b.get("subject_text") or "").strip().lower()
            if not subj_a or not subj_b or subj_a == subj_b:
                continue
            obj_text = str(meta_a.get("object_text") or "").strip()
            m = self._RESPECTIVELY_TWO_VALUE_RE.match(obj_text)
            if not m:
                continue
            first_val, second_val, suffix = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

            # Root-cause fix (targeted correction pass, item 2): the
            # positional assignment below (first value -> whichever
            # relation happens to be group[0]) previously trusted
            # relation-LIST order, which reflects extraction order, not
            # necessarily source-sentence order. Verify instead: find
            # each subject's own distinguishing token's position in the
            # SHARED evidence text (both relations in a group have
            # identical evidence, by construction of `key` above) and
            # assign values in the order the subjects actually appear in
            # the source sentence. If either subject's token cannot be
            # located in the evidence (should not happen given the
            # grouping precondition, but do not guess if it does), skip
            # this group rather than assume an unverified order.
            evidence_for_order = self._relation_evidence_text(group[0]) or ""
            evidence_lower = evidence_for_order.lower()

            def _first_subject_token(meta: Dict[str, Any]) -> str:
                tokens = str(meta.get("subject_text") or "").strip().lower().split()
                return tokens[0] if tokens else ""

            tok_a = _first_subject_token(meta_a)
            tok_b = _first_subject_token(meta_b)
            pos_a = evidence_lower.find(tok_a) if tok_a else -1
            pos_b = evidence_lower.find(tok_b) if tok_b else -1
            if pos_a < 0 or pos_b < 0 or pos_a == pos_b:
                continue
            ordered_group = list(group) if pos_a < pos_b else [group[1], group[0]]

            # Captured BEFORE any reset below, so a pre-resolved shared
            # entity URI can be identified and cleaned out of
            # local_entities afterward (see the removal block after this
            # loop) -- once reset, rel.target no longer carries this
            # information.
            stale_targets = {
                str(getattr(rel, "target", "") or "").strip()
                for rel in ordered_group
                if str(getattr(rel, "target", "") or "").strip().startswith("http")
            }

            for rel, value in zip(ordered_group, (first_val, second_val)):
                new_text = f"{value} {suffix}".strip()
                meta = getattr(rel, "metadata", None) or {}
                meta["object_text"] = new_text
                meta["respectively_split_from"] = obj_text
                rel.metadata = meta
                # Root-cause fix (targeted correction pass, item 2):
                # previously only rewrote rel.target when it still
                # equaled the RAW, unsplit compound text verbatim. When
                # the extractor instead supplied a pre-resolved entity
                # ID/URI for the object (both sibling relations pointing
                # at the SAME standalone "N1 and N2 <unit>" entity minted
                # by the primary entity-extraction pass, confirmed as
                # the actual defect shape via
                # scratchpad/repro_respectively.py and the real
                # annotations.json record for this exact sentence), that
                # equality check never matched, so the stale shared
                # entity binding silently survived even though
                # object_text/object_surface_form were correctly split.
                # Every relation reaching this point is, by construction
                # of `groups` above, one whose raw object denoted the
                # SAME shared, unsplit compound value -- so its CURRENT
                # target (whatever it is: raw text, a literal, or an
                # entity URI) is definitionally stale and must be reset,
                # unconditionally, back to plain corrected text with no
                # literal/entity binding. This forces the ordinary
                # per-relation object-resolution logic that runs AFTER
                # this pre-pass (in the same caller,
                # _resolve_semantic_unit_relations) to resolve the
                # NEW, already-split text fresh, exactly like a normal
                # never-merged quantity mention -- producing its own
                # distinct literal/entity rather than reusing the old
                # shared one.
                rel.target = new_text
                rel.target_literal = None
                rel.target_is_literal = False
                rel.object_type = None
                meta.pop("object_type", None)
                record_event(
                    self.quality_events,
                    category="respectively_split_correction",
                    action="corrected",
                    discourse_id=meta.get("discourse_id"),
                    source_relation_id=str(getattr(rel, "uri", "") or "") or None,
                    subject_surface=str(meta.get("subject_text") or "").strip() or None,
                    predicate_surface=str(getattr(rel, "relation", "") or "").strip() or None,
                    object_surface=new_text or None,
                    evidence_text=self._relation_evidence_text(rel) or None,
                    reason=f"shared compound value {obj_text!r} split positionally between two co-occurring subjects",
                )

            # Remove the stale shared compound entity from local_entities
            # (if any, and only if nothing outside this correction group
            # still needs it) so downstream surface-matching for these
            # two now-reset relations cannot immediately re-select it --
            # e.g. "3.6 basis points more" is a literal trailing
            # substring of "7.9 and 3.6 basis points more", so ordinary
            # permissive surface matching would otherwise silently
            # re-bind the second split value straight back to it.
            if stale_targets and local_entities:
                other_targets = {
                    str(getattr(r, "target", "") or "").strip()
                    for r in relations
                    if r not in ordered_group
                }
                for idx in range(len(local_entities) - 1, -1, -1):
                    ent_uri = str(getattr(local_entities[idx], "uri", "") or "").strip()
                    if ent_uri in stale_targets and ent_uri not in other_targets:
                        del local_entities[idx]

    # Defect A: a scalar value's leading numeric/currency/percentage core,
    # the only part that has to line up between an object_text single value
    # and its position in the source sentence. Domain-independent -- matches
    # numbers, "$" amounts and "%" rates, nothing dataset-specific.
    _SCALAR_VALUE_CORE_RE = re.compile(r"[+\-]?\$?\d[\d.,]*\s*%?")
    # A coordinator joining the parts of a genuine compound-value label
    # ("7.9 and 3.6 ...", "12%, 8% ...", "a & b").
    _VALUE_COORDINATOR_RE = re.compile(r"(?:\band\b|\bor\b|,|&|/)", re.I)

    def _scalar_value_cores(self, text: str) -> List[str]:
        """Every distinct scalar core in `text`, in order of appearance."""
        seen: List[str] = []
        for m in self._SCALAR_VALUE_CORE_RE.finditer(str(text or "")):
            core = re.sub(r"\s+", "", m.group(0)).strip().lower()
            if core and core not in seen:
                seen.append(core)
        return seen

    def _label_is_compound_of_values(self, label: str, value_cores: List[str]) -> bool:
        """True when `label` structurally represents the COMBINATION of
        exactly `value_cores` -- i.e. it joins them with a coordinator and
        carries no scalar core beyond that set. Purely structural; no
        vocabulary."""
        norm = str(label or "").strip().lower()
        if not norm or not self._VALUE_COORDINATOR_RE.search(norm):
            return False
        want = [c for c in value_cores if c]
        if len(want) < 2:
            return False
        label_cores = self._scalar_value_cores(norm)
        if sorted(label_cores) != sorted(want):
            return False
        return True

    def _rebind_stale_shared_respectively_targets(
        self,
        relations: List[Relation],
        local_entities: Optional[List[Entity]] = None,
        *,
        discourse_text: Optional[str] = None,
        discourse_id: Optional[str] = None,
        field: Optional[str] = None,
        origin_key: Optional[str] = None,
        unit: Optional[Dict[str, Any]] = None,
        unit_text: Optional[str] = None,
    ) -> None:
        """Defect A -- stale shared target identity after an upstream
        respectively/value split.

        Two or more sibling relations already carry DISTINCT single-value
        `object_text` values but still share ONE compound value-entity
        `target` URI, because the "respectively" cue that
        _split_shared_respectively_values keys on was removed from their own
        evidence upstream. This pass re-establishes distinct targets, and
        derives the subject->value pairing from SOURCE-TEXT ORDER (never
        relation-list order).

        Every one of these must hold before a group is corrected:
          * same source field and discourse;
          * same relation predicate;
          * same resource (entity, non-literal) target URI;
          * distinct subjects;
          * distinct single-value object_text values (exactly one scalar
            core each);
          * the shared target entity's label structurally represents the
            combined values;
          * authoritative source text (the discourse/unit text) contains
            all the subjects, all the values, and a "respectively" cue,
            with every subject positioned before every value;
          * subject/value pairing established purely from source-text
            order.

        Then, per value: resolve/mint one distinct value entity through the
        normal canonical entity path (_resolve_surface_entity), rebind the
        relation's actual target to it (clearing any literal state), realign
        object_text / object_surface_form, and -- once every sibling is
        rebound -- drop the now-unreferenced compound entity from the local
        pool (and, when it is discourse-local and referenced nowhere else
        in this discourse, from the registry). Nothing is changed unless a
        distinct value entity is obtained for every member.
        """
        if not relations:
            return

        ent_by_uri: Dict[str, Entity] = {}
        registry = getattr(self, "global_registry", None)
        get_all = getattr(registry, "get_all_entities", None)
        registry_entities = list(get_all()) if callable(get_all) else []
        for pool in (list(local_entities or []), registry_entities):
            for ent in pool:
                uri = str(getattr(ent, "uri", "") or "").strip()
                if uri:
                    ent_by_uri.setdefault(uri, ent)

        groups: Dict[Tuple[str, str, str, str], List[Relation]] = {}
        for rel in relations:
            if bool(getattr(rel, "target_is_literal", False)):
                continue
            tgt = str(getattr(rel, "target", "") or "").strip()
            if not tgt.startswith("http"):
                continue
            meta = getattr(rel, "metadata", None) or {}
            pred = str(getattr(rel, "relation", "") or "").strip().lower()
            fld = str(meta.get("field") or meta.get("source_field") or field or "").strip().lower()
            did = str(meta.get("discourse_id") or discourse_id or "").strip()
            if not pred:
                continue
            groups.setdefault((fld, did, pred, tgt), []).append(rel)

        source_text = str(discourse_text or unit_text or "").strip()
        rebound_compound_uris: Set[str] = set()

        for (fld, did, _pred, comp_uri), group in groups.items():
            if len(group) < 2:
                continue

            subjects = [str(getattr(r, "source", "") or "").strip() for r in group]
            if any(not s for s in subjects) or len(set(subjects)) != len(group):
                continue

            metas = [getattr(r, "metadata", None) or {} for r in group]
            subject_texts = [str(m.get("subject_text") or "").strip() for m in metas]
            object_texts = [str(m.get("object_text") or "").strip() for m in metas]
            if any(not t for t in subject_texts) or any(not t for t in object_texts):
                continue

            # Distinct single-value object_text values (exactly one scalar
            # core each -- a still-compound "N1 and N2" object_text is the
            # other pass's job, not this one).
            per_obj_cores = [self._scalar_value_cores(t) for t in object_texts]
            if any(len(c) != 1 for c in per_obj_cores):
                continue
            obj_cores = [c[0] for c in per_obj_cores]
            if len(set(object_texts)) != len(group) or len(set(obj_cores)) != len(group):
                continue

            comp_ent = ent_by_uri.get(comp_uri)
            if comp_ent is None:
                continue
            if not self._label_is_compound_of_values(str(getattr(comp_ent, "label", "") or ""), obj_cores):
                continue

            # Authoritative source-text confirmation.
            src = source_text or self._relation_evidence_text(group[0])
            src_l = src.lower()
            if not src or not self._RESPECTIVELY_CUE.search(src):
                continue

            def _first_token(text: str) -> str:
                toks = str(text or "").strip().lower().split()
                return toks[0] if toks else ""

            subj_tokens = [_first_token(t) for t in subject_texts]
            if any(not tok for tok in subj_tokens):
                continue
            subj_pos = [src_l.find(tok) for tok in subj_tokens]
            core_pos = [src_l.find(core) for core in obj_cores]
            if any(p < 0 for p in subj_pos) or any(p < 0 for p in core_pos):
                continue
            if len(set(subj_pos)) != len(group) or len(set(core_pos)) != len(group):
                continue
            # "A and B ... V1 and V2 respectively": every subject mention
            # must precede every value mention, otherwise source order
            # cannot be safely established -- leave the group untouched.
            if max(subj_pos) >= min(core_pos):
                continue

            order_by_subject = sorted(range(len(group)), key=lambda i: subj_pos[i])
            ordered_cores = [c for _, c in sorted(zip(core_pos, obj_cores), key=lambda z: z[0])]
            if len(ordered_cores) != len(group):
                continue

            core_to_text = {}
            for cores, text in zip(per_obj_cores, object_texts):
                core_to_text.setdefault(cores[0], text)

            synth_unit = dict(unit or {})
            synth_unit.setdefault("start_offset", 0)
            base_meta = metas[order_by_subject[0]]
            synth_unit.setdefault("section", base_meta.get("section") or "")
            synth_unit.setdefault("section_category", base_meta.get("section_category") or "")
            synth_unit.setdefault("unit_id", base_meta.get("semantic_unit_id") or "")
            synth_unit.setdefault("unit_type", base_meta.get("semantic_unit_type") or "")
            synth_unit_text = str(unit_text or discourse_text or src or "")

            # Take the stale compound entity out of the resolution pool
            # BEFORE minting the per-value entities -- but only if nothing
            # outside this group still points at it. A single value's text
            # (e.g. "3.6 basis points more") is a trailing substring of the
            # compound label ("7.9 and 3.6 basis points more"), so leaving
            # it in `local_entities` would let permissive surface matching
            # in _resolve_surface_entity re-select it for that value.
            outside_refs = {
                str(getattr(r, "target", "") or "").strip()
                for r in relations
                if r not in group
            } | {
                str(getattr(r, "source", "") or "").strip()
                for r in relations
                if r not in group
            }
            if comp_uri not in outside_refs and local_entities is not None:
                for pool_idx in range(len(local_entities) - 1, -1, -1):
                    if str(getattr(local_entities[pool_idx], "uri", "") or "").strip() == comp_uri:
                        del local_entities[pool_idx]

            # Phase 1: resolve/mint one distinct value entity per value,
            # source-order aligned. Abandon the whole group on any failure
            # or collision -- never a partial rebind.
            planned: List[Tuple[Relation, Entity, str]] = []
            value_entities: Dict[str, Entity] = {}
            ok = True
            for rank, rel_idx in enumerate(order_by_subject):
                rel = group[rel_idx]
                meta = metas[rel_idx]
                value_text = core_to_text.get(ordered_cores[rank], ordered_cores[rank])
                ve = value_entities.get(value_text)
                if ve is None:
                    ve = self._resolve_surface_entity(
                        surface=value_text,
                        preferred_type="",
                        local_entities=local_entities if local_entities is not None else [],
                        discourse_id=str(meta.get("discourse_id") or discourse_id or did or ""),
                        field=str(meta.get("field") or field or fld or ""),
                        origin_key=str(meta.get("origin_key") or origin_key or ""),
                        unit=synth_unit,
                        unit_text=synth_unit_text,
                    )
                    if ve is None or not str(getattr(ve, "uri", "") or "").strip():
                        ok = False
                        break
                    value_entities[value_text] = ve
                planned.append((rel, ve, value_text))
            if not ok:
                continue
            if len({str(ve.uri) for _, ve, _ in planned}) != len(group):
                continue
            if any(str(ve.uri) == comp_uri for _, ve, _ in planned):
                continue

            # Phase 2: rebind the actual target + realign object surface.
            for rel, ve, value_text in planned:
                meta = getattr(rel, "metadata", None) or {}
                old_target = str(getattr(rel, "target", "") or "")
                rel.set_target_entity(str(ve.uri))
                meta["object_text"] = value_text
                meta["object_surface_form"] = value_text
                meta["object_type"] = "entity"
                meta["shared_compound_respectively_rebound_from"] = old_target
                meta.setdefault(
                    "respectively_split_from",
                    str(getattr(comp_ent, "label", "") or old_target),
                )
                rel.metadata = meta
                record_event(
                    self.quality_events,
                    category="shared_compound_respectively_rebind",
                    action="corrected",
                    discourse_id=meta.get("discourse_id") or did,
                    source_relation_id=str(getattr(rel, "uri", "") or "") or None,
                    subject_surface=str(meta.get("subject_text") or "").strip() or None,
                    predicate_surface=str(getattr(rel, "relation", "") or "").strip() or None,
                    object_surface=value_text or None,
                    evidence_text=self._relation_evidence_text(rel) or None,
                    reason=(
                        f"stale shared compound value entity {old_target!r} "
                        "rebound to a distinct per-subject value entity by "
                        "source-text order"
                    ),
                )
            rebound_compound_uris.add(comp_uri)

        if not rebound_compound_uris:
            return

        # Drop each rebound compound entity when nothing legitimate still
        # points at it.
        still_referenced: Set[str] = set()
        for r in relations:
            for u in (
                str(getattr(r, "source", "") or "").strip(),
                str(getattr(r, "target", "") or "").strip(),
            ):
                if u.startswith("http"):
                    still_referenced.add(u)

        for comp_uri in rebound_compound_uris:
            if comp_uri in still_referenced:
                continue
            if local_entities is not None:
                for idx in range(len(local_entities) - 1, -1, -1):
                    if str(getattr(local_entities[idx], "uri", "") or "").strip() == comp_uri:
                        del local_entities[idx]
            comp_ent = ent_by_uri.get(comp_uri)
            comp_did = str((getattr(comp_ent, "metadata", None) or {}).get("discourse_id") or "")
            reg = getattr(self, "global_registry", None)
            store = getattr(reg, "entities", None) if reg is not None else None
            if (
                isinstance(store, dict)
                and comp_uri in store
                and comp_did
                and comp_did == str(discourse_id or "")
            ):
                store.pop(comp_uri, None)

    def _supersede_stale_respectively_duplicates(
        self, relations: List[Relation], discourse_id: Any,
    ) -> List[Relation]:
        """
        Root-cause fix (bounded correction pass, item 4 -- correction
        lifecycle): _split_shared_respectively_values corrects each
        sibling's OWN object value in place, but only ever sees the
        relations produced by ONE semantic-unit/coverage-recovery call.
        When the identical "N1 and N2 <unit> respectively" statistic is
        repeated in a DIFFERENT section of the same discourse, that
        occurrence gets its own, independently correctly-split pair --
        while the unsplit compound-object relation(s) from the OTHER
        occurrence of the identical statistic remain in the discourse's
        relation list, unrecognised as duplicates, because their object
        text differs from the corrected siblings' (ordinary exact-SPO
        dedup never groups them).

        This targeted, discourse-scoped reconciliation runs once the
        full per-discourse relation list is available (see call site in
        validate_semantic_relations): for every corrected relation
        carrying `respectively_split_from` (the ORIGINAL shared compound
        text it was split away from), find any OTHER, not-itself-
        corrected relation in this SAME discourse with the SAME
        predicate whose CURRENT object text equals that original
        compound text -- that is exactly the stale, unsplit duplicate of
        the SAME underlying statistic -- and supersede it rather than
        serialize it as an independent fact, preserving provenance via a
        quality-report event that names the surviving, corrected
        replacement relation(s).

        Deliberately narrow: matches only on (predicate, exact original-
        compound-text) within one discourse, never on evidence/subject
        similarity or across discourses, so it cannot remove a
        genuinely distinct claim (see caller docstring: "do not use
        broad same-evidence deduplication that removes distinct
        claims").
        """
        corrected_originals: Dict[Tuple[str, str], List[Relation]] = {}
        for rel in relations:
            meta = getattr(rel, "metadata", None) or {}
            orig = str(meta.get("respectively_split_from") or "").strip().lower()
            if not orig:
                continue
            pred = str(getattr(rel, "relation", "") or "").strip().lower()
            corrected_originals.setdefault((pred, orig), []).append(rel)

        if not corrected_originals:
            return relations

        to_drop: Set[int] = set()
        for rel in relations:
            meta = getattr(rel, "metadata", None) or {}
            if str(meta.get("respectively_split_from") or "").strip():
                continue  # a corrected relation is never itself a stale duplicate
            pred = str(getattr(rel, "relation", "") or "").strip().lower()
            current_obj = str(meta.get("object_text") or getattr(rel, "target", "") or "").strip().lower()
            if not current_obj:
                continue
            siblings = corrected_originals.get((pred, current_obj))
            if not siblings:
                continue
            to_drop.add(id(rel))
            replacement_ids = [str(getattr(s, "uri", "") or "") for s in siblings if getattr(s, "uri", None)]
            record_event(
                self.quality_events,
                category="respectively_split_supersession",
                action="superseded",
                discourse_id=discourse_id,
                source_relation_id=str(getattr(rel, "uri", "") or "") or None,
                subject_surface=str(meta.get("subject_text") or "").strip() or None,
                predicate_surface=pred or None,
                object_surface=current_obj or None,
                evidence_text=self._relation_evidence_text(rel) or None,
                reason="stale unsplit compound-value duplicate of a statistic corrected elsewhere in this discourse by the respectively-split pass",
                replacement_relation_ids=replacement_ids or None,
            )

        if not to_drop:
            return relations
        return [r for r in relations if id(r) not in to_drop]

    # CORE/domain-independent: generic English words for "a place that
    # hosts/serves something" -- applicable to any domain, not just
    # AI/dataset documentation. A named hosting platform's own PROPER NAME
    # often does not itself contain one of these words (e.g. "PhysioNet"),
    # which is why a profile-supplied gazetteer of concrete platform names
    # supplements this signal in _correct_hosting_platform_predicate below
    # -- the gazetteer is domain data (see domain_profiles.py), this
    # keyword regex is not.
    _HOSTING_PLATFORM_LABEL_SIGNAL = re.compile(
        r"\b(repository|platform|database|portal|archive|hub|warehouse|registry)\b",
        re.I,
    )

    def _correct_hosting_platform_predicate(
        self, relation: Relation, meta: Dict[str, Any], local_entities: List[Entity], predicate_name: str,
        discourse_id: Optional[str] = None, field: str = "", origin_key: str = "",
        unit: Optional[Dict[str, Any]] = None, unit_text: str = "",
    ) -> None:
        """See call site in _resolve_semantic_unit_relations (category G).
        Conservative: only fires for the bare copula ("is"/"are") or
        hasDescription predicates, and only when the object's own text
        names a hosting repository/platform/database/portal/archive/hub/
        warehouse/registry. Any other predicate/object combination is
        left untouched.

        Root-cause fix (bounded correction pass, item 5): "is"/"are" are
        not registered in PREDICATE_REGISTRY, so declared_entity_only is
        always False for them and the caller's literal/entity branch is
        decided purely by whatever target_is_literal the extractor
        proposed -- a bare copula sentence ("X is UCI Machine Learning
        Repository") very often ends up with a LITERAL object, not a
        resolved entity. The original entity-only check here silently
        never fired for that (common) case. Both shapes are now
        checked against the identical gazetteer/regex signal.
        """
        pred_lower = str(predicate_name or "").strip().lower()
        if pred_lower not in {"is", "are", "hasdescription"}:
            return
        if bool(getattr(relation, "target_is_literal", False)):
            label = str(
                getattr(relation, "target_literal", None)
                or meta.get("object_text")
                or getattr(relation, "target", "")
                or ""
            ).strip()
        else:
            target_uri = str(getattr(relation, "target", "") or "")
            if not target_uri.startswith("http"):
                return
            target_entity = next(
                (e for e in local_entities if str(getattr(e, "uri", "") or "") == target_uri), None
            )
            label = str(getattr(target_entity, "label", "") or "") if target_entity is not None else ""
        is_known_platform = self.domain_profile.matches_gazetteer("hosting_platform_names", label)
        if not is_known_platform and not self._HOSTING_PLATFORM_LABEL_SIGNAL.search(label):
            return

        # Root-cause fix (bounded preflight, item 5): hostedBy is
        # declared an owl:ObjectProperty (see ontology/add_schema.py) --
        # its range is a real entity, never a literal. The literal-
        # object branch above (added to widen this correction to bare-
        # copula sentences whose object never got resolved to an
        # entity) would otherwise rewrite ONLY the predicate and leave
        # an ObjectProperty pointing at a plain string, which is exactly
        # the class of defect PREDICATE_REGISTRY's literal_allowed gate
        # exists to prevent elsewhere in this same method. Mint/reuse a
        # real entity for the hosting-platform label via the same
        # surface-entity resolution used everywhere else in this class,
        # and only then repoint the relation's target to it.
        if bool(getattr(relation, "target_is_literal", False)):
            platform_entity = self._resolve_surface_entity(
                surface=label,
                preferred_type="Resource",
                local_entities=local_entities,
                discourse_id=discourse_id or "",
                field=field,
                origin_key=origin_key,
                unit=unit or {},
                unit_text=unit_text,
            )
            if platform_entity is None:
                # No valid entity endpoint could be produced -- leave the
                # predicate/object untouched rather than assert an
                # ObjectProperty with a literal value.
                return
            relation.set_target_entity(str(platform_entity.uri))

        relation.relation = self.domain_profile.predicate_for("hosting_relation", default="hostedBy")
        meta["predicate_corrected_from"] = predicate_name
        meta["predicate_correction_reason"] = "hosting_platform_object_on_identity_or_description_predicate"
        record_event(
            self.quality_events,
            category="hosting_predicate_correction",
            action="corrected",
            discourse_id=discourse_id,
            source_relation_id=str(getattr(relation, "uri", "") or "") or None,
            subject_surface=str(meta.get("subject_text") or "").strip() or None,
            predicate_surface=predicate_name,
            object_surface=label or None,
            reason="hosting_platform_object_on_identity_or_description_predicate",
        )

    # Predicates for which a bare URL IS the claim being asserted (e.g. "X
    # references/is hosted at/is available from <URL>"), so URL-only
    # evidence is not a grounding gap for them. See
    # _is_bare_url_text / the evidence-sufficiency check above.
    _SOURCE_REFERENCE_PREDICATES: Set[str] = {
        "referencessource", "referencesinstrument", "citedat",
        "availablefrom", "hostedby", "providedby", "citation",
        "haslink", "haswebsite", "sourceurl",
    }

    @staticmethod
    def _is_bare_url_text(text: Any) -> bool:
        return bool(re.fullmatch(r"https?://\S+", str(text or "").strip()))

    # Causal/creation-style predicates under which a near-duplicate
    # subject/object surface (e.g. "AI Team" -> "a happier AI team") is a
    # strong signal of a self-referential resolution error rather than a
    # genuine fact. Deliberately narrow: legitimate same-type-different-
    # entity relations (e.g. "Black Defendants" / "White Defendants", which
    # share only a generic head noun) use other predicates and are
    # unaffected.
    _SELF_TRANSFORMATION_PREDICATES: Set[str] = {
        "lead", "leads", "make", "makes", "create", "creates", "cause",
        "causes", "produce", "produces", "become", "becomes", "result",
        "results", "turn", "turns", "transform", "transforms", "yield",
        "yields", "generate", "generates",
    }
    _REFLEXIVE_GUARD_STOPWORDS: Set[str] = {
        "a", "an", "the", "of", "in", "on", "for", "to", "and", "or",
        "with", "by", "is", "are", "was", "were", "this", "that",
        "these", "those",
    }

    @classmethod
    def _reflexive_guard_tokens(cls, text: Any) -> Set[str]:
        norm = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
        return {t for t in norm.split() if t and t not in cls._REFLEXIVE_GUARD_STOPWORDS}

    @classmethod
    def _is_self_referential_relation_shape(
        cls,
        *,
        subject_text: Any,
        object_text: Any,
        predicate_name: Any,
        source_uri: Any,
        target_uri: Any,
        target_is_literal: bool,
    ) -> bool:
        """True when subject and object denote the same thing -- either an
        exact resolved-endpoint match, or (for a small causal/creation
        predicate family) a near-duplicate surface text. See call site in
        _resolve_semantic_unit_relations."""
        if (
            not target_is_literal
            and source_uri
            and target_uri
            and str(source_uri).strip().lower() == str(target_uri).strip().lower()
        ):
            return True

        pred = str(predicate_name or "").strip().lower()
        if pred not in cls._SELF_TRANSFORMATION_PREDICATES:
            return False

        subj_tokens = cls._reflexive_guard_tokens(subject_text)
        obj_tokens = cls._reflexive_guard_tokens(object_text)
        if not subj_tokens or not obj_tokens:
            return False
        union = subj_tokens | obj_tokens
        if not union:
            return False
        jaccard = len(subj_tokens & obj_tokens) / len(union)
        return jaccard >= 0.6

    def _semantic_coverage_segments(self, unit_text: str) -> List[str]:
        text = str(unit_text or "").strip()
        if not text:
            return []
        parts = [segment.strip(" \t\r\n;,.") for segment in re.split(r"[;\n]+", text) if segment.strip(" \t\r\n;,.")]
        return parts or [text]

    def _segment_looks_semantic(self, segment: str) -> bool:
        norm = normalize_text(segment)
        if not norm:
            return False
        if re.search(r"\b(use|used|uses|trained on|reflects|requires|causes|affects|impacts|mitigates|supports|violates|cites|source|access|url|content|includes)\b", norm, flags=re.I):
            return True
        if re.search(r"(https?://|\$|%|\b\d{4}\b|\b\d+[.,]?\d*\b)", segment):
            return True
        if ":" in segment and len(segment.split()) <= 12:
            return True
        return False

    def _relation_supports_segment(self, relation: Relation, segment: str) -> bool:
        meta = getattr(relation, "metadata", None) or {}
        evidence = str(meta.get("evidence_quote") or meta.get("evidence") or getattr(relation, "context", "") or "").strip()
        if not evidence:
            return False
        return normalize_text(segment) in normalize_text(evidence) or normalize_text(evidence) in normalize_text(segment)

    @staticmethod
    def _relation_evidence_text(relation: Relation) -> str:
        meta = getattr(relation, "metadata", None) or {}
        return str(meta.get("evidence_quote") or meta.get("evidence") or getattr(relation, "context", "") or "").strip()

    def _recovered_relation_grounds_in_target(self, relation: Relation, *, segment: str, unit_text: str) -> bool:
        """Defect-2 residual fix (revised): the invariant is not "does this
        relation's span avoid overlapping any other relation anywhere in
        the discourse" (too broad -- one sentence can legitimately support
        several distinct relations) but "did this relation's own evidence
        actually relate to the semantic unit / recovery segment that
        triggered this specific recovery call". The "Recovery focus"
        prompt narrows what the LLM is ASKED to look for, but not what
        it's ALLOWED to return -- it is handed the full discourse_text as
        Context, so it can (and did, for the $765M+ shape) return a
        relation whose evidence has nothing to do with the segment/unit
        that triggered this call, e.g. "Black and Latino borrowers pay
        $765M+..." produced while recovering an entirely different,
        unrelated bullet elsewhere in the document.

        Deliberately reuses _relation_supports_segment -- the exact same
        textual-relatedness test already used to decide a segment was a
        gap in the first place -- just applied in the other direction:
        there, "does any EXISTING relation's evidence already cover this
        segment"; here, "does THIS newly recovered relation's own evidence
        actually relate to the segment (or, failing that, still the wider
        unit) that asked for it". A relation with no evidence text at all
        is left unchecked -- there is nothing to compare, and grounding-
        completeness is a separate concern handled elsewhere.
        """
        if not self._relation_evidence_text(relation):
            return True
        return self._relation_supports_segment(relation, segment) or self._relation_supports_segment(relation, unit_text)

    @staticmethod
    def _relation_endpoint_key(relation: Relation, side: str) -> str:
        if side == "source":
            return str(getattr(relation, "source", "") or "").strip().lower()
        if bool(getattr(relation, "target_is_literal", False)):
            return normalize_text(str(getattr(relation, "target_literal", "") or ""))
        return str(getattr(relation, "target", "") or "").strip().lower()

    def _is_true_duplicate_relation(self, candidate: Relation, existing_relations: List[Relation]) -> bool:
        """Secondary check, applied ONLY to a candidate that already passed
        the in-target check above. Shared/overlapping evidence text alone
        is NOT sufficient to call two relations duplicates -- e.g. "Black
        and Latino borrowers pay 7.9 and 3.6 basis points more
        respectively" legitimately supports two distinct relations (one
        per figure) from the same sentence. Only when the candidate names
        the same normalised (source, target) endpoint pair AS AN EXISTING
        relation, AND their evidence texts actually relate to each other,
        is it the same underlying fact re-extracted rather than a second,
        genuinely distinct one."""
        cand_src = self._relation_endpoint_key(candidate, "source")
        cand_tgt = self._relation_endpoint_key(candidate, "target")
        if not cand_src or not cand_tgt:
            return False
        cand_evidence = self._relation_evidence_text(candidate)
        for other in existing_relations:
            if self._relation_endpoint_key(other, "source") != cand_src:
                continue
            if self._relation_endpoint_key(other, "target") != cand_tgt:
                continue
            other_evidence = self._relation_evidence_text(other)
            if not cand_evidence or not other_evidence:
                return True
            if normalize_text(cand_evidence) in normalize_text(other_evidence) or normalize_text(other_evidence) in normalize_text(cand_evidence):
                return True
        return False

    def _recover_semantic_coverage_for_unit(
        self,
        *,
        relations: List[Relation],
        unit_text: str,
        support_context: str,
        local_entities: List[Entity],
        discourse_text: str,
        discourse_id: str,
        field: str,
        origin_key: str,
        unit: Dict[str, Any],
    ) -> Tuple[List[Relation], Dict[str, int]]:
        segments = [segment for segment in self._semantic_coverage_segments(unit_text) if self._segment_looks_semantic(segment)]
        gaps = [segment for segment in segments if not any(self._relation_supports_segment(relation, segment) for relation in relations)]
        recovered = 0
        max_passes = int(getattr(self.config, "MAX_COVERAGE_RECOVERY_PASSES", 2) or 2)
        gap_batch = int(getattr(self.config, "MAX_COVERAGE_GAP_BATCH", 8) or 8)
        for _pass in range(max_passes):
            if not gaps:
                break
            new_relations: List[Relation] = []
            for segment in gaps[:gap_batch]:
                recovery_text = f"{unit_text}\nRecovery focus: {segment}"
                recovered_relations = self._extract_relations_with_pair_focus(
                    unit_text=recovery_text,
                    local_entities=local_entities,
                    discourse_id=discourse_id,
                    candidate_pairs=[],
                    context=discourse_text,
                )
                recovered_relations = self._resolve_semantic_unit_relations(
                    relations=recovered_relations,
                    local_entities=local_entities,
                    discourse_text=discourse_text,
                    discourse_id=discourse_id,
                    field=field,
                    origin_key=origin_key,
                    unit=unit,
                    unit_text=unit_text,
                )
                for relation in recovered_relations:
                    # Primary gate: the relation's own evidence must
                    # actually relate to the segment/unit that triggered
                    # this recovery call.
                    if not self._recovered_relation_grounds_in_target(relation, segment=segment, unit_text=unit_text):
                        continue
                    # Secondary gate: even in-target, reject only when it
                    # ALSO demonstrates a true duplicate (same endpoints,
                    # related evidence) against an already-accepted
                    # relation for this same unit -- not merely shared
                    # evidence text with a DIFFERENT endpoint pair.
                    if self._is_true_duplicate_relation(relation, relations) or self._is_true_duplicate_relation(relation, new_relations):
                        continue
                    meta = getattr(relation, "metadata", None) or {}
                    meta["coverage_recovery"] = True
                    relation.metadata = meta
                    new_relations.append(relation)
            if not new_relations:
                break
            recovered += len(new_relations)
            relations.extend(new_relations)
            gaps = [segment for segment in gaps if not any(self._relation_supports_segment(relation, segment) for relation in relations)]
        return relations, {
            "coverage_gaps_detected": len(segments),
            "coverage_gaps_recovered": recovered,
            "unresolved_coverage_gaps": len(gaps),
        }

    def _build_chunk_spans(
        self,
        text: str,
        chunks: List[str],
        discourse_id: Any,
        origin_key: Optional[str] = None,
    ) -> List[Dict[str, int]]:
        did = str(discourse_id or "").strip()
        spans: List[Dict[str, int]] = []
        native_chunks = SemanticChunker.chunk_with_offsets(str(text or ""))
        for idx, item in enumerate(native_chunks):
            start = self._coerce_nonneg_int(item.get("start_offset"))
            end = self._coerce_nonneg_int(item.get("end_offset"))
            if start is None or end is None or end <= start:
                continue
            spans.append(
                {
                    "discourse_id": did,  # type: ignore[dict-item]
                    "chunk_index": int(idx),
                    "start_offset": int(start),
                    "end_offset": int(end),
                }
            )
        if did:
            self._discourse_chunk_spans[did] = spans
        okey = str(origin_key or "").strip()
        if okey and okey != did:
            self._discourse_chunk_spans[okey] = spans
        return spans

    def _get_chunk_spans_for_discourse(
        self,
        discourse_id: Any,
        origin_key: Optional[str] = None,
    ) -> List[Dict[str, int]]:
        did = str(discourse_id or "").strip()
        okey = str(origin_key or "").strip()
        if not did and not okey:
            return []
        spans = []
        if okey:
            spans = self._discourse_chunk_spans.get(okey, [])
        if not spans and did:
            spans = self._discourse_chunk_spans.get(did, [])
        return [dict(s) for s in spans]

    def _extract_relation_span_from_metadata(self, meta: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        s = (
            self._coerce_nonneg_int(meta.get("char_start"))
            if meta.get("char_start") is not None
            else self._coerce_nonneg_int(meta.get("start_char"))
        )
        if s is None:
            s = self._coerce_nonneg_int(meta.get("start_offset"))
        e = (
            self._coerce_nonneg_int(meta.get("char_end"))
            if meta.get("char_end") is not None
            else self._coerce_nonneg_int(meta.get("end_char"))
        )
        if e is None:
            e = self._coerce_nonneg_int(meta.get("end_offset"))
        if s is None or e is None:
            return None
        if e <= s:
            e = s + 1
        return int(s), int(e)

    def _derive_relation_span_from_entities(
        self,
        relation: Relation,
        uri_to_entity: Dict[str, Entity],
    ) -> Optional[Tuple[int, int]]:
        def _entity_span(ent: Optional[Entity]) -> Optional[Tuple[int, int]]:
            if ent is None:
                return None
            meta = getattr(ent, "metadata", None) or {}
            s = self._coerce_nonneg_int(meta.get("start_char"))
            e = self._coerce_nonneg_int(meta.get("end_char"))
            if s is None and getattr(ent, "start_char", None) is not None:
                s = self._coerce_nonneg_int(getattr(ent, "start_char", None))
            if e is None and getattr(ent, "end_char", None) is not None:
                e = self._coerce_nonneg_int(getattr(ent, "end_char", None))
            if s is None or e is None:
                return None
            if e <= s:
                e = s + 1
            return int(s), int(e)

        src_ent = uri_to_entity.get(str(getattr(relation, "source", "") or ""))
        tgt_ent = uri_to_entity.get(str(getattr(relation, "target", "") or ""))
        src_span = _entity_span(src_ent)
        tgt_span = _entity_span(tgt_ent)
        if src_span and tgt_span:
            return min(src_span[0], tgt_span[0]), max(src_span[1], tgt_span[1])
        if src_span:
            return src_span
        if tgt_span:
            return tgt_span
        return None

    def _entity_span_from_uri(
        self, uri: Any, discourse_id: Any = None, *, field: Any = None,
    ) -> Optional[Tuple[int, int]]:
        """Returns this entity's own mention span, SCOPED to the requested
        discourse_id (and, where the mention itself records one, field) --
        never an arbitrary span belonging to a DIFFERENT discourse just
        because the entity was later consolidated with mentions from
        elsewhere (the confirmed root cause of a real cross-discourse
        span-drift bug fixed earlier in serializers/rdf_serializer.py;
        this is the same class of risk at the provenance-grounding layer).

        discourse_id ranges are NOT globally unique across record types in
        this dataset (a Theme and a Question can share the same numeric
        id) -- field is used, where available, as a second scoping key for
        exactly that reason.
        """
        ent_uri = str(uri or "").strip()
        if not ent_uri:
            return None
        ent = None
        registry = getattr(self, "global_registry", None)
        if registry is None:
            return None
        store = getattr(registry, "entities", None)
        if isinstance(store, dict):
            ent = store.get(ent_uri)
        if ent is None:
            get_all = getattr(registry, "get_all_entities", None)
            if callable(get_all):
                for cand in get_all():
                    if str(getattr(cand, "uri", "") or "").strip() == ent_uri:
                        ent = cand
                        break
        if ent is None:
            return None

        meta = getattr(ent, "metadata", None) or {}
        did = str(discourse_id or "").strip()
        req_field = str(field or "").strip()
        mentions = meta.get("mentions")
        if isinstance(mentions, list):
            scoped: List[Tuple[int, int]] = []
            for item in mentions:
                norm = self._normalize_entity_mention(item)
                if norm is None:
                    continue
                if did and str(norm.get("discourse_id", "")).strip() != did:
                    continue
                item_field = str((item or {}).get("field") or "").strip() if isinstance(item, dict) else ""
                # Only reject on field mismatch when BOTH sides actually
                # record one -- a mention with no field info is not
                # rejected (missing evidence, not confirmed
                # incompatibility), it is simply not additionally
                # verified beyond the discourse_id match above.
                if req_field and item_field and item_field != req_field:
                    continue
                scoped.append((int(norm["start_char"]), int(norm["end_char"])))
            if scoped:
                min_start = min(x[0] for x in scoped)
                max_end = max(x[1] for x in scoped)
                if max_end <= min_start:
                    max_end = min_start + 1
                return int(min_start), int(max_end)
            if did:
                # A mentions list EXISTS but none of its entries belong to
                # the requested discourse -- this is real, positive
                # information that this entity was NOT mentioned here (at
                # least not with a tracked span). Falling through to the
                # entity's top-level scalar start_char/end_char below would
                # risk using a span from a completely different discourse
                # (get_or_create's cross-mention scalar drift) -- refuse
                # rather than guess.
                return None

        if did:
            # No mentions list at all: we have no per-occurrence data to
            # verify against. Reject only on an ACTIVE mismatch -- a
            # scalar discourse_id (and, where recorded, field) that
            # positively names a DIFFERENT discourse than the one being
            # grounded (get_or_create's cross-mention scalar drift keeps
            # discourse_id and start_char/end_char in lockstep -- see
            # serializers/rdf_serializer.py's primary_mention fix -- so a
            # drifted scalar span is caught here too). An entity with NO
            # discourse_id recorded at all is missing evidence, not
            # confirmed to belong elsewhere -- allowed through, same as
            # before this fix, rather than rejected on absence alone.
            scalar_did = str(meta.get("discourse_id") or "").strip()
            if scalar_did and scalar_did != did:
                return None
            scalar_field = str(meta.get("field") or "").strip()
            if req_field and scalar_field and scalar_field != req_field:
                return None

        s = self._coerce_nonneg_int(meta.get("start_char"))
        e = self._coerce_nonneg_int(meta.get("end_char"))
        if s is None and getattr(ent, "start_char", None) is not None:
            s = self._coerce_nonneg_int(getattr(ent, "start_char", None))
        if e is None and getattr(ent, "end_char", None) is not None:
            e = self._coerce_nonneg_int(getattr(ent, "end_char", None))
        if s is None or e is None:
            return None
        if e <= s:
            e = s + 1
        return int(s), int(e)

    def _map_span_to_chunk_segments(
        self,
        *,
        discourse_id: str,
        span_start: int,
        span_end: int,
        chunk_spans: List[Dict[str, int]],
    ) -> List[Dict[str, int]]:
        did = str(discourse_id or "").strip()
        segments: List[Dict[str, int]] = []
        for ch in sorted(chunk_spans, key=lambda x: int(x.get("chunk_index", 0))):
            ch_start = self._coerce_nonneg_int(ch.get("start_offset"))
            ch_end = self._coerce_nonneg_int(ch.get("end_offset"))
            ch_idx = self._coerce_nonneg_int(ch.get("chunk_index"))
            if ch_start is None or ch_end is None or ch_idx is None:
                continue
            if ch_end <= span_start or ch_start >= span_end:
                continue
            seg_start = max(span_start, ch_start)
            seg_end = min(span_end, ch_end)
            if seg_end <= seg_start:
                continue
            segments.append(
                {
                    "discourse_id": did,  # type: ignore[dict-item]
                    "chunk_index": int(ch_idx),
                    "char_start": int(seg_start),
                    "char_end": int(seg_end),
                    "start_offset": int(seg_start),
                    "end_offset": int(seg_end),
                }
            )
        return segments

    def _ground_relation_metadata(
        self,
        meta: Dict[str, Any],
        *,
        discourse_id: Any,
        discourse_text: str = "",
        source_uri: Any = None,
        target_uri: Any = None,
        chunk_spans: Optional[List[Dict[str, int]]] = None,
        counters: Optional[Dict[str, int]] = None,
        source_text_lookup: Optional[Any] = None,
    ) -> None:
        did = str(discourse_id or meta.get("discourse_id") or meta.get("doc_id") or "").strip()
        if did:
            meta["discourse_id"] = did
        field = str(meta.get("field") or "").strip()
        origin_key = str(meta.get("origin_key") or self._compose_origin_key(did, field)).strip()
        if origin_key:
            meta["origin_key"] = origin_key
        spans = (
            chunk_spans
            if chunk_spans is not None
            else self._get_chunk_spans_for_discourse(did, origin_key=origin_key)
        )

        # Authoritative-source-text fallback: annotations_dict's own
        # per-record metadata.text (discourse_text, passed in by the
        # caller) is reliably populated for some record types (Theme) but
        # not others (confirmed: Answer/Question records routinely lack
        # it) -- when it's missing, fall back to the injected
        # (field, discourse_id) -> raw text lookup (utils.source_text_
        # lookup.SourceTextLookup, built once from the raw input dataset
        # by the CALLER -- this function never opens any file itself).
        # Same field semantics as validation/span_integrity.py's
        # resolve_source_text callable.
        effective_text = str(discourse_text or "").strip()
        if not effective_text and source_text_lookup is not None and did and field:
            try:
                looked_up = source_text_lookup(field, did)
            except Exception:
                looked_up = None
            if looked_up:
                effective_text = str(looked_up).strip()
                if effective_text:
                    meta["provenance_text_source"] = "authoritative_source_text_lookup"
        discourse_text = effective_text

        span = self._extract_relation_span_from_metadata(meta)
        method = "none"
        status = "ungrounded"
        chunk_index: Optional[int] = None
        evidence_spans: List[Dict[str, int]] = []
        existing_chunk_index = self._coerce_nonneg_int(meta.get("chunk_index"))
        # Set by the priority-(c)-single-endpoint and priority-(d)-unique-
        # chunk-evidence fallbacks below, so the final grounding method
        # stays distinguishable from "span_map" (an explicit relation span
        # or a whole-discourse evidence-quote match) even though both
        # paths reuse the same span-to-chunk mapping utility afterward.
        span_source: Optional[str] = None

        if span is None:
            quote = str(meta.get("evidence_quote") or meta.get("evidence") or "").strip()
            text = str(discourse_text or "").strip()
            if quote and text:
                quote_start = text.find(quote)
                if quote_start >= 0:
                    quote_end = quote_start + len(quote)
                    meta["char_start"] = int(quote_start)
                    meta["char_end"] = int(quote_end)
                    meta["start_offset"] = int(quote_start)
                    meta["end_offset"] = int(quote_end)
                    span = (int(quote_start), int(quote_end))

        if span is None:
            src_span = self._entity_span_from_uri(source_uri, did, field=field)
            tgt_span = self._entity_span_from_uri(target_uri, did, field=field)
            if src_span is not None and tgt_span is not None:
                derived_start = min(int(src_span[0]), int(tgt_span[0]))
                derived_end = max(int(src_span[1]), int(tgt_span[1]))
                if derived_end <= derived_start:
                    derived_end = derived_start + 1
                meta["char_start"] = int(derived_start)
                meta["char_end"] = int(derived_end)
                meta["start_offset"] = int(derived_start)
                meta["end_offset"] = int(derived_end)
                span = (int(derived_start), int(derived_end))
                if counters is not None:
                    counters["derived_span_from_entities"] = int(counters.get("derived_span_from_entities", 0)) + 1
            elif src_span is not None or tgt_span is not None:
                # Priority (c), single-endpoint case: the ORIGINAL two-span
                # derivation above required BOTH endpoints to have a
                # registered mention span, so a relation whose subject (or
                # object) has one but the other genuinely doesn't fell all
                # the way through to "unknown" even though the ONE
                # available endpoint still deterministically anchors the
                # relation to a real chunk. This is intentionally kept
                # distinguishable from the two-endpoint case (a different
                # groundingMethod, single_entity_span, not "span_map" or
                # silently folded into derived_span_from_entities) --
                # weaker evidence than two agreeing endpoints, but still a
                # genuine, non-guessed span, not a nearest-text/label
                # guess.
                only_span = src_span if src_span is not None else tgt_span
                meta["char_start"] = int(only_span[0])
                meta["char_end"] = int(only_span[1])
                meta["start_offset"] = int(only_span[0])
                meta["end_offset"] = int(only_span[1])
                span = (int(only_span[0]), int(only_span[1]))
                span_source = "single_entity_span"
                if counters is not None:
                    counters["derived_span_from_single_entity"] = int(
                        counters.get("derived_span_from_single_entity", 0)
                    ) + 1

        if span is None:
            if counters is not None:
                counters["missing_span_offsets"] = int(counters.get("missing_span_offsets", 0)) + 1
            meta.setdefault("char_start", -1)
            meta.setdefault("char_end", -1)
            meta.setdefault("start_offset", -1)
            meta.setdefault("end_offset", -1)
            if existing_chunk_index is not None:
                chunk_index = int(existing_chunk_index)
                status = "grounded"
                method = str(meta.get("groundingMethod") or "none")
        else:
            span_start, span_end = span
            evidence_spans = self._map_span_to_chunk_segments(
                discourse_id=did,
                span_start=span_start,
                span_end=span_end,
                chunk_spans=spans,
            )
            if not evidence_spans and counters is not None:
                counters["span_out_of_range"] = int(counters.get("span_out_of_range", 0)) + 1

            if len(evidence_spans) == 1:
                chunk_index = int(evidence_spans[0]["chunk_index"])
                method = "span_map"
                status = "grounded"
                seg = evidence_spans[0]
                meta["char_start"] = int(seg["char_start"])
                meta["char_end"] = int(seg["char_end"])
                meta["start_offset"] = int(seg["start_offset"])
                meta["end_offset"] = int(seg["end_offset"])
            elif len(evidence_spans) > 1:
                method = "span_map"
                status = "grounded"
                meta["evidence_spans"] = evidence_spans
                if counters is not None:
                    counters["cross_chunk_evidence"] = int(counters.get("cross_chunk_evidence", 0)) + 1

            if span_source and method == "span_map":
                # Preserve the weaker-evidence-tier label instead of
                # letting it collapse into the same "span_map" used for an
                # explicit relation span or a whole-discourse evidence-quote
                # match -- same chunk-mapping mechanics, genuinely
                # different evidentiary strength, kept visible.
                method = span_source

        if chunk_index is None:
            # Priority (d): every attempt above either found no explicit/
            # derived span at all, or found one whose mapping into chunks
            # came back empty or ambiguous (span_out_of_range /
            # cross_chunk_evidence) -- before giving up, check whether the
            # RAW evidence quote occurs in exactly ONE candidate chunk's
            # own text, independent of whatever position a whole-text
            # .find() landed on (that position can straddle a chunk
            # boundary or hit an earlier, differently-scoped occurrence of
            # the same quote even when a later occurrence sits cleanly
            # inside a single chunk). Never falls back to "nearest chunk"
            # or any similarity heuristic: if the quote appears in zero or
            # more than one chunk, this step does nothing and the relation
            # proceeds to the remaining (weaker) single_chunk_discourse
            # fallback or is left genuinely unknown.
            quote = str(meta.get("evidence_quote") or meta.get("evidence") or "").strip()
            text = str(discourse_text or "").strip()
            if quote and text and spans:
                matching_chunks = []
                for ch in spans:
                    cs = self._coerce_nonneg_int(ch.get("start_offset"))
                    ce = self._coerce_nonneg_int(ch.get("end_offset"))
                    ci = self._coerce_nonneg_int(ch.get("chunk_index"))
                    if cs is None or ce is None or ci is None or ce <= cs or ce > len(text):
                        continue
                    if quote in text[cs:ce]:
                        matching_chunks.append((ci, cs, ce))
                if len(matching_chunks) == 1:
                    ci, cs, ce = matching_chunks[0]
                    local_idx = text[cs:ce].find(quote)
                    abs_start = cs + local_idx
                    abs_end = abs_start + len(quote)
                    meta["char_start"] = int(abs_start)
                    meta["char_end"] = int(abs_end)
                    meta["start_offset"] = int(abs_start)
                    meta["end_offset"] = int(abs_end)
                    meta.pop("evidence_spans", None)
                    chunk_index = int(ci)
                    method = "unique_chunk_evidence_match"
                    status = "grounded"
                    if counters is not None:
                        counters["filled_from_unique_chunk_evidence_match"] = int(
                            counters.get("filled_from_unique_chunk_evidence_match", 0)
                        ) + 1

        if chunk_index is None and len(spans) == 1:
            only_ci = self._coerce_nonneg_int(spans[0].get("chunk_index"))
            if only_ci is not None:
                chunk_index = int(only_ci)
                method = "single_chunk_discourse"
                status = "grounded"

        if chunk_index is None:
            meta["chunk_index"] = -1
            meta["chunk_index_unknown"] = True
            meta["groundingStatus"] = status
            meta["groundingMethod"] = method
            meta.pop("chunk_id", None)
            source_doc_id = meta.get("source_doc_id") or did
            if source_doc_id:
                meta["source_doc_id"] = str(source_doc_id)
            if counters is not None:
                # Count truly ungrounded unknowns; exclude grounded cross-chunk
                # evidence (meta["evidence_spans"] is ONLY ever populated by the
                # len(evidence_spans) > 1 branch above, regardless of which
                # span-priority tier produced the underlying span -- explicit,
                # single-entity, or unique-chunk-evidence-match all funnel
                # through the same mapping call, so checking evidence_spans
                # directly rather than the exact method string keeps this
                # exclusion correct for all of them, not just "span_map").
                if not (status == "grounded" and bool(meta.get("evidence_spans"))):
                    counters["unknown_chunk_index_count"] = int(counters.get("unknown_chunk_index_count", 0)) + 1
        else:
            meta["chunk_index"] = int(chunk_index)
            meta["chunk_index_unknown"] = False
            meta["groundingStatus"] = status
            meta["groundingMethod"] = method
            source_doc_id = meta.get("source_doc_id") or did
            if source_doc_id:
                source_doc_id = str(source_doc_id)
                meta["source_doc_id"] = source_doc_id
                meta["chunk_id"] = str(meta.get("chunk_id") or f"{source_doc_id}:chunk:{chunk_index}")
            if "evidence_spans" in meta and len(evidence_spans) <= 1:
                meta.pop("evidence_spans", None)

    def _ensure_relation_chunk_provenance(
        self,
        relation: Relation,
        fallback_discourse_id: Any = None,
        discourse_text: str = "",
        chunk_spans: Optional[List[Dict[str, int]]] = None,
        counters: Optional[Dict[str, int]] = None,
        source_text_lookup: Optional[Any] = None,
    ) -> None:
        meta = getattr(relation, "metadata", None) or {}
        self._ground_relation_metadata(
            meta,
            discourse_id=fallback_discourse_id,
            discourse_text=discourse_text,
            source_uri=getattr(relation, "source", None),
            target_uri=getattr(relation, "target", None),
            chunk_spans=chunk_spans,
            counters=counters,
            source_text_lookup=source_text_lookup,
        )
        relation.metadata = meta

    def fill_missing_relation_provenance(
        self,
        annotations_dict: Dict[str, Any],
        source_text_lookup: Optional[Any] = None,
        *,
        failures_output_path: Optional[Any] = None,
        run_id: Optional[str] = None,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic pre-serialization provenance fill pass.
        Only fills chunk provenance when:
          a) span->chunk mapping succeeds
          b) discourse has exactly one chunk
        No discourse-majority or first-chunk guessing is performed.

        :param source_text_lookup: optional callable/object
            `(field, discourse_id) -> Optional[str]` resolving the
            AUTHORITATIVE raw source-field text (e.g. a
            utils.source_text_lookup.SourceTextLookup built from the raw
            input dataset by the caller). Used only as a fallback when a
            record's own annotations metadata.text is missing -- this
            function never opens any file itself; the caller decides what
            "authoritative" means and supplies it explicitly.
        :param failures_output_path: when given, and the gate is about to
            fail, every unresolved relation's diagnostic record (see
            utils.checkpoint.build_provenance_failure_record) is written
            to this path (atomic write) BEFORE the RuntimeError is raised,
            so a future failure is diagnosable without needing to
            re-extract. When omitted, no diagnostic file is written
            (unchanged from before this feature existed).
        """
        from utils.checkpoint import build_provenance_failure_record, write_provenance_grounding_failures
        sections = ("themes", "questions", "answers", "votes", "experts", "documents")
        counters: Dict[str, int] = {
            "missing_span_offsets": 0,
            "derived_span_from_entities": 0,
            "span_out_of_range": 0,
            "cross_chunk_evidence": 0,
            "unknown_chunk_index_count": 0,
            "inferred_relations_excluded_from_gate": 0,
            "non_textual_relations_excluded_from_gate": 0,
            "extracted_relations_checked": 0,
            "total_relations_checked": 0,
            "total_relations": 0,
            "filled_from_span_map": 0,
            "filled_from_single_chunk_discourse": 0,
        }
        unknown_by_extractor: Dict[str, int] = {}
        unknown_by_relation: Dict[str, int] = {}
        unknown_by_reason: Dict[str, int] = {}
        failure_records: List[Dict[str, Any]] = []
        _rel_seq = {"n": 0}

        def _inc_bucket(bucket: Dict[str, int], key: str) -> None:
            bucket[key] = int(bucket.get(key, 0)) + 1

        def _as_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))
            if v is None:
                return False
            s = str(v).strip().lower()
            return s in {"true", "1", "yes", "y"}

        def _fill_one(rel_dict: Dict[str, Any], section: str = "unknown") -> None:
            counters["total_relations"] = int(counters.get("total_relations", 0)) + 1
            _rel_seq["n"] += 1
            rel_id = f"rel-{_rel_seq['n']:05d}"
            meta = rel_dict.get("metadata", {}) or {}
            rel_dict["metadata"] = meta
            did = meta.get("discourse_id") or meta.get("doc_id")
            origin_key = str(meta.get("origin_key") or self._compose_origin_key(did, meta.get("field"))).strip()
            if origin_key:
                meta["origin_key"] = origin_key
            extractor = str(meta.get("extractor") or "").strip().lower()
            non_textual_extractor = extractor in {"structural", "vote", "lightweight"}
            relation_type = str(rel_dict.get("relation") or meta.get("relation") or "").strip() or "unknown"
            source_uri = str(rel_dict.get("source") or rel_dict.get("source_id") or "").strip()
            target_uri = str(rel_dict.get("target") or rel_dict.get("target_id") or "").strip()
            discourse_text = str(payload.get("metadata", {}).get("text") or "")
            chunk_spans = self._get_chunk_spans_for_discourse(did, origin_key=origin_key)
            chunk_spans_count = len(chunk_spans)
            relation_span_before = self._extract_relation_span_from_metadata(meta)
            rel_field = meta.get("field")
            source_span_before = self._entity_span_from_uri(source_uri, did, field=rel_field)
            target_span_before = self._entity_span_from_uri(target_uri, did, field=rel_field)
            source_has_span = source_span_before is not None
            target_has_span = target_span_before is not None
            excluded_from_gate = (
                extractor == "reasoner"
                or non_textual_extractor
                or _as_bool(meta.get("isInferred"))
                or _as_bool(meta.get("is_inferred"))
                or _as_bool(rel_dict.get("isInferred"))
                or _as_bool(rel_dict.get("is_inferred"))
            )
            unknown_before = int(counters.get("unknown_chunk_index_count", 0))
            self._ground_relation_metadata(
                meta,
                discourse_id=did,
                discourse_text=discourse_text,
                source_uri=source_uri,
                target_uri=target_uri,
                chunk_spans=chunk_spans,
                counters=counters,
                source_text_lookup=source_text_lookup,
            )
            unknown_after = int(counters.get("unknown_chunk_index_count", 0))
            if excluded_from_gate:
                if non_textual_extractor:
                    counters["non_textual_relations_excluded_from_gate"] = int(
                        counters.get("non_textual_relations_excluded_from_gate", 0)
                    ) + 1
                else:
                    counters["inferred_relations_excluded_from_gate"] = int(
                        counters.get("inferred_relations_excluded_from_gate", 0)
                    ) + 1
                if unknown_after > unknown_before:
                    counters["unknown_chunk_index_count"] = max(0, unknown_before)
            else:
                counters["extracted_relations_checked"] = int(counters.get("extracted_relations_checked", 0)) + 1
                counters["total_relations_checked"] = int(counters.get("total_relations_checked", 0)) + 1
                was_counted_unknown = unknown_after > unknown_before
                if was_counted_unknown:
                    evidence_spans = meta.get("evidence_spans")
                    evidence_spans_len = len(evidence_spans) if isinstance(evidence_spans, list) else 0
                    method = str(meta.get("groundingMethod") or "none")
                    if relation_span_before is None and (not source_has_span) and (not target_has_span):
                        failure_reason = "A_missing_relation_and_entity_spans"
                    elif method == "span_map" and evidence_spans_len > 0:
                        failure_reason = "C_cross_chunk_span_map_unknown_counted"
                    elif relation_span_before is not None and chunk_spans_count > 0 and evidence_spans_len == 0:
                        failure_reason = "B_span_out_of_range_or_chunk_map_miss"
                    else:
                        failure_reason = "D_other"
                    _inc_bucket(unknown_by_extractor, extractor or "unknown")
                    _inc_bucket(unknown_by_relation, relation_type)
                    _inc_bucket(unknown_by_reason, failure_reason)
                    failure_records.append(
                        build_provenance_failure_record(
                            identifier=rel_id,
                            section=section,
                            rel_dict=rel_dict,
                            source_span=source_span_before,
                            target_span=target_span_before,
                            chunk_spans_count=chunk_spans_count,
                            failure_reason=failure_reason,
                        )
                    )

            if str(meta.get("groundingMethod")) == "span_map" and self._coerce_nonneg_int(meta.get("chunk_index")) is not None:
                counters["filled_from_span_map"] = int(counters.get("filled_from_span_map", 0)) + 1
            if str(meta.get("groundingMethod")) == "single_chunk_discourse" and self._coerce_nonneg_int(meta.get("chunk_index")) is not None:
                counters["filled_from_single_chunk_discourse"] = int(
                    counters.get("filled_from_single_chunk_discourse", 0)
                ) + 1

        for section in sections:
            block = annotations_dict.get(section, {})
            if not isinstance(block, dict):
                continue
            for _, payload in block.items():
                if not isinstance(payload, dict):
                    continue
                for rel in payload.get("relations", []):
                    if isinstance(rel, dict):
                        _fill_one(rel, section=section)

        inferred_list = (annotations_dict.get("metadata", {}) or {}).get("inferred_relations", [])
        if isinstance(inferred_list, list):
            for rel in inferred_list:
                if isinstance(rel, dict):
                    _fill_one(rel, section="inferred_relations")

        total_relations = int(counters.get("total_relations", 0))
        total_relations_checked = int(counters.get("total_relations_checked", 0))
        unknown = int(counters.get("unknown_chunk_index_count", 0))
        unknown_rate = (float(unknown) / float(total_relations_checked)) if total_relations_checked > 0 else 0.0
        unknown_by_extractor_pct = {
            key: {
                "count": int(val),
                "pct_of_checked": round((float(val) / float(total_relations_checked)) * 100.0, 3)
                if total_relations_checked > 0
                else 0.0,
            }
            for key, val in sorted(unknown_by_extractor.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        }
        unknown_by_relation_pct = {
            key: {
                "count": int(val),
                "pct_of_checked": round((float(val) / float(total_relations_checked)) * 100.0, 3)
                if total_relations_checked > 0
                else 0.0,
            }
            for key, val in sorted(unknown_by_relation.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        }
        unknown_by_reason_pct = {
            key: {
                "count": int(val),
                "pct_of_checked": round((float(val) / float(total_relations_checked)) * 100.0, 3)
                if total_relations_checked > 0
                else 0.0,
            }
            for key, val in sorted(unknown_by_reason.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        }
        strict_mode = bool(getattr(self.config, "STRICT_PROVENANCE_GROUNDING", False))
        max_unknown = 0.0 if strict_mode else float(getattr(self.config, "MAX_UNKNOWN_CHUNK_INDEX_PCT", 0.02))
        gate_passed = unknown_rate <= max_unknown
        result = {
            "total_relations": total_relations,
            "total_relations_checked": total_relations_checked,
            "extracted_relations_checked": int(counters.get("extracted_relations_checked", 0)),
            "inferred_relations_excluded_from_gate": int(counters.get("inferred_relations_excluded_from_gate", 0)),
            "non_textual_relations_excluded_from_gate": int(
                counters.get("non_textual_relations_excluded_from_gate", 0)
            ),
            "filled_from_span_map": int(counters.get("filled_from_span_map", 0)),
            "filled_from_single_chunk_discourse": int(counters.get("filled_from_single_chunk_discourse", 0)),
            "missing_span_offsets": int(counters.get("missing_span_offsets", 0)),
            "derived_span_from_entities": int(counters.get("derived_span_from_entities", 0)),
            "span_out_of_range": int(counters.get("span_out_of_range", 0)),
            "cross_chunk_evidence": int(counters.get("cross_chunk_evidence", 0)),
            "unknown_chunk_index_count": unknown,
            "unknown_chunk_index_rate": round(unknown_rate, 6),
            "unknown_non_excluded_by_extractor": unknown_by_extractor_pct,
            "unknown_non_excluded_by_relation": unknown_by_relation_pct,
            "unknown_non_excluded_by_reason": unknown_by_reason_pct,
            "unknown_chunk_index_gate_threshold": max_unknown,
            "strict_mode": strict_mode,
            "gate_passed": gate_passed,
        }
        self._provenance_diagnostics = dict(result)
        print("\n[Provenance grounding breakdown (NON-excluded relations)]")
        print(f"  total_relations_checked: {total_relations_checked}")
        print(f"  unknown_chunk_index_count: {unknown}")
        print(f"  unknown_chunk_index_rate: {unknown_rate:.4%}")
        print(f"  missing_span_offsets: {int(counters.get('missing_span_offsets', 0))}")
        print(f"  derived_span_from_entities: {int(counters.get('derived_span_from_entities', 0))}")
        print(f"  span_out_of_range: {int(counters.get('span_out_of_range', 0))}")
        print(f"  cross_chunk_evidence: {int(counters.get('cross_chunk_evidence', 0))}")
        print("  unknown_by_extractor:", json.dumps(unknown_by_extractor_pct, ensure_ascii=False))
        print("  unknown_by_relation:", json.dumps(unknown_by_relation_pct, ensure_ascii=False))
        print("  unknown_by_reason:", json.dumps(unknown_by_reason_pct, ensure_ascii=False))
        if not gate_passed:
            if failures_output_path is not None:
                try:
                    write_provenance_grounding_failures(
                        failures_output_path,
                        failures=failure_records,
                        run_id=run_id,
                        run_context=run_context,
                        summary=result,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist provenance_grounding_failures.json -- "
                        "raising the original gate failure regardless."
                    )
            raise RuntimeError(
                "Provenance grounding gate failed: "
                f"unknown_chunk_index_rate={unknown_rate:.4f} exceeds threshold={max_unknown:.4f}"
            )
        logger.info("Pre-serialization provenance fill: %s", result)
        return result
    
    # ---------------------------------------------------------
    # Entity processing
    # ---------------------------------------------------------
    # -----------------------------------------------------------------
    # Context-aware Dataset-family disambiguation
    #
    # General linguistic/contextual signals only -- no dataset IDs, entity
    # names, or domain-specific phrases. The same closed, generic-English
    # vocabulary here applies identically to any future dataset/domain: a
    # noun phrase led by "study"/"analysis"/"software" etc, or surrounded
    # by generic research/publication/software vocabulary, is not
    # dataset-specific knowledge -- it is ordinary English grammar.
    # -----------------------------------------------------------------
    _DATASET_FAMILY_TYPES: Set[str] = {"Dataset", "Corpus", "DataSource"}

    _DATASET_FAMILY_LABEL_LEAD_MAP: Dict[str, Tuple[str, ...]] = {
        "ResearchStudy": ("study", "studies", "survey", "investigation", "research"),
        "Analysis": ("analysis", "analyses", "review", "evaluation", "examination", "audit"),
        "Publication": ("paper", "article", "report", "publication", "journal"),
        "SoftwareProduct": ("software", "toolkit", "tool", "application", "package", "spreadsheet", "program"),
    }

    # "application" is deliberately excluded as a bare keyword -- it is
    # genuinely ambiguous in general English (a housing/job/loan
    # "application" is not software); only the unambiguous compound forms
    # are used as signals.
    _DATASET_FAMILY_CONTEXT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "SoftwareProduct": ("software", "software application", "web application", "mobile application", "toolkit", "spreadsheet", "programme", "computer program"),
        "Publication": ("published", "publication", "journal", "article", "paper", "authors"),
        "ResearchStudy": ("study", "studies", "researchers", "investigated", "investigation", "conducted", "survey"),
        "Analysis": ("analysis", "analyzed", "analysed", "analyses", "evaluated", "examined"),
    }

    # A label that itself begins with a bare number (optionally currency-
    # or percent-marked) is a quantity/statistic/count phrase, not a named
    # resource -- structural, not lexical: works for any number in any
    # domain. This only stops confident Dataset typing
    # here; full entity-vs-literal handling is the quantity/literal
    # classifier's concern.
    _QUANTITY_LEAD_RE = re.compile(r"^\$?\d[\d,.\-]*\s*%?\+?\b")

    # External-grounding contradiction vocabulary: generic phrases a
    # Wikidata short description would use for a software product,
    # publication, or research project -- never a named product/paper.
    _WIKIDATA_CONTRADICTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "SoftwareProduct": ("software", "computer program", "mobile app", "web application", "software application"),
        "Publication": ("journal article", "research paper", "academic paper", "scholarly article", "conference paper"),
        "ResearchStudy": ("research project", "scientific study", "research study"),
    }

    # External-grounding POSITIVE corroboration vocabulary --
    # generic terms a Wikidata short description would use for a genuine
    # data resource. Only ever used to restore Dataset from an already-
    # abstained Concept state, never to override a DIFFERENT type context
    # already independently assigned.
    _WIKIDATA_DATASET_CORROBORATION_KEYWORDS: Tuple[str, ...] = ("dataset", "database", "corpus")

    # A bullet/list-item line, or a blank line, is structurally part of
    # the same list block as its neighbours -- a general document-
    # structure pattern (works for any markdown/plain-text bulleted list,
    # not a domain-specific rule).
    _LIST_ITEM_LINE_RE = re.compile(r"^\s*(?:[*\-+]|\d+[.)])\s+")

    # URLs/paths routinely contain incidental keyword-shaped segments
    # (e.g. ".../compas-analysis", ".../data-and-analysis") that are
    # identifier slugs, not natural-language signal -- strip them before
    # any keyword matching so a hyperlink never masquerades as prose.
    _URL_RE = re.compile(r"https?://\S+")

    # Entity_type values (after entity_normalizer canonicalises
    # casing/slash/space variants) that fall within the population/group
    # family and are therefore in scope for the entity-vs-finding decision.
    _POPULATION_GROUP_FAMILY_TYPES: Set[str] = {"PopulationGroup", "DemographicGroup"}

    # General closed English grammatical classes (auxiliary/modal verbs,
    # comparison markers) -- not protected-characteristic or domain
    # vocabulary. Presence of either is a strong, portable signal that a
    # label is clause/finding-shaped (a fact or comparison ABOUT a group)
    # rather than a bare noun phrase naming the group itself.
    _POPULATION_GROUP_AUX_MODAL_VERBS: Set[str] = {
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had",
        "do", "does", "did",
        "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    }
    _POPULATION_GROUP_COMPARATIVE_MARKERS: Set[str] = {
        "more", "less", "fewer", "greater", "higher", "lower", "worse", "better",
        "than", "compared", "versus", "vs",
    }

    # Entity_type values (after entity_normalizer canonicalises
    # casing/variant forms) that fall within the AISystem family and are
    # therefore in scope for the entity-vs-process/framework/organisation
    # decision.
    _AISYSTEM_FAMILY_TYPES: Set[str] = {"AISystem", "Model", "System"}

    # General closed legal-entity-suffix class (not AI/domain vocabulary --
    # the same general closed set already reused for fuzzy-merge grouping
    # in utils/canonical_linker.py's ORG_SUFFIXES) -- a label ending in one
    # of these is a strong, portable, structural institution signal.
    _AISYSTEM_ORG_SUFFIXES: Set[str] = {"inc", "ltd", "corp", "company", "co", "llc", "gmbh", "plc"}

    # General closed English indefinite-pronoun/common-noun class whose
    # members happen to end in the letters "ing" without being gerunds
    # (cf. "the"/"a"/"an" article-stripping elsewhere in this file) -- not
    # AI/domain vocabulary, just a guard against the gerund-suffix check
    # misfiring on ordinary function words.
    _AISYSTEM_NON_GERUND_ING_WORDS: Set[str] = {
        "something", "anything", "nothing", "everything",
        "morning", "evening", "spring", "king", "ring", "thing", "things",
    }

    # Entity_type-agnostic generic-predicate values in scope for
    # refinement.
    _GENERIC_PREDICATE_FAMILY: Set[str] = {"is", "has", "relatedTo"}

    # General document-structure pattern (not lexical/domain-specific): a
    # short "Label:" lead followed by genuine descriptive prose (not a bare
    # URL) -- the structural shape repeatedly justifying hasDescription
    # when auditing "is"-predicate relations.
    _FIELD_LABEL_DESCRIPTION_RE = re.compile(r"^[^:\n]{2,80}:\s*(?!https?://)(\S.{14,})$")

    def _dataset_family_context_window(self, entity: Entity, discourse_text: str) -> str:
        start = self._coerce_nonneg_int(getattr(entity, "start_char", None))
        if start is None:
            start = self._coerce_nonneg_int((getattr(entity, "metadata", None) or {}).get("start_char"))
        end = self._coerce_nonneg_int(getattr(entity, "end_char", None))
        if end is None:
            end = self._coerce_nonneg_int((getattr(entity, "metadata", None) or {}).get("end_char"))
        text = str(discourse_text or "")
        if start is None or end is None or start < 0 or end <= start or end > len(text):
            return ""
        window_start = max(0, start - 160)
        window_end = min(len(text), end + 160)
        # Do not cross a line break: source records are frequently
        # formatted as bullet-point "Field: value" lists (a general
        # document-structure pattern, not domain-specific), where each
        # line is a distinct, unrelated field -- letting the window bleed
        # into a neighbouring bullet lets an unrelated field's wording
        # (e.g. a separate "Case Study" reference two lines down)
        # masquerade as context about the entity's own line.
        prev_break = text.rfind("\n", 0, start)
        if prev_break != -1 and prev_break >= window_start:
            window_start = prev_break + 1
        next_break = text.find("\n", end)
        if next_break != -1 and next_break <= window_end:
            window_end = next_break
        window = text[window_start:window_end]
        return self._URL_RE.sub(" ", window)

    def _dataset_family_block_context(self, entity: Entity, discourse_text: str) -> str:
        """The structurally-linked list block surrounding the
        entity's own line -- consecutive bullet/list-item lines (or blank
        lines) extending outward from the entity's line, stopping at the
        first non-list, non-blank line, and capped so it cannot grow
        unbounded. General document-structure signal (works for any
        markdown/plain-text bulleted list), used ONLY for the narrower,
        more specific "quantity-attached positive content" check below --
        never for the broader category-ambiguity check, which stays
        confined to the entity's own line to avoid the exact neighbouring-
        bullet contamination that check exists to prevent."""
        start = self._coerce_nonneg_int(getattr(entity, "start_char", None))
        if start is None:
            start = self._coerce_nonneg_int((getattr(entity, "metadata", None) or {}).get("start_char"))
        end = self._coerce_nonneg_int(getattr(entity, "end_char", None))
        if end is None:
            end = self._coerce_nonneg_int((getattr(entity, "metadata", None) or {}).get("end_char"))
        text = str(discourse_text or "")
        if start is None or end is None or start < 0 or end <= start or end > len(text):
            return ""

        lines = text.split("\n")
        offsets = []
        pos = 0
        for ln in lines:
            offsets.append((pos, pos + len(ln)))
            pos += len(ln) + 1

        own_idx = None
        for i, (line_start, line_end) in enumerate(offsets):
            if line_start <= start <= line_end:
                own_idx = i
                break
        if own_idx is None:
            return ""

        def _is_list_or_blank(line: str) -> bool:
            stripped = line.strip()
            return not stripped or bool(self._LIST_ITEM_LINE_RE.match(line))

        lo = own_idx
        while lo > 0 and _is_list_or_blank(lines[lo - 1]) and (own_idx - lo) < 6:
            lo -= 1
        hi = own_idx
        while hi < len(lines) - 1 and _is_list_or_blank(lines[hi + 1]) and (hi - own_idx) < 6:
            hi += 1

        block = "\n".join(lines[lo:hi + 1])[:1200]
        return self._URL_RE.sub(" ", block)

    @staticmethod
    def _keyword_present(text: str, keyword: str) -> bool:
        """Word-boundary phrase match (not naive substring containment),
        so a compound like "software application" never accidentally
        matches inside an unrelated word/phrase such as "housing
        applications"."""
        pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
        return re.search(pattern, text) is not None

    def _classify_dataset_family_deterministic(
        self, *, label: str, context_window: str
    ) -> Tuple[Optional[str], str]:
        """Cheap, confident, general-grammar PRE-filter only --
        never a full decision by itself. Returns (type, reason) for a
        confident deterministic decision (a structural label-lead signal,
        or a single unambiguous same-line category keyword), or
        (None, "needs_llm_classification") to escalate to the
        DatasetFamilyTypeClassifier for anything this filter cannot
        confidently resolve.

        Unlike the broader disambiguation logic, this filter never itself
        confirms "stays Dataset" from bare keyword presence (e.g.
        "records"/"data" nearby) or from wider-block quantity proximity --
        an earlier audit found that produced false positives when an unrelated
        neighbouring entity (a repository/toolkit) shared a list block
        with a genuine dataset's record count. Only an explicit
        structural label-lead match, an unambiguous same-line category
        keyword, or the LLM classifier's own contextual judgement can now
        confirm or restore Dataset."""
        norm_label = " ".join(str(label or "").strip().split())
        if not norm_label:
            return None, "needs_llm_classification"

        if self._QUANTITY_LEAD_RE.match(norm_label):
            return "Concept", "quantity_or_statistic_like_label"

        lead_tokens = [t.strip(".,:;") for t in re.split(r"\s+", norm_label.lower())]
        # A leading English article is a closed-class function word, not
        # domain vocabulary -- strip it so "A study of..."/"The analysis
        # of..." are recognised the same as a bare "Study of...".
        if lead_tokens and lead_tokens[0] in {"a", "an", "the"} and len(lead_tokens) > 1:
            lead_tokens = lead_tokens[1:]
        lead_token = lead_tokens[0] if lead_tokens else ""
        for candidate_type, leads in self._DATASET_FAMILY_LABEL_LEAD_MAP.items():
            if lead_token in leads:
                return candidate_type, f"label_structurally_denotes_{candidate_type.lower()}"

        norm_context = normalize_text(context_window) if context_window else ""
        if norm_context:
            hits: Dict[str, int] = {}
            for candidate_type, keywords in self._DATASET_FAMILY_CONTEXT_KEYWORDS.items():
                if any(self._keyword_present(norm_context, kw) for kw in keywords):
                    hits[candidate_type] = 1
            if len(hits) == 1:
                only_type = next(iter(hits))
                return only_type, "local_context_keyword_match"
            # len(hits) == 0 (no same-line category signal at all) or > 1
            # (genuinely conflicting same-line signals) both escalate to
            # the LLM classifier rather than guessing deterministically.

        return None, "needs_llm_classification"

    def _classify_population_group_deterministic(
        self, *, label: str
    ) -> Tuple[Optional[str], str]:
        """Cheap, confident PRE-filter only -- never a full
        decision by itself. A label is confidently NOT a genuine group only
        when it is clause/finding-shaped (carries a general-English
        auxiliary/modal verb, or a comparison/disparity marker) -- both are
        closed grammatical-function-word classes, never protected-
        characteristic or domain vocabulary, so this never special-cases
        any specific group. Everything else -- including a short, clean
        noun phrase that plausibly names a group, and a long noun phrase
        with no clause/comparison signal -- escalates to the
        PopulationGroupTypeClassifier LLM call rather than guessing
        deterministically which class of group (or non-group) it is."""
        norm_label = " ".join(str(label or "").strip().split())
        if not norm_label:
            return None, "needs_llm_classification"
        tokens = [t.strip(".,:;\"'").lower() for t in re.split(r"\s+", norm_label)]
        tokens = [t for t in tokens if t]
        if not tokens:
            return None, "needs_llm_classification"
        if any(t in self._POPULATION_GROUP_AUX_MODAL_VERBS for t in tokens):
            return "Outcome", "clause_shaped_not_a_group"
        if any(t in self._POPULATION_GROUP_COMPARATIVE_MARKERS for t in tokens):
            return "Outcome", "comparison_or_disparity_marker_present"
        return None, "needs_llm_classification"

    def _classify_ai_system_family_deterministic(
        self, *, label: str
    ) -> Tuple[Optional[str], str]:
        """Cheap, confident PRE-filter only -- never a full
        decision by itself. Resolves only two structurally obvious cases,
        both using closed general-grammar/legal-suffix classes rather than
        any AI/audit/framework vocabulary list:
        - a label ending in a general legal-entity suffix (Inc/Ltd/Corp/...)
          is confidently an Organization;
        - a label that is clause-shaped (a general auxiliary/modal verb
          present) or gerund-shaped (its first or last token is a
          sufficiently long -ing activity word, e.g. "Red Teaming",
          "Testing Procedures") confidently denotes an activity, not a
          system or model.
        Everything else -- including any mention actually naming a system,
        model, framework, or resource -- escalates to the
        AiSystemFamilyTypeClassifier LLM call rather than guessing which
        of the closed vocabulary's many remaining categories applies."""
        norm_label = " ".join(str(label or "").strip().split())
        if not norm_label:
            return None, "needs_llm_classification"
        tokens = [t.strip(".,:;\"'").lower() for t in re.split(r"\s+", norm_label)]
        tokens = [t for t in tokens if t]
        if not tokens:
            return None, "needs_llm_classification"
        last_token = tokens[-1].rstrip(".,")
        if last_token in self._AISYSTEM_ORG_SUFFIXES:
            return "Organization", "legal_entity_suffix_present"
        if any(t in self._POPULATION_GROUP_AUX_MODAL_VERBS for t in tokens):
            return "Process", "clause_shaped_activity_not_a_system"
        if any(
            t.endswith("ing") and len(t) > 4 and t not in self._AISYSTEM_NON_GERUND_ING_WORDS
            for t in (tokens[0], tokens[-1])
        ):
            return "Process", "gerund_shaped_activity_label"
        return None, "needs_llm_classification"

    def _predicate_domain_range_plausible(
        self, predicate: str, relation: Relation, entities_by_uri: Dict[str, Entity]
    ) -> bool:
        """Use the EXISTING predicate
        domain/range registry purely as a VALIDATION gate on an already-
        proposed predicate -- never to invent one. The SUBJECT side always
        uses the existing entity-family domain check unchanged. The OBJECT
        side branches on whether the target is a literal: an entity object
        still uses the pre-existing entity-family range check unchanged;
        a literal object instead checks the predicate's explicit literal-
        object permission (PredicateRegistryEntry.literal_allowed/
        literal_datatypes) -- a literal never automatically satisfies (or
        automatically fails) range_families, since it has no meaningful
        entity family of its own."""
        entry = PREDICATE_REGISTRY.get(predicate)
        if entry is None:
            return False
        policy = getattr(self, "semantic_policy", None)
        if policy is None:
            return False
        src = entities_by_uri.get(str(getattr(relation, "source", "") or ""))
        src_family = policy.entity_family(src)
        if not policy._family_matches(src_family, entry.domain_families):
            return False

        if bool(getattr(relation, "target_is_literal", False)):
            meta = getattr(relation, "metadata", None) or {}
            object_type = str(getattr(relation, "object_type", None) or meta.get("object_type") or "")
            target_datatype = str(getattr(relation, "target_datatype", None) or "")
            return literal_datatype_permitted(entry, object_type, target_datatype)

        tgt = entities_by_uri.get(str(getattr(relation, "target", "") or ""))
        tgt_family = policy.entity_family(tgt)
        return policy._family_matches(tgt_family, entry.range_families)

    def _refine_generic_predicate_deterministic(
        self, relation: Relation, entities_by_uri: Dict[str, Entity]
    ) -> Tuple[Optional[str], str]:
        """Cheap, confident PRE-filter only -- never a full
        decision by itself. Resolves only two structurally obvious cases:
        - the relation's own ORIGINALLY-CAPTURED raw predicate text
          (metadata["predicate_text"]) already resolves, via the existing
          general predicate-canonicalisation table, to a DIFFERENT,
          specific predicate than the generic one it was collapsed to --
          reusing that already-established mapping is zero-risk IF the
          proposed predicate's domain/range is plausible for this
          relation's actual subject/object (validation, not invention).
          This is the "specific_predicate_available" case.
        - the relation is "is" with a literal object whose evidence
          matches a general "Label: descriptive text" structural pattern
          (not lexical -- any bulleted/colon-separated field-description
          shape in any domain) -- confidently hasDescription.
        Everything else escalates to the PredicateRefinementClassifier LLM
        call rather than guessing which specific relation is entailed."""
        pred = str(getattr(relation, "relation", "") or "").strip()
        if pred not in self._GENERIC_PREDICATE_FAMILY:
            return None, "not_applicable"
        meta = getattr(relation, "metadata", None) or {}

        predicate_text = str(meta.get("predicate_text") or "").strip()
        if predicate_text:
            resolved = normalize_predicate_strict(predicate_text)
            if resolved and resolved != pred and resolved not in self._GENERIC_PREDICATE_FAMILY:
                if self._predicate_domain_range_plausible(resolved, relation, entities_by_uri):
                    return resolved, "predicate_text_reresolved_domain_range_valid"

        if pred in {"is", "has"} and bool(getattr(relation, "target_is_literal", False)):
            # The same "Label: descriptive text"
            # structural shape surfaces under either raw generic copula
            # ("is"/"has"), depending only on which generic verb the
            # upstream extractor happened to pick for a given run -- the
            # structural signal is identical regardless of which one it was.
            evidence = str(meta.get("evidence_quote") or meta.get("evidence") or "")
            if self._FIELD_LABEL_DESCRIPTION_RE.match(evidence.strip()):
                return "hasDescription", "field_description_bullet_pattern"

        return None, "needs_llm_classification"

    def _refine_generic_predicates_for_relations(
        self, relations: List[Relation], entities_by_uri: Dict[str, Entity], discourse_text: str
    ) -> None:
        """Entity-vs-finding-style predicate refinement, applied
        to already-ACCEPTED relations whose final predicate is exactly
        is/has/relatedTo. Mutates relation objects in place (mirrors how
        canonical_linker's type-promotion mutates Entity objects
        in place after acceptance) -- never changes which relations exist,
        their source/target/evidence/provenance, only the predicate/
        ontology_uri/mapping_status fields."""
        for relation in relations:
            pred = str(getattr(relation, "relation", "") or "").strip()
            if pred not in self._GENERIC_PREDICATE_FAMILY:
                continue
            meta = getattr(relation, "metadata", None) or {}
            if not isinstance(meta, dict):
                meta = {}
                relation.metadata = meta

            new_pred, reason = self._refine_generic_predicate_deterministic(relation, entities_by_uri)
            mapping_status = "mapped"
            if new_pred is None and reason == "needs_llm_classification":
                classifier = getattr(self, "predicate_refinement_classifier", None)
                if classifier is not None:
                    subject_entity = entities_by_uri.get(str(getattr(relation, "source", "") or ""))
                    object_entity = None
                    if not bool(getattr(relation, "target_is_literal", False)):
                        object_entity = entities_by_uri.get(str(getattr(relation, "target", "") or ""))
                    subject_label = str(getattr(subject_entity, "label", "") or meta.get("subject_text") or "")
                    subject_type = str(getattr(subject_entity, "entity_type", "") or meta.get("subject_type") or "")
                    is_literal = bool(getattr(relation, "target_is_literal", False))
                    object_value = (
                        str(getattr(relation, "target_literal", None) or getattr(relation, "target", "") or "")
                        if is_literal
                        else str(getattr(object_entity, "label", "") or meta.get("object_text") or "")
                    )
                    object_type = (
                        str(meta.get("object_type") or "")
                        if is_literal
                        else str(getattr(object_entity, "entity_type", "") or meta.get("object_type") or "")
                    )
                    context_window = self._dataset_family_context_window(
                        subject_entity, discourse_text
                    ) if subject_entity is not None else discourse_text
                    result = classifier.classify(
                        subject=subject_label,
                        subject_type=subject_type,
                        object_value=object_value,
                        object_type=object_type,
                        is_literal_object=is_literal,
                        original_predicate=pred,
                        sentence=context_window or discourse_text,
                        evidence=str(meta.get("evidence_quote") or meta.get("evidence") or ""),
                    )
                    decision = result.get("decision")
                    confidence = result.get("confidence")
                    reasoning_category = result.get("reasoning_category")
                    if decision == "upgrade" and confidence == "confident":
                        candidate = str(result.get("final_predicate") or "").strip()
                        if candidate in PREDICATE_REGISTRY and self._predicate_domain_range_plausible(
                            candidate, relation, entities_by_uri
                        ):
                            new_pred = candidate
                            reason = f"llm_upgrade:{reasoning_category}"
                        else:
                            new_pred = None
                            reason = "llm_upgrade_rejected_domain_range_invalid"
                    elif decision == "local_specific" and confidence == "confident":
                        candidate = str(result.get("final_predicate") or "").strip()
                        candidate = re.sub(r"[^A-Za-z0-9]+", " ", candidate).strip()
                        pieces = [p for p in candidate.split() if p]
                        if pieces:
                            new_pred = pieces[0][:1].lower() + pieces[0][1:] + "".join(p.capitalize() for p in pieces[1:])
                            reason = f"llm_local_specific:{reasoning_category}"
                            mapping_status = "local_unmapped"
                        else:
                            new_pred = None
                            reason = "llm_local_specific_empty_predicate"
                    else:
                        new_pred = None
                        reason = f"llm_keep_generic:{reasoning_category}" if reasoning_category else "llm_keep_generic"
                else:
                    reason = "no_confident_signal_abstained_no_classifier"
            elif new_pred is not None and reason == "predicate_text_reresolved_domain_range_valid":
                mapping_status = "mapped"
            elif new_pred is not None and reason == "field_description_bullet_pattern":
                mapping_status = "local_unmapped"

            if new_pred is not None and new_pred != pred:
                meta["generic_predicate_original"] = pred
                meta["predicate_refinement_reason"] = reason
                relation.relation = new_pred
                relation.predicate_mapping_status = mapping_status
                meta["predicate_mapping_status"] = mapping_status
            elif reason:
                meta["predicate_refinement_reason"] = reason
            relation.metadata = meta

    def _dataset_family_wikidata_contradiction(
        self, *, entity: Entity, wikidata_description: Optional[str]
    ) -> Optional[Tuple[str, str]]:
        """External-grounding consistency check: a successful
        Wikidata match is corroborating-or-contradicting evidence for an
        entity that entered this run typed Dataset/Corpus/DataSource --
        whether it is still typed that way, or was already abstained to
        the generic Concept fallback by the context-aware pass above
        (leaving the ambiguity for exactly this kind of corroborating
        evidence to resolve). Never touches any entity that did not
        originate as Dataset-family. Reads the grounded external
        description generically -- no named entity is ever special-cased."""
        current_type = str(getattr(entity, "entity_type", "") or "")
        original_type = str((getattr(entity, "metadata", None) or {}).get("dataset_family_original_type") or "")
        if current_type not in self._DATASET_FAMILY_TYPES and original_type not in self._DATASET_FAMILY_TYPES:
            return None
        desc = normalize_text(str(wikidata_description or ""))
        if not desc:
            return None
        # Positive corroboration -- only ever restores Dataset
        # from an already-abstained Concept (ambiguous) state; never
        # overrides a DIFFERENT type the context pass already
        # independently, confidently assigned.
        if current_type == "Concept" and any(
            self._keyword_present(desc, kw) for kw in self._WIKIDATA_DATASET_CORROBORATION_KEYWORDS
        ):
            return "Dataset", "wikidata_description_corroborates_dataset"
        # Contradiction may only redirect an entity that is
        # STILL literally proposed as Dataset -- external grounding must
        # never override a DIFFERENT, already-confident classification
        # (Analysis/ResearchStudy/Publication/SoftwareProduct/Resource)
        # the deterministic pre-filter or the LLM classifier already made
        # from genuine source context.
        if current_type != "Dataset":
            return None
        for candidate_type, keywords in self._WIKIDATA_CONTRADICTION_KEYWORDS.items():
            if candidate_type == current_type:
                continue
            if any(self._keyword_present(desc, kw) for kw in keywords):
                return candidate_type, "wikidata_description_contradicts_dataset"
        return None

    def _reground_and_remint_entity(self, e: Entity, *, scope_key: Optional[str]) -> Entity:
        """Re-run ontology mapping + URI reminting after an entity_type
        change made after the normal per-entity grounding pass already
        ran (the post-Wikidata contradiction correction above), so type/
        ontology_uri/uri never drift apart from each other."""
        mapped = self.ontology_mapper.map_entity(e) if self.config.USE_ONTOLOGY_MAPPING else None
        chosen = None
        if isinstance(mapped, str) and mapped.strip():
            chosen = mapped.strip()
        elif isinstance(mapped, list):
            for m in mapped:
                if isinstance(m, str) and m.strip():
                    chosen = m.strip()
                    break
        if isinstance(mapped, list) and mapped:
            e.metadata["ontology_classes_curie"] = [str(x).strip() for x in mapped if str(x).strip()]
        elif isinstance(mapped, str) and mapped.strip():
            e.metadata["ontology_classes_curie"] = [mapped.strip()]
        e.ontology_uri = expand_curie(chosen) if chosen else None
        if e.ontology_uri:
            e.metadata["ontology_classes"] = [e.ontology_uri]
        else:
            e.metadata.pop("ontology_classes", None)

        self._ensure_entity_ontology_grounding(e)

        reminted_uri = self.entity_normalizer.canonical_uri(
            label=e.label,
            entity_type=e.entity_type,
            ontology_uri=e.ontology_uri or (e.metadata or {}).get("ontology_uri"),
            discourse_scope=scope_key,
        )
        if reminted_uri != e.uri:
            old_uri = e.uri
            e.uri = reminted_uri
            e = self.global_registry.get_or_create(e)
            if old_uri != reminted_uri and hasattr(self.global_registry, "replace_entity"):
                self.global_registry.replace_entity(old_uri, reminted_uri)
        return e

    def _process_entities_batch(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> List[List[Entity]]:
        """
        Process entities for multiple texts in batch.
        Uses:
            - coreference resolution
            - semantic chunking
            - LLM entity extraction per chunk
            - registry deduplication
            - ontology enrichment
            - metadata propagation
        """
        all_entities: List[List[Entity]] = []
    
        for text, metadata in zip(texts, metadata_list):

            # -------------------------------------------------------
            # 1. Resolve coreference
            # -------------------------------------------------------
            resolved_text = self.coref_resolver.resolve(text)

            # -------------------------------------------------------
            # 2. Semantic chunking
            # -------------------------------------------------------
            chunks = SemanticChunker.chunk(resolved_text)
            section_spans = SemanticChunker.parse_structured_sections(
                resolved_text,
                field=str(metadata.get("field") or "").strip(),
            )
            discourse_id = metadata.get("discourse_id")
            origin_key = metadata.get("origin_key") or self._compose_origin_key(discourse_id, metadata.get("field"))
            chunk_spans = self._build_chunk_spans(
                resolved_text,
                chunks,
                discourse_id,
                origin_key=str(origin_key) if origin_key is not None else None,
            )
            chunk_span_by_idx: Dict[int, Dict[str, int]] = {
                int(span["chunk_index"]): span for span in chunk_spans if "chunk_index" in span
            }

            # -------------------------------------------------------
            # 3. Extract entities FROM CHUNKS (only ONCE)
            # -------------------------------------------------------
            entities: List[Entity] = []

            doc_id = f"{metadata.get('discourse_id')}|{metadata.get('field', 'unknown')}"
            discourse_id_raw = metadata.get("discourse_id")
            discourse_id = str(discourse_id_raw).strip() if discourse_id_raw is not None else None
            
            for idx, chunk in enumerate(chunks):
                extracted = self.entity_extractor.extract(chunk, doc_id=doc_id, context=resolved_text)

                if extracted:
                    for e in extracted:
                        if not isinstance(getattr(e, "metadata", None), dict):
                            e.metadata = {}

                        e.metadata["chunk_index"] = idx
                        e.metadata["chunk_total"] = len(chunks)
                        if discourse_id:
                            e.metadata["discourse_id"] = discourse_id
                        if origin_key:
                            e.metadata["origin_key"] = str(origin_key)

                        # Prefer extractor-provided span values, then model fields.
                        local_start = self._coerce_nonneg_int(e.metadata.get("start_char"))
                        local_end = self._coerce_nonneg_int(e.metadata.get("end_char"))
                        if local_start is None and getattr(e, "start_char", None) is not None:
                            local_start = self._coerce_nonneg_int(getattr(e, "start_char", None))
                        if local_end is None and getattr(e, "end_char", None) is not None:
                            local_end = self._coerce_nonneg_int(getattr(e, "end_char", None))

                        span = chunk_span_by_idx.get(idx)
                        if span is not None:
                            e.metadata["start_offset"] = int(span.get("start_offset", 0))
                            e.metadata["end_offset"] = int(span.get("end_offset", 0))

                            # Offset local chunk spans by deterministic chunk start.
                            chunk_start = self._coerce_nonneg_int(span.get("start_offset"))
                            abs_start = None
                            abs_end = None
                            if local_start is not None and chunk_start is not None:
                                abs_start = int(local_start + chunk_start)
                            if local_end is not None and chunk_start is not None:
                                abs_end = int(local_end + chunk_start)
                            if abs_start is not None:
                                e.metadata["start_char"] = abs_start
                                e.start_char = abs_start
                            else:
                                e.metadata.setdefault("start_char", -1)
                            if abs_end is not None:
                                e.metadata["end_char"] = abs_end
                                e.end_char = abs_end
                            else:
                                e.metadata.setdefault("end_char", -1)

                            if abs_start is not None and abs_end is not None:
                                # Register a mention for EVERY literal occurrence of this
                                # entity's surface form within the chunk (metadata["spans"],
                                # chunk-local offsets from the extractor), not just the
                                # single primary span -- otherwise a later semantic unit
                                # containing a repeated mention of the same entity is
                                # incorrectly treated as not having it locally available.
                                # _append_entity_mention already dedupes by
                                # (discourse_id, chunk_index, start_char, end_char), so this
                                # is safe to call once per occurrence without creating
                                # duplicate mention records or duplicate entity nodes.
                                local_spans = e.metadata.get("spans")
                                if not isinstance(local_spans, list) or not local_spans:
                                    local_spans = [(local_start, local_end)]
                                for local_s, local_e in local_spans:
                                    coerced_s = self._coerce_nonneg_int(local_s)
                                    coerced_e = self._coerce_nonneg_int(local_e)
                                    if coerced_s is None or coerced_e is None or coerced_e <= coerced_s:
                                        continue
                                    self._append_entity_mention(
                                        e,
                                        discourse_id=discourse_id,
                                        chunk_index=idx,
                                        start_char=int(coerced_s + chunk_start),
                                        end_char=int(coerced_e + chunk_start),
                                        field=str(metadata.get("field") or "").strip(),
                                    )
                                self._annotate_entity_section_metadata(
                                    e,
                                    section_spans=section_spans,
                                )
                        else:
                            e.metadata.setdefault("start_char", -1)
                            e.metadata.setdefault("end_char", -1)

                    entities.extend(extracted) 

            self._augment_lexical_entities(
                text=resolved_text,
                metadata=metadata,
                entities=entities,
                section_spans=section_spans,
                chunk_spans=chunk_spans,
            )

            # -------------------------------------------------------
            # 4. Normalize + merge with global registry
            # -------------------------------------------------------
            normalized_entities: List[Entity] = []

            for e in entities:
              
                # Skip malformed outputs from LLM extractor
                if not isinstance(e, Entity):
                    continue

                # Backwards compatibility for any older code paths
                if not hasattr(e, "name"):
                    e.name = e.label
                if not hasattr(e, "type"):
                    e.type = e.entity_type

                # ðŸ”¹ NEW: normalize label before registry & enrichment
                #e.label = self.entity_normalizer.normalize_label(e.label)
                #e.name = e.label  # keep name in sync with label

                # Normalize entity before registry + enrichment
                self.entity_normalizer.normalize_entity(e)

                # Context-aware Dataset-family
                # disambiguation -- runs only on entities currently
                # proposed as Dataset (or a Dataset-synonym type), using
                # the entity's own label PLUS its local source context,
                # before any URI/ontology grounding locks that type in.
                # Never touches any other type family. A cheap
                # deterministic pre-filter resolves confident cases (a
                # structural label-lead signal, or a single unambiguous
                # same-line category keyword); anything it cannot
                # confidently resolve escalates to the narrowly-scoped
                # DatasetFamilyTypeClassifier LLM call, which
                # judges genuine grammatical/semantic role using the full
                # sentence plus structurally-linked context -- never a
                # blind "quantity/keyword present nearby" rule.
                if e.entity_type in self._DATASET_FAMILY_TYPES:
                    context_window = self._dataset_family_context_window(e, resolved_text)
                    new_type, reason = self._classify_dataset_family_deterministic(
                        label=e.label, context_window=context_window,
                    )
                    if new_type is None and reason == "needs_llm_classification":
                        classifier = getattr(self, "dataset_family_classifier", None)
                        if classifier is not None:
                            block_context = self._dataset_family_block_context(e, resolved_text)
                            result = classifier.classify(
                                surface=e.label,
                                sentence=context_window,
                                linked_context=block_context,
                                proposed_type=e.entity_type,
                            )
                            label_out = str(result.get("label") or "Ambiguous")
                            confidence = str(result.get("confidence") or "ambiguous")
                            if label_out == "Ambiguous" or confidence == "ambiguous":
                                new_type = "Concept"
                                reason = "llm_classification_ambiguous"
                            else:
                                new_type = label_out
                                reason = "llm_classification_confident"
                        else:
                            # No classifier available (e.g. a pipeline
                            # built without __init__, the established
                            # test-harness pattern in this codebase) --
                            # conservative fallback: abstain rather than
                            # guess or silently keep Dataset.
                            new_type = "Concept"
                            reason = "no_confident_signal_abstained_no_classifier"
                    if new_type is not None and new_type != e.entity_type:
                        e.metadata["dataset_family_original_type"] = e.entity_type
                        e.metadata["dataset_family_typing_reason"] = reason
                        e.entity_type = new_type
                        if hasattr(e, "type"):
                            e.type = new_type
                    elif reason:
                        e.metadata["dataset_family_typing_reason"] = reason

                # General entity-vs-literal decision -- applies to
                # every candidate entity regardless of entity_type. A scalar
                # value (percentage/count/currency amount/multiplier/ratio)
                # should not become a graph Entity merely because it appears
                # as a noun phrase. Deliberately runs AFTER the Dataset-family
                # block above (not before it): that block's own quantity-lead
                # guard needs to see the entity's ORIGINAL, unmodified label
                # -- rewriting the label first would silently defeat it for
                # any Dataset-proposed quantity-lead phrase (the label would
                # no longer start with a digit by the time it got there).
                # Structural/morphological only -- no dataset IDs, entity
                # names, or domain vocabulary; applies identically to any
                # future dataset/domain.
                quantity_decision = classify_quantity_candidate(e.label)
                if quantity_decision is not None and quantity_decision.is_pure_quantity:
                    # No genuine referent remains once the quantity
                    # expression is removed -- do not mint a node for a
                    # bare scalar value.
                    continue
                if quantity_decision is not None and quantity_decision.is_decomposable and quantity_decision.residual_label:
                    e.metadata["quantity_literal_original_label"] = e.label
                    e.metadata["quantity_literal_value"] = quantity_decision.literal_value
                    e.metadata["quantity_literal_datatype"] = quantity_decision.datatype
                    e.metadata["quantity_literal_object_type"] = quantity_decision.object_type
                    if quantity_decision.qualifiers:
                        e.metadata["quantity_literal_qualifiers"] = quantity_decision.qualifiers
                    e.metadata["quantity_literal_reason"] = quantity_decision.reason
                    e.label = quantity_decision.residual_label
                    e.name = e.label

                # Entity-vs-finding decision for the population/
                # group family -- runs AFTER the quantity/entity
                # decomposition above (so a percentage/count prefix has
                # already been stripped to its residual referent) and only
                # engages entities already canonically typed PopulationGroup
                # or DemographicGroup by this point. A cheap deterministic
                # pre-filter resolves confidently clause/comparison-shaped
                # labels (general closed grammatical-function-word classes
                # only -- never protected-characteristic or domain
                # vocabulary); anything it cannot confidently resolve
                # escalates to the narrowly-scoped PopulationGroupTypeClassifier
                # LLM call, which judges the mention's genuine grammatical/
                # semantic role using the full sentence context.
                if e.entity_type in self._POPULATION_GROUP_FAMILY_TYPES:
                    new_pg_type, pg_reason = self._classify_population_group_deterministic(label=e.label)
                    if new_pg_type is None and pg_reason == "needs_llm_classification":
                        pg_classifier = getattr(self, "population_group_classifier", None)
                        if pg_classifier is not None:
                            pg_context = self._dataset_family_context_window(e, resolved_text)
                            pg_result = pg_classifier.classify(
                                surface=e.label,
                                sentence=pg_context,
                                proposed_type=e.entity_type,
                            )
                            pg_label_out = str(pg_result.get("label") or "Ambiguous")
                            pg_confidence = str(pg_result.get("confidence") or "ambiguous")
                            if pg_label_out == "Ambiguous" or pg_confidence == "ambiguous":
                                new_pg_type = "Concept"
                                pg_reason = "llm_classification_ambiguous"
                            else:
                                new_pg_type = pg_label_out
                                pg_reason = "llm_classification_confident"
                        else:
                            # No classifier available (e.g. a pipeline built
                            # without __init__, the established test-harness
                            # pattern in this codebase) -- conservative
                            # fallback: abstain rather than guess or
                            # silently keep a possibly-wrong group type.
                            new_pg_type = "Concept"
                            pg_reason = "no_confident_signal_abstained_no_classifier"
                    if new_pg_type is not None and new_pg_type != e.entity_type:
                        e.metadata["population_group_original_type"] = e.entity_type
                        e.metadata["population_group_typing_reason"] = pg_reason
                        e.entity_type = new_pg_type
                        if hasattr(e, "type"):
                            e.type = new_pg_type
                    elif pg_reason:
                        e.metadata["population_group_typing_reason"] = pg_reason

                # Entity-vs-process/framework/organisation
                # decision for the AISystem family -- only engages entities
                # already canonically typed AISystem/Model/System. A cheap
                # deterministic pre-filter resolves confidently clause/
                # gerund-shaped activity labels and legal-entity-suffix
                # organisation labels (general closed grammar/legal-suffix
                # classes only -- never AI/audit/framework vocabulary);
                # anything it cannot confidently resolve escalates to the
                # narrowly-scoped AiSystemFamilyTypeClassifier LLM call.
                if e.entity_type in self._AISYSTEM_FAMILY_TYPES:
                    new_ai_type, ai_reason = self._classify_ai_system_family_deterministic(label=e.label)
                    if new_ai_type is None and ai_reason == "needs_llm_classification":
                        ai_classifier = getattr(self, "ai_system_family_classifier", None)
                        if ai_classifier is not None:
                            ai_context = self._dataset_family_context_window(e, resolved_text)
                            ai_result = ai_classifier.classify(
                                surface=e.label,
                                sentence=ai_context,
                                proposed_type=e.entity_type,
                            )
                            ai_label_out = str(ai_result.get("label") or "Ambiguous")
                            ai_confidence = str(ai_result.get("confidence") or "ambiguous")
                            if ai_label_out == "Ambiguous" or ai_confidence == "ambiguous":
                                new_ai_type = "Concept"
                                ai_reason = "llm_classification_ambiguous"
                            else:
                                new_ai_type = ai_label_out
                                ai_reason = "llm_classification_confident"
                        else:
                            # No classifier available (e.g. a pipeline built
                            # without __init__, the established test-harness
                            # pattern in this codebase) -- conservative
                            # fallback: abstain rather than guess or
                            # silently keep a possibly-wrong system/model type.
                            new_ai_type = "Concept"
                            ai_reason = "no_confident_signal_abstained_no_classifier"
                    if new_ai_type is not None and new_ai_type != e.entity_type:
                        e.metadata["ai_system_family_original_type"] = e.entity_type
                        e.metadata["ai_system_family_typing_reason"] = ai_reason
                        e.entity_type = new_ai_type
                        if hasattr(e, "type"):
                            e.type = new_ai_type
                    elif ai_reason:
                        e.metadata["ai_system_family_typing_reason"] = ai_reason

                # An entity whose own label is still referential
                # -shaped (reusing the unresolved-endpoint guard's exact
                # structural detector -- no new/duplicate lexical rule) has
                # not been resolved to any
                # genuine antecedent at this point -- its identity must stay
                # discourse-local rather than merge globally with the same
                # surface text from an unrelated discourse. Scope by
                # origin_key (field + discourse_id) so the SAME phrase in
                # two different discourses never produces the same URI.
                if self._is_unresolved_coreference_candidate(e.label):
                    e.metadata["discourse_scoped_identity"] = True
                scope_key = str(origin_key or discourse_id or "") if e.metadata.get("discourse_scoped_identity") else None

                # Deterministic canonical URI (type-aware + alias-normalized)
                # Use any ontology hint already present in metadata for stability.
                ontology_hint = getattr(e, "ontology_uri", None) or (e.metadata or {}).get("ontology_uri")
                e.uri = self.entity_normalizer.canonical_uri(
                    label=e.label,
                    entity_type=e.entity_type,
                    ontology_uri=ontology_hint,
                    discourse_scope=scope_key,
                )

                # Deduplicate across chunks & documents
                e = self.global_registry.get_or_create(e)

                # Attach top-level metadata (discourse_id, theme_id, etc.)
                for k, v in metadata.items():
                    if v is not None:
                      e.metadata[k] = v


                # -------------------------------------------------------
                # Ontology mapping (production: ONE expanded IRI in ontology_uri)
                # -------------------------------------------------------
                mapped = None
                if self.config.USE_ONTOLOGY_MAPPING:
                    mapped = self.ontology_mapper.map_entity(e)

                    chosen = None
                    if isinstance(mapped, str) and mapped.strip():
                        chosen = mapped.strip()
                    elif isinstance(mapped, list):
                        for m in mapped:
                            if isinstance(m, str) and m.strip():
                                chosen = m.strip()
                                break

                    # store full mapping for traceability
                    if isinstance(mapped, list) and mapped:
                        e.metadata["ontology_classes_curie"] = [str(x).strip() for x in mapped if str(x).strip()]
                    elif isinstance(mapped, str) and mapped.strip():
                        e.metadata["ontology_classes_curie"] = [mapped.strip()]

                    e.ontology_uri = expand_curie(chosen) if chosen else None

                # --- final guard: ontology_uri must NEVER be list/stringified list ---
                ou = getattr(e, "ontology_uri", None)
                if isinstance(ou, list) and ou:
                    ou = str(ou[0])
                if isinstance(ou, str) and ou.strip().startswith("[") and ou.strip().endswith("]"):
                    try:
                        parsed = ast.literal_eval(ou)
                        if isinstance(parsed, list) and parsed:
                            ou = str(parsed[0])
                    except Exception:
                        pass

                e.ontology_uri = expand_curie(ou.strip()) if isinstance(ou, str) and ou.strip() else None

                # Add ontology alignment to metadata for metrics/serializer
                if e.ontology_uri:
                    e.metadata["ontology_classes"] = [e.ontology_uri]
                else:
                    e.metadata.pop("ontology_classes", None)

                # Coverage guard: always keep one ontology grounding.
                self._ensure_entity_ontology_grounding(e)

                # If ontology mapping changed canonical identity, remint URI deterministically
                # and keep registry references consistent through redirect.
                # Preserve the same discourse scope used at initial minting above.
                reminted_uri = self.entity_normalizer.canonical_uri(
                    label=e.label,
                    entity_type=e.entity_type,
                    ontology_uri=e.ontology_uri or (e.metadata or {}).get("ontology_uri"),
                    discourse_scope=scope_key,
                )
                if reminted_uri != e.uri:
                    old_uri = e.uri
                    e.uri = reminted_uri
                    e = self.global_registry.get_or_create(e)
                    if old_uri != reminted_uri and hasattr(self.global_registry, "replace_entity"):
                        self.global_registry.replace_entity(old_uri, reminted_uri)

                # Wikidata linking -- never send an unresolved referential
                # mention's raw surface text to external entity linking.
                if (
                    getattr(self.config, "USE_WIKIDATA_LINKING", False)
                    and self.wikidata_linker
                    and not e.metadata.get("discourse_scoped_identity")
                ):
                    # Prefer the genuine source mention over e.label: entity_normalizer
                    # rewrites .label to a normalized/canonical form (see
                    # EntityNormalizer.normalize_entity), while metadata["surface_form"]
                    # is set at entity-creation time (entity_extractor / semantic
                    # assertion resolution / structural-node registration) and is never
                    # overwritten by normalization -- confirmed present for every one of
                    # the 17 CSV-flagged mismatches when re-inspected against the
                    # production TTL. Falls back to e.label only if surface_form is
                    # genuinely absent.
                    wikidata_query = str(e.metadata.get("surface_form") or e.label or "").strip() or e.label
                    # Prefer explicit, externally-grounded ontology class(es)
                    # over the bare local type for the publication-type-
                    # compatibility signal (see WikidataLinker._is_publication_type) --
                    # already computed above via ontology mapping / coverage
                    # guard, so no new mapping work is introduced here.
                    entity_ontology_types = e.metadata.get("ontology_classes") or (
                        [e.ontology_uri] if getattr(e, "ontology_uri", None) else None
                    )
                    # ONLY genuinely-observed source text is passed as
                    # trusted identity evidence: the verbatim surface_form,
                    # plus metadata["source_grounded_aliases"] (an explicit
                    # allow-list written only by EntityNormalizer -- for a
                    # real extracted mention, never an LLM resolved_label --
                    # and _attach_local_surface_aliases). The full
                    # metadata["aliases"] collection (which also carries
                    # inferred/consolidated forms) is NOT passed here.
                    _sg_source = e.metadata.get("source_grounded_aliases")
                    _sg_aliases = list(_sg_source) if isinstance(_sg_source, list) else []
                    _sf = str(e.metadata.get("surface_form") or "").strip()
                    if _sf and _sf not in _sg_aliases:
                        _sg_aliases.insert(0, _sf)
                    wd = self.wikidata_linker.link(
                        wikidata_query,
                        entity_type=e.entity_type,
                        context_text=self._dataset_family_context_window(e, resolved_text),
                        ontology_types=entity_ontology_types,
                        source_grounded_aliases=_sg_aliases or None,
                    )
                    if wd and "qid" in wd:
                        qid = wd["qid"]
                        # 1) For quality_metrics (linking_rate)
                        e.metadata["wikidata_uri"] = f"http://www.wikidata.org/entity/{qid}"

                        # 2) For RDF serializer (_add_wikidata_metadata expects 'wikidata'* keys)
                        e.metadata["wikidata"] = wd
                        e.metadata["wikidataLabel"] = wd.get("label")
                        e.metadata["wikidataDescription"] = wd.get("description")
                        e.metadata["wikidataScore"] = wd.get("score")

                        # 3) Optional: keep your original debug blob if you like
                        e.metadata["wikidata_match"] = wd

                        # External-ID safety (design instruction 4): flat,
                        # easy-to-read provenance for any downstream
                        # consumer that treats a Wikidata QID as
                        # same-referent evidence -- this is a HEURISTIC API
                        # link, not an asserted identifier. Consolidation
                        # logic is not changed in this pass.
                        e.metadata["wikidata_identity_evidence"] = wd.get("identity_evidence")
                        e.metadata["wikidata_authoritative"] = bool(wd.get("authoritative", False))

                        # External-grounding consistency check --
                        # a successful Wikidata match is corroborating or
                        # CONTRADICTING evidence for an entity that entered
                        # this run Dataset-family, never an unconditional
                        # override. Only acts on entities that originated
                        # as Dataset/Corpus/DataSource (whether still typed
                        # that way or already abstained to Concept above);
                        # any other type family is left untouched.
                        contradiction = self._dataset_family_wikidata_contradiction(
                            entity=e,
                            wikidata_description=wd.get("description"),
                        )
                        if contradiction is not None:
                            new_type, reason = contradiction
                            e.metadata.setdefault("dataset_family_original_type", e.entity_type)
                            e.metadata["dataset_family_wikidata_consistency_reason"] = reason
                            e.entity_type = new_type
                            if hasattr(e, "type"):
                                e.type = new_type
                            e = self._reground_and_remint_entity(e, scope_key=scope_key)

                normalized_entities.append(e)

            # after the loop
            all_entities.append(normalized_entities)

        return all_entities

    # ---------------------------------------------------------
    # Relation processing
    # ---------------------------------------------------------

    def _process_relations_batch(
        self,
        texts: List[str],
        entities_list: List[List[Entity]],
        discourse_ids: List[str],
        provenance_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[List[Relation]]:
        """Process relations for multiple texts in batch.

        Design choice:
        - Text is stored as a literal on the DiscourseUnit via metadata["text"]
        - No Text node, no hasText triple
        - DiscourseUnit URIs are stable (derived from real discourse_ids)
        """
        resolved_texts = [self.coref_resolver.resolve(text) for text in texts]

        print("ðŸ§  Using LLM-based relation extraction...")
        all_relations: List[List[Relation]] = []

        # ------------------------------------------------------------
        # Ensure a DiscourseUnit node exists for each text (typed and queryable),
        # store raw text as a literal via metadata["text"],
        # and standardise provenance key to "discourse_id".
        # ------------------------------------------------------------
        for idx, text in enumerate(texts):
            relations: List[Relation] = []
            text = texts[idx]
            resolved_text = resolved_texts[idx] if idx < len(resolved_texts) else text
            uri_to_entity: Dict[str, Entity] = {}

            for ent in entities_list[idx]:
                u = getattr(ent, "uri", None)
                if u:
                    uri_to_entity[u] = ent

            # Use stable discourse_id from upstream
            did = discourse_ids[idx] if idx < len(discourse_ids) else None
            if did is None:
                did = f"doc_{idx}"
            did = str(did).strip()
            if not did:
                did = f"doc_{idx}"

            prov_meta = (
                provenance_list[idx]
                if provenance_list is not None and idx < len(provenance_list) and isinstance(provenance_list[idx], dict)
                else {}
            )
            field = prov_meta.get("field") if isinstance(prov_meta, dict) else None
            origin_key = (
                str(prov_meta.get("origin_key")).strip()
                if isinstance(prov_meta, dict) and prov_meta.get("origin_key") is not None
                else self._compose_origin_key(did, field)
            )

            chunk_spans = self._get_chunk_spans_for_discourse(did, origin_key=origin_key)
            if not chunk_spans:
                chunk_spans = self._build_chunk_spans(
                    resolved_text,
                    SemanticChunker.chunk(resolved_text),
                    did,
                    origin_key=origin_key,
                )

            # Stable URI for the discourse container node
            # (prefixing avoids accidental clashes with other entity IDs)
            doc_uri = make_uri(f"discourse_{make_uri_safe(origin_key or did)}")

            # Create a DiscourseUnit entity for the document context
            doc_entity = Entity(
                uri=doc_uri,
                label=f"DiscourseUnit {did}",
                entity_type="DiscourseUnit",
                confidence=1.0,               
                metadata={
                    "discourse_id": did,
                    "origin_key": origin_key,
                    "field": field,
                    "surface_form": "discourse unit",
                    "context_source": "lightweight",
                    "text": text,
                },    
            )

            # Some of your pipeline code expects these aliases:
            if not hasattr(doc_entity, "name") or doc_entity.name is None:
                doc_entity.name = doc_entity.label
            if not hasattr(doc_entity, "type") or doc_entity.type is None:
                doc_entity.type = doc_entity.entity_type
            self._ensure_entity_ontology_grounding(doc_entity)

            # Register the DiscourseUnit entity so it gets rdf:type in TTL
            if hasattr(self.global_registry, "get_or_create"):
                doc_entity = self.global_registry.get_or_create(doc_entity)
                self._ensure_entity_ontology_grounding(doc_entity)
                doc_uri = str(doc_entity.uri)
            elif hasattr(self.global_registry, "add_entity"):
                self.global_registry.add_entity(doc_entity)
            elif hasattr(self.global_registry, "add"):
                self.global_registry.add(doc_entity)
            else:
                if hasattr(self.global_registry, "entities"):
                    self.global_registry.entities[doc_uri] = doc_entity

            semantic_units = SemanticChunker.semantic_relation_units(
                resolved_text,
                field=str(field or "").strip(),
            )
            # Populated as each unit finishes processing below:
            # relations already extracted for a given semantic unit, keyed by
            # unit_id -- lets a LATER unit's propositional-deixis resolution
            # inspect the immediately preceding unit's own assertions (e.g.
            # to detect that it expressed more than one distinct claim)
            # without re-deriving anything.
            unit_relations_by_id: Dict[str, List[Relation]] = {}
            if self._field_allows_semantic_extraction(field):
                for unit_index, unit in enumerate(semantic_units):
                    unit_text = str(unit.get("text") or "").strip()
                    section_context = str(unit.get("section_text") or "").strip()
                    support_context = self._compose_semantic_support_context(
                        unit_text=unit_text,
                        unit=unit,
                        section_text=section_context,
                        discourse_text=resolved_text,
                    )
                    diag_row: Dict[str, Any] = {
                        "source_field": str(field or "").strip(),
                        "source_id": did,
                        "local_unit_id": str(unit.get("unit_id") or ""),
                        "local_unit_type": str(unit.get("unit_type") or ""),
                        "local_sentence": unit_text,
                        "predicate_cues_detected": [],
                        "extracted_local_entities": [],
                        "candidate_pairs_considered": [],
                        "llm_relations_returned": [],
                        "lexical_relations_returned": [],
                        "pre_policy_rejection_reason": "",
                        "sent_to_policy": False,
                    }
                    if str(unit.get("section_category") or "").strip() != "semantic":
                        diag_row["pre_policy_rejection_reason"] = "non_semantic_section"
                        self._record_candidate_generation_diag(diag_row)
                        continue
                    local_entities = self._local_entities_for_unit(entities_list[idx], unit, did)
                    diag_row["extracted_local_entities"] = [self._entity_debug_row(entity) for entity in local_entities]
                    unit_start = self._coerce_nonneg_int(unit.get("start_offset")) or 0
                    unit_end = self._coerce_nonneg_int(unit.get("end_offset")) or unit_start + len(unit_text)
                    if local_entities:
                        self._attach_local_surface_aliases(
                            entities=local_entities,
                            discourse_text=resolved_text,
                            discourse_id=did,
                            unit_start=unit_start,
                            unit_end=unit_end,
                        )
                    candidate_pairs = self._candidate_pairs_for_unit(
                        unit_text=unit_text,
                        entities=local_entities,
                    )
                    diag_row["candidate_pairs_considered"] = candidate_pairs
                    diag_row["predicate_cues_detected"] = sorted(
                        {
                            str(pair.get("cue") or "").strip()
                            for pair in candidate_pairs
                            if str(pair.get("cue") or "").strip()
                        }
                    )

                    # Only when this unit has NO locally-mentioned system/process
                    # actor at all, offer a single unambiguous actor established
                    # earlier in the same field as an additional entity hint (not
                    # a literal local mention). candidate_pairs above is left
                    # untouched -- it stays cue/text-matched to genuinely local
                    # entities only.
                    carry_forward_entity = None
                    extraction_entities = local_entities
                    if not any(self._entity_semantic_family(e) == "system" for e in local_entities):
                        carry_forward_entity = self._carry_forward_actor_for_unit(
                            entities=entities_list[idx], unit=unit, discourse_id=did,
                        )
                        if carry_forward_entity is not None:
                            hinted = copy.copy(carry_forward_entity)
                            hinted.metadata = dict(carry_forward_entity.metadata or {})
                            hinted.metadata["carried_forward_from_field_context"] = True
                            extraction_entities = local_entities + [hinted]

                    extracted_relations = self._extract_relations_with_pair_focus(
                        unit_text=unit_text,
                        local_entities=extraction_entities,
                        discourse_id=did,
                        candidate_pairs=candidate_pairs,
                        context=resolved_text,
                    )
                    # Deterministic endpoint-resolution step, run before
                    # semantic relation validation: resolve referential
                    # subject/object endpoints ("this", "it", "them", "this
                    # tool", ...) to an unambiguous same-discourse entity
                    # where entity extraction/mentions already make that
                    # possible, BEFORE _resolve_semantic_unit_relations would
                    # otherwise fabricate a brand-new entity from the raw
                    # pronoun text. Entities it resolves to are appended to
                    # extraction_entities so the URI lookup below finds them.
                    extracted_relations, anaphora_resolved_entities = self._resolve_anaphoric_relation_endpoints(
                        relations=extracted_relations,
                        entities=entities_list[idx],
                        unit=unit,
                        discourse_id=did,
                        unit_index=unit_index,
                        semantic_units=semantic_units,
                    )
                    if anaphora_resolved_entities:
                        existing_uris = {str(getattr(e, "uri", "") or "") for e in extraction_entities}
                        extraction_entities = extraction_entities + [
                            e for e in anaphora_resolved_entities if str(getattr(e, "uri", "") or "") not in existing_uris
                        ]
                    # Propositional/discourse-deictic endpoint
                    # resolution -- only engages on endpoints the nominal
                    # step above left unresolved (still non-URI text), and
                    # only for the closed bare-pronoun class compatible
                    # with proposition-level reference. Also runs BEFORE
                    # _resolve_semantic_unit_relations to intercept
                    # fabrication, exactly like the nominal step.
                    extracted_relations, propositional_resolved_entities = self._resolve_propositional_relation_endpoints(
                        relations=extracted_relations,
                        unit=unit,
                        unit_index=unit_index,
                        semantic_units=semantic_units,
                        unit_relations_by_id=unit_relations_by_id,
                        discourse_id=did,
                        origin_key=origin_key,
                        field=str(field or "").strip(),
                    )
                    if propositional_resolved_entities:
                        existing_uris = {str(getattr(e, "uri", "") or "") for e in extraction_entities}
                        extraction_entities = extraction_entities + [
                            e for e in propositional_resolved_entities if str(getattr(e, "uri", "") or "") not in existing_uris
                        ]
                    extracted_relations = self._resolve_semantic_unit_relations(
                        relations=extracted_relations,
                        local_entities=extraction_entities,
                        discourse_text=resolved_text,
                        discourse_id=did,
                        field=str(field or "").strip(),
                        origin_key=origin_key,
                        unit=unit,
                        unit_text=unit_text,
                    )
                    # Possessive-reference endpoint semantics. Runs
                    # AFTER subject/object resolution above (so the
                    # relation's own subject is a guaranteed resolved URI,
                    # needed for the reflexive-subject fallback) and never
                    # touches the occurrence-level nominal-pronoun resolution
                    # above.
                    extracted_relations = self._apply_possessive_endpoint_semantics(
                        relations=extracted_relations,
                        entities=entities_list[idx],
                        local_pool_entities=extraction_entities,
                        unit=unit,
                        discourse_id=did,
                    )
                    if carry_forward_entity is not None:
                        carried_uri = str(getattr(carry_forward_entity, "uri", "") or "")
                        carried_label = str(getattr(carry_forward_entity, "label", "") or "")
                        kept_relations = []
                        for relation in extracted_relations:
                            if str(getattr(relation, "source", "") or "") != carried_uri:
                                kept_relations.append(relation)
                                continue
                            rel_meta = getattr(relation, "metadata", None) or {}
                            evidence_anchor = str(
                                rel_meta.get("evidence_quote") or rel_meta.get("evidence") or ""
                            ).strip()
                            # The model may claim this relation belongs to the
                            # carried-forward actor while actually grounding it in
                            # Context (e.g. reusing a sentence from elsewhere in the
                            # field) instead of this unit's own Text -- reject rather
                            # than accept evidence that isn't from the target chunk.
                            if not evidence_anchor or evidence_anchor not in unit_text:
                                continue
                            # Widen ONLY the validation-time local context (never the
                            # provenance evidence/evidence_quote, which stay anchored
                            # to the target chunk) so the carried-forward actor's
                            # established name and this unit's own claim can be seen
                            # co-occurring in one local proposition, exactly as an
                            # explicit local mention would be -- without touching
                            # semantic_relation_policy.py itself.
                            widened_context = f"{carried_label}, {unit_text}"
                            rel_meta["local_context"] = widened_context
                            rel_meta["subject_resolved_from_field_context"] = True
                            relation.metadata = rel_meta
                            # validate() reads relation.context before
                            # metadata["local_context"] (see semantic_relation_policy
                            # .validate()) -- set both, and the unconditional
                            # `rel.context = support_context` below is guarded to
                            # skip relations already tagged here.
                            relation.context = widened_context
                            kept_relations.append(relation)
                        extracted_relations = kept_relations

                        # Deterministic fallback for passive/agent-elliptical local
                        # units: extraction sometimes still attributes the relation
                        # to a local "group"-family entity (a demographic/population
                        # -type entity -- generically the affected party, not a
                        # plausible agent, per _label_has_group_cue/_entity_semantic
                        # _family) instead of using the carried-forward actor hint --
                        # this choice was found to be non-deterministic across
                        # otherwise-identical LLM calls, so it cannot be relied on
                        # via prompting alone. Only apply when Text names no OTHER
                        # locally-available entity of an agent-plausible family (i.e.
                        # no explicit competing actor), and only when the group
                        # entity's own literal target survives as evidence text --
                        # the group entity itself becomes the object/participant.
                        no_competing_local_agent = not any(
                            self._entity_semantic_family(e) in {"system", "policy", "organisation", "person"}
                            for e in local_entities
                        )
                        if no_competing_local_agent:
                            corrected_relations = []
                            for relation in extracted_relations:
                                rel_source_uri = str(getattr(relation, "source", "") or "")
                                # _resolve_semantic_unit_relations resolves unmatched
                                # subject_text against extraction_entities and appends
                                # any newly-created surface entity to that list (not
                                # local_entities) -- look the subject up there so a
                                # dynamically-resolved local entity (e.g. one only
                                # named in Text, not extracted up front) is found too.
                                subject_entity = next(
                                    (e for e in extraction_entities if str(getattr(e, "uri", "") or "") == rel_source_uri),
                                    None,
                                )
                                if (
                                    subject_entity is None
                                    or self._entity_semantic_family(subject_entity) != "group"
                                    or not getattr(relation, "target_is_literal", False)
                                ):
                                    corrected_relations.append(relation)
                                    continue
                                rel_meta = getattr(relation, "metadata", None) or {}
                                evidence_anchor = str(
                                    rel_meta.get("evidence_quote") or rel_meta.get("evidence") or ""
                                ).strip()
                                if not evidence_anchor or evidence_anchor not in unit_text:
                                    corrected_relations.append(relation)
                                    continue
                                relation.source = carried_uri
                                # The group entity moves from
                                # subject to object here -- the relation's
                                # PREVIOUS target was literal-valued text
                                # describing something else entirely, so
                                # that stale target_literal/target_datatype
                                # must not survive onto the new Entity
                                # target (this is what caused themes/12 and
                                # themes/13's stale-literal inconsistency).
                                relation.set_target_entity(str(getattr(subject_entity, "uri", "") or ""))
                                # Root-cause fix (bounded correction pass,
                                # item 1): subject_text/object_text are
                                # raw LLM output captured BEFORE this
                                # swap and describe the PRE-swap roles
                                # (old subject_text names the group that
                                # is now the object; old object_text was
                                # the literal that is now discarded).
                                # Left untouched, they survive as stale
                                # text that the RDF serializer's
                                # endpoint/surface compatibility check
                                # later reads as ground truth for the
                                # POST-swap roles, causing a false
                                # "entity vs surface form" mismatch (or,
                                # worse, silently passing due to the
                                # check's own ordering gap). Refresh both
                                # to describe the entities in their new
                                # roles instead of leaving pre-swap text.
                                old_subject_text = str(rel_meta.get("subject_text") or "").strip()
                                rel_meta["subject_text"] = carried_label
                                if old_subject_text:
                                    rel_meta["object_text"] = old_subject_text
                                rel_meta.pop("subject_surface_form", None)
                                rel_meta.pop("object_surface_form", None)
                                widened_context = f"{carried_label}, {unit_text}"
                                rel_meta["local_context"] = widened_context
                                rel_meta["subject_resolved_from_field_context"] = True
                                relation.metadata = rel_meta
                                relation.context = widened_context
                                corrected_relations.append(relation)
                            extracted_relations = corrected_relations
                    lexical_relations = self._lexical_semantic_relations_for_unit(
                        unit_text=support_context,
                        entities=local_entities,
                        discourse_id=did,
                        field=str(field or "").strip(),
                        origin_key=origin_key,
                        unit=unit,
                        llm_relations=extracted_relations,
                    )
                    entity_index = {
                        str(getattr(entity, "uri", "") or "").strip(): entity
                        for entity in extraction_entities
                        if str(getattr(entity, "uri", "") or "").strip()
                    }
                    diag_row["llm_relations_returned"] = [
                        self._relation_debug_row(relation, entity_index)
                        for relation in extracted_relations
                    ]
                    diag_row["lexical_relations_returned"] = [
                        self._relation_debug_row(relation, entity_index)
                        for relation in lexical_relations
                    ]
                    extracted_relations.extend(lexical_relations)
                    extracted_relations, coverage_meta = self._recover_semantic_coverage_for_unit(
                        relations=extracted_relations,
                        unit_text=unit_text,
                        support_context=support_context,
                        local_entities=local_entities,
                        discourse_text=resolved_text,
                        discourse_id=did,
                        field=str(field or "").strip(),
                        origin_key=origin_key,
                        unit=unit,
                    )
                    diag_row["coverage_gaps_detected"] = int(coverage_meta.get("coverage_gaps_detected", 0))
                    diag_row["coverage_gaps_recovered"] = int(coverage_meta.get("coverage_gaps_recovered", 0))
                    diag_row["unresolved_coverage_gaps"] = int(coverage_meta.get("unresolved_coverage_gaps", 0))
                    unit_relations_by_id[str(unit.get("unit_id") or "")] = list(extracted_relations)
                    if not extracted_relations:
                        diag_row["pre_policy_rejection_reason"] = (
                            "no_relations_generated_without_pair_hints"
                            if not candidate_pairs
                            else "no_relations_generated"
                        )
                        self._record_candidate_generation_diag(diag_row)
                        continue
                    diag_row["sent_to_policy"] = True
                    self._record_candidate_generation_diag(diag_row)
                    unit_offset = unit_start
                    for rel in extracted_relations:
                        meta = getattr(rel, "metadata", None) or {}
                        for key in ("char_start", "char_end", "start_offset", "end_offset"):
                            val = self._coerce_int(meta.get(key))
                            if val is not None and val >= 0:
                                meta[key] = int(val + unit_offset)
                        meta.setdefault("discourse_id", did)
                        meta.setdefault("field", str(field or "").strip())
                        meta.setdefault("source_field", str(field or "").strip())
                        meta.setdefault("origin_key", origin_key)
                        meta.setdefault("source_doc_id", str(did))
                        meta["section"] = str(unit.get("section") or "")
                        meta["section_category"] = str(unit.get("section_category") or "")
                        meta["semantic_unit_id"] = str(unit.get("unit_id") or "")
                        meta["semantic_unit_type"] = str(unit.get("unit_type") or "")
                        if str(field or "").strip() == "question.body":
                            meta.setdefault("discourse_mode", "interrogative")
                            meta.setdefault("epistemic_status", "questioned")
                        meta["relation_kind"] = "extracted_semantic"
                        if not meta.get("subject_resolved_from_field_context"):
                            rel.context = support_context
                        meta.setdefault("local_context", support_context)
                        rel.metadata = meta
                        self._get_semantic_policy().apply_relation_metadata(
                            rel,
                            field=str(field or "").strip(),
                            evidence_text=unit_text,
                        )
                        self._ensure_relation_chunk_provenance(
                            rel,
                            fallback_discourse_id=did,
                            chunk_spans=chunk_spans,
                        )
                        relations.append(rel)

            # Ensure every extracted relation has strict chunk provenance.
            for rel in relations:
                meta = getattr(rel, "metadata", None) or {}
                if origin_key:
                    meta.setdefault("origin_key", origin_key)
                span = self._extract_relation_span_from_metadata(meta)
                if span is None:
                    derived = self._derive_relation_span_from_entities(rel, uri_to_entity)
                    if derived is not None:
                        meta["char_start"] = int(derived[0])
                        meta["char_end"] = int(derived[1])
                        meta["start_offset"] = int(derived[0])
                        meta["end_offset"] = int(derived[1])
                rel.metadata = meta
                self._ensure_relation_chunk_provenance(
                    rel,
                    fallback_discourse_id=did,
                    chunk_spans=chunk_spans,
                )

            all_relations.append(relations)


        # Backward compatibility at extraction stage.
        for relations in all_relations:
            for relation in relations:
                if "discourse_id" not in relation.metadata and "doc_id" in relation.metadata:
                    relation.metadata["discourse_id"] = relation.metadata.get("doc_id")

        return all_relations

    # ---------------------------------------------------------
    # Core structural node registration (Theme/Question/Answer)
    # ---------------------------------------------------------
    def _register_structural_node(
        self,
        *,
        uri: str,
        label: str,
        entity_type: str,
        discourse_id: str,
        context_source: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Ensure structural nodes exist in registry so they serialize into TTL."""
        meta = {
            "discourse_id": str(discourse_id),
            "surface_form": str(label),
            "context_source": context_source,
            "source": "structural",
            "source_doc_id": str(discourse_id),
            "chunk_index": -1,
            "chunk_index_unknown": True,
        }
        if extra_meta:
            meta.update(extra_meta)
        if "origin_key" not in meta:
            meta["origin_key"] = self._compose_origin_key(meta.get("discourse_id"), meta.get("field"))

        node = Entity(
            uri=uri,
            label=label,
            entity_type=entity_type,
            confidence=1.0,
            metadata=meta,
        )

        # compatibility aliases used elsewhere in your pipeline
        if not hasattr(node, "name") or node.name is None:
            node.name = node.label
        if not hasattr(node, "type") or node.type is None:
            node.type = node.entity_type
        self._ensure_entity_ontology_grounding(node)
        self.entity_normalizer.normalize_entity(node)
        ontology_hint = getattr(node, "ontology_uri", None) or meta.get("ontology_uri")
        node.uri = self.entity_normalizer.canonical_uri(
            label=node.label,
            entity_type=node.entity_type,
            ontology_uri=ontology_hint,
        )

        # register (same pattern you use for DiscourseUnit) :contentReference[oaicite:1]{index=1}
        if hasattr(self.global_registry, "get_or_create"):
            node = self.global_registry.get_or_create(node)
            self._ensure_entity_ontology_grounding(node)
        elif hasattr(self.global_registry, "add_entity"):
            self.global_registry.add_entity(node)
        elif hasattr(self.global_registry, "add"):
            self.global_registry.add(node)
        else:
            if hasattr(self.global_registry, "entities"):
                self.global_registry.entities[uri] = node
        return node



    # ---------------------------------------------------------
    # Themes
    # ---------------------------------------------------------
    def process_themes_batch(self, themes: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple themes in batch."""
        texts = []
        metadata_list = []
        
        for theme in themes:
            texts.append(theme.get('description', ''))
            metadata_list.append({
                "domain": theme.get("domain_category", ""),
                "discourse_id": theme.get("id"),
                "field": "theme.description",
                "origin_key": self._compose_origin_key(theme.get("id"), "theme.description"),
                "theme_id": theme.get("id"),
                "expert_id": theme.get("expert_id"),
                "question_id": None,
                "answer_id": None,
                "context_source": "dataset_theme",
            })


        entities_list = self._process_entities_batch(texts, metadata_list)

        discourse_ids = [m.get("discourse_id") for m in metadata_list]
        relations_list = self._process_relations_batch(
            texts,
            entities_list,
            discourse_ids,
            provenance_list=metadata_list,
        )
        
        # Attach discourse_id to each relation for this batch
        for i, theme in enumerate(themes):
            theme_id = theme.get('id')
            expert_id = theme.get('expert_id')

            # âœ… FIX: register Theme node so it serialises (prevents dangling theme_<id>)
            theme_uri = f"{self.config.NAMESPACES['etd']}theme_{theme_id}"
            theme_node = self._register_structural_node(
                uri=theme_uri,
                label=f"Theme {theme_id}",
                entity_type="Theme",
                discourse_id=str(theme_id),
                context_source="dataset_theme",
                extra_meta={
                    "theme_id": theme_id,
                    "expert_id": expert_id,
                    "field": "theme",
                    "domain": theme.get("domain_category", ""),
                    "problem_category": theme.get("problem_category", ""),
                },
            )

            for rel in relations_list[i]:
                rel.metadata.setdefault("discourse_id", theme_id)
                rel.metadata.setdefault("field", "theme.description")
                rel.metadata.setdefault("source_entity_uri", str(theme_node.uri))
                rel.metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(theme_id, "theme.description"),
                )
                self._ensure_relation_chunk_provenance(rel, fallback_discourse_id=theme_id)

            if expert_id:
                e_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
                e_node = self._register_structural_node(
                    uri=e_uri,
                    label=f"Expert {expert_id}",
                    entity_type="Expert",
                    discourse_id=str(theme_id),
                    context_source="dataset_theme",
                    extra_meta={
                        "expert_id": expert_id,
                        "field": "expert",
                    },
                )
                rel_uri = _make_rel_uri(f"theme_{theme_id}_expert_{expert_id}")
                relations_list[i].append(make_relation(
                    str(theme_node.uri),
                    "createdBy",
                    str(e_node.uri),
                    discourse_id=str(theme_id),
                    extractor="structural",
                    context_source="dataset_theme",
                    confidence=0.96,
                    evidence=_mk_evidence_text(
                        text=None,
                        fallback=f"Structural link: theme_{theme_id} createdBy expert_{expert_id} (dataset_theme).",
                    ),
                ))
                relations_list[i][-1].uri = rel_uri
                relations_list[i][-1].metadata.setdefault("field", "theme.description")
                relations_list[i][-1].metadata.setdefault("source_entity_uri", str(theme_node.uri))
                relations_list[i][-1].metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(str(theme_id), "theme.description"),
                )
                relations_list[i][-1].metadata.setdefault("source_doc_id", str(theme_id))
                relations_list[i][-1].metadata["ontology_properties"] = ["prov:wasAttributedTo"]

        results: List[ProcessingResult] = []
        for i, theme in enumerate(themes):
            problem_category = theme.get('problem_category', '')
            domain = theme.get('domain_category', '')
            
            problem_ontologies = self.ontology_mapper.map_problem_category(problem_category)
            vair_domain = self.ontology_mapper.map_domain(domain)
            
            results.append(ProcessingResult(
                entities=entities_list[i],
                relations=relations_list[i],
                metadata={
                    'text': texts[i],
                    'text_length': len(texts[i]),
                    'num_entities': len(entities_list[i]),
                    'num_relations': len(relations_list[i]),
                    'problem_category': problem_category,
                    'problem_ontologies': problem_ontologies,
                    'domain_category': domain,
                    'vair_domain': vair_domain,
                    'model_category': theme.get('model_category', ''),
                    'extraction_method': 'enhanced_llm_ontology'
                }
            ))
        
        return results

    # ---------------------------------------------------------
    # Questions
    # ---------------------------------------------------------
    def process_questions_batch(self, questions: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple questions in batch."""
        texts = []
        metadata_list = []
        
        for question in questions:
            texts.append(question.get('body', question.get('title', '')))       
            metadata_list.append({
                "domain": None,
                "discourse_id": question.get("id"),
                "field": "question.body",
                "origin_key": self._compose_origin_key(question.get("id"), "question.body"),
                "theme_id": question.get("theme_id"),
                "question_id": question.get("id"),
                "expert_id": question.get("expert_id"),
                "answer_id": None,
                "context_source": "dataset_question",
            })

        
        entities_list = self._process_entities_batch(texts, metadata_list)
        
        discourse_ids = [m.get("discourse_id") for m in metadata_list]
        relations_list = self._process_relations_batch(
            texts,
            entities_list,
            discourse_ids,
            provenance_list=metadata_list,
        )

                
        #Attach discourse_id to each relation in this question
        for i, question in enumerate(questions):
            qid = question.get('id')
            for rel in relations_list[i]:
                rel.metadata.setdefault("discourse_id", qid)
                rel.metadata.setdefault("field", "question.body")
                rel.metadata.setdefault("discourse_mode", "interrogative")
                rel.metadata.setdefault("epistemic_status", "questioned")
                rel.metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(qid, "question.body"),
                )
                self._ensure_relation_chunk_provenance(rel, fallback_discourse_id=qid)

        results: List[ProcessingResult] = []
        for i, question in enumerate(questions):
            question_id = question.get('id')
            theme_id = question.get('theme_id')
            expert_id = question.get('expert_id')
            
            # âœ… FIX: register Question node so it serialises (prevents dangling question_<id>)
            q_uri = f"{self.config.NAMESPACES['etd']}question_{question_id}"
            q_node = self._register_structural_node(
                uri=q_uri,
                label=f"Question {question_id}",
                entity_type="Question",
                discourse_id=str(question_id),
                context_source="dataset_question",
                extra_meta={
                    "question_id": question_id,
                    "theme_id": theme_id,
                    "expert_id": expert_id,
                    "field": "question",
                },
            )         
            for rel in relations_list[i]:
                rel.metadata.setdefault("source_entity_uri", str(q_node.uri))
            
            if theme_id:
                # Structural relation: Question -> Theme
                # âœ… FIX: ensure Theme node exists before linking (prevents dangling theme_<id>)
                t_uri = f"{self.config.NAMESPACES['etd']}theme_{theme_id}"
                t_node = self._register_structural_node(
                    uri=t_uri,
                    label=f"Theme {theme_id}",
                    entity_type="Theme",
                    discourse_id=str(theme_id),
                    context_source="dataset_theme",
                    extra_meta={
                        "theme_id": theme_id,
                        "field": "theme",
                    },
                )
                
                source_uri = str(q_node.uri)
                target_uri = str(t_node.uri)
                rel_uri = _make_rel_uri(f"question_{question_id}_theme_{theme_id}")
                relations_list[i].append(make_relation(
                    source_uri,
                    "belongsToTheme",
                    target_uri,
                    discourse_id=str(question_id),
                    extractor="structural",
                    context_source="dataset_question",
                    confidence=0.98,
                    evidence=_mk_evidence_text(
                        text=None,
                        fallback=f"Structural link: question_{question_id} belongsToTheme theme_{theme_id} (dataset_question).",
                    ),
                ))
                relations_list[i][-1].uri = rel_uri
                relations_list[i][-1].metadata.setdefault("field", "question.body")
                relations_list[i][-1].metadata.setdefault("source_entity_uri", str(q_node.uri))
                relations_list[i][-1].metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(str(question_id), "question.body"),
                )
                relations_list[i][-1].metadata.setdefault("source_doc_id", str(question_id))
                
                # keep ontology mapping info
                relations_list[i][-1].metadata["ontology_properties"] = ["aipo:hasTheme"]

            if expert_id:
                e_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
                e_node = self._register_structural_node(
                    uri=e_uri,
                    label=f"Expert {expert_id}",
                    entity_type="Expert",
                    discourse_id=str(question_id),
                    context_source="dataset_question",
                    extra_meta={
                        "expert_id": expert_id,
                        "field": "expert",
                    },
                )

                rel_uri = _make_rel_uri(f"question_{question_id}_expert_{expert_id}")
                relations_list[i].append(make_relation(
                    str(q_node.uri),
                    "askedBy",
                    str(e_node.uri),
                    discourse_id=str(question_id),
                    extractor="structural",
                    context_source="dataset_question",
                    confidence=0.96,
                    evidence=_mk_evidence_text(
                        text=None,
                        fallback=f"Structural link: question_{question_id} askedBy expert_{expert_id} (dataset_question).",
                    ),
                ))
                relations_list[i][-1].uri = rel_uri
                relations_list[i][-1].metadata.setdefault("field", "question.body")
                relations_list[i][-1].metadata.setdefault("source_entity_uri", str(q_node.uri))
                relations_list[i][-1].metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(str(question_id), "question.body"),
                )
                relations_list[i][-1].metadata.setdefault("source_doc_id", str(question_id))
                relations_list[i][-1].metadata["ontology_properties"] = ["prov:wasAttributedTo"]

            question_type = self._classify_question_type(texts[i])
            has_ethical = any(e.entity_type == 'EthicalConcept' for e in entities_list[i])

            results.append(ProcessingResult(
                entities=entities_list[i],
                relations=relations_list[i],
                metadata={
                    'text': texts[i],
                    'text_length': len(texts[i]),
                    'num_entities': len(entities_list[i]),
                    'num_relations': len(relations_list[i]),
                    'question_type': question_type,
                    'has_ethical_dimension': has_ethical
                }
            ))
        
        return results

    # ---------------------------------------------------------
    # Answers
    # ---------------------------------------------------------
    def process_answers_batch(self, answers: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple answers in batch."""
        texts = []
        metadata_list = []
        
        for answer in answers:
            texts.append(answer.get('description', ''))
            metadata_list.append({
                "domain": None,
                "discourse_id": answer.get("id"),
                "field": "answer.description",
                "origin_key": self._compose_origin_key(answer.get("id"), "answer.description"),
                "theme_id": answer.get("theme_id"),
                "question_id": answer.get("question_id"),
                "answer_id": answer.get("id"),
                "context_source": "dataset_answer",
            })

        entities_list = self._process_entities_batch(texts, metadata_list)
        
        discourse_ids = [m.get("discourse_id") for m in metadata_list]
        relations_list = self._process_relations_batch(
            texts,
            entities_list,
            discourse_ids,
            provenance_list=metadata_list,
        )
        
        # Attach discourse_id to each relation in this answer
        for i, answer in enumerate(answers):
            aid = answer.get('id')
            for rel in relations_list[i]:
                rel.metadata.setdefault("discourse_id", aid)
                rel.metadata.setdefault("field", "answer.description")
                rel.metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(aid, "answer.description"),
                )
                self._ensure_relation_chunk_provenance(rel, fallback_discourse_id=aid)

        results: List[ProcessingResult] = []
        for i, answer in enumerate(answers):
            answer_id = answer.get('id')
            question_id = answer.get('question_id')
            expert_id = answer.get('expert_id')
            
            # âœ… FIX: register Answer node so it serialises (prevents dangling answer_<id>)
            a_uri = f"{self.config.NAMESPACES['etd']}answer_{answer_id}"
            a_node = self._register_structural_node(
                uri=a_uri,
                label=f"Answer {answer_id}",
                entity_type="Answer",
                discourse_id=str(answer_id),
                context_source="dataset_answer",
                extra_meta={
                    "answer_id": answer_id,
                    "question_id": question_id,
                    "theme_id": answer.get("theme_id"),
                    "field": "answer",
                },
            )
            for rel in relations_list[i]:
                rel.metadata.setdefault("source_entity_uri", str(a_node.uri))
                           
            if question_id:
                # âœ… FIX: ensure Question node exists before linking (prevents dangling question_<id>)
                q_uri = f"{self.config.NAMESPACES['etd']}question_{question_id}"
                q_node = self._register_structural_node(
                    uri=q_uri,
                    label=f"Question {question_id}",
                    entity_type="Question",
                    discourse_id=str(question_id),
                    context_source="dataset_question",
                    extra_meta={
                        "question_id": question_id,
                        "field": "question",
                    },
                )

                source_uri = str(a_node.uri)
                target_uri = str(q_node.uri)
                rel_uri = _make_rel_uri(f"answer_{answer_id}_question_{question_id}")
                relations_list[i].append(make_relation(
                    source_uri,
                    "answersQuestion",
                    target_uri,
                    discourse_id=str(answer_id),
                    extractor="structural",
                    context_source="dataset_answer",
                    confidence=0.98,
                    evidence=_mk_evidence_text(
                        text=None,
                        fallback=f"Structural link: answer_{answer_id} answersQuestion question_{question_id} (dataset_answer).",
                    ),
                ))
                relations_list[i][-1].uri = rel_uri
                relations_list[i][-1].metadata.setdefault("field", "answer.description")
                relations_list[i][-1].metadata.setdefault("source_entity_uri", str(a_node.uri))
                relations_list[i][-1].metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(str(answer_id), "answer.description"),
                )
                relations_list[i][-1].metadata.setdefault("source_doc_id", str(answer_id))
                relations_list[i][-1].metadata["ontology_properties"] = ["sioc:reply_of"]

            if expert_id:
                # âœ… FIX: ensure Expert node exists before linking (prevents dangling expert_<id>)
                e_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
                e_node = self._register_structural_node(
                    uri=e_uri,
                    label=f"Expert {expert_id}",
                    entity_type="Expert",
                    discourse_id=str(answer_id),  # discourse unit is the answer
                    context_source="dataset_answer",
                    extra_meta={
                        "expert_id": expert_id,
                        "field": "expert",
                    },
                )

                source_uri = str(a_node.uri)
                target_uri = str(e_node.uri)
                rel_uri = _make_rel_uri(f"answer_{answer_id}_expert_{expert_id}")
                relations_list[i].append(make_relation(
                    source_uri,
                    "authoredBy",
                    target_uri,
                    discourse_id=str(answer_id),
                    extractor="structural",
                    context_source="dataset_answer",
                    confidence=0.96,
                    evidence=_mk_evidence_text(
                        text=None,
                        fallback=f"Structural link: answer_{answer_id} authoredBy expert_{expert_id} (dataset_answer).",
                    ),
                ))
                relations_list[i][-1].uri = rel_uri
                relations_list[i][-1].metadata.setdefault("field", "answer.description")
                relations_list[i][-1].metadata.setdefault("source_entity_uri", str(a_node.uri))
                relations_list[i][-1].metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(str(answer_id), "answer.description"),
                )
                relations_list[i][-1].metadata.setdefault("source_doc_id", str(answer_id))
                relations_list[i][-1].metadata["ontology_properties"] = ["prov:wasAttributedTo"]


            results.append(ProcessingResult(
                entities=entities_list[i],
                relations=relations_list[i],
                metadata={
                    'text': texts[i],
                    'text_length': len(texts[i]),
                    'num_entities': len(entities_list[i]),
                    'num_relations': len(relations_list[i])
                }
            ))
        
        return results

    # ---------------------------------------------------------
    # Votes
    # ---------------------------------------------------------
    def process_vote(self, vote_dict: Dict[str, Any]) -> ProcessingResult:
        """Process a vote (expert endorsement)."""
        expert_id = vote_dict.get('expert_id')
        answer_id = vote_dict.get('answer_id')
        vote_raw = vote_dict.get('vote_value', None)

        # normalised text form (existing behaviour)
        vote_value = str(vote_raw).strip().lower() if vote_raw is not None else ''

        # raw integer form for RDF literal (new)
        vote_raw_int = None
        try:
            # handles 1, 0, -1, "1", "0", "-1"
            vote_raw_int = int(vote_raw)
        except (TypeError, ValueError):
            vote_raw_int = None
        
        vote_id = vote_dict.get('id')
        vote_node_id = vote_id if vote_id is not None else f"{expert_id}_{answer_id}"

        if vote_value in {'upvote', 'approve', 'endorse', 'like', 'agree', 'support', 'recommend', 'positive', '1', 'true', 'yes'}:
            vote_concept_local = "VoteValueSchemeUpvote"
        elif vote_value in {'downvote', 'disagree', 'reject', '0', 'oppose', 'criticize', 'negative', 'false', 'no'}:
            vote_concept_local = "VoteValueSchemeDownvote"
        elif vote_value in {'neutral', 'abstain', 'skip', 'maybe', 'undecided'}:
            vote_concept_local = "VoteValueSchemeNeutral"
        else:
            vote_concept_local = "VoteValueSchemeNeutral"

        vote_value_concept_uri = f"{self.config.NAMESPACES['etd']}{vote_concept_local}"

        # Create Expert entity
        expert_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
        expert_entity = Entity(
            uri=expert_uri,
            label=f"Expert {expert_id}",
            entity_type="Expert",
            confidence=0.95,
            metadata={
                'airo_type': 'Stakeholder',
                "discourse_id": str(vote_node_id),
                "field": "vote.vote_value",
                "surface_form": f"Expert {expert_id}",
                "context_source": "dataset_vote",
                "source": "structural",
                "source_doc_id": str(vote_node_id),
                "chunk_index": -1,
                "chunk_index_unknown": True,
                "origin_key": self._compose_origin_key(str(vote_node_id), "vote.vote_value"),
            }
        )
        # backwards compat
        expert_entity.name = expert_entity.label
        expert_entity.type = expert_entity.entity_type
        self._ensure_entity_ontology_grounding(expert_entity)

        expert_entity = self.global_registry.get_or_create(expert_entity)
        self._ensure_entity_ontology_grounding(expert_entity)

        # Create Answer entity
        answer_uri = f"{self.config.NAMESPACES['etd']}answer_{answer_id}"
        answer_entity = Entity(
            uri=answer_uri,
            label=f"Answer {answer_id}",
            entity_type="Answer",
            confidence=0.90,
            metadata={
                'airo_type': 'Document',
                "discourse_id": str(vote_node_id),
                "field": "vote.vote_value",
                "surface_form": f"Answer {answer_id}",
                "context_source": "dataset_vote",
                "source": "structural",
                "source_doc_id": str(vote_node_id),
                "chunk_index": -1,
                "chunk_index_unknown": True,
                "origin_key": self._compose_origin_key(str(vote_node_id), "vote.vote_value"),
            }
        )
        answer_entity.name = answer_entity.label
        answer_entity.type = answer_entity.entity_type
        self._ensure_entity_ontology_grounding(answer_entity)

        answer_entity = self.global_registry.get_or_create(answer_entity)
        self._ensure_entity_ontology_grounding(answer_entity)

        # Create Vote entity for structural backbone links
        vote_uri = f"{self.config.NAMESPACES['etd']}vote_{vote_node_id}"
        vote_entity = Entity(
            uri=vote_uri,
            label=f"Vote {vote_node_id}",
            entity_type="Vote",
            confidence=1.0,
            metadata={
                "id": vote_id,
                "vote_id": vote_id,
                "expert_id": expert_id,
                "answer_id": answer_id,
                "discourse_id": str(vote_node_id),
                "context_source": "dataset_vote",
                "source": "structural",
                "source_doc_id": str(vote_node_id),
                "field": "vote.vote_value",
                "surface_form": f"Vote {vote_node_id}",
                "chunk_index": -1,
                "chunk_index_unknown": True,
                "origin_key": self._compose_origin_key(str(vote_node_id), "vote.vote_value"),
                "vote_value": vote_raw_int if vote_raw_int is not None else vote_value,
                "raw_vote_value": vote_raw_int,
                "vote_value_concept": vote_value_concept_uri,
                "created_at": vote_dict.get("created_at"),
                "updated_at": vote_dict.get("updated_at"),
            },
        )
        vote_entity.name = vote_entity.label
        vote_entity.type = vote_entity.entity_type
        self._ensure_entity_ontology_grounding(vote_entity)
        vote_entity = self.global_registry.get_or_create(vote_entity)
        self._ensure_entity_ontology_grounding(vote_entity)

        # Backbone type guards (hard invariant for vote structure)
        if getattr(vote_entity, "entity_type", None) != "Vote":
            raise RuntimeError(f"Backbone invariant failed: vote entity is not Vote ({vote_entity.uri})")
        if getattr(expert_entity, "entity_type", None) not in {"Expert", "Person"}:
            raise RuntimeError(f"Backbone invariant failed: castBy target must be Expert/Person ({expert_entity.uri})")
        if getattr(answer_entity, "entity_type", None) != "Answer":
            raise RuntimeError(f"Backbone invariant failed: receivesVote source must be Answer ({answer_entity.uri})")

        # Vote values stay structural. They are preserved as raw and SKOS-normalized
        # metadata on the Vote node, but they do not generate semantic relations.

        # Deterministic backbone edges
        cast_by_rel = make_relation(
            vote_entity.uri,
            "castBy",
            expert_entity.uri,
            discourse_id=str(vote_node_id),
            extractor="structural",
            context_source="dataset_vote",
            confidence=0.99,
            evidence=_mk_evidence_text(
                text=None,
                fallback=f"Structural link: vote_{vote_node_id} castBy expert_{expert_id}.",
            ),
        )
        cast_by_rel.uri = _make_rel_uri(f"vote_{vote_node_id}_castBy_{expert_id}")
        cast_by_rel.metadata.setdefault("field", "vote.vote_value")
        cast_by_rel.metadata.setdefault("source_entity_uri", str(vote_entity.uri))
        cast_by_rel.metadata.setdefault("relation_kind", "structural")
        cast_by_rel.metadata.setdefault(
            "origin_key",
            self._compose_origin_key(str(vote_node_id), "vote.vote_value"),
        )
        cast_by_rel.metadata.setdefault("source_doc_id", str(vote_node_id))

        receives_vote_rel = make_relation(
            answer_entity.uri,
            "receivesVote",
            vote_entity.uri,
            discourse_id=str(vote_node_id),
            extractor="structural",
            context_source="dataset_vote",
            confidence=0.99,
            evidence=_mk_evidence_text(
                text=None,
                fallback=f"Structural link: answer_{answer_id} receivesVote vote_{vote_node_id}.",
            ),
        )
        receives_vote_rel.uri = _make_rel_uri(f"answer_{answer_id}_receivesVote_vote_{vote_node_id}")
        receives_vote_rel.metadata.setdefault("field", "vote.vote_value")
        receives_vote_rel.metadata.setdefault("source_entity_uri", str(vote_entity.uri))
        receives_vote_rel.metadata.setdefault("relation_kind", "structural")
        receives_vote_rel.metadata.setdefault(
            "origin_key",
            self._compose_origin_key(str(vote_node_id), "vote.vote_value"),
        )
        receives_vote_rel.metadata.setdefault("source_doc_id", str(vote_node_id))

        for rel in (cast_by_rel, receives_vote_rel):
            self._ensure_relation_chunk_provenance(rel, fallback_discourse_id=str(vote_node_id))

        return ProcessingResult(
            entities=[expert_entity, answer_entity, vote_entity],
            relations=[cast_by_rel, receives_vote_rel],
            metadata={
                'vote_id': vote_id,
                'expert_id': expert_id,
                'answer_id': answer_id,
                'vote_value': vote_value,
                'raw_vote_value': vote_raw_int,
                'vote_value_concept': vote_value_concept_uri
            }
        )

    # ---------------------------------------------------------
    # Experts
    # ---------------------------------------------------------
    def process_expert(self, expert_dict: Dict[str, Any]) -> ProcessingResult:
        """Process an expert profile."""
        expert_id = expert_dict.get('id')
        user_id = expert_dict.get('user_id')
        expertise = expert_dict.get('area_of_expertise', '')
        bio = expert_dict.get('bio', '')

        expert_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
        label = expert_dict.get('display_name', f"Expert {expert_id}")
        
        expert_entity = Entity(
            uri=expert_uri,
            label=label,
            entity_type="Expert",
            confidence=0.95,
            metadata={
                'airo_type': 'Stakeholder',
                'ontology_classes': ['foaf:Person', 'prov:Agent'],
                'area_of_expertise': expertise,
                'bio': bio,
                'user_id': user_id,
                'discourse_id': str(expert_id),
                'field': 'expert.profile',
                'surface_form': label,
                'context_source': 'dataset_expert',
                'source': 'structural',
                'source_doc_id': str(expert_id),
                'chunk_index': -1,
                'chunk_index_unknown': True,
                'origin_key': self._compose_origin_key(str(expert_id), 'expert.profile'),
                # Explicit, span-less structural source occurrence for this
                # dataset Expert record. Preserved across canonical-entity
                # merges by GlobalEntityRegistry._merge_source_occurrences
                # even when the Expert was first registered through another
                # field (vote/answer/question/theme) and first-writer-wins
                # keeps that earlier occurrence as the primary scalars. The
                # serializer materialises the expert.profile DiscourseUnit
                # from THIS occurrence, not from the primary scalars or the
                # canonical URI.
                'source_occurrences': [
                    {
                        'field': 'expert.profile',
                        'discourse_id': str(expert_id),
                        'source_doc_id': str(expert_id),
                        'origin_key': self._compose_origin_key(str(expert_id), 'expert.profile'),
                        'source': 'structural',
                        'context_source': 'dataset_expert',
                        'chunk_index': -1,
                        'chunk_index_unknown': True,
                    }
                ],
            }
        )

        expert_entity.name = expert_entity.label
        expert_entity.type = expert_entity.entity_type
        self._ensure_entity_ontology_grounding(expert_entity)
        
        expert_entity = self.global_registry.get_or_create(expert_entity)
        self._ensure_entity_ontology_grounding(expert_entity)
        
        return ProcessingResult(
            entities=[expert_entity],
            relations=[],
            metadata={
                'user_id': user_id,
                'expert_id': expert_id,
                'area_of_expertise': expertise
            }
        )

    # ---------------------------------------------------------
    # Documents
    # ---------------------------------------------------------
    def process_document(self, document_dict: Dict[str, Any]) -> ProcessingResult:
        """Process a document (report, file)."""
        discourse_id = document_dict.get('id')
        title = document_dict.get('title', f"Document {discourse_id}")
        description = document_dict.get('description', '')
        url = document_dict.get('url', '')

        doc_uri = f"{self.config.NAMESPACES['etd']}document_{discourse_id}"
        
        document_entity = Entity(
            uri=doc_uri,
            label=title,
            entity_type="Resource",
            confidence=0.90,
            metadata={
                'airo_type': 'Document',
                'ontology_classes': ['bibo:Document', 'foaf:Document'],
                'url': url,
                'description': description,
                'discourse_id': str(discourse_id),
                'field': 'document.description',
                'surface_form': title,
                'context_source': 'dataset_document',
                'source': 'structural',
                'source_doc_id': str(discourse_id),
                'chunk_index': -1,
                'chunk_index_unknown': True,
                'origin_key': self._compose_origin_key(str(discourse_id), 'document.description'),
            }
        )

        document_entity.name = document_entity.label
        document_entity.type = document_entity.entity_type
        self._ensure_entity_ontology_grounding(document_entity)
        
        document_entity = self.global_registry.get_or_create(document_entity)
        self._ensure_entity_ontology_grounding(document_entity)

        # For now, we skip URL extraction from description to avoid
        # dependency on the old entity_extractor._extract_urls().
        # This can be re-added later as a small utility.
        relations: List[Relation] = []

        return ProcessingResult(
            entities=[document_entity],
            relations=relations,
            metadata={
                'discourse_id': discourse_id,
                'title': title,
                'text': description or title,
            }
        )

    # ---------------------------------------------------------
    # Question classification
    # ---------------------------------------------------------
    def _classify_question_type(self, text: str) -> str:
        """Classify question type."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['what', 'define', 'definition']):
            return 'definitional'
        elif any(word in text_lower for word in ['how', 'explain', 'method']):
            return 'explanatory'
        elif any(word in text_lower for word in ['why', 'reason', 'cause']):
            return 'causal'
        elif any(word in text_lower for word in ['should', 'ought', 'must']):
            return 'normative'
        return 'general'

    # ---------------------------------------------------------
    # Batch processing orchestration
    # ---------------------------------------------------------
    def process_batch(self, data: Dict[str, Any]) -> AnnotationBatch:
        """Process all documents in batches."""
        annotations = AnnotationBatch()
        self._candidate_generation_diagnostics = []
        batch_size = max(1, int(self.config.BATCH_SIZE))
        
        themes = data.get('themes', data.get('Theme', []))
        if themes:
            print(f"\nâš™ï¸ Processing {len(themes)} themes...")
            theme_batches = (len(themes) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iterator(themes, batch_size), total=theme_batches):
                results = self.process_themes_batch(batch)
                for theme, result in zip(batch, results):
                    annotations.themes[theme.get('id')] = result
                cleanup_memory()
        
        questions = data.get('questions', data.get('Question', []))
        if questions:
            print(f"\nâš™ï¸ Processing {len(questions)} questions...")
            question_batches = (len(questions) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iterator(questions, batch_size), total=question_batches):
                results = self.process_questions_batch(batch)
                for question, result in zip(batch, results):
                    annotations.questions[question.get('id')] = result
                cleanup_memory()
        
        answers = data.get('answers', data.get('Answer', []))
        if answers:
            print(f"\nâš™ï¸ Processing {len(answers)} answers...")
            answer_batches = (len(answers) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iterator(answers, batch_size), total=answer_batches):
                results = self.process_answers_batch(batch)
                for answer, result in zip(batch, results):
                    annotations.answers[answer.get('id')] = result
                cleanup_memory()
        
        votes = data.get('votes', data.get('Vote', []))
        if votes:
            print(f"\nâš™ï¸ Processing {len(votes)} votes...")
            vote_batches = (len(votes) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iterator(votes, batch_size), total=vote_batches):
                for vote in batch:
                    result = safe_process_with_retry(
                        vote,
                        lambda v: self.process_vote(v)
                    )
                    annotations.votes[vote.get('id')] = result
                cleanup_memory()
        
        experts = data.get('experts', data.get('Expert', []))
        if experts:
            print(f"\nâš™ï¸ Processing {len(experts)} experts...")
            for expert in tqdm(experts):
                result = safe_process_with_retry(
                    expert,
                    lambda e: self.process_expert(e)
                )
                annotations.experts[expert.get('id')] = result
        
        documents = data.get('documents', data.get('Document', []))
        if documents:
            print(f"\nâš™ï¸ Processing {len(documents)} documents...")
            for document in tqdm(documents):
                result = safe_process_with_retry(
                    document,
                    lambda d: self.process_document(d)
                )
                annotations.documents[document.get('id')] = result

        # ðŸ”¹ ENTITY CONSOLIDATION (NEW STEP)
        print("\nðŸ”„ Consolidating entities across the entire registry...")

        consolidator = EntityConsolidator(
            self.global_registry,
            self.entity_normalizer,
            output_dir=self.config.OUTPUT_DIR,
        )

        # 1) Pull all entities from the registry (production-safe)
        store = getattr(self.global_registry, "entities", None)

        if isinstance(store, dict):
            all_entities = list(store.values())
        elif isinstance(store, list):
            all_entities = list(store)
        else:
            all_entities = []
            if hasattr(self.global_registry, "get_all_entities"):
                all_entities = list(self.global_registry.get_all_entities())


        # 2) Consolidate (returns a LIST of canonical entities)
        consolidated_entities = consolidator.consolidate(all_entities)

        # 3) Write back into registry (DO NOT replace the registry object)
        if hasattr(self.global_registry, "entities"):
            # Keep registry storage as dict keyed by entity URI (backward compatible)
            self.global_registry.entities = {e.uri: e for e in consolidated_entities if getattr(e, "uri", None)}

            # Optional: rebuild registry indexes if supported
            if hasattr(self.global_registry, "reindex"):
                self.global_registry.reindex()
        elif hasattr(self.global_registry, "set_entities"):
            self.global_registry.set_entities(consolidated_entities)
        else:
            # last resort: keep registry as-is, but avoid crashing
            raise TypeError("GlobalEntityRegistry has no 'entities' or set_entities() API to store consolidated entities.")

        print(f"âœ“ Entity consolidation complete. Total unique entities: {len(consolidated_entities)}")
        self._apply_entity_redirects_to_annotations(annotations, getattr(consolidator, "redirects", {}))

        # 4) Canonical linking across FULL registry (cross-chunk/cross-document)
        print("\n🔗 Running canonical linking across full registry...")
        coverage = self.run_canonical_linking(annotations)
        try:
            save_json(coverage, self.config.OUTPUT_DIR / "canonical_linking_report.json")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to persist canonical_linking_report.json: %s", exc)
        print(
            f"✓ Canonical linking complete. Raw={coverage.get('raw_entities')} "
            f"Canonical={coverage.get('canonical_entities')} Merges={coverage.get('merges_count')}"
        )

        # Ensure ontology grounding coverage for the full canonical registry.
        get_all = getattr(self.global_registry, "get_all_entities", None)
        if callable(get_all):
            for ent in get_all():
                self._ensure_entity_ontology_grounding(ent)


        # 5) Predicate precision + ontology enrichment after canonical linking.
        self.apply_predicate_precision_and_enrichment(annotations)
        self.validate_semantic_relations(annotations)

        return annotations

    @staticmethod
    def _relation_grounded_chunk_participation(
        relation_lists: List[List[Relation]],
    ) -> Dict[str, Set[Tuple[str, int]]]:
        """{endpoint entity URI -> {(discourse_id, chunk_index)}} for chunks
        a GROUNDED relation affirmatively places that endpoint in. Only
        relations whose evidence is grounded to a concrete, non-unknown
        chunk contribute; literal objects are ignored. Gives the canonical
        linker's same-chunk cross-type reconciliation the affirmative
        both-sides-in-one-concrete-chunk evidence it requires for an
        endpoint whose own metadata carries no span."""
        def _truthy(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in {"true", "1", "yes"}

        out: Dict[str, Set[Tuple[str, int]]] = {}

        def _record(uri: Any, did: str, ci: Any) -> None:
            u = str(uri or "").strip()
            if not u or not did:
                return
            try:
                c = int(ci)
            except (TypeError, ValueError):
                return
            if c < 0:
                return
            out.setdefault(u, set()).add((did, c))

        for rel_list in relation_lists or []:
            for rel in rel_list or []:
                meta = getattr(rel, "metadata", None) or {}
                status = str(meta.get("groundingStatus") or "").strip().lower()
                if status and status not in {"grounded", "chunk_grounded"}:
                    continue
                did = str(meta.get("discourse_id") or meta.get("doc_id")
                          or getattr(rel, "doc_id", "") or "").strip()
                endpoints = [getattr(rel, "source", None)]
                if not getattr(rel, "target_is_literal", False):
                    endpoints.append(getattr(rel, "target", None))
                contexts: List[Tuple[str, Any]] = []
                if did and not _truthy(meta.get("chunk_index_unknown")):
                    contexts.append((did, meta.get("chunk_index")))
                spans = meta.get("evidence_spans")
                if isinstance(spans, list):
                    for seg in spans:
                        if isinstance(seg, dict):
                            sdid = str(seg.get("discourse_id") or did or "").strip()
                            contexts.append((sdid, seg.get("chunk_index")))
                for uri in endpoints:
                    for cdid, cci in contexts:
                        _record(uri, cdid, cci)
        return out

    def run_canonical_linking(self, annotations: AnnotationBatch) -> Dict[str, Any]:
        pre_link_count = len(self.global_registry.get_all_entities()) if hasattr(self.global_registry, "get_all_entities") else 0
        participation = self._relation_grounded_chunk_participation(
            self._iter_annotation_relation_lists(annotations)
        )
        if participation and hasattr(self.canonical_linker, "set_entity_chunk_participation"):
            self.canonical_linker.set_entity_chunk_participation(participation)
        canonical_entities = self.canonical_linker.run_full_registry()
        self._apply_entity_redirects_to_annotations(annotations, getattr(self.canonical_linker, "redirects", {}))
        for rel_list in self._iter_annotation_relation_lists(annotations):
            self.canonical_linker.rewrite_relation_endpoints(rel_list)
        self._reconcile_relation_endpoints(annotations)
        return self.canonical_linker.get_report(pre_link_count, len(canonical_entities))

    def _reconcile_relation_endpoints(self, annotations: AnnotationBatch) -> Dict[str, int]:
        """Final invariant after all entity redirects have been applied: every
        non-literal relation endpoint URI must be a URI that a registered
        canonical entity actually holds. A multi-step merge (A -> B in one
        pass, B -> C in a later one) can otherwise strand an endpoint on an
        intermediate URI that no surviving entity holds and that the RDF
        serializer would then materialise as a bare, provenance-less
        ``etd:Entity`` stub (no datasetId/runId/field/source_doc_id/span).

        Repair is redirect-only -- resolve the stranded URI through the union
        of every redirect map produced this run (transitively) and, when that
        lands on a registered entity, rewrite the endpoint to it. Anything
        still unresolved is recorded (never silently materialised); it is a
        genuine defect for the provenance gate / manual review to surface.
        Generic: no dependency on any label, type, dataset or discourse.
        """
        diagnostics = getattr(self, "_provenance_diagnostics", None)
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            self._provenance_diagnostics = diagnostics

        registered: Set[str] = set()
        get_all = getattr(self.global_registry, "get_all_entities", None)
        if callable(get_all):
            registered = {
                str(getattr(ent, "uri", "") or "").strip()
                for ent in get_all()
                if str(getattr(ent, "uri", "") or "").strip()
            }

        combined: Dict[str, str] = {}
        combined.update(getattr(self, "_entity_redirects", {}) or {})
        combined.update(getattr(self.canonical_linker, "redirects", {}) or {})

        def _resolve(uri: str) -> str:
            cur = uri
            seen: Set[str] = set()
            while cur in combined and cur not in seen:
                seen.add(cur)
                nxt = str(combined[cur] or "")
                if not nxt or nxt == cur:
                    break
                cur = nxt
            return cur

        repaired = 0
        unresolved: List[Dict[str, Any]] = []
        for rel in self._iter_all_annotation_relations(annotations):
            meta = getattr(rel, "metadata", None) or {}
            endpoints = [("source", str(getattr(rel, "source", "") or "").strip())]
            target_is_literal = bool(getattr(rel, "target_is_literal", False)) or (
                getattr(rel, "target_literal", None) is not None
            )
            if not target_is_literal:
                endpoints.append(("target", str(getattr(rel, "target", "") or "").strip()))

            for side, uri in endpoints:
                if not uri or not uri.startswith("http") or uri in registered:
                    continue
                resolved = _resolve(uri)
                if resolved != uri and resolved in registered:
                    if side == "source":
                        rel.source = resolved
                    elif hasattr(rel, "set_target_entity"):
                        rel.set_target_entity(resolved)
                    else:
                        rel.target = resolved
                    repaired += 1
                    continue
                unresolved.append(
                    {
                        "relation_uri": str(getattr(rel, "uri", "") or ""),
                        "side": side,
                        "endpoint_uri": uri,
                        "resolved_to": resolved if resolved != uri else None,
                        "predicate": str(getattr(rel, "relation", "") or ""),
                        "discourse_id": str(meta.get("discourse_id") or ""),
                        "field": str(meta.get("field") or meta.get("source_field") or ""),
                        "surface_form": str(
                            meta.get("subject_text" if side == "source" else "object_text") or ""
                        ),
                    }
                )
                record_event(
                    self.quality_events,
                    category="late_endpoint_unreconciled",
                    action="flagged",
                    discourse_id=meta.get("discourse_id"),
                    source_relation_id=str(getattr(rel, "uri", "") or "") or None,
                    subject_surface=str(meta.get("subject_text") or "").strip() or None,
                    predicate_surface=str(getattr(rel, "relation", "") or "") or None,
                    object_surface=str(meta.get("object_text") or "").strip() or None,
                    reason=f"{side} endpoint URI has no registered canonical entity after all redirects",
                )

        diagnostics["reconciled_relation_endpoints"] = int(repaired)
        diagnostics["unresolved_relation_endpoints"] = unresolved
        if unresolved:
            logger.warning(
                "Relation-endpoint reconciliation: %d repaired, %d still unresolved",
                repaired,
                len(unresolved),
            )
        return {"repaired": repaired, "unresolved": len(unresolved)}

    def _iter_all_annotation_relations(self, annotations: AnnotationBatch) -> List[Relation]:
        out: List[Relation] = []
        for rel_list in self._iter_annotation_relation_lists(annotations):
            out.extend(rel_list)
        return out
    def apply_predicate_precision_and_enrichment(self, annotations: AnnotationBatch) -> None:
        """Apply predicate precision and ontology enrichment after canonical linking."""
        # Use FULL canonical registry for type-aware predicate validation.
        # This preserves structural backbone predicates even when per-document
        # `pr.entities` does not contain reminted canonical endpoints.
        canonical_entities: List[Entity] = []
        get_all = getattr(self.global_registry, "get_all_entities", None)
        if callable(get_all):
            canonical_entities = list(get_all())

        for block in (
            annotations.themes,
            annotations.questions,
            annotations.answers,
            annotations.votes,
            annotations.experts,
            annotations.documents,
        ):
            for pr in block.values():
                discourse_text = None
                if isinstance(pr.metadata, dict):
                    discourse_text = pr.metadata.get("text")
                refined_relations = self.predicate_precision.refine_relations(
                    relations=pr.relations,
                    entities=canonical_entities or pr.entities,
                    discourse_text=discourse_text,
                )
                if getattr(self.config, "STRICT_PREDICATE_INVENTORY", False):
                    for relation in refined_relations:
                        if relation.relation not in self.predicate_precision.allowed:
                            raise RuntimeError(
                                f"Predicate precision violation: '{relation.relation}' is outside allowed inventory"
                            )
                for relation in refined_relations:
                    enrichment = self.ontology_mapper.enrich_relation(relation)
                    if isinstance(enrichment, dict) and enrichment:
                        relation.metadata.update(enrichment)
                    if "discourse_id" not in relation.metadata and "doc_id" in relation.metadata:
                        relation.metadata["discourse_id"] = relation.metadata.get("doc_id")
                    relation.metadata.setdefault("source_field", str(relation.metadata.get("field") or "").strip())
                pr.relations = refined_relations

    def validate_semantic_relations(self, annotations: AnnotationBatch) -> Dict[str, int]:
        canonical_entities: Dict[str, Entity] = {}
        get_all = getattr(self.global_registry, "get_all_entities", None)
        if callable(get_all):
            canonical_entities = {
                str(getattr(ent, "uri", "") or ""): ent
                for ent in get_all()
                if str(getattr(ent, "uri", "") or "").strip()
            }

        kept = 0
        dropped = 0
        for block in (
            annotations.themes,
            annotations.questions,
            annotations.answers,
            annotations.votes,
            annotations.experts,
            annotations.documents,
        ):
            for pr in block.values():
                discourse_text = ""
                if isinstance(pr.metadata, dict):
                    discourse_text = str(pr.metadata.get("text") or "")
                pr_discourse_id = None
                pr_field = None
                pr_origin_key = None
                if pr.relations:
                    first_meta = getattr(pr.relations[0], "metadata", None) or {}
                    pr_discourse_id = first_meta.get("discourse_id")
                    pr_field = first_meta.get("field") or first_meta.get("source_field")
                    pr_origin_key = first_meta.get("origin_key")

                # Defect A (global scope): the per-semantic-unit
                # _rebind_stale_shared_respectively_targets calls only ever
                # see the relations of ONE unit, so when the sibling
                # relations of a "N1 and N2 <unit> respectively" statistic
                # are produced by SEPARATE unit-resolution calls, they are
                # never co-present as a >=2 group there. Re-run the SAME
                # detector once per discourse now that pr.relations holds
                # every surviving relation for this record with canonical
                # URI endpoints. Registry-only (local_entities=None): the
                # compound entity is resolved from self.global_registry,
                # Phase-1 mints distinct per-value entities via the normal
                # canonical path, and its `str(ve.uri) == comp_uri` guard
                # abandons cleanly if a value ever re-matched the compound.
                # Runs BEFORE _supersede_stale_respectively_duplicates so
                # the `respectively_split_from` markers it writes are
                # visible to that pass. Idempotent: once rebound, the
                # sibling targets are distinct so no >=2 group re-forms.
                self._rebind_stale_shared_respectively_targets(
                    pr.relations,
                    local_entities=None,
                    discourse_text=discourse_text,
                    discourse_id=pr_discourse_id,
                    field=pr_field,
                    origin_key=pr_origin_key,
                    unit_text=discourse_text,
                )

                pr.relations = self._supersede_stale_respectively_duplicates(pr.relations, pr_discourse_id)
                filtered: List[Relation] = []
                for relation in pr.relations:
                    meta = getattr(relation, "metadata", None) or {}
                    meta.setdefault("source_field", str(meta.get("field") or "").strip())
                    pred = str(getattr(relation, "relation", "") or "").strip()
                    relation_kind = str(meta.get("relation_kind") or "").strip()

                    # An endpoint the unresolved-endpoint guard left
                    # explicitly unresolved must never reach materialization
                    # -- checked BEFORE the
                    # predicate-registry routing below, since an
                    # unregistered/local predicate (e.g. one not in
                    # SEMANTIC_RELATION_PREDICATES) would otherwise bypass
                    # self._validate_semantic_relation_candidate() entirely
                    # and be kept unvalidated.
                    if bool(meta.get("subject_endpoint_unresolved_coreference")) or bool(
                        meta.get("object_endpoint_unresolved_coreference")
                    ):
                        meta["semantic_validation_status"] = "rejected"
                        meta["semantic_validation_code"] = "unresolved_coreference"
                        meta["semantic_validation_reason"] = (
                            "unresolved coreference endpoint "
                            f"(subject={bool(meta.get('subject_endpoint_unresolved_coreference'))}, "
                            f"object={bool(meta.get('object_endpoint_unresolved_coreference'))})"
                        )
                        relation.metadata = meta
                        dropped += 1
                        record_event(
                            self.quality_events,
                            category="unresolved_coreference_drop",
                            action="dropped",
                            discourse_id=meta.get("discourse_id"),
                            source_relation_id=str(getattr(relation, "uri", "") or "") or None,
                            subject_surface=str(
                                meta.get("subject_endpoint_unresolved_original_surface")
                                or meta.get("subject_text") or ""
                            ).strip() or None,
                            predicate_surface=pred or None,
                            object_surface=str(
                                meta.get("object_endpoint_unresolved_original_surface")
                                or meta.get("object_text") or ""
                            ).strip() or None,
                            evidence_text=str(meta.get("evidence") or meta.get("evidence_quote") or "").strip() or None,
                            reason=str(
                                meta.get("subject_endpoint_unresolved_reason")
                                or meta.get("object_endpoint_unresolved_reason")
                                or "unresolved_coreference"
                            ),
                        )
                        continue

                    if pred in STRUCTURAL_RELATION_PREDICATES:
                        meta.setdefault("relation_kind", "structural")
                        relation.metadata = meta
                        filtered.append(relation)
                        kept += 1
                        continue
                    if relation_kind in {"structural", "provenance"}:
                        if pred in SEMANTIC_RELATION_PREDICATES:
                            meta["semantic_validation_status"] = "rejected"
                            meta["semantic_validation_reason"] = "structural_semantic_predicate_blocked"
                            relation.metadata = meta
                            dropped += 1
                            continue
                        relation.metadata = meta
                        filtered.append(relation)
                        kept += 1
                        continue
                    if pred not in SEMANTIC_RELATION_PREDICATES:
                        filtered.append(relation)
                        kept += 1
                        continue
                    accepted, reason = self._validate_semantic_relation_candidate(
                        relation,
                        entities_by_uri=canonical_entities,
                        discourse_text=discourse_text,
                    )
                    if accepted:
                        meta["relation_kind"] = "extracted_semantic" if not getattr(relation, "is_inferred", False) else "inferred_semantic"
                        meta["semantic_validation_status"] = "accepted"
                        meta["semantic_validation_reason"] = reason
                        relation.metadata = meta
                        filtered.append(relation)
                        kept += 1
                    else:
                        meta["semantic_validation_status"] = "rejected"
                        meta["semantic_validation_reason"] = reason
                        relation.metadata = meta
                        dropped += 1
                filtered, shadowed = self._consolidate_competing_semantic_relations(filtered)
                if shadowed:
                    kept -= len(shadowed)
                    dropped += len(shadowed)
                # Entity-vs-finding-style predicate refinement --
                # runs only on the FINAL, already-accepted relation list for
                # this discourse record, after validation/consolidation, so
                # it can never change which relations were kept or dropped,
                # only the predicate on already-accepted is/has/relatedTo
                # relations.
                self._refine_generic_predicates_for_relations(filtered, canonical_entities, discourse_text)
                pr.relations = filtered
        summary = {"kept": kept, "dropped": dropped}
        logger.info("Semantic relation validation summary: %s", summary)
        return summary

    def _consolidate_competing_semantic_relations(
        self,
        relations: List[Relation],
    ) -> Tuple[List[Relation], List[Relation]]:
        if len(relations) < 2:
            return relations, []

        policy = self._get_semantic_policy()
        passthrough: List[Relation] = []
        groups: Dict[Tuple[str, str, str, str, str, str], List[Relation]] = {}

        for relation in relations:
            meta = getattr(relation, "metadata", None) or {}
            pred = str(getattr(relation, "relation", "") or "").strip()
            family = policy.predicate_competition_family(pred)
            if not family:
                passthrough.append(relation)
                continue
            proposition_text = normalize_text(
                str(
                    getattr(relation, "context", None)
                    or meta.get("local_context")
                    or meta.get("evidence")
                    or meta.get("evidence_quote")
                    or ""
                )
            )
            if not proposition_text:
                passthrough.append(relation)
                continue
            key = (
                str(meta.get("field") or meta.get("source_field") or "").strip(),
                str(meta.get("source_doc_id") or meta.get("discourse_id") or meta.get("doc_id") or "").strip(),
                str(getattr(relation, "source", "") or "").strip(),
                str(getattr(relation, "target", "") or "").strip(),
                proposition_text,
                family,
            )
            groups.setdefault(key, []).append(relation)

        kept_ids: Set[int] = set(id(rel) for rel in passthrough)
        shadowed: List[Relation] = []
        for bucket in groups.values():
            if len(bucket) == 1:
                kept_ids.add(id(bucket[0]))
                continue
            ranked = sorted(
                bucket,
                key=lambda rel: (
                    policy.predicate_specificity_rank(str(getattr(rel, "relation", "") or "").strip()),
                    float(getattr(rel, "confidence", 0.0) or 0.0),
                ),
                reverse=True,
            )
            winner = ranked[0]
            kept_ids.add(id(winner))
            for loser in ranked[1:]:
                loser_meta = getattr(loser, "metadata", None) or {}
                loser_meta["semantic_validation_status"] = "rejected"
                loser_meta["semantic_validation_code"] = "specificity:shadowed_by_more_specific_predicate"
                loser_meta["semantic_validation_reason"] = (
                    f"predicate '{getattr(loser, 'relation', '')}' shadowed by more specific "
                    f"predicate '{getattr(winner, 'relation', '')}' for the same local proposition"
                )
                loser.metadata = loser_meta
                shadowed.append(loser)

        kept: List[Relation] = []
        for relation in relations:
            if id(relation) in kept_ids:
                kept.append(relation)
        return kept, shadowed

    def _iter_annotation_relation_lists(self, annotations: AnnotationBatch) -> List[List[Relation]]:
        relation_lists: List[List[Relation]] = []
        for block in (
            annotations.themes,
            annotations.questions,
            annotations.answers,
            annotations.votes,
            annotations.experts,
            annotations.documents,
        ):
            for pr in block.values():
                relation_lists.append(pr.relations)
        return relation_lists
    def _apply_entity_redirects_to_annotations(self, annotations: AnnotationBatch, redirects: Dict[str, str]) -> None:
        """Rewrite entity and relation URIs to canonical redirects after consolidation."""
        if not redirects:
            return

        # Keep a cumulative view of every redirect ever applied to the
        # annotations, so the post-canonical-linking reconciliation sweep can
        # resolve an endpoint through the union of the EntityConsolidator and
        # CanonicalLinker maps.
        redirect_union = getattr(self, "_entity_redirects", None)
        if redirect_union is None:
            redirect_union = {}
            self._entity_redirects = redirect_union
        redirect_union.update(redirects)

        def _final(uri: Any) -> Optional[str]:
            """Follow the redirect chain to its end (cycle-guarded). A merge
            pass can produce A -> B and, later, B -> C; a single dict lookup
            would strand an endpoint on the intermediate B, which no surviving
            entity holds -- exactly the shape that leaves a provenance-less
            bare endpoint URI in the graph."""
            cur = uri
            seen: Set[Any] = set()
            while cur in redirects and cur not in seen:
                seen.add(cur)
                nxt = redirects[cur]
                if not nxt or nxt == cur:
                    break
                cur = nxt
            return cur if cur != uri else None

        def _rewrite_result(pr: ProcessingResult) -> None:
            for ent in pr.entities:
                uri = getattr(ent, "uri", None)
                resolved = _final(uri)
                if resolved is not None:
                    ent.uri = resolved
            for rel in pr.relations:
                src = getattr(rel, "source", None)
                tgt = getattr(rel, "target", None)
                src_final = _final(src)
                tgt_final = _final(tgt)
                if src_final is not None:
                    rel.source = src_final
                if tgt_final is not None:
                    # Same invariant as CanonicalLinker
                    # .rewrite_relation_endpoints -- a redirect target is
                    # always a genuine internal Entity/Proposition URI, so
                    # any stale literal-typed state left over from an
                    # earlier stage must not survive the redirect.
                    if hasattr(rel, "set_target_entity"):
                        rel.set_target_entity(tgt_final)
                    else:
                        rel.target = tgt_final

        for block in (
            annotations.themes,
            annotations.questions,
            annotations.answers,
            annotations.votes,
            annotations.experts,
            annotations.documents,
        ):
            for pr in block.values():
                _rewrite_result(pr)

    # ---------------------------------------------------------
    # Reasoning & quality
    # ---------------------------------------------------------
    def apply_reasoning(self, annotations: Any) -> List[Relation]:
        """Apply multi-hop reasoning to infer new relations.

        Works with:
          - AnnotationBatch (has get_all_relations)
          - dict (nested structure: {doc_type: {doc_id: {relations:[...]}}})
        """
        if not self.config.USE_REASONING:
            return []

        print("\nðŸ§  Applying multi-hop reasoning...")

        # ---------------------------------------------------------
        # Collect relations (supports both object + dict)
        # ---------------------------------------------------------
        all_relations: List[Union[Relation, Dict[str, Any]]] = []

        # Case A: model/object API
        if hasattr(annotations, "get_all_relations") and callable(getattr(annotations, "get_all_relations")):
            all_relations = annotations.get_all_relations()

        # Case B: dict API (annotations_dict)
        elif isinstance(annotations, dict):
            for doc_type, docs in annotations.items():
                if not isinstance(docs, dict):
                    continue
                for _, doc in docs.items():
                    if isinstance(doc, dict):
                        rels = doc.get("relations", [])
                        if isinstance(rels, list):
                            all_relations.extend(rels)

        else:
            raise TypeError(f"apply_reasoning() expected AnnotationBatch or dict, got {type(annotations)}")

        # Build deterministic discourse priors from extracted relations.
        entity_did_counts: Dict[str, Dict[str, int]] = {}
        entity_origin_counts: Dict[str, Dict[str, int]] = {}
        entity_field_counts: Dict[str, Dict[str, int]] = {}

        def _inc(counter: Dict[Any, int], key: Any) -> None:
            counter[key] = int(counter.get(key, 0)) + 1

        def _best_key(counter: Optional[Dict[Any, int]]) -> Optional[Any]:
            if not counter:
                return None
            return sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0][0]

        for item in all_relations:
            if isinstance(item, Relation):
                src = str(getattr(item, "source", "") or "")
                tgt = str(getattr(item, "target", "") or "")
                meta = getattr(item, "metadata", None) or {}
            elif isinstance(item, dict):
                src = str(item.get("source") or item.get("source_id") or "")
                tgt = str(item.get("target") or item.get("target_id") or "")
                meta = item.get("metadata", {}) or {}
            else:
                continue

            did_raw = meta.get("discourse_id") or meta.get("doc_id")
            did = str(did_raw).strip() if did_raw is not None else ""
            field_raw = meta.get("field")
            field = str(field_raw).strip() if field_raw is not None else ""
            origin_raw = meta.get("origin_key")
            origin = str(origin_raw).strip() if origin_raw is not None else ""
            if not origin:
                origin = self._compose_origin_key(did, field)

            if did:
                for ent_uri in (src, tgt):
                    if ent_uri:
                        ent_counter = entity_did_counts.setdefault(ent_uri, {})
                        _inc(ent_counter, did)
            if origin:
                for ent_uri in (src, tgt):
                    if ent_uri:
                        origin_counter = entity_origin_counts.setdefault(ent_uri, {})
                        _inc(origin_counter, origin)
            if field:
                for ent_uri in (src, tgt):
                    if ent_uri:
                        field_counter = entity_field_counts.setdefault(ent_uri, {})
                        _inc(field_counter, field)

        inferred = self.reasoner.infer_relations(all_relations)
        self.canonical_linker.rewrite_relation_endpoints(inferred)
        canonical_entities: Dict[str, Entity] = {}
        get_all = getattr(self.global_registry, "get_all_entities", None)
        if callable(get_all):
            canonical_entities = {
                str(getattr(ent, "uri", "") or ""): ent
                for ent in get_all()
                if str(getattr(ent, "uri", "") or "").strip()
            }

        validated_inferred: List[Relation] = []
        for rel in inferred:
            meta = getattr(rel, "metadata", None) or {}
            src = str(getattr(rel, "source", "") or "")
            tgt = str(getattr(rel, "target", "") or "")

            did_raw = meta.get("discourse_id") or meta.get("doc_id") or getattr(rel, "doc_id", None)
            did = str(did_raw).strip() if did_raw is not None else ""
            if not did:
                did = str(_best_key(entity_did_counts.get(src)) or _best_key(entity_did_counts.get(tgt)) or "").strip()

            origin_raw = meta.get("origin_key")
            origin = str(origin_raw).strip() if origin_raw is not None else ""
            if not origin:
                origin = str(
                    _best_key(entity_origin_counts.get(src))
                    or _best_key(entity_origin_counts.get(tgt))
                    or ""
                ).strip()

            field_raw = meta.get("field")
            field = str(field_raw).strip() if field_raw is not None else ""
            if not field:
                field = str(
                    _best_key(entity_field_counts.get(src))
                    or _best_key(entity_field_counts.get(tgt))
                    or ""
                ).strip()
            if not field and origin and "|" in origin:
                field = origin.split("|", 1)[0].strip()
            if not origin:
                origin = self._compose_origin_key(did, field)

            if did:
                meta["discourse_id"] = did
            if field:
                meta["field"] = field
                meta.setdefault("source_field", field)
            if origin:
                meta["origin_key"] = origin
            meta.setdefault("extractor", "reasoner")
            meta.setdefault("context_source", "reasoner_inference")
            rel.metadata = meta
            self._ensure_relation_chunk_provenance(
                rel,
                fallback_discourse_id=did or getattr(rel, "doc_id", None),
                chunk_spans=self._get_chunk_spans_for_discourse(did, origin_key=origin),
            )
            evidence_text = str(meta.get("evidence_quote") or meta.get("evidence") or getattr(rel, "context", None) or "").strip()
            accepted, reason = self._validate_semantic_relation_candidate(
                rel,
                entities_by_uri=canonical_entities,
                discourse_text=evidence_text,
            )
            if not accepted:
                continue
            meta["relation_kind"] = "inferred_semantic"
            meta["semantic_validation_status"] = "accepted"
            meta["semantic_validation_reason"] = reason
            rel.metadata = meta
            validated_inferred.append(rel)
        print(f"  âœ“ Inferred {len(validated_inferred)} new relations")
        return validated_inferred


    def detect_conflicts(self, annotations: Any) -> List[Dict[str, Any]]:
        """Detect conflicts in the knowledge graph.

        Supports:
          - AnnotationBatch (has to_dict())
          - dict (already serialized annotations)
        """
        if not self.config.USE_CONFLICT_DETECTION:
            return []

        print("\nðŸ” Detecting conflicts...")

        # ---------------------------------------------------------
        # Normalize input to dict
        # ---------------------------------------------------------
        if isinstance(annotations, dict):
            ann_dict = annotations
        elif hasattr(annotations, "to_dict") and callable(getattr(annotations, "to_dict")):
            ann_dict = annotations.to_dict()
        else:
            raise TypeError(
                f"detect_conflicts() expected dict or AnnotationBatch, got {type(annotations)}"
            )

        conflicts = self.conflict_detector.detect(ann_dict)

        if conflicts:
            print(f"  âš ï¸  Found {len(conflicts)} conflicts")
        else:
            print("  âœ“ No conflicts detected")

        return conflicts


    def compute_quality_metrics(self, annotations: Any) -> dict:
        """Compute quality metrics.

        Supports:
          - AnnotationBatch (has to_dict())
          - dict (already serialized annotations)
        """
        print("\nðŸ“Š Computing quality metrics...")

        # ---------------------------------------------------------
        # Normalize input to dict
        # ---------------------------------------------------------
        if isinstance(annotations, dict):
            ann_dict = annotations
        elif hasattr(annotations, "to_dict") and callable(getattr(annotations, "to_dict")):
            ann_dict = annotations.to_dict()
        else:
            raise TypeError(
                f"compute_quality_metrics() expected dict or AnnotationBatch, got {type(annotations)}"
            )

        metrics_computer = QualityMetricsComputer(
            ann_dict,
            self.global_registry
        )

        metrics = metrics_computer.compute_all()
        metrics["provenance_diagnostics"] = dict(getattr(self, "_provenance_diagnostics", {}))
        summary = metrics_computer.get_summary_metrics()

        print(f"\nðŸ“ˆ Quality Metrics:")
        print(f"  Overall Quality Score: {summary.overall_quality_score}/10")
        print(f"  Total Entities: {summary.total_entities}")
        print(f"  Total Relations: {summary.total_relations}")
        print(f"  Entity Coverage: {summary.entity_coverage:.1%}")
        print(f"  Relation Coverage: {summary.relation_coverage:.1%}")

        return metrics












