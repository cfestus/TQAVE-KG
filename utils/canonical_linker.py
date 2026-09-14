"""Deterministic cross-chunk canonical entity linker."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from config import Config
from ontology_mappings import ENTITY_TYPE_TO_ONTOLOGY
from utils.entity_normalizer import containment_adds_material_token, is_initialism_pair
from utils.helpers import expand_curie

# Lazily-built, cached local-class-name -> ALL rdfs:subClassOf
# ancestor local-class-names, derived from the actual declared ETD schema
# (ontology/add_schema.py) -- never a hardcoded pair list. Used to decide
# whether two entity_type values disagreeing on the same external identity
# are a compatible specificity difference (one is an ancestor/descendant of
# the other) or a genuine incompatible disagreement.
_ETD_ANCESTOR_CACHE: Optional[Dict[str, Set[str]]] = None


def _etd_class_ancestors() -> Dict[str, Set[str]]:
    global _ETD_ANCESTOR_CACHE
    if _ETD_ANCESTOR_CACHE is not None:
        return _ETD_ANCESTOR_CACHE
    from rdflib import Graph, Namespace, RDFS
    from ontology.add_schema import add_schema

    etd_ns = Namespace(Config.NAMESPACES["etd"])
    g = Graph()
    add_schema(g, etd_ns)
    etd_str = str(etd_ns)
    direct_parents: Dict[str, Set[str]] = defaultdict(set)
    for s, _p, o in g.triples((None, RDFS.subClassOf, None)):
        s_str, o_str = str(s), str(o)
        if s_str.startswith(etd_str) and o_str.startswith(etd_str):
            direct_parents[s_str[len(etd_str):]].add(o_str[len(etd_str):])

    ancestors: Dict[str, Set[str]] = {}

    def _collect(node: str, seen: Set[str]) -> Set[str]:
        if node in ancestors:
            return ancestors[node]
        result: Set[str] = set()
        for parent in direct_parents.get(node, ()):
            if parent in seen:
                continue
            result.add(parent)
            result |= _collect(parent, seen | {parent})
        ancestors[node] = result
        return result

    for cls in list(direct_parents.keys()):
        _collect(cls, {cls})
    _ETD_ANCESTOR_CACHE = ancestors
    return ancestors


def _types_ontology_compatible(type_a: str, type_b: str) -> bool:
    """True if type_a and type_b are equal, or one is a declared
    rdfs:subClassOf* ancestor of the other -- a structural hierarchy check,
    never a hardcoded pair list."""
    a, b = str(type_a or "").strip(), str(type_b or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    ancestors = _etd_class_ancestors()
    return b in ancestors.get(a, set()) or a in ancestors.get(b, set())


def _most_specific_compatible_type(types: Set[str]) -> Optional[str]:
    """Given a set of 2+ distinct type names, return the one that is a
    descendant-or-equal of every other type in the set (i.e. the whole set
    forms one compatible ancestor chain), or None if no such type exists
    (a genuine incompatible disagreement)."""
    if len(types) == 1:
        return next(iter(types))
    ancestors = _etd_class_ancestors()
    for candidate in sorted(types):
        others = types - {candidate}
        if all(other in ancestors.get(candidate, set()) for other in others):
            return candidate
    return None


def _etd_class_depth(type_name: str) -> int:
    """Number of declared rdfs:subClassOf* ancestors a local ETD class has
    (0 for a root / unknown class). A larger value means a structurally
    more specific class -- used only as a deterministic tie-break when two
    same-referent members carry confidently-classified but incompatible
    types, never as a hardcoded pair ranking."""
    return len(_etd_class_ancestors().get(str(type_name or "").strip(), set()))


@dataclass
class LinkDecision:
    input_label: str
    norm_label: str
    chosen_canonical_uri: str
    rule: str
    scores: Dict[str, float]
    source_ids: List[str]


class CanonicalLinker:
    # CORE_GENERIC: bare structural/resource/process nouns whose genericity
    # makes fuzzy merging unsafe across essentially any subject-matter
    # domain (a legal-filings, medical-records, or agricultural-data
    # deployment of this pipeline would need every one of these protected
    # just as much as the AI-ethics deployment does). Domain-specific
    # blocked terms (e.g. AI-ethics vocabulary like "bias"/"fairness") no
    # longer live here -- they are supplied by the active domain profile's
    # "generic_merge_blocklist" gazetteer instead (see
    # domain_profiles.py::AI_ETHICS_PROFILE and _can_fuzzy_merge below,
    # which consults BOTH this list and the profile's).
    DO_NOT_MERGE: Set[str] = {
        "data",
        "dataset",
        "datasets",
        "model",
        "system",
        "audit",
        "audits",
    }
    STRUCTURAL_TYPES: Set[str] = {"Theme", "Question", "Answer", "Vote", "Expert", "Document", "DiscourseUnit"}
    ORG_SUFFIXES: Set[str] = {"inc", "ltd", "corp", "company", "co"}
    CLASS_NS_PREFIXES: Tuple[str, ...] = (
        Config.NAMESPACES.get("aipo", ""),
        Config.NAMESPACES.get("dpv", ""),
        Config.NAMESPACES.get("vair", ""),
        Config.NAMESPACES.get("airo", ""),
        Config.NAMESPACES.get("fmo", ""),
        Config.NAMESPACES.get("hudock", ""),
        Config.NAMESPACES.get("relaieo", ""),
        Config.NAMESPACES.get("dcat", ""),
        Config.NAMESPACES.get("schema", ""),
        Config.NAMESPACES.get("org", ""),
        Config.NAMESPACES.get("foaf", ""),
    )

    ALIAS_TABLE: Dict[str, str] = {
        "hmda": "Home Mortgage Disclosure Act",
        "home mortgage disclosure act": "Home Mortgage Disclosure Act",
        "gdpr": "General Data Protection Regulation",
        "general data protection regulation": "General Data Protection Regulation",
        "eu ai act": "Artificial Intelligence Act",
        "artificial intelligence act": "Artificial Intelligence Act",
        "unesco ai ethics recommendation": "UNESCO Recommendation on the Ethics of Artificial Intelligence",
        "unesco recommendation on the ethics of artificial intelligence": "UNESCO Recommendation on the Ethics of Artificial Intelligence",
        "oecd ai principles": "OECD Principles on Artificial Intelligence",
        "oecd principles on artificial intelligence": "OECD Principles on Artificial Intelligence",
        "compas": "COMPAS",
        "correctional offender management profiling for alternative sanctions": "COMPAS",
        "mimic-cxr": "MIMIC-CXR",
        "mimic cxr": "MIMIC-CXR",
    }
    CLAUSE_LIKE_VERBS: Set[str] = {
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
    CLAUSE_LIKE_STARTERS: Set[str] = {
        "beyond",
        "especially",
        "even",
        "like",
        "than",
        "versus",
        "vs",
    }

    def __init__(
        self,
        registry: Any,
        output_dir: Optional[Union[Path, str]] = None,
        domain_profile: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.alias_index: Dict[str, str] = {}
        self.alias_index_raw: Dict[str, str] = {}
        self.redirects: Dict[str, str] = {}
        self.decisions: List[LinkDecision] = []
        # Active domain profile (see domain_profiles.py). Callers may pass
        # domain_profile=<DomainProfile> explicitly (e.g. tests exercising
        # a synthetic profile); otherwise resolved from Config.DOMAIN_PROFILE
        # -- this reproduces current AI-ethics behaviour by default, so
        # existing callers see no behaviour change.
        if domain_profile is not None:
            self.domain_profile = domain_profile
        else:
            from domain_profiles import get_active_profile

            self.domain_profile = get_active_profile()

        # Optional, caller-injected {entity URI -> {(discourse_id, chunk_index)}}
        # affirmative chunk participation derived from GROUNDED relation /
        # Evidence provenance (see EnhancedSemanticPipeline.run_canonical_
        # linking). Used only to give the same-chunk cross-type
        # reconciliation pass concrete evidence for an entity whose own
        # metadata carries no span (e.g. a semantic-assertion-derived
        # endpoint). Empty by default -> pass falls back to entity metadata
        # alone and simply does not unify when neither side has a concrete
        # chunk.
        self._entity_chunk_participation: Dict[str, Set[Tuple[str, int]]] = {}

    def set_entity_chunk_participation(self, mapping: Dict[str, Any]) -> None:
        """Merge in {entity URI -> iterable of (discourse_id, chunk_index)}
        concrete chunk participation. Non-int / negative chunk indices and
        blank discourse ids are dropped. Additive: safe to call more than
        once."""
        if not isinstance(mapping, dict):
            return
        for uri, pairs in mapping.items():
            u = str(uri or "").strip()
            if not u or pairs is None:
                continue
            bucket = self._entity_chunk_participation.setdefault(u, set())
            for pair in pairs:
                try:
                    did, ci = pair
                    did = str(did or "").strip()
                    ci = int(ci)
                except (TypeError, ValueError):
                    continue
                if did and ci >= 0:
                    bucket.add((did, ci))

    def run_full_registry(self) -> List[Any]:
        entities = self._get_registry_entities()
        linked = self.link_entities(entities)
        if hasattr(self.registry, "entities"):
            self.registry.entities = {e.uri: e for e in linked if getattr(e, "uri", None)}
        elif hasattr(self.registry, "set_entities"):
            self.registry.set_entities(linked)
        return linked

    @staticmethod
    def _entity_scope(e: Any) -> Optional[str]:
        """Identity scope for entities not yet resolved to a
        genuine antecedent -- e.g. unresolved referential mentions,
        unit-scoped Propositions -- driven purely by a generic
        metadata flag set by the producer, never by label vocabulary. None
        for genuine entities, which keep merging/canonicalising by label
        alone as before."""
        meta = getattr(e, "metadata", None) or {}
        if not meta.get("discourse_scoped_identity"):
            return None
        return str(meta.get("origin_key") or meta.get("discourse_id") or "")

    @staticmethod
    def _member_type_is_uncertain(e: Any) -> bool:
        """General, portable fallback/abstention detector.
        Reads ANY metadata key ending in "_typing_reason" (the convention
        already used by every narrowly-scoped typing stage in this
        pipeline) and treats a value containing "ambiguous" or "abstained"
        as "no confident type opinion" -- never a hardcoded family/label
        list, so it transparently covers any current or future classifier
        following the same convention."""
        meta = getattr(e, "metadata", None) or {}
        for key, value in meta.items():
            if str(key).endswith("_typing_reason"):
                v = str(value or "").lower()
                if "ambiguous" in v or "abstained" in v:
                    return True
        return False

    def _resolve_cluster_type(self, cluster: List[Any]) -> Optional[str]:
        """The type this cluster's members should share, considering only
        CONFIDENT (non-uncertain) member types -- a fallback/abstained
        member never contributes an opinion, so a single confident
        classification elsewhere in the cluster silently outvotes it,
        satisfying the "ambiguous mention vs confident mention" case
        without ever forcing a merge across two independently CONFIDENT
        but incompatible classifications. Returns None when no confident
        member type exists at all (the whole cluster is uncertain) or when
        confident members disagree incompatibly."""
        confident_types = {
            str(getattr(m, "entity_type", "") or "")
            for m in cluster
            if getattr(m, "entity_type", None) and not self._member_type_is_uncertain(m)
        }
        if not confident_types:
            return None
        if len(confident_types) == 1:
            return next(iter(confident_types))
        return _most_specific_compatible_type(confident_types)

    def _resolve_same_referent_cluster_type(self, cluster: List[Any]) -> Optional[str]:
        """Type for a cluster already established to denote ONE referent
        (identical label + concrete same-chunk co-occurrence). Prefer the
        normal compatible-chain resolution; when confident members disagree
        incompatibly, fall back to the structurally most specific confident
        type (deepest declared subclass), broken deterministically by
        member support / confidence / name. Returns None only when no
        member carries a confident type at all."""
        rt = self._resolve_cluster_type(cluster)
        if rt:
            return rt
        confident: List[str] = [
            str(getattr(m, "entity_type", "") or "")
            for m in cluster
            if getattr(m, "entity_type", None) and not self._member_type_is_uncertain(m)
        ]
        if not confident:
            return None
        counts: Dict[str, int] = defaultdict(int)
        best_conf: Dict[str, float] = defaultdict(float)
        for m in cluster:
            t = str(getattr(m, "entity_type", "") or "")
            if t not in confident:
                continue
            counts[t] += 1
            best_conf[t] = max(best_conf[t], float(getattr(m, "confidence", 0.0) or 0.0))
        return max(
            set(confident),
            key=lambda t: (_etd_class_depth(t), counts[t], best_conf[t], t),
        )

    @staticmethod
    def _has_explicit_distinct_referent_evidence(members: List[Any]) -> bool:
        """OPTIONAL forward-compatible safeguard -- not required for
        correctness (concrete same-chunk evidence on BOTH sides is what
        authorises a merge; this only ever BLOCKS one). No producer
        currently emits these keys; the pass is safe without them. True
        when a producer has explicitly recorded that these same-label,
        same-chunk mentions denote DIFFERENT referents: a truthy metadata
        key named ``distinct_referent`` or ending in ``_distinct_referent``
        (the same suffix convention used elsewhere in this pipeline), or a
        ``distinct_referent_from`` list naming another member's URI/label.
        Never a label/type heuristic."""
        uris = {str(getattr(m, "uri", "") or "").strip() for m in members}
        labels = {str(getattr(m, "label", "") or "").strip() for m in members}
        for m in members:
            meta = getattr(m, "metadata", None) or {}
            for key, value in meta.items():
                k = str(key)
                if (k == "distinct_referent" or k.endswith("_distinct_referent")) and bool(value):
                    return True
                if k == "distinct_referent_from":
                    marks = value if isinstance(value, (list, set, tuple)) else [value]
                    for mk in marks:
                        s = str(mk or "").strip()
                        if s and (s in uris or s in labels):
                            return True
        return False

    @staticmethod
    def _metadata_chunk_contexts(e: Any) -> Set[Tuple[str, int]]:
        """Concrete (discourse_id, chunk_index>=0) pairs this entity's OWN
        metadata affirms -- top-level source identity, per-mention records
        and structural source occurrences. A record whose chunk index is
        missing, negative, or explicitly flagged unknown contributes
        nothing (no discourse-level fallback)."""
        out: Set[Tuple[str, int]] = set()
        meta = getattr(e, "metadata", None) or {}

        def _truthy(v: Any) -> bool:
            return str(v).strip().lower() in {"true", "1", "yes"} if not isinstance(v, bool) else v

        def _add(did_raw: Any, ci_raw: Any, unknown_flag: Any = None) -> None:
            did = str(did_raw if did_raw is not None else "").strip()
            if not did or _truthy(unknown_flag):
                return
            try:
                ci = int(ci_raw)
            except (TypeError, ValueError):
                return
            if ci >= 0:
                out.add((did, ci))

        _add(meta.get("discourse_id") or meta.get("doc_id"),
             meta.get("chunk_index"), meta.get("chunk_index_unknown"))
        for bucket_key in ("mentions", "source_occurrences"):
            bucket = meta.get(bucket_key)
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                if isinstance(item, dict):
                    _add(item.get("discourse_id") or item.get("source_doc_id"),
                         item.get("chunk_index"), item.get("chunk_index_unknown"))
        return out

    def _entity_chunk_contexts(self, e: Any) -> Set[Tuple[str, int]]:
        """All concrete (discourse_id, chunk_index) pairs affirmatively
        evidenced for this entity: its own metadata PLUS caller-injected
        grounded-relation/Evidence participation for its URI."""
        ctx = set(self._metadata_chunk_contexts(e))
        uri = str(getattr(e, "uri", "") or "").strip()
        if uri:
            ctx |= self._entity_chunk_participation.get(uri, set())
        return ctx

    def _same_chunk_co_occurrence(self, a: Any, b: Any) -> bool:
        """True ONLY when a concrete (discourse_id, chunk_index) pair is
        affirmatively evidenced for BOTH a and b. Same discourse with an
        unknown / absent chunk on either side is NOT sufficient -- there
        must be a shared concrete chunk both mentions are actually placed
        in."""
        return bool(self._entity_chunk_contexts(a) & self._entity_chunk_contexts(b))

    def _unify_same_chunk_cross_type(self, remainder: List[Any]) -> None:
        """Pass 1.5 -- before typed exact-key grouping. Two no-external-ID
        entities with an identical normalized label and the same identity
        scope that are grounded in the SAME concrete discourse/chunk denote
        one referent unless a producer says otherwise; a mere INFERRED
        coarse-type disagreement between them must not survive as two
        nodes. This only unifies the members' entity_type (+ ontology_uri,
        atomically) onto the resolved type, then lets the unchanged
        exact-key pass fold them into a single canonical entity the normal
        way. Entities that already agree on coarse type, or that carry
        explicit distinct-referent evidence, or that never co-occur in a
        concrete chunk, are left completely untouched."""
        groups: Dict[Tuple[str, Optional[str]], List[Any]] = defaultdict(list)
        for e in remainder:
            key = self._normalized_match_label(getattr(e, "label", "") or "")
            if not key:
                continue
            groups[(key, self._entity_scope(e))].append(e)

        for gkey in sorted(groups.keys(), key=lambda x: (str(x[0]), str(x[1] or ""))):
            members = sorted(groups[gkey], key=self._entity_sort_key)
            if len(members) < 2:
                continue
            # connected components over concrete same-chunk co-occurrence
            parent = list(range(len(members)))

            def _find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if self._same_chunk_co_occurrence(members[i], members[j]):
                        parent[_find(i)] = _find(j)

            comps: Dict[int, List[Any]] = defaultdict(list)
            for idx, m in enumerate(members):
                comps[_find(idx)].append(m)

            for comp in comps.values():
                if len(comp) < 2:
                    continue
                if len({self._coarse_type(m) for m in comp}) < 2:
                    continue  # already one coarse type -- exact-key pass handles it
                if self._has_explicit_distinct_referent_evidence(comp):
                    continue
                resolved = self._resolve_same_referent_cluster_type(comp)
                if not resolved:
                    continue
                for m in comp:
                    if str(getattr(m, "entity_type", "") or "") == resolved:
                        continue
                    meta = getattr(m, "metadata", None)
                    if not isinstance(meta, dict):
                        meta = {}
                        m.metadata = meta
                    meta.setdefault("same_chunk_type_reconciled_from", str(getattr(m, "entity_type", "") or ""))
                    meta["same_chunk_type_reconciled_to"] = resolved
                    self._promote_entity_type(m, resolved)

    def _partition_by_type_compatibility(self, members: List[Any]) -> List[List[Any]]:
        """External-ID equality alone must not silently erase
        an incompatible semantic type disagreement. Members sharing an
        external identity are kept in ONE cluster (to be merged into one
        canonical entity) only when their CONFIDENT type opinions form a
        single compatible ancestor/descendant chain (or there is at most
        one confident opinion at all). Otherwise the group is split by its
        own entity_type -- each resulting sub-cluster still benefits from
        external-ID-based dedup among members that DO agree -- and every
        member is flagged so the external link can be audited as suspect
        rather than silently trusted."""
        if len(members) <= 1:
            return [members]
        confident_types = {
            str(getattr(m, "entity_type", "") or "")
            for m in members
            if getattr(m, "entity_type", None) and not self._member_type_is_uncertain(m)
        }
        if len(confident_types) <= 1 or _most_specific_compatible_type(confident_types) is not None:
            return [members]

        clusters: Dict[str, List[Any]] = defaultdict(list)
        for m in members:
            clusters[str(getattr(m, "entity_type", "") or "")].append(m)
            meta = getattr(m, "metadata", None)
            if isinstance(meta, dict):
                meta["external_id_type_conflict"] = True
                meta["external_id_type_conflict_types"] = sorted(confident_types)
                meta["external_link_suspect"] = True
        return list(clusters.values())

    @staticmethod
    def _promote_entity_type(e: Any, new_type: str) -> None:
        """Atomically move entity_type + ontology_uri together (Task
        9.4.1 section 6) -- no member may retain an ontology_uri
        describing its OLD type after a type promotion. Mirrors the same
        direct entity_type -> ontology_uri lookup the pipeline's own
        ontology-mapping stage uses, so a promoted type is grounded
        exactly as if it had been proposed with that type from the start."""
        old_type = str(getattr(e, "entity_type", "") or "")
        if old_type == new_type:
            return
        meta = getattr(e, "metadata", None)
        if not isinstance(meta, dict):
            meta = {}
            e.metadata = meta
        meta.setdefault("external_id_type_reconciled_from", old_type)
        meta["external_id_type_reconciled_to"] = new_type
        e.entity_type = new_type
        if hasattr(e, "type"):
            e.type = new_type
        mapped = ENTITY_TYPE_TO_ONTOLOGY.get(new_type)
        if mapped:
            # The promoted type has its own specific grounding -- this
            # supersedes whatever ontology_uri the entity carried under
            # its OLD type, which may no longer describe it accurately.
            new_ontology_uri = expand_curie(mapped[0])
            e.ontology_uri = new_ontology_uri
            meta["ontology_uri"] = new_ontology_uri
            meta["ontology_classes"] = [new_ontology_uri]
            meta.pop("ontology_classes_curie", None)
        elif not getattr(e, "ontology_uri", None):
            # No type-specific grounding available for the promoted type,
            # and nothing existing to preserve -- the same generic
            # fallback _ensure_entity_ontology_grounding would assign.
            fallback_uri = expand_curie("prov:Entity")
            e.ontology_uri = fallback_uri
            meta["ontology_uri"] = fallback_uri
            meta["ontology_classes"] = [fallback_uri]
        # else: the promoted type has no dedicated mapping, but the
        # entity already carries a more specific external grounding (e.g.
        # from lexical concept-matching) -- preserve it rather than
        # degrading to the generic prov:Entity fallback.

    def link_entities(self, entities: List[Any]) -> List[Any]:
        if not entities:
            self._persist()
            return []

        ordered_entities = sorted(
            [e for e in entities if e is not None],
            key=self._entity_sort_key,
        )

        # Pass 1: absolute identity by normalized external ID (type-agnostic).
        # This prevents same external entity from leaking into multiple canonical URIs
        # due to coarse-type/path differences.
        external_groups: Dict[str, List[Any]] = defaultdict(list)
        remainder: List[Any] = []
        for e in ordered_entities:
            ext_id = self._external_id(e)
            if ext_id:
                external_groups[ext_id].append(e)
            else:
                remainder.append(e)

        canonicals: List[Any] = []
        for ext_id in sorted(external_groups.keys(), key=lambda x: str(x)):
            members = sorted(external_groups[ext_id], key=self._entity_sort_key)
            # External-ID equality alone must not silently
            # erase an incompatible semantic type disagreement -- split
            # into type-compatible sub-clusters first, each canonicalised
            # independently (still under the same external ID, so members
            # that DO agree keep deduplicating exactly as before).
            for cluster in self._partition_by_type_compatibility(members):
                resolved_type = self._resolve_cluster_type(cluster)
                c = self._pick_canonical(cluster)
                if resolved_type and str(getattr(c, "entity_type", "") or "") != resolved_type:
                    self._promote_entity_type(c, resolved_type)
                ctype = self._coarse_type(c)
                minted = self._mint_canonical_uri(ctype, f"ext:{ext_id}")
                old_can_uri = str(getattr(c, "uri", "") or "")
                c.uri = minted
                if old_can_uri and old_can_uri != minted:
                    self.redirects[old_can_uri] = minted
                for s in cluster:
                    # Confidence-aware reconciliation: promote every
                    # member (not just the canonical) to the resolved
                    # compatible type BEFORE merging, so the registry-level
                    # redirect/merge never has to arbitrate a type
                    # mismatch itself -- entity_type/ontology_uri/URI
                    # bucket stay mutually coherent for every member, not
                    # just the chosen canonical representative.
                    if resolved_type and str(getattr(s, "entity_type", "") or "") != resolved_type:
                        self._promote_entity_type(s, resolved_type)
                    self._register_alias(s, c, "external_id_exact", {"score": 1.0})
                    if s is c:
                        continue
                    self._merge(c, s, "external_id_exact", {"score": 1.0})
                canonicals.append(c)

        # Pass 1.5: same normalized label + same identity scope + concrete
        # same-chunk co-occurrence => one referent. Unify a purely INFERRED
        # coarse-type disagreement between such mentions onto the resolved
        # type BEFORE Pass 2, so the exact-key pass folds them into one
        # canonical entity instead of minting a per-type duplicate node.
        self._unify_same_chunk_cross_type(remainder)

        # Pass 2: typed exact-key grouping for entities without external IDs.
        # Fold in identity scope (None for genuine entities, so
        # they group/mint exactly as before) so an unresolved referential
        # mention or a unit-scoped Proposition from one discourse never
        # lands in the same group -- and never mints the same canonical
        # URI -- as the same-labelled thing from an unrelated discourse.
        groups: Dict[Tuple[str, str, Optional[str]], List[Any]] = defaultdict(list)
        for e in remainder:
            ctype = self._coarse_type(e)
            key = self._normalized_match_label(getattr(e, "label", "") or "")
            groups[(ctype, key, self._entity_scope(e))].append(e)

        for (ctype, key, scope) in sorted(groups.keys(), key=lambda x: (str(x[0]), str(x[1]), str(x[2] or ""))):
            members = sorted(groups[(ctype, key, scope)], key=self._entity_sort_key)
            c = self._pick_canonical(members)
            mint_key = f"{key}|scope:{scope}" if scope else key
            minted = self._mint_canonical_uri(ctype, mint_key)
            old_can_uri = str(getattr(c, "uri", "") or "")
            c.uri = minted
            if old_can_uri and old_can_uri != minted:
                self.redirects[old_can_uri] = minted
            for s in members:
                self._register_alias(s, c, "exact_key", {"score": 1.0})
                if s is c:
                    continue
                self._merge(c, s, "exact_key", {"score": 1.0})
            canonicals.append(c)

        canonicals = self._fuzzy_cluster_and_merge(canonicals)
        self._persist()
        return canonicals

    def rewrite_relation_endpoints(self, all_relations: Iterable[Any]) -> None:
        if not self.redirects:
            return
        for rel in all_relations:
            src = getattr(rel, "source", None)
            tgt = getattr(rel, "target", None)
            if src in self.redirects:
                rel.source = self._resolve_redirect(src)
            if tgt in self.redirects:
                # A redirect key/value is always a genuine
                # internal Entity/Proposition URI (this dict is only ever
                # populated from entity canonicalisation), so once `target`
                # is rewritten to one, the relation's object is
                # unambiguously an entity reference now -- even if `target`
                # had incidentally been carrying literal-typed state (e.g.
                # a pre-canonicalisation entity URI stored as a literal
                # value by an earlier stage). set_target_entity clears that
                # stale literal state instead of leaving it stranded on a
                # now-resolved entity target.
                resolved = self._resolve_redirect(tgt)
                if hasattr(rel, "set_target_entity"):
                    rel.set_target_entity(resolved)
                else:
                    rel.target = resolved

    def _resolve_redirect(self, uri: str) -> str:
        seen: Set[str] = set()
        cur = uri
        while cur in self.redirects and cur not in seen:
            seen.add(cur)
            nxt = self.redirects[cur]
            if not nxt or nxt == cur:
                break
            cur = nxt
        return cur

    def get_report(self, raw_entities: int, canonical_entities: int) -> Dict[str, Any]:
        rows = []
        for e in self._get_registry_entities():
            aliases = (getattr(e, "metadata", {}) or {}).get("aliases", [])
            rows.append({"uri": str(getattr(e, "uri", "")), "alias_count": len(aliases) if isinstance(aliases, list) else 0})
        rows.sort(key=lambda x: (-int(x["alias_count"]), str(x["uri"])))
        return {
            "raw_entities": int(raw_entities),
            "canonical_entities": int(canonical_entities),
            "merges_count": max(0, int(raw_entities) - int(canonical_entities)),
            "top_entities_by_alias_count": rows[:20],
        }

    def _fuzzy_cluster_and_merge(self, entities: List[Any]) -> List[Any]:
        # Union-find clustering per coarse type with capped stabilization loop.
        by_type: Dict[str, List[Any]] = defaultdict(list)
        for e in entities:
            by_type[self._coarse_type(e)].append(e)

        merged_out: List[Any] = []
        for ctype in sorted(by_type.keys(), key=lambda x: str(x)):
            typed = sorted(by_type[ctype], key=self._entity_sort_key)
            index = {str(getattr(e, "uri", "")): i for i, e in enumerate(typed)}
            parent = list(range(len(typed)))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            # bucketed candidate generation (no global O(n^2)).
            buckets: Dict[str, List[int]] = defaultdict(list)
            for i, e in enumerate(typed):
                toks = sorted(list(self._tokenize_for_match(getattr(e, "label", "") or "")))
                if not toks:
                    buckets["_empty"].append(i)
                else:
                    for t in toks[:2]:
                        buckets[t].append(i)

            seen_pairs: Set[Tuple[int, int]] = set()
            for _bucket_key in sorted(buckets.keys(), key=lambda x: str(x)):
                idxs = sorted(set(buckets[_bucket_key]))
                for a_pos in range(len(idxs)):
                    for b_pos in range(a_pos + 1, len(idxs)):
                        ia, ib = idxs[a_pos], idxs[b_pos]
                        p = (min(ia, ib), max(ia, ib))
                        if p in seen_pairs:
                            continue
                        seen_pairs.add(p)
                        a = typed[ia]
                        b = typed[ib]
                        if not self._can_fuzzy_merge(a, b, ctype):
                            continue
                        jacc, lev = self._similarity_scores(str(getattr(a, "label", "")), str(getattr(b, "label", "")))
                        score = (jacc + lev) / 2.0
                        if score >= 0.92:
                            union(ia, ib)

            clusters: Dict[int, List[Any]] = defaultdict(list)
            for i, e in enumerate(typed):
                clusters[find(i)].append(e)

            for root in sorted(clusters.keys()):
                members = sorted(clusters[root], key=self._entity_sort_key)
                c = self._pick_canonical(members)
                key = self._external_id(c) or self._normalized_match_label(getattr(c, "label", "") or "")
                # _can_fuzzy_merge already refuses to union entities with
                # different scopes, so every member of this cluster shares
                # one scope -- safe to read it from any one of them.
                scope = self._entity_scope(c)
                mint_key = f"{key}|scope:{scope}" if scope else key
                minted = self._mint_canonical_uri(ctype, mint_key)
                old = str(getattr(c, "uri", "") or "")
                c.uri = minted
                if old and old != minted:
                    self.redirects[old] = minted
                for s in members:
                    self._register_alias(s, c, "fuzzy_cluster", {"score": 0.92})
                    if s is c:
                        continue
                    self._merge(c, s, "fuzzy_cluster", {"score": 0.92})
                merged_out.append(c)

        return merged_out

    def _mint_canonical_uri(self, coarse_type: str, key: str) -> str:
        base = Config.NAMESPACES["etd"].rstrip("#/")
        ctype = self._slug(coarse_type) or "entity"
        label_slug = self._slug(key) or "item"
        h = hashlib.sha1(f"{coarse_type}|{key}".encode("utf-8")).hexdigest()[:8]
        return f"{base}/entity/{ctype}/{label_slug}-{h}"

    def _pick_canonical(self, group: List[Any]) -> Any:
        if len(group) == 1:
            return group[0]
        return max(
            group,
            key=lambda x: (
                self._label_preference_key(str(getattr(x, "label", "") or "")),
                float(getattr(x, "confidence", 0.0) or 0.0),
                str(getattr(x, "label", "") or ""),
                str(getattr(x, "uri", "") or ""),
            ),
        )

    @staticmethod
    def _entity_sort_key(e: Any) -> Tuple[str, str, str]:
        return (
            str(getattr(e, "uri", "") or ""),
            str(getattr(e, "entity_type", "") or ""),
            str(getattr(e, "label", "") or ""),
        )

    def _register_alias(self, source: Any, canonical: Any, rule: str, scores: Dict[str, float]) -> None:
        raw_label = str(getattr(source, "label", "") or "")
        norm_label = self._normalized_match_label(raw_label)
        can_uri = str(getattr(canonical, "uri", "") or "")
        if norm_label:
            self.alias_index[norm_label] = can_uri
        if raw_label:
            self.alias_index_raw[raw_label] = can_uri
        self.decisions.append(
            LinkDecision(
                input_label=raw_label,
                norm_label=norm_label,
                chosen_canonical_uri=can_uri,
                rule=rule,
                scores={k: float(v) for k, v in scores.items()},
                source_ids=[str(getattr(source, "uri", "") or ""), can_uri],
            )
        )

    def _merge(self, canonical: Any, source: Any, rule: str, scores: Dict[str, float]) -> None:
        canonical.metadata = canonical.metadata or {}
        source.metadata = source.metadata or {}

        aliases = canonical.metadata.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        for lbl in (getattr(canonical, "label", None), getattr(source, "label", None)):
            if lbl and lbl not in aliases:
                aliases.append(lbl)
        canonical.metadata["aliases"] = aliases

        source_ids = canonical.metadata.get("source_ids")
        if not isinstance(source_ids, list):
            source_ids = []
        for sid in (str(getattr(source, "uri", "") or ""), str(getattr(canonical, "uri", "") or "")):
            if sid and sid not in source_ids:
                source_ids.append(sid)
        canonical.metadata["source_ids"] = source_ids

        self._safe_merge_metadata(canonical.metadata, source.metadata)

        if getattr(source, "ontology_uri", None) and not getattr(canonical, "ontology_uri", None):
            canonical.ontology_uri = source.ontology_uri

        if float(getattr(source, "confidence", 0.0) or 0.0) > float(getattr(canonical, "confidence", 0.0) or 0.0):
            canonical.confidence = source.confidence
            if getattr(source, "label", None) and self._label_preference_key(
                str(getattr(source, "label", "") or "")
            ) > self._label_preference_key(str(getattr(canonical, "label", "") or "")):
                canonical.label = source.label

        old_uri = str(getattr(source, "uri", "") or "")
        new_uri = str(getattr(canonical, "uri", "") or "")
        if old_uri and new_uri and old_uri != new_uri:
            self.redirects[old_uri] = new_uri
            if hasattr(self.registry, "replace_entity"):
                self.registry.replace_entity(old_uri, new_uri)

    def _normalize_mention(self, mention: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(mention, dict):
            return None
        did_raw = mention.get("discourse_id")
        did = str(did_raw).strip() if did_raw is not None else ""
        if not did:
            return None
        try:
            start = int(mention.get("start_char"))
            end = int(mention.get("end_char"))
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start:
            return None
        out: Dict[str, Any] = {
            "discourse_id": did,
            "start_char": int(start),
            "end_char": int(end),
        }
        try:
            ci = int(mention.get("chunk_index"))
            if ci >= 0:
                out["chunk_index"] = int(ci)
        except (TypeError, ValueError):
            pass
        return out

    def _merge_mentions(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        merged: List[Dict[str, Any]] = []
        for bucket in (target.get("mentions"), source.get("mentions")):
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                norm = self._normalize_mention(item)
                if norm is not None:
                    merged.append(norm)
        if not merged:
            return
        dedup: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
        for m in merged:
            key = (
                str(m.get("discourse_id", "")),
                int(m.get("chunk_index")) if m.get("chunk_index") is not None else -1,
                int(m.get("start_char", -1)),
                int(m.get("end_char", -1)),
            )
            dedup[key] = m
        target["mentions"] = [dedup[k] for k in sorted(dedup.keys())]

    def _safe_merge_metadata(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        self._merge_mentions(target, source or {})
        for k, v in (source or {}).items():
            if k == "mentions":
                continue
            if v is None:
                continue
            if k not in target or target[k] is None:
                target[k] = v
                continue
            tv = target[k]
            if isinstance(tv, dict) and isinstance(v, dict):
                self._safe_merge_metadata(tv, v)
            elif isinstance(tv, list):
                incoming = v if isinstance(v, list) else [v]
                for item in incoming:
                    if item not in tv:
                        tv.append(item)
            elif tv != v:
                target[k] = [tv, v] if not isinstance(tv, list) else tv + ([v] if v not in tv else [])

    def _can_fuzzy_merge(self, a: Any, b: Any, coarse_type: str) -> bool:
        if self._coarse_type(a) != coarse_type or self._coarse_type(b) != coarse_type:
            return False
        if self._entity_scope(a) != self._entity_scope(b):
            return False
        if coarse_type in self.STRUCTURAL_TYPES:
            return False
        la = self._normalized_match_label(getattr(a, "label", "") or "")
        lb = self._normalized_match_label(getattr(b, "label", "") or "")
        if not la or not lb:
            return False
        if self._is_coordinated_superset_pair(la, lb):
            return False
        if re.fullmatch(r"(theme|question|answer|vote|expert|document|discourseunit)\s*[_-]?\s*\d+", la):
            return False
        if re.fullmatch(r"(theme|question|answer|vote|expert|document|discourseunit)\s*[_-]?\s*\d+", lb):
            return False
        if la in self.DO_NOT_MERGE or lb in self.DO_NOT_MERGE:
            return False
        # Domain-specific blocked terms (e.g. AI-ethics vocabulary) come
        # from the active domain profile, not a hardcoded list here -- a
        # different profile can supply an entirely different blocklist
        # without any change to this file.
        if self.domain_profile.matches_gazetteer("generic_merge_blocklist", la) or self.domain_profile.matches_gazetteer("generic_merge_blocklist", lb):
            return False
        if len(la) < 3 or len(lb) < 3:
            return False
        if self._is_bare_generic_label(la) != self._is_bare_generic_label(lb):
            # Root-cause fix (manual semantic-QA pass, category E): one
            # side is a bare generic noun (e.g. "prediction"/"the
            # prediction" -- at most one content word once a leading
            # determiner is stripped), the other a longer, more specific
            # phrase (e.g. "Performativity Of Predictions"). Lexical
            # overlap/similarity alone (a shared root, a high Jaccard or
            # sequence-matcher score) must not be treated as identity
            # evidence across discourses in this asymmetric case --
            # generalizes DO_NOT_MERGE's fixed word list into a shape
            # rule, since a generic noun's exact spelling can't be
            # enumerated in advance. A genuine repeat of the SAME named
            # entity across discourses (both sides specific, or both
            # sides equally generic) is unaffected.
            return False

        a_ext = self._external_id(a)
        b_ext = self._external_id(b)
        if a_ext and b_ext and a_ext != b_ext:
            return False

        a_ont = str(getattr(a, "ontology_uri", "") or (getattr(a, "metadata", {}) or {}).get("ontology_uri") or "")
        b_ont = str(getattr(b, "ontology_uri", "") or (getattr(b, "metadata", {}) or {}).get("ontology_uri") or "")
        if a_ont and b_ont and a_ont != b_ont:
            return False

        # C.1: even at/above the similarity threshold, a containment pair
        # whose longer label adds a material content token (qualifier /
        # modifier / extra head noun -- broader vs narrower, population vs
        # subgroup, "non-"/opposite, generic vs qualified) is a DIFFERENT
        # referent unless an independent signal says otherwise. Mirrors
        # EntityConsolidator._should_merge_fuzzy. The override for "shared
        # external identity" uses an AUTHORITATIVE id only (DOI /
        # authoritative URL / explicit external_id / wikidata_authoritative
        # =True) -- a shared heuristic Wikidata QID is not enough on its
        # own to force a broader/narrower merge.
        if la != lb and (la in lb or lb in la):
            short_key, long_key = (la, lb) if len(la) <= len(lb) else (lb, la)
            auth_a = self._authoritative_external_id(a)
            auth_b = self._authoritative_external_id(b)
            if (
                len(short_key) >= 4
                and len(long_key) >= 4
                and containment_adds_material_token(short_key, long_key)
                and not (auth_a and auth_b and auth_a == auth_b)
                and not is_initialism_pair(getattr(a, "label", "") or "", getattr(b, "label", "") or "")
            ):
                return False
        return True

    @staticmethod
    def _authoritative_external_id(e: Any) -> str:
        """See EntityConsolidator._authoritative_external_id -- an ASSERTED
        external identity, not a heuristic Wikidata alignment."""
        meta = getattr(e, "metadata", {}) or {}
        for key in ("doi", "url", "external_id"):
            v = str(meta.get(key) or "").strip().lower()
            if v:
                return v
        if bool(meta.get("wikidata_authoritative")):
            for key in ("wikidata_uri", "wikidata_id", "qid"):
                v = str(meta.get(key) or "").strip().lower()
                if v:
                    return v
        return ""

    @staticmethod
    def _is_coordinated_superset_pair(left_key: str, right_key: str) -> bool:
        left = str(left_key or "").strip()
        right = str(right_key or "").strip()
        if not left or not right:
            return False
        longer, shorter = (left, right) if len(left) >= len(right) else (right, left)
        if not re.search(r"\b(?:and|or)\b", longer):
            return False
        parts = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"\s+(?:and|or)\s+", longer)
            if part.strip()
        ]
        return shorter in parts

    def _external_id(self, e: Any) -> str:
        meta = getattr(e, "metadata", {}) or {}
        candidates: List[Any] = [
            meta.get("external_id"),
            meta.get("wikidata_uri"),
            meta.get("doi"),
            meta.get("url"),
            meta.get("qid"),
            meta.get("wikidata_id"),
        ]

        wd_obj = meta.get("wikidata")
        if isinstance(wd_obj, dict):
            candidates.append(wd_obj.get("qid"))
            candidates.append(wd_obj.get("id"))
            candidates.append(wd_obj.get("uri"))

        wd_match = meta.get("wikidata_match")
        if isinstance(wd_match, dict):
            candidates.append(wd_match.get("qid"))
            candidates.append(wd_match.get("id"))
            candidates.append(wd_match.get("uri"))

        for v in candidates:
            norm = self._normalize_external_id(v)
            if norm:
                return norm
        return ""

    def _normalize_external_id(self, value: Any) -> str:
        if value is None:
            return ""
        s = str(value).strip()
        if not s:
            return ""

        # Wikidata normalization to a stable token.
        m = re.search(r"(?:^|/|:)(Q\d+)$", s, flags=re.IGNORECASE)
        if m:
            return f"wikidata:{m.group(1).upper()}"

        if "wikidata.org/entity/" in s.lower() or "wikidata.org/wiki/" in s.lower():
            tail = s.rstrip("/").split("/")[-1]
            if re.fullmatch(r"Q\d+", tail, flags=re.IGNORECASE):
                return f"wikidata:{tail.upper()}"

        # DOI normalization.
        if s.lower().startswith("doi:"):
            return f"doi:{s[4:].strip().lower()}"
        if "doi.org/" in s.lower():
            doi_tail = s.split("doi.org/", 1)[-1].strip().lower()
            if doi_tail:
                return f"doi:{doi_tail}"

        # URL normalization.
        if s.startswith("http://") or s.startswith("https://"):
            return self._normalize_url(s)

        return s.casefold()

    def _coarse_type(self, e: Any) -> str:
        ont = str(getattr(e, "ontology_uri", "") or "").strip()
        if ont and self._is_class_uri(ont):
            local = ont.split("#")[-1].split("/")[-1]
            if local:
                return local
        et = str(getattr(e, "entity_type", "Entity") or "Entity").strip()
        return et if et else "Entity"

    def _is_class_uri(self, uri: str) -> bool:
        if not uri:
            return False
        for ns in self.CLASS_NS_PREFIXES:
            if ns and uri.startswith(ns):
                return True
        return False

    _GENERIC_LABEL_LEADING_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those"}

    def _is_bare_generic_label(self, normalized_label: str) -> bool:
        """True when a normalized label reduces to at most one content
        token once a leading determiner is stripped (e.g. "prediction",
        "the prediction", "a model") -- used to keep a bare generic noun
        from fuzzy-merging with an unrelated, more specific multi-word
        concept purely on lexical overlap. See _can_fuzzy_merge."""
        tokens = [t for t in str(normalized_label or "").split() if t]
        if tokens and tokens[0] in self._GENERIC_LABEL_LEADING_DETERMINERS:
            tokens = tokens[1:]
        return len(tokens) <= 1

    def _normalized_match_label(self, label: str) -> str:
        s = self._normalize_text(label)
        if s in self.ALIAS_TABLE:
            s = self._normalize_text(self.ALIAS_TABLE[s])
        return s

    def _normalize_text(self, text: str) -> str:
        s = unicodedata.normalize("NFKC", str(text or ""))
        s = s.strip()
        if not s:
            return ""
        if s.startswith("http://") or s.startswith("https://"):
            s = self._normalize_url(s)
        s = s.casefold()
        s = re.sub(r"[\u2013\u2014]", "-", s)
        s = re.sub(r"[^\w\s:/.-]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _label_is_clause_like(self, label: str) -> bool:
        tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", str(label or ""))
        if not tokens:
            return True
        norm_tokens = [self._normalize_text(tok) for tok in tokens]
        if norm_tokens[0] in self.CLAUSE_LIKE_STARTERS:
            return True
        if any(tok in self.CLAUSE_LIKE_VERBS for tok in norm_tokens):
            return True
        norm_label = self._normalize_text(label)
        if re.search(r"\b(?:that|which|who)\b", norm_label, flags=re.IGNORECASE):
            return True
        return False

    def _label_preference_key(self, label: str) -> Tuple[int, int, int, str]:
        tokens = re.findall(r"[A-Za-z0-9][\w'/-]*", str(label or ""))
        if not tokens:
            return (0, 0, 0, "")
        clean = 0 if self._label_is_clause_like(label) else 1
        informative = 1 if len(tokens) > 1 else 0
        return (
            clean,
            informative,
            len(tokens),
            self._normalize_text(label),
        )

    def _normalize_url(self, raw_url: str) -> str:
        p = urlparse(raw_url.strip())
        scheme = p.scheme.lower() or "https"
        netloc = p.netloc.lower()
        path = re.sub(r"/+$", "", p.path or "")
        query_pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if not k.lower().startswith("utm_")]
        query = urlencode(query_pairs)
        return urlunparse((scheme, netloc, path, "", query, ""))

    def _tokenize_for_match(self, label: str) -> Set[str]:
        s = self._normalized_match_label(label)
        toks = [t for t in re.split(r"[\s_/:-]+", s) if t]
        if self._looks_organization_label(label):
            toks = [t for t in toks if t not in self.ORG_SUFFIXES]
        return set(toks)

    def _looks_organization_label(self, label: str) -> bool:
        s = self._normalize_text(label)
        return any(re.search(rf"\b{re.escape(suf)}\.?$", s) for suf in self.ORG_SUFFIXES)

    def _similarity_scores(self, a: str, b: str) -> Tuple[float, float]:
        ta = self._tokenize_for_match(a)
        tb = self._tokenize_for_match(b)
        if not ta or not tb:
            return 0.0, 0.0
        jaccard = len(ta.intersection(tb)) / len(ta.union(tb))
        lev = SequenceMatcher(None, self._normalized_match_label(a), self._normalized_match_label(b)).ratio()
        return jaccard, lev

    def _slug(self, s: str) -> str:
        t = self._normalize_text(s)
        t = re.sub(r"[^a-z0-9]+", "-", t)
        t = re.sub(r"-+", "-", t).strip("-")
        return t

    def _persist(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        alias_payload = {
            "normalized": dict(sorted(self.alias_index.items())),
            "raw": dict(sorted(self.alias_index_raw.items())),
        }
        (self.output_dir / "alias_map.json").write_text(
            json.dumps(alias_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (self.output_dir / "canonical_linking_log.jsonl").open("w", encoding="utf-8") as f:
            ordered_decisions = sorted(
                self.decisions,
                key=lambda d: (
                    d.chosen_canonical_uri,
                    d.norm_label,
                    d.input_label,
                    d.rule,
                    "|".join(d.source_ids),
                ),
            )
            for d in ordered_decisions:
                f.write(
                    json.dumps(
                        {
                            "input_label": d.input_label,
                            "norm_label": d.norm_label,
                            "chosen_canonical_uri": d.chosen_canonical_uri,
                            "rule": d.rule,
                            "scores": d.scores,
                            "source_ids": d.source_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _get_registry_entities(self) -> List[Any]:
        get_all = getattr(self.registry, "get_all_entities", None)
        if callable(get_all):
            return list(get_all())
        store = getattr(self.registry, "entities", None)
        if isinstance(store, dict):
            return list(store.values())
        if isinstance(store, list):
            return list(store)
        return []


