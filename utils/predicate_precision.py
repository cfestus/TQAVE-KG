"""
Deterministic predicate precision layer.

Runs after extraction/canonicalization and before reasoning/RDF.
Returns NEW Relation instances (no in-place mutation).
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from models import Entity, Relation
from utils.predicate_map import (
    is_malformed_local_predicate,
    normalize_local_predicate,
    normalize_predicate_strict,
)
from utils.semantic_relation_policy import PREDICATE_REGISTRY, SemanticRelationPolicy

logger = logging.getLogger(__name__)


# Predicates that SemanticRelationPolicy treats as generic relation fallbacks.
# A raw predicate may only normalize to one of these when it is itself an
# explicitly registered generic phrase (see PredicatePrecisionLayer._normalize_candidate).
_GENERIC_FALLBACK_PREDICATES = {"relatedTo", "associatedWith"}


@dataclass(frozen=True)
class PredicateSpec:
    domain_types: Set[str]
    range_types: Set[str]
    min_confidence: float


def _to_coarse_type(raw: Any) -> str:
    s = str(raw or "").strip()
    return s if s else "Entity"


def _infer_type_from_uri(uri: str) -> str:
    s = str(uri or "").lower()
    if (
        "/entity/person/expert-" in s
        or "/entity/person/expert/" in s
        or "#entity/person/expert-" in s
        or "#entity/person/expert/" in s
    ):
        return "Expert"
    if "#expert_" in s or "/expert_" in s:
        return "Expert"
    if "/expert/" in s:
        return "Expert"
    if "#answer_" in s or "/answer_" in s:
        return "Answer"
    if "/answer/" in s:
        return "Answer"
    if "#question_" in s or "/question_" in s:
        return "Question"
    if "/question/" in s:
        return "Question"
    if "#theme_" in s or "/theme_" in s:
        return "Theme"
    if "/theme/" in s:
        return "Theme"
    if "#vote_" in s or "/vote_" in s:
        return "Vote"
    if "/vote/" in s:
        return "Vote"
    if "#document_" in s or "/document_" in s:
        return "Document"
    if "/document/" in s:
        return "Document"
    if "#discourse_" in s or "/discourse_" in s:
        return "DiscourseUnit"
    return "Entity"


class PredicatePrecisionLayer:
    def __init__(self, semantic_policy: Optional[SemanticRelationPolicy] = None) -> None:
        self.policy = semantic_policy or SemanticRelationPolicy()
        any_t = {"*"}
        policy_targets = {"Standard", "Law", "Regulation", "PolicyFramework", "LegalRequirement", "RegulatoryReference"}
        dataset_targets = {"Dataset", "DataSource", "Corpus"}

        self.specs: Dict[str, PredicateSpec] = {
            "relatedTo": PredicateSpec(any_t, any_t, 0.30),
            "associatedWith": PredicateSpec(any_t, any_t, 0.35),
            "mentions": PredicateSpec({"Theme", "Question", "Answer", "Document", "DiscourseUnit"}, any_t, 0.20),
            "uses": PredicateSpec(any_t, any_t, 0.45),
            "trainedOn": PredicateSpec(any_t, dataset_targets, 0.50),
            "reflects": PredicateSpec(any_t, any_t, 0.50),
            "hasRisk": PredicateSpec(any_t, any_t, 0.50),
            "exhibitsRisk": PredicateSpec(any_t, any_t, 0.50),
            "causes": PredicateSpec(any_t, any_t, 0.55),
            "causedBy": PredicateSpec(any_t, any_t, 0.55),
            "affects": PredicateSpec(any_t, any_t, 0.50),
            "affectedBy": PredicateSpec(any_t, any_t, 0.50),
            "particularlyAffects": PredicateSpec(any_t, any_t, 0.55),
            "partOfDomain": PredicateSpec(any_t, any_t, 0.45),
            "mitigatedBy": PredicateSpec(any_t, any_t, 0.55),
            "mitigates": PredicateSpec(any_t, any_t, 0.55),
            "supports": PredicateSpec(any_t, any_t, 0.50),
            "criticizes": PredicateSpec(any_t, any_t, 0.50),
            "endorses": PredicateSpec(any_t, any_t, 0.50),
            "disagreesWith": PredicateSpec(any_t, any_t, 0.50),
            "opposes": PredicateSpec(any_t, any_t, 0.50),
            "compliesWith": PredicateSpec(any_t, policy_targets, 0.55),
            "violates": PredicateSpec(any_t, policy_targets, 0.55),
            "requiresComplianceWith": PredicateSpec(any_t, policy_targets, 0.55),
            "requires": PredicateSpec(any_t, any_t, 0.50),
            "referencesInstrument": PredicateSpec(any_t, policy_targets, 0.45),
            "referencesSource": PredicateSpec(any_t, any_t, 0.45),
            "cites": PredicateSpec(any_t, any_t, 0.45),
            "answersQuestion": PredicateSpec({"Answer"}, {"Question"}, 0.70),
            "authoredBy": PredicateSpec({"Answer"}, {"Expert"}, 0.70),
            "askedBy": PredicateSpec({"Question"}, {"Expert"}, 0.70),
            "createdBy": PredicateSpec({"Theme"}, {"Expert"}, 0.70),
            "belongsToTheme": PredicateSpec({"Question"}, {"Theme"}, 0.70),
            "castBy": PredicateSpec({"Vote"}, {"Expert"}, 0.70),
            "receivesVote": PredicateSpec({"Answer"}, {"Vote"}, 0.70),
        }
        self.allowed: Set[str] = set(self.specs.keys())
        self.upgrade_blocked_contexts: Set[str] = {
            "cooccurrence_chunk",
            "lightweight_mentions",
            "dataset_vote",
            "structural",
        }
        self.semantic_predicates: Set[str] = set(self.policy.semantic_predicates)
        self.structural_predicates: Set[str] = {
            "mentions",
            "answersQuestion",
            "authoredBy",
            "askedBy",
            "createdBy",
            "belongsToTheme",
            "castBy",
            "receivesVote",
        }
        self.allowed.update(self.semantic_predicates)
        self.allowed.update(self.structural_predicates)

    def refine_relations(
        self,
        relations: List[Relation],
        entities: Iterable[Entity],
        discourse_text: Optional[str] = None,
    ) -> List[Relation]:
        type_by_uri: Dict[str, str] = {}
        label_by_uri: Dict[str, str] = {}
        entity_by_uri: Dict[str, Entity] = {}
        for e in entities:
            u = str(getattr(e, "uri", "") or "")
            if not u:
                continue
            entity_by_uri[u] = e
            type_by_uri[u] = _to_coarse_type(getattr(e, "entity_type", "Entity"))
            label_by_uri[u] = str(getattr(e, "label", "") or "")

        strong_pairs: Set[Tuple[str, str]] = set()
        for r in relations:
            p = self._normalize_candidate(str(getattr(r, "relation", "")))
            if p not in {"relatedTo", "mentions"}:
                strong_pairs.add((str(r.source), str(r.target)))
                strong_pairs.add((str(r.target), str(r.source)))

        out: List[Relation] = []
        for rel in relations:
            src = str(rel.source)
            tgt = str(rel.target)
            src_t = type_by_uri.get(src) or _infer_type_from_uri(src)
            tgt_t = type_by_uri.get(tgt) or _infer_type_from_uri(tgt)
            src_entity = entity_by_uri.get(src)
            tgt_entity = entity_by_uri.get(tgt)
            meta = copy.deepcopy(getattr(rel, "metadata", {}) or {})

            raw_pred = str(getattr(rel, "relation", "") or "")
            pred = self._normalize_candidate(raw_pred)
            rel_target_is_literal = bool(getattr(rel, "target_is_literal", False)) or getattr(rel, "target_literal", None) is not None
            allow_local_predicate = bool(meta.get("allow_local_predicate"))
            meta.setdefault(
                "relation_kind",
                self._infer_relation_kind(
                    meta=meta,
                    pred=pred,
                    is_inferred=bool(getattr(rel, "is_inferred", False)),
                ),
            )
            if pred not in self.allowed:
                if allow_local_predicate and raw_pred.strip():
                    # E: deterministic local-predicate cleanup. Casing /
                    # punctuation / a single conservative regular-inflection
                    # fold only -- never aggressive stemming, never merging
                    # predicates that differ by a content word.
                    cleaned = normalize_local_predicate(raw_pred)
                    if is_malformed_local_predicate(cleaned):
                        # Deterministically not a valid relation predicate
                        # (function-word-only / empty). Drop rather than
                        # serialize a contentless edge.
                        self._log(meta, raw_pred, "", "reject:malformed_local_predicate")
                        logger.warning(
                            "Dropping relation %s: predicate %r is not a valid relation predicate "
                            "(function-word-only / empty)",
                            getattr(rel, "uri", "") or f"{src}->{tgt}",
                            raw_pred,
                        )
                        continue
                    # A cleaned form may now match an already-established
                    # canonical alias (no threshold change -- same strict
                    # table). Honour that mapping only when the relation's
                    # existing object representation stays valid under the
                    # canonical predicate; otherwise keep the cleaned local
                    # predicate rather than losing the relation.
                    remapped = normalize_predicate_strict(cleaned)
                    if (
                        remapped
                        and remapped != "relatedTo"
                        and remapped in self.allowed
                        and self._object_state_valid_for_predicate(remapped, rel)
                    ):
                        self._log(meta, raw_pred, remapped, "normalize:local_alias_to_canonical")
                        meta["predicate_mapping_status"] = "mapped"
                        pred = remapped
                    else:
                        if cleaned != raw_pred.strip():
                            self._log(meta, raw_pred, cleaned, "normalize:local_predicate_alias")
                        pred = cleaned
                        meta["predicate_mapping_status"] = "local_unmapped"
                else:
                    self._log(meta, raw_pred, "relatedTo", "downgrade:unknown_predicate")
                    pred = "relatedTo"

            # Type validation BEFORE evidence upgrade.
            if pred in self.allowed and not self._type_valid(
                pred, src_t, tgt_t, source_entity=src_entity, target_entity=tgt_entity,
                target_is_literal=rel_target_is_literal,
            ):
                downgraded = "mentions" if pred == "mentions" else "relatedTo"
                self._log(meta, pred, downgraded, f"type_guard:{src_t}->{tgt_t}")
                pred = downgraded

            evidence = str(meta.get("evidence") or getattr(rel, "context", None) or discourse_text or "")
            upgraded_pred, upgrade_rule = self._candidate_upgrade(evidence)
            if pred in {"relatedTo", "mentions"} and upgraded_pred and upgraded_pred in self.allowed:
                # Upgrade only when upgraded predicate passes type checks and is supported
                # by the local evidence for the actual source/target pair.
                if (
                    self._upgrade_allowed(meta)
                    and self._evidence_mentions_pair(evidence, label_by_uri.get(src, ""), label_by_uri.get(tgt, ""))
                    and self._type_valid(
                        upgraded_pred, src_t, tgt_t, source_entity=src_entity, target_entity=tgt_entity,
                        target_is_literal=rel_target_is_literal,
                    )
                    and self._upgrade_locally_supported(upgraded_pred, src_entity, tgt_entity, evidence)
                ):
                    self._log(meta, pred, upgraded_pred, f"upgrade:{upgrade_rule}")
                    pred = upgraded_pred

            # drop weak co-occurrence if pair already has stronger relation
            context_source = str(meta.get("context_source", ""))
            if pred == "relatedTo" and context_source == "cooccurrence_chunk" and (src, tgt) in strong_pairs:
                self._log(meta, pred, pred, "drop:cooccurrence_shadowed")
                continue

            # Defect-2 residual fix: revalidate the object representation
            # against whatever predicate this relation ends up with. A
            # literal object that was valid under an earlier predicate is
            # not automatically valid under a reassigned one -- and this
            # layer has no entity-minting capability to correct it, so the
            # deterministic, non-weakening action is to drop the relation
            # rather than serialize an invalid predicate/object combination.
            if not self._object_state_valid_for_predicate(pred, rel):
                self._log(meta, pred, pred, "reject:literal_object_invalid_for_reassigned_predicate")
                logger.warning(
                    "Dropping relation %s: predicate '%s' does not permit a literal object "
                    "(object representation was not revalidated after reassignment)",
                    getattr(rel, "uri", "") or f"{src}->{tgt}",
                    pred,
                )
                continue

            conf = float(getattr(rel, "confidence", 0.0) or 0.0)
            conf2 = self._adjust_confidence(conf, pred, meta)

            out.append(
                Relation(
                    uri=getattr(rel, "uri", ""),
                    source=src,
                    relation=pred,
                    target=tgt,
                    confidence=conf2,
                    ontology_uri=getattr(rel, "ontology_uri", None),
                    is_inferred=bool(getattr(rel, "is_inferred", False)),
                    context=getattr(rel, "context", None),
                    doc_id=getattr(rel, "doc_id", None),
                    metadata=meta,
                    # Preserve literal-object state verbatim -- this layer
                    # only refines the predicate, it must never silently
                    # reclassify an already-correct literal/entity endpoint
                    # by omitting these fields from the rebuilt Relation.
                    target_literal=getattr(rel, "target_literal", None),
                    target_is_literal=bool(getattr(rel, "target_is_literal", False)),
                    target_datatype=getattr(rel, "target_datatype", None),
                    object_type=getattr(rel, "object_type", None),
                )
            )
        return out

    @staticmethod
    def _object_state_valid_for_predicate(pred: str, rel: Relation) -> bool:
        """Defect-2 residual fix: this layer can reassign a relation's
        predicate (type-guard downgrade to relatedTo/mentions, or an
        evidence-based upgrade) without ever touching its object
        representation, which is preserved verbatim by design (see the
        comment on the Relation(...) construction below). That is correct
        when the object representation stays valid for whatever predicate
        the relation ends up with -- it silently stopped being correct
        whenever the FINAL predicate turned out to be declared object/
        entity-only (literal_allowed=False in PREDICATE_REGISTRY) while the
        object was still literal-valued from a PRIOR, literal-permitting
        predicate. Checked once, on the final pred, right before a relation
        is emitted -- covers every reassignment path uniformly (including a
        relation that arrived already mismatched, without needing a
        separate check duplicated at each individual reassignment site).
        Predicates absent from PREDICATE_REGISTRY (local/unmapped) are left
        unconstrained, matching this registry's existing default-off
        semantics elsewhere.
        """
        if not bool(getattr(rel, "target_is_literal", False)) and getattr(rel, "target_literal", None) is None:
            return True
        entry = PREDICATE_REGISTRY.get(pred)
        if entry is None:
            return True
        return bool(entry.literal_allowed)

    def _normalize_candidate(self, raw_pred: str) -> str:
        raw = str(raw_pred or "").strip()
        if not raw:
            return "relatedTo"
        normalized = self.policy.canonicalize_predicate(raw)
        if normalized and normalized not in _GENERIC_FALLBACK_PREDICATES:
            return normalized
        # `normalized` is either empty or a generic fallback predicate.
        # Only trust that fallback when the raw predicate itself is an
        # explicitly registered generic phrase (e.g. "related to", "linked
        # to", "associated with"). Otherwise this is simply an unrecognized
        # predicate, and returning "" here (instead of silently defaulting
        # to relatedTo/associatedWith) lets refine_relations() route it
        # through the local-unmapped-predicate path so specific-but-
        # unregistered verbs are preserved rather than erased.
        strict = normalize_predicate_strict(raw)
        if strict in _GENERIC_FALLBACK_PREDICATES:
            return strict
        return ""

    def _type_valid(
        self,
        pred: str,
        src_t: str,
        tgt_t: str,
        *,
        source_entity: Optional[Entity] = None,
        target_entity: Optional[Entity] = None,
        target_is_literal: bool = False,
    ) -> bool:
        # Defect-2 residual fix: a literal-valued object always has
        # target_entity=None (there is no Entity to resolve), which
        # previously fell through to the entity-only branches below and
        # returned False unconditionally for ANY predicate not in
        # self.specs -- including hasDescription/hasAttribute/
        # hasSubjectDescription, which are genuinely literal_allowed=True.
        # That forced an incorrect type-guard downgrade to relatedTo/
        # mentions for a legitimately literal-valued relation, which then
        # carried its (still valid-for-the-old-predicate) literal object
        # into relatedTo/mentions, silently violating THEIR declared
        # object-only kind. Consult the registry first for any literal
        # object so the genuinely-literal case is judged by policy, not by
        # the entity-presence side effect of being literal-valued.
        if target_is_literal:
            entry = PREDICATE_REGISTRY.get(pred)
            if entry is not None:
                return bool(entry.literal_allowed)
        if pred in self.semantic_predicates and source_entity is not None and target_entity is not None:
            probe = Relation(
                source=str(getattr(source_entity, "uri", "") or ""),
                relation=pred,
                target=str(getattr(target_entity, "uri", "") or ""),
                confidence=1.0,
                metadata={},
            )
            return self.policy.domain_range_valid(
                probe,
                {
                    str(getattr(source_entity, "uri", "") or ""): source_entity,
                    str(getattr(target_entity, "uri", "") or ""): target_entity,
                },
            )
        spec = self.specs.get(pred)
        if spec is None:
            return False
        return self._match(spec.domain_types, src_t) and self._match(spec.range_types, tgt_t)

    @staticmethod
    def _match(allowed: Set[str], actual: str) -> bool:
        return "*" in allowed or actual in allowed

    def _candidate_upgrade(self, evidence: str) -> Tuple[Optional[str], Optional[str]]:
        if not evidence:
            return None, None
        pred, _phrase, source = self.policy.best_predicate_match(evidence, include_generic=False)
        if not pred or not source:
            return None, None
        return pred, source

    def _upgrade_locally_supported(
        self,
        predicate: str,
        source_entity: Optional[Entity],
        target_entity: Optional[Entity],
        evidence: str,
    ) -> bool:
        if source_entity is None or target_entity is None:
            return False
        source_variants = self.policy.entity_surface_variants(source_entity)
        target_variants = self.policy.entity_surface_variants(target_entity)
        if not source_variants or not target_variants:
            return False
        return self.policy.predicate_locally_entailed(predicate, source_variants, target_variants, evidence)

    def _adjust_confidence(self, conf: float, pred: str, metadata: Dict[str, Any]) -> float:
        """Small bounded adjustments only; no hard floors."""
        c = max(0.0, min(1.0, float(conf)))
        logs = metadata.get("predicate_precision_logs") or []
        if any(str(x.get("rule", "")).startswith("upgrade:") for x in logs if isinstance(x, dict)):
            c = min(1.0, c + 0.05)
        if any(str(x.get("rule", "")).startswith("type_guard:") for x in logs if isinstance(x, dict)):
            c = max(0.0, c - 0.07)
        if pred in {"relatedTo", "associatedWith"}:
            c = max(0.0, c - 0.10)
        if (
            str(metadata.get("extractor", "") or "").strip() == "lexical"
            and pred in self.semantic_predicates
            and not bool(metadata.get("llm_semantic_agreement"))
        ):
            c = max(0.0, c - 0.05)
        if bool(metadata.get("impact_direction_ambiguous")):
            c = max(0.0, c - 0.08)
        if pred == "mentions":
            c = max(0.0, c - 0.15)
        return c

    def _upgrade_allowed(self, metadata: Dict[str, Any]) -> bool:
        context_source = str(metadata.get("context_source", "") or "").strip()
        relation_kind = str(metadata.get("relation_kind", "") or "").strip()
        if context_source in self.upgrade_blocked_contexts:
            return False
        if relation_kind in {"structural", "provenance"}:
            return False
        return True

    def _infer_relation_kind(self, meta: Dict[str, Any], pred: str, is_inferred: bool) -> str:
        if is_inferred:
            return "inferred_semantic"
        extractor = str(meta.get("extractor", "") or meta.get("source", "") or "").strip()
        context_source = str(meta.get("context_source", "") or "").strip()
        if extractor == "structural" or context_source == "dataset_vote" or pred in self.structural_predicates:
            return "structural"
        if pred == "mentions":
            return "provenance"
        return "extracted_semantic"

    @staticmethod
    def _evidence_mentions_pair(evidence: str, source_label: str, target_label: str) -> bool:
        text = str(evidence or "").casefold()
        src = str(source_label or "").strip().casefold()
        tgt = str(target_label or "").strip().casefold()
        if not text or not src or not tgt:
            return False
        return src in text and tgt in text

    @staticmethod
    def _log(meta: Dict[str, Any], inp: str, out: str, rule: str) -> None:
        logs = meta.get("predicate_precision_logs")
        if not isinstance(logs, list):
            logs = []
        logs.append({"input": inp, "output": out, "rule": rule})
        meta["predicate_precision_logs"] = logs
