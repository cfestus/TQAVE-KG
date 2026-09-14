from __future__ import annotations

import ast
import json
import os
import re
import uuid
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, Optional, List, Set, Union, Tuple

from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS, OWL, SKOS, XSD, DCTERMS

from models import Relation
from utils.helpers import canonicalize_local_name, dedup_mentions, make_uri_safe, mention_from_source_identity, safe_hash
from ontology.add_schema import (
    add_schema,
    apply_property_lifecycle,
    canonicalize_class_local,
    normalized_etd_local_name,
)
from utils.predicate_map import predicate_label_from_local_name
from ontology.property_registry import is_registered_etd_predicate
from ontology.semantic_typing import SemanticTyper
from utils.semantic_relation_policy import STRUCTURAL_RELATION_PREDICATES
from utils.evidence_alignment import align_evidence_span
from utils.semantic_quality_events import record_event

logger = logging.getLogger(__name__)
PROV = Namespace("http://www.w3.org/ns/prov#")


# Predicates whose Relation nodes already carry a deterministic URI minted
# directly from their canonicalized (subject, predicate, object, discourse_id)
# at emission time (see _emit_reified_backbone_relation). Consolidation still
# groups/merges these like any other relation (defense in depth), but does not
# re-mint their URI -- that would be redundant and risks diverging from the
# separate, already-correct backbone URI scheme these depend on elsewhere.
_STRUCTURAL_BACKBONE_PREDICATES = frozenset(STRUCTURAL_RELATION_PREDICATES)


def _relation_object_identity(
    graph: Graph, rel_node: URIRef, p_object: URIRef
) -> Optional[Tuple[str, str, str, str]]:
    """
    Canonical semantic identity of a Relation node's object/target.

    Entity targets identify by canonical URI. Literal targets identify by
    normalized lexical value + RDF datatype + language tag
    rather than being treated as interchangeable with entity URIs or with each
    other merely because their lexical value looks similar.
    """
    seen: Set[Tuple[str, str, str, str]] = set()
    for v in graph.objects(rel_node, p_object):
        if isinstance(v, URIRef):
            seen.add(("entity", str(v), "", ""))
        elif isinstance(v, Literal):
            dtype = str(v.datatype) if v.datatype is not None else ""
            lang = str(v.language) if v.language else ""
            seen.add(("literal", str(v), dtype, lang))
    if len(seen) != 1:
        return None
    return next(iter(seen))


def _evidence_content_fingerprint(graph: Graph, ev: URIRef, etd_ns: Namespace) -> Tuple[Tuple[str, ...], ...]:
    def _vals(pred: URIRef) -> Tuple[str, ...]:
        return tuple(sorted(str(v) for v in graph.objects(ev, pred)))

    return (
        _vals(etd_ns.evidenceText),
        _vals(etd_ns.discourse_id),
        _vals(etd_ns.chunk_index),
        _vals(etd_ns.char_start),
        _vals(etd_ns.char_end),
    )


def _dedupe_identical_evidence_on_relation(graph: Graph, etd_ns: Namespace, rnode: URIRef) -> int:
    """
    Collapse Evidence nodes attached to rnode that are content-identical
    (same evidenceText, discourse_id, chunk_index, and offsets) onto one
    node. Evidence node identity itself is untouched (still minted per raw
    extraction occurrence, per the existing provenance architecture) -- this
    only removes redundancy that surfaces specifically when two occurrences
    of the SAME final assertion, each carrying byte-identical evidence, are
    consolidated onto one Relation. Evidence
    nodes that differ in any of these fields are left fully distinct.
    """
    p_has_evidence = etd_ns.hasEvidence
    evs = [e for e in graph.objects(rnode, p_has_evidence) if isinstance(e, URIRef)]
    if len(evs) <= 1:
        return 0

    by_fingerprint: Dict[Tuple[Tuple[str, ...], ...], List[URIRef]] = {}
    for ev in evs:
        by_fingerprint.setdefault(_evidence_content_fingerprint(graph, ev, etd_ns), []).append(ev)

    removed = 0
    for fingerprint, members in by_fingerprint.items():
        if len(members) <= 1:
            continue
        # An all-empty fingerprint (no evidenceText/discourse_id/chunk/offset
        # recorded at all) is absence of information, not proof of identity
        # -- never collapse those; only merge nodes with a genuinely matching
        # non-empty evidenceText.
        evidence_text_component = fingerprint[0]
        if not evidence_text_component:
            continue
        keep = sorted(members, key=str)[0]
        for dup in members:
            if dup == keep:
                continue
            for t in list(graph.triples((dup, None, None))):
                graph.remove(t)
            for t in list(graph.triples((None, None, dup))):
                graph.remove(t)
                graph.add((t[0], t[1], keep))
            removed += 1
    return removed


def consolidate_duplicate_reified_relations(
    graph: Graph, etd_ns: Namespace, quality_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """
    Consolidate duplicate reified ETD relations that represent the same final
    semantic assertion (canonical subject, final predicate, canonical object-
    or-literal) within the same discourse scope (discourse_id + source field).

    This is the FINAL semantic identity, resolved after coreference, entity
    canonicalization, literal/quantity conversion, and predicate refinement
    have all already run -- all of which are complete by the time relations
    reach this point in RDF serialization (subject/object canonicalization and
    literal typing happen in _add_relations just before this call; predicate
    refinement runs earlier, per-record, in pipeline.py).

    Deterministic winner selection (governs which occurrence's non-identity
    assertion-level metadata -- e.g. predicate_mapping_status, ontology
    predicate URI, semantic_validation_status -- survives on the merged node):
      1) highest confidence
      2) prefer isInferred=false over true
      3) lexicographically smallest relation URI

    Evidence links from losing relations are preserved by attaching them to the
    winner -- evidence is never deleted merely because its Relation was
    deduplicated. Each Evidence node retains its own
    per-occurrence confidence, unaffected by this merge.

    The surviving node is then re-minted onto a URI computed purely from the
    canonical identity components (discourse scope, subject, predicate,
    object-or-literal identity) -- for every relation, not only ones with
    duplicates this run -- so the same resolved assertion always produces the
    same Relation URI regardless of extraction order, chunk order, evidence
    wording, or how many raw occurrences happened to be extracted.
    Structural backbone relations are excluded from re-minting
    since they already carry a deterministic URI from their own emission path.
    """
    rel_type = etd_ns.Relation
    p_subject = etd_ns.subject
    p_predicate = etd_ns.predicate
    p_object = etd_ns.object
    p_confidence = etd_ns.confidence
    p_is_inferred = etd_ns.isInferred
    p_has_evidence = etd_ns.hasEvidence
    p_discourse_id = etd_ns.discourse_id
    p_field = etd_ns.field

    def _single_iri(rel_node: URIRef, pred: URIRef) -> Optional[URIRef]:
        vals = [v for v in graph.objects(rel_node, pred) if isinstance(v, URIRef)]
        if len(vals) != 1:
            return None
        return vals[0]

    def _confidence(rel_node: URIRef) -> float:
        vals = []
        for v in graph.objects(rel_node, p_confidence):
            try:
                vals.append(float(v))
            except Exception:
                continue
        return max(vals) if vals else 0.0

    def _is_inferred(rel_node: URIRef) -> bool:
        vals = list(graph.objects(rel_node, p_is_inferred))
        for v in vals:
            if str(v).strip().lower() in {"true", "1"}:
                return True
        return False

    def _literal_values(node: URIRef, pred: URIRef) -> List[str]:
        out: Set[str] = set()
        for v in graph.objects(node, pred):
            if isinstance(v, Literal):
                s = str(v).strip()
                if s:
                    out.add(s)
        return sorted(out)

    def _discourse_scope(rel_node: URIRef) -> Tuple[str, str]:
        # Relation-level discourse_id/field are not yet finalized at this
        # pipeline point (_align_relation_evidence_provenance/
        # _finalize_relation_provenance run afterward), so prefer them when
        # present but fall back to the evidence-level values, which are
        # reliably stamped by _add_evidence_node before this call runs.
        rel_dids = _literal_values(rel_node, p_discourse_id)
        rel_fields = _literal_values(rel_node, p_field)
        ev_dids: Set[str] = set()
        ev_fields: Set[str] = set()
        for ev in graph.objects(rel_node, p_has_evidence):
            if not isinstance(ev, URIRef):
                continue
            ev_dids.update(_literal_values(ev, p_discourse_id))
            ev_fields.update(_literal_values(ev, p_field))
        did = "|".join(rel_dids) if rel_dids else "|".join(sorted(ev_dids))
        field = "|".join(rel_fields) if rel_fields else "|".join(sorted(ev_fields))
        return did, field

    # Assertion-level (non-evidence-specific) metadata properties whose
    # disagreement across a duplicate group must be reported rather than
    # silently overwritten by whichever occurrence happens to win.
    # Deliberately excludes evidence-specific/per-occurrence
    # fields (confidence is intentionally handled separately).
    # Root-cause fix (bounded correction pass, item 2 -- non-asserted
    # direct SPO / consolidation): negated and epistemic_status were
    # NOT part of a duplicate group's identity key, so a positively-
    # asserted relation and a negated/non-asserted relation sharing the
    # identical (discourse, field, subject, predicate, object) could be
    # merged as if they were redundant re-extractions of the SAME fact.
    # When the negated/non-asserted occurrence won the merge (see
    # winner-selection below), the OTHER, positively-asserted
    # occurrence's already-emitted direct SPO triple (added earlier in
    # _add_relations, before consolidation ever runs) was left orphaned
    # in the graph -- unqualified, and no longer backed by any
    # surviving relation node that actually asserts it. Including these
    # two fields in the key means a positive and a negated/non-asserted
    # version of the same SPO are never treated as duplicates of each
    # other -- both survive as distinct, independently-asserted
    # relations, which is the correct semantics (contradictory or
    # differently-qualified assertions about the same SPO are DISTINCT
    # facts, not redundant restatements of one fact). A true duplicate
    # (same SPO, same negated/epistemic_status -- an honest redundant
    # re-extraction) still merges exactly as before. See also
    # _enforce_direct_spo_assertion_invariant, the belt-and-suspenders
    # pass that runs after this function and removes any direct triple
    # that ends up with no surviving positively-asserted justification,
    # regardless of how it got that way.
    _ASSERTION_LEVEL_PROPS = (
        "predicate_mapping_status",
        "canonical_predicate",
        "relation_origin",
        "semantic_validation_status",
        "relation_kind",
        "negated",
        "epistemic_status",
    )

    def _object_identity_key(oid: Tuple[str, str, str, str]) -> str:
        return "\x1f".join(oid)

    def _negated_flag(rel_node: URIRef) -> bool:
        for v in graph.objects(rel_node, etd_ns.negated):
            if str(v).strip().lower() in {"true", "1"}:
                return True
        return False

    def _epistemic_status(rel_node: URIRef) -> str:
        vals = _literal_values(rel_node, etd_ns.epistemic_status)
        return vals[0] if vals else ""

    groups: Dict[Tuple[str, str, str, str, str, bool, str], List[URIRef]] = {}
    skipped = 0

    for rel in graph.subjects(RDF.type, rel_type):
        if not isinstance(rel, URIRef):
            skipped += 1
            continue

        s = _single_iri(rel, p_subject)
        p = _single_iri(rel, p_predicate)
        oid = _relation_object_identity(graph, rel, p_object)
        if s is None or p is None or oid is None:
            skipped += 1
            logger.warning("Skipping relation with missing/ambiguous SPO: %s", rel)
            continue

        did, field = _discourse_scope(rel)
        key = (
            did, field, str(s), str(p), _object_identity_key(oid),
            _negated_flag(rel), _epistemic_status(rel),
        )
        groups.setdefault(key, []).append(rel)

    duplicate_groups = 0
    removed = 0
    evidence_moved = 0
    reminted = 0
    conflicting_metadata_groups = 0
    identical_evidence_deduped = 0

    for key, rel_nodes in groups.items():
        unique_rels = sorted(set(rel_nodes), key=str)
        did, field, s_str, p_str, _oid_key, _key_negated, _key_epistemic = key

        if len(unique_rels) > 1:
            duplicate_groups += 1

            conflict_found = False
            for prop_local in _ASSERTION_LEVEL_PROPS:
                pred = etd_ns[prop_local]
                values = set()
                for r in unique_rels:
                    values.update(str(v).strip() for v in graph.objects(r, pred) if str(v).strip())
                if len(values) > 1:
                    conflict_found = True
                    logger.warning(
                        "Conflicting assertion-level metadata %r across duplicate group (%s): %s",
                        prop_local,
                        key,
                        sorted(values),
                    )
            if conflict_found:
                conflicting_metadata_groups += 1

            winner = sorted(
                unique_rels,
                key=lambda r: (-_confidence(r), 0 if not _is_inferred(r) else 1, str(r)),
            )[0]

            losers = [r for r in unique_rels if r != winner]
            for loser in losers:
                for ev in list(graph.objects(loser, p_has_evidence)):
                    if (winner, p_has_evidence, ev) not in graph:
                        graph.add((winner, p_has_evidence, ev))
                        evidence_moved += 1

                for t in list(graph.triples((loser, None, None))):
                    graph.remove(t)
                for t in list(graph.triples((None, None, loser))):
                    graph.remove(t)
                removed += 1
                if quality_events is not None:
                    record_event(
                        quality_events,
                        category="duplicate_relation_consolidation",
                        action="merged",
                        discourse_id=did or None,
                        source_relation_id=str(loser),
                        predicate_surface=_uri_localname(URIRef(p_str)),
                        reason=f"redundant re-extraction of the same (subject, predicate, object, negated={_key_negated}, epistemic_status={_key_epistemic!r}) assertion",
                        replacement_relation_ids=[str(winner)],
                    )

            identical_evidence_deduped += _dedupe_identical_evidence_on_relation(graph, etd_ns, winner)

            # Re-mint the surviving node's URI deterministically from the
            # canonical identity -- but ONLY for actual
            # duplicate-merge groups. A singleton relation (no duplicate this
            # run) keeps the URI it was assigned at extraction time: that URI
            # is a pre-existing, load-bearing invariant elsewhere in the
            # pipeline/tests (stable relation URIs for lookup, provenance
            # chains, downstream export), and leaving it untouched avoids
            # collateral, out-of-scope rewrites of every relation node just
            # to cover a case (a resolved assertion happening to be extracted
            # exactly once vs. more than once across runs) that does not
            # arise from duplicate consolidation in the first place. Within
            # an actual duplicate group, which raw URI "wins" previously
            # depended on which raw occurrences happened to be extracted
            # this run (order/wording-dependent); re-minting from the shared
            # canonical identity removes that dependency for the case this
            # task targets.
            pred_local = _uri_localname(URIRef(p_str))
            if pred_local not in _STRUCTURAL_BACKBONE_PREDICATES:
                canonical_seed = "|".join([did, field, s_str, p_str, _oid_key])
                canonical_uri = URIRef(f"{str(etd_ns)}rel_{safe_hash(canonical_seed)}")
                if canonical_uri != winner:
                    for t in list(graph.triples((winner, None, None))):
                        graph.remove(t)
                        graph.add((canonical_uri, t[1], t[2]))
                    for t in list(graph.triples((None, None, winner))):
                        graph.remove(t)
                        graph.add((t[0], t[1], canonical_uri))
                    reminted += 1

    return {
        "duplicate_spo_groups_count": duplicate_groups,
        "relations_removed_count": removed,
        "evidence_links_moved_count": evidence_moved,
        "rels_skipped_missing_spo_count": skipped,
        "relations_reminted_count": reminted,
        "conflicting_metadata_groups": conflicting_metadata_groups,
        "identical_evidence_deduped_count": identical_evidence_deduped,
    }


def _uri_localname(node: URIRef) -> str:
    s = str(node)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _first_literal_for_predicate(graph: Graph, subject: URIRef, predicate: URIRef) -> Optional[str]:
    vals = sorted(
        [str(o).strip() for o in graph.objects(subject, predicate) if isinstance(o, Literal) and str(o).strip()],
        key=lambda x: x.lower(),
    )
    return vals[0] if vals else None


def _normalized_label_key(label: str) -> str:
    """
    Normalized key for alias equivalence checks only.
    Keeps display labels untouched and deterministic.
    """
    s = unicodedata.normalize("NFKC", str(label or ""))
    s = s.casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def enforce_single_en_labels(graph: Graph, etd_ns: Namespace) -> Dict[str, int]:
    """
    Guarantee exactly one rdfs:label@en for ETD Entity/Vote/Relation/Evidence/Export/FieldAssertion nodes.
    Existing labels are normalized to a single deterministic @en literal.
    """
    target_types = [
        etd_ns.Entity,
        etd_ns.DiscourseUnit,
        etd_ns.Vote,
        etd_ns.Relation,
        etd_ns.Evidence,
        etd_ns.Export,
        etd_ns.FieldAssertion,
    ]
    nodes: Set[URIRef] = set()
    for t in target_types:
        for n in graph.subjects(RDF.type, t):
            if isinstance(n, URIRef):
                nodes.add(n)

    # One-pass index for reified vote edges to avoid O(V*R) scans.
    reified_cast_by: Dict[URIRef, Set[URIRef]] = {}
    reified_receives: Dict[URIRef, Set[URIRef]] = {}
    for rel in graph.subjects(RDF.type, etd_ns.Relation):
        if not isinstance(rel, URIRef):
            continue
        subj = next((o for o in graph.objects(rel, etd_ns.subject) if isinstance(o, URIRef)), None)
        pred = next((o for o in graph.objects(rel, etd_ns.predicate) if isinstance(o, URIRef)), None)
        obj = next((o for o in graph.objects(rel, etd_ns.object) if isinstance(o, URIRef)), None)
        if subj is None or pred is None or obj is None:
            continue
        if pred == etd_ns.castBy:
            reified_cast_by.setdefault(subj, set()).add(obj)
        elif pred == etd_ns.receivesVote:
            reified_receives.setdefault(subj, set()).add(obj)

    converted = 0
    generated = 0
    deduped = 0

    for node in sorted(nodes, key=str):
        classes = {c for c in graph.objects(node, RDF.type) if isinstance(c, URIRef)}
        is_relation = etd_ns.Relation in classes
        is_evidence = etd_ns.Evidence in classes
        is_vote = etd_ns.Vote in classes
        is_discourse_unit = etd_ns.DiscourseUnit in classes
        is_export = etd_ns.Export in classes
        is_field_assertion = etd_ns.FieldAssertion in classes

        existing = [o for o in graph.objects(node, RDFS.label) if isinstance(o, Literal)]
        en_labels = sorted(
            [str(l).strip() for l in existing if str(l).strip() and str(getattr(l, "language", "")).lower() == "en"],
            key=lambda x: x.lower(),
        )
        any_labels = sorted([str(l).strip() for l in existing if str(l).strip()], key=lambda x: x.lower())

        if len(existing) == 1 and len(en_labels) == 1:
            # Already in canonical shape: exactly one label@en.
            continue

        chosen: Optional[str] = None
        if en_labels:
            chosen = en_labels[0]
            if len(en_labels) > 1 or len(existing) > 1:
                deduped += 1
        elif any_labels:
            chosen = any_labels[0]
            converted += 1
            if len(any_labels) > 1:
                deduped += 1
        else:
            if is_relation:
                s = next((o for o in graph.objects(node, etd_ns.subject) if isinstance(o, URIRef)), None)
                p = next((o for o in graph.objects(node, etd_ns.predicate) if isinstance(o, URIRef)), None)
                o = next((o for o in graph.objects(node, etd_ns.object) if isinstance(o, URIRef)), None)
                pred_l = _uri_localname(p) if p is not None else "unknownPredicate"
                subj_l = _uri_localname(s) if s is not None else "unknownSubject"
                obj_l = _uri_localname(o) if o is not None else "unknownObject"
                chosen = f"Relation: {pred_l} ({subj_l} -> {obj_l})"
            elif is_evidence:
                snippet = _first_literal_for_predicate(graph, node, etd_ns.snippet_id)
                base = snippet or _uri_localname(node)
                if not snippet:
                    source = _first_literal_for_predicate(graph, node, etd_ns.source)
                    if source:
                        base = source
                base = (base or _uri_localname(node)).strip()
                prefix = "Evidence: "
                max_total = 80
                max_base = max_total - len(prefix)
                if max_base < 4:
                    max_base = 4
                if len(base) > max_base:
                    base = base[: max_base - 3] + "..."
                chosen = f"{prefix}{base}"
            elif is_vote:
                cast = next((o for o in graph.objects(node, etd_ns.castBy) if isinstance(o, URIRef)), None)
                recv = next((o for o in graph.objects(node, etd_ns.receivesVote) if isinstance(o, URIRef)), None)
                if cast is None:
                    cands = sorted(reified_cast_by.get(node, set()), key=str)
                    cast = cands[0] if cands else None
                if recv is None:
                    cands = sorted(reified_receives.get(node, set()), key=str)
                    recv = cands[0] if cands else None
                if cast is not None and recv is not None:
                    chosen = f"Vote by {_uri_localname(cast)} for {_uri_localname(recv)}"
                else:
                    chosen = _uri_localname(node)
            elif is_discourse_unit:
                chosen = f"Discourse unit: {_uri_localname(node)}"
            elif is_export:
                ds = _first_literal_for_predicate(graph, node, etd_ns.datasetId)
                run = _first_literal_for_predicate(graph, node, etd_ns.runId)
                if ds and run:
                    chosen = f"Export: {ds} ({run})"
                else:
                    chosen = _uri_localname(node)
            elif is_field_assertion:
                source_section = _first_literal_for_predicate(graph, node, etd_ns.sourceSection)
                field_name = _first_literal_for_predicate(graph, node, etd_ns.fieldName)
                record_id = _first_literal_for_predicate(graph, node, etd_ns.recordId)
                if source_section and field_name:
                    chosen = f"Field assertion: {source_section}.{field_name}"
                elif field_name:
                    chosen = f"Field assertion: {field_name}"
                elif record_id:
                    chosen = f"Field assertion: record {record_id}"
                else:
                    chosen = f"Field assertion: {_uri_localname(node)}"
            else:
                fallback: Optional[str] = None
                for p, o in graph.predicate_objects(node):
                    if not isinstance(p, URIRef) or not isinstance(o, Literal):
                        continue
                    local = _uri_localname(p).lower()
                    if local in {"name", "title"} and str(o).strip():
                        val = str(o).strip()
                        if fallback is None or val.lower() < fallback.lower():
                            fallback = val
                chosen = fallback or _uri_localname(node)
            generated += 1

        assert chosen is not None
        for old in list(graph.objects(node, RDFS.label)):
            graph.remove((node, RDFS.label, old))
        graph.add((node, RDFS.label, Literal(chosen, lang="en")))

    return {
        "nodes_processed": len(nodes),
        "labels_generated": generated,
        "labels_converted_to_en": converted,
        "labels_deduped": deduped,
    }


def prune_isolated_entities(graph: Graph, etd_ns: Namespace) -> Dict[str, int]:
    """
    Remove isolated etd:Entity nodes not referenced by any reified relation
    as etd:subject or etd:object.
    Keep provenance anchor entities referenced via etd:hasSourceEntity.
    """
    p_subject = etd_ns.subject
    p_object = etd_ns.object
    p_has_source_entity = etd_ns.hasSourceEntity
    t_entity = etd_ns.Entity
    t_relation = etd_ns.Relation

    connected: Set[URIRef] = set()
    for rel in graph.subjects(RDF.type, t_relation):
        for node in graph.objects(rel, p_subject):
            if isinstance(node, URIRef):
                connected.add(node)
        for node in graph.objects(rel, p_object):
            if isinstance(node, URIRef):
                connected.add(node)
        for node in graph.objects(rel, p_has_source_entity):
            if isinstance(node, URIRef):
                connected.add(node)

    for node in graph.objects(None, p_has_source_entity):
        if isinstance(node, URIRef):
            connected.add(node)

    isolated: Set[URIRef] = set()
    for ent in graph.subjects(RDF.type, t_entity):
        if isinstance(ent, URIRef) and ent not in connected:
            isolated.add(ent)

    triples_to_remove: Set[Tuple[Node, Node, Node]] = set()
    for ent in isolated:
        for t in graph.triples((ent, None, None)):
            triples_to_remove.add(t)
        for t in graph.triples((None, None, ent)):
            triples_to_remove.add(t)

    for t in triples_to_remove:
        graph.remove(t)

    return {
        "isolated_entities_found": len(isolated),
        "isolated_entities_removed": len(isolated),
        "triples_removed": len(triples_to_remove),
    }

class RDFSerializer:
    """
    Serialize ETD-Hub KG into RDF formats using canonical URIs.
    """

    def __init__(
        self,
        annotations_dict: Dict[str, Any],
        registry: Any = None,
        namespaces: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        # ---- 1) Normalise annotations -----------------------------------------
        if hasattr(annotations_dict, "to_dict"):
            annotations_dict = annotations_dict.to_dict()

        if not isinstance(annotations_dict, dict):
            raise TypeError(
                f"annotations_dict must be a dict, got {type(annotations_dict)}"
            )

        self.annotations: Dict[str, Any] = annotations_dict

        # ---- 2) Registry -------------------------------------------------------
        if registry is None and "registry" in kwargs:
            registry = kwargs["registry"]

        self.registry = registry

        # ---- 3) Namespaces -----------------------------------------------------
        config = kwargs.get("config")
        self.config = config
        if config is None:
            try:
                from config import Config as _RuntimeConfig  # lazy import to avoid hard dependency at module import

                self.config = _RuntimeConfig
            except Exception:
                self.config = None

        # Active domain profile (see domain_profiles.py). Callers may pass
        # domain_profile=<DomainProfile> explicitly (e.g. tests exercising
        # a synthetic profile); otherwise resolved from Config.DOMAIN_PROFILE
        # -- this reproduces the AI-ethics profile by default, so existing
        # callers see no behaviour change.
        domain_profile_override = kwargs.get("domain_profile")
        if domain_profile_override is not None:
            self.domain_profile = domain_profile_override
        else:
            from domain_profiles import get_active_profile

            self.domain_profile = get_active_profile()

        prune_override = kwargs.get("prune_isolated_entities")
        if prune_override is None:
            self.prune_isolated_entities = bool(getattr(self.config, "PRUNE_ISOLATED_ENTITIES", True))
        else:
            self.prune_isolated_entities = bool(prune_override)
        self.strict_relation_backbone = bool(getattr(self.config, "STRICT_RELATION_BACKBONE", False))
        self.strict_etd_property_registry = bool(getattr(self.config, "STRICT_ETD_PROPERTY_REGISTRY", True))
        self.materialize_raw_fields = bool(getattr(self.config, "MATERIALIZE_RAW_FIELDS", True))
        self.materialize_sensitive_fields = bool(getattr(self.config, "MATERIALIZE_SENSITIVE_FIELDS", False))
        self.media_base_url = str(getattr(self.config, "MEDIA_BASE_URL", "") or "").strip()
        self._raw_datetime_warning_keys: Set[str] = set()
        # Defect-1 root-cause fix: entities whose surface_form _add_entities()
        # had to derive from ent.label (no genuine extracted-mention text was
        # ever provided) -- tracked so _finalize_entity_provenance_completion
        # can later REPLACE this generic placeholder with a genuine
        # relation-sourced value once relations exist, instead of treating
        # "a value is already present" as "nothing to do here".
        self._generic_surface_form_entities: Set[URIRef] = set()

        # Semantic-quality instrumentation events raised during graph
        # construction/consolidation/invariant enforcement (see
        # utils/semantic_quality_events.py). Merged with
        # EnhancedSemanticPipeline.quality_events (pre-serialization
        # drops) and persisted once per run to
        # output/semantic_quality_report.json by
        # write_semantic_quality_report() (see main.py). Callers may
        # pass an existing list via quality_events= to share one sink
        # with the pipeline that built `registry`; otherwise a fresh
        # list is used (e.g. when RDFSerializer is exercised standalone
        # in tests).
        quality_events_override = kwargs.get("quality_events")
        self.quality_events: List[Dict[str, Any]] = (
            quality_events_override if isinstance(quality_events_override, list) else []
        )

        # C.2/C.3: optional callable(field, discourse_id) -> raw source text
        # (e.g. utils.source_text_lookup.SourceTextLookup). When supplied,
        # the primary-mention chosen for an entity's emitted
        # (field, discourse_id, start_char, end_char) tuple is one whose
        # pair actually resolves AND whose offsets bound the entity's own
        # surface form (or a preserved alias / normalization form). When
        # NONE of an entity's mentions verify, the numeric span is withheld
        # rather than silently truncated or emitted as a false grounded
        # claim -- identity (field, discourse_id, surface_form) is kept.
        stl = kwargs.get("source_text_lookup")
        self._source_text_lookup = stl if callable(stl) else None

        if namespaces is None:
            if "namespaces" in kwargs and kwargs["namespaces"] is not None:
                namespaces = kwargs["namespaces"]
            elif config is not None and hasattr(config, "NAMESPACES"):
                namespaces = getattr(config, "NAMESPACES")

        if namespaces is None:
            raise ValueError("RDFSerializer requires a namespaces dict (e.g., Config.NAMESPACES)")

        self.namespaces: Dict[str, str] = namespaces
        # Export metadata must be initialized once after config/prune resolution.
        self._init_export_metadata(kwargs)
        etd_base = str(self.namespaces["etd"])
        self.export_node = URIRef(
            f"{etd_base}export_{safe_hash(self.dataset_id + '|' + self.run_id)}"
        )

        # ---- 4) RDF Graph ------------------------------------------------------
        self.graph: Graph = Graph()

        # Bind namespaces cleanly
        for pfx, uri in self.namespaces.items():
            self.graph.bind(pfx, Namespace(uri))

        # Bind SKOS prefix for readability in TTL output
        self.graph.bind("skos", SKOS)
        # Guarantee canonical PROV namespace binding for provenance triples.
        self.graph.bind("prov", PROV)

        # ---- 5) Build entity index ---------------------------------------------
        # Key → entity (label, id, uri)
        self.entity_index: Dict[str, Any] = self._build_entity_index()
        self.vote_entity_index: Dict[str, Any] = self._build_vote_entity_index()

        # Track which ETD classes we actually used, for Option A alignment
        self._seen_entity_classes: Set[str] = set()
        self._class_alignment_targets: Dict[str, URIRef] = {}

        # --- Add schema early ---
        #etd_ns = Namespace(self.namespaces["etd"])
        #add_schema(self.graph, etd_ns)

        #add_schema(self.graph, Namespace(self.namespaces["etd"]))

        #self.semantic_typer = SemanticTyper(self.graph, Namespace(self.namespaces["etd"]))
        
        # --- Add schema early (single call) ---
        etd_ns = Namespace(self.namespaces["etd"])
        allowed = add_schema(self.graph, etd_ns)   # add_schema returns set[str]
        self.semantic_typer = SemanticTyper(self.graph, etd_ns, allowed_classes=allowed, domain_profile=self.domain_profile)



        # ---- 6) Construct RDF graph --------------------------------------------
        self._add_ontology_header()
        self._add_entities()
        self._external_to_internal_index = self._build_external_to_internal_uri_index(etd_ns)
        self._add_alignment_axioms()   # Option A: minimal conceptual ontology alignment
        self._add_relations()
        self._assert_relation_spine_atomicity(etd_ns)
        # Root-cause fix (bounded preflight, item 3): endpoint recovery
        # MUST run before consolidation, not after. Recovery can repoint
        # a mismatched relation's subject/object onto an entity another
        # relation already uses -- if that happens after consolidation
        # has already taken its one pass over the graph, the newly-
        # identical (subject, predicate, object) pair it just created
        # would never be recognised as a duplicate and both would
        # survive as separate reified relations. Running recovery first
        # means any duplicate identity it produces is caught by the
        # consolidation pass that follows, exactly like a duplicate
        # produced by ordinary extraction would be.
        self.endpoint_surface_compatibility_final = self._enforce_endpoint_surface_compatibility_final(etd_ns)
        logger.info("Final endpoint/surface compatibility enforcement: %s", self.endpoint_surface_compatibility_final)
        self.duplicate_relation_consolidation = consolidate_duplicate_reified_relations(
            self.graph, etd_ns, quality_events=self.quality_events,
        )
        logger.info("Duplicate reified relation consolidation: %s", self.duplicate_relation_consolidation)
        # The direct-SPO assertion invariant is the final backstop and
        # must run LAST, after consolidation -- consolidation (a merge
        # of duplicate reified relations) is the one operation most
        # likely to leave a direct triple without a surviving positively
        # -asserted justification (see the function's own docstring).
        self.direct_spo_invariant_enforcement = self._enforce_direct_spo_assertion_invariant(etd_ns)
        logger.info("Direct-SPO assertion invariant enforcement: %s", self.direct_spo_invariant_enforcement)
        # Root-cause fix (targeted correction pass, item 1): final,
        # reference-aware Evidence cleanup -- must run after endpoint
        # recovery, consolidation, AND the direct-SPO invariant above,
        # since any of those can remove a relation's own hasEvidence
        # link without knowing whether some OTHER surviving relation
        # still needs that Evidence node. No later step in this
        # __init__ adds a new etd:hasEvidence triple (the only call site
        # that ever does runs during _add_relations, long before this
        # point), so nothing downstream can be preempted by running this
        # here.
        self.orphan_evidence_reconciliation = self._reconcile_orphan_evidence_nodes(etd_ns)
        logger.info("Orphan evidence reconciliation: %s", self.orphan_evidence_reconciliation)
        self.relation_evidence_alignment = self._align_relation_evidence_provenance(etd_ns)
        logger.info("Relation/evidence provenance alignment: %s", self.relation_evidence_alignment)
        self.local_predicate_declaration = self._declare_local_relation_predicates(etd_ns)
        logger.info("Local relation predicate declaration: %s", self.local_predicate_declaration)
        # Create export metadata node before label enforcement so Export always gets rdfs:label@en.
        self._add_dataset_metadata()
        self.final_relation_provenance = self._finalize_relation_provenance(etd_ns)
        logger.info("Final relation provenance completion: %s", self.final_relation_provenance)
        self.relation_source_traceability = self._finalize_relation_source_traceability(etd_ns)
        logger.info("Relation source traceability completion: %s", self.relation_source_traceability)
        self.entity_provenance_completion = self._finalize_entity_provenance_completion(etd_ns)
        self.label_enforcement = enforce_single_en_labels(self.graph, etd_ns)
        logger.info("Label enforcement metrics: %s", self.label_enforcement)
        if self.prune_isolated_entities:
            self.isolated_entity_prune_metrics = prune_isolated_entities(self.graph, etd_ns)
        else:
            self.isolated_entity_prune_metrics = {
                "isolated_entities_found": 0,
                "isolated_entities_removed": 0,
                "triples_removed": 0,
            }
        logger.info("Isolated entity prune metrics: %s", self.isolated_entity_prune_metrics)

    def _init_export_metadata(self, kwargs: Dict[str, Any]) -> None:
        """
        Initialize run-scoped export metadata once per serializer instance.
        """
        meta = self.annotations.get("metadata", {}) or {}
        input_path = (
            kwargs.get("input_path")
            or getattr(self.config, "INPUT_JSON", "")
            or meta.get("input_path")
            or "unknown_input"
        )
        dataset_id = kwargs.get("dataset_id") or meta.get("datasetId") or Path(str(input_path)).stem
        run_id = kwargs.get("run_id") or meta.get("runId")
        if not run_id:
            now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"run-{safe_hash(str(dataset_id) + '|' + now)}"

        self.dataset_id = str(dataset_id)
        self.run_id = str(run_id)
        self.generated_at_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.pipeline_version = str(
            kwargs.get("pipeline_version")
            or getattr(self.config, "PIPELINE_VERSION", None)
            or "5.1"
        )
        git_commit = kwargs.get("git_commit") or os.getenv("GIT_COMMIT")
        if not git_commit:
            try:
                import subprocess

                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode("utf-8").strip()
            except Exception:
                git_commit = "unknown"
        self.git_commit = str(git_commit)
        self.shacl_profile_version = str(
            kwargs.get("shacl_profile_version")
            or getattr(self.config, "SHACL_SHAPES_PATH", None)
            or "ontology/shacl_shapes.ttl"
        )
        self.export_mode = f"ai;prune_isolated_entities={str(bool(self.prune_isolated_entities)).lower()}"

    # ------------------------------------------------------------------
    # Internal: entity index + resolution
    # ------------------------------------------------------------------
    def _build_entity_index(self) -> Dict[str, Any]:
        """
        Build a lookup from various keys → entity objects to
        resolve literals / IDs back to canonical entity URIs.

        Keys we store:
          - label.lower()
          - name.lower()
          - str(ent.id) / str(ent.idx) / str(ent.index) if present
          - str(ent.uri)
        """
        index: Dict[str, Any] = {}

        if self.registry is None:
            return index

        get_all = getattr(self.registry, "get_all_entities", None)
        if not callable(get_all):
            return index

        for ent in get_all():
            label = getattr(ent, "label", None) or getattr(ent, "name", None)
            if label:
                key = label.strip()
                if key:
                    index[key.lower()] = ent

            # Potential numeric / string identifiers
            for attr in ("id", "idx", "index"):
                val = getattr(ent, attr, None)
                if val is not None:
                    index[str(val)] = ent

            # Also index by canonical URI string if available
            ent_uri = getattr(ent, "uri", None)
            if ent_uri:
                index[str(ent_uri).strip()] = ent

        return index

    def _build_vote_entity_index(self) -> Dict[str, Any]:
        """
        Build discourse-id -> canonical Vote entity lookup.
        Vote entities are created upstream in the pipeline and should be reused
        as the single canonical vote resource during RDF export.
        """
        index: Dict[str, Any] = {}

        if self.registry is None:
            return index

        get_all = getattr(self.registry, "get_all_entities", None)
        if not callable(get_all):
            return index

        for ent in get_all():
            if str(getattr(ent, "entity_type", "") or "") != "Vote":
                continue
            meta = getattr(ent, "metadata", None) or {}
            did = meta.get("discourse_id")
            if did is None:
                continue
            key = str(did).strip()
            if not key:
                continue
            index.setdefault(key, ent)

        return index

    def _resolve_entity_uri_from_value(self, value: Any) -> Optional[URIRef]:
        """
        Try hard to map a value to an existing entity's URIRef:

        1. If it matches an existing entity label/name/id/uri → entity.uri
        2. If it looks like a full HTTP(S) IRI → safe URIRef
        3. If it looks like a CURIE with known prefix → expanded safe URIRef

        Otherwise returns None (caller decides what to do).
        """
        # Already a URIRef
        if isinstance(value, URIRef):
            return value

        if not isinstance(value, str):
            return None

        s = value.strip()
        if not s:
            return None

        # Entity index hit: try as-is, then lowercase (labels)
        ent = self.entity_index.get(s) or self.entity_index.get(s.lower())
        if ent is not None and getattr(ent, "uri", None):
            return _safe_uriref(str(ent.uri), self.namespaces)

        # HTTP(S) IRI
        if s.startswith("http://") or s.startswith("https://"):
            return _safe_uriref(s, self.namespaces)

        # CURIE prefix:local with known prefix
        if ":" in s and " " not in s:
            prefix, local = s.split(":", 1)
            if prefix in self.namespaces:
                iri = self.namespaces[prefix] + local
                return _safe_uriref(iri, self.namespaces)

        return None

    def _canonical_vote_uri_for_relation(
        self,
        rel: Relation,
        subj: Any,
        obj: Any,
        meta: Dict[str, Any],
        etd_ns: Namespace,
    ) -> URIRef:
        """
        Resolve the single canonical Vote URI for semantic vote materialisation.
        Reuse the upstream /entity/vote/... entity when present; otherwise mint a
        deterministic canonical vote IRI (never an etd:vote_* local node).
        """
        discourse_id = meta.get("discourse_id")
        did = str(discourse_id).strip() if discourse_id is not None else ""
        if did:
            ent = self.vote_entity_index.get(did)
            if ent is not None and getattr(ent, "uri", None):
                return _safe_uriref(str(ent.uri), self.namespaces)

        src_key = str(getattr(rel, "source", "") or "").strip()
        if src_key:
            src_ent = self.entity_index.get(src_key)
            if src_ent is not None and str(getattr(src_ent, "entity_type", "") or "") == "Vote":
                return _safe_uriref(str(src_ent.uri), self.namespaces)

        base_etd = self.namespaces["etd"].rstrip("#/")
        local_id = did or safe_hash(f"{str(subj)}|{str(obj)}|{str(meta.get('raw_vote_value'))}")
        slug = make_uri_safe(str(local_id).strip() or "vote")
        return URIRef(f"{base_etd}/entity/vote/vote-{slug}-{safe_hash(str(local_id))}")

    def _set_vote_labels(
        self,
        vote_uri: URIRef,
        vote_label: str,
    ) -> None:
        """
        Ensure canonical votes carry the richer interaction label.
        Replace generic/missing rdfs:label values and keep exactly one prefLabel.
        """
        existing_labels = list(self.graph.objects(vote_uri, RDFS.label))
        keep_existing = False
        if len(existing_labels) == 1:
            sval = str(existing_labels[0]).strip()
            if sval and "Vote by" in sval:
                keep_existing = True

        if not keep_existing:
            for old in existing_labels:
                self.graph.remove((vote_uri, RDFS.label, old))
            self.graph.add((vote_uri, RDFS.label, Literal(vote_label, lang="en")))

        for old in list(self.graph.objects(vote_uri, SKOS.prefLabel)):
            self.graph.remove((vote_uri, SKOS.prefLabel, old))
        self.graph.add((vote_uri, SKOS.prefLabel, Literal(vote_label, lang="en")))

    def _is_internal_etd_uri(self, uri: URIRef) -> bool:
        return isinstance(uri, URIRef) and str(uri).startswith(str(self.namespaces["etd"]))

    def _extract_wikidata_qid(self, uri: URIRef) -> Optional[str]:
        s = str(uri).strip()
        if not s:
            return None
        m = re.search(r"(Q\d+)$", s, flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(1).upper()

    def _build_external_to_internal_uri_index(self, etd_ns: Namespace) -> Dict[str, List[URIRef]]:
        """
        Build a deterministic map external-identity-URI -> canonical internal ETD URI candidates.
        """
        index: Dict[str, Set[URIRef]] = {}

        def _add(ext_uri: URIRef, internal_uri: URIRef) -> None:
            if not isinstance(ext_uri, URIRef) or not isinstance(internal_uri, URIRef):
                return
            if not self._is_internal_etd_uri(internal_uri):
                return
            key = str(ext_uri)
            index.setdefault(key, set()).add(internal_uri)
            qid = self._extract_wikidata_qid(ext_uri)
            if qid:
                index.setdefault(f"http://www.wikidata.org/entity/{qid}", set()).add(internal_uri)
                index.setdefault(f"https://www.wikidata.org/entity/{qid}", set()).add(internal_uri)

        for pred in (OWL.sameAs, etd_ns.wikidata):
            for subj, obj in self.graph.subject_objects(pred):
                if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                    _add(obj, subj)

        # Registry-backed fallback mapping for cases where sameAs isn't materialized yet.
        get_all = getattr(self.registry, "get_all_entities", None)
        if callable(get_all):
            for ent in get_all():
                ent_uri = getattr(ent, "uri", None)
                if not ent_uri:
                    continue
                internal = _safe_uriref(str(ent_uri), self.namespaces)
                if not self._is_internal_etd_uri(internal):
                    continue
                meta = getattr(ent, "metadata", None) or {}
                ext_candidates: List[str] = []
                wdu = meta.get("wikidata_uri")
                if isinstance(wdu, str) and wdu.strip():
                    ext_candidates.append(wdu.strip())
                wd = meta.get("wikidata")
                if isinstance(wd, dict):
                    qid = wd.get("qid")
                    if isinstance(qid, str) and qid.strip():
                        ext_candidates.append(f"http://www.wikidata.org/entity/{qid.strip()}")
                        ext_candidates.append(f"https://www.wikidata.org/entity/{qid.strip()}")
                for ext in ext_candidates:
                    _add(_safe_uriref(ext, self.namespaces), internal)

        return {k: sorted(v, key=str) for k, v in index.items()}

    def _canonicalize_relation_endpoint_uri(self, endpoint: Any) -> Any:
        """
        Canonicalize relation endpoints:
        - keep non-URI nodes unchanged
        - keep internal ETD URIs unchanged
        - replace external identity URI with canonical internal ETD URI when mapped
        """
        if not isinstance(endpoint, URIRef):
            return endpoint
        if self._is_internal_etd_uri(endpoint):
            return endpoint

        candidates = self._external_to_internal_index.get(str(endpoint), [])
        if not candidates:
            qid = self._extract_wikidata_qid(endpoint)
            if qid:
                candidates = (
                    self._external_to_internal_index.get(f"http://www.wikidata.org/entity/{qid}", [])
                    or self._external_to_internal_index.get(f"https://www.wikidata.org/entity/{qid}", [])
                )

        if candidates:
            chosen = sorted(candidates, key=str)[0]
            if len(candidates) > 1:
                logger.warning(
                    "Multiple internal mappings for external endpoint %s; chosen=%s candidates=%s",
                    endpoint,
                    chosen,
                    [str(c) for c in sorted(candidates, key=str)],
                )
            return chosen
        return endpoint

    def _ensure_relation_node_identity(
        self,
        rnode: URIRef,
        subj: Any,
        pred_uri: URIRef,
        obj: Any,
        etd_ns: Namespace,
    ) -> URIRef:
        """
        Reified relation backbone is immutable:
        if rnode already carries conflicting SPO, remint deterministically
        (or raise in strict mode).
        """
        existing_s = sorted(set(self.graph.objects(rnode, etd_ns.subject)), key=str)
        existing_p = sorted(set(self.graph.objects(rnode, etd_ns.predicate)), key=str)
        existing_o = sorted(set(self.graph.objects(rnode, etd_ns.object)), key=str)
        if not existing_s and not existing_p and not existing_o:
            return rnode

        same_spo = (
            len(existing_s) == 1 and existing_s[0] == subj and
            len(existing_p) == 1 and existing_p[0] == pred_uri and
            len(existing_o) == 1 and existing_o[0] == obj
        )
        if same_spo:
            return rnode

        msg = (
            f"Conflicting SPO write for relation node {rnode}: "
            f"existing_s={existing_s}, existing_p={existing_p}, existing_o={existing_o}, "
            f"new=({subj}, {pred_uri}, {obj})"
        )
        if self.strict_relation_backbone:
            raise RuntimeError(msg)

        base = self.namespaces["etd"]
        seed = f"{str(rnode)}|{self._spo_key(subj, pred_uri, obj)}"
        salt = 0
        while True:
            suffix = "" if salt == 0 else f"|{salt}"
            candidate = URIRef(f"{base}rel_{safe_hash(seed + suffix)}")
            c_s = sorted(set(self.graph.objects(candidate, etd_ns.subject)), key=str)
            c_p = sorted(set(self.graph.objects(candidate, etd_ns.predicate)), key=str)
            c_o = sorted(set(self.graph.objects(candidate, etd_ns.object)), key=str)
            if not c_s and not c_p and not c_o:
                logger.warning("%s. Reminted relation URI: %s", msg, candidate)
                return candidate
            if (
                len(c_s) == 1 and c_s[0] == subj and
                len(c_p) == 1 and c_p[0] == pred_uri and
                len(c_o) == 1 and c_o[0] == obj
            ):
                return candidate
            salt += 1

    def _assert_relation_spine_atomicity(self, etd_ns: Namespace) -> None:
        """
        Fail-fast structural guard: every reified relation must have exactly one S/P/O.
        """
        violations: List[str] = []
        for rnode in sorted(self.graph.subjects(RDF.type, etd_ns.Relation), key=str):
            if not isinstance(rnode, URIRef):
                continue
            s_count = len(set(self.graph.objects(rnode, etd_ns.subject)))
            p_count = len(set(self.graph.objects(rnode, etd_ns.predicate)))
            o_count = len(set(self.graph.objects(rnode, etd_ns.object)))
            if s_count != 1 or p_count != 1 or o_count != 1:
                violations.append(
                    f"{rnode} subjectCount={s_count} predicateCount={p_count} objectCount={o_count}"
                )
        if violations:
            sample = "; ".join(violations[:10])
            raise RuntimeError(
                f"Relation spine cardinality violation ({len(violations)}): {sample}"
            )

    # ------------------------------------------------------------------
    # Ontology header
    # ------------------------------------------------------------------
    def _add_ontology_header(self) -> None:
        """Ensure a single canonical ontology resource exists."""
        graph = self.graph
        ns = self.namespaces

        etd_ns = Namespace(ns["etd"])

        # Canonical ontology IRI: <ETD_NS>ontology
        onto_uri = etd_ns["ontology"]

        # add_schema() already materialises ontology metadata on etd:ontology.
        # Here we only ensure ontology typing exists if schema injection changes.
        graph.add((onto_uri, RDF.type, OWL.Ontology))


    def _maybe_record_alignment_target(self, class_local: str, external_uri: URIRef) -> None:
        """
        Record that ETD class `class_local` should be aligned to `external_uri`.
        Only records mappings for recognised external ontology namespaces.
        """
        if not isinstance(external_uri, URIRef):
            return

        external_prefixes = {
            "dcat",
            "schema",
            "aipo",
            "relaieo",
            "vair",
            "airo",
            "hudock",
            "fmo",
            "dpv",
        }

        uri_str = str(external_uri)
        for pfx in external_prefixes:
            base = self.namespaces.get(pfx)
            if base and uri_str.startswith(base):
                if class_local not in self._class_alignment_targets:
                    self._class_alignment_targets[class_local] = external_uri
                return

    def _add_alignment_axioms(self) -> None:
        """
        Option A: Minimal conceptual ontology alignment.

        Primary strategy (fixes case-mismatch):
        ---------------------------------------
        For each ETD class where we have seen an individual with an
        `etd:ontology_uri`, we use that URI as the rdfs:subClassOf target:

            etd:<ClassLocal> rdfs:subClassOf <external-class-IRI> .

        Fallback strategy (heuristic, only when no ontology_uri-based
        mapping is available):
        -------------------------------------------------------------
        For each ETD class name matching <Prefix>_<Local> where
        Prefix.lower() is a known external namespace, we add:

            etd:<Prefix_Local> rdfs:subClassOf <prefix>:<Local> .
        """
        etd_ns = Namespace(self.namespaces["etd"])

        # 1) Preferred: use mappings learned from ontology_uri on entities
        for class_local, external_uri in sorted(self._class_alignment_targets.items()):
            etd_class_uri = etd_ns[class_local]
            # subclass to the exact external URI we observed (correct case)
            self.graph.add((etd_class_uri, RDFS.subClassOf, external_uri))
            # explicitly type the ETD proxy as a class
            self.graph.add((etd_class_uri, RDF.type, OWL.Class))
            self._annotate_dynamic_etd_class(etd_class_uri, class_local)

        # 2) Fallback: heuristic from class_local if we have no explicit target
        external_prefixes = {
            "dcat",
            "schema",
            "aipo",
            "relaieo",
            "vair",
            "airo",
            "hudock",
            "fmo",
            "dpv",
        }

        for class_local in sorted(self._seen_entity_classes):
            # Skip classes already covered by explicit ontology_uri mapping
            if class_local in self._class_alignment_targets:
                continue

            if "_" not in class_local:
                continue

            prefix, local = class_local.split("_", 1)
            pfx_lower = prefix.lower()

            if pfx_lower not in external_prefixes:
                continue

            base = self.namespaces.get(pfx_lower)
            if not base:
                continue

            external_class_uri = URIRef(base + local)
            etd_class_uri = etd_ns[class_local]

            self.graph.add((etd_class_uri, RDFS.subClassOf, external_class_uri))
            self.graph.add((etd_class_uri, RDF.type, OWL.Class))
            self._annotate_dynamic_etd_class(etd_class_uri, class_local)

    def _annotate_dynamic_etd_class(self, cls_uri: URIRef, class_local: str) -> None:
        """
        Ensure classes minted during alignment remain self-describing.
        """
        if not any(True for _ in self.graph.objects(cls_uri, RDFS.label)):
            human = re.sub(r"[_\-]+", " ", str(class_local or "").strip())
            human = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", human)
            human = re.sub(r"\s+", " ", human).strip() or "Entity Class"
            label = " ".join(w.capitalize() for w in human.split())
            self.graph.add((cls_uri, RDFS.label, Literal(label, lang="en")))

        if not any(True for _ in self.graph.objects(cls_uri, RDFS.comment)):
            self.graph.add(
                (
                    cls_uri,
                    RDFS.comment,
                    Literal(
                        "ETD proxy class materialised during ontology alignment; used to preserve stable local typing.",
                        lang="en",
                    ),
                )
            )


    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------
    def _add_entities(self) -> None:
        """Serialise entities using canonical entity.uri and ontology classes."""
        if self.registry is None:
            return

        etd_ns = Namespace(self.namespaces["etd"])

        get_all = getattr(self.registry, "get_all_entities", None)
        if not callable(get_all):
            return

        for ent in get_all():
            uri_str = getattr(ent, "uri", None)
            if not uri_str:
                # Fallback: derive from label
                label = getattr(ent, "label", None) or getattr(ent, "name", None) or "Entity"
                local = self._sanitize(label)
                uri_str = f"{self.namespaces['etd']}{local}"

            uri = _safe_uriref(str(uri_str), self.namespaces)

            # Ensure every entity is at least typed as etd:Entity
            self.graph.add((uri, RDF.type, etd_ns.Entity))


            # rdf:type based on entity_type
            entity_type = getattr(ent, "entity_type", None) or getattr(ent, "type", None)
            entity_class_was_corrected = False
            if entity_type:
                local_class = canonicalize_class_local(self._sanitize(str(entity_type)))
                pre_correction_class = local_class
                label_hint = getattr(ent, "label", None) or getattr(ent, "name", None)
                local_class = self._corrected_entity_class(local_class, label_hint)
                if local_class != pre_correction_class:
                    entity_class_was_corrected = True
                    record_event(
                        self.quality_events,
                        category="entity_class_correction",
                        action="corrected",
                        discourse_id=str((getattr(ent, "metadata", None) or {}).get("discourse_id") or "") or None,
                        subject_surface=str(label_hint or "").strip() or None,
                        object_surface=local_class,
                        reason=f"class-correction rule: {pre_correction_class} -> {local_class} (insufficient label evidence for the declared class)",
                    )
                self._seen_entity_classes.add(local_class)
                if local_class in self.semantic_typer.allowed_classes:
                    self.graph.add((uri, RDF.type, etd_ns[local_class]))


            # rdfs:label
            label = getattr(ent, "label", None) or getattr(ent, "name", None)
            if label:
                self.graph.add((uri, RDFS.label, Literal(label, lang="en")))
            meta_for_alias = getattr(ent, "metadata", None) or {}
            canonical_label = str(label or "").strip()
            canonical_key = _normalized_label_key(canonical_label) if canonical_label else ""
            alt_labels: Set[str] = set()
            hidden_labels: Set[str] = set()

            def _collect_alias_values(raw_val: Any, force_hidden: bool = False) -> None:
                if raw_val is None:
                    return
                if isinstance(raw_val, (list, tuple, set)):
                    values = list(raw_val)
                else:
                    values = [raw_val]
                for v in values:
                    s = str(v or "").strip()
                    if not s:
                        continue
                    if canonical_label and s == canonical_label:
                        continue
                    nkey = _normalized_label_key(s)
                    if not nkey:
                        continue
                    if force_hidden or (canonical_key and nkey == canonical_key):
                        hidden_labels.add(s)
                    else:
                        alt_labels.add(s)

            for k in ("aliases", "alias", "alt_labels", "alternative_labels"):
                _collect_alias_values(meta_for_alias.get(k), force_hidden=False)
            for k in ("hidden_labels", "noisy_aliases", "normalized_label", "label_normalized", "surface_form"):
                _collect_alias_values(meta_for_alias.get(k), force_hidden=True)

            for alt in sorted(alt_labels, key=lambda x: x.lower()):
                self.graph.add((uri, SKOS.altLabel, Literal(alt, lang="en")))
            for hidden in sorted(hidden_labels - alt_labels, key=lambda x: x.lower()):
                self.graph.add((uri, SKOS.hiddenLabel, Literal(hidden, lang="en")))

            # ----------------------------------------------------
            # Semantic Type Classification
            # ----------------------------------------------------
            # Runs unconditionally, once per entity, regardless of
            # incidental hidden-label count -- deliberately not nested
            # inside the hidden-label loop above, which would make it run
            # zero or multiple times depending on how many hidden labels an
            # entity happens to have. SemanticTyper._add_type also gates
            # every candidate against ontology-compatibility with types
            # already on the node, so this lexical fallback layer can never
            # union in types that conflict with the entity's already-
            # resolved primary type from entity_type above.
            if label:
                try:
                    self.semantic_typer.classify(uri, label.lower().strip())
                except Exception as e:
                    logger.warning("Semantic typing failed for %s: %s", label, e)


            # confidence (if any)
            conf = getattr(ent, "confidence", None)
            if conf is not None:
                self.graph.add((uri, etd_ns.confidence, self._confidence_literal(conf)))

            # ontology_uri (if mapped directly on entity)
            # Root-cause fix (bounded correction pass, item 5 -- stale
            # external-ontology typing surviving a label-driven local-
            # class correction): ent.ontology_uri/metadata["ontology_uri"]
            # is computed upstream, in pipeline.py, from the entity's
            # RAW (pre-correction) entity_type -- entirely independent of
            # _corrected_entity_class's rule engine above. Left as-is, a
            # corrected entity (e.g. mistyped "AlgorithmicBias" relabelled
            # to "AlgorithmicProcess" because the label carries no bias-
            # specific evidence) still gets the OLD external class (e.g.
            # airo:BiasRisk) materialised as an independent rdf:type at
            # line ~1439 below, via a code path _corrected_entity_class
            # never touches -- producing two simultaneously-asserted,
            # semantically incompatible types on one entity. When a
            # correction fired for this entity, treat its raw
            # ontology_uri as stale and fall back to the existing
            # heuristic ontology-alignment strategy for the CORRECTED
            # local_class instead (see _add_alignment_axioms), rather
            # than trust an external class computed for a declared type
            # this entity was just found not to actually support.
            onto_uri = None if entity_class_was_corrected else getattr(ent, "ontology_uri", None)
            metadata = getattr(ent, "metadata", None) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            if entity_class_was_corrected:
                # Root-cause fix (bounded preflight, item 5): suppressing
                # the LOCAL `onto_uri` variable above only stops THIS
                # block's own rdf:type/etd:ontology_uri emission --
                # _add_entity_metadata(), called later in this same
                # per-entity pass (and reusing this SAME metadata dict
                # object), independently re-reads
                # metadata["ontology_uri"] and would re-assert the stale
                # external class regardless, leaving an inconsistent
                # etd:ontology_uri triple with no matching rdf:type.
                # Clearing it here on the entity's own metadata (and the
                # ontology_uri/ontology_classes attributes, in case any
                # other consumer reads those directly) keeps every
                # downstream reader of this entity consistent with the
                # corrected class.
                metadata.pop("ontology_uri", None)
                metadata.pop("ontology_classes", None)
                if hasattr(ent, "ontology_uri"):
                    ent.ontology_uri = None
            if not str(metadata.get("surface_form") or "").strip():
                metadata["surface_form"] = str(getattr(ent, "label", "") or "").strip()
                # Task (Defect 1): this entity had no genuine extracted
                # surface_form of its own -- ent.label is used only as a
                # placeholder so downstream SHACL-facing fields are never
                # left blank. Tracked so _finalize_entity_provenance_
                # completion() can replace it with genuine relation-sourced
                # text later, once relations exist, rather than treating a
                # value being present here as "already correct".
                self._generic_surface_form_entities.add(uri)
            metadata.setdefault(
                "context_source",
                str(metadata.get("context_source") or getattr(ent, "source", "unknown") or "unknown").strip(),
            )
            metadata.setdefault(
                "source",
                str(metadata.get("source") or getattr(ent, "source", None) or metadata.get("context_source") or "unknown").strip(),
            )
            if not metadata.get("field") and str(entity_type or getattr(ent, "entity_type", "") or "") == "Vote":
                metadata["field"] = "vote.vote_value"
                metadata.setdefault("source_doc_id", metadata.get("discourse_id"))
                metadata.setdefault("chunk_index", -1)
                metadata.setdefault("chunk_index_unknown", True)
            if not onto_uri and not entity_class_was_corrected:
                onto_uri = metadata.get("ontology_uri")
            if not onto_uri and not entity_class_was_corrected:
                oc = metadata.get("ontology_classes")
                if isinstance(oc, list) and oc:
                    onto_uri = oc[0]
            if onto_uri:
                # ----------------------------------------------------
                # FIX: prevent ontology_uri being a stringified list
                # e.g. "['dcat:Dataset']" -> "dcat:Dataset"
                # ----------------------------------------------------
                if isinstance(onto_uri, list) and onto_uri:
                    onto_uri = onto_uri[0]

                if isinstance(onto_uri, str):
                    s = onto_uri.strip()

                    # stringified list
                    if s.startswith("[") and s.endswith("]"):
                        try:
                            parsed = ast.literal_eval(s)
                            if isinstance(parsed, list) and parsed:
                                onto_uri = parsed[0]
                        except Exception:
                            pass

                    # final string cleanup
                    if isinstance(onto_uri, str):
                        onto_uri = onto_uri.strip()

                onto_ref: Optional[Union[URIRef, Literal]] = self._resolve_entity_uri_from_value(onto_uri)
                onto_str = str(onto_uri).strip() if onto_uri is not None else ""

                # Ensure ontology grounding is materialized whenever ontology_uri exists.
                # Prefer full IRI/CURIE expansion; fallback to a stable ETD class URI.
                if not isinstance(onto_ref, URIRef) and onto_str:
                    if onto_str.startswith("http://") or onto_str.startswith("https://"):
                        onto_ref = _safe_uriref(onto_str, self.namespaces)
                    elif ":" in onto_str and " " not in onto_str:
                        prefix, local = onto_str.split(":", 1)
                        if prefix in self.namespaces:
                            onto_ref = _safe_uriref(self.namespaces[prefix] + local, self.namespaces)
                    if not isinstance(onto_ref, URIRef):
                        onto_ref = URIRef(f"{self.namespaces['etd']}{self._sanitize(onto_str)}")

                if onto_ref is None:
                    onto_ref = Literal(str(onto_uri), datatype=XSD.string)

                self.graph.add((uri, etd_ns.ontology_uri, onto_ref))
                if isinstance(onto_ref, URIRef):
                    # Materialized ontology grounding for SPARQL type queries.
                    self.graph.add((uri, RDF.type, onto_ref))

                # Record alignment target if ontology_uri is a real external class URI
                if entity_type and isinstance(onto_ref, URIRef):
                    local_class = canonicalize_class_local(self._sanitize(str(entity_type)))
                    self._maybe_record_alignment_target(local_class, onto_ref)
                src = metadata.get("ontology_mapping_source")
                if src:
                    self.graph.add((uri, etd_ns.ontologyMappingSource, Literal(str(src), datatype=XSD.string)))
                mconf = metadata.get("ontology_mapping_confidence")
                if mconf is not None:
                    try:
                        self.graph.add((uri, etd_ns.ontologyConfidence, Literal(float(mconf), datatype=XSD.float)))
                    except Exception:
                        pass


            # Entity-level metadata (including surface_form, context_source, discourse_id, etc.)
            self._add_entity_metadata(uri, metadata, etd_ns)
            raw_fields = metadata.get("raw_fields", metadata)
            kind_hint = str(entity_type or getattr(ent, "entity_type", "") or "").strip()
            self._emit_raw_fields(uri, kind_hint, raw_fields, etd_ns, provenance_dict=metadata)
            if kind_hint == "Vote":
                concept = metadata.get("vote_value_concept")
                concept_ref = self._resolve_entity_uri_from_value(concept)
                if concept_ref is None and isinstance(concept, str):
                    c = concept.strip()
                    if c and ":" not in c and not (c.startswith("http://") or c.startswith("https://")):
                        concept_ref = URIRef(f"{self.namespaces['etd']}{c.lstrip('#/')}")
                if isinstance(concept_ref, URIRef):
                    self.graph.add((uri, etd_ns.voteValue, concept_ref))

                polarity = metadata.get("vote_value")
                if polarity is not None and str(polarity).strip():
                    self.graph.add((uri, etd_ns.polarity, Literal(str(polarity), datatype=XSD.string)))
            # Wikidata is a special sub-case
            self._add_wikidata_metadata(uri, metadata, etd_ns)

            # ----------------------------------------------------
            # Production: guarantee every entity has an rdfs:label (human-friendly)
            # ----------------------------------------------------
            if not any(self.graph.objects(uri, RDFS.label)):
                meta = getattr(ent, "metadata", None) or {}

                # Prefer meaningful metadata labels if available
                fallback = (
                    (meta.get("surface_form") or "").strip()
                    or (meta.get("canonical_label") or "").strip()
                    or (meta.get("normalized_label") or "").strip()
                )

                if not fallback:
                    # Final fallback: last segment of URI (still better than missing label)
                    try:
                        fallback = str(uri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
                    except Exception:
                        fallback = "Unknown"

                # Clamp for UI safety
                if len(fallback) > 120:
                    fallback = fallback[:120]

                self.graph.add((uri, RDFS.label, Literal(fallback, lang="en")))

    def _coerce_int(self, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)

        s = str(v).strip()
        if not s:
            return None

        # stringified list like "[174, 717]"
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    return int(parsed[0])
            except Exception:
                pass

        # first integer substring
        m = re.search(r"-?\d+", s)
        if m:
            return int(m.group(0))

        return None

    def _coerce_nonneg_int(self, v: Any) -> Optional[int]:
        num = self._coerce_int(v)
        if num is None or int(num) < 0:
            return None
        return int(num)

    def _confidence_literal(self, value: Any) -> Literal:
        """
        Emit confidence as a strict xsd:float typed literal.
        """
        try:
            fval = float(value)
        except Exception:
            fval = 0.0
        return Literal(float(fval), datatype=XSD.float)

    # Modalities under which an unqualified positive direct <s> <p> <o>
    # triple would misrepresent the source proposition. Applies whether the
    # signal is carried on etd:epistemic_status OR inside etd:modality_tags.
    # The reified etd:Relation node (with etd:negated / etd:epistemic_status
    # / etd:modality_tags) and its Evidence are always kept regardless --
    # only the plain direct triple is withheld.
    _NON_ASSERTED_MODALITIES = frozenset({
        "questioned", "hypothetical", "uncertain",
        "conditional", "normative", "reported",
    })
    # Epistemic-status values that DO represent an asserted proposition.
    # Missing / legacy-empty is treated as asserted for backward
    # compatibility (no project policy says otherwise).
    _ASSERTED_EPISTEMIC_STATUSES = frozenset({"asserted", ""})

    # Kept as an alias so external callers / tests referencing the old name
    # still see the tightened set.
    _NON_ASSERTED_EPISTEMIC_STATUSES = _NON_ASSERTED_MODALITIES

    @staticmethod
    def _modality_tag_tokens(raw: Any) -> Set[str]:
        """Tokenise an etd:modality_tags value into a lowercased token set.
        Accepts a list/tuple/set or a single string with '|', ',', ';' or
        '/' separators (e.g. 'questioned|conditional')."""
        if raw is None:
            return set()
        items = raw if isinstance(raw, (list, tuple, set)) else re.split(r"[|,;/]+", str(raw))
        return {str(t).strip().lower() for t in items if str(t).strip()}

    def _relation_provides_positive_asserted_support(
        self,
        *,
        negated: Any,
        epistemic_status: Any,
        modality_tags: Any,
    ) -> bool:
        """THE single rule shared by direct-SPO emission and
        _enforce_direct_spo_assertion_invariant. A relation provides
        positive asserted support for its exact (subject, predicate,
        object) iff ALL of:
          * it is not negated;
          * its epistemic status represents an asserted proposition
            (explicit 'asserted', or missing/legacy-empty for
            compatibility);
          * its modality_tags carry no non-asserted modality
            (questioned / hypothetical / uncertain / conditional /
            normative / reported).
        Otherwise the proposition is preserved on the reified node but must
        not authorise an unqualified direct triple."""
        if self._coerce_bool(negated):
            return False
        status = str(epistemic_status or "").strip().lower()
        if status not in self._ASSERTED_EPISTEMIC_STATUSES:
            return False
        if self._modality_tag_tokens(modality_tags) & self._NON_ASSERTED_MODALITIES:
            return False
        return True

    def _direct_spo_suppressed_by_assertion_status(self, meta: Dict[str, Any]) -> bool:
        """True when the direct SPO triple must be withheld because this
        relation does not provide positive asserted support for it (see
        call site in _add_relations). Reified node + Evidence stay."""
        return not self._relation_provides_positive_asserted_support(
            negated=meta.get("negated"),
            epistemic_status=meta.get("epistemic_status"),
            modality_tags=meta.get("modality_tags"),
        )

    def _enforce_direct_spo_assertion_invariant(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Root-cause fix (bounded correction pass, item 2). Final,
        authoritative enforcement of the invariant: a direct positive
        SPO triple exists only if at least one surviving, valid,
        positively-asserted relation supports it. Runs AFTER
        consolidate_duplicate_reified_relations, over the graph's FINAL
        set of surviving reified relations, so it catches an orphaned
        direct triple regardless of the specific mechanism that
        orphaned it (a consolidation merge choosing a negated/non-
        asserted duplicate as winner over a positively-asserted one
        that had already emitted its own direct triple; any future
        change to consolidation/re-linking that repoints or removes a
        relation). Deliberately scoped to
        relation_kind="extracted_semantic": structural backbone
        relations carry no negated/epistemic_status semantics.

        Grouping is by (subject, predicate, object) ONLY (unlike
        consolidation's discourse-scoped key) -- this is the identity
        that determines whether a specific direct triple is present in
        the graph at all, independent of which discourse(s) assert it.
        A group with at least one positively-asserted member keeps its
        direct triple (other, differently-qualified members of the same
        group -- negated or non-asserted restatements -- remain as
        their own, separate reified relations; nothing about them is
        touched here). A group with NO positively-asserted member has
        its direct triple removed if present.
        """
        graph = self.graph
        rel_type = etd_ns.Relation
        groups: Dict[Tuple[Any, Any, Any], List[URIRef]] = {}
        for rel in graph.subjects(RDF.type, rel_type):
            if not isinstance(rel, URIRef):
                continue
            kind_vals = [str(v) for v in graph.objects(rel, etd_ns.relation_kind)]
            if kind_vals and kind_vals[0] != "extracted_semantic":
                continue
            s_vals = [v for v in graph.objects(rel, etd_ns.subject) if isinstance(v, URIRef)]
            p_vals = [v for v in graph.objects(rel, etd_ns.predicate) if isinstance(v, URIRef)]
            o_vals = list(graph.objects(rel, etd_ns.object))
            if len(s_vals) != 1 or len(p_vals) != 1 or len(o_vals) != 1:
                continue
            key = (s_vals[0], p_vals[0], o_vals[0])
            groups.setdefault(key, []).append(rel)

        checked = 0
        removed = 0
        for (s, p, o), rels in groups.items():
            checked += 1
            has_positive = False
            for rel in rels:
                negated = any(
                    str(v).strip().lower() in {"true", "1"} for v in graph.objects(rel, etd_ns.negated)
                )
                epi_vals = [str(v) for v in graph.objects(rel, etd_ns.epistemic_status)]
                mt_vals = [str(v) for v in graph.objects(rel, etd_ns.modality_tags)]
                if self._relation_provides_positive_asserted_support(
                    negated=negated,
                    epistemic_status=(epi_vals[0] if epi_vals else ""),
                    modality_tags=(mt_vals[0] if mt_vals else ""),
                ):
                    has_positive = True
                    break
            if has_positive:
                continue
            if (s, p, o) in graph:
                graph.remove((s, p, o))
                removed += 1
                rel_ids = [str(r) for r in rels]
                record_event(
                    self.quality_events,
                    category="non_asserted_direct_spo_removed",
                    action="dropped",
                    source_relation_id=rel_ids[0] if rel_ids else None,
                    subject_surface=str(s),
                    predicate_surface=_uri_localname(p) if isinstance(p, URIRef) else str(p),
                    object_surface=str(o),
                    reason="no surviving relation for this (subject, predicate, object) is positively asserted; removed orphaned direct triple",
                    replacement_relation_ids=rel_ids or None,
                )
        return {"spo_groups_checked": checked, "orphaned_direct_triples_removed": removed}

    def _enforce_endpoint_surface_compatibility_final(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Root-cause fix (bounded correction pass, item 1). Authoritative,
        final endpoint/surface-form compatibility check, run after
        consolidation -- i.e. after every operation in this pipeline
        that can change which relation node/endpoint survives. Reuses
        the EXACT SAME multi-signal matching logic as the per-relation
        check in _add_relations (_endpoint_surface_form_compatible), so
        the two can never disagree; this pass differs only in reading
        the FINAL persisted triples instead of in-flight metadata, and
        in attempting recovery before dropping.

        On a confirmed mismatch, before dropping, looks for exactly one
        OTHER entity already associated with this relation's own
        discourse (via hasSourceEntity/subject/object on any relation
        sharing that discourse_id) whose label is plausibly compatible
        with the captured surface form; if found, the relation is
        repointed to it (reified subject/object triple rewritten, and
        the direct SPO triple too, if one existed) instead of being
        dropped. Only when no such unambiguous candidate exists is the
        relation dropped entirely (reified triples and any direct
        triple removed) -- never a silent drop when a valid grounded
        endpoint could be recovered, and never a guess when more than
        one candidate is plausible.
        """
        graph = self.graph
        rel_type = etd_ns.Relation

        def _relation_discourse_ids(rel: URIRef) -> Set[str]:
            dids = {str(v) for v in graph.objects(rel, etd_ns.discourse_id)}
            if dids:
                return dids
            for ev in graph.objects(rel, etd_ns.hasEvidence):
                dids.update(str(v) for v in graph.objects(ev, etd_ns.discourse_id))
            return dids

        discourse_entities: Dict[str, Set[URIRef]] = {}
        all_semantic_rels = [
            r for r in graph.subjects(RDF.type, rel_type)
            if isinstance(r, URIRef)
            and (lambda kv: not kv or kv[0] == "extracted_semantic")(
                [str(v) for v in graph.objects(r, etd_ns.relation_kind)]
            )
        ]
        for rel in all_semantic_rels:
            dids = _relation_discourse_ids(rel)
            if not dids:
                continue
            candidates: Set[URIRef] = set()
            for pred in (etd_ns.subject, etd_ns.object, etd_ns.hasSourceEntity):
                for v in graph.objects(rel, pred):
                    if isinstance(v, URIRef):
                        candidates.add(v)
            for did in dids:
                discourse_entities.setdefault(did, set()).update(candidates)

        def _find_recovery(candidate_pool: Set[URIRef], bad_node: Any, surface: str) -> Optional[URIRef]:
            if not surface:
                return None
            surface_tokens = self._surface_match_tokens(surface)
            if not surface_tokens:
                return None
            matches = []
            for cand in candidate_pool:
                if cand == bad_node:
                    continue
                cand_label = next((str(v) for v in graph.objects(cand, RDFS.label)), "")
                if not cand_label:
                    continue
                if self._surface_match_tokens(cand_label) & surface_tokens:
                    matches.append(cand)
            return matches[0] if len(matches) == 1 else None

        checked = 0
        recovered = 0
        dropped = 0
        for rel in all_semantic_rels:
            s_vals = [v for v in graph.objects(rel, etd_ns.subject) if isinstance(v, URIRef)]
            p_vals = [v for v in graph.objects(rel, etd_ns.predicate) if isinstance(v, URIRef)]
            o_vals = list(graph.objects(rel, etd_ns.object))
            if len(s_vals) != 1 or len(p_vals) != 1 or len(o_vals) != 1:
                continue
            s, p, o = s_vals[0], p_vals[0], o_vals[0]
            subj_surface = next((str(v) for v in graph.objects(rel, etd_ns.subject_surface_form)), "")
            obj_surface = next((str(v) for v in graph.objects(rel, etd_ns.object_surface_form)), "")
            meta = {"subject_surface_form": subj_surface, "object_surface_form": obj_surface}
            checked += 1
            subj_ok = self._endpoint_surface_form_compatible(s, meta, "subject")
            obj_ok = self._endpoint_surface_form_compatible(o, meta, "object") if isinstance(o, URIRef) else True
            if subj_ok and obj_ok:
                continue

            dids = _relation_discourse_ids(rel)
            candidate_pool: Set[URIRef] = set()
            for did in dids:
                candidate_pool.update(discourse_entities.get(did, set()))

            new_s, new_o = s, o
            did_recover = False
            if not subj_ok:
                cand = _find_recovery(candidate_pool, s, subj_surface)
                if cand is not None:
                    new_s = cand
                    did_recover = True
            if not obj_ok and isinstance(o, URIRef):
                cand = _find_recovery(candidate_pool, o, obj_surface)
                if cand is not None:
                    new_o = cand
                    did_recover = True

            recheck_meta = {"subject_surface_form": subj_surface, "object_surface_form": obj_surface}
            still_bad = (
                not self._endpoint_surface_form_compatible(new_s, recheck_meta, "subject")
                or (
                    isinstance(new_o, URIRef)
                    and not self._endpoint_surface_form_compatible(new_o, recheck_meta, "object")
                )
            )

            pred_local = _uri_localname(p)
            if did_recover and not still_bad:
                if (s, p, o) in graph:
                    graph.remove((s, p, o))
                    graph.add((new_s, p, new_o))
                if (rel, etd_ns.subject, s) in graph:
                    graph.remove((rel, etd_ns.subject, s))
                    graph.add((rel, etd_ns.subject, new_s))
                if (rel, etd_ns.object, o) in graph:
                    graph.remove((rel, etd_ns.object, o))
                    graph.add((rel, etd_ns.object, new_o))
                recovered += 1
                record_event(
                    self.quality_events,
                    category="endpoint_surface_mismatch_recovered",
                    action="recovered",
                    discourse_id=next(iter(dids), None),
                    source_relation_id=str(rel),
                    subject_surface=subj_surface or None,
                    predicate_surface=pred_local,
                    object_surface=obj_surface or None,
                    reason="repointed to a discourse-scoped entity whose label matches the captured surface form",
                )
                continue

            if (s, p, o) in graph:
                graph.remove((s, p, o))
            for t in list(graph.triples((rel, None, None))):
                graph.remove(t)
            for t in list(graph.triples((None, None, rel))):
                graph.remove(t)
            dropped += 1
            record_event(
                self.quality_events,
                category="endpoint_surface_mismatch_drop",
                action="dropped",
                discourse_id=next(iter(dids), None),
                source_relation_id=str(rel),
                subject_surface=subj_surface or None,
                predicate_surface=pred_local,
                object_surface=obj_surface or None,
                reason="final invariant check found a confirmed subject/object surface-form mismatch with no unambiguous recoverable endpoint in this discourse",
            )

        return {"relations_checked": checked, "relations_recovered": recovered, "relations_dropped": dropped}

    def _reconcile_orphan_evidence_nodes(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Root-cause fix (targeted correction pass, item 1 -- orphan
        Evidence lifecycle). Final, graph-wide, reference-aware Evidence
        cleanup. Must run LAST -- after endpoint recovery
        (_enforce_endpoint_surface_compatibility_final), consolidation
        (consolidate_duplicate_reified_relations), and the direct-SPO
        invariant (_enforce_direct_spo_assertion_invariant) -- since any
        of those can remove a relation's own triples (including its
        etd:hasEvidence link) without itself knowing whether the
        Evidence node it stopped referencing is still needed by some
        OTHER surviving relation. Rather than trying to track "is this
        evidence still used" separately at every removal site (fragile,
        easy to miss a future one), this pass re-derives the answer
        authoritatively from the FINAL graph state, once, the same way
        scripts/kg_quality_gate.py itself measures orphan_evidence_count
        (see that script's `evidence_inbound` computation) -- so this
        method's own postcondition and the audit gate's pass/fail
        criterion can never disagree.

        An Evidence node is inbound-referenced if, and only if, at least
        one SURVIVING etd:Relation node has an etd:hasEvidence triple
        pointing at it -- exactly kg_quality_gate.py's own definition.
        Evidence shared by two or more relations keeps a positive
        inbound count as long as ANY one of them survives, so it is
        never touched merely because one of its owners was dropped;
        only a genuine zero-inbound (truly orphaned) Evidence node is
        deleted, and only that node's own triples are removed -- nothing
        about the surviving relation(s)/entities that still reference
        other Evidence is touched.
        """
        graph = self.graph
        evidence_nodes = set(graph.subjects(RDF.type, etd_ns.Evidence))
        if not evidence_nodes:
            return {"evidence_nodes_checked": 0, "orphaned_evidence_removed": 0}

        inbound: Dict[URIRef, int] = {}
        for rel in graph.subjects(RDF.type, etd_ns.Relation):
            if not isinstance(rel, URIRef):
                continue
            for ev in graph.objects(rel, etd_ns.hasEvidence):
                if isinstance(ev, URIRef):
                    inbound[ev] = inbound.get(ev, 0) + 1

        removed = 0
        for ev in evidence_nodes:
            if inbound.get(ev, 0) > 0:
                continue
            for t in list(graph.triples((ev, None, None))):
                graph.remove(t)
            for t in list(graph.triples((None, None, ev))):
                graph.remove(t)
            removed += 1
            record_event(
                self.quality_events,
                category="orphan_evidence_removed",
                action="dropped",
                source_relation_id=None,
                reason="Evidence node had zero surviving etd:hasEvidence referencers after endpoint recovery/consolidation/direct-SPO reconciliation",
            )

        return {"evidence_nodes_checked": len(evidence_nodes), "orphaned_evidence_removed": removed}

    def _corrected_entity_class(self, local_class: str, label: Optional[str]) -> str:
        """
        CORE, domain-independent mechanism (manual semantic-QA pass,
        category H; made domain-agnostic in the subsequent architectural-
        cleanup pass): a "loaded/specialized" ontology class requires its
        own textual evidence before being asserted; absent that evidence,
        fall back to a configured, less-specific class in the same branch
        of the hierarchy.

        This method is a pure rule INTERPRETER over
        self.domain_profile.class_correction_rules (see
        domain_profiles.py::ClassCorrectionRule) -- it does not know the
        name "AlgorithmicBias", "AISystem", or any other domain class
        itself. The active domain profile supplies, per rule: which
        declared class it governs, what signal justifies (or disqualifies)
        it, and which fallback class(es) to use instead. Every class/label
        combination not covered by a rule passes through unchanged.
        """
        text = str(label or "")
        for rule in self.domain_profile.class_correction_rules:
            if local_class != rule.declared_class:
                continue
            if rule.disqualifying_signal is not None and rule.disqualifying_signal.search(text):
                return rule.fallback_class
            if rule.required_signal is not None and not rule.required_signal.search(text):
                if rule.secondary_signal is not None and rule.secondary_signal.search(text):
                    return rule.secondary_fallback_class or rule.fallback_class
                return rule.fallback_class
        return local_class

    # Category B (manual semantic-QA pass): a bare pronoun/deictic surface
    # form is exempt from the endpoint/surface consistency check -- its
    # resolution is governed separately by the pipeline's own anaphora-
    # resolution guard, not by lexical overlap with a label.
    _ENDPOINT_PRONOUN_SURFACES = frozenset({
        "it", "its", "itself", "this", "that", "these", "those", "they",
        "them", "their", "themselves", "he", "she", "him", "her",
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
        "ourselves", "you", "your", "yours", "yourself", "yourselves",
    })
    _SURFACE_MATCH_STOPWORDS = frozenset({
        "a", "an", "the", "of", "in", "on", "for", "to", "and", "or",
        "with", "by", "is", "are", "was", "were", "this", "that",
        "these", "those",
    })

    def _surface_match_tokens(self, text: Any) -> Set[str]:
        norm = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
        return {t for t in norm.split() if t and t not in self._SURFACE_MATCH_STOPWORDS}

    def _endpoint_surface_form_compatible(self, node: Any, meta: Dict[str, Any], side: str) -> bool:
        """
        Root-cause fix (manual semantic-QA pass, category B): if a relation
        endpoint resolves to a canonical entity URI, that entity's own
        label must be plausibly compatible with the SURFACE TEXT the
        extractor actually captured for this endpoint
        (subject_surface_form / object_surface_form) -- otherwise some
        upstream stage (canonical linking, entity consolidation, anaphora
        resolution, or a colliding normalized-URI computation -- any of
        these can independently repoint an endpoint at the wrong entity)
        has attached this relation to an entity the extracted text does
        not actually denote.

        Multi-signal, not naive exact-string equality:
          - normalized-token overlap (handles case/whitespace/article
            differences, and legitimate partial mentions)
          - acronym match in either direction
          - alias-list membership (entity.metadata["aliases"])

        Returns True (compatible / not checkable) whenever the endpoint is
        not a URIRef, no surface form was captured, the surface form is a
        pronoun (governed separately, see _ENDPOINT_PRONOUN_SURFACES), or
        no entity record is found for the URI -- this check only ever
        REJECTS a positively-confirmed mismatch, never invents a judgement
        from missing data. The caller drops the whole relation on a
        confirmed mismatch rather than re-resolving it to a guessed
        alternative entity, since no evidence establishes what the correct
        endpoint should have been.
        """
        if not isinstance(node, URIRef):
            return True
        surface = str(meta.get(f"{side}_surface_form") or "").strip()
        if not surface:
            return True
        if surface.lower() in self._ENDPOINT_PRONOUN_SURFACES:
            return True

        # Root-cause fix (bounded correction pass, item 1): an entity
        # whose OWN label is a synthetic URI-slug placeholder --
        # _add_entities() had no genuine extracted surface_form for it
        # and fell back to ent.label purely so downstream SHACL-facing
        # fields are never blank (see _generic_surface_form_entities) --
        # carries no genuine claim about what it denotes. Comparing a
        # real relation's surface text against that placeholder always
        # looks like a "mismatch" (the placeholder is definitionally
        # not descriptive text), which would incorrectly flag/drop a
        # perfectly valid relation whose only fault is referencing an
        # entity that was first minted as a bare relation endpoint. Not
        # checkable, so treated as compatible, same as "no label".
        if node in self._generic_surface_form_entities:
            return True

        ent = self.entity_index.get(str(node))
        if ent is None:
            return True
        label = str(getattr(ent, "label", "") or "").strip()
        if not label:
            return True

        surface_tokens = self._surface_match_tokens(surface)
        label_tokens = self._surface_match_tokens(label)
        if not surface_tokens or not label_tokens:
            return True
        if surface_tokens & label_tokens:
            return True

        # Acronym check, either direction -- e.g. label "Artificial
        # Intelligence" / surface "AI", or label "General Data Protection
        # Regulation" / surface "the GDPR". Checked as token MEMBERSHIP
        # (the acronym appears as one whole word in the other phrase),
        # not whole-string equality, so a determiner/qualifier alongside
        # the acronym ("the GDPR") does not defeat the match.
        surface_words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", surface).split() if w]
        label_words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", label).split() if w]
        surface_words_lower = {w.lower() for w in surface_words}
        label_words_lower = {w.lower() for w in label_words}
        if label_words:
            label_acronym = "".join(w[0] for w in label_words).lower()
            if len(label_acronym) >= 2 and label_acronym in surface_words_lower:
                return True
        if surface_words:
            surface_acronym = "".join(w[0] for w in surface_words).lower()
            if len(surface_acronym) >= 2 and surface_acronym in label_words_lower:
                return True

        ent_meta = getattr(ent, "metadata", None) or {}
        aliases = ent_meta.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                if self._surface_match_tokens(alias) & surface_tokens:
                    return True

        return False

    def _coerce_bool(self, v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(int(v))
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        return None

    def _registered_etd_predicate(self, local_name: str, etd_ns: Namespace) -> Optional[URIRef]:
        """
        Resolve ETD predicate only if it is explicitly registered.
        """
        key = str(local_name or "").strip()
        if not key:
            return None
        if not is_registered_etd_predicate(key):
            msg = f"Unregistered ETD predicate emission blocked: etd:{key}"
            if self.strict_etd_property_registry:
                raise RuntimeError(msg)
            logger.warning(msg)
            return None
        return etd_ns[key]

    def _emit_registered_etd_literal(
        self,
        subject: Union[URIRef, BNode],
        local_name: str,
        value: Any,
        etd_ns: Namespace,
        *,
        datatype: Optional[URIRef] = None,
        lang: Optional[str] = None,
    ) -> None:
        pred = self._registered_etd_predicate(local_name, etd_ns)
        if pred is None:
            return
        if lang is not None:
            self.graph.add((subject, pred, Literal(str(value), lang=lang)))
            return
        if datatype is not None:
            self.graph.add((subject, pred, Literal(value, datatype=datatype)))
            return
        self.graph.add((subject, pred, Literal(value)))

    def _emit_registered_etd_object(
        self,
        subject: Union[URIRef, BNode],
        local_name: str,
        obj: URIRef,
        etd_ns: Namespace,
    ) -> None:
        pred = self._registered_etd_predicate(local_name, etd_ns)
        if pred is None:
            return
        self.graph.add((subject, pred, obj))

    def _replace_registered_etd_literal(
        self,
        subject: Union[URIRef, BNode],
        local_name: str,
        value: Any,
        etd_ns: Namespace,
        *,
        datatype: Optional[URIRef] = None,
        lang: Optional[str] = None,
    ) -> None:
        pred = self._registered_etd_predicate(local_name, etd_ns)
        if pred is None:
            return
        for old in list(self.graph.objects(subject, pred)):
            self.graph.remove((subject, pred, old))
        self._emit_registered_etd_literal(subject, local_name, value, etd_ns, datatype=datatype, lang=lang)

    def _pick_single_value(self, val: Any) -> Any:
        """
        Deterministically pick one value:
        - scalar -> scalar
        - list/tuple/set -> lexicographically smallest non-empty string representation
        """
        if isinstance(val, (list, tuple, set)):
            candidates = sorted(
                [str(x).strip() for x in val if str(x).strip()],
                key=lambda x: x.lower(),
            )
            return candidates[0] if candidates else None
        return val

    def _coerce_datetime_literal(self, raw_val: Any, field_key: str) -> Optional[Literal]:
        if raw_val is None:
            return None
        raw = str(raw_val).strip()
        if not raw:
            return None
        candidate = raw
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            norm = dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
            return Literal(norm, datatype=XSD.dateTime)
        except Exception:
            if field_key not in self._raw_datetime_warning_keys:
                logger.warning(
                    "Raw field '%s' datetime parse failed; emitting xsd:string fallback",
                    field_key,
                )
                self._raw_datetime_warning_keys.add(field_key)
            return Literal(raw, datatype=XSD.string)

    def _looks_like_url(self, s: str) -> bool:
        if not s:
            return False
        try:
            p = urlparse(s)
            return p.scheme in {"http", "https"} and bool(p.netloc)
        except Exception:
            return False

    def _emit_raw_field_with_provenance(
        self,
        subject_uri: URIRef,
        predicate_uri: URIRef,
        object_node: Node,
        field_name: str,
        record_id: str,
        section_name: str,
        raw_dict: Dict[str, Any],
        etd_ns: Namespace,
    ) -> None:
        if not self.materialize_raw_fields:
            return
        self.graph.add((subject_uri, predicate_uri, object_node))
        prov_key = f"{str(subject_uri)}|{str(predicate_uri)}|{field_name}|{record_id}"
        field_node = URIRef(f"{self.namespaces['etd']}field_{safe_hash(prov_key)}")

        self.graph.add((field_node, RDF.type, etd_ns.FieldAssertion))
        self.graph.add((subject_uri, etd_ns.hasFieldAssertion, field_node))
        self.graph.add((field_node, etd_ns.fieldName, Literal(str(field_name), datatype=XSD.string)))
        self.graph.add((field_node, etd_ns.recordId, Literal(str(record_id), datatype=XSD.string)))
        self.graph.add((field_node, etd_ns.sourceSection, Literal(str(section_name), datatype=XSD.string)))
        self.graph.add((field_node, etd_ns.assertedPredicate, predicate_uri))
        self.graph.add((field_node, etd_ns.assertedValue, object_node))
        self.graph.add((field_node, etd_ns.datasetId, Literal(self.dataset_id, datatype=XSD.string)))
        self.graph.add((field_node, etd_ns.runId, Literal(self.run_id, datatype=XSD.string)))
        self.graph.add((field_node, PROV.generatedAtTime, Literal(self.generated_at_time, datatype=XSD.dateTime)))
        self.graph.add((field_node, PROV.wasDerivedFrom, self.export_node))

        source_val = raw_dict.get("source") or raw_dict.get("context_source")
        if source_val is not None and str(source_val).strip():
            self.graph.add((field_node, etd_ns.source, Literal(str(source_val).strip(), datatype=XSD.string)))
        if raw_dict.get("discourse_id") is not None:
            self._add_discourse_id_literal(field_node, raw_dict.get("discourse_id"), etd_ns)
        origin_node = self._ensure_discourse_unit_node(raw_dict, etd_ns)
        if origin_node is not None:
            self.graph.add((field_node, etd_ns.extractedFrom, origin_node))
        if raw_dict.get("chunk_index") is not None:
            cidx = self._coerce_int(raw_dict.get("chunk_index"))
            if cidx is not None:
                self.graph.add((field_node, etd_ns.chunk_index, Literal(cidx, datatype=XSD.integer)))
                self.graph.add((field_node, etd_ns.chunk_index_unknown, Literal(bool(int(cidx) < 0), datatype=XSD.boolean)))
        elif raw_dict.get("chunk_index_unknown") is not None:
            bidx = self._coerce_bool(raw_dict.get("chunk_index_unknown"))
            if bidx is not None:
                self.graph.add((field_node, etd_ns.chunk_index_unknown, Literal(bool(bidx), datatype=XSD.boolean)))

    def _emit_raw_fields(
        self,
        node: URIRef,
        kind: str,
        raw_dict: Dict[str, Any],
        etd_ns: Namespace,
        provenance_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.materialize_raw_fields:
            return
        if not isinstance(raw_dict, dict) or not raw_dict:
            return
        provenance_context: Dict[str, Any] = {}
        if isinstance(provenance_dict, dict):
            provenance_context.update(provenance_dict)
        provenance_context.update(raw_dict)

        allowlist: Dict[str, Dict[str, str]] = {
            "Theme": {
                "theme_id": "themeId",
                "created_at": "createdAt",
                "views": "views",
                "expert_id": "expertId",
                "problem_category": "problemCategory",
                "model_category": "modelCategory",
                "domain_category": "domainCategory",
                "name": "name",
                "description": "description",
            },
            "Question": {
                "created_at": "createdAt",
                "views": "views",
                "expert_id": "expertId",
                "theme_id": "themeId",
                "title": "title",
                "body": "body",
            },
            "Answer": {
                "created_at": "createdAt",
                "question_id": "questionId",
                "expert_id": "expertId",
                "parent_id": "parentId",
                "description": "description",
            },
            "Vote": {
                "created_at": "createdAt",
                "updated_at": "updatedAt",
                "expert_id": "expertId",
                "answer_id": "answerId",
                "vote_value": "rawVoteValue",
            },
            "Expert": {
                "user_id": "userId",
                "is_deleted": "isDeleted",
                "area_of_expertise": "areaOfExpertise",
                "date_joined": "dateJoined",
                "profile_picture": "profilePicture",
                "bio": "bio",
            },
        }
        section_map = {
            "Theme": "themes",
            "Question": "questions",
            "Answer": "answers",
            "Vote": "votes",
            "Expert": "experts",
        }
        section_name = section_map.get(kind, "unknown")

        def _record_id_for_kind() -> str:
            candidates_by_kind = {
                "Theme": ["id", "theme_id", "discourse_id"],
                "Question": ["id", "question_id", "discourse_id"],
                "Answer": ["id", "answer_id", "discourse_id"],
                "Vote": ["id", "vote_id", "discourse_id"],
                "Expert": ["id", "expert_id", "user_id", "discourse_id"],
            }
            for key in candidates_by_kind.get(kind, []):
                val = provenance_context.get(key)
                if val is not None and str(val).strip():
                    return str(self._pick_single_value(val)).strip()
            return _uri_localname(node)

        record_id = _record_id_for_kind()

        datetime_fields = {"created_at", "updated_at", "date_joined"}
        int_fields = {
            "views",
            "expert_id",
            "theme_id",
            "question_id",
            "answer_id",
            "parent_id",
            "user_id",
            "vote_value",
        }
        bool_fields = {"is_deleted"}
        string_fields = {
            "problem_category",
            "model_category",
            "domain_category",
            "area_of_expertise",
            "bio",
            "name",
            "title",
            "body",
            "description",
        }

        kind_map = allowlist.get(kind, {})
        for raw_key, pred_local in kind_map.items():
            if raw_key == "bio" and not self.materialize_sensitive_fields:
                continue
            if raw_key not in raw_dict:
                continue

            picked = self._pick_single_value(raw_dict.get(raw_key))
            if picked is None:
                continue
            if isinstance(picked, str) and not picked.strip():
                continue

            pred = self._registered_etd_predicate(pred_local, etd_ns)
            if pred is None:
                continue
            for old in list(self.graph.objects(node, pred)):
                self.graph.remove((node, pred, old))

            if raw_key in datetime_fields:
                lit = self._coerce_datetime_literal(picked, raw_key)
                if lit is not None:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=lit,
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=provenance_context,
                        etd_ns=etd_ns,
                    )
                continue

            if raw_key in int_fields:
                ival = self._coerce_int(picked)
                if ival is not None:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=Literal(ival, datatype=XSD.integer),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=provenance_context,
                        etd_ns=etd_ns,
                    )
                continue

            if raw_key in bool_fields:
                bval = self._coerce_bool(picked)
                if bval is not None:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=Literal(bval, datatype=XSD.boolean),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=provenance_context,
                        etd_ns=etd_ns,
                    )
                continue

            if raw_key == "profile_picture":
                sval = str(picked).strip()
                if not sval:
                    continue
                if self._looks_like_url(sval):
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=URIRef(sval),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=provenance_context,
                        etd_ns=etd_ns,
                    )
                else:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=Literal(sval, datatype=XSD.string),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=provenance_context,
                        etd_ns=etd_ns,
                    )
                    if self.media_base_url:
                        base = self.media_base_url.rstrip("/")
                        rel = sval.lstrip("/")
                        self.graph.add((node, PROV.atLocation, URIRef(f"{base}/{rel}")))
                continue

            if raw_key in string_fields:
                s = str(picked).strip()
                if s:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=Literal(s, datatype=XSD.string),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=provenance_context,
                        etd_ns=etd_ns,
                    )

    def _add_discourse_id_literal(self, subject: Union[URIRef, BNode], did: Any, etd_ns: Namespace) -> None:
        """
        Emit etd:discourse_id with robust typing:
        - None/empty -> skip
        - numeric -> xsd:integer
        - non-numeric -> warn and emit xsd:string
        """
        if did is None:
            return
        raw = str(did).strip()
        if not raw:
            return
        num = self._coerce_int(raw)
        if num is not None:
            self.graph.add((subject, etd_ns.discourse_id, Literal(int(num), datatype=XSD.integer)))
            return
        logger.warning(
            "[RDFSerializer] Non-numeric discourse_id '%s'; emitting as xsd:string",
            raw,
        )
        self.graph.add((subject, etd_ns.discourse_id, Literal(raw, datatype=XSD.string)))

    def _resolve_origin_discourse_uri(self, metadata: Optional[Dict[str, Any]]) -> Optional[URIRef]:
        """
        Resolve deterministic DiscourseUnit URI from provenance metadata.
        Uses origin_key when available, else field|discourse_id, else discourse_id.
        """
        if not isinstance(metadata, dict):
            return None
        origin_key = metadata.get("origin_key")
        if origin_key is not None and str(origin_key).strip():
            token = str(origin_key).strip()
        else:
            did_raw = metadata.get("discourse_id") or metadata.get("doc_id")
            did = str(did_raw).strip() if did_raw is not None else ""
            if not did:
                return None
            field = metadata.get("field")
            fld = str(field).strip() if field is not None else ""
            token = f"{fld}|{did}" if fld else did
        safe = make_uri_safe(token)
        if not safe:
            return None
        return URIRef(f"{self.namespaces['etd']}discourse_{safe}")

    @staticmethod
    def _norm_for_span_match(s: Any) -> str:
        return " ".join(str(s or "").strip().split()).lower()

    def _mention_span_verifies(
        self,
        mention: Dict[str, Any],
        *,
        surface_form: Optional[str],
        aliases: Optional[List[str]],
    ) -> Optional[bool]:
        """True/False when a source-text resolver is available and the
        mention's (field, discourse_id) resolves; None when it cannot be
        checked (no resolver, or the pair does not resolve -- i.e. a
        nonexistent source record)."""
        if self._source_text_lookup is None:
            return None
        field = mention.get("field")
        did = mention.get("discourse_id")
        try:
            text = self._source_text_lookup(field, did)
        except Exception:
            text = None
        if not text:
            return None
        try:
            s, e = int(mention.get("start_char")), int(mention.get("end_char"))
        except (TypeError, ValueError):
            return False
        if s < 0 or e <= s or e > len(text):
            return False
        excerpt = self._norm_for_span_match(text[s:e])
        if surface_form is not None and excerpt == self._norm_for_span_match(surface_form):
            return True
        for alt in aliases or []:
            if excerpt == self._norm_for_span_match(alt):
                return True
        return False

    def _primary_mention(
        self,
        metadata: Optional[Dict[str, Any]],
        *,
        surface_form: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        verify: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(metadata, dict):
            return None
        mentions = metadata.get("mentions")
        if not isinstance(mentions, list) or not mentions:
            # Single-occurrence entity with no explicit mentions list: fall
            # back to the top-level source-identity scalars so C.2/C.3
            # verification still applies to it.
            scalar_mention = mention_from_source_identity(metadata)
            mentions = [scalar_mention] if scalar_mention is not None else []
        if not mentions:
            return None
        top_field = str(metadata.get("field") or metadata.get("source_field") or "").strip()
        normalized: List[Dict[str, Any]] = []
        for item in mentions:
            if not isinstance(item, dict):
                continue
            did_raw = item.get("discourse_id")
            did = str(did_raw).strip() if did_raw is not None else ""
            start = self._coerce_nonneg_int(item.get("start_char"))
            end = self._coerce_nonneg_int(item.get("end_char"))
            chunk = self._coerce_int(item.get("chunk_index"))
            if not did or start is None or end is None or end <= start:
                continue
            # C.3: a mention's own field stays paired with its own
            # discourse_id; only fall back to the top-level field when the
            # mention has none (older records) -- never mix a field from
            # one occurrence with a discourse_id from another.
            m_field = str(item.get("field") or item.get("source_field") or "").strip()
            normalized.append({
                "discourse_id": did,
                "field": m_field or top_field,
                "chunk_index": int(chunk) if chunk is not None else -1,
                "start_char": int(start),
                "end_char": int(end),
            })
        deduped = dedup_mentions(normalized)
        if not deduped:
            return None

        def _sort_key(m: Dict[str, Any]) -> tuple:
            return (
                1 if int(m.get("chunk_index", -1)) < 0 else 0,
                str(m.get("field", "")),
                str(m.get("discourse_id", "")),
                int(m.get("chunk_index", -1)),
                int(m.get("start_char", -1)),
                int(m.get("end_char", -1)),
            )

        ordered = sorted(deduped, key=_sort_key)
        if verify and self._source_text_lookup is not None and surface_form:
            verifying = [
                m for m in ordered
                if self._mention_span_verifies(m, surface_form=surface_form, aliases=aliases) is True
            ]
            if verifying:
                return verifying[0]
            # Nothing verified: still return a deterministic pick, flagged
            # so the caller withholds the (known-suspect) numeric offsets
            # rather than truncating or emitting a false grounded span.
            chosen = dict(ordered[0])
            chosen["_span_unverified"] = True
            return chosen
        return ordered[0]

    def _ensure_discourse_unit_node(
        self,
        metadata: Optional[Dict[str, Any]],
        etd_ns: Namespace,
        origin_node: Optional[URIRef] = None,
    ) -> Optional[URIRef]:
        """
        Materialize a fully-provenanced DiscourseUnit node from origin metadata.
        """
        meta = dict(metadata or {})
        if origin_node is None:
            origin_node = self._resolve_origin_discourse_uri(meta)
        if origin_node is None:
            return None

        self.graph.add((origin_node, RDF.type, etd_ns.DiscourseUnit))

        did_raw = meta.get("discourse_id") or meta.get("doc_id")
        mention = self._primary_mention(meta)
        if did_raw is None and mention is not None:
            did_raw = mention.get("discourse_id")
        did = str(did_raw).strip() if did_raw is not None else ""
        if did:
            for old in list(self.graph.objects(origin_node, etd_ns.discourse_id)):
                self.graph.remove((origin_node, etd_ns.discourse_id, old))
            self._add_discourse_id_literal(origin_node, did, etd_ns)

        existing_field = next((str(o).strip() for o in self.graph.objects(origin_node, etd_ns.field) if str(o).strip()), "")
        field = str(meta.get("field") or existing_field).strip()
        if field:
            self._replace_registered_etd_literal(origin_node, "field", field, etd_ns, datatype=XSD.string)

        existing_source = next((str(o).strip() for o in self.graph.objects(origin_node, etd_ns.source) if str(o).strip()), "")
        source = str(meta.get("source") or meta.get("extractor") or meta.get("context_source") or existing_source or "unknown").strip()
        if source:
            self._replace_registered_etd_literal(origin_node, "source", source, etd_ns, datatype=XSD.string)
            context_source = str(meta.get("context_source") or source).strip()
            if context_source:
                self._replace_registered_etd_literal(origin_node, "context_source", context_source, etd_ns, datatype=XSD.string)

        source_doc_id = meta.get("source_doc_id")
        if source_doc_id is None and did:
            source_doc_id = did
        existing_source_doc_id = next(
            (str(o).strip() for o in self.graph.objects(origin_node, etd_ns.source_doc_id) if str(o).strip()),
            "",
        )
        if source_doc_id is None and existing_source_doc_id:
            source_doc_id = existing_source_doc_id
        if source_doc_id is not None and str(source_doc_id).strip():
            source_doc_id = str(source_doc_id).strip()
            self._replace_registered_etd_literal(origin_node, "source_doc_id", source_doc_id, etd_ns, datatype=XSD.string)

        candidate_chunk = self._coerce_int(meta.get("chunk_index"))
        existing_chunks = [self._coerce_int(o) for o in self.graph.objects(origin_node, etd_ns.chunk_index)]
        existing_chunks = [int(v) for v in existing_chunks if v is not None]
        chunk_index = candidate_chunk
        if chunk_index is None and mention is not None:
            chunk_index = self._coerce_int(mention.get("chunk_index"))
        if (chunk_index is None or chunk_index < 0) and existing_chunks:
            nonneg_existing = sorted(v for v in existing_chunks if v >= 0)
            chunk_index = nonneg_existing[0] if nonneg_existing else existing_chunks[0]
        if chunk_index is None:
            chunk_index = -1
        chunk_unknown = chunk_index < 0
        self._replace_registered_etd_literal(origin_node, "chunk_index", int(chunk_index), etd_ns, datatype=XSD.integer)
        self._replace_registered_etd_literal(origin_node, "chunk_index_unknown", bool(chunk_unknown), etd_ns, datatype=XSD.boolean)

        existing_totals = [self._coerce_int(o) for o in self.graph.objects(origin_node, etd_ns.chunk_total)]
        existing_totals = [int(v) for v in existing_totals if v is not None]
        chunk_total = self._coerce_int(meta.get("chunk_total"))
        if chunk_total is None and existing_totals:
            chunk_total = max(existing_totals)
        if chunk_total is None:
            chunk_total = 1
        self._replace_registered_etd_literal(origin_node, "chunk_total", int(chunk_total), etd_ns, datatype=XSD.integer)

        chunk_id = meta.get("chunk_id")
        existing_chunk_id = next((str(o).strip() for o in self.graph.objects(origin_node, etd_ns.chunk_id) if str(o).strip()), "")
        if (chunk_id is None or not str(chunk_id).strip()) and existing_chunk_id:
            chunk_id = existing_chunk_id
        if (chunk_id is None or not str(chunk_id).strip()) and source_doc_id is not None and chunk_index >= 0:
            chunk_id = f"{str(source_doc_id).strip()}:chunk:{int(chunk_index)}"
        if (chunk_id is None or not str(chunk_id).strip()) and source_doc_id is not None:
            chunk_id = f"{str(source_doc_id).strip()}:chunk:unknown"
        if chunk_id is not None and str(chunk_id).strip():
            self._replace_registered_etd_literal(origin_node, "chunk_id", str(chunk_id).strip(), etd_ns, datatype=XSD.string)

        label = f"DiscourseUnit: {field or did or _uri_localname(origin_node)}"
        self.graph.set((origin_node, RDFS.label, Literal(label, lang="en")))

        # Source_Context -- the larger sentence/semantic-unit/
        # field text needed to interpret a relation's full assertion,
        # distinct from Evidence.evidenceText (the minimal supporting
        # span). Shared across every relation whose Evidence links to this
        # same DiscourseUnit node (idempotent replace: relations from the
        # same field agree on the same text, so re-setting it is a no-op
        # in content, never a duplicate).
        context_text = str(meta.get("source_context") or meta.get("text") or "").strip()
        if context_text:
            # evidenceText must remain an exact
            # substring of contextText wherever exact grounding is
            # claimed -- truncating here would silently break that
            # guarantee for any relation whose evidence span happens to
            # fall past the cutoff in a long multi-item source field. Only
            # guard against a truly pathological/runaway value, well above
            # any real field length in practice.
            if len(context_text) > 50000:
                context_text = context_text[:50000]
            self._replace_registered_etd_literal(origin_node, "contextText", context_text, etd_ns, datatype=XSD.string)

        self._emit_registered_etd_literal(origin_node, "datasetId", self.dataset_id, etd_ns, datatype=XSD.string)
        self._emit_registered_etd_literal(origin_node, "runId", self.run_id, etd_ns, datatype=XSD.string)
        if (origin_node, etd_ns.extractedFrom, origin_node) not in self.graph:
            self.graph.add((origin_node, etd_ns.extractedFrom, origin_node))
        source_entities = self._resolve_source_entities_from_provenance(meta, etd_ns)
        source_entities = self._normalize_source_entities(source_entities, meta, etd_ns)
        self._replace_source_entity_links(origin_node, source_entities, etd_ns)
        return origin_node

    def _entity_nodes_for_type_and_discourse(self, type_local: str, discourse_id: str, etd_ns: Namespace) -> List[URIRef]:
        out: List[URIRef] = []
        did = str(discourse_id or "").strip()
        if not did:
            return out
        type_ref = etd_ns[type_local]
        for node in self.graph.subjects(RDF.type, type_ref):
            if not isinstance(node, URIRef):
                continue
            for o in self.graph.objects(node, etd_ns.discourse_id):
                if str(o).strip() == did:
                    out.append(node)
                    break
        return sorted(set(out), key=str)

    def _entity_nodes_for_type_and_id_literal(
        self,
        type_local: str,
        pred_local: str,
        raw_id: Any,
        etd_ns: Namespace,
    ) -> List[URIRef]:
        did = str(raw_id or "").strip()
        if not did:
            return []
        pred_ref = etd_ns[pred_local]
        type_ref = etd_ns[type_local]
        out: List[URIRef] = []
        for node in self.graph.subjects(RDF.type, type_ref):
            if not isinstance(node, URIRef):
                continue
            for o in self.graph.objects(node, pred_ref):
                if str(o).strip() == did:
                    out.append(node)
                    break
        return sorted(set(out), key=str)

    def _replace_source_entity_links(self, subject: URIRef, entities: List[URIRef], etd_ns: Namespace) -> None:
        for old in list(self.graph.objects(subject, etd_ns.hasSourceEntity)):
            self.graph.remove((subject, etd_ns.hasSourceEntity, old))
        for ent in sorted({e for e in entities if isinstance(e, URIRef)}, key=str):
            self.graph.add((subject, etd_ns.hasSourceEntity, ent))

    def _theme_nodes_for_source_theme_id(self, raw_id: Any, etd_ns: Namespace) -> List[URIRef]:
        """
        Resolve source Theme nodes by stable Theme-record identifiers only.

        Do not use etd:discourse_id here. Theme entities can legitimately
        accumulate discourse occurrences from downstream provenance, and using
        discourse_id for theme-description source resolution can collapse
        Theme 16/17 onto Theme 10.
        """
        theme_id = str(raw_id or "").strip()
        if not theme_id:
            return []

        candidates: List[URIRef] = []
        candidates.extend(self._entity_nodes_for_type_and_id_literal("Theme", "themeId", theme_id, etd_ns))
        candidates.extend(self._entity_nodes_for_type_and_id_literal("Theme", "source_doc_id", theme_id, etd_ns))
        return sorted({c for c in candidates if isinstance(c, URIRef)}, key=str)

    def _is_source_expert_record_uri(self, node: URIRef, source_id: Optional[str] = None) -> bool:
        if not isinstance(node, URIRef):
            return False
        uri_str = str(node)
        local = _uri_localname(node)
        expected = str(source_id or "").strip()
        patterns = []
        if expected:
            patterns.extend(
                [
                    f"/entity/person/expert-{expected}-",
                    f"/entity/expert/expert-{expected}-",
                    f"/entity/expert/expert_{expected}",
                    f"/entity/person/expert_{expected}",
                ]
            )
            if any(token in uri_str for token in patterns):
                return True
            if local == f"expert_{expected}" or local.startswith(f"expert-{expected}-"):
                return True
            return False

        return (
            "/entity/person/expert-" in uri_str
            or "/entity/expert/" in uri_str
            or local.startswith("expert_")
            or local.startswith("expert-")
        )

    def _expert_record_nodes_for_source_id(self, raw_id: Any, etd_ns: Namespace) -> List[URIRef]:
        """
        Resolve source dataset expert records only.

        Do not use generic Person/discourse matching here: extracted named people
        can share discourse provenance with expert.profile content, but they are
        not source dataset expert records.
        """
        source_id = str(raw_id or "").strip()
        if not source_id:
            return []

        candidates: Set[URIRef] = set()
        for node in self.graph.subjects(RDF.type, etd_ns.Expert):
            if not isinstance(node, URIRef):
                continue
            if not self._is_source_expert_record_uri(node, source_id):
                continue
            candidates.add(node)

        if candidates:
            return sorted(candidates, key=str)

        # Legacy fallback for tests or older exports that may only carry a local expert_<id> URI.
        for node in self.graph.subjects(RDF.type, etd_ns.Expert):
            if not isinstance(node, URIRef):
                continue
            if _uri_localname(node) == f"expert_{source_id}":
                candidates.add(node)
        return sorted(candidates, key=str)

    def _normalize_source_entities(
        self,
        entities: List[URIRef],
        meta: Dict[str, Any],
        etd_ns: Namespace,
    ) -> List[URIRef]:
        normalized = sorted({e for e in entities if isinstance(e, URIRef)}, key=str)
        if not isinstance(meta, dict):
            return normalized

        field = str(meta.get("field") or "").strip()
        if field == "expert.profile":
            expected_expert_id = str(meta.get("source_doc_id") or meta.get("discourse_id") or "").strip()
            if not expected_expert_id:
                return normalized
            authoritative_experts = self._expert_record_nodes_for_source_id(expected_expert_id, etd_ns)
            authoritative_expert = authoritative_experts[0] if authoritative_experts else None
            if authoritative_expert is None:
                return normalized
            filtered: List[URIRef] = [authoritative_expert]
            for ent in normalized:
                if ent == authoritative_expert:
                    continue
                if (ent, RDF.type, etd_ns.Expert) in self.graph:
                    continue
                filtered.append(ent)
            return sorted({e for e in filtered if isinstance(e, URIRef)}, key=str)

        if field != "theme.description":
            return normalized

        expected_theme_id = str(meta.get("source_doc_id") or meta.get("discourse_id") or "").strip()
        authoritative_theme: Optional[URIRef] = None

        source_entity_uri = str(meta.get("source_entity_uri") or "").strip()
        if source_entity_uri:
            source_ref = _safe_uriref(source_entity_uri, self.namespaces)
            if isinstance(source_ref, URIRef):
                source_ref = self._canonicalize_relation_endpoint_uri(source_ref)
                if isinstance(source_ref, URIRef) and (source_ref, RDF.type, etd_ns.Theme) in self.graph:
                    authoritative_theme = source_ref

        def _theme_ids(node: URIRef) -> Set[str]:
            vals: Set[str] = set()
            for pred in (etd_ns.themeId, etd_ns.source_doc_id):
                for obj in self.graph.objects(node, pred):
                    sval = str(obj).strip()
                    if sval:
                        vals.add(sval)
            return vals

        if authoritative_theme is None and expected_theme_id:
            for ent in normalized:
                if (ent, RDF.type, etd_ns.Theme) not in self.graph:
                    continue
                if expected_theme_id in _theme_ids(ent):
                    authoritative_theme = ent
                    break

        if authoritative_theme is None and expected_theme_id:
            for cand in self._theme_nodes_for_source_theme_id(expected_theme_id, etd_ns):
                if isinstance(cand, URIRef) and (cand, RDF.type, etd_ns.Theme) in self.graph:
                    authoritative_theme = cand
                    break

        if authoritative_theme is None:
            return normalized

        filtered: List[URIRef] = [authoritative_theme]
        for ent in normalized:
            if ent == authoritative_theme:
                continue
            if (ent, RDF.type, etd_ns.Theme) in self.graph:
                continue
            filtered.append(ent)
        return sorted({e for e in filtered if isinstance(e, URIRef)}, key=str)

    def _expand_source_entity_closure(self, anchors: List[URIRef], etd_ns: Namespace) -> List[URIRef]:
        out: Set[URIRef] = set()
        queue: List[URIRef] = [a for a in anchors if isinstance(a, URIRef)]
        seen: Set[URIRef] = set()

        def _expert_from_literal(subject: URIRef) -> List[URIRef]:
            expert_ids = [str(o).strip() for o in self.graph.objects(subject, etd_ns.expertId) if str(o).strip()]
            found: Set[URIRef] = set()
            for eid in expert_ids:
                found.update(self._entity_nodes_for_type_and_discourse("Expert", eid, etd_ns))
                found.update(self._entity_nodes_for_type_and_discourse("Person", eid, etd_ns))
                found.update(self._entity_nodes_for_type_and_id_literal("Expert", "expertId", eid, etd_ns))
                found.update(self._entity_nodes_for_type_and_id_literal("Expert", "userId", eid, etd_ns))
            return sorted(found, key=str)

        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            out.add(node)

            if (node, RDF.type, etd_ns.Answer) in self.graph:
                for q in self.graph.objects(node, etd_ns.answersQuestion):
                    if isinstance(q, URIRef):
                        queue.append(q)
                for ex in self.graph.objects(node, etd_ns.authoredBy):
                    if isinstance(ex, URIRef):
                        queue.append(ex)

            if (node, RDF.type, etd_ns.Question) in self.graph:
                for th in self.graph.objects(node, etd_ns.belongsToTheme):
                    if isinstance(th, URIRef):
                        queue.append(th)
                for ex in self.graph.objects(node, etd_ns.askedBy):
                    if isinstance(ex, URIRef):
                        queue.append(ex)
                queue.extend(_expert_from_literal(node))

            if (node, RDF.type, etd_ns.Theme) in self.graph:
                for ex in self.graph.objects(node, etd_ns.createdBy):
                    if isinstance(ex, URIRef):
                        queue.append(ex)
                queue.extend(_expert_from_literal(node))

            if (node, RDF.type, etd_ns.Vote) in self.graph:
                for ans in self.graph.subjects(etd_ns.receivesVote, node):
                    if isinstance(ans, URIRef):
                        queue.append(ans)
                for ans in self.graph.subjects(etd_ns.hasVote, node):
                    if isinstance(ans, URIRef):
                        queue.append(ans)
                for ex in self.graph.objects(node, etd_ns.castBy):
                    if isinstance(ex, URIRef):
                        queue.append(ex)

        return sorted(out, key=str)

    def _resolve_source_entities_from_provenance(self, meta: Dict[str, Any], etd_ns: Namespace) -> List[URIRef]:
        if not isinstance(meta, dict):
            return []
        source_entity_uri = str(meta.get("source_entity_uri") or "").strip()
        if source_entity_uri:
            source_ref = _safe_uriref(source_entity_uri, self.namespaces)
            if isinstance(source_ref, URIRef):
                canonical_ref = self._canonicalize_relation_endpoint_uri(source_ref)
                if isinstance(canonical_ref, URIRef):
                    expanded = self._expand_source_entity_closure([canonical_ref], etd_ns)
                    return self._normalize_source_entities(expanded, meta, etd_ns)
        field = str(meta.get("field") or "").strip()
        source_doc_id = str(meta.get("source_doc_id") or meta.get("discourse_id") or "").strip()
        if not field or not source_doc_id:
            return []

        field_root = field.split(".", 1)[0].strip().lower()
        anchors: List[URIRef] = []
        if field_root == "answer":
            anchors.extend(self._entity_nodes_for_type_and_discourse("Answer", source_doc_id, etd_ns))
            anchors.extend(self._entity_nodes_for_type_and_id_literal("Answer", "answerId", source_doc_id, etd_ns))
        elif field_root == "question":
            anchors.extend(self._entity_nodes_for_type_and_discourse("Question", source_doc_id, etd_ns))
            anchors.extend(self._entity_nodes_for_type_and_id_literal("Question", "questionId", source_doc_id, etd_ns))
        elif field_root == "theme":
            anchors.extend(self._theme_nodes_for_source_theme_id(source_doc_id, etd_ns))
        elif field_root == "vote":
            anchors.extend(self._entity_nodes_for_type_and_discourse("Vote", source_doc_id, etd_ns))
            anchors.extend(self._entity_nodes_for_type_and_id_literal("Vote", "voteId", source_doc_id, etd_ns))
        elif field_root == "expert":
            anchors.extend(self._expert_record_nodes_for_source_id(source_doc_id, etd_ns))
        expanded = self._expand_source_entity_closure(anchors, etd_ns)
        return self._normalize_source_entities(expanded, meta, etd_ns)

    def _infer_structural_entity_type_from_uri(
        self,
        node: URIRef,
        rel: Optional[Relation] = None,
        role: Optional[str] = None,
    ) -> Optional[str]:
        s = str(node)
        for marker, etype in (
            ("/entity/answer/", "Answer"),
            ("/entity/question/", "Question"),
            ("/entity/theme/", "Theme"),
            ("/entity/vote/", "Vote"),
            ("/entity/expert/", "Expert"),
            ("/entity/person/", "Person"),
            ("/entity/discourseunit/", "DiscourseUnit"),
        ):
            if marker in s:
                return etype
        local = _uri_localname(node).lower()
        if local.startswith("answer_"):
            return "Answer"
        if local.startswith("question_"):
            return "Question"
        if local.startswith("theme_"):
            return "Theme"
        if local.startswith("vote_"):
            return "Vote"
        if local.startswith("expert_"):
            return "Expert"
        if local.startswith("discourse_"):
            return "DiscourseUnit"
        pred = str(getattr(rel, "relation", "") or "")
        if pred == "answersQuestion" and role == "subject":
            return "Answer"
        if pred == "answersQuestion" and role == "object":
            return "Question"
        if pred == "authoredBy" and role == "subject":
            return "Answer"
        if pred == "authoredBy" and role == "object":
            return "Expert"
        if pred == "askedBy" and role == "subject":
            return "Question"
        if pred == "askedBy" and role == "object":
            return "Expert"
        if pred == "createdBy" and role == "subject":
            return "Theme"
        if pred == "createdBy" and role == "object":
            return "Expert"
        if pred == "belongsToTheme" and role == "subject":
            return "Question"
        if pred == "belongsToTheme" and role == "object":
            return "Theme"
        if pred == "castBy" and role == "subject":
            return "Vote"
        if pred == "castBy" and role == "object":
            return "Expert"
        if pred == "receivesVote" and role == "subject":
            return "Answer"
        if pred == "receivesVote" and role == "object":
            return "Vote"
        return None

    def _hydrate_relation_endpoint_provenance(
        self,
        node: Any,
        rel: Relation,
        meta: Dict[str, Any],
        role: str,
        etd_ns: Namespace,
    ) -> None:
        if not isinstance(node, URIRef):
            return
        etype = self._infer_structural_entity_type_from_uri(node, rel=rel, role=role)
        if etype is None:
            return
        self.graph.add((node, RDF.type, etd_ns.Entity))
        if hasattr(etd_ns, etype):
            self.graph.add((node, RDF.type, etd_ns[etype]))
        if not any(True for _ in self.graph.objects(node, RDFS.label)):
            local = _uri_localname(node).replace("-", " ").replace("_", " ")
            self.graph.add((node, RDFS.label, Literal(local or etype, lang="en")))
        endpoint_meta = dict(meta or {})
        endpoint_meta.setdefault("surface_form", _first_literal_for_predicate(self.graph, node, RDFS.label) or etype)
        endpoint_meta.setdefault("context_source", endpoint_meta.get("context_source") or endpoint_meta.get("source") or "structural")
        endpoint_meta.setdefault("source", endpoint_meta.get("source") or endpoint_meta.get("extractor") or "structural")
        if endpoint_meta.get("field") is None:
            if etype in {"Answer", "Question", "Theme", "Vote"}:
                endpoint_meta["field"] = etype.lower()
        if endpoint_meta.get("discourse_id") is None and endpoint_meta.get("source_doc_id") is not None:
            endpoint_meta["discourse_id"] = endpoint_meta.get("source_doc_id")
        if endpoint_meta.get("source_doc_id") is None and endpoint_meta.get("discourse_id") is not None:
            endpoint_meta["source_doc_id"] = endpoint_meta.get("discourse_id")
        if endpoint_meta.get("chunk_index") is None:
            endpoint_meta["chunk_index"] = -1
        if etype == "DiscourseUnit":
            endpoint_meta.setdefault("chunk_total", 1)
            source_doc_id = endpoint_meta.get("source_doc_id") or endpoint_meta.get("discourse_id")
            chunk_index = self._coerce_int(endpoint_meta.get("chunk_index"))
            if source_doc_id is not None and chunk_index is not None:
                chunk_token = str(chunk_index) if chunk_index >= 0 else "unknown"
                endpoint_meta.setdefault("chunk_id", f"{str(source_doc_id).strip()}:chunk:{chunk_token}")
        self._add_entity_metadata(node, endpoint_meta, etd_ns)

    _ISO_DT_RE = re.compile(
        r"^\d{4}-\d{2}-\d{2}"                  # date
        r"(?:[T ]\d{2}:\d{2}:\d{2}"            # time
        r"(?:\.\d+)?"                          # optional fractional
        r"(?:Z|[+-]\d{2}:\d{2})?)?$"           # optional TZ
    )

    def _is_iso_datetime(self, s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        return bool(self._ISO_DT_RE.match(s))

    def _add_entity_metadata(
        self,
        ent_uri: URIRef,
        metadata: Dict[str, Any],
        etd_ns: Namespace,
    ) -> None:
        """
        Add core, SHACL-friendly metadata for entities.

        At minimum we support:
          - surface_form      (xsd:string)
          - context_source    (xsd:string)
          - discourse_id      (xsd:string)
          - chunk_index       (xsd:integer)
          - chunk_total       (xsd:integer)
          - start_char        (xsd:integer)
          - end_char          (xsd:integer)
          - ontology_uri      (URIRef if resolvable, else xsd:string)
        """
        if not metadata:
            return

        # C.2: the preserved forms the emitted span is allowed to match --
        # surface_form, EntityNormalizer's recorded pre-normalization
        # aliases, and the entity's own rdfs:label (which can legitimately
        # be the longer captured mention, e.g. label "OpenAI API" while
        # surface_form was normalised down to "OpenAI").
        _entity_surface_form = str(metadata.get("surface_form") or metadata.get("canonical_label") or "").strip() or None
        _raw_aliases = metadata.get("aliases")
        _verify_aliases = [a for a in _raw_aliases if isinstance(a, str) and a.strip()] if isinstance(_raw_aliases, list) else []
        _lbl = _first_literal_for_predicate(self.graph, ent_uri, RDFS.label)
        if _lbl and _lbl not in _verify_aliases:
            _verify_aliases.append(_lbl)

        # C.2 span verification applies only to genuine mention-grounded
        # entities. A CONSTRUCTED endpoint (e.g. a Proposition standing for
        # a clause) is grounded by its clause offsets, not by a verbatim
        # surface form, so its span must not be verified against / withheld.
        _verify_span = str(metadata.get("mention_grounding") or "").strip().lower() != "constructed"

        primary_mention = self._primary_mention(
            metadata, surface_form=_entity_surface_form, aliases=_verify_aliases,
            verify=_verify_span,
        )
        span_unverified = bool(primary_mention.get("_span_unverified")) if primary_mention is not None else False

        # A primary_mention represents a DISTINCT source occurrence (which
        # its whole source-identity tuple, including source_doc_id and the
        # primary extractedFrom DiscourseUnit, must follow) only when it
        # carries its OWN field. A mention with no field of its own is an
        # accumulated downstream discourse-id reference, not a separate
        # source record -- for those the entity's scalar source identity
        # is authoritative and stays as-is.
        primary_mention_is_own_occurrence = False
        if primary_mention is not None:
            _raw_ms = metadata.get("mentions")
            if isinstance(_raw_ms, list):
                _pd = str(primary_mention.get("discourse_id") or "").strip()
                _ps = self._coerce_int(primary_mention.get("start_char"))
                _pe = self._coerce_int(primary_mention.get("end_char"))
                for _m in _raw_ms:
                    if not isinstance(_m, dict):
                        continue
                    if str(_m.get("discourse_id") or "").strip() != _pd:
                        continue
                    if (self._coerce_int(_m.get("start_char")) != _ps
                            or self._coerce_int(_m.get("end_char")) != _pe):
                        continue
                    if str(_m.get("field") or _m.get("source_field") or "").strip():
                        primary_mention_is_own_occurrence = True
                        break

        # Backward compatibility: accept doc_id if discourse_id is missing
        if "discourse_id" not in metadata and "doc_id" in metadata:
            metadata["discourse_id"] = metadata.get("doc_id")
        if "discourse_id" not in metadata and primary_mention is not None:
            metadata["discourse_id"] = primary_mention.get("discourse_id")

        # String fields that are safe to append as-is.
        for key in ("text",):
            if key in metadata and metadata[key] is not None:
                if isinstance(metadata[key], (dict, list)):
                    continue  # or flatten intentionally
                val = str(metadata[key]).strip()
                if val:
                    self._emit_registered_etd_literal(ent_uri, key, val, etd_ns, datatype=XSD.string)

        # C.3: field is one half of the atomic (field, discourse_id) source
        # pair -- take it from the SAME primary_mention record the
        # discourse_id / span below come from, so the emitted pair always
        # identifies ONE real record. Fall back to the top-level scalar
        # only when the chosen mention carries no field of its own.
        field_val = ""
        if primary_mention is not None:
            field_val = str(primary_mention.get("field") or "").strip()
        if not field_val:
            field_val = str(metadata.get("field") or metadata.get("source_field") or "").strip()
        if field_val:
            self._replace_registered_etd_literal(ent_uri, "field", field_val, etd_ns, datatype=XSD.string)

        fallback_surface = str(metadata.get("surface_form") or metadata.get("canonical_label") or "").strip()
        if not fallback_surface:
            fallback_surface = _first_literal_for_predicate(self.graph, ent_uri, RDFS.label) or ""
        if fallback_surface:
            self._replace_registered_etd_literal(ent_uri, "surface_form", fallback_surface, etd_ns, datatype=XSD.string)

        fallback_context = str(metadata.get("context_source") or metadata.get("source") or "unknown").strip()
        if fallback_context:
            self._replace_registered_etd_literal(ent_uri, "context_source", fallback_context, etd_ns, datatype=XSD.string)

        fallback_source = str(metadata.get("source") or metadata.get("context_source") or "unknown").strip()
        if fallback_source:
            self._replace_registered_etd_literal(ent_uri, "source", fallback_source, etd_ns, datatype=XSD.string)

        # discourse_id -- ALWAYS paired with the SAME mention record as
        # start_char/end_char below, so a consumer reading this entity's
        # single (discourse_id, start_char, end_char) triple set gets a
        # genuinely verifiable span into ONE real source text, never a
        # discourse_id from one mention paired with an offset copied from
        # a different one.
        #
        # Root cause (confirmed 2026-09 linker-validation preflight):
        # GlobalEntityRegistry.get_or_create (utils/helpers.py) correctly
        # accumulates every occurrence of a cross-discourse-consolidated
        # entity into metadata["mentions"] (via _merge_mentions), but the
        # TOP-LEVEL scalar metadata["discourse_id"]/["start_char"]/
        # ["end_char"] are separately overwritten by whichever mention
        # happens to be merged LAST (plain dict-key overwrite, since these
        # are not list/dict-valued). This code previously emitted
        # discourse_id from BOTH primary_mention (deterministic: the
        # earliest/lowest-numbered mention) AND the independently-drifting
        # metadata scalar, while start_char/end_char came ONLY from the
        # scalar -- so for any entity merged across 2+ discourse mentions,
        # the emitted start_char/end_char could silently describe a
        # DIFFERENT discourse_id than the one a reader would naturally
        # pair it with. Confirmed empirically against 6 real entities
        # (Cultural Bias, communities, model, users, Algorithms, and one
        # of two OpenAI mentions) where the recorded span matched the
        # SECOND emitted discourse_id, not the first.
        #
        # Fix: emit exactly ONE discourse_id, and derive it (and
        # start_char/end_char below) from the SAME primary_mention record
        # when one exists. No provenance is lost -- every individual
        # mention (including the ones no longer chosen as "primary") still
        # gets its own reified discourse-unit node via the `mentions` loop
        # further below in this method.
        primary_discourse_id = (
            primary_mention.get("discourse_id") if primary_mention is not None else metadata.get("discourse_id")
        )
        self._add_discourse_id_literal(ent_uri, primary_discourse_id, etd_ns)

        # Integer-like fields (production-safe: coerce + preserve raw).
        # start_char/end_char come from the SAME primary_mention record as
        # field/discourse_id above. C.2: when a source-text resolver is
        # available and NONE of this entity's mentions verify (the excerpt
        # matches neither surface_form nor a preserved alias/label, or the
        # pair does not resolve at all), the numeric offsets are WITHHELD
        # -- never truncated to fit, never emitted as a false grounded
        # span. The entity keeps its (field, discourse_id, surface_form)
        # identity; a consumer simply sees "no verifiable offset".
        emit_span = not span_unverified
        for key in ("chunk_total", "start_char", "end_char"):
            if key in ("start_char", "end_char") and not emit_span:
                continue
            raw_val = None
            if key in ("start_char", "end_char") and primary_mention is not None:
                raw_val = primary_mention.get(key)
            if raw_val is None and key in metadata:
                raw_val = metadata.get(key)
            if raw_val is not None:
                num = self._coerce_int(raw_val)

                if num is not None:
                    self._emit_registered_etd_literal(ent_uri, key, num, etd_ns, datatype=XSD.integer)

        if span_unverified:
            record_event(
                self.quality_events,
                category="entity_span_withheld_unverifiable",
                action="suppressed",
                discourse_id=primary_discourse_id,
                subject_surface=_entity_surface_form or None,
                reason=(
                    "no mention's (field, discourse_id) span matched the "
                    "surface form or a preserved alias/label; numeric "
                    "offsets withheld rather than truncated"
                ),
            )

        # Same consistency rule as discourse_id/start_char/end_char above:
        # prefer primary_mention's own chunk_index over the top-level
        # scalar (which is subject to the same last-merge-wins drift), so
        # chunk_index always describes the SAME occurrence as the emitted
        # span.
        chunk_index = None
        if primary_mention is not None:
            chunk_index = self._coerce_int(primary_mention.get("chunk_index"))
        if chunk_index is None:
            chunk_index = self._coerce_int(metadata.get("chunk_index"))
        if chunk_index is None:
            chunk_index = -1
        self._replace_registered_etd_literal(ent_uri, "chunk_index", int(chunk_index), etd_ns, datatype=XSD.integer)
        self._replace_registered_etd_literal(ent_uri, "chunk_index_unknown", bool(int(chunk_index) < 0), etd_ns, datatype=XSD.boolean)

        # C.3 serialization residue fix: when the primary_mention is a
        # distinct source occurrence, EVERY single-valued primary source
        # scalar the entity emits must describe THAT occurrence -- not a
        # different one left on the top-level scalar tuple by
        # first-writer-wins. A mention carries no source_doc_id of its own;
        # its own discourse_id IS its source doc id (the same equivalence
        # the scalar fallback already uses).
        if primary_mention_is_own_occurrence:
            source_doc_id = str(primary_mention.get("discourse_id") or "").strip() or None
        else:
            source_doc_id = metadata.get("source_doc_id") or metadata.get("discourse_id") or (primary_mention or {}).get("discourse_id")
        if source_doc_id is not None and str(source_doc_id).strip():
            self._replace_registered_etd_literal(ent_uri, "source_doc_id", str(source_doc_id).strip(), etd_ns, datatype=XSD.string)

        is_discourse_unit = (ent_uri, RDF.type, etd_ns.DiscourseUnit) in self.graph
        if is_discourse_unit:
            chunk_total = self._coerce_int(metadata.get("chunk_total"))
            if chunk_total is None:
                chunk_total = 1
            self._replace_registered_etd_literal(ent_uri, "chunk_total", int(chunk_total), etd_ns, datatype=XSD.integer)

            chunk_id = metadata.get("chunk_id")
            if (chunk_id is None or not str(chunk_id).strip()) and source_doc_id is not None:
                chunk_token = str(int(chunk_index)) if int(chunk_index) >= 0 else "unknown"
                chunk_id = f"{str(source_doc_id).strip()}:chunk:{chunk_token}"
            if chunk_id is not None and str(chunk_id).strip():
                self._replace_registered_etd_literal(ent_uri, "chunk_id", str(chunk_id).strip(), etd_ns, datatype=XSD.string)

        def _occurrence_origin_meta(
            occ_field: str, occ_did: str, occ_chunk: Any,
            occ_start: Any, occ_end: Any,
        ) -> Dict[str, Any]:
            """Origin metadata for ONE source occurrence, keyed strictly by
            that occurrence's own (field, discourse_id): its discourse_id
            doubles as its source_doc_id, its origin_key is derived from
            exactly that pair, and any conflicting top-level
            source_doc_id / origin_key / chunk_id from a DIFFERENT
            occurrence is dropped so the resulting DiscourseUnit cannot
            inherit provenance from the entity's scalar tuple. All other
            (non-source-identity) metadata is preserved."""
            om = dict(metadata)
            om.pop("mentions", None)
            om.pop("chunk_id", None)
            f = str(occ_field or "").strip()
            d = str(occ_did or "").strip()
            if f:
                om["field"] = f
            if d:
                om["discourse_id"] = d
                om["source_doc_id"] = d
            om["origin_key"] = (f"{f}|{d}" if f and d else d)
            ci = self._coerce_int(occ_chunk)
            if ci is not None:
                om["chunk_index"] = ci
            if occ_start is not None:
                om["start_char"] = occ_start
            if occ_end is not None:
                om["end_char"] = occ_end
            return om

        origin_nodes: List[URIRef] = []
        if primary_mention_is_own_occurrence:
            # Primary extractedFrom DiscourseUnit == the SAME occurrence the
            # emitted field / discourse_id / source_doc_id / span /
            # chunk_index all describe.
            origin_node = self._ensure_discourse_unit_node(
                _occurrence_origin_meta(
                    primary_mention.get("field"),
                    primary_mention.get("discourse_id"),
                    primary_mention.get("chunk_index"),
                    primary_mention.get("start_char"),
                    primary_mention.get("end_char"),
                ),
                etd_ns,
            )
        else:
            origin_node = self._ensure_discourse_unit_node(metadata, etd_ns)
        if origin_node is not None:
            origin_nodes.append(origin_node)
        mentions = metadata.get("mentions")
        if isinstance(mentions, list):
            for item in mentions:
                if not isinstance(item, dict):
                    continue
                did = item.get("discourse_id")
                if did is None or not str(did).strip():
                    continue
                # Each mention's DiscourseUnit is keyed by that mention's
                # OWN field / discourse_id / origin_key / span / chunk --
                # never the entity's top-level scalar tuple. A mention with
                # no field of its own falls back to the top-level field
                # (same rule _primary_mention uses), NOT to the top-level
                # origin_key/source_doc_id. Deduped against the primary
                # node below via `desired_origins`.
                mention_meta = _occurrence_origin_meta(
                    (item.get("field") or item.get("source_field")
                     or metadata.get("field") or metadata.get("source_field")),
                    did,
                    item.get("chunk_index"),
                    item.get("start_char"),
                    item.get("end_char"),
                )
                mn = self._ensure_discourse_unit_node(mention_meta, etd_ns)
                if mn is not None:
                    origin_nodes.append(mn)

        # Span-less STRUCTURAL source occurrences (e.g. a dataset Expert /
        # document record). Preserved verbatim across canonical-entity
        # merges by GlobalEntityRegistry._merge_source_occurrences, so this
        # occurrence's own DiscourseUnit is materialised from ITS provenance
        # tuple -- never from the entity's primary scalars (which
        # first-writer-wins may have locked to a different field) and never
        # from the canonical URI. The 1:1 source link is asserted directly:
        # the entity that declared the occurrence IS its source.
        occurrences = metadata.get("source_occurrences")
        if isinstance(occurrences, list):
            for item in occurrences:
                if not isinstance(item, dict):
                    continue
                occ_field = str(item.get("field") or item.get("source_field") or "").strip()
                occ_did = str(item.get("discourse_id") or item.get("source_doc_id") or "").strip()
                occ_okey = str(item.get("origin_key") or "").strip()
                if not occ_field or not (occ_did or occ_okey):
                    continue
                occ_meta = {
                    "field": occ_field,
                    "discourse_id": occ_did or occ_okey.split("|")[-1],
                    "source_doc_id": str(item.get("source_doc_id") or occ_did or "").strip(),
                    "origin_key": occ_okey or f"{occ_field}|{occ_did}",
                    "source": str(item.get("source") or "structural").strip(),
                    "context_source": str(item.get("context_source") or item.get("source") or "").strip(),
                    "chunk_index": item.get("chunk_index", -1),
                    "chunk_index_unknown": item.get("chunk_index_unknown", True),
                    "chunk_total": item.get("chunk_total", 1),
                }
                on = self._ensure_discourse_unit_node(occ_meta, etd_ns)
                if on is None:
                    continue
                origin_nodes.append(on)
                self.graph.add((on, etd_ns.hasSourceEntity, ent_uri))

        desired_origins = sorted({n for n in origin_nodes if isinstance(n, URIRef)}, key=str)
        if desired_origins:
            for old in list(self.graph.objects(ent_uri, etd_ns.extractedFrom)):
                self.graph.remove((ent_uri, etd_ns.extractedFrom, old))
            for onode in desired_origins:
                self.graph.add((ent_uri, etd_ns.extractedFrom, onode))

        # Genuine entity-level provenance only -- e.g. a Proposition entity
        # that derives its content from one specific supporting
        # relation, when exactly that relation was available. Generic: any
        # entity carrying this metadata key gets the same treatment, not
        # special-cased to one entity_type.
        derived_from_relation_uris = metadata.get("derived_from_relation_uris")
        if isinstance(derived_from_relation_uris, list):
            for item in derived_from_relation_uris:
                raw = str(item or "").strip()
                if not raw:
                    continue
                ref = _safe_uriref(raw, self.namespaces) if raw.startswith("http") else _safe_uriref(
                    f"{self.namespaces['etd']}{raw.lstrip('#/')}", self.namespaces
                )
                self.graph.add((ent_uri, PROV.wasDerivedFrom, ref))

        self._emit_registered_etd_literal(ent_uri, "datasetId", self.dataset_id, etd_ns, datatype=XSD.string)
        self._emit_registered_etd_literal(ent_uri, "runId", self.run_id, etd_ns, datatype=XSD.string)


        # ontology_uri as a proper resource if possible
        if "ontology_uri" in metadata and metadata["ontology_uri"] is not None:
            ou = metadata["ontology_uri"]

            # ----------------------------------------------------
            # FIX: prevent ontology_uri being list or stringified list
            # e.g. ["dcat:Dataset"] or "['dcat:Dataset']"
            # ----------------------------------------------------
            if isinstance(ou, list) and ou:
                ou = ou[0]

            if isinstance(ou, str):
                s = ou.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, list) and parsed:
                            ou = parsed[0]
                    except Exception:
                        pass

            raw = str(ou).strip()
            if raw:
                ref = self._resolve_entity_uri_from_value(raw)
                if ref is None:
                    ref = Literal(raw, datatype=XSD.string)
                self.graph.add((ent_uri, etd_ns.ontology_uri, ref))

    #-----------------------------------------
    # Adding wikidata_metadata
    # ----------------------------------------
    def _add_wikidata_metadata(
        self,
        ent_uri: URIRef,
        metadata: Dict[str, Any],
        etd_ns: Namespace,
    ) -> None:
        """
        Handle Wikidata-related metadata so we don't store raw JSON strings.

        Expected keys (by convention in your pipeline):
        - 'wikidata'             → dict or string with 'qid' and 'label'
        - 'wikidataLabel'        → label string
        - 'wikidataDescription'  → description string
        - 'wikidataScore'        → float
        """
        if not metadata:
            return

        def _replace_value(p: URIRef) -> None:
            for o in list(self.graph.objects(ent_uri, p)):
                self.graph.remove((ent_uri, p, o))

        wrote_label = False
        wrote_desc = False
        wrote_score = False

        def _truthy(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in {"true", "1", "yes"}

        wikidata_val = metadata.get("wikidata")
        if wikidata_val:
            if isinstance(wikidata_val, str):
                try:
                    wikidata_val = ast.literal_eval(wikidata_val)
                except Exception:
                    wikidata_val = None

            if isinstance(wikidata_val, dict):
                qid = wikidata_val.get("qid")
                if qid:
                    wd_uri = f"http://www.wikidata.org/entity/{qid}"
                    wd_ref = _safe_uriref(wd_uri, self.namespaces)

                    # Identity semantics: reserve owl:sameAs for an
                    # AUTHORITATIVE asserted identity; a heuristic linker
                    # best-guess is a non-transitive semantic alignment
                    # (skos:closeMatch), never hard identity. The QID,
                    # score, evidence, authority flag and decision reason
                    # are always preserved on the entity regardless.
                    authoritative = _truthy(
                        wikidata_val.get("authoritative")
                        if wikidata_val.get("authoritative") is not None
                        else metadata.get("wikidata_authoritative")
                    )
                    self.graph.add((ent_uri, etd_ns.wikidata, wd_ref))
                    if authoritative:
                        self.graph.add((ent_uri, OWL.sameAs, wd_ref))
                    else:
                        # never assert hard identity for a heuristic match;
                        # clear any stale owl:sameAs to this QID too.
                        self.graph.remove((ent_uri, OWL.sameAs, wd_ref))
                        self.graph.add((ent_uri, SKOS.closeMatch, wd_ref))

                    _replace_value(etd_ns.wikidataAuthoritative)
                    self.graph.add((
                        ent_uri, etd_ns.wikidataAuthoritative,
                        Literal(bool(authoritative), datatype=XSD.boolean),
                    ))
                    ident_ev = str(
                        wikidata_val.get("identity_evidence")
                        or metadata.get("wikidata_identity_evidence") or ""
                    ).strip()
                    if ident_ev:
                        _replace_value(etd_ns.wikidataIdentityEvidence)
                        self.graph.add((
                            ent_uri, etd_ns.wikidataIdentityEvidence,
                            Literal(ident_ev, datatype=XSD.string),
                        ))
                    reason = str(
                        wikidata_val.get("decision_reason")
                        or metadata.get("wikidata_decision_reason") or ""
                    ).strip()
                    if reason:
                        _replace_value(etd_ns.wikidataDecisionReason)
                        self.graph.add((
                            ent_uri, etd_ns.wikidataDecisionReason,
                            Literal(reason, datatype=XSD.string),
                        ))

                label = wikidata_val.get("label")
                if label:
                    _replace_value(etd_ns.wikidataLabel)
                    self.graph.add(
                        (ent_uri, etd_ns.wikidataLabel, Literal(str(label).strip(), lang="en"))
                    )
                    wrote_label = True

                desc = wikidata_val.get("description")
                if desc:
                    _replace_value(etd_ns.wikidataDescription)
                    self.graph.add(
                        (ent_uri, etd_ns.wikidataDescription, Literal(str(desc).strip(), lang="en"))
                    )
                    wrote_desc = True

                score = wikidata_val.get("score")
                if score is not None:
                    try:
                        sc = float(score)
                    except Exception:
                        sc = 0.0
                    _replace_value(etd_ns.wikidataScore)
                    self.graph.add((ent_uri, etd_ns.wikidataScore, Literal(sc, datatype=XSD.float)))
                    wrote_score = True

        # Direct convenience keys (only if not already written from wikidata dict)
        lbl = metadata.get("wikidataLabel")
        if lbl and not wrote_label:
            _replace_value(etd_ns.wikidataLabel)
            self.graph.add((ent_uri, etd_ns.wikidataLabel, Literal(str(lbl).strip(), lang="en")))

        desc2 = metadata.get("wikidataDescription")
        if desc2 and not wrote_desc:
            _replace_value(etd_ns.wikidataDescription)
            self.graph.add((ent_uri, etd_ns.wikidataDescription, Literal(str(desc2).strip(), lang="en")))

        score2 = metadata.get("wikidataScore")
        if score2 is not None and not wrote_score:
            try:
                sc2 = float(score2)
            except Exception:
                sc2 = 0.0
            _replace_value(etd_ns.wikidataScore)
            self.graph.add((ent_uri, etd_ns.wikidataScore, Literal(sc2, datatype=XSD.float)))

    # ------------------------------------------------------------------
    # Relations (direct + relation node with metadata)
    # ------------------------------------------------------------------
    def _get_all_relations(self) -> Iterable[Relation]:
        """
        Flatten relations from all document groups in annotations
        into a list[Relation].
        """
        rels: List[Relation] = []

        base_etd = self.namespaces.get("etd")
        if not base_etd:
            raise ValueError("Missing 'etd' namespace in RDFSerializer.namespaces")

        def _compose_origin_key(discourse_id: Any, field: Any) -> str:
            did = str(discourse_id).strip() if discourse_id is not None else ""
            fld = str(field).strip() if field is not None else ""
            if did and fld:
                return f"{fld}|{did}"
            return did

        def collect(section_key: str) -> None:
            section = self.annotations.get(section_key, {})
            for _, doc in section.items():
                # Skip non-dict entries
                if not isinstance(doc, dict):
                    continue
                doc_meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}

                for rel_data in doc.get("relations", []):
                    # Skip non-dicts just in case
                    if not isinstance(rel_data, dict):
                        continue

                    src = rel_data.get("source_id") or rel_data.get("source")
                    tgt = rel_data.get("target_id") or rel_data.get("target")
                    pred = rel_data.get("relation")
                    rel_meta = dict(rel_data.get("metadata", {}) or {})
                    did = rel_meta.get("discourse_id")
                    if did is None:
                        did = doc_meta.get("discourse_id")
                    if did is not None:
                        rel_meta.setdefault("discourse_id", did)
                    field = rel_meta.get("field")
                    if field is None:
                        field = doc_meta.get("field")
                    if field is not None and str(field).strip():
                        rel_meta.setdefault("field", str(field).strip())
                    origin_key = rel_meta.get("origin_key")
                    if origin_key is None:
                        origin_key = _compose_origin_key(rel_meta.get("discourse_id"), rel_meta.get("field"))
                    if origin_key:
                        rel_meta.setdefault("origin_key", str(origin_key).strip())

                    raw = f"{src}|{pred}|{tgt}|{did}"
                    uri = rel_data.get("uri") or f"{base_etd}rel_{safe_hash(raw)}"

                    rels.append(
                        Relation(
                            uri=uri,
                            source=src,
                            target=tgt,
                            relation=pred,
                            confidence=rel_data.get("confidence", 0.0),
                            target_literal=rel_data.get("target_literal"),
                            target_datatype=rel_data.get("target_datatype"),
                            target_is_literal=bool(rel_data.get("target_is_literal", False)),
                            object_type=rel_data.get("object_type"),
                            qualifiers=rel_data.get("qualifiers", {}) or {},
                            predicate_text=rel_data.get("predicate_text"),
                            predicate_mapping_status=rel_data.get("predicate_mapping_status", "mapped"),
                            negated=bool(rel_data.get("negated", False)),
                            temporal_information=rel_data.get("temporal_information", {}) or {},
                            quantitative_information=rel_data.get("quantitative_information", {}) or {},
                            context=rel_data.get("evidence") or rel_data.get("context"),
                            is_inferred=bool(rel_data.get("is_inferred", False)),
                            metadata=rel_meta,
                        )
                    )


        for key in ["themes", "questions", "answers", "votes", "experts"]:
            if key in self.annotations:
                collect(key)

        # Include inferred relations persisted at metadata level.
        # Contract: annotations["metadata"]["inferred_relations"] is list[dict].
        metadata = self.annotations.get("metadata", {})
        if isinstance(metadata, dict):
            inferred = metadata.get("inferred_relations", [])
            if isinstance(inferred, list):
                for rel_data in inferred:
                    if not isinstance(rel_data, dict):
                        continue

                    src = rel_data.get("source_id") or rel_data.get("source")
                    tgt = rel_data.get("target_id") or rel_data.get("target")
                    pred = rel_data.get("relation")
                    rel_meta = rel_data.get("metadata", {}) or {}
                    did = rel_meta.get("discourse_id")
                    field = rel_meta.get("field")
                    if field is not None and str(field).strip():
                        rel_meta["field"] = str(field).strip()
                    if rel_meta.get("origin_key") is None:
                        ok = _compose_origin_key(did, rel_meta.get("field"))
                        if ok:
                            rel_meta["origin_key"] = ok

                    raw = f"{src}|{pred}|{tgt}|{did}|inferred"
                    uri = rel_data.get("uri") or f"{base_etd}rel_{safe_hash(raw)}"

                    rels.append(
                        Relation(
                            uri=uri,
                            source=src,
                            target=tgt,
                            relation=pred,
                            confidence=rel_data.get("confidence", 0.0),
                            target_literal=rel_data.get("target_literal"),
                            target_datatype=rel_data.get("target_datatype"),
                            target_is_literal=bool(rel_data.get("target_is_literal", False)),
                            object_type=rel_data.get("object_type"),
                            qualifiers=rel_data.get("qualifiers", {}) or {},
                            predicate_text=rel_data.get("predicate_text"),
                            predicate_mapping_status=rel_data.get("predicate_mapping_status", "mapped"),
                            negated=bool(rel_data.get("negated", False)),
                            temporal_information=rel_data.get("temporal_information", {}) or {},
                            quantitative_information=rel_data.get("quantitative_information", {}) or {},
                            context=rel_data.get("evidence") or rel_data.get("context"),
                            is_inferred=True,
                            metadata=rel_meta,
                        )
                    )

        return rels

    def _mint_subject_iri(self, label: str) -> URIRef:
        """Mint a stable ETD IRI for unresolved relation subjects (SHACL requires IRI)."""
        etd_base = self.namespaces["etd"]
        etd_ns = Namespace(etd_base)
        txt = (label or "Unknown").strip() or "Unknown"
        u = URIRef(f"{etd_base}entity_{safe_hash(txt)}")
        # Materialise minimal node
        self.graph.add((u, RDF.type, etd_ns.Entity))
        self.graph.add((u, RDFS.label, Literal(txt, lang="en")))
        return u

    def _resolve_relation_node(
        self,
        endpoint: Any,
        for_subject: bool,
    ) -> Union[URIRef, Literal]:
        """
        Resolve a relation endpoint:

        1. If it's already an RDF node, return as-is (except: literal subjects are converted to minted ETD IRIs).
        2. If it matches a known entity (by id, label, uri) → that entity URIRef.
        3. If it looks like a full IRI or CURIE → URIRef.
        4. Otherwise:
        - for subjects: mint a stable ETD IRI (URIRef) with rdfs:label @en
        - for objects: use an xsd:string literal
        """
        # Already an RDF node
        if isinstance(endpoint, (URIRef, Literal)):
            if isinstance(endpoint, Literal) and for_subject:
                # SHACL requires IRI subjects → mint stable ETD IRI instead of BNode
                return self._mint_subject_iri(str(endpoint))            
            
            return endpoint

        # Map numeric index/id
        if isinstance(endpoint, int):
            ent = self.entity_index.get(str(endpoint))
            if ent is not None and getattr(ent, "uri", None):
                return _safe_uriref(str(ent.uri), self.namespaces)

        # String handling
        if isinstance(endpoint, str):
            s = endpoint.strip()
            if not s:
                if for_subject:
                    return self._mint_subject_iri("Unknown")                
               
                return Literal("", datatype=XSD.string)

            # Exact entity hit: id / uri / label
            ent = (
                self.entity_index.get(s)
                or self.entity_index.get(s.lower())
            )
            if ent is not None and getattr(ent, "uri", None):
                return _safe_uriref(str(ent.uri), self.namespaces)

            # Full IRI
            if s.startswith("http://") or s.startswith("https://"):
                return _safe_uriref(s, self.namespaces)

            # CURIE with known prefix
            if ":" in s and not s.startswith("urn:"):
                prefix, local = s.split(":", 1)
                if prefix in self.namespaces:
                    iri = self.namespaces[prefix] + local
                    return _safe_uriref(iri, self.namespaces)

            # Otherwise fallback
            if for_subject:
                return self._mint_subject_iri(s)            
        
        # Fallback: stringify
        if for_subject:
            return self._mint_subject_iri(str(endpoint))

        # Object fallback: always return a literal (prevents None)
        return Literal(str(endpoint).strip(), datatype=XSD.string)

    def _relation_literal_object(self, rel: Relation) -> Optional[Literal]:
        if not bool(getattr(rel, "target_is_literal", False)) and getattr(rel, "target_literal", None) is None:
            return None
        raw = getattr(rel, "target_literal", None)
        if raw is None:
            raw = getattr(rel, "target", None)
        datatype = str(getattr(rel, "target_datatype", None) or "").strip().lower()
        if raw is None:
            return None
        if datatype in {"xsd:integer", "integer", "count"}:
            try:
                return Literal(int(raw), datatype=XSD.integer)
            except Exception:
                return Literal(str(raw), datatype=XSD.string)
        if datatype in {"xsd:float", "float", "double", "decimal", "percentage", "money"}:
            try:
                return Literal(float(str(raw).replace(",", "")), datatype=XSD.float)
            except Exception:
                return Literal(str(raw), datatype=XSD.string)
        if datatype in {"xsd:boolean", "boolean"}:
            sval = str(raw).strip().lower()
            if sval in {"true", "1", "yes"}:
                return Literal(True, datatype=XSD.boolean)
            if sval in {"false", "0", "no"}:
                return Literal(False, datatype=XSD.boolean)
        if datatype in {"xsd:date", "date"}:
            return Literal(str(raw), datatype=XSD.date)
        if datatype in {"xsd:datetime", "datetime", "xsd:dateTime".lower()}:
            return Literal(str(raw), datatype=XSD.dateTime)
        return Literal(str(raw), datatype=XSD.string)

    def _ensure_typed_node(self, node: Any, etd_ns: Namespace) -> None:
        """
        Graph-closure: ensure any URIRef endpoint used in relations is materialised
        with at least rdf:type etd:Entity. This prevents dangling endpoints.
        """
        if not isinstance(node, URIRef):
            return

        # If node already has any triple, it's materialised enough
        if (node, None, None) in self.graph:
            return

        # Minimal typing (safe default)
        self.graph.add((node, RDF.type, etd_ns.Entity))


    def _map_rating_value_to_skos(self, pred_local: str, obj: Any) -> Optional[URIRef]:
        """
        Map literal rating values used in risk assessments (e.g., 'High') to
        SKOS concept URIs minted in the ETD namespace (e.g., etd:SeverityScaleHigh).

        This assumes add_schema.py creates:
        - etd:SeverityScale, etd:LikelihoodScale, etd:RiskLevelScale (skos:ConceptScheme)
        - etd:SeverityScaleHigh/Medium/Low (skos:Concept), etc.
        """
        if obj is None:
            return None

        # If object is already a URI/BNode, keep it
        if isinstance(obj, (URIRef, BNode)):
            return None

        # Convert Literal -> string
        if isinstance(obj, Literal):
            val = str(obj)
        else:
            val = str(obj)

        val = val.strip()
        if not val:
            return None

        # Normalise common variants
        norm = val.lower()
        normalise = {
            "very high": "VeryHigh",
            "high": "High",
            "medium": "Medium",
            "moderate": "Medium",
            "low": "Low",
            "very low": "VeryLow",
        }
        if norm not in normalise:
            # If you later add numeric scales, handle them here
            return None

        level = normalise[norm]

        scheme_for_pred = {
        "hasSeverity": "SeverityScale",
            "hasLikelihood": "LikelihoodScale",
            "hasRiskLevel": "RiskLevelScale",
        }
        scheme = scheme_for_pred.get(pred_local)
        if not scheme:
            return None

        etd_ns = Namespace(self.namespaces["etd"])
        return etd_ns[f"{scheme}{level}"]


    def _is_vote_relation(self, rel: Relation) -> bool:
        meta = getattr(rel, "metadata", None) or {}
        # Only semantic vote relations should trigger vote-node materialization.
        # Structural backbone relations (castBy / receivesVote) are already
        # emitted directly from relation records and must not be remapped here.
        return meta.get("extractor") == "vote"

    def _label_for_node(self, node: Any) -> str:
        """Best-effort human label for URI/BNode/Literal nodes."""
        if isinstance(node, URIRef):
            existing = next(self.graph.objects(node, RDFS.label), None)
            if isinstance(existing, Literal) and str(existing).strip():
                return str(existing).strip()
            s = str(node)
            tail = s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            return tail or s
        if isinstance(node, Literal):
            return str(node).strip() or "Unknown"
        return str(node).strip() or "Unknown"

    def _spo_key(self, subj: Any, pred_uri: Any, obj: Any) -> str:
        return f"{str(subj)}|{str(pred_uri)}|{str(obj)}"

    def _has_reified_relation_for_spo(self, subj: Any, pred_uri: Any, obj: Any, etd_ns: Namespace) -> bool:
        key = self._spo_key(subj, pred_uri, obj)
        if not hasattr(self, "_reified_spo_keys"):
            self._reified_spo_keys = set()
        if key in self._reified_spo_keys:
            return True

        for rel_node in self.graph.subjects(RDF.type, etd_ns.Relation):
            if (
                (rel_node, etd_ns.subject, subj) in self.graph
                and (rel_node, etd_ns.predicate, pred_uri) in self.graph
                and (rel_node, etd_ns.object, obj) in self.graph
            ):
                self._reified_spo_keys.add(key)
                return True
        return False

    def _emit_reified_backbone_relation(
        self,
        *,
        subj: Any,
        pred_local: str,
        obj: Any,
        etd_ns: Namespace,
        discourse_id: Any = None,
        chunk_index: Any = None,
        field: Optional[str] = None,
        origin_key: Optional[str] = None,
        evidence_text: str,
        source: str = "dataset_vote",
    ) -> None:
        """
        Emit direct triple + deterministic reified Relation node for a backbone edge.
        Deduplicates if an equivalent reified SPO already exists.
        """
        pred_uri = etd_ns[pred_local]
        subj = self._canonicalize_relation_endpoint_uri(subj)
        obj = self._canonicalize_relation_endpoint_uri(obj)

        # Maintain contract: direct triple must always exist.
        self.graph.add((subj, pred_uri, obj))

        if self._has_reified_relation_for_spo(subj, pred_uri, obj, etd_ns):
            return

        rid_raw = f"{str(subj)}|{pred_local}|{str(obj)}|{source}|{discourse_id}"
        rnode = URIRef(f"{self.namespaces['etd']}rel_{safe_hash(rid_raw)}")
        rnode = self._ensure_relation_node_identity(rnode, subj, pred_uri, obj, etd_ns)
        self.graph.add((rnode, RDF.type, etd_ns.Relation))
        self.graph.add((rnode, etd_ns.subject, subj))
        self.graph.add((rnode, etd_ns.predicate, pred_uri))
        self.graph.add((rnode, etd_ns.object, obj))
        self.graph.add((rnode, etd_ns.confidence, Literal(1.0, datatype=XSD.float)))
        self.graph.add((rnode, etd_ns.isInferred, Literal(False, datatype=XSD.boolean)))

        rel_meta = {
            "extractor": "structural",
            "context_source": source,
            "evidence": evidence_text,
        }
        if discourse_id is not None:
            rel_meta["discourse_id"] = discourse_id
        if chunk_index is not None:
            rel_meta["chunk_index"] = chunk_index
        if field is not None and str(field).strip():
            rel_meta["field"] = str(field).strip()
        if origin_key is not None and str(origin_key).strip():
            rel_meta["origin_key"] = str(origin_key).strip()
        if rel_meta.get("discourse_id") is not None:
            rel_meta["source_doc_id"] = str(rel_meta.get("discourse_id"))
        ci = self._coerce_int(chunk_index)
        if ci is not None and ci >= 0:
            rel_meta["groundingStatus"] = "chunk_grounded"
            rel_meta["groundingMethod"] = "single_chunk_discourse"
            rel_meta["chunk_index_unknown"] = False
        else:
            rel_meta["groundingStatus"] = "ungrounded"
            rel_meta["groundingMethod"] = "structural"
            rel_meta["chunk_index_unknown"] = True

        self._add_relation_metadata(rnode, rel_meta, etd_ns)

        rel_obj = Relation(
            uri=str(rnode),
            source=str(subj),
            relation=pred_local,
            target=str(obj),
            confidence=1.0,
            is_inferred=False,
            metadata=rel_meta,
        )
        self._add_evidence_node(rnode, rel_obj, etd_ns)
        self._reified_spo_keys.add(self._spo_key(subj, pred_uri, obj))

    def _relation_confidence_for_spo(self, subj: URIRef, pred_uri: URIRef, obj: URIRef, etd_ns: Namespace) -> float:
        """Best-effort confidence lookup from existing reified relation nodes."""
        best = 0.0
        for rel_node in self.graph.subjects(RDF.type, etd_ns.Relation):
            if (
                (rel_node, etd_ns.subject, subj) in self.graph
                and (rel_node, etd_ns.predicate, pred_uri) in self.graph
                and (rel_node, etd_ns.object, obj) in self.graph
            ):
                for c in self.graph.objects(rel_node, etd_ns.confidence):
                    try:
                        best = max(best, float(c))
                    except Exception:
                        continue
        return best

    def _remove_reified_backbone_spo(self, subj: URIRef, pred_uri: URIRef, obj: URIRef, etd_ns: Namespace) -> None:
        """Remove reified relation nodes (and attached evidence) for a specific SPO."""
        to_remove: List[URIRef] = []
        for rel_node in self.graph.subjects(RDF.type, etd_ns.Relation):
            if (
                (rel_node, etd_ns.subject, subj) in self.graph
                and (rel_node, etd_ns.predicate, pred_uri) in self.graph
                and (rel_node, etd_ns.object, obj) in self.graph
            ):
                to_remove.append(rel_node)

        for rel_node in to_remove:
            evidence_nodes = list(self.graph.objects(rel_node, etd_ns.hasEvidence))
            # Remove all triples where relation node is subject/object
            for t in list(self.graph.triples((rel_node, None, None))):
                self.graph.remove(t)
            for t in list(self.graph.triples((None, None, rel_node))):
                self.graph.remove(t)

            # Remove evidence nodes fully to avoid dangling evidence
            for ev in evidence_nodes:
                for t in list(self.graph.triples((ev, None, None))):
                    self.graph.remove(t)
                for t in list(self.graph.triples((None, None, ev))):
                    self.graph.remove(t)

        # Keep de-dupe cache in sync
        if hasattr(self, "_reified_spo_keys"):
            self._reified_spo_keys.discard(self._spo_key(subj, pred_uri, obj))

    def _enforce_vote_castby_backbone(self, etd_ns: Namespace) -> None:
        """
        Guarantee every Vote has exactly one castBy IRI.
        - Materialise inverse from (expert castVote vote) when needed.
        - Resolve multi-candidate deterministically (highest confidence, then sorted URI).
        - Reify castBy edge 1:1 with deterministic evidence.
        """
        strict = bool(getattr(self.config, "FAIL_ON_SHACL_VIOLATIONS", False))
        missing_cast_by = 0
        cast_by_uri = etd_ns.castBy
        cast_vote_uri = etd_ns.castVote

        vote_nodes = sorted(
            {v for v in self.graph.subjects(RDF.type, etd_ns.Vote) if isinstance(v, URIRef)},
            key=str,
        )
        for vote in vote_nodes:
            candidate_scores: Dict[URIRef, float] = {}

            # Direct candidates from existing castBy triples
            for expert in self.graph.objects(vote, cast_by_uri):
                if isinstance(expert, URIRef):
                    score = self._relation_confidence_for_spo(vote, cast_by_uri, expert, etd_ns)
                    candidate_scores[expert] = max(candidate_scores.get(expert, 0.0), score)

            # Inverse materialization candidates from expert castVote vote
            for expert in self.graph.subjects(cast_vote_uri, vote):
                if isinstance(expert, URIRef):
                    score = self._relation_confidence_for_spo(expert, cast_vote_uri, vote, etd_ns)
                    candidate_scores[expert] = max(candidate_scores.get(expert, 0.0), score)

            if not candidate_scores:
                missing_cast_by += 1
                logger.error("Vote missing castBy candidate: %s", vote)
                if strict:
                    raise RuntimeError(f"Vote data-contract violation: missing castBy for vote {vote}")
                continue

            # Deterministic pick: highest score, then lexicographic URI.
            chosen = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
            if len(candidate_scores) > 1:
                logger.warning(
                    "Multiple castBy candidates for %s; chosen=%s candidates=%s",
                    vote,
                    chosen,
                    [str(u) for u in sorted(candidate_scores.keys(), key=str)],
                )

            # Remove any non-chosen castBy edges + reified equivalents.
            for expert in list(self.graph.objects(vote, cast_by_uri)):
                if isinstance(expert, URIRef) and expert != chosen:
                    self.graph.remove((vote, cast_by_uri, expert))
                    self._remove_reified_backbone_spo(vote, cast_by_uri, expert, etd_ns)

            # Ensure chosen direct triple exists.
            self.graph.add((vote, cast_by_uri, chosen))

            # Reify castBy 1:1 with deterministic evidence, dedup-safe.
            did_candidates: Set[str] = set()
            for v in self.graph.objects(vote, etd_ns.discourse_id):
                raw = str(v).strip()
                if raw:
                    did_candidates.add(raw)
            chosen_did = sorted(did_candidates)[0] if did_candidates else None

            chunk_candidates: Set[int] = set()
            for v in self.graph.objects(vote, etd_ns.chunk_index):
                c = self._coerce_int(v)
                if c is not None and c >= 0:
                    chunk_candidates.add(int(c))
            chosen_chunk = sorted(chunk_candidates)[0] if chunk_candidates else None

            field_candidates: Set[str] = set()
            for v in self.graph.objects(vote, etd_ns.field):
                raw = str(v).strip()
                if raw:
                    field_candidates.add(raw)
            chosen_field = sorted(field_candidates)[0] if field_candidates else "vote.vote_value"
            chosen_origin = (
                f"{chosen_field}|{chosen_did}"
                if (chosen_did and chosen_field)
                else (chosen_did or None)
            )

            vote_lbl = self._label_for_node(vote)
            expert_lbl = self._label_for_node(chosen)
            evidence_text = f"Vote record: {expert_lbl} cast {vote_lbl}"
            self._emit_reified_backbone_relation(
                subj=vote,
                pred_local="castBy",
                obj=chosen,
                etd_ns=etd_ns,
                discourse_id=chosen_did,
                chunk_index=chosen_chunk,
                field=chosen_field,
                origin_key=chosen_origin,
                evidence_text=evidence_text,
                source="dataset_vote",
            )

        if missing_cast_by > 0:
            meta = self.annotations.setdefault("metadata", {})
            diagnostics = meta.setdefault("diagnostics", {})
            diagnostics["missing_castBy_votes"] = missing_cast_by
            logger.warning("Vote castBy diagnostics: missing_castBy_votes=%d", missing_cast_by)

    def _assert_backbone_relation_types(self, rel: Relation, pred_local: str) -> None:
        """Hard invariant: validate backbone relation endpoint types before emit."""
        if pred_local not in {"castBy", "receivesVote"}:
            return

        src_ent = self.entity_index.get(str(rel.source).strip())
        tgt_ent = self.entity_index.get(str(rel.target).strip())
        src_type = str(getattr(src_ent, "entity_type", "") or "")
        tgt_type = str(getattr(tgt_ent, "entity_type", "") or "")

        if pred_local == "castBy":
            if src_type != "Vote":
                raise RuntimeError(
                    f"Backbone invariant failed: castBy subject must be Vote, got {src_type or 'Unknown'} ({rel.source})"
                )
            if tgt_type not in {"Expert", "Person"}:
                raise RuntimeError(
                    f"Backbone invariant failed: castBy object must be Expert/Person, got {tgt_type or 'Unknown'} ({rel.target})"
                )

        if pred_local == "receivesVote":
            if src_type != "Answer":
                raise RuntimeError(
                    f"Backbone invariant failed: receivesVote subject must be Answer, got {src_type or 'Unknown'} ({rel.source})"
                )
            if tgt_type != "Vote":
                raise RuntimeError(
                    f"Backbone invariant failed: receivesVote object must be Vote, got {tgt_type or 'Unknown'} ({rel.target})"
                )


    def _add_relations(self) -> None:
        """
        Emit relations in two forms:

        (1) DIRECT SEMANTIC TRIPLE:
            <subject>  etd:<predicate>  <object>

        (2) RELATION METADATA NODE:
            _:r  a etd:Relation ;
                  etd:subject <subject> ;
                  etd:predicate etd:<predicate> ;
                  etd:object <object> ;
                  etd:confidence xsd:float ;
                  etd:context_source xsd:string ;
                  etd:isInferred xsd:boolean ;
                  etd:discourse_id xsd:string ;
                  ... (other metadata)

          This ensures relations ALWAYS appear in TTL.
          """
        etd_ns = Namespace(self.namespaces["etd"])

        base_etd = self.namespaces.get("etd", "")
        if not base_etd:
            raise ValueError("Missing 'etd' namespace in RDFSerializer.namespaces")

        # Flatten relation objects
        all_rels = self._get_all_relations()
        self._reified_spo_keys = set()

        for rel in all_rels:
            meta = getattr(rel, "metadata", None) or {}

            # Root-cause fix (bounded correction pass, item 1 -- endpoint/
            # surface-form ordering bug): this reconciliation previously
            # lived ONLY in _add_relation_metadata(), which runs further
            # down this same loop AFTER _endpoint_surface_form_compatible()
            # has already been evaluated (see below). Since
            # subject_surface_form/object_surface_form are only ever
            # populated upstream for relations whose predicate passes
            # through SemanticRelationPolicy.validate() (see
            # utils/semantic_relation_policy.py), any relation with an
            # unregistered/local predicate reached the compatibility check
            # with an EMPTY surface form, which the check's own documented
            # "nothing to compare against -> compatible" fallback then let
            # straight through -- even when the raw subject_text/
            # object_text genuinely disagreed with the bound entity's
            # label. Performing the same fill-only reconciliation here,
            # before the check runs, closes that gap; _add_relation_
            # metadata()'s own copy of this logic is now a no-op for any
            # relation already reconciled here (it only ever fills a gap,
            # never overwrites).
            if not meta.get("subject_surface_form") and meta.get("subject_text"):
                meta["subject_surface_form"] = meta["subject_text"]
            if not meta.get("object_surface_form") and meta.get("object_text"):
                meta["object_surface_form"] = meta["object_text"]

            def _maybe_expand_local_to_etd(value: Any) -> Any:
                if not isinstance(value, str) or not base_etd:
                    return value
                s = value.strip()
                if not s:
                    return value

                # If already absolute IRI, keep
                if s.startswith("http://") or s.startswith("https://"):
                    return s

                # If CURIE-like (prefix:local), keep (resolver will expand later)
                if ":" in s and " " not in s:
                    prefix = s.split(":", 1)[0]
                    if prefix in self.namespaces:
                        return s

                # Otherwise treat as ETD-local and expand
                return base_etd + s.lstrip("#/")

            # Use it:
            rel.source = _maybe_expand_local_to_etd(rel.source)
            rel.target = _maybe_expand_local_to_etd(rel.target)


            # -------------------------
            # 1. Determine predicate
            # -------------------------
            raw_pred = rel.relation or "relatedTo"
            pred_local = self._sanitize(str(raw_pred))
            pred_uri = etd_ns[pred_local]   # etd:uses, etd:hasRisk, etc.
            relation_kind = str(meta.get("relation_kind", "") or "").strip()

            if relation_kind in {"structural", "provenance"} and pred_local in {
                "relatedTo",
                "uses",
                "trainedOn",
                "hasRisk",
                "exhibitsRisk",
                "causes",
                "affects",
                "mitigates",
                "mitigatedBy",
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
            }:
                logger.warning(
                    "Skipping structural/provenance relation with semantic predicate %s for %s -> %s",
                    pred_local,
                    rel.source,
                    rel.target,
                )
                continue

            # Enforce strict type contract for structural vote backbone edges.
            self._assert_backbone_relation_types(rel, pred_local)

            # -------------------------
            # 2. Resolve SUBJECT / OBJECT
            # -------------------------
            subj = self._resolve_relation_node(rel.source, for_subject=True)
            obj = self._relation_literal_object(rel)
            if obj is None:
                obj = self._resolve_relation_node(rel.target, for_subject=False)
            subj = self._canonicalize_relation_endpoint_uri(subj)
            if not isinstance(obj, Literal):
                obj = self._canonicalize_relation_endpoint_uri(obj)

            self._hydrate_relation_endpoint_provenance(subj, rel, meta, "subject", etd_ns)
            self._hydrate_relation_endpoint_provenance(obj, rel, meta, "object", etd_ns)

            # ✅ FIX: graph closure (prevent dangling endpoints)
            self._ensure_typed_node(subj, etd_ns)
            self._ensure_typed_node(obj, etd_ns)

            # Root-cause fix (manual semantic-QA pass, category B; made
            # recovery-capable in the bounded correction pass): a
            # relation whose resolved endpoint entity is textually
            # incompatible with the surface text actually captured for
            # it (e.g. subject resolved to "AI Recruitment Tool" while
            # subject_surface_form is "names associated with Black
            # women") must not survive. This used to be enforced HERE,
            # by unconditionally dropping the relation -- but that ran
            # per-relation, in isolation, with no visibility into
            # sibling relations elsewhere in the SAME discourse that
            # might name the entity this surface form actually refers
            # to, so a genuinely recoverable case (the true referent
            # already exists as another entity in this discourse) was
            # always dropped rather than repointed. The check itself
            # (_endpoint_surface_form_compatible) is unchanged and still
            # used, but enforcement is now deferred to
            # _enforce_endpoint_surface_compatibility_final, which runs
            # once after every relation (and consolidation) has been
            # materialised, so it can attempt that discourse-scoped
            # recovery before falling back to dropping. See that
            # method's docstring for the full recovery/drop policy.

            # ----------------------------------------------------
            # SKOS: map rating literals to SKOS Concepts
            # ----------------------------------------------------
            if pred_local in {"hasSeverity", "hasLikelihood", "hasRiskLevel"}:
                mapped = self._map_rating_value_to_skos(pred_local, obj)
                if mapped is not None:
                    obj = mapped


            # -------------------------
            # 3. Normalize SUBJECT / OBJECT node types (safety net; always
            #    applied so downstream rnode construction sees valid RDF
            #    terms regardless of whether the direct triple is emitted)
            # -------------------------
            if not isinstance(subj, (URIRef, BNode)):
                subj = self._mint_subject_iri(str(subj))
            if not isinstance(obj, (URIRef, BNode, Literal)):
                obj = Literal(str(obj), datatype=XSD.string)

            # -------------------------
            # 3b. Emit DIRECT triple -- only when the relation provides
            #     POSITIVE ASSERTED SUPPORT for this exact (s, p, o).
            #
            # A reified relation may correctly be negated, or carry a
            # non-asserted modality (questioned / hypothetical / uncertain /
            # conditional / normative / reported) on etd:epistemic_status OR
            # inside etd:modality_tags, but an unqualified direct
            # <subject> <predicate> <object> triple has no way to carry that
            # qualification -- it reads as a plain positive fact to any
            # consumer querying direct triples without inspecting the
            # reified node. The single shared rule
            # (_relation_provides_positive_asserted_support, also used by
            # _enforce_direct_spo_assertion_invariant) decides. The reified
            # node (etd:subject/predicate/object + etd:negated /
            # etd:epistemic_status / etd:modality_tags) and its Evidence are
            # emitted exactly as before, further down in this same loop.
            # -------------------------
            if not self._direct_spo_suppressed_by_assertion_status(meta):
                self.graph.add((subj, pred_uri, obj))
            else:
                record_event(
                    self.quality_events,
                    category="direct_spo_suppressed",
                    action="suppressed",
                    discourse_id=meta.get("discourse_id"),
                    source_relation_id=str(getattr(rel, "uri", "") or "") or None,
                    subject_surface=str(meta.get("subject_surface_form") or meta.get("subject_text") or "").strip() or None,
                    predicate_surface=pred_local,
                    object_surface=str(meta.get("object_surface_form") or meta.get("object_text") or "").strip() or None,
                    reason=(
                        f"negated={bool(meta.get('negated'))}, "
                        f"epistemic_status={meta.get('epistemic_status')!r}, "
                        f"modality_tags={meta.get('modality_tags')!r} -- "
                        "no positive asserted support; unqualified direct triple would "
                        "misrepresent a non-asserted proposition as a plain fact"
                    ),
                )

            # Ensure relation URI is absolute
            if rel.uri and isinstance(rel.uri, str) and not rel.uri.startswith("http"):
                rel.uri = base_etd + rel.uri.lstrip("#/")

            # -------------------------
            # 4. Create RELATION METADATA NODE
            # -------------------------
            rnode = URIRef(rel.uri) if rel.uri else BNode()
            if isinstance(rnode, URIRef):
                rnode = self._ensure_relation_node_identity(rnode, subj, pred_uri, obj, etd_ns)
                rel.uri = str(rnode)
            self.graph.add((rnode, RDF.type, etd_ns.Relation))

            # ----------------------------------------------------
            # VOTE SUPPORT: attach an explicit etd:Vote node + SKOS voteValue
            # ----------------------------------------------------
            if self._is_vote_relation(rel):
                vote_uri = self._canonical_vote_uri_for_relation(rel, subj, obj, meta, etd_ns)

                # Canonical vote typing
                self.graph.add((vote_uri, RDF.type, etd_ns.Vote))
                self.graph.add((vote_uri, RDF.type, etd_ns.Entity))
                self.graph.add((vote_uri, RDF.type, PROV.Activity))

                # Deterministic vote labels for UX/explainability.
                expert_lbl = self._label_for_node(subj)
                answer_lbl = self._label_for_node(obj)
                vote_label = f"Vote by {expert_lbl} for {answer_lbl}"
                self._set_vote_labels(vote_uri, vote_label)

                # Link Vote to the participants using auxiliary predicates only.
                # Backbone predicates (castBy / receivesVote) are emitted only
                # from explicit structural relations so they are always reified.
                discourse_id = meta.get("discourse_id")
                chunk_index = meta.get("chunk_index")
                vote_field = str(meta.get("field") or "vote.vote_value").strip()
                if discourse_id is not None:
                    self._add_discourse_id_literal(vote_uri, discourse_id, etd_ns)
                if vote_field:
                    self._emit_registered_etd_literal(vote_uri, "field", vote_field, etd_ns, datatype=XSD.string)
                ci = self._coerce_int(chunk_index)
                if ci is not None and ci >= 0:
                    self.graph.add((vote_uri, etd_ns.chunk_index, Literal(int(ci), datatype=XSD.integer)))
                vote_evidence = f"Vote record: {expert_lbl} cast {vote_label} for {answer_lbl}"

                self._emit_reified_backbone_relation(
                    subj=obj,
                    pred_local="receivesVote",
                    obj=vote_uri,
                    etd_ns=etd_ns,
                    discourse_id=discourse_id,
                    chunk_index=chunk_index,
                    field=vote_field,
                    origin_key=meta.get("origin_key"),
                    evidence_text=vote_evidence,
                    source="dataset_vote",
                )

                # Optional convenience links (if you want the inverse materialised)
                self._emit_reified_backbone_relation(
                    subj=subj,
                    pred_local="castVote",
                    obj=vote_uri,
                    etd_ns=etd_ns,
                    discourse_id=discourse_id,
                    chunk_index=chunk_index,
                    field=vote_field,
                    origin_key=meta.get("origin_key"),
                    evidence_text=vote_evidence,
                    source="dataset_vote",
                )
                self.graph.add((obj, etd_ns.hasVote, vote_uri))

                # rawVoteValue literal (xsd:integer)
                raw_int = meta.get("raw_vote_value")
                if raw_int is not None:
                    try:
                        self.graph.add((vote_uri, etd_ns.rawVoteValue, Literal(int(raw_int), datatype=XSD.integer)))
                    except Exception:
                        pass

                # voteValue as SKOS concept (object property)
                concept = meta.get("vote_value_concept")
                if concept:
                    c = str(concept).strip()

                    # 1) Try normal resolution (full IRI / CURIE / indexed entity)
                    concept_ref = self._resolve_entity_uri_from_value(c)

                    # 2) Fallback: ETD-local concept (e.g., "VoteValueSchemeUpvote")
                    if concept_ref is None and ":" not in c and not (
                        c.startswith("http://") or c.startswith("https://")
                    ):
                        concept_ref = URIRef(f"{self.namespaces['etd']}{c.lstrip('#/')}")


                    if isinstance(concept_ref, URIRef):
                        self.graph.add((vote_uri, etd_ns.voteValue, concept_ref))


                # polarity as string (optional, keeps your old normalised value)
                pol = meta.get("vote_value")
                if pol:
                    self.graph.add((vote_uri, etd_ns.polarity, Literal(str(pol), datatype=XSD.string)))

                vote_meta = dict(meta or {})
                vote_meta.setdefault("surface_form", f"Vote {discourse_id}" if discourse_id is not None else "Vote")
                vote_meta.setdefault("context_source", vote_meta.get("context_source") or "dataset_vote")
                vote_meta.setdefault("source", vote_meta.get("source") or vote_meta.get("extractor") or "vote")
                vote_meta.setdefault("source_doc_id", str(discourse_id).strip() if discourse_id is not None else None)
                vote_meta.setdefault("chunk_index", self._coerce_int(meta.get("chunk_index")) if meta.get("chunk_index") is not None else -1)
                if discourse_id is not None:
                    vote_meta.setdefault("discourse_id", discourse_id)
                if vote_field:
                    vote_meta.setdefault("field", vote_field)
                self._add_entity_metadata(vote_uri, vote_meta, etd_ns)

            # link S, P, O
            self.graph.add((rnode, etd_ns.subject, subj))
            self.graph.add((rnode, etd_ns.predicate, pred_uri))
            self.graph.add((rnode, etd_ns.object, obj))
            self._reified_spo_keys.add(self._spo_key(subj, pred_uri, obj))


            # -------------------------
            # 5. Confidence
            # -------------------------
            self.graph.add((rnode, etd_ns.confidence, self._confidence_literal(rel.confidence or 0.0)))

            # -------------------------
            # 7. isInferred flag
            # -------------------------
            self.graph.add(
                (
                    rnode,
                    etd_ns.isInferred,
                    Literal(bool(getattr(rel, "is_inferred", False)), datatype=XSD.boolean),
                )
            )

            # -------------------------
            # 8. Additional metadata
            # -------------------------
            rel_meta = dict(meta)
            if rel_meta.get("qualifiers") in (None, {}, []):
                rel_meta["qualifiers"] = getattr(rel, "qualifiers", None) or {}
            if rel_meta.get("temporal_information") in (None, {}, []):
                rel_meta["temporal_information"] = getattr(rel, "temporal_information", None) or {}
            if rel_meta.get("quantitative_information") in (None, {}, []):
                rel_meta["quantitative_information"] = getattr(rel, "quantitative_information", None) or {}
            if rel_meta.get("target_is_literal") is None:
                rel_meta["target_is_literal"] = bool(getattr(rel, "target_is_literal", False))
            if rel_meta.get("target_datatype") in (None, "") and getattr(rel, "target_datatype", None):
                rel_meta["target_datatype"] = getattr(rel, "target_datatype", None)
            if rel_meta.get("object_type") in (None, "") and getattr(rel, "object_type", None):
                rel_meta["object_type"] = getattr(rel, "object_type", None)
            if rel_meta.get("predicate_mapping_status") in (None, "") and getattr(rel, "predicate_mapping_status", None):
                rel_meta["predicate_mapping_status"] = getattr(rel, "predicate_mapping_status", None)
            if rel_meta.get("negated") is None:
                rel_meta["negated"] = bool(getattr(rel, "negated", False))
            rel.metadata = rel_meta
            self._add_relation_metadata(rnode, rel_meta, etd_ns)
            if bool(getattr(rel, "is_inferred", False)):
                self._add_inferred_provenance(rnode, rel, etd_ns)

            self._add_evidence_node(rnode, rel, etd_ns)

        # Final vote backbone enforcement pass:
        # every Vote must have exactly one castBy IRI and castBy must be reified.
        self._enforce_vote_castby_backbone(etd_ns)


    def _add_evidence_node(self, rnode: URIRef, rel: Relation, etd_ns: Namespace) -> None:
        """
        Create one or more etd:Evidence nodes for a relation and attach provenance + evidence text.
        If rel.metadata["evidence_spans"] is present, emit one Evidence node per span.
        """
        base_meta = dict(getattr(rel, "metadata", None) or {})
        raw_spans = base_meta.get("evidence_spans")
        payloads: List[Tuple[int, Dict[str, Any]]] = []
        if isinstance(raw_spans, list) and raw_spans:
            seen_keys: Set[str] = set()
            for idx, item in enumerate(raw_spans):
                if not isinstance(item, dict):
                    continue
                p = dict(base_meta)
                p.pop("evidence_spans", None)
                p.update(item)
                pidx = self._coerce_int(p.get("chunk_index"))
                ps = self._coerce_int(p.get("char_start", p.get("start_offset")))
                pe = self._coerce_int(p.get("char_end", p.get("end_offset")))
                dedupe_key = f"{pidx}|{ps}|{pe}"
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                payloads.append((idx, p))
        if not payloads:
            payloads = [(0, dict(base_meta))]

        for payload_idx, meta in payloads:
            rid = str(rnode)

            # --- resolve evidence text EARLY (before minting ev_uri) ---
            # The Evidence node's identity key must fold in the evidence
            # text itself. Without this, two genuinely different evidence
            # occurrences attached to the same reified relation (e.g. the
            # same subject/predicate/object triple extracted independently
            # from two different semantic units/chunks, which resolve to
            # the SAME relation node since relation identity is keyed on
            # subject+predicate+target+discourse only, not on evidence or
            # chunk position) would collide onto one Evidence node -- each
            # occurrence would then attach its own evidenceText literal to
            # that single node, violating the exactly-one constraint.
            # Folding the resolved text into the key means: different text
            # -> different Evidence node (both preserved, distinguished);
            # identical text -> the same node (idempotent re-add, no
            # violation, no redundant node).
            did_for_evidence = meta.get("discourse_id") or getattr(rel, "discourse_id", None) or getattr(rel, "doc_id", None)
            src_for_evidence = str(meta.get("source") or meta.get("extractor") or meta.get("context_source") or "unknown").strip()
            chunk_num_for_evidence = self._coerce_int(meta.get("chunk_index"))
            derived_chunk_index_for_evidence = (
                int(chunk_num_for_evidence) if (chunk_num_for_evidence is not None and chunk_num_for_evidence >= 0) else -1
            )
            start_for_evidence = self._coerce_int(meta.get("start_offset", meta.get("start_char")))
            end_for_evidence = self._coerce_int(meta.get("end_offset", meta.get("end_char")))
            valid_span_for_evidence = (
                start_for_evidence is not None
                and end_for_evidence is not None
                and int(end_for_evidence) > int(start_for_evidence)
            )
            grounding_status_for_evidence = str(meta.get("groundingStatus") or "").strip()
            if derived_chunk_index_for_evidence < 0:
                if not grounding_status_for_evidence:
                    grounding_status_for_evidence = "ungrounded"
            elif valid_span_for_evidence:
                if not grounding_status_for_evidence or grounding_status_for_evidence == "ungrounded":
                    grounding_status_for_evidence = "grounded"
            else:
                grounding_status_for_evidence = "chunk_grounded"

            evidence_raw = (meta.get("evidence") or getattr(rel, "context", None) or "")
            raw_candidate = str(evidence_raw).strip()

            # The LLM's evidence_quote is a locator hint, not
            # authoritative source text. Resolve it deterministically
            # against Source_Context (already established above,
            # unmodified here) before it is ever serialized as evidenceText.
            # Synthetic placeholder text (raw_candidate empty) is left
            # entirely untouched -- it was never a quote to begin with.
            alignment_status: Optional[str] = None
            alignment_method: Optional[str] = None
            alignment_original_candidate: Optional[str] = None
            alignment_context_start: Optional[int] = None
            alignment_context_end: Optional[int] = None

            if raw_candidate:
                source_context_for_alignment = str(meta.get("source_context") or "").strip()
                if source_context_for_alignment:
                    alignment = align_evidence_span(
                        raw_candidate,
                        source_context_for_alignment,
                        hint_char_offset=start_for_evidence,
                    )
                    txt = alignment.text
                    alignment_status = alignment.status
                    alignment_method = alignment.method
                    alignment_original_candidate = alignment.original_candidate
                    alignment_context_start = alignment.context_char_start
                    alignment_context_end = alignment.context_char_end
                else:
                    txt = raw_candidate
            else:
                txt = ""

            if not txt:
                field_for_evidence = str(meta.get("field") or "unknown").strip()
                did_text_for_evidence = str(did_for_evidence).strip() if did_for_evidence is not None else ""
                if grounding_status_for_evidence == "grounded":
                    txt = f"Grounded evidence from {field_for_evidence} discourse {did_text_for_evidence} chunk {derived_chunk_index_for_evidence}."
                elif grounding_status_for_evidence == "chunk_grounded":
                    txt = f"Chunk-grounded provenance from {field_for_evidence} discourse {did_text_for_evidence} chunk {derived_chunk_index_for_evidence}."
                else:
                    txt = f"Structural provenance from {src_for_evidence} for {field_for_evidence} discourse {did_text_for_evidence}."
            if len(txt) > 500:
                txt = txt[:500]
                # Keep the exact-substring invariant intact for the
                # (rare) truncated case: a prefix of an exact raw-source
                # substring is still an exact raw-source substring at the
                # same start offset.
                if alignment_context_start is not None:
                    alignment_context_end = alignment_context_start + len(txt)

            ev_key = "|".join(
                [
                    rid,
                    str(payload_idx),
                    str(meta.get("chunk_index", "")),
                    str(meta.get("char_start", meta.get("start_offset", ""))),
                    str(meta.get("char_end", meta.get("end_offset", ""))),
                    safe_hash(txt),
                ]
            )
            ev_uri = URIRef(f"{self.namespaces['etd']}evidence_{safe_hash(ev_key)}")

            self.graph.add((ev_uri, RDF.type, etd_ns.Evidence))
            self.graph.add((rnode, etd_ns.hasEvidence, ev_uri))
            self._emit_registered_etd_literal(ev_uri, "datasetId", self.dataset_id, etd_ns, datatype=XSD.string)
            self._emit_registered_etd_literal(ev_uri, "runId", self.run_id, etd_ns, datatype=XSD.string)

            # extractor/source
            src = src_for_evidence
            self._emit_registered_etd_literal(ev_uri, "source", src, etd_ns, datatype=XSD.string)

            # Evidence-alignment provenance -- only present when
            # alignment was actually attempted (i.e. not for synthetic
            # structural placeholder text, which was never a quote).
            if alignment_status is not None:
                self._emit_registered_etd_literal(ev_uri, "evidence_alignment_status", alignment_status, etd_ns, datatype=XSD.string)
                self._emit_registered_etd_literal(ev_uri, "evidence_alignment_method", alignment_method or "", etd_ns, datatype=XSD.string)
                if alignment_original_candidate:
                    self._emit_registered_etd_literal(ev_uri, "original_evidence_candidate", alignment_original_candidate, etd_ns, datatype=XSD.string)
                if alignment_context_start is not None and alignment_context_end is not None:
                    self._emit_registered_etd_literal(ev_uri, "context_char_start", int(alignment_context_start), etd_ns, datatype=XSD.integer)
                    self._emit_registered_etd_literal(ev_uri, "context_char_end", int(alignment_context_end), etd_ns, datatype=XSD.integer)

            # confidence (required exactly once on Evidence)
            try:
                cval = float(getattr(rel, "confidence", 0.0) or 0.0)
            except Exception:
                cval = 0.0
            cval = max(0.0, min(1.0, cval))
            self._emit_registered_etd_literal(ev_uri, "confidence", float(cval), etd_ns, datatype=XSD.float)

            # discourse_id
            did = meta.get("discourse_id") or getattr(rel, "discourse_id", None) or getattr(rel, "doc_id", None)
            self._add_discourse_id_literal(ev_uri, did, etd_ns)
            origin_meta = dict(meta)
            if did is not None:
                origin_meta.setdefault("discourse_id", did)
            origin_meta.setdefault("source", src)
            origin_node = self._ensure_discourse_unit_node(origin_meta, etd_ns)
            if origin_node is not None:
                self.graph.add((ev_uri, etd_ns.extractedFrom, origin_node))

            # --- derive locals (do not mutate meta) ---
            chunk_num = self._coerce_int(meta.get("chunk_index"))
            derived_chunk_index = int(chunk_num) if (chunk_num is not None and chunk_num >= 0) else -1
            derived_chunk_unknown = True if derived_chunk_index == -1 else False

            derived_source_doc_id = None
            sdoc = meta.get("source_doc_id")
            if sdoc is not None and str(sdoc).strip():
                derived_source_doc_id = str(sdoc).strip()
            elif did is not None and str(did).strip():
                derived_source_doc_id = str(did).strip()

            derived_chunk_id = None
            cid = meta.get("chunk_id")
            if cid is not None and str(cid).strip():
                derived_chunk_id = str(cid).strip()
            elif derived_source_doc_id is not None and not derived_chunk_unknown:
                derived_chunk_id = f"{derived_source_doc_id}:chunk:{derived_chunk_index}"

            derived_start_offset = self._coerce_int(meta.get("start_offset", meta.get("start_char")))
            derived_end_offset = self._coerce_int(meta.get("end_offset", meta.get("end_char")))

            derived_span_hash = None
            sh = meta.get("span_hash")
            if sh is not None and str(sh).strip():
                derived_span_hash = str(sh).strip()
            elif derived_source_doc_id is not None and (
                derived_chunk_id
                or not derived_chunk_unknown
                or derived_start_offset is not None
                or derived_end_offset is not None
            ):
                span_key = "|".join(
                    [
                        derived_source_doc_id or "",
                        derived_chunk_id or "",
                        str(derived_chunk_index if derived_chunk_index >= 0 else ""),
                        str(derived_start_offset if derived_start_offset is not None else ""),
                        str(derived_end_offset if derived_end_offset is not None else ""),
                    ]
                )
                derived_span_hash = safe_hash(span_key)

            derived_meta = {
                "source_doc_id": derived_source_doc_id,
                "chunk_id": derived_chunk_id,
                "chunk_index": derived_chunk_index,
                "chunk_index_unknown": derived_chunk_unknown,
                "char_start": derived_start_offset,
                "char_end": derived_end_offset,
                "start_offset": derived_start_offset,
                "end_offset": derived_end_offset,
                "span_hash": derived_span_hash,
            }

            valid_span_offsets = (
                derived_start_offset is not None
                and derived_end_offset is not None
                and int(derived_end_offset) > int(derived_start_offset)
            )
            grounding_status = str(meta.get("groundingStatus") or "").strip()
            grounding_method = str(meta.get("groundingMethod") or "").strip()
            if derived_chunk_index < 0:
                if not grounding_status:
                    grounding_status = "ungrounded"
                if not grounding_method:
                    grounding_method = "structural" if src in {"structural", "vote", "reasoner"} else "none"
            else:
                if valid_span_offsets:
                    if not grounding_status or grounding_status == "ungrounded":
                        grounding_status = "grounded"
                    if not grounding_method or grounding_method == "none":
                        grounding_method = "span_map"
                else:
                    grounding_status = "chunk_grounded"
                    if not grounding_method or grounding_method == "none":
                        grounding_method = "single_chunk_discourse"
            derived_meta["groundingStatus"] = grounding_status
            derived_meta["groundingMethod"] = grounding_method

            # field + chunking provenance (production-safe)
            for key in (
                "field",
                "source_doc_id",
                "chunk_id",
                "chunk_index",
                "chunk_index_unknown",
                "groundingStatus",
                "groundingMethod",
                "chunk_total",
                "chunk_start",
                "start_char",
                "end_char",
                "char_start",
                "char_end",
                "start_offset",
                "end_offset",
                "span_hash",
                "batch_id",
                "ingestedAt",
            ):
                val = derived_meta.get(key, meta.get(key))
                if val is None:
                    continue

                if key in (
                    "chunk_index",
                    "chunk_total",
                    "chunk_start",
                    "start_char",
                    "end_char",
                    "char_start",
                    "char_end",
                    "start_offset",
                    "end_offset",
                ):
                    num = self._coerce_int(val)
                    if num is not None:
                        self._emit_registered_etd_literal(ev_uri, key, num, etd_ns, datatype=XSD.integer)
                    continue

                if key == "chunk_index_unknown":
                    bval = self._coerce_bool(val)
                    if bval is not None:
                        self._emit_registered_etd_literal(ev_uri, key, bool(bval), etd_ns, datatype=XSD.boolean)
                    continue

                if key == "ingestedAt":
                    raw_str = str(val).strip()
                    if not raw_str:
                        continue

                    if self._is_iso_datetime(raw_str):
                        self._emit_registered_etd_literal(ev_uri, key, raw_str, etd_ns, datatype=XSD.dateTime)
                    else:
                        self._emit_registered_etd_literal(ev_uri, key, raw_str, etd_ns, datatype=XSD.string)
                    continue

                self._emit_registered_etd_literal(ev_uri, key, str(val).strip(), etd_ns, datatype=XSD.string)

            # evidence text (belt-and-braces: always emit evidenceText) --
            # already resolved above, before ev_uri was minted, so the
            # Evidence node's identity key and its literal value are
            # guaranteed to agree.
            self.graph.add((ev_uri, etd_ns.evidenceText, Literal(txt, datatype=XSD.string)))

            qh = meta.get("quote_hash")
            qh = str(qh).strip() if (qh is not None and str(qh).strip()) else (safe_hash(txt) if txt else None)
            if qh:
                self._emit_registered_etd_literal(ev_uri, "quote_hash", qh, etd_ns, datatype=XSD.string)

    def _align_relation_evidence_provenance(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Align relation-level provenance to linked evidence without mutating evidence:
        - Relation provenance is stamped only when evidence provides one unambiguous value.
        - chunk_index=-1 evidence values are treated as unknown and excluded from alignment.
        """
        processed = 0
        ambiguous_did = 0
        ambiguous_chunk = 0
        stamped_did = 0
        stamped_chunk = 0

        for rnode in sorted(self.graph.subjects(RDF.type, etd_ns.Relation), key=str):
            if not isinstance(rnode, URIRef):
                continue
            processed += 1

            evidence_nodes = [ev for ev in self.graph.objects(rnode, etd_ns.hasEvidence) if isinstance(ev, URIRef)]
            if not evidence_nodes:
                continue

            ev_dids: Set[str] = set()
            ev_chunks: Set[int] = set()

            for ev in evidence_nodes:
                for did in self.graph.objects(ev, etd_ns.discourse_id):
                    did_raw = str(did).strip()
                    if did_raw:
                        ev_dids.add(did_raw)
                for chunk in self.graph.objects(ev, etd_ns.chunk_index):
                    cnum = self._coerce_int(chunk)
                    if cnum is not None and cnum >= 0:
                        ev_chunks.add(cnum)

            for old in list(self.graph.objects(rnode, etd_ns.discourse_id)):
                self.graph.remove((rnode, etd_ns.discourse_id, old))
            for old in list(self.graph.objects(rnode, etd_ns.chunk_index)):
                self.graph.remove((rnode, etd_ns.chunk_index, old))

            if len(ev_dids) == 1:
                self._add_discourse_id_literal(rnode, sorted(ev_dids)[0], etd_ns)
                stamped_did += 1
            elif len(ev_dids) > 1:
                ambiguous_did += 1

            if len(ev_chunks) == 1:
                self.graph.add((rnode, etd_ns.chunk_index, Literal(next(iter(ev_chunks)), datatype=XSD.integer)))
                stamped_chunk += 1
            elif len(ev_chunks) > 1:
                ambiguous_chunk += 1

        return {
            "relations_processed": processed,
            "relations_stamped_did": stamped_did,
            "relations_stamped_chunk": stamped_chunk,
            "ambiguous_did": ambiguous_did,
            "ambiguous_chunk": ambiguous_chunk,
        }

    def _finalize_relation_provenance(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Final deterministic provenance completion for relation nodes.
        Runs after all relation generation/enforcement and before serialization.

        Guarantees:
        - every asserted etd:Relation has exactly one etd:chunk_index (xsd:integer)
        - relation-level datasetId/runId are present once when available
        """

        def _int_values(subject: URIRef, predicate: URIRef, *, non_negative_only: bool) -> List[int]:
            vals: Set[int] = set()
            for o in self.graph.objects(subject, predicate):
                n = self._coerce_int(o)
                if n is None:
                    continue
                if non_negative_only and n < 0:
                    continue
                vals.add(int(n))
            return sorted(vals)

        def _string_values(subject: URIRef, predicate: URIRef) -> List[str]:
            vals: Set[str] = set()
            for o in self.graph.objects(subject, predicate):
                s = str(o).strip()
                if s:
                    vals.add(s)
            return sorted(vals)

        # Build discourse -> known chunk index set from existing graph evidence/entity nodes.
        discourse_chunks: Dict[str, Set[int]] = {}
        for node in list(self.graph.subjects(RDF.type, etd_ns.Evidence)) + list(self.graph.subjects(RDF.type, etd_ns.Entity)):
            if not isinstance(node, URIRef):
                continue
            dids = _string_values(node, etd_ns.discourse_id)
            chunks = _int_values(node, etd_ns.chunk_index, non_negative_only=True)
            if len(dids) == 1 and chunks:
                discourse_chunks.setdefault(dids[0], set()).update(chunks)

        metrics: Dict[str, int] = {
            "relations_processed": 0,
            "filled_from_evidence": 0,
            "filled_fallback_unknown": 0,
            "dataset_id_set": 0,
            "run_id_set": 0,
            "source_set": 0,
            "source_doc_id_set": 0,
        }

        for rnode in sorted(self.graph.subjects(RDF.type, etd_ns.Relation), key=str):
            if not isinstance(rnode, URIRef):
                continue
            metrics["relations_processed"] += 1

            evidence_nodes = [ev for ev in self.graph.objects(rnode, etd_ns.hasEvidence) if isinstance(ev, URIRef)]
            subj = next((o for o in self.graph.objects(rnode, etd_ns.subject) if isinstance(o, URIRef)), None)
            obj = next((o for o in self.graph.objects(rnode, etd_ns.object) if isinstance(o, URIRef)), None)

            chosen_chunk: Optional[int] = None
            ev_chunk_values: Set[int] = set()
            ev_sources: Set[str] = set()
            ev_source_docs: Set[str] = set()
            ev_statuses: Set[str] = set()
            ev_methods: Set[str] = set()
            for ev in evidence_nodes:
                ev_chunk_values.update(_int_values(ev, etd_ns.chunk_index, non_negative_only=False))
                ev_sources.update(_string_values(ev, etd_ns.source))
                ev_source_docs.update(_string_values(ev, etd_ns.source_doc_id))
                ev_statuses.update(_string_values(ev, etd_ns.groundingStatus))
                ev_methods.update(_string_values(ev, etd_ns.groundingMethod))

            if len(ev_chunk_values) == 1:
                chosen_chunk = sorted(ev_chunk_values)[0]
                metrics["filled_from_evidence"] += 1
            else:
                chosen_chunk = -1
                metrics["filled_fallback_unknown"] += 1

            # Enforce exactly one chunk_index triple, typed xsd:integer.
            for old in list(self.graph.objects(rnode, etd_ns.chunk_index)):
                self.graph.remove((rnode, etd_ns.chunk_index, old))
            self.graph.add((rnode, etd_ns.chunk_index, Literal(int(chosen_chunk), datatype=XSD.integer)))
            for old in list(self.graph.objects(rnode, etd_ns.chunk_index_unknown)):
                self.graph.remove((rnode, etd_ns.chunk_index_unknown, old))
            self.graph.add((rnode, etd_ns.chunk_index_unknown, Literal(bool(int(chosen_chunk) < 0), datatype=XSD.boolean)))

            chosen_status: Optional[str] = None
            chosen_method: Optional[str] = None
            if len(ev_chunk_values) > 1:
                if ev_statuses and ev_statuses.issubset({"grounded", "chunk_grounded"}):
                    chosen_status = "grounded"
                    chosen_method = "multi_chunk_evidence"
                else:
                    chosen_status = "ungrounded"
                    chosen_method = "multi_chunk_evidence"
            elif chosen_chunk is not None and chosen_chunk >= 0:
                if len(ev_statuses) == 1:
                    chosen_status = sorted(ev_statuses)[0]
                if len(ev_methods) == 1:
                    chosen_method = sorted(ev_methods)[0]
            else:
                if len(ev_statuses) == 1:
                    chosen_status = sorted(ev_statuses)[0]
                if len(ev_methods) == 1:
                    chosen_method = sorted(ev_methods)[0]

            if chosen_status:
                for old in list(self.graph.objects(rnode, etd_ns.groundingStatus)):
                    self.graph.remove((rnode, etd_ns.groundingStatus, old))
                self.graph.add((rnode, etd_ns.groundingStatus, Literal(chosen_status, datatype=XSD.string)))
            if chosen_method:
                for old in list(self.graph.objects(rnode, etd_ns.groundingMethod)):
                    self.graph.remove((rnode, etd_ns.groundingMethod, old))
                self.graph.add((rnode, etd_ns.groundingMethod, Literal(chosen_method, datatype=XSD.string)))

            # Relation-level datasetId / runId completion.
            rel_ds = _string_values(rnode, etd_ns.datasetId)
            rel_run = _string_values(rnode, etd_ns.runId)
            ev_ds: Set[str] = set()
            ev_run: Set[str] = set()
            for ev in evidence_nodes:
                ev_ds.update(_string_values(ev, etd_ns.datasetId))
                ev_run.update(_string_values(ev, etd_ns.runId))

            chosen_ds = (rel_ds[0] if rel_ds else (sorted(ev_ds)[0] if ev_ds else str(self.dataset_id or "").strip()))
            chosen_run = (rel_run[0] if rel_run else (sorted(ev_run)[0] if ev_run else str(self.run_id or "").strip()))

            for old in list(self.graph.objects(rnode, etd_ns.datasetId)):
                self.graph.remove((rnode, etd_ns.datasetId, old))
            for old in list(self.graph.objects(rnode, etd_ns.runId)):
                self.graph.remove((rnode, etd_ns.runId, old))

            if chosen_ds:
                self.graph.add((rnode, etd_ns.datasetId, Literal(chosen_ds, datatype=XSD.string)))
                metrics["dataset_id_set"] += 1
            if chosen_run:
                self.graph.add((rnode, etd_ns.runId, Literal(chosen_run, datatype=XSD.string)))
                metrics["run_id_set"] += 1

            chosen_source = ""
            if len(ev_sources) == 1:
                chosen_source = sorted(ev_sources)[0]
            elif len(ev_sources) > 1:
                chosen_source = "mixed"
            if chosen_source:
                for old in list(self.graph.objects(rnode, etd_ns.source)):
                    self.graph.remove((rnode, etd_ns.source, old))
                self.graph.add((rnode, etd_ns.source, Literal(chosen_source, datatype=XSD.string)))
                metrics["source_set"] += 1

            chosen_source_doc = sorted(ev_source_docs)[0] if ev_source_docs else ""
            if chosen_source_doc:
                for old in list(self.graph.objects(rnode, etd_ns.source_doc_id)):
                    self.graph.remove((rnode, etd_ns.source_doc_id, old))
                self.graph.add((rnode, etd_ns.source_doc_id, Literal(chosen_source_doc, datatype=XSD.string)))
                metrics["source_doc_id_set"] += 1

        return metrics

    def _finalize_entity_provenance_completion(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Backfill missing entity-level provenance (source, extractedFrom,
        chunk_index, chunk_index_unknown, discourse_id, context_source,
        surface_form) for any etd:Entity node whose upstream construction
        path left these unset -- e.g. an entity minted only as an LLM
        relation's unresolved subject/object endpoint.

        Runs after all relations exist so it can derive values from the
        unique reified etd:Relation node(s) referencing the entity as
        etd:subject/etd:object. Only fills a field that is completely
        absent; never overwrites an existing value. Only backfills when
        every referencing relation agrees on a field's value (or there is
        exactly one referencing relation) -- if referencing relations
        disagree, that field is left as-is rather than guessed, since a
        genuinely ambiguous value must not be invented.

        surface_form specifically is backfilled from the referencing
        relation's etd:subject_surface_form / etd:object_surface_form
        (the genuine LLM-identified mention text -- see
        _finalize_relation_provenance and the subject_text/object_text ->
        subject_surface_form/object_surface_form reconciliation in
        _add_relation_metadata), never from the entity's own URI or label.
        """
        backfill_fields = (
            etd_ns.source,
            etd_ns.extractedFrom,
            etd_ns.chunk_index,
            etd_ns.chunk_index_unknown,
            etd_ns.discourse_id,
            etd_ns.context_source,
        )
        metrics = {
            "entities_checked": 0,
            "entities_incomplete": 0,
            "entities_backfilled_full": 0,
            "entities_backfilled_partial": 0,
            "entities_surface_form_backfilled": 0,
            "entities_left_incomplete_no_referencing_relation": 0,
            "entities_left_incomplete_ambiguous": 0,
        }

        def _single_agreed_value(rels: List[URIRef], pred: URIRef) -> Optional[Any]:
            values = set()
            for rel in rels:
                for v in self.graph.objects(rel, pred):
                    values.add(v)
            if len(values) == 1:
                return next(iter(values))
            return None

        entity_nodes = sorted(
            (n for n in self.graph.subjects(RDF.type, etd_ns.Entity) if isinstance(n, URIRef)),
            key=str,
        )
        for ent in entity_nodes:
            metrics["entities_checked"] += 1
            missing = [p for p in backfill_fields if not list(self.graph.objects(ent, p))]
            existing_surface_form = list(self.graph.objects(ent, etd_ns.surface_form))
            # Task (Defect 1): a value being present is NOT sufficient here --
            # _add_entities() always writes SOME surface_form (falling back to
            # ent.label as a placeholder when nothing genuine was provided),
            # so "missing" alone would never catch that case. Entities flagged
            # by _generic_surface_form_entities are eligible for replacement
            # by a genuine relation-sourced value below, same as a genuinely
            # absent one.
            missing_surface_form = (not existing_surface_form) or (ent in self._generic_surface_form_entities)
            if not missing and not missing_surface_form:
                continue
            metrics["entities_incomplete"] += 1

            rels_as_subject = [r for r in self.graph.subjects(etd_ns.subject, ent) if isinstance(r, URIRef)]
            rels_as_object = [r for r in self.graph.subjects(etd_ns.object, ent) if isinstance(r, URIRef)]
            all_rels = sorted(set(rels_as_subject) | set(rels_as_object), key=str)

            if not all_rels:
                metrics["entities_left_incomplete_no_referencing_relation"] += 1
                continue

            filled_any = False
            still_missing = False
            for pred in list(missing):
                val = _single_agreed_value(all_rels, pred)
                if val is not None:
                    self.graph.add((ent, pred, val))
                    filled_any = True
                else:
                    still_missing = True

            if missing_surface_form:
                surface_pred = etd_ns.subject_surface_form if rels_as_subject else etd_ns.object_surface_form
                surface_rels = rels_as_subject if rels_as_subject else rels_as_object
                surface_val = _single_agreed_value(surface_rels, surface_pred)
                if surface_val is not None and str(surface_val).strip():
                    # Replace, not just add -- when existing_surface_form is a
                    # generic ent.label placeholder, this is the genuine
                    # extracted-mention text taking its place, not a second
                    # competing value.
                    for old_val in existing_surface_form:
                        self.graph.remove((ent, etd_ns.surface_form, old_val))
                    self.graph.add((ent, etd_ns.surface_form, surface_val))
                    metrics["entities_surface_form_backfilled"] += 1
                    filled_any = True
                else:
                    still_missing = True

            if filled_any and not still_missing:
                metrics["entities_backfilled_full"] += 1
            elif filled_any:
                metrics["entities_backfilled_partial"] += 1
            else:
                metrics["entities_left_incomplete_ambiguous"] += 1

        logger.info("Entity provenance completion: %s", metrics)
        return metrics

    def _finalize_relation_source_traceability(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Resolve explicit source-entity links after the full graph exists.
        This avoids order-dependent misses while relations are still being added.
        """
        metrics = {
            "discourse_units_processed": 0,
            "relations_processed": 0,
            "discourse_units_linked": 0,
            "relations_linked": 0,
        }

        def _meta_from_node(node: URIRef) -> Dict[str, Any]:
            field = next((str(o).strip() for o in self.graph.objects(node, etd_ns.field) if str(o).strip()), "")
            source_doc_id = next(
                (str(o).strip() for o in self.graph.objects(node, etd_ns.source_doc_id) if str(o).strip()),
                "",
            )
            discourse_id = next(
                (str(o).strip() for o in self.graph.objects(node, etd_ns.discourse_id) if str(o).strip()),
                "",
            )
            return {
                "field": field,
                "source_doc_id": source_doc_id,
                "discourse_id": discourse_id,
            }

        for du in sorted(self.graph.subjects(RDF.type, etd_ns.DiscourseUnit), key=str):
            if not isinstance(du, URIRef):
                continue
            metrics["discourse_units_processed"] += 1
            du_meta = _meta_from_node(du)
            existing_sources = [o for o in self.graph.objects(du, etd_ns.hasSourceEntity) if isinstance(o, URIRef)]
            sources = self._resolve_source_entities_from_provenance(du_meta, etd_ns)
            if not sources:
                sources = existing_sources
            sources = self._normalize_source_entities(sources, du_meta, etd_ns)
            self._replace_source_entity_links(du, sources, etd_ns)
            if sources:
                metrics["discourse_units_linked"] += 1

        for rel in sorted(self.graph.subjects(RDF.type, etd_ns.Relation), key=str):
            if not isinstance(rel, URIRef):
                continue
            metrics["relations_processed"] += 1
            rel_meta = _meta_from_node(rel)
            existing_sources = [o for o in self.graph.objects(rel, etd_ns.hasSourceEntity) if isinstance(o, URIRef)]
            sources: List[URIRef] = []
            for du in self.graph.objects(rel, etd_ns.extractedFrom):
                if isinstance(du, URIRef):
                    sources.extend([o for o in self.graph.objects(du, etd_ns.hasSourceEntity) if isinstance(o, URIRef)])
            if not sources:
                sources = self._resolve_source_entities_from_provenance(rel_meta, etd_ns)
            if not sources:
                sources = existing_sources
            sources = self._normalize_source_entities(sources, rel_meta, etd_ns)
            self._replace_source_entity_links(rel, sources, etd_ns)
            if sources:
                metrics["relations_linked"] += 1

        return metrics

    def _assert_evidence_contract(self) -> None:
        """Raise if any Evidence node misses required audit-grade fields."""
        etd_ns = Namespace(self.namespaces["etd"])
        violations: List[str] = []
        for ev in self.graph.subjects(RDF.type, etd_ns.Evidence):
            if len(list(self.graph.objects(ev, etd_ns.evidenceText))) != 1:
                violations.append(f"{ev} evidenceText!=1")
            if len(list(self.graph.objects(ev, etd_ns.confidence))) != 1:
                violations.append(f"{ev} confidence!=1")
            if len(list(self.graph.objects(ev, etd_ns.chunk_index))) != 1:
                violations.append(f"{ev} chunk_index!=1")
        if violations:
            sample = "; ".join(violations[:10])
            raise RuntimeError(
                f"Evidence contract violation ({len(violations)} issues): {sample}"
            )


    def _add_relation_metadata(self, rnode: BNode, metadata: Dict[str, Any], etd_ns: Namespace) -> None:
        """
        Attach selected provenance metadata for relations:
        - discourse_id
        - surface_form
        - context_source
        - chunk_index, chunk_total
        - start_char, end_char
        """
        if not metadata:
            return

        # Backward compatibility: accept doc_id if discourse_id missing
        if "discourse_id" not in metadata and "doc_id" in metadata:
            metadata["discourse_id"] = metadata.get("doc_id")

        # discourse_id
        self._add_discourse_id_literal(rnode, metadata.get("discourse_id"), etd_ns)
        origin_node = self._ensure_discourse_unit_node(metadata, etd_ns)
        if origin_node is not None:
            self.graph.add((rnode, etd_ns.extractedFrom, origin_node))
        relation_source_entities: List[URIRef] = []
        if origin_node is not None:
            relation_source_entities.extend(
                [o for o in self.graph.objects(origin_node, etd_ns.hasSourceEntity) if isinstance(o, URIRef)]
            )
        if not relation_source_entities:
            relation_source_entities.extend(self._resolve_source_entities_from_provenance(metadata, etd_ns))
        relation_source_entities = self._normalize_source_entities(relation_source_entities, metadata, etd_ns)
        self._replace_source_entity_links(rnode, relation_source_entities, etd_ns)

        chunk_num = self._coerce_int(metadata.get("chunk_index"))
        if metadata.get("groundingStatus") in (None, ""):
            if chunk_num is not None and int(chunk_num) >= 0:
                metadata["groundingStatus"] = "chunk_grounded"
            else:
                metadata["groundingStatus"] = "ungrounded"
        if metadata.get("groundingMethod") in (None, ""):
            if chunk_num is not None and int(chunk_num) >= 0:
                metadata["groundingMethod"] = "single_chunk_discourse"
            else:
                src = str(metadata.get("source") or metadata.get("extractor") or "").strip()
                metadata["groundingMethod"] = src if src in {"structural", "reasoner", "vote", "lightweight"} else "none"
        if metadata.get("chunk_index_unknown") is None and chunk_num is not None:
            metadata["chunk_index_unknown"] = bool(int(chunk_num) < 0)

        # Root-cause fix: extractors/llm_relation_extractor.py captures the
        # genuine LLM-identified mention text as metadata["subject_text"]/
        # ["object_text"], but this field-emission loop (and downstream
        # entity-provenance backfill) reads "subject_surface_form"/
        # "object_surface_form" -- a different key name. Without this
        # reconciliation, genuine extracted mention text that IS already
        # captured never reaches the RDF graph. Only fills the gap;
        # never overwrites a value already set by another code path.
        if not metadata.get("subject_surface_form") and metadata.get("subject_text"):
            metadata["subject_surface_form"] = metadata["subject_text"]
        if not metadata.get("object_surface_form") and metadata.get("object_text"):
            metadata["object_surface_form"] = metadata["object_text"]

        # surface_form, context_source, field, source_doc_id, source
        for key in (
            "surface_form",
            "context_source",
            "groundingStatus",
            "groundingMethod",
            "field",
            "source_doc_id",
            "source",
            "relation_kind",
            "section",
            "section_category",
            "semantic_validation_status",
            "semantic_validation_reason",
            "semantic_validation_code",
            "relation_origin",
            "predicate_policy_source",
            "predicate_surface_phrase",
            "canonical_predicate",
            "predicate_mapping_status",
            "direction_normalized",
            "locality_scope",
            "discourse_mode",
            "epistemic_status",
            "subject_surface_form",
            "object_surface_form",
            "subject_canonical_entity",
            "object_canonical_entity",
            "subject_family",
            "object_family",
            "object_type",
            "entity_role_check",
            "domain_range_check",
            "entailment_check",
            "modality_check",
            "modality_tags",
        ):
            if metadata.get(key) is not None:
                val = str(metadata[key]).strip()
                if val:
                    self._emit_registered_etd_literal(rnode, key, val, etd_ns, datatype=XSD.string)


        # Integer-like fields (production-safe: coerce + preserve raw)
        for key in ("chunk_index", "chunk_total", "start_char", "end_char"):
            if metadata.get(key) is not None:
                raw_val = metadata.get(key)
                num = self._coerce_int(raw_val)

                if num is not None:
                    self._emit_registered_etd_literal(rnode, key, num, etd_ns, datatype=XSD.integer)
        if metadata.get("chunk_index_unknown") is not None:
            bval = self._coerce_bool(metadata.get("chunk_index_unknown"))
            if bval is not None:
                self._emit_registered_etd_literal(rnode, "chunk_index_unknown", bool(bval), etd_ns, datatype=XSD.boolean)
        target_is_literal = self._coerce_bool(metadata.get("target_is_literal"))
        if target_is_literal is not None:
            self._emit_registered_etd_literal(rnode, "target_is_literal", bool(target_is_literal), etd_ns, datatype=XSD.boolean)
        if metadata.get("target_datatype") is not None and str(metadata.get("target_datatype")).strip():
            self._emit_registered_etd_literal(rnode, "target_datatype", str(metadata.get("target_datatype")).strip(), etd_ns, datatype=XSD.string)
        for key, local_name in (
            ("qualifiers", "qualifiers_json"),
            ("temporal_information", "temporal_information_json"),
            ("quantitative_information", "quantitative_information_json"),
        ):
            value = metadata.get(key)
            if not value:
                continue
            try:
                encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                encoded = str(value)
            if encoded.strip():
                self._emit_registered_etd_literal(rnode, local_name, encoded, etd_ns, datatype=XSD.string)
        coverage_recovery = self._coerce_bool(metadata.get("coverage_recovery"))
        if coverage_recovery is not None:
            self._emit_registered_etd_literal(rnode, "coverage_recovery", bool(coverage_recovery), etd_ns, datatype=XSD.boolean)
        negated = self._coerce_bool(metadata.get("negated"))
        if negated is not None:
            self._emit_registered_etd_literal(rnode, "negated", bool(negated), etd_ns, datatype=XSD.boolean)

        self._emit_registered_etd_literal(rnode, "datasetId", self.dataset_id, etd_ns, datatype=XSD.string)
        self._emit_registered_etd_literal(rnode, "runId", self.run_id, etd_ns, datatype=XSD.string)

    def _add_inferred_provenance(self, rnode: Union[URIRef, BNode], rel: Relation, etd_ns: Namespace) -> None:
        """
        Attach audit-grade provenance to inferred relations:
        - Prefer prov:wasDerivedFrom when supporting relation URIs exist.
        - Fallback to etd:inferenceRule.
        """
        meta = getattr(rel, "metadata", None) or {}

        derived_candidates: List[Any] = []
        for key in (
            "supporting_relation_uris",
            "derived_from",
            "was_derived_from",
            "source_relation_uris",
            "support_relations",
        ):
            val = meta.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                derived_candidates.extend(val)
            else:
                derived_candidates.append(val)

        added_derivation = False
        for item in derived_candidates:
            if isinstance(item, dict):
                item = item.get("uri") or item.get("relation_uri") or item.get("id")
            if item is None:
                continue
            raw = str(item).strip()
            if not raw:
                continue
            if raw.startswith("http://") or raw.startswith("https://"):
                ref = _safe_uriref(raw, self.namespaces)
            else:
                ref = _safe_uriref(f"{self.namespaces['etd']}{raw.lstrip('#/')}", self.namespaces)
            self.graph.add((rnode, PROV.wasDerivedFrom, ref))
            added_derivation = True

        if not added_derivation:
            rule = (
                meta.get("inferenceRule")
                or meta.get("inference_rule")
                or meta.get("inference_type")
                or meta.get("rule")
                or "unspecified_inference_rule"
            )
            self.graph.add((rnode, etd_ns.inferenceRule, Literal(str(rule), datatype=XSD.string)))

    # ------------------------------------------------------------------
    # Dataset-level metadata
    # ------------------------------------------------------------------
    def _add_dataset_metadata(self) -> None:
        """
        Add dataset/run/export metadata as RDF triples on a single export node.
        """
        etd_ns = Namespace(self.namespaces["etd"])
        onto_uri = Namespace(self.namespaces["etd"])["ontology"]
        export_node = self.export_node

        meta = self.annotations.get("metadata", {}) or {}

        # Export node for AI traceability/queryability.
        self.graph.add((export_node, RDF.type, etd_ns.Export))
        self.graph.add((export_node, etd_ns.datasetId, Literal(self.dataset_id, datatype=XSD.string)))
        self.graph.add((export_node, etd_ns.runId, Literal(self.run_id, datatype=XSD.string)))
        self.graph.add((export_node, PROV.generatedAtTime, Literal(self.generated_at_time, datatype=XSD.dateTime)))
        self.graph.add((export_node, etd_ns.pipelineVersion, Literal(self.pipeline_version, datatype=XSD.string)))
        self.graph.add((export_node, etd_ns.gitCommit, Literal(self.git_commit, datatype=XSD.string)))
        self.graph.add((export_node, etd_ns.shaclProfileVersion, Literal(self.shacl_profile_version, datatype=XSD.string)))
        self.graph.add((export_node, etd_ns.exportMode, Literal(self.export_mode, datatype=XSD.string)))
        self.graph.add((export_node, PROV.wasDerivedFrom, onto_uri))

        # Avoid duplicate ontology titles; schema layer already sets a default title.
        title = meta.get("title")
        if title:
            self.graph.add((onto_uri, DCTERMS.title, Literal(title, lang="en")))

        desc = meta.get("description")
        if desc:
            self.graph.add(
                (onto_uri, DCTERMS.description, Literal(desc, lang="en"))
            )

        creator = meta.get("creator")
        if creator:
            self.graph.add(
                (onto_uri, DCTERMS.creator, Literal(creator, datatype=XSD.string))
            )

        version = meta.get("version")
        if version:
            self.graph.add(
                (onto_uri, etd_ns.pipelineVersion, Literal(version, datatype=XSD.string))
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _to_rdf_node(self, value: Any) -> Union[Literal, URIRef]:
        """
        Convert Python value → RDF node.

        - Numbers / booleans → typed literals
        - Strings that look like IRIs or CURIEs → safe URIRefs
        - Everything else → xsd:string literal
        """
        if isinstance(value, URIRef):
            return value

        if isinstance(value, bool):
            return Literal(value, datatype=XSD.boolean)

        if isinstance(value, int):
            return Literal(value, datatype=XSD.integer)

        if isinstance(value, float):
            return Literal(value, datatype=XSD.float)

        if isinstance(value, str):
            s = value.strip()

            # HTTP(S) IRI
            if s.startswith("http://") or s.startswith("https://"):
                return _safe_uriref(s, self.namespaces)

            # CURIE with known prefix
            if ":" in s and " " not in s:
                prefix, local = s.split(":", 1)
                if prefix in self.namespaces:
                    iri = self.namespaces[prefix] + local
                    return _safe_uriref(iri, self.namespaces)

            return Literal(s, datatype=XSD.string)

        # Fallback: string literal
        return Literal(str(value), datatype=XSD.string)

    def _declare_local_relation_predicates(self, etd_ns: Namespace) -> Dict[str, Any]:
        """E: every local (unmapped) relation predicate actually used in the
        graph gets a first-class RDF/OWL declaration.

          * property kind from ACTUAL object usage across all its reified
            relations: all-literal -> owl:DatatypeProperty; any resource
            object -> owl:ObjectProperty (matches this project's existing
            pattern where a relation object-property may still carry a
            literal, e.g. hasDescription);
          * a human-readable rdfs:label (de-camelCased) + a generic
            rdfs:comment + the governance lifecycle metadata the quality
            gate requires -- via ontology.add_schema.apply_property_lifecycle,
            the SAME single lifecycle policy used for statically declared
            ETD properties (root cause: this pass runs after add_schema's
            static stamping loop, so those predicates were previously
            missed);
          * NO domain/range axioms are invented.

        A predicate whose case/separator-insensitive local name collides
        with an already-declared ETD class local name (or a differently
        cased ETD property local name) -- per the same normalisation the
        quality gate enforces -- is NOT declared: no colliding ETD local
        name is created and no dataset-specific rename is invented; the
        predicate is left unresolved/flagged (still local_unmapped,
        undeclared, which the gate treats as non-blocking).

        Predicates already declared anywhere in the schema, structural
        backbone predicates, and serializer-metadata predicates are left
        untouched.
        """
        metrics: Dict[str, Any] = {
            "declared": [], "object_property": 0, "datatype_property": 0,
            "mixed_use": [], "already_declared_skipped": 0,
            "lifecycle_stamped": 0, "collision_skipped": [],
        }
        declared_types = (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property)

        # Case/separator-insensitive index of every ETD local name already
        # present as a class or property -> guards against emitting a
        # class/property (or property/property) local-name collision the
        # quality gate would flag. Same normalisation as
        # kg_quality_gate._etd_local_name_collisions._norm.
        norm_index: Dict[str, str] = {}
        for _t in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property):
            for _term in self.graph.subjects(RDF.type, _t):
                if isinstance(_term, URIRef) and str(_term).startswith(str(etd_ns)):
                    _tl = str(_term)[len(str(etd_ns)):]
                    norm_index.setdefault(normalized_etd_local_name(_tl), _tl)

        # collect distinct predicate URIs used as a reified etd:predicate value
        used: Dict[URIRef, List[URIRef]] = {}
        for rel in self.graph.subjects(RDF.type, etd_ns.Relation):
            for p in self.graph.objects(rel, etd_ns.predicate):
                if isinstance(p, URIRef):
                    used.setdefault(p, []).append(rel)

        etd_prefix = str(etd_ns)
        for pred_uri, rel_nodes in sorted(used.items(), key=lambda kv: str(kv[0])):
            if not str(pred_uri).startswith(etd_prefix):
                continue
            local = str(pred_uri)[len(etd_prefix):]
            if any((pred_uri, RDF.type, t) in self.graph for t in declared_types):
                metrics["already_declared_skipped"] += 1
                continue

            nlocal = normalized_etd_local_name(local)
            clash = norm_index.get(nlocal)
            if clash is not None and clash != local:
                # Would create an ETD local-name collision -- leave the
                # predicate unresolved/flagged rather than declare it or
                # rename it.
                metrics["collision_skipped"].append(local)
                continue

            # object shape across every reified usage of this predicate
            saw_literal = False
            saw_resource = False
            for rel in rel_nodes:
                for o in self.graph.objects(rel, etd_ns.object):
                    if isinstance(o, Literal):
                        saw_literal = True
                    elif isinstance(o, URIRef):
                        saw_resource = True
            if saw_resource:
                kind = OWL.ObjectProperty
                metrics["object_property"] += 1
            elif saw_literal:
                kind = OWL.DatatypeProperty
                metrics["datatype_property"] += 1
            else:
                # no object at all observed -- default to the more permissive
                # object-property kind rather than guess a datatype range.
                kind = OWL.ObjectProperty
                metrics["object_property"] += 1
            if saw_literal and saw_resource:
                metrics["mixed_use"].append(local)

            self.graph.add((pred_uri, RDF.type, kind))
            label = predicate_label_from_local_name(local) or local
            self.graph.add((pred_uri, RDFS.label, Literal(label, lang="en")))
            self.graph.add((
                pred_uri, RDFS.comment,
                Literal(
                    "Local relation predicate preserved from extraction with no external "
                    "ontology mapping; property kind inferred from observed object usage.",
                    lang="en",
                ),
            ))
            # Same governance lifecycle metadata as a statically declared
            # ETD property -- add_schema's stamping loop already ran (this
            # pass is later in serialization), so stamp it here via the
            # shared policy.
            apply_property_lifecycle(self.graph, etd_ns, pred_uri, local)
            metrics["lifecycle_stamped"] += 1
            norm_index[nlocal] = local
            metrics["declared"].append(local)

        metrics["declared"] = sorted(metrics["declared"])
        metrics["mixed_use"] = sorted(metrics["mixed_use"])
        metrics["collision_skipped"] = sorted(metrics["collision_skipped"])
        return metrics

    def _sanitize(self, text: str) -> str:
        """Convert arbitrary string into a safe NCName-like local part."""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return "Resource"
        text = re.sub(r"[^\w]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_") or "Resource"

    # ------------------------------------------------------------------
    # Public serialization methods
    # ------------------------------------------------------------------
    def serialize_to_turtle(self, filepath: Path) -> None:
        """Turtle serialization (.ttl)."""
        try:
            self._assert_evidence_contract()
            self.graph.serialize(destination=str(filepath), format="turtle")
            print(f"  ✓ {filepath.name}")
        except Exception as e:
            print(f"  ✗ Turtle serialization failed: {e}")

    def serialize_to_rdf(self, filepath: Path) -> None:
        """RDF/XML serialization (.rdf)."""
        try:
            self._assert_evidence_contract()
            self.graph.serialize(destination=str(filepath), format="xml")
            print(f"  ✓ {filepath.name}")
        except Exception as e:
            print(f"  ✗ RDF/XML serialization failed: {e}")

    def serialize_to_jsonld(self, filepath: Path) -> None:
        """JSON-LD serialization (.jsonld)."""
        try:
            self._assert_evidence_contract()
            self.graph.serialize(destination=str(filepath), format="json-ld")
            print(f"  ✓ {filepath.name}")
        except Exception as e:
            print(f"  ✗ JSON-LD serialization failed: {e}")

    def serialize_to_owl(self, filepath: Path) -> None:
        """OWL/XML serialization (.owl) – for tools expecting ontology-like serialisation."""
        try:
            self._assert_evidence_contract()
            self.graph.serialize(destination=str(filepath), format="xml")
            print(f"  ✓ {filepath.name}")
        except Exception as e:
            print(f"  ✗ OWL serialization failed: {e}")


# ----------------------------------------------------------------------
# Helper: safe URIRef using namespaces + make_uri_safe
# ----------------------------------------------------------------------
def _safe_uriref(iri_or_curie: str, namespaces: Dict[str, str]) -> URIRef:
    """
    Expand CURIEs and produce a safe URIRef, BUT with special handling
    for ETD-local IRIs so entity URIs do not get percent-encoded.
    """
    if not isinstance(iri_or_curie, str):
        iri_or_curie = str(iri_or_curie)

    iri_or_curie = iri_or_curie.strip()

    # Try CURIE expansion first
    if ":" in iri_or_curie and not iri_or_curie.startswith("http"):
        prefix, local = iri_or_curie.split(":", 1)
        base = namespaces.get(prefix)
        if base:
            iri = base + local
        else:
            iri = iri_or_curie
    else:
        iri = iri_or_curie

    # ------------------------------------------------------------
    # 🔥 FIX: Normalise ETD local names BEFORE encoding
    # ------------------------------------------------------------
    etd_base = namespaces.get("etd")
    if etd_base and isinstance(iri, str) and iri.startswith(etd_base):
        tail = iri[len(etd_base):]

        # If tail contains spaces → canonicalize instead of percent-encoding
        if any(ch.isspace() for ch in tail):
            local = canonicalize_local_name(tail)
            return URIRef(etd_base + local)

    # For all other IRIs (Wikidata, DPV, VAIR, URLs), use normal encoding
    return make_uri_safe(iri)



