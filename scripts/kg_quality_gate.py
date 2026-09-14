#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from validation.shacl_validator import SHACLValidator
from scripts.audit_harness import (
    _resolve_etd_namespace,
    _entity_nodes,
    _isolated_entities,
    _vote_typing_issues,
    _duplicate_reified_relation_spo,
    _same_chunk_duplicate_labels_count,
    _cross_chunk_duplicate_labels,
)
from ontology.add_schema import CLASS_CANONICAL_ALIASES
from ontology.property_registry import ETD_PROPERTY_REGISTRY
from utils.semantic_relation_policy import PREDICATE_REGISTRY


def _sum_vote_issues(v: Dict[str, int]) -> int:
    return sum(int(x) for x in v.values())


def _count_query(graph: Graph, query: str) -> int:
    rows = list(graph.query(query))
    if not rows:
        return 0
    return int(rows[0][0])


def _relation_evidence_provenance_mismatches(graph: Graph, etd: Namespace) -> Dict[str, int]:
    mismatched_did = 0
    mismatched_chunk = 0

    for rel in graph.subjects(RDF.type, etd.Relation):
        rel_dids = {str(v).strip() for v in graph.objects(rel, etd.discourse_id) if str(v).strip()}
        rel_chunks = set()
        for v in graph.objects(rel, etd.chunk_index):
            try:
                n = int(str(v))
            except Exception:
                continue
            if n >= 0:
                rel_chunks.add(n)

        for ev in graph.objects(rel, etd.hasEvidence):
            ev_dids = {str(v).strip() for v in graph.objects(ev, etd.discourse_id) if str(v).strip()}
            ev_chunks = set()
            for v in graph.objects(ev, etd.chunk_index):
                try:
                    n = int(str(v))
                except Exception:
                    continue
                if n >= 0:
                    ev_chunks.add(n)

            if rel_dids and ev_dids and rel_dids.isdisjoint(ev_dids):
                mismatched_did += 1
            if rel_chunks and ev_chunks and rel_chunks.isdisjoint(ev_chunks):
                mismatched_chunk += 1

    return {
        "relation_evidence_mismatched_did": mismatched_did,
        "relation_evidence_mismatched_chunk": mismatched_chunk,
    }


def _same_origin_duplicate_reified_relation_spo(graph: Graph, etd_ns: str) -> Dict[str, int]:
    etd = Namespace(etd_ns)
    groups: Dict[Tuple[str, str, str, str], List[str]] = {}
    for rel in graph.subjects(RDF.type, etd.Relation):
        if not isinstance(rel, URIRef):
            continue
        subjects = [str(o) for o in graph.objects(rel, etd.subject) if isinstance(o, URIRef)]
        predicates = [str(o) for o in graph.objects(rel, etd.predicate) if isinstance(o, URIRef)]
        objects = [str(o) for o in graph.objects(rel, etd.object) if isinstance(o, URIRef)]
        origins = [str(o) for o in graph.objects(rel, etd.extractedFrom) if isinstance(o, URIRef)]
        if not origins:
            dids = [str(o).strip() for o in graph.objects(rel, etd.discourse_id) if str(o).strip()]
            fields = [str(o).strip() for o in graph.objects(rel, etd.field) if str(o).strip()]
            if dids:
                origins = [f"{(fields[0] if fields else '')}|{dids[0]}"]
        for s in subjects:
            for p in predicates:
                for o in objects:
                    for origin in origins:
                        groups.setdefault((s, p, o, origin), []).append(str(rel))
    dup_groups = 0
    dup_rels = 0
    for rels in groups.values():
        uniq = sorted(set(rels))
        if len(uniq) > 1:
            dup_groups += 1
            dup_rels += len(uniq)
    return {
        "same_origin_duplicate_relation_spo_groups_count": dup_groups,
        "same_origin_duplicate_relation_spo_total_relations_involved": dup_rels,
    }


def _literal_strings(graph: Graph, subject: URIRef, predicate: URIRef) -> List[str]:
    vals: Set[str] = set()
    for obj in graph.objects(subject, predicate):
        sval = str(obj).strip()
        if sval:
            vals.add(sval)
    return sorted(vals)


def _int_literals(graph: Graph, subject: URIRef, predicate: URIRef) -> List[int]:
    vals: Set[int] = set()
    for obj in graph.objects(subject, predicate):
        try:
            vals.add(int(str(obj)))
        except Exception:
            continue
    return sorted(vals)


def _provenance_completeness_metrics(graph: Graph, etd: Namespace) -> Dict[str, object]:
    allowed_status = {"grounded", "chunk_grounded", "ungrounded"}
    allowed_methods = {
        "span_map",
        "single_entity_span",
        "single_chunk_discourse",
        "structural",
        "reasoner",
        "vote",
        "lightweight",
        "none",
    }

    def _count_missing(nodes: List[URIRef], predicate: URIRef) -> int:
        return sum(1 for n in nodes if not any(True for _ in graph.objects(n, predicate)))

    evidence_nodes = sorted([n for n in graph.subjects(RDF.type, etd.Evidence) if isinstance(n, URIRef)], key=str)
    relation_nodes = sorted([n for n in graph.subjects(RDF.type, etd.Relation) if isinstance(n, URIRef)], key=str)
    entity_nodes = sorted([n for n in graph.subjects(RDF.type, etd.Entity) if isinstance(n, URIRef)], key=str)
    discourse_nodes = sorted([n for n in graph.subjects(RDF.type, etd.DiscourseUnit) if isinstance(n, URIRef)], key=str)

    grounded_placeholder = 0
    evidence_missing_grounding = 0
    evidence_inconsistent_chunk_unknown = 0
    evidence_invalid_span = 0
    evidence_orphaned = 0
    evidence_unexpected_status = 0
    evidence_unexpected_method = 0
    evidence_unknown_by_source: Dict[str, int] = {}
    evidence_unknown_by_field: Dict[str, int] = {}

    evidence_inbound: Dict[URIRef, int] = {}
    for rel in relation_nodes:
        for ev in graph.objects(rel, etd.hasEvidence):
            if isinstance(ev, URIRef):
                evidence_inbound[ev] = int(evidence_inbound.get(ev, 0)) + 1

    for ev in evidence_nodes:
        chunks = _int_literals(graph, ev, etd.chunk_index)
        chunk = chunks[0] if chunks else None
        flags = [str(v).strip().lower() in {"true", "1"} for v in graph.objects(ev, etd.chunk_index_unknown)]
        flag = flags[0] if flags else None
        status = _literal_strings(graph, ev, etd.groundingStatus)
        method = _literal_strings(graph, ev, etd.groundingMethod)
        char_starts = _int_literals(graph, ev, etd.char_start) or _int_literals(graph, ev, etd.start_char)
        char_ends = _int_literals(graph, ev, etd.char_end) or _int_literals(graph, ev, etd.end_char)
        ev_text = " ".join(_literal_strings(graph, ev, etd.evidenceText))

        if chunk is None or flag is None or ((chunk < 0) != bool(flag)):
            evidence_inconsistent_chunk_unknown += 1
        if not status or not method:
            evidence_missing_grounding += 1
        if status and any(s not in allowed_status for s in status):
            evidence_unexpected_status += 1
        if method and any(m not in allowed_methods for m in method):
            evidence_unexpected_method += 1
        if any(True for _ in graph.objects(ev, etd.extractedFrom)) is False:
            evidence_orphaned += 1
        if any(s in {"grounded", "chunk_grounded"} for s in status) and "Evidence unavailable" in ev_text:
            grounded_placeholder += 1

        if status:
            primary_status = status[0]
            primary_method = method[0] if method else ""
            has_valid_span = bool(char_starts and char_ends and char_ends[0] > char_starts[0] >= 0)
            if primary_status == "grounded" and primary_method == "span_map" and not has_valid_span:
                evidence_invalid_span += 1
            if primary_status == "grounded" and primary_method == "single_chunk_discourse" and not has_valid_span:
                evidence_invalid_span += 1

        if chunk is not None and chunk < 0:
            src = (_literal_strings(graph, ev, etd.source) or ["<missing>"])[0]
            fld = (_literal_strings(graph, ev, etd.field) or ["<missing>"])[0]
            evidence_unknown_by_source[src] = int(evidence_unknown_by_source.get(src, 0)) + 1
            evidence_unknown_by_field[fld] = int(evidence_unknown_by_field.get(fld, 0)) + 1

    relation_missing_source = 0
    relation_missing_has_evidence = 0
    relation_missing_field = 0
    relation_missing_inferred_flag = 0
    relation_missing_grounding = 0
    relation_has_source_entity = 0
    relation_traces_to_answer = 0
    relation_traces_to_question = 0
    relation_traces_to_theme = 0
    relation_traces_to_expert = 0
    relation_traces_to_vote = 0
    theme_description_total = 0
    theme_description_with_source = 0
    theme_description_missing_source = 0
    theme_description_wrong_theme = 0
    theme_description_wrong_examples: List[str] = []
    relation_inconsistent_chunk_unknown = 0
    relation_invalid_grounding_semantics = 0
    relation_bad_source_sync = 0
    for rel in relation_nodes:
        if not any(True for _ in graph.objects(rel, etd.hasEvidence)):
            relation_missing_has_evidence += 1
        if not any(True for _ in graph.objects(rel, etd.field)):
            relation_missing_field += 1
        if not any(True for _ in graph.objects(rel, etd.isInferred)):
            relation_missing_inferred_flag += 1
        if not any(True for _ in graph.objects(rel, etd.source)):
            relation_missing_source += 1
        if not any(True for _ in graph.objects(rel, etd.groundingStatus)) or not any(True for _ in graph.objects(rel, etd.groundingMethod)):
            relation_missing_grounding += 1
        source_entities = [o for o in graph.objects(rel, etd.hasSourceEntity) if isinstance(o, URIRef)]
        if source_entities:
            relation_has_source_entity += 1
            source_types: Set[str] = set()
            for src in source_entities:
                source_types.update(
                    str(t).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
                    for t in graph.objects(src, RDF.type)
                    if isinstance(t, URIRef)
                )
            if "Answer" in source_types:
                relation_traces_to_answer += 1
            if "Question" in source_types:
                relation_traces_to_question += 1
            if "Theme" in source_types:
                relation_traces_to_theme += 1
            if "Expert" in source_types or "Person" in source_types:
                relation_traces_to_expert += 1
            if "Vote" in source_types:
                relation_traces_to_vote += 1
        fields_for_rel = _literal_strings(graph, rel, etd.field)
        source_doc_ids_for_rel = _literal_strings(graph, rel, etd.source_doc_id)
        if "theme.description" in fields_for_rel:
            theme_description_total += 1
            if source_entities:
                theme_description_with_source += 1
            else:
                theme_description_missing_source += 1
            expected_theme_id = source_doc_ids_for_rel[0] if source_doc_ids_for_rel else ""
            if source_entities and expected_theme_id:
                theme_sources = []
                for src in source_entities:
                    src_types = {
                        str(t).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
                        for t in graph.objects(src, RDF.type)
                        if isinstance(t, URIRef)
                    }
                    if "Theme" not in src_types:
                        continue
                    theme_ids = _literal_strings(graph, src, etd.themeId)
                    if not theme_ids:
                        theme_ids = _literal_strings(graph, src, etd.source_doc_id)
                    if not theme_ids:
                        theme_ids = _literal_strings(graph, src, etd.discourse_id)
                    theme_sources.extend(theme_ids)
                if expected_theme_id not in theme_sources:
                    theme_description_wrong_theme += 1
                    if len(theme_description_wrong_examples) < 10:
                        theme_description_wrong_examples.append(
                            f"{rel} expected_theme_id={expected_theme_id} resolved_theme_ids={sorted(set(theme_sources))}"
                        )
        chunks = _int_literals(graph, rel, etd.chunk_index)
        chunk = chunks[0] if chunks else None
        flags = [str(v).strip().lower() in {"true", "1"} for v in graph.objects(rel, etd.chunk_index_unknown)]
        flag = flags[0] if flags else None
        statuses = _literal_strings(graph, rel, etd.groundingStatus)
        methods = _literal_strings(graph, rel, etd.groundingMethod)
        if chunk is None or flag is None or ((chunk < 0) != bool(flag)):
            relation_inconsistent_chunk_unknown += 1
        if chunk is not None and chunk < 0 and any(m in {"span_map", "single_chunk_discourse"} for m in methods):
            relation_invalid_grounding_semantics += 1
        rel_sources = set(_literal_strings(graph, rel, etd.source))
        ev_sources: Set[str] = set()
        for ev in graph.objects(rel, etd.hasEvidence):
            if isinstance(ev, URIRef):
                ev_sources.update(_literal_strings(graph, ev, etd.source))
        if rel_sources and ev_sources:
            if len(rel_sources) != 1 or not rel_sources.issubset(ev_sources | {"mixed"}):
                relation_bad_source_sync += 1

    entity_missing_source = _count_missing(entity_nodes, etd.source)
    entity_missing_extracted_from = _count_missing(entity_nodes, etd.extractedFrom)
    entity_missing_surface_form = _count_missing(entity_nodes, etd.surface_form)
    entity_missing_discourse_id = _count_missing(entity_nodes, etd.discourse_id)
    entity_missing_context_source = _count_missing(entity_nodes, etd.context_source)
    entity_missing_chunk_index = _count_missing(entity_nodes, etd.chunk_index)
    entity_inconsistent_chunk_unknown = 0
    for ent in entity_nodes:
        chunks = _int_literals(graph, ent, etd.chunk_index)
        chunk = chunks[0] if chunks else None
        flags = [str(v).strip().lower() in {"true", "1"} for v in graph.objects(ent, etd.chunk_index_unknown)]
        flag = flags[0] if flags else None
        if chunk is None or flag is None or ((chunk < 0) != bool(flag)):
            entity_inconsistent_chunk_unknown += 1

    discourse_missing_source_doc_id = _count_missing(discourse_nodes, etd.source_doc_id)
    discourse_missing_chunk_index = _count_missing(discourse_nodes, etd.chunk_index)
    discourse_missing_chunk_total = _count_missing(discourse_nodes, etd.chunk_total)
    discourse_missing_field = _count_missing(discourse_nodes, etd.field)
    discourse_missing_extracted_from = _count_missing(discourse_nodes, etd.extractedFrom)
    discourse_missing_chunk_id = _count_missing(discourse_nodes, etd.chunk_id)
    discourse_missing_discourse_id = _count_missing(discourse_nodes, etd.discourse_id)
    discourse_inconsistent_chunk_unknown = 0
    for du in discourse_nodes:
        chunks = _int_literals(graph, du, etd.chunk_index)
        chunk = chunks[0] if chunks else None
        flags = [str(v).strip().lower() in {"true", "1"} for v in graph.objects(du, etd.chunk_index_unknown)]
        flag = flags[0] if flags else None
        if chunk is None or flag is None or ((chunk < 0) != bool(flag)):
            discourse_inconsistent_chunk_unknown += 1

    return {
        "total_relation_nodes": len(relation_nodes),
        "total_entity_nodes": len(entity_nodes),
        "total_discourse_unit_nodes": len(discourse_nodes),
        "evidence_missing_grounding_count": evidence_missing_grounding,
        "evidence_inconsistent_chunk_unknown_count": evidence_inconsistent_chunk_unknown,
        "grounded_placeholder_evidence_count": grounded_placeholder,
        "evidence_invalid_span_count": evidence_invalid_span,
        "orphan_evidence_count": sum(1 for ev in evidence_nodes if evidence_inbound.get(ev, 0) == 0),
        "evidence_unexpected_status_count": evidence_unexpected_status,
        "evidence_unexpected_method_count": evidence_unexpected_method,
        "evidence_unknown_chunk_by_source": dict(sorted(evidence_unknown_by_source.items())),
        "evidence_unknown_chunk_by_field": dict(sorted(evidence_unknown_by_field.items())),
        "relation_missing_source_count": relation_missing_source,
        "relation_missing_has_evidence_count": relation_missing_has_evidence,
        "relation_missing_field_count": relation_missing_field,
        "relation_missing_inferred_flag_count": relation_missing_inferred_flag,
        "relation_missing_grounding_count": relation_missing_grounding,
        "relation_has_source_entity_count": relation_has_source_entity,
        "relation_missing_source_entity_count": max(0, len(relation_nodes) - relation_has_source_entity),
        "relation_traces_to_answer_count": relation_traces_to_answer,
        "relation_traces_to_question_count": relation_traces_to_question,
        "relation_traces_to_theme_count": relation_traces_to_theme,
        "relation_traces_to_expert_count": relation_traces_to_expert,
        "relation_traces_to_vote_count": relation_traces_to_vote,
        "theme_description_relation_total_count": theme_description_total,
        "theme_description_with_source_entity_count": theme_description_with_source,
        "theme_description_missing_source_entity_count": theme_description_missing_source,
        "theme_description_incorrect_theme_mapping_count": theme_description_wrong_theme,
        "theme_description_incorrect_theme_mapping_examples": theme_description_wrong_examples,
        "relation_inconsistent_chunk_unknown_count": relation_inconsistent_chunk_unknown,
        "relation_invalid_grounding_semantics_count": relation_invalid_grounding_semantics,
        "relation_source_sync_mismatch_count": relation_bad_source_sync,
        "entity_missing_source_count": entity_missing_source,
        "entity_missing_extractedFrom_count": entity_missing_extracted_from,
        "entity_missing_chunk_index_count": entity_missing_chunk_index,
        "entity_missing_surface_form_count": entity_missing_surface_form,
        "entity_missing_discourse_id_count": entity_missing_discourse_id,
        "entity_missing_context_source_count": entity_missing_context_source,
        "entity_inconsistent_chunk_unknown_count": entity_inconsistent_chunk_unknown,
        "discourse_missing_source_doc_id_count": discourse_missing_source_doc_id,
        "discourse_missing_chunk_index_count": discourse_missing_chunk_index,
        "discourse_missing_chunk_total_count": discourse_missing_chunk_total,
        "discourse_missing_field_count": discourse_missing_field,
        "discourse_missing_extractedFrom_count": discourse_missing_extracted_from,
        "discourse_missing_chunk_id_count": discourse_missing_chunk_id,
        "discourse_missing_discourse_id_count": discourse_missing_discourse_id,
        "discourse_inconsistent_chunk_unknown_count": discourse_inconsistent_chunk_unknown,
    }


def _is_true_literal(node: object) -> bool:
    sval = str(node).strip().lower()
    return sval in {"true", "1"}


def _local_local_sameas_warnings(graph: Graph, etd_ns: str) -> Dict[str, object]:
    """
    Warn on local<->local owl:sameAs identity assertions unless explicitly justified.
    Justification flags supported on either endpoint:
      - etd:sameAsJustified true
      - etd:identityJustified true
    """
    etd = Namespace(etd_ns)
    local_pairs: List[Tuple[str, str]] = []
    justified = 0
    unjustified = 0
    examples: List[str] = []

    for s, _, o in graph.triples((None, OWL.sameAs, None)):
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        ss = str(s)
        oo = str(o)
        if not (ss.startswith(etd_ns) and oo.startswith(etd_ns)):
            continue

        key = tuple(sorted((ss, oo)))
        if key in local_pairs:
            continue
        local_pairs.append(key)

        s_flags = list(graph.objects(s, etd.sameAsJustified)) + list(graph.objects(s, etd.identityJustified))
        o_flags = list(graph.objects(o, etd.sameAsJustified)) + list(graph.objects(o, etd.identityJustified))
        ok = any(_is_true_literal(v) for v in s_flags + o_flags)
        if ok:
            justified += 1
        else:
            unjustified += 1
            if len(examples) < 10:
                examples.append(f"{key[0]} == {key[1]}")

    return {
        "local_local_sameas_total": len(local_pairs),
        "local_local_sameas_justified": justified,
        "local_local_sameas_unjustified": unjustified,
        "local_local_sameas_examples": examples,
    }


def _etd_class_name_collisions(graph: Graph, etd_ns: str) -> Dict[str, object]:
    """
    Detect ETD class-name collisions that differ only by case/lexical form.
    """
    class_names: Dict[str, str] = {}
    for cls in graph.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue
        s = str(cls)
        if not s.startswith(etd_ns):
            continue
        local = s[len(etd_ns):]
        class_names[local] = s

    by_key: Dict[str, List[str]] = {}
    for local in sorted(class_names.keys(), key=lambda x: x.lower()):
        norm = "".join(ch for ch in local.lower() if ch.isalnum())
        by_key.setdefault(norm, []).append(local)

    collisions = {k: sorted(v) for k, v in by_key.items() if len(v) > 1}
    return {
        "etd_class_name_collision_groups_count": len(collisions),
        "etd_class_name_collisions": collisions,
    }


def _deprecated_etd_class_usage(graph: Graph, etd_ns: str) -> Dict[str, object]:
    """
    Detect deprecated/non-canonical ETD class spellings both in declarations and instance rdf:type usage.
    """
    alias_map = {k: v for k, v in CLASS_CANONICAL_ALIASES.items() if v}

    def _norm(local: str) -> str:
        return "".join(ch for ch in str(local or "").lower() if ch.isalnum())

    deprecated_declared: Dict[str, str] = {}
    for cls in graph.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue
        s = str(cls)
        if not s.startswith(etd_ns):
            continue
        local = s[len(etd_ns):]
        key = _norm(local)
        canonical = alias_map.get(key)
        if canonical and local != canonical:
            deprecated_declared[s] = f"{etd_ns}{canonical}"

    deprecated_instance_usage: Dict[str, int] = {}
    for _, _, cls in graph.triples((None, RDF.type, None)):
        if not isinstance(cls, URIRef):
            continue
        s = str(cls)
        if not s.startswith(etd_ns):
            continue
        local = s[len(etd_ns):]
        key = _norm(local)
        canonical = alias_map.get(key)
        if canonical and local != canonical:
            deprecated_instance_usage[s] = int(deprecated_instance_usage.get(s, 0)) + 1

    return {
        "deprecated_etd_class_declarations_count": len(deprecated_declared),
        "deprecated_etd_class_declarations": deprecated_declared,
        "deprecated_etd_instance_type_usage_count": sum(deprecated_instance_usage.values()),
        "deprecated_etd_instance_type_usage": deprecated_instance_usage,
    }


def _declared_etd_predicate_kinds(graph: Graph, etd_ns: str) -> Dict[str, str]:
    declared: Dict[str, str] = {}
    for pred in graph.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(pred, URIRef) and str(pred).startswith(etd_ns):
            declared[str(pred)] = "datatype"
    for pred in graph.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(pred, URIRef) and str(pred).startswith(etd_ns):
            declared[str(pred)] = "object"
    for pred in graph.subjects(RDF.type, RDF.Property):
        if isinstance(pred, URIRef) and str(pred).startswith(etd_ns):
            declared.setdefault(str(pred), "rdf_property")
    return declared


def _declared_etd_terms(graph: Graph, etd_ns: str) -> Tuple[Set[str], Set[str]]:
    classes: Set[str] = set()
    props: Set[str] = set()
    for cls in graph.subjects(RDF.type, OWL.Class):
        if isinstance(cls, URIRef) and str(cls).startswith(etd_ns):
            classes.add(str(cls))
    for ptype in (OWL.DatatypeProperty, OWL.ObjectProperty, RDF.Property):
        for pred in graph.subjects(RDF.type, ptype):
            if isinstance(pred, URIRef) and str(pred).startswith(etd_ns):
                props.add(str(pred))
    return classes, props


def _etd_term_documentation_coverage(graph: Graph, etd_ns: str) -> Dict[str, object]:
    classes, props = _declared_etd_terms(graph, etd_ns)
    all_terms = sorted(set(classes) | set(props))

    missing_label = []
    missing_comment = []
    for t in all_terms:
        term = URIRef(t)
        if not any(True for _ in graph.objects(term, RDFS.label)):
            missing_label.append(t)
        if not any(True for _ in graph.objects(term, RDFS.comment)):
            missing_comment.append(t)

    return {
        "etd_declared_terms_count": len(all_terms),
        "etd_declared_classes_count": len(classes),
        "etd_declared_properties_count": len(props),
        "etd_terms_missing_label_count": len(missing_label),
        "etd_terms_missing_label": missing_label,
        "etd_terms_missing_comment_count": len(missing_comment),
        "etd_terms_missing_comment": missing_comment,
    }


def _etd_property_lifecycle_coverage(graph: Graph, etd_ns: str) -> Dict[str, object]:
    _, props = _declared_etd_terms(graph, etd_ns)
    status_pred = URIRef(f"{etd_ns}lifecycleStatus")
    note_pred = URIRef(f"{etd_ns}lifecycleNote")
    valid_status = {"active", "conditional", "reserved", "deprecated"}

    missing_status: List[str] = []
    invalid_status: Dict[str, List[str]] = {}
    missing_note_for_non_active: List[str] = []

    for p_str in sorted(props):
        p = URIRef(p_str)
        vals = [str(v).strip().lower() for v in graph.objects(p, status_pred) if str(v).strip()]
        if not vals:
            missing_status.append(p_str)
            continue
        bad = sorted({v for v in vals if v not in valid_status})
        if bad:
            invalid_status[p_str] = bad
            continue
        is_non_active = any(v in {"conditional", "reserved", "deprecated"} for v in vals)
        has_note = any(str(v).strip() for v in graph.objects(p, note_pred))
        if is_non_active and not has_note:
            missing_note_for_non_active.append(p_str)

    return {
        "etd_property_lifecycle_missing_count": len(missing_status),
        "etd_property_lifecycle_missing": missing_status,
        "etd_property_lifecycle_invalid_count": len(invalid_status),
        "etd_property_lifecycle_invalid": invalid_status,
        "etd_property_lifecycle_note_missing_count": len(missing_note_for_non_active),
        "etd_property_lifecycle_note_missing": missing_note_for_non_active,
    }


def _etd_local_name_collisions(graph: Graph, etd_ns: str) -> Dict[str, object]:
    """
    Detect case/lexical local-name collisions across ETD classes and ETD properties.
    """
    classes, props = _declared_etd_terms(graph, etd_ns)
    by_norm: Dict[str, List[Tuple[str, str]]] = {}

    def _norm(local: str) -> str:
        return "".join(ch for ch in local.lower() if ch.isalnum())

    for cls in classes:
        local = cls[len(etd_ns):]
        by_norm.setdefault(_norm(local), []).append(("class", local))
    for p in props:
        local = p[len(etd_ns):]
        by_norm.setdefault(_norm(local), []).append(("property", local))

    allowlist = {
        "source": {("class", "Source"), ("property", "source")},
    }

    allowed: Dict[str, List[str]] = {}
    unexpected: Dict[str, List[str]] = {}
    for key, vals in by_norm.items():
        uniq = sorted(set(vals), key=lambda x: (x[0], x[1].lower(), x[1]))
        if len(uniq) <= 1:
            continue
        labels = [f"{kind}:{name}" for kind, name in uniq]
        if key in allowlist and set(uniq) == allowlist[key]:
            allowed[key] = labels
        else:
            unexpected[key] = labels

    return {
        "etd_local_name_collision_groups_count": len(allowed) + len(unexpected),
        "etd_local_name_collisions_allowed_count": len(allowed),
        "etd_local_name_collisions_allowed": allowed,
        "etd_local_name_collisions_unexpected_count": len(unexpected),
        "etd_local_name_collisions_unexpected": unexpected,
    }


def _used_etd_predicates(graph: Graph, etd_ns: str) -> List[str]:
    used = set()
    for _, p, _ in graph:
        if isinstance(p, URIRef) and str(p).startswith(etd_ns):
            used.add(str(p))
    return sorted(used)


def _observed_predicate_value_shape(graph: Graph, predicate_uri: URIRef) -> str:
    has_literal = False
    has_resource = False
    for _, _, o in graph.triples((None, predicate_uri, None)):
        if isinstance(o, Literal):
            has_literal = True
        elif isinstance(o, (URIRef, BNode)):
            has_resource = True
        if has_literal and has_resource:
            return "mixed"
    if has_literal:
        return "datatype"
    if has_resource:
        return "object"
    return "none"


def _reified_relation_object_shape(graph: Graph, etd: Namespace, predicate_uri: URIRef) -> str:
    """Object shape of ``predicate_uri`` across the SURVIVING reified
    relation population only -- the ``etd:object`` values of every ``?rel``
    such that ``?rel rdf:type etd:Relation`` and
    ``?rel etd:predicate <predicate_uri>``.

    This is the SAME population
    ``serializers/rdf_serializer.py::_declare_local_relation_predicates``
    infers a dynamically declared local predicate's property kind from, so
    validating the declared kind against it (rather than against the
    direct-SPO projection) compares like with like: the direct-SPO
    projection legitimately omits negated / non-asserted relations, which
    would otherwise make a genuinely mixed-object predicate look
    literal-only. Ontology / schema triples and structural relation-node
    metadata are never counted -- only actual relation instances whose
    ``etd:predicate`` is this predicate.

    Returns ``"datatype"`` (literal objects only), ``"object"`` (resource
    objects only), ``"mixed"`` (both), or ``"none"`` (no such relation
    instance carries an ``etd:object``)."""
    has_literal = False
    has_resource = False
    for rel in graph.subjects(etd.predicate, predicate_uri):
        if (rel, RDF.type, etd.Relation) not in graph:
            continue
        for o in graph.objects(rel, etd.object):
            if isinstance(o, Literal):
                has_literal = True
            elif isinstance(o, (URIRef, BNode)):
                has_resource = True
        if has_literal and has_resource:
            return "mixed"
    if has_literal:
        return "datatype"
    if has_resource:
        return "object"
    return "none"


def _external_import_mapping(config_namespaces: Dict[str, str]) -> Dict[str, str]:
    """
    Map namespace base URI -> ontology import IRI.
    """
    mapping: Dict[str, str] = {}
    for key in (
        "aipo",
        "airo",
        "dpv",
        "vair",
        "dcat",
        "prov",
        "skos",
        "foaf",
        "org",
        "sioc",
        "schema",
        "bibo",
        "dcterms",
        "fmo",
        "oa",
        "odrl",
        "cnt",
    ):
        ns = str(config_namespaces.get(key, "")).strip()
        if ns:
            mapping[ns] = ns.rstrip("#/")

    # Canonical ontology document IRIs where namespace != ontology doc IRI.
    if "prov" in config_namespaces:
        mapping[str(config_namespaces["prov"]).strip()] = "http://www.w3.org/ns/prov-o"
    if "skos" in config_namespaces:
        mapping[str(config_namespaces["skos"]).strip()] = "http://www.w3.org/2004/02/skos/core"
    if "dcat" in config_namespaces:
        mapping[str(config_namespaces["dcat"]).strip()] = "http://www.w3.org/ns/dcat"
    if "aipo" in config_namespaces:
        mapping[str(config_namespaces["aipo"]).strip()] = "https://w3id.org/aipo"
    if "airo" in config_namespaces:
        mapping[str(config_namespaces["airo"]).strip()] = "https://w3id.org/airo"
    if "dpv" in config_namespaces:
        mapping[str(config_namespaces["dpv"]).strip()] = "https://w3id.org/dpv"
    if "vair" in config_namespaces:
        mapping[str(config_namespaces["vair"]).strip()] = "https://w3id.org/vair"
    return mapping


def _external_import_coverage(
    graph: Graph,
    etd_ns: str,
    config_namespaces: Dict[str, str],
) -> Dict[str, object]:
    """
    Ensure reused external ontology namespaces are covered by owl:imports.
    """
    mapping = _external_import_mapping(config_namespaces)
    used_namespaces: set = set()
    for s, p, o in graph:
        for n in (s, p, o):
            if not isinstance(n, URIRef):
                continue
            u = str(n)
            if u.startswith(etd_ns) or u.startswith(etd_ns.rstrip("#") + "/"):
                continue
            if u.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#"):
                continue
            if u.startswith("http://www.w3.org/2000/01/rdf-schema#"):
                continue
            if u.startswith("http://www.w3.org/2002/07/owl#"):
                continue
            if u.startswith("http://www.w3.org/2001/XMLSchema#"):
                continue
            for ns_base in mapping.keys():
                if u.startswith(ns_base):
                    used_namespaces.add(ns_base)
                    break

    imported_iris: set = set()
    for ont in graph.subjects(RDF.type, OWL.Ontology):
        for imp in graph.objects(ont, OWL.imports):
            if isinstance(imp, URIRef):
                imported_iris.add(str(imp).rstrip("#/"))

    required_imports = sorted({mapping[ns] for ns in used_namespaces})
    missing_imports = sorted([iri for iri in required_imports if iri.rstrip("#/") not in imported_iris])
    return {
        "external_namespaces_used": sorted(used_namespaces),
        "external_namespaces_used_count": len(used_namespaces),
        "required_external_imports": required_imports,
        "required_external_imports_count": len(required_imports),
        "owl_imports_declared": sorted(imported_iris),
        "owl_imports_declared_count": len(imported_iris),
        "missing_external_imports": missing_imports,
        "missing_external_imports_count": len(missing_imports),
    }


def _expert_typing_metrics(graph: Graph, etd_ns: str) -> Dict[str, object]:
    etd = Namespace(etd_ns)
    expert_nodes = sorted({n for n in graph.subjects(RDF.type, etd.Expert) if isinstance(n, URIRef)}, key=str)

    def _is_source_expert(node: URIRef) -> bool:
        s = str(node)
        local = s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        return (
            "/entity/person/expert-" in s
            or "/entity/expert/" in s
            or local.startswith("expert_")
            or local.startswith("expert-")
        )

    def _is_external_person(node: URIRef) -> bool:
        return "/entity/person/wikidata-" in str(node)

    source_expert_nodes = [n for n in expert_nodes if _is_source_expert(n)]
    external_person_experts = [n for n in expert_nodes if _is_external_person(n)]

    expert_profile_external_links: List[str] = []
    expert_profile_discourse_units: List[URIRef] = []
    expert_profile_multi_source_examples: List[str] = []
    expert_profile_wrong_source_examples: List[str] = []
    expert_profile_correct_one_to_one = 0
    subjects = list(graph.subjects(RDF.type, etd.Relation)) + list(graph.subjects(RDF.type, etd.DiscourseUnit))
    for subject in subjects:
        if not isinstance(subject, URIRef):
            continue
        if "expert.profile" not in _literal_strings(graph, subject, etd.field):
            continue
        source_entities = [src for src in graph.objects(subject, etd.hasSourceEntity) if isinstance(src, URIRef)]
        if (subject, RDF.type, etd.DiscourseUnit) in graph:
            expert_profile_discourse_units.append(subject)
            expected_expert_id = ""
            subject_str = str(subject)
            if "%7C" in subject_str:
                expected_expert_id = subject_str.rsplit("%7C", 1)[-1]
            if not expected_expert_id:
                dids = _literal_strings(graph, subject, etd.discourse_id)
                expected_expert_id = dids[0] if dids else ""

            if len(source_entities) != 1:
                if len(expert_profile_multi_source_examples) < 10:
                    expert_profile_multi_source_examples.append(
                        f"{subject} source_entities={[str(src) for src in source_entities[:10]]}"
                    )
            else:
                only_source = str(source_entities[0])
                is_expected = (
                    bool(expected_expert_id)
                    and (
                        f"/entity/person/expert-{expected_expert_id}-" in only_source
                        or f"/entity/expert/expert-{expected_expert_id}-" in only_source
                        or only_source.endswith(f"expert_{expected_expert_id}")
                    )
                )
                if is_expected:
                    expert_profile_correct_one_to_one += 1
                elif len(expert_profile_wrong_source_examples) < 10:
                    expert_profile_wrong_source_examples.append(
                        f"{subject} expected_expert_id={expected_expert_id} source_entities={[str(src) for src in source_entities]}"
                    )
        for src in source_entities:
            if isinstance(src, URIRef) and _is_external_person(src):
                if len(expert_profile_external_links) < 10:
                    expert_profile_external_links.append(f"{subject} -> {src}")

    dataset_id = ""
    export_nodes = list(graph.subjects(RDF.type, etd.Export))
    if export_nodes:
        dataset_vals = _literal_strings(graph, export_nodes[0], etd.datasetId)
        dataset_id = dataset_vals[0] if dataset_vals else ""
    expected_count = 57 if dataset_id == "ai_ethics_datasets" else None

    return {
        "total_expert_nodes": len(expert_nodes),
        "source_expert_record_count": len(source_expert_nodes),
        "canonical_expert_node_count": len(source_expert_nodes),
        "external_person_nodes_typed_as_expert": len(external_person_experts),
        "external_person_nodes_typed_as_expert_examples": [str(n) for n in external_person_experts[:10]],
        "expert_profile_external_person_source_links_count": len(expert_profile_external_links),
        "expert_profile_external_person_source_link_examples": expert_profile_external_links,
        "expert_profile_discourse_unit_count": len(expert_profile_discourse_units),
        "expert_profile_correct_one_to_one_source_links": expert_profile_correct_one_to_one,
        "expert_profile_multi_source_entity_count": sum(
            1
            for subject in expert_profile_discourse_units
            if len([src for src in graph.objects(subject, etd.hasSourceEntity) if isinstance(src, URIRef)]) != 1
        ),
        "expert_profile_multi_source_entity_examples": expert_profile_multi_source_examples,
        "expert_profile_wrong_source_entity_count": len(expert_profile_wrong_source_examples),
        "expert_profile_wrong_source_entity_examples": expert_profile_wrong_source_examples,
        "expected_source_expert_record_count": expected_count,
    }


def run_quality_gate(
    ttl_path: Path,
    shapes_path: Path,
    output_dir: Path,
    etd_namespace: Optional[str] = None,
) -> Tuple[bool, Dict[str, object], List[str]]:
    from config import Config

    fail_reasons: List[str] = []
    metrics: Dict[str, object] = {}

    graph = Graph()
    try:
        graph.parse(str(ttl_path), format="turtle")
    except Exception as exc:
        fail_reasons.append(f"ttl_parse_failed:{exc}")
        return False, metrics, fail_reasons
    metrics["total_triples"] = len(graph)

    etd_ns = _resolve_etd_namespace(etd_namespace)
    etd = Namespace(etd_ns)
    prov = Namespace("http://www.w3.org/ns/prov#")

    validator = SHACLValidator(shapes_path=shapes_path, output_dir=output_dir)
    shacl = validator.validate_graph(graph)
    counts = shacl.get("counts") or {}
    shacl_conforms = bool(shacl.get("conforms", False))
    shacl_violations = int(counts.get("violations", 0))
    metrics["shacl_conforms"] = shacl_conforms
    metrics["shacl_violations"] = shacl_violations
    metrics["shacl_warnings"] = int(counts.get("warnings", 0))
    metrics["shacl_infos"] = int(counts.get("infos", 0))
    metrics["shacl_report_path"] = shacl.get("report_path")
    if (not shacl_conforms) or shacl_violations > 0:
        fail_reasons.append("shacl_failed")

    entity_nodes = _entity_nodes(graph, etd_ns)
    isolated = _isolated_entities(graph, entity_nodes, etd_ns)
    vote_issues = _vote_typing_issues(graph, etd_ns)
    dup_spo = _duplicate_reified_relation_spo(graph, etd_ns, output_dir)
    same_origin_dup_spo = _same_origin_duplicate_reified_relation_spo(graph, etd_ns)

    metrics["duplicate_relation_spo_groups_count"] = int(dup_spo["duplicate_relation_spo_groups_count"])
    metrics["same_origin_duplicate_relation_spo_groups_count"] = int(
        same_origin_dup_spo["same_origin_duplicate_relation_spo_groups_count"]
    )
    metrics["isolated_entities_count"] = len(isolated)
    metrics["vote_typing_issues"] = vote_issues
    metrics["vote_typing_issues_total"] = _sum_vote_issues(vote_issues)

    if int(same_origin_dup_spo["same_origin_duplicate_relation_spo_groups_count"]) != 0:
        fail_reasons.append("duplicate_reified_spo")

    prune_enabled = bool(getattr(Config, "PRUNE_ISOLATED_ENTITIES", True))
    metrics["prune_isolated_entities_enabled"] = prune_enabled
    if prune_enabled and len(isolated) != 0:
        fail_reasons.append("isolated_entities_nonzero")

    if _sum_vote_issues(vote_issues) != 0:
        fail_reasons.append("vote_typing_issues")

    mismatch = _relation_evidence_provenance_mismatches(graph, etd)
    metrics.update(mismatch)
    if int(mismatch["relation_evidence_mismatched_did"]) > 0 or int(mismatch["relation_evidence_mismatched_chunk"]) > 0:
        fail_reasons.append("relation_evidence_provenance_mismatch")

    prov_cov = _provenance_completeness_metrics(graph, etd)
    metrics.update(prov_cov)
    for key in (
        "evidence_missing_grounding_count",
        "evidence_inconsistent_chunk_unknown_count",
        "grounded_placeholder_evidence_count",
        "evidence_invalid_span_count",
        "orphan_evidence_count",
        "evidence_unexpected_status_count",
        "evidence_unexpected_method_count",
        "relation_missing_source_count",
        "relation_missing_has_evidence_count",
        "relation_missing_field_count",
        "relation_missing_inferred_flag_count",
        "relation_missing_grounding_count",
        "relation_inconsistent_chunk_unknown_count",
        "relation_invalid_grounding_semantics_count",
        "relation_source_sync_mismatch_count",
        "entity_missing_source_count",
        "entity_missing_extractedFrom_count",
        "entity_missing_chunk_index_count",
        "entity_missing_surface_form_count",
        "entity_missing_discourse_id_count",
        "entity_missing_context_source_count",
        "entity_inconsistent_chunk_unknown_count",
        "discourse_missing_source_doc_id_count",
        "discourse_missing_chunk_index_count",
        "discourse_missing_chunk_total_count",
        "discourse_missing_field_count",
        "discourse_missing_extractedFrom_count",
        "discourse_missing_chunk_id_count",
        "discourse_missing_discourse_id_count",
        "discourse_inconsistent_chunk_unknown_count",
    ):
        if int(prov_cov.get(key, 0)) > 0:
            fail_reasons.append(key)

    sameas_warn = _local_local_sameas_warnings(graph, etd_ns)
    metrics.update(sameas_warn)
    doc_cov = _etd_term_documentation_coverage(graph, etd_ns)
    metrics.update(doc_cov)
    if int(doc_cov.get("etd_terms_missing_label_count", 0)) > 0:
        fail_reasons.append("etd_terms_missing_label")
    if int(doc_cov.get("etd_terms_missing_comment_count", 0)) > 0:
        fail_reasons.append("etd_terms_missing_comment")

    lifecycle_cov = _etd_property_lifecycle_coverage(graph, etd_ns)
    metrics.update(lifecycle_cov)
    if int(lifecycle_cov.get("etd_property_lifecycle_missing_count", 0)) > 0:
        fail_reasons.append("etd_property_lifecycle_missing")
    if int(lifecycle_cov.get("etd_property_lifecycle_invalid_count", 0)) > 0:
        fail_reasons.append("etd_property_lifecycle_invalid")
    if int(lifecycle_cov.get("etd_property_lifecycle_note_missing_count", 0)) > 0:
        fail_reasons.append("etd_property_lifecycle_note_missing")

    local_name_collisions = _etd_local_name_collisions(graph, etd_ns)
    metrics.update(local_name_collisions)
    if int(local_name_collisions.get("etd_local_name_collisions_unexpected_count", 0)) > 0:
        fail_reasons.append("etd_local_name_collisions_unexpected")

    import_cov = _external_import_coverage(graph, etd_ns, Config.NAMESPACES)
    metrics.update(import_cov)
    if int(import_cov.get("missing_external_imports_count", 0)) > 0:
        fail_reasons.append("missing_external_owl_imports")
    expert_metrics = _expert_typing_metrics(graph, etd_ns)
    metrics.update(expert_metrics)
    expected_source_expert_count = expert_metrics.get("expected_source_expert_record_count")
    if expected_source_expert_count is not None and expert_metrics.get("source_expert_record_count") != expected_source_expert_count:
        fail_reasons.append("source_expert_record_count_mismatch")
    if expert_metrics.get("total_expert_nodes") != expert_metrics.get("canonical_expert_node_count"):
        fail_reasons.append("expert_total_count_mismatch")
    if int(expert_metrics.get("external_person_nodes_typed_as_expert", 0)) > 0:
        fail_reasons.append("external_person_nodes_typed_as_expert")
    if int(expert_metrics.get("expert_profile_external_person_source_links_count", 0)) > 0:
        fail_reasons.append("expert_profile_external_person_source_links")
    if expected_source_expert_count is not None and expert_metrics.get("expert_profile_discourse_unit_count") != expected_source_expert_count:
        fail_reasons.append("expert_profile_discourse_unit_count_mismatch")
    if expected_source_expert_count is not None and expert_metrics.get("expert_profile_correct_one_to_one_source_links") != expected_source_expert_count:
        fail_reasons.append("expert_profile_correct_one_to_one_source_links_mismatch")
    if int(expert_metrics.get("expert_profile_multi_source_entity_count", 0)) > 0:
        fail_reasons.append("expert_profile_multi_source_entity")
    if int(expert_metrics.get("expert_profile_wrong_source_entity_count", 0)) > 0:
        fail_reasons.append("expert_profile_wrong_source_entity")
    class_collisions = _etd_class_name_collisions(graph, etd_ns)
    metrics.update(class_collisions)
    if int(class_collisions.get("etd_class_name_collision_groups_count", 0)) > 0:
        fail_reasons.append("etd_class_name_collisions")
    deprecated_classes = _deprecated_etd_class_usage(graph, etd_ns)
    metrics.update(deprecated_classes)
    if int(deprecated_classes.get("deprecated_etd_class_declarations_count", 0)) > 0:
        fail_reasons.append("deprecated_etd_class_declarations_present")
    if int(deprecated_classes.get("deprecated_etd_instance_type_usage_count", 0)) > 0:
        fail_reasons.append("deprecated_etd_instance_types_present")

    declared_predicates = _declared_etd_predicate_kinds(graph, etd_ns)
    used_predicates = _used_etd_predicates(graph, etd_ns)
    undeclared_used = sorted([p for p in used_predicates if p not in declared_predicates])
    banned_dynamic_raw = sorted([p for p in used_predicates if p[len(etd_ns):].endswith("_raw")])

    # --- Reconciliation with intended graph design (see CHANGELOG.md) ---
    #
    # utils/semantic_relation_policy.py deliberately extracts an open
    # vocabulary of surface predicates and explicitly tracks, per reified
    # etd:Relation, whether its predicate was mapped onto a declared
    # ontology term (`predicate_mapping_status`). "local_unmapped" is a
    # first-class, intended outcome of that policy -- not an
    # ontology-completeness defect -- so an undeclared predicate whose
    # *every* reified usage is tagged local_unmapped should not, by
    # itself, fail the gate. Still fully reported below either way.
    def _undeclared_is_local_unmapped_only(pred_uri_str: str) -> bool:
        pred_uri = URIRef(pred_uri_str)
        rels = list(graph.subjects(etd.predicate, pred_uri))
        if not rels:
            return False
        return all(
            "local_unmapped" in _literal_strings(graph, rel, etd.predicate_mapping_status)
            for rel in rels
        )

    undeclared_used_local_unmapped = sorted(
        p for p in undeclared_used if _undeclared_is_local_unmapped_only(p)
    )
    undeclared_used_unexplained = sorted(
        p for p in undeclared_used if p not in set(undeclared_used_local_unmapped)
    )

    registry_declared_kind_mismatch: Dict[str, Dict[str, str]] = {}
    for local_name, spec in ETD_PROPERTY_REGISTRY.items():
        pred_uri = f"{etd_ns}{local_name}"
        declared_kind = declared_predicates.get(pred_uri)
        expected_kind = str(spec.get("kind", "datatype")).strip().lower()
        if declared_kind is None:
            registry_declared_kind_mismatch[pred_uri] = {
                "expected": expected_kind,
                "declared": "missing",
            }
            continue
        if expected_kind in {"datatype", "object"} and declared_kind != expected_kind:
            registry_declared_kind_mismatch[pred_uri] = {
                "expected": expected_kind,
                "declared": declared_kind,
            }

    # --- Reconciliation with intended graph design (see CHANGELOG.md) ---
    #
    # A dynamically declared local relation predicate has its property kind
    # inferred by serializers/rdf_serializer.py::
    # _declare_local_relation_predicates from the object shape across ALL of
    # its surviving reified etd:Relation instances (any resource object ->
    # owl:ObjectProperty; literal objects still permitted). This gate must
    # validate the declared kind against that SAME population, so it uses
    # _reified_relation_object_shape -- the etd:object values of the
    # relations whose etd:predicate is this predicate -- as the
    # authoritative observed shape whenever such a population exists. The
    # direct-SPO projection is NOT authoritative here: rdf_serializer
    # legitimately withholds the direct triple for every negated /
    # non-asserted (questioned / hypothetical / uncertain / conditional /
    # normative / reported) relation, so a genuinely mixed-object predicate
    # whose only resource-valued relation is non-asserted would otherwise
    # look literal-only. The direct-SPO shape is still recorded per
    # predicate as a diagnostic (etd_predicate_object_shapes), and is used
    # as the fallback observed shape only for a predicate that has no
    # reified relation population at all (e.g. a non-relation metadata
    # predicate).
    #
    # For a predicate declared owl:ObjectProperty, an observed shape of
    # "mixed" (both resource- and literal-valued relation objects) is
    # exactly what the extraction design produces when an extracted target
    # sometimes does and sometimes doesn't resolve to a minted entity -- it
    # is not a schema violation. A *pure* datatype observation for a
    # declared object-property is only reconciled when
    # utils/semantic_relation_policy.py's own PREDICATE_REGISTRY entry for
    # that predicate explicitly declares literal_allowed=True. Predicates
    # with neither a mixed observation nor such a declaration remain
    # reported as a mismatch.
    observed_kind_mismatch: Dict[str, Dict[str, str]] = {}
    observed_kind_reconciled: Dict[str, Dict[str, str]] = {}
    predicate_object_shapes: Dict[str, Dict[str, str]] = {}
    # Evaluate every predicate used as a plain triple predicate PLUS every
    # predicate that appears as an etd:predicate value on a reified
    # etd:Relation -- the latter may have no surviving direct SPO triple at
    # all (all its relations negated / non-asserted) yet still be a
    # dynamically declared relation predicate whose declared kind must be
    # validated against its reified object usage.
    reified_relation_predicates = {
        str(p)
        for rel in graph.subjects(RDF.type, etd.Relation)
        for p in graph.objects(rel, etd.predicate)
        if isinstance(p, URIRef) and str(p).startswith(etd_ns)
    }
    kind_check_predicates = sorted(set(used_predicates) | reified_relation_predicates)
    for pred_uri_str in kind_check_predicates:
        declared_kind = declared_predicates.get(pred_uri_str)
        if declared_kind not in {"datatype", "object"}:
            continue
        pred_uri = URIRef(pred_uri_str)
        direct_shape = _observed_predicate_value_shape(graph, pred_uri)
        reified_shape = _reified_relation_object_shape(graph, etd, pred_uri)
        # Validate against the population the declaration was inferred from
        # (reified relation objects); fall back to the direct-SPO shape only
        # when there is no reified relation population for this predicate.
        observed_kind = reified_shape if reified_shape != "none" else direct_shape
        if direct_shape != "none" or reified_shape != "none":
            predicate_object_shapes[pred_uri_str] = {
                "reified_object_shape": reified_shape,
                "direct_spo_shape": direct_shape,
                "validated_against": "reified" if reified_shape != "none" else "direct",
            }
        local_name = pred_uri_str[len(etd_ns):]
        policy_entry = PREDICATE_REGISTRY.get(local_name)
        literal_allowed = bool(getattr(policy_entry, "literal_allowed", False))

        if observed_kind == "none":
            continue
        if observed_kind == "mixed":
            if declared_kind == "object":
                observed_kind_reconciled[pred_uri_str] = {
                    "declared": declared_kind,
                    "observed": observed_kind,
                    "reason": "reified relation objects are literal-or-resource; declared ObjectProperty per _declare_local_relation_predicates",
                }
            else:
                observed_kind_mismatch[pred_uri_str] = {"declared": declared_kind, "observed": observed_kind}
            continue
        if observed_kind != declared_kind:
            if declared_kind == "object" and observed_kind == "datatype" and literal_allowed:
                observed_kind_reconciled[pred_uri_str] = {
                    "declared": declared_kind,
                    "observed": observed_kind,
                    "reason": "utils/semantic_relation_policy.py PREDICATE_REGISTRY declares literal_allowed=True",
                }
            else:
                observed_kind_mismatch[pred_uri_str] = {"declared": declared_kind, "observed": observed_kind}

    metrics["undeclared_etd_predicates_used_count"] = len(undeclared_used)
    metrics["undeclared_etd_predicates_used"] = undeclared_used
    metrics["undeclared_etd_predicates_used_local_unmapped_count"] = len(undeclared_used_local_unmapped)
    metrics["undeclared_etd_predicates_used_local_unmapped"] = undeclared_used_local_unmapped
    metrics["undeclared_etd_predicates_used_unexplained_count"] = len(undeclared_used_unexplained)
    metrics["undeclared_etd_predicates_used_unexplained"] = undeclared_used_unexplained
    metrics["banned_dynamic_etd_raw_predicates_count"] = len(banned_dynamic_raw)
    metrics["banned_dynamic_etd_raw_predicates"] = banned_dynamic_raw
    metrics["etd_registry_declared_kind_mismatch_count"] = len(registry_declared_kind_mismatch)
    metrics["etd_registry_declared_kind_mismatch"] = registry_declared_kind_mismatch
    metrics["etd_declared_vs_observed_kind_mismatch_count"] = len(observed_kind_mismatch)
    metrics["etd_declared_vs_observed_kind_mismatch"] = observed_kind_mismatch
    metrics["etd_declared_vs_observed_kind_reconciled_count"] = len(observed_kind_reconciled)
    metrics["etd_declared_vs_observed_kind_reconciled"] = observed_kind_reconciled
    metrics["etd_predicate_object_shapes"] = predicate_object_shapes

    if undeclared_used_unexplained:
        fail_reasons.append("undeclared_etd_predicates_used")
    if banned_dynamic_raw:
        fail_reasons.append("banned_dynamic_etd_raw_predicates_present")
    if registry_declared_kind_mismatch:
        fail_reasons.append("etd_registry_declared_kind_mismatch")
    if observed_kind_mismatch:
        fail_reasons.append("etd_declared_vs_observed_kind_mismatch")

    same_chunk_collisions = int(_same_chunk_duplicate_labels_count(graph, etd_ns))
    cross_chunk_data = _cross_chunk_duplicate_labels(graph, etd_ns, output_dir)
    cross_chunk_collisions = int(cross_chunk_data.get("cross_chunk_collisions_count", 0))
    metrics["same_chunk_collisions_count"] = same_chunk_collisions
    metrics["cross_chunk_collisions_count"] = cross_chunk_collisions
    metrics["cross_chunk_merges_count"] = int(cross_chunk_data.get("cross_chunk_merges_count", 0))
    metrics["cross_chunk_linking_report_path"] = cross_chunk_data.get("cross_chunk_linking_report_path")

    max_same_chunk_collisions = int(getattr(Config, "MAX_SAME_CHUNK_COLLISIONS", 0))
    metrics["max_same_chunk_collisions"] = max_same_chunk_collisions
    if same_chunk_collisions > max_same_chunk_collisions:
        fail_reasons.append("same_chunk_collisions_threshold_exceeded")

    q_total_ev = f"""
    SELECT (COUNT(?ev) AS ?n) WHERE {{
      ?ev a <{etd_ns}Evidence> .
    }}
    """
    q_ev_with_ids = f"""
    SELECT (COUNT(?ev) AS ?n) WHERE {{
      ?ev a <{etd_ns}Evidence> ;
          <{etd_ns}datasetId> ?ds ;
          <{etd_ns}runId> ?run .
    }}
    """
    total_rows = list(graph.query(q_total_ev))
    total_ev = int(total_rows[0][0]) if total_rows else 0
    ev_rows = list(graph.query(q_ev_with_ids))
    ev_with_ids = int(ev_rows[0][0]) if ev_rows else 0
    metrics["total_evidence_nodes"] = total_ev
    metrics["evidence_with_dataset_run"] = ev_with_ids
    if ev_with_ids != total_ev:
        fail_reasons.append("evidence_dataset_run_mismatch")

    q_unknown_chunk = f"""
    SELECT (COUNT(?ev) AS ?n) WHERE {{
      ?ev a <{etd_ns}Evidence> ;
          <{etd_ns}chunk_index> ?ci .
      FILTER(str(?ci) = "-1")
    }}
    """
    unknown_rows = list(graph.query(q_unknown_chunk))
    unknown_chunk = int(unknown_rows[0][0]) if unknown_rows else 0
    unknown_pct = (float(unknown_chunk) / float(total_ev)) if total_ev > 0 else 0.0
    max_unknown_pct = float(getattr(Config, "MAX_UNKNOWN_CHUNK_INDEX_PCT", 0.02))
    metrics["evidence_chunk_index_unknown_count"] = unknown_chunk
    metrics["evidence_chunk_index_unknown_pct"] = round(unknown_pct * 100.0, 2)
    metrics["max_unknown_chunk_index_pct"] = round(max_unknown_pct * 100.0, 2)

    export_nodes = list(graph.subjects(RDF.type, etd.Export))
    metrics["export_node_count"] = len(export_nodes)
    export_en_labels = 0
    if len(export_nodes) == 1:
        export_en_labels = len(
            [l for l in graph.objects(export_nodes[0], RDFS.label) if getattr(l, "language", None) == "en"]
        )
    metrics["export_en_label_count"] = export_en_labels
    if len(export_nodes) != 1 or export_en_labels != 1:
        fail_reasons.append("export_node_or_label_invalid")

    materialize_raw = bool(getattr(Config, "MATERIALIZE_RAW_FIELDS", True))
    metrics["materialize_raw_fields_enabled"] = materialize_raw
    if materialize_raw:
        bad_fa = 0
        field_assertion_inconsistent_chunk_unknown = 0
        field_assertion_missing_origin = 0
        export_node = export_nodes[0] if len(export_nodes) == 1 else None
        for fa in graph.subjects(RDF.type, etd.FieldAssertion):
            has_field_name = any(True for _ in graph.objects(fa, etd.fieldName))
            has_record_id = any(True for _ in graph.objects(fa, etd.recordId))
            has_source_section = any(True for _ in graph.objects(fa, etd.sourceSection))
            preds = list(graph.objects(fa, etd.assertedPredicate))
            vals = list(graph.objects(fa, etd.assertedValue))
            has_ds = any(True for _ in graph.objects(fa, etd.datasetId))
            has_run = any(True for _ in graph.objects(fa, etd.runId))
            has_export_link = export_node is not None and (fa, prov.wasDerivedFrom, export_node) in graph
            has_origin_link = any(True for _ in graph.objects(fa, etd.extractedFrom))
            chunks = _int_literals(graph, fa, etd.chunk_index)
            chunk = chunks[0] if chunks else None
            flags = [str(v).strip().lower() in {"true", "1"} for v in graph.objects(fa, etd.chunk_index_unknown)]
            flag = flags[0] if flags else None

            pred_iri_ok = len(preds) >= 1 and all(isinstance(p, URIRef) for p in preds)
            val_ok = len(vals) >= 1
            origin_ok = has_origin_link or has_export_link
            if chunk is not None and (flag is None or ((chunk < 0) != bool(flag))):
                field_assertion_inconsistent_chunk_unknown += 1
            if not origin_ok:
                field_assertion_missing_origin += 1

            if not (
                has_field_name
                and has_record_id
                and has_source_section
                and pred_iri_ok
                and val_ok
                and has_ds
                and has_run
                and has_export_link
            ):
                bad_fa += 1

        metrics["field_assertion_invalid_count"] = bad_fa
        metrics["field_assertion_inconsistent_chunk_unknown_count"] = field_assertion_inconsistent_chunk_unknown
        metrics["field_assertion_missing_origin_count"] = field_assertion_missing_origin
        if bad_fa > 0:
            fail_reasons.append("field_assertion_integrity_failed")
        if field_assertion_inconsistent_chunk_unknown > 0:
            fail_reasons.append("field_assertion_chunk_unknown_inconsistent")
        if field_assertion_missing_origin > 0:
            fail_reasons.append("field_assertion_missing_origin")
    else:
        metrics["field_assertion_invalid_count"] = 0
        metrics["field_assertion_inconsistent_chunk_unknown_count"] = 0
    metrics["field_assertion_missing_origin_count"] = 0

    if metrics.get("theme_description_missing_source_entity_count", 0) > 0:
        fail_reasons.append("theme_description_missing_source_entity")
    if metrics.get("theme_description_incorrect_theme_mapping_count", 0) > 0:
        fail_reasons.append("theme_description_incorrect_theme_mapping")

    return len(fail_reasons) == 0, metrics, fail_reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic TTL quality gate")
    parser.add_argument("ttl_positional", nargs="?", default=None)
    parser.add_argument("--ttl", default="output/etd_hub_enhanced_kg.ttl")
    parser.add_argument("--shapes", default="ontology/shacl_shapes.ttl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--etd-namespace", default=None)
    args = parser.parse_args()

    ttl_path = Path(args.ttl_positional or args.ttl)
    shapes_path = Path(args.shapes)
    output_dir = Path(args.output_dir)

    if not ttl_path.exists():
        print(f"[FAIL] TTL not found: {ttl_path}")
        return 1
    if not shapes_path.exists():
        print(f"[FAIL] SHACL shapes not found: {shapes_path}")
        return 1

    ok, metrics, reasons = run_quality_gate(
        ttl_path=ttl_path,
        shapes_path=shapes_path,
        output_dir=output_dir,
        etd_namespace=args.etd_namespace,
    )

    print("=== KG Quality Gate ===")
    print(f"SHACL conforms: {metrics.get('shacl_conforms')}")
    print(f"SHACL violations: {metrics.get('shacl_violations')}")
    print(f"SHACL warnings: {metrics.get('shacl_warnings')}")
    print(f"SHACL infos: {metrics.get('shacl_infos')}")
    print(f"SHACL report_path: {metrics.get('shacl_report_path')}")
    print(f"total_triples: {metrics.get('total_triples')}")
    print(f"duplicate_relation_spo_groups_count: {metrics.get('duplicate_relation_spo_groups_count')}")
    print(f"isolated_entities_count: {metrics.get('isolated_entities_count')}")
    print(f"vote_typing_issues: {metrics.get('vote_typing_issues')}")
    print(f"relation_evidence_mismatched_did: {metrics.get('relation_evidence_mismatched_did')}")
    print(f"relation_evidence_mismatched_chunk: {metrics.get('relation_evidence_mismatched_chunk')}")
    print(f"local_local_sameas_total: {metrics.get('local_local_sameas_total')}")
    print(f"local_local_sameas_unjustified: {metrics.get('local_local_sameas_unjustified')}")
    if metrics.get("local_local_sameas_examples"):
        print(f"local_local_sameas_examples: {metrics.get('local_local_sameas_examples')}")
    print(f"etd_terms_missing_label_count: {metrics.get('etd_terms_missing_label_count')}")
    if metrics.get("etd_terms_missing_label"):
        print(f"etd_terms_missing_label: {metrics.get('etd_terms_missing_label')}")
    print(f"etd_terms_missing_comment_count: {metrics.get('etd_terms_missing_comment_count')}")
    if metrics.get("etd_terms_missing_comment"):
        print(f"etd_terms_missing_comment: {metrics.get('etd_terms_missing_comment')}")
    print(f"etd_property_lifecycle_missing_count: {metrics.get('etd_property_lifecycle_missing_count')}")
    if metrics.get("etd_property_lifecycle_missing"):
        print(f"etd_property_lifecycle_missing: {metrics.get('etd_property_lifecycle_missing')}")
    print(f"etd_property_lifecycle_invalid_count: {metrics.get('etd_property_lifecycle_invalid_count')}")
    if metrics.get("etd_property_lifecycle_invalid"):
        print(f"etd_property_lifecycle_invalid: {metrics.get('etd_property_lifecycle_invalid')}")
    print(f"etd_property_lifecycle_note_missing_count: {metrics.get('etd_property_lifecycle_note_missing_count')}")
    if metrics.get("etd_property_lifecycle_note_missing"):
        print(f"etd_property_lifecycle_note_missing: {metrics.get('etd_property_lifecycle_note_missing')}")
    print(f"etd_local_name_collision_groups_count: {metrics.get('etd_local_name_collision_groups_count')}")
    print(f"etd_local_name_collisions_allowed_count: {metrics.get('etd_local_name_collisions_allowed_count')}")
    if metrics.get("etd_local_name_collisions_allowed"):
        print(f"etd_local_name_collisions_allowed: {metrics.get('etd_local_name_collisions_allowed')}")
    print(f"etd_local_name_collisions_unexpected_count: {metrics.get('etd_local_name_collisions_unexpected_count')}")
    if metrics.get("etd_local_name_collisions_unexpected"):
        print(f"etd_local_name_collisions_unexpected: {metrics.get('etd_local_name_collisions_unexpected')}")
    print(f"external_namespaces_used_count: {metrics.get('external_namespaces_used_count')}")
    if metrics.get("external_namespaces_used"):
        print(f"external_namespaces_used: {metrics.get('external_namespaces_used')}")
    print(f"required_external_imports_count: {metrics.get('required_external_imports_count')}")
    if metrics.get("required_external_imports"):
        print(f"required_external_imports: {metrics.get('required_external_imports')}")
    print(f"owl_imports_declared_count: {metrics.get('owl_imports_declared_count')}")
    if metrics.get("owl_imports_declared"):
        print(f"owl_imports_declared: {metrics.get('owl_imports_declared')}")
    print(f"missing_external_imports_count: {metrics.get('missing_external_imports_count')}")
    if metrics.get("missing_external_imports"):
        print(f"missing_external_imports: {metrics.get('missing_external_imports')}")
    print(f"etd_class_name_collision_groups_count: {metrics.get('etd_class_name_collision_groups_count')}")
    if metrics.get("etd_class_name_collisions"):
        print(f"etd_class_name_collisions: {metrics.get('etd_class_name_collisions')}")
    print(f"deprecated_etd_class_declarations_count: {metrics.get('deprecated_etd_class_declarations_count')}")
    if metrics.get("deprecated_etd_class_declarations"):
        print(f"deprecated_etd_class_declarations: {metrics.get('deprecated_etd_class_declarations')}")
    print(f"deprecated_etd_instance_type_usage_count: {metrics.get('deprecated_etd_instance_type_usage_count')}")
    if metrics.get("deprecated_etd_instance_type_usage"):
        print(f"deprecated_etd_instance_type_usage: {metrics.get('deprecated_etd_instance_type_usage')}")
    print(f"undeclared_etd_predicates_used_count: {metrics.get('undeclared_etd_predicates_used_count')}")
    if metrics.get("undeclared_etd_predicates_used"):
        print(f"undeclared_etd_predicates_used: {metrics.get('undeclared_etd_predicates_used')}")
    print(
        "  of which local_unmapped (reconciled, non-blocking): "
        f"{metrics.get('undeclared_etd_predicates_used_local_unmapped_count')}"
    )
    print(
        "  of which unexplained (still gates the FAIL): "
        f"{metrics.get('undeclared_etd_predicates_used_unexplained_count')}"
    )
    if metrics.get("undeclared_etd_predicates_used_unexplained"):
        print(f"undeclared_etd_predicates_used_unexplained: {metrics.get('undeclared_etd_predicates_used_unexplained')}")
    print(f"banned_dynamic_etd_raw_predicates_count: {metrics.get('banned_dynamic_etd_raw_predicates_count')}")
    if metrics.get("banned_dynamic_etd_raw_predicates"):
        print(f"banned_dynamic_etd_raw_predicates: {metrics.get('banned_dynamic_etd_raw_predicates')}")
    print(f"etd_registry_declared_kind_mismatch_count: {metrics.get('etd_registry_declared_kind_mismatch_count')}")
    if metrics.get("etd_registry_declared_kind_mismatch"):
        print(f"etd_registry_declared_kind_mismatch: {metrics.get('etd_registry_declared_kind_mismatch')}")
    print(f"etd_declared_vs_observed_kind_mismatch_count: {metrics.get('etd_declared_vs_observed_kind_mismatch_count')}")
    if metrics.get("etd_declared_vs_observed_kind_mismatch"):
        print(f"etd_declared_vs_observed_kind_mismatch: {metrics.get('etd_declared_vs_observed_kind_mismatch')}")
    print(f"etd_declared_vs_observed_kind_reconciled_count: {metrics.get('etd_declared_vs_observed_kind_reconciled_count')}")
    if metrics.get("etd_declared_vs_observed_kind_reconciled"):
        print(f"etd_declared_vs_observed_kind_reconciled: {metrics.get('etd_declared_vs_observed_kind_reconciled')}")
    if metrics.get("etd_predicate_object_shapes"):
        print(f"etd_predicate_object_shapes (reified vs direct-SPO, diagnostic): {metrics.get('etd_predicate_object_shapes')}")
    print(f"same_chunk_collisions_count: {metrics.get('same_chunk_collisions_count')}")
    print(f"max_same_chunk_collisions: {metrics.get('max_same_chunk_collisions')}")
    print(f"cross_chunk_collisions_count (warning-only): {metrics.get('cross_chunk_collisions_count')}")
    print(f"cross_chunk_merges_count (audit metric): {metrics.get('cross_chunk_merges_count')}")
    print(f"cross_chunk_linking_report_path: {metrics.get('cross_chunk_linking_report_path')}")
    print(f"total_evidence_nodes: {metrics.get('total_evidence_nodes')}")
    print(f"total_relation_nodes: {metrics.get('total_relation_nodes')}")
    print(f"total_entity_nodes: {metrics.get('total_entity_nodes')}")
    print(f"total_discourse_unit_nodes: {metrics.get('total_discourse_unit_nodes')}")
    print(f"evidence_with_dataset_run: {metrics.get('evidence_with_dataset_run')}")
    print(f"evidence_chunk_index_unknown_count: {metrics.get('evidence_chunk_index_unknown_count')}")
    print(f"evidence_chunk_index_unknown_pct: {metrics.get('evidence_chunk_index_unknown_pct')}")
    print(f"evidence_missing_grounding_count: {metrics.get('evidence_missing_grounding_count')}")
    print(f"evidence_inconsistent_chunk_unknown_count: {metrics.get('evidence_inconsistent_chunk_unknown_count')}")
    print(f"grounded_placeholder_evidence_count: {metrics.get('grounded_placeholder_evidence_count')}")
    print(f"evidence_invalid_span_count: {metrics.get('evidence_invalid_span_count')}")
    print(f"orphan_evidence_count: {metrics.get('orphan_evidence_count')}")
    print(f"evidence_unexpected_status_count: {metrics.get('evidence_unexpected_status_count')}")
    print(f"evidence_unexpected_method_count: {metrics.get('evidence_unexpected_method_count')}")
    print(f"evidence_unknown_chunk_by_source: {metrics.get('evidence_unknown_chunk_by_source')}")
    print(f"evidence_unknown_chunk_by_field: {metrics.get('evidence_unknown_chunk_by_field')}")
    print(f"relation_missing_source_count: {metrics.get('relation_missing_source_count')}")
    print(f"relation_missing_has_evidence_count: {metrics.get('relation_missing_has_evidence_count')}")
    print(f"relation_missing_field_count: {metrics.get('relation_missing_field_count')}")
    print(f"relation_missing_inferred_flag_count: {metrics.get('relation_missing_inferred_flag_count')}")
    print(f"relation_missing_grounding_count: {metrics.get('relation_missing_grounding_count')}")
    print(f"relation_has_source_entity_count: {metrics.get('relation_has_source_entity_count')}")
    print(f"relation_missing_source_entity_count: {metrics.get('relation_missing_source_entity_count')}")
    print(f"relation_traces_to_answer_count: {metrics.get('relation_traces_to_answer_count')}")
    print(f"relation_traces_to_question_count: {metrics.get('relation_traces_to_question_count')}")
    print(f"relation_traces_to_theme_count: {metrics.get('relation_traces_to_theme_count')}")
    print(f"relation_traces_to_expert_count: {metrics.get('relation_traces_to_expert_count')}")
    print(f"relation_traces_to_vote_count: {metrics.get('relation_traces_to_vote_count')}")
    print(f"source_expert_record_count: {metrics.get('source_expert_record_count')}")
    print(f"canonical_expert_node_count: {metrics.get('canonical_expert_node_count')}")
    print(f"total_expert_nodes: {metrics.get('total_expert_nodes')}")
    print(f"external_person_nodes_typed_as_expert: {metrics.get('external_person_nodes_typed_as_expert')}")
    if metrics.get("external_person_nodes_typed_as_expert_examples"):
        print(f"external_person_nodes_typed_as_expert_examples: {metrics.get('external_person_nodes_typed_as_expert_examples')}")
    print(f"expert_profile_external_person_source_links_count: {metrics.get('expert_profile_external_person_source_links_count')}")
    if metrics.get("expert_profile_external_person_source_link_examples"):
        print(f"expert_profile_external_person_source_link_examples: {metrics.get('expert_profile_external_person_source_link_examples')}")
    print(f"expert_profile_discourse_unit_count: {metrics.get('expert_profile_discourse_unit_count')}")
    print(f"expert_profile_correct_one_to_one_source_links: {metrics.get('expert_profile_correct_one_to_one_source_links')}")
    print(f"expert_profile_multi_source_entity_count: {metrics.get('expert_profile_multi_source_entity_count')}")
    if metrics.get("expert_profile_multi_source_entity_examples"):
        print(f"expert_profile_multi_source_entity_examples: {metrics.get('expert_profile_multi_source_entity_examples')}")
    print(f"expert_profile_wrong_source_entity_count: {metrics.get('expert_profile_wrong_source_entity_count')}")
    if metrics.get("expert_profile_wrong_source_entity_examples"):
        print(f"expert_profile_wrong_source_entity_examples: {metrics.get('expert_profile_wrong_source_entity_examples')}")
    print(f"theme_description_relation_total_count: {metrics.get('theme_description_relation_total_count')}")
    print(f"theme_description_with_source_entity_count: {metrics.get('theme_description_with_source_entity_count')}")
    print(f"theme_description_missing_source_entity_count: {metrics.get('theme_description_missing_source_entity_count')}")
    print(f"theme_description_incorrect_theme_mapping_count: {metrics.get('theme_description_incorrect_theme_mapping_count')}")
    if metrics.get("theme_description_incorrect_theme_mapping_examples"):
        print(f"theme_description_incorrect_theme_mapping_examples: {metrics.get('theme_description_incorrect_theme_mapping_examples')}")
    print(f"relation_inconsistent_chunk_unknown_count: {metrics.get('relation_inconsistent_chunk_unknown_count')}")
    print(f"relation_invalid_grounding_semantics_count: {metrics.get('relation_invalid_grounding_semantics_count')}")
    print(f"relation_source_sync_mismatch_count: {metrics.get('relation_source_sync_mismatch_count')}")
    print(f"entity_missing_source_count: {metrics.get('entity_missing_source_count')}")
    print(f"entity_missing_extractedFrom_count: {metrics.get('entity_missing_extractedFrom_count')}")
    print(f"entity_missing_chunk_index_count: {metrics.get('entity_missing_chunk_index_count')}")
    print(f"entity_missing_surface_form_count: {metrics.get('entity_missing_surface_form_count')}")
    print(f"entity_missing_discourse_id_count: {metrics.get('entity_missing_discourse_id_count')}")
    print(f"entity_missing_context_source_count: {metrics.get('entity_missing_context_source_count')}")
    print(f"entity_inconsistent_chunk_unknown_count: {metrics.get('entity_inconsistent_chunk_unknown_count')}")
    print(f"discourse_missing_source_doc_id_count: {metrics.get('discourse_missing_source_doc_id_count')}")
    print(f"discourse_missing_chunk_index_count: {metrics.get('discourse_missing_chunk_index_count')}")
    print(f"discourse_missing_chunk_total_count: {metrics.get('discourse_missing_chunk_total_count')}")
    print(f"discourse_missing_field_count: {metrics.get('discourse_missing_field_count')}")
    print(f"discourse_missing_extractedFrom_count: {metrics.get('discourse_missing_extractedFrom_count')}")
    print(f"discourse_missing_chunk_id_count: {metrics.get('discourse_missing_chunk_id_count')}")
    print(f"discourse_missing_discourse_id_count: {metrics.get('discourse_missing_discourse_id_count')}")
    print(f"discourse_inconsistent_chunk_unknown_count: {metrics.get('discourse_inconsistent_chunk_unknown_count')}")
    print(f"max_unknown_chunk_index_pct: {metrics.get('max_unknown_chunk_index_pct')}")
    print(f"export_node_count: {metrics.get('export_node_count')}")
    print(f"export_en_label_count: {metrics.get('export_en_label_count')}")
    print(f"field_assertion_invalid_count: {metrics.get('field_assertion_invalid_count')}")
    print(f"field_assertion_inconsistent_chunk_unknown_count: {metrics.get('field_assertion_inconsistent_chunk_unknown_count')}")
    print(f"field_assertion_missing_origin_count: {metrics.get('field_assertion_missing_origin_count')}")

    if ok:
        print("[PASS] KG quality gate passed")
        return 0
    print(f"[FAIL] KG quality gate failed: {', '.join(reasons)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
