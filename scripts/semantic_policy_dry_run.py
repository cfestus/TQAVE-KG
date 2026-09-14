import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import Config
from pipeline import EnhancedSemanticPipeline, SEMANTIC_RELATION_PREDICATES
from utils.helpers import load_json, validate_input_json


FIELD_ORDER = [
    ("themes", "theme.description", "description"),
    ("questions", "question.body", "body"),
    ("answers", "answer.description", "description"),
]


def iter_blocks(annotations):
    return [
        annotations.themes,
        annotations.questions,
        annotations.answers,
        annotations.votes,
        annotations.experts,
        annotations.documents,
    ]


def reason_bucket(code: str) -> str:
    value = str(code or "").strip()
    if value.startswith("entity_role:"):
        return "entity-role ambiguity"
    if value.startswith("predicate_ambiguity:"):
        return "predicate ambiguity"
    if value.startswith("domain_range:"):
        return "domain/range"
    if value.startswith("locality:"):
        return "locality"
    if value.startswith("entailment:"):
        return "entailment"
    if value.startswith("modality:"):
        return "modality"
    if value.startswith("resource_filter:"):
        return "resource filtering"
    if value.startswith("duplicate:"):
        return "duplicate"
    return "other"


def accepted_quality(row: Dict[str, Any]) -> str:
    predicate_source = str(row.get("predicate_policy_source") or "").strip()
    epistemic = str(row.get("epistemic_status") or "").strip()
    if predicate_source in {
        "explicit_lexical_relation",
        "canonical_ontology_equivalent",
        "strongly_entailed_semantic_mapping",
    } and epistemic in {"", "asserted", "questioned", "hypothetical"}:
        return "clearly supported"
    if predicate_source == "generic_meaningful_association" or epistemic in {
        "questioned",
        "hypothetical",
        "conditional",
        "uncertain",
    }:
        return "weak / ambiguous"
    return "unsupported"


def likely_valid_rejection(row: Dict[str, Any]) -> bool:
    if str(row.get("section_category") or "").strip() != "semantic":
        return False
    if str(row.get("predicate") or "").strip() in {"relatedTo", "associatedWith", "mentions"}:
        return False
    return reason_bucket(str(row.get("semantic_validation_code") or "")) in {
        "locality",
        "entailment",
        "entity-role ambiguity",
    }


def relation_row(relation, pr, entity_index, stage, reason_override=None):
    meta = getattr(relation, "metadata", None) or {}
    src = entity_index.get(str(getattr(relation, "source", "") or ""))
    tgt = entity_index.get(str(getattr(relation, "target", "") or ""))
    evidence = str(meta.get("evidence_quote") or meta.get("evidence") or getattr(relation, "context", None) or "").strip()
    local_source_text = str(getattr(relation, "context", None) or (pr.metadata or {}).get("text") or "").strip()
    reason = str(reason_override or meta.get("semantic_validation_reason") or "").strip()
    code = str(meta.get("semantic_validation_code") or "").strip()
    row = {
        "field": str(meta.get("field") or meta.get("source_field") or "").strip(),
        "source_field": str(meta.get("source_field") or meta.get("field") or "").strip(),
        "source_id": str(meta.get("source_doc_id") or meta.get("discourse_id") or ""),
        "section": str(meta.get("section") or ""),
        "section_category": str(meta.get("section_category") or ""),
        "local_source_text": local_source_text,
        "evidence": evidence,
        "subject": str(getattr(src, "label", "") or ""),
        "subject_surface_form": str(meta.get("subject_surface_form") or ""),
        "subject_canonical_entity": str(meta.get("subject_canonical_entity") or ""),
        "subject_type": str(getattr(src, "entity_type", "") or ""),
        "predicate": str(getattr(relation, "relation", "") or ""),
        "predicate_surface_phrase": str(meta.get("predicate_surface_phrase") or ""),
        "canonical_predicate": str(meta.get("canonical_predicate") or ""),
        "object": str(getattr(tgt, "label", "") or ""),
        "object_surface_form": str(meta.get("object_surface_form") or ""),
        "object_canonical_entity": str(meta.get("object_canonical_entity") or ""),
        "object_type": str(getattr(tgt, "entity_type", "") or ""),
        "confidence": float(getattr(relation, "confidence", 0.0) or 0.0),
        "relation_kind": str(meta.get("relation_kind") or ""),
        "semantic_validation_status": str(meta.get("semantic_validation_status") or ""),
        "semantic_validation_code": code,
        "semantic_validation_reason": reason,
        "relation_origin": str(meta.get("relation_origin") or ""),
        "predicate_policy_source": str(meta.get("predicate_policy_source") or ""),
        "direction": str(meta.get("direction_normalized") or ""),
        "locality_scope": str(meta.get("locality_scope") or ""),
        "discourse_mode": str(meta.get("discourse_mode") or ""),
        "epistemic_status": str(meta.get("epistemic_status") or ""),
        "modality_tags": str(meta.get("modality_tags") or ""),
        "subject_family": str(meta.get("subject_family") or ""),
        "object_family": str(meta.get("object_family") or ""),
        "entity_role_check": str(meta.get("entity_role_check") or ""),
        "domain_range_check": str(meta.get("domain_range_check") or ""),
        "entailment_check": str(meta.get("entailment_check") or ""),
        "modality_check": str(meta.get("modality_check") or ""),
        "source_uri": str(getattr(relation, "source", "") or ""),
        "target_uri": str(getattr(relation, "target", "") or ""),
        "stage": stage,
    }
    row["accepted_quality"] = accepted_quality(row) if stage == "accepted" else ""
    row["rejection_bucket"] = reason_bucket(code) if stage == "rejected" else ""
    row["likely_valid_rejection"] = likely_valid_rejection(row) if stage == "rejected" else False
    row["rejection_seems_correct"] = False if row["likely_valid_rejection"] else True
    return row


def semantic_signature(row: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("source_field") or "").strip(),
        str(row.get("source_id") or "").strip(),
        str(row.get("subject_canonical_entity") or row.get("source_uri") or "").strip(),
        str(row.get("canonical_predicate") or row.get("predicate") or "").strip(),
        str(row.get("object_canonical_entity") or row.get("target_uri") or "").strip(),
        str(row.get("epistemic_status") or "").strip(),
        str(row.get("relation_origin") or "").strip(),
    )


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        sig = semantic_signature(row)
        existing = deduped.get(sig)
        if existing is None:
            deduped[sig] = row
            continue
        current_score = (
            float(row.get("confidence") or 0.0),
            len(str(row.get("semantic_validation_reason") or "")),
            str(row.get("predicate_policy_source") or ""),
        )
        existing_score = (
            float(existing.get("confidence") or 0.0),
            len(str(existing.get("semantic_validation_reason") or "")),
            str(existing.get("predicate_policy_source") or ""),
        )
        if current_score > existing_score:
            deduped[sig] = row
    return sorted(
        deduped.values(),
        key=lambda row: (
            row["source_field"],
            row["source_id"],
            row["subject"],
            row["predicate"],
            row["object"],
            row["relation_origin"],
        ),
    )


def collect_final_accepted_rows(annotations, entity_index):
    rows: List[Dict[str, Any]] = []
    for block in iter_blocks(annotations):
        for pr in block.values():
            for relation in pr.relations:
                meta = getattr(relation, "metadata", None) or {}
                pred = str(getattr(relation, "relation", "") or "").strip()
                if pred not in SEMANTIC_RELATION_PREDICATES:
                    continue
                if str(meta.get("semantic_validation_status") or "").strip() != "accepted":
                    continue
                if str(meta.get("relation_kind") or "").strip() in {"structural", "provenance"}:
                    continue
                rows.append(relation_row(relation, pr, entity_index, "accepted"))
    return dedupe_rows(rows)


def make_entity_index(pipeline):
    get_all = getattr(pipeline.global_registry, "get_all_entities", None)
    if callable(get_all):
        return {
            str(getattr(ent, "uri", "") or ""): ent
            for ent in get_all()
            if str(getattr(ent, "uri", "") or "").strip()
        }
    return {}


def main():
    out_dir = REPO / "output" / "semantic_policy_dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    Config.OUTPUT_DIR = out_dir
    Config.BATCH_SIZE = 100

    input_path = REPO / Config.INPUT_JSON
    data = load_json(input_path)
    validate_input_json(data)

    audit: Dict[str, Any] = {
        "input_file": str(input_path),
        "eligible_source_records": 0,
        "eligible_records_by_field": {},
        "initial_semantic_candidates": 0,
        "accepted_semantic_candidates": 0,
        "rejected_semantic_candidates": 0,
        "accepted_for_materialisation": 0,
        "accepted_by_field": {},
        "rejected_by_field": {},
        "rejections_by_bucket": {},
        "accepted_audit": [],
        "rejected_audit": [],
        "rejected_valid_looking_audit": [],
    }

    for bucket_key, field_name, text_key in FIELD_ORDER:
        raw_bucket = data.get(bucket_key, data.get(bucket_key[:-1].title(), []))
        eligible = [row for row in raw_bucket if str(row.get(text_key) or "").strip()]
        audit["eligible_records_by_field"][field_name] = len(eligible)
        audit["eligible_source_records"] += len(eligible)

    pipeline = EnhancedSemanticPipeline(Config)
    policy = pipeline._get_semantic_policy()
    (out_dir / "predicate_registry.json").write_text(policy.registry_json(), encoding="utf-8")

    orig_apply = pipeline.apply_predicate_precision_and_enrichment
    orig_validate = pipeline.validate_semantic_relations
    orig_validate_candidate = pipeline._validate_semantic_relation_candidate

    def audited_apply(annotations):
        count = 0
        for block in iter_blocks(annotations):
            for pr in block.values():
                for relation in pr.relations:
                    meta = getattr(relation, "metadata", None) or {}
                    pred = str(getattr(relation, "relation", "") or "").strip()
                    relation_kind = str(meta.get("relation_kind") or "").strip()
                    if pred in SEMANTIC_RELATION_PREDICATES and relation_kind not in {"structural", "provenance"}:
                        count += 1
        audit["initial_semantic_candidates"] = count
        return orig_apply(annotations)

    def audited_validate(annotations):
        entity_index = make_entity_index(pipeline)
        pre_rows = {}
        decisions = {}
        structural_rejections = []

        for block in iter_blocks(annotations):
            for pr in block.values():
                for relation in pr.relations:
                    meta = getattr(relation, "metadata", None) or {}
                    pred = str(getattr(relation, "relation", "") or "").strip()
                    if pred not in SEMANTIC_RELATION_PREDICATES:
                        continue
                    pre_rows[id(relation)] = (relation, pr)
                    if str(meta.get("relation_kind") or "").strip() in {"structural", "provenance"}:
                        structural_rejections.append((relation, pr))

        def wrapped_validate_candidate(relation, **kwargs):
            accepted, reason = orig_validate_candidate(relation, **kwargs)
            decisions[id(relation)] = (accepted, reason)
            return accepted, reason

        pipeline._validate_semantic_relation_candidate = wrapped_validate_candidate
        try:
            summary = orig_validate(annotations)
        finally:
            pipeline._validate_semantic_relation_candidate = orig_validate_candidate

        accepted_rows: List[Dict[str, Any]] = []
        rejected_rows: List[Dict[str, Any]] = []
        accepted_by_field = Counter()
        rejected_by_field = Counter()
        rejected_by_bucket = Counter()

        for relation, pr in structural_rejections:
            row = relation_row(relation, pr, entity_index, "rejected")
            rejected_rows.append(row)
            rejected_by_field[row["source_field"]] += 1
            rejected_by_bucket[row["rejection_bucket"]] += 1

        for rel_id, (relation, pr) in pre_rows.items():
            if rel_id not in decisions:
                continue
            accepted, reason = decisions[rel_id]
            if accepted:
                continue
            row = relation_row(relation, pr, entity_index, "rejected", reason_override=reason)
            rejected_rows.append(row)
            rejected_by_field[row["source_field"]] += 1
            rejected_by_bucket[row["rejection_bucket"]] += 1

        accepted_rows = collect_final_accepted_rows(annotations, entity_index)
        for row in accepted_rows:
            accepted_by_field[row["source_field"]] += 1

        rejected_rows.sort(key=lambda row: (row["source_field"], row["source_id"], row["subject"], row["predicate"], row["object"]))

        audit["accepted_audit"] = accepted_rows
        audit["rejected_audit"] = rejected_rows
        audit["accepted_semantic_candidates"] = len(accepted_rows)
        audit["rejected_semantic_candidates"] = len(rejected_rows)
        audit["accepted_for_materialisation"] = len(accepted_rows)
        audit["accepted_by_field"] = dict(sorted(accepted_by_field.items()))
        audit["rejected_by_field"] = dict(sorted(rejected_by_field.items()))
        audit["rejections_by_bucket"] = dict(sorted(rejected_by_bucket.items()))
        audit["validation_summary"] = summary
        audit["rejected_valid_looking_audit"] = sorted(
            [row for row in rejected_rows if row["likely_valid_rejection"]],
            key=lambda row: (-float(row["confidence"]), row["source_field"], row["source_id"], row["predicate"]),
        )[:30]
        return summary

    pipeline.apply_predicate_precision_and_enrichment = audited_apply
    pipeline.validate_semantic_relations = audited_validate

    annotations = pipeline.process_batch(data)
    inferred = pipeline.apply_reasoning(annotations)
    audit["reasoner_inferred_validated"] = len(inferred or [])

    audit_path = out_dir / "semantic_policy_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "eligible_source_records": audit["eligible_source_records"],
        "eligible_records_by_field": audit["eligible_records_by_field"],
        "initial_semantic_candidates": audit["initial_semantic_candidates"],
        "accepted_semantic_candidates": audit["accepted_semantic_candidates"],
        "rejected_semantic_candidates": audit["rejected_semantic_candidates"],
        "accepted_for_materialisation": audit["accepted_for_materialisation"],
        "accepted_by_field": audit["accepted_by_field"],
        "rejected_by_field": audit["rejected_by_field"],
        "rejections_by_bucket": audit["rejections_by_bucket"],
        "reasoner_inferred_validated": audit["reasoner_inferred_validated"],
        "predicate_registry_path": str(out_dir / "predicate_registry.json"),
        "audit_path": str(audit_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
