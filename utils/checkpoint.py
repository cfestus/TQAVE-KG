"""
utils/checkpoint.py
--------------------
Crash-safe checkpointing for the extraction -> provenance -> serialization
pipeline. Two artifacts:

  - write_pre_provenance_checkpoint(): a NON-CANONICAL snapshot of the
    complete in-memory extraction result (annotations_dict + entity
    registry), written immediately before the provenance-grounding gate
    runs, so a RuntimeError from that gate does not discard expensive
    LLM-extraction work that already completed. This is explicitly not a
    finished or validated output.

  - write_provenance_grounding_failures(): a diagnostic dump of every
    relation the provenance gate could not ground, written immediately
    before the gate raises, so a future failure is diagnosable without
    needing to re-run extraction (see the 2026-09 controlled-regeneration
    failure this was built to prevent a repeat of -- that run's two
    failing relations' actual content was unrecoverable because nothing
    like this existed at the time).

Both use the same atomic write/rename pattern: write to a temp file in the
SAME directory as the target, fsync it, then os.replace() it into place --
os.replace is atomic on both POSIX and Windows, so a process killed
mid-write can never leave a partially-written file at the target path, and
a reader never observes a half-written checkpoint.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

PRE_PROVENANCE_CHECKPOINT_STATUS = "PRE_PROVENANCE_CHECKPOINT"


def _atomic_write_json(path: Union[str, Path], payload: Dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(target))
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return str(target)


def write_pre_provenance_checkpoint(
    path: Union[str, Path],
    *,
    annotations_dict: Dict[str, Any],
    entities: Optional[Iterable[Any]] = None,
    run_id: Optional[str] = None,
    run_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persists the complete in-memory extraction result needed to replay
    provenance-completion and RDF serialization without another LLM call:
    `annotations_dict` (relations + structural nodes, exactly what
    fill_missing_relation_provenance/RDFSerializer consume) plus every
    entity from the global registry (via Entity.to_dict(), so
    Entity.from_dict() can reconstruct them later).

    Explicitly marked as NOT a finished output -- never expose this file
    as, or substitute it for, the final TTL/annotations output:
      canonical=False, validation_complete=False,
      status="PRE_PROVENANCE_CHECKPOINT"

    :param run_context: non-secret run/configuration identity only
        (dataset_id, pipeline_version, git_commit, timestamp, domain
        profile) -- same discipline as
        utils/semantic_quality_events.write_semantic_quality_report.
    """
    entity_dicts: List[Dict[str, Any]] = []
    for ent in entities or []:
        to_dict = getattr(ent, "to_dict", None)
        entity_dicts.append(to_dict() if callable(to_dict) else ent)

    payload = {
        "canonical": False,
        "validation_complete": False,
        "status": PRE_PROVENANCE_CHECKPOINT_STATUS,
        "run_id": run_id,
        "run_context": dict(run_context or {}),
        "annotations": annotations_dict,
        "entities": entity_dicts,
    }
    return _atomic_write_json(path, payload)


def load_pre_provenance_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a checkpoint written by write_pre_provenance_checkpoint. Does
    NOT itself validate `status`/`canonical` -- callers that intend to
    replay a checkpoint should check those fields themselves and refuse
    anything that isn't status == PRE_PROVENANCE_CHECKPOINT and
    canonical == False, to avoid ever mistaking a checkpoint for a
    finished output."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Fields deliberately allow-listed for the failure dump -- never the raw
# metadata dict wholesale, so nothing unexpected (were a future extractor
# to ever attach something sensitive to relation metadata) can leak in.
_SAFE_META_FIELDS = (
    "discourse_id", "field", "origin_key", "extractor", "evidence_quote", "evidence",
    "char_start", "char_end", "start_offset", "end_offset", "chunk_index", "chunk_index_unknown",
    "groundingStatus", "groundingMethod", "provenance_text_source", "evidence_spans",
)


def build_provenance_failure_record(
    *,
    identifier: str,
    section: str,
    rel_dict: Dict[str, Any],
    source_span: Optional[Any],
    target_span: Optional[Any],
    chunk_spans_count: int,
    failure_reason: str,
) -> Dict[str, Any]:
    meta = rel_dict.get("metadata", {}) or {}
    safe_meta = {k: meta.get(k) for k in _SAFE_META_FIELDS if k in meta}
    return {
        "id": identifier,
        "section": section,
        "subject": rel_dict.get("source") or rel_dict.get("source_id"),
        "predicate": rel_dict.get("relation"),
        "object": rel_dict.get("target") or rel_dict.get("target_id"),
        "metadata": safe_meta,
        "source_entity_span": list(source_span) if source_span is not None else None,
        "target_entity_span": list(target_span) if target_span is not None else None,
        "chunk_spans_count": int(chunk_spans_count),
        "grounding_evidence_available": {
            "explicit_relation_span": bool(meta.get("char_start") is not None or meta.get("start_char") is not None),
            "evidence_quote": bool(str(meta.get("evidence_quote") or meta.get("evidence") or "").strip()),
            "source_entity_span": source_span is not None,
            "target_entity_span": target_span is not None,
            "authoritative_text_used": meta.get("provenance_text_source") == "authoritative_source_text_lookup",
        },
        "failure_classification": failure_reason,
    }


def write_provenance_grounding_failures(
    path: Union[str, Path],
    *,
    failures: List[Dict[str, Any]],
    run_id: Optional[str] = None,
    run_context: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Written immediately before fill_missing_relation_provenance raises
    because unknown_rate exceeds the threshold -- one record per
    unresolved relation (see build_provenance_failure_record), never
    including credentials/secrets (only the allow-listed metadata fields
    above ever go in)."""
    payload = {
        "run_id": run_id,
        "run_context": dict(run_context or {}),
        "summary": dict(summary or {}),
        "failures": failures,
    }
    return _atomic_write_json(path, payload)
