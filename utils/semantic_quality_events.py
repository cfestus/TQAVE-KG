"""Shared semantic-quality instrumentation event recorder.

`EnhancedSemanticPipeline` (drop/suppression/correction decisions made
before RDF serialization) and `RDFSerializer` (drop/correction/
consolidation/invariant-enforcement decisions made during and after graph
construction) each hold their own `quality_events: List[Dict[str, Any]]`
list and append to it via `record_event()`. The two lists are merged and
persisted by `write_semantic_quality_report()`, called once per run,
after serialization, so drop history survives independently of whether
the dropped fact ever reached the TTL (see module docstring intent: a
relation dropped before serialization has no RDF trace at all, so its
audit history must live here, not in the graph).
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional


def _clip(value: Any, limit: int = 300) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def record_event(
    sink: List[Dict[str, Any]],
    *,
    category: str,
    action: str,
    discourse_id: Optional[str] = None,
    source_relation_id: Optional[str] = None,
    subject_surface: Optional[str] = None,
    predicate_surface: Optional[str] = None,
    object_surface: Optional[str] = None,
    evidence_text: Optional[str] = None,
    reason: str = "",
    replacement_relation_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Append one structured quality event to `sink` and return it.

    `category` is a stable, code-reviewable identifier for the
    mechanism that fired (e.g. "self_reference_drop",
    "endpoint_surface_mismatch_recovered"); `action` is one of
    "dropped" / "corrected" / "suppressed" / "recovered" / "superseded"
    / "reused" / "merged". Only discourse ids, relation ids, and short
    surface/evidence text already present in the annotations/graph are
    ever recorded here -- no credentials or external-system data of any
    kind flow through this function, so there is nothing to redact.
    """
    event: Dict[str, Any] = {
        "category": category,
        "action": action,
        "discourse_id": discourse_id,
        "source_relation_id": source_relation_id,
        "subject_surface": _clip(subject_surface),
        "predicate_surface": predicate_surface,
        "object_surface": _clip(object_surface),
        "evidence_text": _clip(evidence_text),
        "reason": reason,
        "replacement_relation_ids": list(replacement_relation_ids or []),
    }
    sink.append(event)
    return event


def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category = Counter(str(e.get("category") or "unknown") for e in events)
    by_action = Counter(str(e.get("action") or "unknown") for e in events)
    by_category_action: Counter = Counter(
        (str(e.get("category") or "unknown"), str(e.get("action") or "unknown")) for e in events
    )
    return {
        "total_events": len(events),
        "by_category": dict(sorted(by_category.items())),
        "by_action": dict(sorted(by_action.items())),
        "by_category_and_action": {
            f"{cat}|{act}": count for (cat, act), count in sorted(by_category_action.items())
        },
    }


def write_semantic_quality_report(
    events: List[Dict[str, Any]],
    output_path: str,
    *,
    run_id: Optional[str] = None,
    run_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist the full event list plus a summary to `output_path` as JSON.

    Deliberately independent of the TTL: a dropped relation/entity has
    no RDF trace by definition, so this report is the only place its
    audit history can be recovered from after the run completes. Written
    unconditionally whenever a graph was successfully built, regardless
    of whether SHACL validation, GraphDB upload, or any other later
    step subsequently fails -- see the call site in main.py, placed
    immediately after RDFSerializer construction rather than after those
    later steps, precisely so a failing run still leaves this audit
    trail behind.

    `run_context` carries only non-secret run/configuration identity
    (dataset id, pipeline version, git commit, domain profile name,
    generation timestamp) -- never credentials, API keys, or raw config
    dumps; callers must not pass anything else through it.
    """
    report: Dict[str, Any] = {
        "run_id": run_id,
        "run_context": run_context or {},
        "summary": summarize_events(events),
        "events": events,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
