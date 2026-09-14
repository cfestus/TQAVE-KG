"""
utils/source_text_lookup.py
----------------------------
Deterministic (field, discourse_id) -> exact raw source-field text lookup,
built once from the raw input dataset JSON.

Same field semantics pipeline.py itself uses to build the text it feeds to
extraction (theme.description -> Theme[id].description, question.body ->
Question[id].body, answer.description -> Answer[id].description -- see
pipeline.py's _process_themes/_process_questions/_process_answers-style
methods around the "field": "..." literals), and the same semantics
validation/span_integrity.py's check_entity_span_integrity expects from its
injected `resolve_source_text(field, discourse_id)` callable.

This module intentionally has NO knowledge of provenance-grounding, span
math, or entity/relation structure -- it only answers "what is the exact
raw text for this (field, discourse_id)", or None if that isn't known.
Callers (fill_missing_relation_provenance, span-integrity checks, ad hoc
audits) decide what to do with that.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Maps a pipeline "field" literal to the (top-level JSON array key, text
# attribute) it is drawn from in the raw dataset.
FIELD_SOURCE_MAP = {
    "theme.description": ("Theme", "description"),
    "question.body": ("Question", "body"),
    "answer.description": ("Answer", "description"),
}


class SourceTextLookup:
    def __init__(self, dataset: Optional[Dict[str, Any]] = None):
        self._by_field: Dict[str, Dict[str, str]] = {}
        if dataset:
            self.load(dataset)

    def load(self, dataset: Dict[str, Any]) -> None:
        for field, (section, key) in FIELD_SOURCE_MAP.items():
            records = dataset.get(section)
            if not isinstance(records, list):
                continue
            table = self._by_field.setdefault(field, {})
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                rid = rec.get("id")
                if rid is None:
                    continue
                table[str(rid)] = str(rec.get(key) or "")

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "SourceTextLookup":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def get(self, field: Any, discourse_id: Any) -> Optional[str]:
        field_key = str(field or "").strip()
        did = str(discourse_id or "").strip()
        if not field_key or not did:
            return None
        table = self._by_field.get(field_key)
        if table is None:
            return None
        return table.get(did)

    # Makes an instance directly usable wherever a plain
    # `resolve_source_text(field, discourse_id)` callable is expected (see
    # validation/span_integrity.py::check_entity_span_integrity).
    def __call__(self, field: Any, discourse_id: Any) -> Optional[str]:
        return self.get(field, discourse_id)

    def __bool__(self) -> bool:
        return bool(self._by_field)
