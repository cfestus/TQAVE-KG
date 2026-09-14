"""
validation/span_integrity.py
-----------------------------
Source-span integrity validator: checks that an entity's recorded
(discourse_id, field, start_char, end_char) genuinely bounds its own
surface form within the exact source-field text used at extraction time.

Built out of the 2026-09 linker-validation preflight, which found (and
fixed, in serializers/rdf_serializer.py) a cross-mention consolidation bug
where a serialized span could describe a DIFFERENT mention than the one a
reader would pair it with, plus two still-open, narrower truncation cases
this validator is designed to keep catching even though they were not
repaired (see the preflight report for why: no single deterministic,
general root cause could be pinned down for those two within the bounded
scope of that pass).

This module has no dependency on any particular dataset's raw JSON shape --
callers supply the resolved source-field text (however they obtain it) and
the entity's own metadata dict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _normalize(s: str) -> str:
    return " ".join(str(s or "").strip().split()).lower()


def check_span_against_text(
    source_field_text: Optional[str],
    start_char: Any,
    end_char: Any,
    surface_form: Optional[str],
    *,
    preserved_original_mentions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Verifies source_field_text[start_char:end_char] against surface_form.

    Does NOT require exact equality when an explicitly recorded, justified
    normalization transformation exists: if the raw excerpt doesn't match
    `surface_form` but DOES match one of `preserved_original_mentions`
    (e.g. entity.metadata["aliases"], which EntityNormalizer.normalize_entity
    populates with the pre-normalization original label whenever
    normalization actually changed it -- see utils/entity_normalizer.py),
    the span is verified against that preserved original instead, and the
    result reports which one matched rather than silently treating a
    normalization-driven difference as a real defect.

    Returns one of these `status` values:
      "OK"                 -- excerpt matches surface_form exactly (normalized
                               for case/whitespace only).
      "OK_VIA_NORMALIZATION" -- excerpt matches a preserved original mention,
                               not the (transformed) surface_form directly.
      "MISMATCH"            -- excerpt matches neither.
      "OUT_OF_BOUNDS"       -- start/end don't fall within the source text.
      "MISSING_SOURCE_TEXT" -- source_field_text unavailable -- never
                               silently treated as a pass.
      "MISSING_SPAN"        -- start_char/end_char not usable integers.
    """
    if not source_field_text:
        return {"status": "MISSING_SOURCE_TEXT", "excerpt": None}

    try:
        s, e = int(start_char), int(end_char)
    except (TypeError, ValueError):
        return {"status": "MISSING_SPAN", "excerpt": None}

    if s < 0 or e <= s or e > len(source_field_text):
        return {"status": "OUT_OF_BOUNDS", "excerpt": None, "text_len": len(source_field_text)}

    excerpt = source_field_text[s:e]
    excerpt_norm = _normalize(excerpt)

    if surface_form is not None and excerpt_norm == _normalize(surface_form):
        return {"status": "OK", "excerpt": excerpt, "matched_against": "surface_form"}

    for original in preserved_original_mentions or []:
        if excerpt_norm == _normalize(original):
            return {
                "status": "OK_VIA_NORMALIZATION", "excerpt": excerpt,
                "matched_against": original,
                "note": "excerpt does not equal the (normalized) surface_form, but matches a "
                        "preserved pre-normalization original mention -- not treated as a defect.",
            }

    return {
        "status": "MISMATCH", "excerpt": excerpt,
        "expected_surface_form": surface_form,
        "expected_originals_checked": list(preserved_original_mentions or []),
    }


def check_entity_span_integrity(
    metadata: Dict[str, Any],
    resolve_source_text,
    *,
    primary_mention: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for an entity's own metadata dict, mirroring
    serializers/rdf_serializer.py's _primary_mention preference (the SAME
    mention record start_char/end_char/discourse_id/field all come from --
    see that module for why this consistency matters).

    :param metadata: the entity's metadata dict (surface_form, field,
        discourse_id, start_char, end_char, aliases, mentions, ...).
    :param resolve_source_text: callable(field, discourse_id) -> str | None,
        supplied by the caller (this module makes no assumption about
        where source text comes from).
    :param primary_mention: optional pre-computed primary mention dict
        ({"discourse_id","start_char","end_char","field"?}); if omitted,
        the top-level metadata scalars are used directly (the common,
        single-mention case).
    """
    if primary_mention is not None:
        field = primary_mention.get("field") or metadata.get("field")
        discourse_id = primary_mention.get("discourse_id")
        start_char = primary_mention.get("start_char")
        end_char = primary_mention.get("end_char")
    else:
        field = metadata.get("field")
        discourse_id = metadata.get("discourse_id")
        start_char = metadata.get("start_char")
        end_char = metadata.get("end_char")

    source_text = resolve_source_text(field, discourse_id)
    aliases = metadata.get("aliases")
    preserved = [a for a in aliases if isinstance(a, str)] if isinstance(aliases, list) else []

    result = check_span_against_text(
        source_text, start_char, end_char, metadata.get("surface_form"),
        preserved_original_mentions=preserved,
    )
    result["field"] = field
    result["discourse_id"] = discourse_id
    result["start_char"] = start_char
    result["end_char"] = end_char
    return result
