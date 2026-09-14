"""
utils/helpers.py
----------------
Modernised utilities for the ETD-Hub Semantic Knowledge Graph Pipeline.
Fully compatible with:
- new Entity(uri, label, entity_type)
- new LLM-based entity extractor
- ontology-driven metadata
- canonical URI-based deduplication
"""

import gc
import time
import hashlib
import json
import logging
import requests
import re
import traceback
from typing import Callable, Any, List, Optional, Dict
from pathlib import Path
from rdflib import URIRef
from urllib.parse import urlsplit, urlunsplit, quote
from models import Relation  
from models import Entity
from models import ProcessingResult
from config import Config
from utils.wikidata_linker import WikidataLinker

logger = logging.getLogger(__name__)

# ============================================================
# SOURCE-IDENTITY (C.3)
# ============================================================
# The (field, discourse_id) pair, plus the offsets/chunk that only make
# sense *relative to that exact record*, form ONE atomic source-identity
# tuple. They must never be updated one key at a time from a different
# occurrence during a merge -- doing so produces an entity whose emitted
# field points at one record while its span/discourse_id point at another
# (or at a record that does not exist). Every merge path keeps the
# already-present tuple intact and records the other occurrence in
# metadata["mentions"] instead.
SOURCE_IDENTITY_SCALAR_KEYS = (
    "field",
    "source_field",
    "discourse_id",
    "source_doc_id",
    "origin_key",
    "start_char",
    "end_char",
    "chunk_index",
)


def dedup_mentions(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse mention records that describe the same occurrence. Identity
    is (field, discourse_id, start_char, end_char) -- a record synthesised
    from top-level scalars (no chunk_index) and the same occurrence's
    explicit mention (with chunk_index) are ONE occurrence, so chunk_index
    (prefer >= 0) and field (prefer non-empty) are merged rather than
    producing two near-duplicate rows."""
    out: Dict[tuple, Dict[str, Any]] = {}
    for m in mentions:
        if not isinstance(m, dict):
            continue
        did = str(m.get("discourse_id", "")).strip()
        try:
            s = int(m.get("start_char"))
            e = int(m.get("end_char"))
        except (TypeError, ValueError):
            continue
        field = str(m.get("field") or "").strip()
        key = (field, did, s, e)
        merged_key = None
        for existing_key in out:
            if existing_key == key:
                merged_key = existing_key
                break
            # a field-less record folds into the same occurrence that DOES
            # carry a field, and vice versa
            if existing_key[1:] == key[1:] and ("" in (existing_key[0], field)):
                merged_key = existing_key
                break
        if merged_key is None:
            out[key] = dict(m)
            continue
        cur = out[merged_key]
        if not str(cur.get("field") or "").strip() and field:
            cur["field"] = field
            new_key = (field, did, s, e)
            out[new_key] = out.pop(merged_key)
            cur = out[new_key]
        cur_ci = cur.get("chunk_index")
        new_ci = m.get("chunk_index")
        if (cur_ci is None or int(cur_ci) < 0) and new_ci is not None and int(new_ci) >= 0:
            cur["chunk_index"] = int(new_ci)
    return [out[k] for k in sorted(out.keys(), key=lambda t: (t[0], t[1], t[2], t[3]))]


def mention_from_source_identity(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build one mention record from an entity metadata dict's top-level
    source-identity scalars (field carried through), or None if there is no
    usable (discourse_id, start_char, end_char)."""
    if not isinstance(meta, dict):
        return None
    did_raw = meta.get("discourse_id")
    did = str(did_raw).strip() if did_raw is not None else ""
    if not did:
        return None
    try:
        start = int(meta.get("start_char"))
        end = int(meta.get("end_char"))
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None
    out: Dict[str, Any] = {
        "discourse_id": did,
        "start_char": int(start),
        "end_char": int(end),
    }
    field = str(meta.get("field") or meta.get("source_field") or "").strip()
    if field:
        out["field"] = field
    try:
        ci = int(meta.get("chunk_index"))
        if ci >= 0:
            out["chunk_index"] = int(ci)
    except (TypeError, ValueError):
        pass
    return out


# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_get(dictionary: dict, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(dictionary, dict):
            dictionary = dictionary.get(key, default)
        else:
            return default
    return dictionary


def normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def generate_hash(text: str) -> str:
    """Generate a short MD5 hash for text."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


# Personal-pronoun clause openers that, when immediately followed by a
# cognition/modal verb, mark a finite clause fragment rather than a nominal
# phrase (e.g. "I think there needs", "we need to") -- see
# is_finite_clause_fragment.
_CLAUSE_FRAGMENT_OPENERS = {"i", "we", "you", "he", "she", "they"}
_CLAUSE_FRAGMENT_VERBS = {
    "think", "thinks", "thought", "believe", "believes", "believed",
    "feel", "feels", "felt", "guess", "guesses", "suppose", "supposes",
    "assume", "assumes", "argue", "argues", "suggest", "suggests",
    "recommend", "recommends", "need", "needs", "want", "wants", "wish",
    "wishes", "should", "must", "would", "could", "will", "shall", "can",
    "know", "knows", "hope", "hopes", "require", "requires",
}


def is_finite_clause_fragment(text: str) -> bool:
    """True when `text` is shaped like a finite clause / speaker-intent
    fragment -- a personal-pronoun subject immediately followed by a
    cognition/modal verb (e.g. "I think there needs", "I believe that",
    "we need to") -- rather than a genuine named concept.

    Deliberately conservative and purely syntactic (subject-pronoun +
    finite-verb marks the start of a full clause, not a nominal phrase):
    only that specific two-token opening shape is rejected. A multi-word
    named concept that happens to contain a verb-like word elsewhere (e.g.
    "Algorithmic Decision-Making", "Automated Content Moderation") is left
    untouched, since its first token is not a personal pronoun.
    """
    norm = " ".join(str(text or "").strip().lower().split())
    if not norm:
        return False
    tokens = norm.split()
    if len(tokens) < 2:
        return False
    if tokens[0] not in _CLAUSE_FRAGMENT_OPENERS:
        return False
    return tokens[1] in _CLAUSE_FRAGMENT_VERBS


def batch_iterator(items: List[Any], batch_size: int):
    """Yield fixed-size batches."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def cleanup_memory():
    """Force GC + clear GPU."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def validate_input_json(data: dict) -> bool:
    """Ensure minimal input structure exists."""
    # Accept both modern plural keys and legacy singular keys.
    expected_aliases = {
        "themes": ["themes", "Theme"],
        "questions": ["questions", "Question"],
        "answers": ["answers", "Answer"],
    }
    for canonical, aliases in expected_aliases.items():
        if not any(alias in data for alias in aliases):
            logger.warning("Input JSON key missing: %s", canonical)
    return True


# ============================================================
# RETRY WRAPPER
# ============================================================

def safe_process_with_retry(
    item: Any,
    process_func: Callable,
    max_retries: int = None
) -> ProcessingResult:
    """Perform processing with retries and always return ProcessingResult."""
    max_retries = max_retries or Config.MAX_RETRIES

    def _coerce_processing_result(result: Any) -> ProcessingResult:
        if isinstance(result, ProcessingResult):
            return result

        if isinstance(result, dict):
            entities_raw = result.get("entities", []) or []
            relations_raw = result.get("relations", []) or []
            metadata_raw = result.get("metadata", {}) or {}
            errors_raw = result.get("errors", []) or []

            entities: List[Entity] = []
            for e in entities_raw:
                if isinstance(e, Entity):
                    entities.append(e)
                elif isinstance(e, dict):
                    try:
                        entities.append(Entity.from_dict(e))
                    except Exception:
                        continue

            relations: List[Relation] = []
            for r in relations_raw:
                if isinstance(r, Relation):
                    relations.append(r)
                elif isinstance(r, dict):
                    try:
                        relations.append(Relation.from_dict(r))
                    except Exception:
                        continue

            metadata = metadata_raw if isinstance(metadata_raw, dict) else {"raw_metadata": metadata_raw}
            errors = errors_raw if isinstance(errors_raw, list) else [str(errors_raw)]

            return ProcessingResult(
                entities=entities,
                relations=relations,
                metadata=metadata,
                errors=errors,
            )

        raise TypeError(f"Expected ProcessingResult or dict, got {type(result)}")

    last_exc: Optional[Exception] = None
    last_tb: str = ""
    for attempt in range(max_retries):
        try:
            result = process_func(item)
            return _coerce_processing_result(result)
        except Exception as e:
            last_exc = e
            last_tb = traceback.format_exc()
            logger.warning(
                "safe_process_with_retry failed on attempt %d/%d: %s",
                attempt + 1,
                max_retries,
                str(e)[:120],
            )

            if attempt == max_retries - 1:
                break

            time.sleep(2 ** attempt)

    return ProcessingResult(
        entities=[],
        relations=[],
        metadata={
            "error": str(last_exc)[:200] if last_exc else "Max retries exceeded",
            "exception_type": type(last_exc).__name__ if last_exc else "UnknownError",
            "retries": int(max_retries),
            "traceback": last_tb,
            "processing_failed": True,
            "item_id": item.get("id", "unknown") if isinstance(item, dict) else getattr(item, "id", "unknown"),
        },
        errors=[str(last_exc)] if last_exc else ["Max retries exceeded"],
    )


# ============================================================
# GLOBAL ENTITY REGISTRY (CRITICAL COMPONENT)
# ============================================================

class GlobalEntityRegistry:
    """
    Canonical entity registry across entire pipeline.
    NEW RULES:
    - Deduplicate entities by **entity.uri** (canonical)
    - Preserve entity.label + entity.entity_type
    - Merge metadata and keep highest confidence
    """

    def __init__(self):
        # Map canonical URI → Entity instance
        self.entities: Dict[str, Entity] = {}

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
        # C.3: a mention carries its own field, so (field, discourse_id)
        # survives a merge as one pair rather than field drifting to a
        # top-level scalar the span no longer matches.
        field = str(mention.get("field") or mention.get("source_field") or "").strip()
        if field:
            out["field"] = field
        try:
            ci = int(mention.get("chunk_index"))
            if ci >= 0:
                out["chunk_index"] = int(ci)
        except (TypeError, ValueError):
            pass
        return out

    def _merge_mentions(self, existing_meta: Dict[str, Any], incoming_meta: Dict[str, Any]) -> None:
        merged: List[Dict[str, Any]] = []
        # Seed from both explicit mention buckets AND each side's top-level
        # source-identity scalars -- an entity minted through the semantic-
        # assertion path carries only the scalars (no "mentions" list), so
        # without this its occurrence would be silently lost on merge.
        for meta in (existing_meta, incoming_meta):
            bucket = meta.get("mentions")
            if isinstance(bucket, list):
                for item in bucket:
                    norm = self._normalize_mention(item)
                    if norm is not None:
                        merged.append(norm)
            scalar_mention = mention_from_source_identity(meta)
            if scalar_mention is not None:
                merged.append(scalar_mention)
        if not merged:
            return
        existing_meta["mentions"] = dedup_mentions(merged)

    def _merge_source_occurrences(self, existing_meta: Dict[str, Any], incoming_meta: Dict[str, Any]) -> None:
        """Preserve span-less STRUCTURAL source occurrences (e.g. a dataset
        Expert / document record) across a canonical-entity merge.

        Unlike metadata["mentions"], these carry no character span, so
        _normalize_mention / mention_from_source_identity (both span-gated)
        would silently drop them. They are kept in their own bucket, keyed
        on the atomic (field, discourse_id, origin_key) provenance tuple:
        a later occurrence from a different field never displaces an
        earlier one -- every distinct occurrence is retained so a
        downstream consumer can materialise its own provenance node."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for meta in (existing_meta, incoming_meta):
            bucket = meta.get("source_occurrences") if isinstance(meta, dict) else None
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                if not isinstance(item, dict):
                    continue
                field = str(item.get("field") or item.get("source_field") or "").strip()
                did = str(item.get("discourse_id") or item.get("source_doc_id") or "").strip()
                okey = str(item.get("origin_key") or "").strip()
                if not field or not (did or okey):
                    continue
                key = (field, did, okey)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(dict(item))
        if merged:
            existing_meta["source_occurrences"] = merged

    def _merge_scalar_metadata(self, existing_meta: Dict[str, Any], incoming_meta: Dict[str, Any]) -> None:
        """Non-mention scalar/dict/list metadata merge. The atomic
        source-identity tuple (C.3) is NEVER overwritten key-by-key: once a
        coherent tuple is present, a differing occurrence only lands in
        metadata["mentions"] (via _merge_mentions, already called by the
        caller). metadata["source_occurrences"] is likewise owned by
        _merge_source_occurrences and never touched here."""
        for k, v in (incoming_meta or {}).items():
            if k in ("mentions", "source_occurrences"):
                continue
            if k in SOURCE_IDENTITY_SCALAR_KEYS:
                if not str(existing_meta.get(k) or "").strip():
                    existing_meta[k] = v
                continue
            if k not in existing_meta:
                existing_meta[k] = v
                continue
            cur = existing_meta.get(k)
            if isinstance(cur, dict) and isinstance(v, dict):
                merged = dict(cur)
                merged.update(v)
                existing_meta[k] = merged
            elif isinstance(cur, list) and isinstance(v, list):
                seen = set()
                out = []
                for item in cur + v:
                    marker = repr(item)
                    if marker not in seen:
                        seen.add(marker)
                        out.append(item)
                existing_meta[k] = out
            else:
                existing_meta[k] = v

    # -----------------------------------------
    # Registration
    # -----------------------------------------
    def get_or_create(self, entity: Entity) -> Entity:
        """
        Deduplicate using the canonical entity URI.
        Merge confidence + metadata when entity already exists.
        """
        uri = entity.uri

        if uri in self.entities:
            existing = self.entities[uri]

            # Higher confidence wins
            if entity.confidence > existing.confidence:
                existing.confidence = entity.confidence

            # Production-safe metadata merge: merge dicts, extend lists, else
            # overwrite -- EXCEPT the atomic source-identity tuple (C.3),
            # which _merge_scalar_metadata never breaks apart.
            self._merge_mentions(existing.metadata, entity.metadata or {})
            self._merge_source_occurrences(existing.metadata, entity.metadata or {})
            self._merge_scalar_metadata(existing.metadata, entity.metadata or {})

            # Keep label if longer / more informative
            if len(entity.label) > len(existing.label):
                existing.label = entity.label

            # Keep entity_type if new one seems better (same logic)
            if entity.entity_type != existing.entity_type:
                existing.entity_type = entity.entity_type

            # Keep entity.name / entity.type for compatibility
            if hasattr(entity, "name"):
                existing.name = entity.name
            if hasattr(entity, "type"):
                existing.type = entity.type

            return existing

        # NEW entity
        self.entities[uri] = entity

        # backwards compatibility:
        if not hasattr(entity, "name"):
            entity.name = entity.label
        if not hasattr(entity, "type"):
            entity.type = entity.entity_type

        return entity

    # -----------------------------------------
    # Retrieval
    # -----------------------------------------
    def get_all_entities(self) -> List[Entity]:
        """
        Return all Entity objects regardless of whether self.entities is a dict or list.
        Production-safe for streaming refactors.
        """
        store = getattr(self, "entities", None)

        if isinstance(store, dict):
            return list(store.values())

        if isinstance(store, list):
            return list(store)

        return []


    def get_count(self) -> int:
        return len(self.entities)


    def replace_entity(self, old_uri: str, new_uri: str):
        """
        Redirect all references from old_uri → new_uri.
        Merge metadata, confidence, type, label.
        """
        if old_uri == new_uri:
            return

        if old_uri not in self.entities:
            return

        old_ent = self.entities[old_uri]

        if new_uri not in self.entities:
            # If canonical doesn't yet exist, promote old entity to canonical
            self.entities[new_uri] = old_ent
            del self.entities[old_uri]
            return

        # Merge into canonical
        canon = self.entities[new_uri]

        # Keep best confidence
        if old_ent.confidence > canon.confidence:
            canon.confidence = old_ent.confidence

        # Choose richer label
        if len(old_ent.label) > len(canon.label):
            canon.label = old_ent.label

        # Merge metadata -- keep the canonical entity's coherent
        # source-identity tuple (C.3); the redirected entity's own
        # occurrence is preserved in canon.metadata["mentions"].
        self._merge_mentions(canon.metadata, old_ent.metadata or {})
        self._merge_scalar_metadata(canon.metadata, old_ent.metadata or {})

        # Merge entity_type if different
        if old_ent.entity_type != canon.entity_type:
            canon.entity_type = old_ent.entity_type

        # Remove old
        del self.entities[old_uri]


# ============================================================
# COREFERENCE RESOLVER (placeholder)
# ============================================================

class CoreferenceResolver:
    """Optional text coreference resolver (disabled by default)."""

    def __init__(self):
        self.enabled = False

    def resolve(self, text: str) -> str:
        return text


# ============================================================
# FILE UTILITIES
# ============================================================

def create_output_directory(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", output_dir)


def save_json(data: dict, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON: %s", filepath.name)


def load_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# URI SANITISATION
# ============================================================


def canonicalize_local_name(label: str) -> str:
    """
    Convert labels into RDF-safe camelCase local names.
    Removes punctuation and spaces without percent-encoding.
    """
    if not label:
        return "unnamed"

    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", label)
    parts = cleaned.strip().split()  # split on spaces

    if not parts:
        return "unnamed"

    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])

def make_uri_safe(iri: str) -> URIRef:
    """Percent-encode illegal characters for RDF stores."""
    s = str(iri).strip()
    parts = urlsplit(s)
    path = quote(parts.path, safe="/:@-._~")
    query = quote(parts.query, safe="/:@-._~&=?")
    fragment = quote(parts.fragment, safe="/:@-._~")
    return URIRef(urlunsplit((parts.scheme, parts.netloc, path, query, fragment)))

def safe_uri(iri: str) -> URIRef:
    """
    Convert arbitrary strings into a valid ETD URIRef.
    Uses Config.NAMESPACES['etd'] as base IRI unless input is already absolute.
    """
   
    if iri is None:
        raise ValueError("safe_uri() received None")

    iri = str(iri).strip()

    # If already an absolute HTTP IRI, just percent-encode safely
    if iri.startswith("http://") or iri.startswith("https://"):
        return make_uri_safe(iri)

    # Otherwise treat as an ETD-local identifier
    base = Config.NAMESPACES["etd"]

    # Replace unsafe characters before percent-encoding
    cleaned = re.sub(r"[^A-Za-z0-9\-_]", "_", iri)  # strict safe fragment

    return make_uri_safe(base + cleaned)


def expand_curie(curie: str) -> str:
    """
    Expand CURIE like 'prov:wasAttributedTo' → full URI.
    Falls back to CURIE as-is if prefix unknown.
    """
    if ":" in curie:
        prefix, local = curie.split(":", 1)
        base = Config.NAMESPACES.get(prefix)
        if base:
            return f"{base}{local}"
    return curie

def is_acronym(text: str) -> bool:
    """
    Returns True if the string is an acronym:
    - All uppercase letters (e.g., “AI”, “NLP”, “FBI”)
    - Or uppercase with digits (e.g., “GPT4”, “COVID19”)
    """
    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) <= 1:
        return False

    # No spaces, contains only A–Z or digits
    return text.isupper() and text.replace("-", "").replace("_", "").isalnum()


def safe_hash(value: str) -> str:
    """
    Generate a stable, URL-safe hash for KG identifiers.
    """
    if value is None:
        value = ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def make_relation(
    source_uri: str,
    predicate: str,
    target_uri: str,
    *,
    discourse_id: Optional[str] = None,
    evidence: str = "",
    extractor: str = "pipeline",
    confidence: float = 0.0,
    context_source: str = "unknown",
    is_inferred: bool = False,
) -> Relation:
    """
    Production-grade relation builder:
    - stable URI hash
    - guaranteed metadata provenance
    """
    raw_key = f"{discourse_id}|{source_uri}|{predicate}|{target_uri}"
    rel_hash = safe_hash(raw_key)
    uri = f"{Config.NAMESPACES['etd']}rel_{rel_hash}"

    meta = {
        "discourse_id": discourse_id,
        "extractor": extractor,
        "context_source": context_source,
        "evidence": evidence or "",
    }

    return Relation(
        uri=uri,
        source=source_uri,
        relation=predicate,
        target=target_uri,
        confidence=float(confidence or 0.0),
        is_inferred=bool(is_inferred),
        metadata=meta,
    )

