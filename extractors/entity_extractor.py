"""
LLM-based Entity Extraction (OpenAI JSON Mode Version)
ETD-Hub - Semantic Knowledge Graph Pipeline
"""

import json
import time
import hashlib
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

from config import Config
from models import Entity
from utils.helpers import is_finite_clause_fragment
import logging

logger = logging.getLogger(__name__)


def safe_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:15]


class EntityJSONParseError(ValueError):
    """Raised when LLM entity-extraction output cannot be parsed as JSON,
    even after repair attempts. Treated as a retryable failure -- a single
    malformed response is often a one-off model hiccup."""


class LLMEntityExtractor:

    # OpenAI Structured Outputs (response_format={"type": "json_schema", ...}).
    # Stronger than plain JSON mode: the API constrains token generation to
    # match this schema, so a syntactically-valid completion is guaranteed
    # to have this exact shape (strict mode requires every property listed
    # in "required" -- optional fields stay as plain "" per the existing
    # prompt convention, so this changes no field semantics). It does NOT
    # guarantee a *complete* completion if max_tokens is hit mid-generation,
    # which is why a repair fallback still exists for the truncated case.
    _RESPONSE_FORMAT: Dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "surface_form": {"type": "string"},
                                "resolved_label": {"type": "string"},
                                "entity_type": {"type": "string"},
                                "ontology_uri": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": [
                                "surface_form",
                                "resolved_label",
                                "entity_type",
                                "ontology_uri",
                                "confidence",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["entities"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        ontology_terms: Dict[str, str],
        model: str = None,
        max_retries: int = 3,
        min_confidence: float = 0.35
    ):
        self.ontology_terms = ontology_terms
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        self.model = model or Config.LLM_MODEL
        self.retry_base_delay = 0.5
        self.retry_max_delay = 4.0

        # Use the same key loading strategy as the rest of the pipeline
        Config.load_llm_api_key()
        self.client = OpenAI(api_key=getattr(Config, "LLM_API_KEY", None))

        self.system_prompt = """
You are an expert AI entity extractor. You MUST return ONLY valid JSON.

Output Example:
{
  "entities": [
    {
      "surface_form": "COMPAS",
      "resolved_label": "",
      "entity_type": "Dataset",
      "ontology_uri": "aipo:Dataset",
      "confidence": 0.92
    }
  ]
}

surface_form MUST always be the verbatim mention text copied exactly from Text -- it is the
literal provenance span and must never be altered, even when resolved_label is set.

resolved_label is OPTIONAL. Set it only when surface_form in Text is a pronoun, shorthand
reference, or abbreviation whose full/canonical identity is established in Context (e.g.
surface_form "it" or "the tool" in Text, with Context establishing that referent's full name).
Leave resolved_label empty ("") when surface_form is already the specific, canonical form --
do not duplicate surface_form into resolved_label.

You may be given a "Context" section (the surrounding source field) in addition to the "Text"
section (the specific span to extract entities from). Extract entities only for mentions that
appear in Text. Use Context ONLY to resolve what a pronoun, shorthand reference, or abbreviation
in Text refers to -- never to extract an entity that appears only in Context and is not
mentioned, even indirectly via pronoun or shorthand, in Text.
"""

    # =============================================================
    # Public API
    # =============================================================
    def extract(self, text: str, doc_id: Optional[str] = None, context: Optional[str] = None) -> List[Entity]:
        raw = self._call_with_retries(text, context=context)
        cleaned = self._clean_entities(raw)
        converted = self._convert_to_entities(cleaned, text, doc_id)
        deduped = self._deduplicate(converted)
        return deduped

    # =============================================================
    # Retry wrapper
    # =============================================================
    def _call_with_retries(self, text: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_model(text, context=context)
            except Exception as e:
                last_error = e
                retryable = self._is_transient_error(e)
                logger.warning(
                    f"[LLM Entity Extraction] attempt {attempt}/{self.max_retries} failed "
                    f"(retryable={retryable}): {e}"
                )
                if not retryable:
                    logger.error("[LLM Entity Extraction] non-transient failure; not retrying.")
                    raise RuntimeError(f"LLM entity extraction failed (non-transient): {e}") from e
                if attempt == self.max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(f"[LLM Entity Extraction] retrying in {delay:.2f}s")
                time.sleep(delay)
        if last_error is not None:
            logger.error(f"[LLM Entity Extraction] final failure after {self.max_retries} attempts: {last_error}")
            raise RuntimeError(
                f"LLM entity extraction failed after {self.max_retries} attempts: {last_error}"
            ) from last_error
        raise RuntimeError("LLM entity extraction failed with unknown error")

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, min(0.25, delay * 0.25))
        return delay + jitter

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            code = getattr(exc, "status_code", None)
            return code == 429 or (isinstance(code, int) and 500 <= code <= 599)
        if isinstance(exc, EntityJSONParseError):
            # A single malformed/truncated completion is usually a one-off
            # model hiccup -- worth the existing retry budget rather than
            # failing the whole extraction outright.
            return True
        return False

    # =============================================================
    # Direct OpenAI JSON-mode call
    # =============================================================
    def _build_prompt(self, text: str, context: Optional[str] = None) -> str:
        context_block = ""
        norm_context = str(context or "").strip()
        if norm_context and norm_context != str(text or "").strip():
            context_block = f"""
Context (full source field -- use ONLY to resolve what a pronoun, shorthand reference, or
abbreviation in Text refers to; do NOT extract entities that appear only in Context):
{norm_context}
"""

        return f"""
Extract all entities from Text that could serve as meaningful semantic relation endpoints.

Requirements:
- Be exhaustive: extract every distinct entity explicitly supported by Text, not just the
  single most prominent one per sentence.
- When Text lists or coordinates multiple entities (e.g. "X or Y", "X, Y, and Z"), extract
  EACH member of the list as its own separate entity, not only the first one.
- Preserve the most specific local noun phrase that appears verbatim in Text.
- Prefer a compound phrase (e.g. "AI recruitment tool", "historical workforce data") over a
  shorter generic form (e.g. "AI", "data") ONLY when that more specific compound phrase is
  itself present in Text for the same referent. When Text expresses the actual subject
  or object of a statement using only a short or generic term (a bare acronym, or a general
  noun such as "the system", "the model", "the algorithm", "the AI") and no more specific
  compound phrase for that same referent appears locally, still extract that short form as its
  own entity. Never omit a sentence's actual grammatical subject or object just because it is
  short or generic -- the "prefer compound phrases" rule governs which surface form to use, it
  must never cause an entity to be dropped entirely.
- Include explicit systems/models/tools, datasets/data phrases, populations/groups, domains,
  harms, metrics, and concept objects when they are locally stated.
- Do NOT collapse a specific phrase to a generic stand-in if the specific phrase is present.
- surface_form MUST be copied exactly from Text.
- Do NOT invent entities that are not explicitly written in Text.
- Do NOT return pronouns by themselves unless they are part of a longer named phrase, UNLESS
  Context resolves that pronoun to a specific referent -- in that case extract the pronoun's
  literal text as surface_form and put the resolved canonical name in resolved_label.

Ontology classes (JSON):
{json.dumps(self.ontology_terms, indent=2)}

{context_block}

Text (the target span to extract entities from; surface_form MUST come from here):
{text}
"""

    def _call_model(self, text: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(text, context=context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
            response_format=self._RESPONSE_FORMAT,
        )

        content = response.choices[0].message.content
        data = self._parse_entities_json(content)

        if not isinstance(data, dict):
            return []

        ents = data.get("entities", [])
        if not isinstance(ents, list):
            return []

        # ensure list[dict]
        clean = [x for x in ents if isinstance(x, dict)]
        return clean

    # =============================================================
    # JSON parsing + repair
    # =============================================================
    def _parse_entities_json(self, content: Any) -> Dict[str, Any]:
        """Parse LLM entity-extraction output as JSON, tolerating minor
        formatting defects: markdown code fences, leading/trailing
        commentary text, trailing commas, and a truncated tail (e.g. the
        completion was cut off mid-entity by max_tokens).

        Every repair only ever narrows the text to a complete, valid JSON
        subset of what the model actually returned -- nothing is ever
        invented or filled in. If the last entity in a truncated response
        is itself incomplete, it is dropped, not completed.

        Raises EntityJSONParseError if no repair yields valid JSON.
        """
        raw = content if isinstance(content, str) else ""
        if not raw.strip():
            logger.error("[LLM Entity Extraction] model returned empty content")
            raise EntityJSONParseError("empty response content")

        # 1) Strict parse -- the common case, especially with json_schema
        # structured outputs.
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            first_error = exc

        # 2) Slice to the outermost {...} span. This also transparently
        # strips markdown code fences (```json ... ```) and any leading or
        # trailing commentary, since those characters fall outside the
        # first '{' / last '}' boundary.
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        sliced = raw[first_brace:last_brace + 1] if first_brace != -1 and last_brace > first_brace else raw

        # 3) Remove trailing commas immediately before a closing } or ].
        no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", sliced)

        for candidate in (sliced, no_trailing_commas):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        # 4) Truncated tail: cut back to the last point at which a JSON
        # object/array fully closed (dropping only the incomplete trailing
        # entity) and close whatever brackets are still open at that point.
        for candidate in (no_trailing_commas, sliced, raw):
            repaired = self._repair_truncated_json(candidate)
            if repaired is None:
                continue
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue

        preview_len = 200
        preview = raw[:preview_len] + ("...<truncated>" if len(raw) > preview_len else "")
        logger.error(
            "[LLM Entity Extraction] unrecoverable malformed JSON "
            "(len=%d, parse_error=%s): %r",
            len(raw), first_error, preview,
        )
        raise EntityJSONParseError(f"unrecoverable malformed JSON (len={len(raw)}): {first_error}")

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[str]:
        """Cut `text` back to the last position at which a JSON object or
        array fully closed, then append closers for whatever brackets are
        still open at that point. Returns None when there is no safe cut
        point (e.g. genuinely mismatched brackets) or nothing to repair."""
        stack: List[str] = []
        in_string = False
        escape = False
        last_safe_cut: Optional[int] = None
        last_safe_stack: Optional[List[str]] = None

        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    return None
                opener = stack.pop()
                if (opener, ch) not in (("{", "}"), ("[", "]")):
                    return None
                last_safe_cut = i + 1
                last_safe_stack = list(stack)

        if last_safe_cut is None or not last_safe_stack:
            return None
        if last_safe_cut == len(text) and not stack and not in_string:
            return None  # already fully balanced -- nothing to repair here

        closing = "".join("}" if c == "{" else "]" for c in reversed(last_safe_stack))
        return text[:last_safe_cut] + closing

    # =============================================================
    # Confidence + ontology cleaning
    # =============================================================
    def _clean_entities(self, ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for e in ents:
            raw_conf = e.get("confidence", 0.0)
            try:
                conf = float(raw_conf) if raw_conf is not None and str(raw_conf).strip() != "" else 0.0
            except Exception:
                conf = 0.0

            conf = max(0.0, min(1.0, conf))

            if conf < self.min_confidence:
                continue

            # Root-cause fix (manual semantic-QA pass, category C): reject
            # finite clause / speaker-intent fragments (e.g.
            # "I Think There Needs", "I believe that") before they ever
            # become a candidate Entity -- checked on both the literal
            # mention and the LLM's own "resolved" label, since either can
            # carry the fragment shape.
            surface_form = e.get("surface_form", "")
            resolved_label = e.get("resolved_label", "")
            if is_finite_clause_fragment(surface_form) or is_finite_clause_fragment(resolved_label):
                continue

            out.append(e)
        return out

    # =============================================================
    # Convert JSON entity objects to Entity model
    # =============================================================
    def _convert_to_entities(
        self,
        ents: List[Dict[str, Any]],
        text: str,
        doc_id: Optional[str]
    ) -> List[Entity]:
        output: List[Entity] = []

        for e in ents:
            surface = e.get("surface_form", "")
            surface = surface.strip() if isinstance(surface, str) else ""
            if not surface:
                continue  # avoid empty needle causing infinite/huge matches

            resolved_label = e.get("resolved_label", "")
            resolved_label = resolved_label.strip() if isinstance(resolved_label, str) else ""
            # The resolved (context-derived) name is used for identity/display only when it
            # actually differs from the literal mention; otherwise treat as unresolved.
            #
            # NOTE (manual semantic-QA pass, category E -- cross-discourse
            # over-canonicalisation): an earlier version of this fix tried
            # to require resolved_label to be textually grounded in the
            # local `text` chunk before trusting it, to stop a generic
            # mention like "the prediction" from resolving to an unrelated
            # specific concept ("Performativity Of Predictions") purely on
            # the LLM's own say-so. That check was reverted: `text` here is
            # only the current chunk, not the full document, so it also
            # rejected entirely legitimate resolutions to an antecedent
            # established earlier in the SAME document but outside this
            # chunk (e.g. "the tool" -> a name introduced two paragraphs
            # earlier) -- there is no way to tell those two cases apart
            # from this function's local view of the text. The actual
            # fix for category E is downstream, in
            # utils/canonical_linker.py's cross-discourse fuzzy-merge gate
            # (_is_bare_generic_label), which correctly has visibility into
            # both entities being compared.
            is_resolved = bool(resolved_label) and resolved_label.lower() != surface.lower()
            label = resolved_label if is_resolved else surface

            etype = e.get("entity_type", "Entity") or "Entity"
            ont = e.get("ontology_uri", "") or ""
            conf = float(e.get("confidence", 0.0) or 0.0)

            # --- locate ALL spans in this chunk/text ---
            # Provenance MUST stay truthful: always search for the literal surface_form
            # (the actual mention text), never the resolved label, and only within the
            # target chunk `text` -- never any wider context.
            spans = self._surface_spans(text, surface)
            if not spans:
                raw_start = self._coerce_nonneg_int(e.get("start_char"))
                raw_end = self._coerce_nonneg_int(e.get("end_char"))
                if raw_start is not None and raw_end is not None and raw_end > raw_start:
                    spans = [(int(raw_start), int(raw_end))]

            # choose best span (first match) but store all
            start_char, end_char = spans[0] if spans else (None, None)

            # --- entity URI (stable identity) ---
            # Keyed on the resolved label when available so that different mentions of the
            # same referent (e.g. a named entity and a later pronoun/shorthand resolved to
            # it) converge on the same identity, exactly as a directly-named mention would.
            norm_label = " ".join(label.lower().split())
            identity_key = f"{norm_label}|{etype}|{ont}"

            uri = f"{Config.NAMESPACES['etd']}{safe_hash(identity_key)}"

            # --- metadata ---
            metadata = {
                "doc_id": doc_id,
                "source": "llm",
                "surface_form": surface,
                "start_char": start_char,
                "end_char": end_char,
                "spans": spans,
                "ontology_uri": ont,
                "mention_resolved_from_context": is_resolved,
            }
            if is_resolved:
                metadata["resolved_label"] = resolved_label

            ent_obj = Entity(
                uri=uri,
                label=label,
                entity_type=etype,
                confidence=conf,
                metadata=metadata
            )

            output.append(ent_obj)

        return output

    @staticmethod
    def _coerce_nonneg_int(value: Any) -> Optional[int]:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        return iv if iv >= 0 else None

    @staticmethod
    def _surface_pattern(surface: str) -> Optional[re.Pattern]:
        tokens = [re.escape(tok) for tok in re.findall(r"[A-Za-z0-9]+(?:['/-][A-Za-z0-9]+)*", str(surface or ""))]
        if not tokens:
            return None
        joined = r"(?:\s+|[-/]|['’]s\s+|['’])".join(tokens)
        return re.compile(rf"(?<!\w){joined}(?!\w)", flags=re.IGNORECASE)

    def _surface_spans(self, text: str, surface: str) -> List[Tuple[int, int]]:
        raw_text = str(text or "")
        raw_surface = str(surface or "").strip()
        if not raw_text or not raw_surface:
            return []

        spans: List[Tuple[int, int]] = []
        lower_text = raw_text.lower()
        needle = raw_surface.lower()
        pos = 0
        while True:
            start = lower_text.find(needle, pos)
            if start == -1:
                break
            spans.append((start, start + len(raw_surface)))
            pos = start + len(raw_surface)
        if spans:
            return spans

        pattern = self._surface_pattern(raw_surface)
        if pattern is None:
            return []
        return [(int(match.start()), int(match.end())) for match in pattern.finditer(raw_text)]

    # =============================================================
    # Deduplicate entities by label
    # =============================================================
# =============================================================
# Deduplicate entities (strong key, confidence-aware)
# =============================================================
    def _deduplicate(self, ents: List[Entity]) -> List[Entity]:
        seen = {}

        for e in ents:
            if e is None:
                continue

            label = (e.label or "").strip().lower()
            etype = getattr(e, "entity_type", "Entity") or "Entity"

            # ontology_uri may live on entity or in metadata
            ont = ""
            if hasattr(e, "metadata") and e.metadata:
                ont = e.metadata.get("ontology_uri") or ""
            if not ont:
                ont = getattr(e, "ontology_uri", "") or ""

            key = (label, etype, str(ont))

            # keep the highest-confidence instance
            if key not in seen:
                seen[key] = e
            else:
                prev = seen[key]
                prev_conf = float(getattr(prev, "confidence", 0.0) or 0.0)
                curr_conf = float(getattr(e, "confidence", 0.0) or 0.0)
                if curr_conf > prev_conf:
                    seen[key] = e

        return list(seen.values())

