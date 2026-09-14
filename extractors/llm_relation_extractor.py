"""
LLM-based Relation Extraction (OpenAI JSON Mode Version)
ETD-Hub - Semantic Knowledge Graph Pipeline

This extractor:
- Uses the native OpenAI client
- Forces strict JSON output
- Produces clean Relation objects
"""

import json
import time
import hashlib
import random
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from utils.predicate_map import normalize_predicate
from utils.helpers import normalize_text
from config import Config
from models import Relation
import logging

logger = logging.getLogger(__name__)


def safe_hash(value: str) -> str:
    """Stable short hash for relation identity."""
    return hashlib.sha256(value.encode()).hexdigest()[:15]


class LLMRelationExtractor:

    def __init__(
        self,
        model: str = None,
        max_retries: int = 3,
        relation_confidence_threshold: float = 0.35
    ):
        self.max_retries = max_retries
        self.model = model or Config.LLM_MODEL
        self.relation_confidence_threshold = relation_confidence_threshold
        self.retry_base_delay = 0.5
        self.retry_max_delay = 4.0

        # Direct OpenAI client
        Config.load_llm_api_key()
        self.client = OpenAI(api_key=getattr(Config, "LLM_API_KEY", None))

        # System prompt
        self.system_prompt = """
You are an expert semantic proposition extraction model.
You MUST output ONLY valid JSON in the following format:

{
  "assertions": [
    {
      "subject_id": "<entity_id or empty string>",
      "subject_text": "<verbatim subject surface from the text>",
      "subject_type": "<best local type guess>",
      "predicate": "<predicate phrase or canonical predicate label>",
      "object_id": "<entity_id or empty string>",
      "object_text": "<verbatim object surface from the text when entity-valued>",
      "object_type": "<entity|literal|quantity|money|date|percentage|count|temporal_range|comparison|concept|population|dataset|system>",
      "target_literal": "<literal object value when object is not entity-valued>",
      "target_datatype": "<xsd datatype, literal kind, or empty string>",
      "qualifiers": {},
      "modality": {},
      "negated": false,
      "temporal_information": {},
      "quantitative_information": {},
      "confidence": 0.92,
      "evidence_quote": "<verbatim quote from the text>",
      "evidence": "<short quote/span from the text>"
    }
  ]
}

Rules:
- Enumerate every independently meaningful proposition explicitly supported by the supplied text.
- Be exhaustive. Do not select only salient, important, key, representative, or main propositions.
- Do not omit later propositions because earlier ones seem more important.
- The provided entity hints are optional anchors, not hard constraints.
- Use subject_id/object_id only when an entity hint exactly matches the local proposition endpoint.
- Do not require both endpoints to already exist in the entity hints.
- Use target_literal plus qualifiers / quantitative_information / temporal_information for literal-valued objects.
- Comparative/likelihood statements such as "X was/were/is/are <N> times/percent more/less
  likely to <VERB> <Y> [than <Z>]", "X has a higher/lower <RATE> than <Z>", or "X occurs <N>%
  more/less often than <Z>" express a specific relation between X and the outcome/rate Y, NOT a
  generic "is/are/was/were/has" copula relation, and the comparison itself is a QUALIFIER on
  that relation, not the object. For these:
  - predicate MUST be the verb or relation phrase that actually connects X to Y (e.g. "receive",
    "occur", "experience", "approval rate"), copied or lightly conjugated from the text -- NEVER
    use the bare copula (is/are/was/were/has) as predicate, and NEVER fold the comparison words
    ("times", "more", "less", "percent") into the predicate itself.
  - object_text MUST be only the outcome/rate noun phrase (e.g. "additional care", "approval
    rate"), never the comparison number or the full comparative clause.
  - When the statement has NO separate outcome noun phrase -- a passive/intransitive verb (e.g.
    "X is more likely to be rejected than Z") or a bare rate comparison (e.g. "X has a higher
    approval rate than Z", "X occurs more often than Z") -- there is nothing to put in object_text
    for the outcome itself, so use the comparison target Z as object_text instead, and let
    predicate name the compared verb/rate (e.g. "rejected", "approvalRate", "occurs"). Every
    relation MUST still have a non-empty object_text or target_literal; never leave both empty.
  - target_literal / target_is_literal MUST NOT be set to the bare comparison number -- the
    number belongs in quantitative_information, not as the relation's target.
  - Record the comparison in quantitative_information as
    {"factor": <number>, "unit": "times"|"percent", "direction": "more"|"less"}.
  - Only set qualifiers.compared_to when the text states an EXPLICIT comparison target using one
    of: "than <Z>", "vs <Z>", "versus <Z>", "compared to/with <Z>". Do NOT set compared_to from
    any other clause, and NEVER infer or guess a comparison target when none of those markers is
    present -- omit compared_to entirely in that case rather than filling it with the nearest
    unrelated clause.
  - A concessive clause ("despite <X>", "although <X>", "even though <X>") is NOT a comparison
    target and MUST NOT be put in compared_to. If present, record it as qualifiers.concession
    (verbatim from the text); if a proposition has no true comparison target and you are unsure
    whether the remaining clause is a concession, simply leave both compared_to and concession
    out rather than misclassifying it.
  Examples:
  "Group A was 3 times less likely to receive Outcome B than Group C." ->
  subject_text="Group A", predicate="receive", object_text="Outcome B", target_literal="",
  quantitative_information={"factor": 3, "unit": "times", "direction": "less"},
  qualifiers={"compared_to": "Group C"}.
  "Group A is 3 times more likely to be rejected than Group C." ->
  subject_text="Group A", predicate="rejected", object_text="Group C", target_literal="",
  quantitative_information={"factor": 3, "unit": "times", "direction": "more"},
  qualifiers={"compared_to": "Group C"}.
  "Group A was 2 times less likely to receive Outcome B despite greater need." -> (no "than"/
  "vs"/"versus"/"compared to" marker, so no true comparison target is stated)
  subject_text="Group A", predicate="receive", object_text="Outcome B", target_literal="",
  quantitative_information={"factor": 2, "unit": "times", "direction": "less"},
  qualifiers={"concession": "despite greater need"} (compared_to omitted).
- Only extract propositions explicitly stated or directly supported by the local text.
- You may be given a "Context" section (the surrounding source field) in addition to the "Text"
  section (the specific span to extract from). Extract propositions only from Text. Use Context
  only to resolve what a pronoun, shorthand reference, or ambiguous actor name in Text refers to
  (e.g. what "it" or "this system" means) -- never use Context to introduce a proposition that
  is not itself expressed in Text, and never extract a proposition that only appears in Context.
- evidence_quote and evidence MUST be verbatim substrings copied from Text, never from Context.
- An entity hint marked "field_context_actor": true identifies a system/process actor
  established earlier in the same field (visible in Context) that Text itself does not name.
  You MAY use that hint's id as subject_id -- even though its name is not repeated in Text --
  ONLY when Text expresses a claim (e.g. an effect, outcome, or comparison) that is clearly about
  that same actor's behavior and Text does not name a DIFFERENT entity actively performing that
  behavior. A local entity that Text names only as the affected/impacted party -- e.g. the
  grammatical subject of a passive construction such as "X was charged/affected/penalized..." or
  "X faced/experienced Y" where X does not itself act -- is not a competing actor and does not
  block use of the hint; in that case use the hint as subject_id and assign the affected party as
  object_id/object_text instead. Do not use the hint if Text names its own entity actively
  performing the action, or if Text's claim could plausibly be about a different actor.
  evidence_quote/evidence still MUST be copied verbatim from Text only.
  Example: Context earlier establishes "System C" (entity hint {"id": "<system_c_id>", "label":
  "System C", "field_context_actor": true}). Text: "Category D: Group E is flagged at a
  substantially higher frequency" -- Group E is named first and is grammatically the local
  subject, but it is the party something is done TO, not an entity doing the flagging, and Text
  names no other active agent. Correct: subject_id="<system_c_id>", predicate reflects System C's
  action (e.g. "flags"), object_id/object_text="Group E" -- do NOT set subject_id to Group E's id.
  If Text instead read "Category D: Component F flags Group E at a substantially higher
  frequency", Component F is a different entity actively performing the action, so the hint must
  NOT be used; extract subject_text="Component F" from Text as normal.
- Use canonical relation labels from this schema set when they are clearly supported:
  [uses, trainedOn, reflects, hasRisk, exhibitsRisk, causes, causedBy, affects, affectedBy,
   particularlyAffects, mitigates, mitigatedBy, supports, criticizes, endorses, disagreesWith,
   opposes, compliesWith, requiresComplianceWith, requires, violates, referencesInstrument,
   referencesSource, cites, associatedWith, partOfDomain, relatedTo].
- When no canonical predicate fits exactly, preserve the locally supported predicate phrase in predicate instead of collapsing to a generic fallback.
- Prefer the most specific predicate supported by the text.
- Do NOT use "mentions" for ordinary domain claims. Structural predicates are handled elsewhere.
- confidence MUST be in [0,1].
"""

    # =====================================================================
    # Public API
    # =====================================================================

    def extract_batch(self, texts: List[str], entities_per_doc: List[List[Any]], discourse_ids: Optional[List[str]] = None) -> List[List[Relation]]:
        all_relations: List[List[Relation]] = []
        for i, (text, ents) in enumerate(zip(texts, entities_per_doc)):
            did = discourse_ids[i] if discourse_ids and i < len(discourse_ids) else None
            relations = self.extract(text, ents, discourse_id=did)
            all_relations.append(relations)
        return all_relations

    def extract(
        self,
        text: str,
        entities: List[Any],
        discourse_id: Optional[str] = None,
        candidate_pairs: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
    ) -> List[Relation]:
        raw = self._call_with_retries(text, entities, candidate_pairs=candidate_pairs, context=context)

        # Allowed IDs are only valid when the extractor explicitly reuses entity hints.
        allowed_ids = {
            getattr(e, "uri", None)
            for e in entities
            if getattr(e, "uri", None)
        }

        cleaned = self._clean_relations(raw, allowed_ids=allowed_ids)
        converted = self._convert_to_relations(cleaned, discourse_id=discourse_id, text=text, entities=entities)
        return converted


    # =====================================================================
    # Retry wrapper
    # =====================================================================
    def _call_with_retries(
        self,
        text: str,
        entities: List[Any],
        candidate_pairs: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_model(text, entities, candidate_pairs=candidate_pairs, context=context)
            except Exception as e:
                last_error = e
                retryable = self._is_transient_error(e)
                logger.warning(
                    f"[Relation Extraction] attempt {attempt}/{self.max_retries} failed "
                    f"(retryable={retryable}): {e}"
                )
                if not retryable:
                    logger.error("[Relation Extraction] non-transient failure; not retrying.")
                    raise RuntimeError(f"LLM relation extraction failed (non-transient): {e}") from e
                if attempt == self.max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(f"[Relation Extraction] retrying in {delay:.2f}s")
                time.sleep(delay)
        if last_error is not None:
            logger.error(f"[Relation Extraction] final failure after {self.max_retries} attempts: {last_error}")
            raise RuntimeError(
                f"LLM relation extraction failed after {self.max_retries} attempts: {last_error}"
            ) from last_error
        raise RuntimeError("LLM relation extraction failed with unknown error")

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
        return False

    # =====================================================================
    # DIRECT OpenAI JSON-mode call
    # =====================================================================
    def _parse_relations_json(self, content: Any) -> Dict[str, Any]:
        """
        Robust JSON parsing for LLM outputs.
        Always returns a dict with an 'assertions' key.
        """
        fallback: Dict[str, Any] = {"relations": []}

        def _validate_payload(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, dict):
                if isinstance(obj.get("relations"), list) and not isinstance(obj.get("assertions"), list):
                    obj["assertions"] = obj.get("relations", [])
                if isinstance(obj.get("assertions"), list) and not isinstance(obj.get("relations"), list):
                    obj["relations"] = obj.get("assertions", [])
                if not isinstance(obj.get("assertions"), list):
                    obj["assertions"] = []
                if not isinstance(obj.get("relations"), list):
                    obj["relations"] = list(obj.get("assertions", []))
                return obj
            return fallback

        if not isinstance(content, str):
            logger.error("LLM returned non-string content: %r", type(content))
            return fallback

        raw = content.strip()
        if not raw:
            logger.error("LLM returned empty content")
            return fallback

        # 1) Strict JSON parse
        try:
            return _validate_payload(json.loads(raw))
        except Exception:
            pass

        # 2) Slice from first '{' to last '}'.
        # 3) If no closing '}', append one.
        # 4) If relations array is unterminated, append ']}'.
        repaired_candidates: List[str] = []
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1:
            if last != -1 and last >= first:
                sliced = raw[first:last + 1]
            else:
                sliced = raw[first:] + "}"
            repaired_candidates.append(sliced)

            rel_key_pos = sliced.find('"assertions"')
            if rel_key_pos == -1:
                rel_key_pos = sliced.find('"relations"')
            rel_open = sliced.find("[", rel_key_pos) if rel_key_pos != -1 else -1
            rel_close = sliced.rfind("]")
            if rel_open != -1 and rel_close < rel_open:
                repaired_candidates.append(sliced.rstrip() + "]}")

        for cand in repaired_candidates:
            try:
                return _validate_payload(json.loads(cand))
            except Exception:
                pass

        # 5) Lightweight repair (single quotes + trailing commas)
        candidates = [raw]
        candidates.extend(repaired_candidates)
        for cand in candidates:
            repaired = cand.replace("'", '"')
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            try:
                return _validate_payload(json.loads(repaired))
            except Exception:
                continue

        # 6) Final fallback: log and return empty
        logger.error("Failed to parse LLM JSON output. Raw content: %s", raw[:4000])
        return fallback

    def _build_prompt(
        self,
        text: str,
        entities: List[Any],
        candidate_pairs: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
    ) -> str:
        entity_list = []
        for e in entities:
            uri = getattr(e, "uri", None)
            label = getattr(e, "label", None)
            if not uri or not label:
                continue
            hint = {"id": uri, "label": label, "type": getattr(e, "entity_type", None)}
            meta = getattr(e, "metadata", None) or {}
            if meta.get("carried_forward_from_field_context"):
                hint["field_context_actor"] = True
            entity_list.append(hint)

        pair_list = []
        for pair in candidate_pairs or []:
            if not isinstance(pair, dict):
                continue
            source_id = str(pair.get("source_id") or "").strip()
            target_id = str(pair.get("target_id") or "").strip()
            if not source_id or not target_id:
                continue
            pair_list.append(
                {
                    "source_id": source_id,
                    "source_label": str(pair.get("source_label") or "").strip(),
                    "target_id": target_id,
                    "target_label": str(pair.get("target_label") or "").strip(),
                    "predicate_hints": list(pair.get("predicate_hints") or []),
                    "cue": str(pair.get("cue") or "").strip(),
                    "selection_reason": str(pair.get("selection_reason") or "").strip(),
                }
            )

        max_entity_hints = int(getattr(Config, "MAX_ENTITY_HINTS_PER_PROMPT", 150) or 150)
        entity_block = ""
        if entity_list and len(entity_list) <= max_entity_hints:
            entity_block = f"""
Entity hints (optional anchors only; use ids only when they exactly fit the proposition):
{json.dumps(entity_list, indent=2)}
"""
        elif entity_list:
            entity_block = (
                f"\nEntity hints omitted because {len(entity_list)} candidates would over-expand the prompt. "
                "You may still extract propositions directly from the text.\n"
            )

        pair_block = ""
        if pair_list:
            pair_block = f"""
Candidate proposition anchors (optional, not exhaustive):
{json.dumps(pair_list, indent=2)}
"""

        context_block = ""
        norm_context = str(context or "").strip()
        if norm_context and norm_context != str(text or "").strip():
            context_block = f"""
Context (full source field -- use ONLY to resolve pronouns, shorthand, or ambiguous actor
identity referenced in Text; do NOT extract propositions from Context that are not also
supported by Text):
{norm_context}
"""

        prompt = f"""
Extract all explicitly supported semantic propositions from Text below.

{entity_block}

{pair_block}

{context_block}

Text (the target span to extract propositions from; evidence_quote MUST come from here):
{text}

Return only valid JSON.
"""
        return prompt

    def _call_model(
        self,
        text: str,
        entities: List[Any],
        candidate_pairs: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(text, entities, candidate_pairs=candidate_pairs, context=context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"}  # STRICT JSON MODE
        )

        content = response.choices[0].message.content

        parsed = self._parse_relations_json(content)
        rels = parsed.get("assertions", [])
        if not isinstance(rels, list):
            rels = parsed.get("relations", [])
        if not isinstance(rels, list):
            rels = []

        clean = [x for x in rels if isinstance(x, dict)]
        return clean

    # =====================================================================
    # Filter + clean relations
    # =====================================================================
    def _clean_relations(
        self,
        rels: List[Dict[str, Any]],
        *,
        allowed_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        allowed_ids = allowed_ids or set()

        for r in rels:
            try:
                conf = float(r.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0            
            if conf < self.relation_confidence_threshold:
                continue

            sid = r.get("source_id") or r.get("subject_id")
            tid = r.get("target_id") or r.get("object_id")
            subject_text = str(r.get("subject_text") or "").strip()
            object_text = str(r.get("object_text") or "").strip()
            target_literal = r.get("target_literal")
            predicate = str(r.get("predicate") or r.get("relation") or "").strip()

            if not predicate:
                continue
            if not sid and not subject_text:
                continue
            if not tid and not object_text and (target_literal is None or str(target_literal).strip() == ""):
                continue

            if sid and not str(sid).startswith("http"):
                sid = None
            if tid and not str(tid).startswith("http"):
                tid = None

            if allowed_ids:
                if sid and sid not in allowed_ids:
                    sid = None
                if tid and tid not in allowed_ids:
                    tid = None
                if not sid and not subject_text:
                    continue
                if not tid and not object_text and (target_literal is None or str(target_literal).strip() == ""):
                    continue

            r["subject_id"] = sid
            r["object_id"] = tid
            out.append(r)

        return out


    # =====================================================================
    # Convert JSON relations to Relation objects
    # =====================================================================
    @staticmethod
    def _all_occurrences(text: str, quote: str) -> List[Tuple[int, int]]:
        occurrences: List[Tuple[int, int]] = []
        if not text or not quote:
            return occurrences
        start = 0
        while True:
            idx = text.find(quote, start)
            if idx == -1:
                break
            occurrences.append((idx, idx + len(quote)))
            start = idx + 1
        return occurrences

    @staticmethod
    def _coerce_nonneg_int(value: Any) -> Optional[int]:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        return iv if iv >= 0 else None

    def _mention_midpoint_for_entity(self, entity: Any, discourse_id: Optional[str]) -> Optional[float]:
        meta = getattr(entity, "metadata", None) or {}
        did = str(discourse_id or "").strip()
        mentions = meta.get("mentions")
        if isinstance(mentions, list):
            scoped: List[Tuple[int, int]] = []
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                m_did = str(m.get("discourse_id", "") or "").strip()
                if did and m_did != did:
                    continue
                s = self._coerce_nonneg_int(m.get("start_char"))
                e = self._coerce_nonneg_int(m.get("end_char"))
                if s is None or e is None or e <= s:
                    continue
                scoped.append((int(s), int(e)))
            if scoped:
                min_s = min(s for s, _ in scoped)
                max_e = max(e for _, e in scoped)
                return float(min_s + max_e) / 2.0
        s = self._coerce_nonneg_int(meta.get("start_char"))
        e = self._coerce_nonneg_int(meta.get("end_char"))
        if s is not None and e is not None and e > s:
            return float(s + e) / 2.0
        return None

    def _pair_midpoint(
        self,
        source_uri: str,
        target_uri: str,
        entities: List[Any],
        discourse_id: Optional[str],
    ) -> Optional[float]:
        by_uri: Dict[str, Any] = {}
        for e in entities:
            u = str(getattr(e, "uri", "") or "").strip()
            if u:
                by_uri[u] = e
        src_ent = by_uri.get(str(source_uri or "").strip())
        tgt_ent = by_uri.get(str(target_uri or "").strip())
        if src_ent is None or tgt_ent is None:
            return None
        src_mid = self._mention_midpoint_for_entity(src_ent, discourse_id)
        tgt_mid = self._mention_midpoint_for_entity(tgt_ent, discourse_id)
        if src_mid is None or tgt_mid is None:
            return None
        return (float(src_mid) + float(tgt_mid)) / 2.0

    def _select_quote_span(
        self,
        text: str,
        evidence_quote: str,
        source_uri: str,
        target_uri: str,
        entities: List[Any],
        discourse_id: Optional[str],
    ) -> Optional[Tuple[int, int]]:
        quote = str(evidence_quote or "").strip()
        if not quote:
            return None
        occurrences = self._all_occurrences(text, quote)
        if not occurrences:
            return None
        if len(occurrences) == 1:
            return occurrences[0]
        # Duplicate-safe deterministic tie break:
        # use pair midpoint if both source/target mention spans exist in this discourse.
        pair_mid = self._pair_midpoint(source_uri, target_uri, entities, discourse_id)
        if pair_mid is None:
            return sorted(occurrences, key=lambda x: x[0])[0]
        ranked = sorted(
            occurrences,
            key=lambda span: (abs(((span[0] + span[1]) / 2.0) - pair_mid), span[0]),
        )
        return ranked[0]

    def _convert_to_relations(
        self,
        rels: List[Dict[str, Any]],
        discourse_id: Optional[str] = None,
        text: str = "",
        entities: Optional[List[Any]] = None,
    ) -> List[Relation]:
        results: List[Relation] = []
        ent_list = entities or []
        for r in rels:
            source = str(r.get("source_id") or r.get("subject_id") or r.get("subject_text") or "").strip()
            raw_predicate = str(r.get("predicate") or r.get("relation") or "").strip()
            rel_label = normalize_predicate(raw_predicate)
            raw_norm = normalize_text(raw_predicate)
            generic_aliases = {
                "relatedto",
                "related to",
                "relates to",
                "is related to",
                "linked to",
                "connects to",
                "associatedwith",
                "associated with",
            }
            if rel_label == "relatedTo" and raw_predicate and raw_norm not in generic_aliases:
                compact = re.sub(r"[^A-Za-z0-9]+", " ", raw_predicate).strip()
                if compact and normalize_predicate(compact) == "relatedTo":
                    pieces = [p for p in re.split(r"\s+", compact) if p]
                    if pieces:
                        rel_label = pieces[0].lower() + "".join(p.capitalize() for p in pieces[1:])

            target = str(r.get("target_id") or r.get("object_id") or r.get("object_text") or "").strip()
            target_literal = r.get("target_literal")
            object_type = str(r.get("object_type") or "").strip()
            target_is_literal = bool(
                r.get("target_is_literal")
                or object_type.lower() in {"literal", "quantity", "money", "date", "percentage", "count", "temporal_range", "comparison"}
                or (target_literal is not None and str(target_literal).strip() != "")
            )
            if target_is_literal:
                if target_literal is None or str(target_literal).strip() == "":
                    target_literal = target
                target = str(target_literal or "").strip()
            try:
                confidence = float(r.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            if not source or not rel_label:
                continue
            if not target:
                continue

            raw_object_key = target_literal if target_is_literal else target
            raw_key = f"{discourse_id}|{source}|{rel_label}|{raw_object_key}"
            rel_hash = safe_hash(raw_key)
            uri = f"{Config.NAMESPACES['etd']}rel_{rel_hash}"
            evidence_quote_raw = str(r.get("evidence_quote") or "").strip()
            evidence = (r.get("evidence") or evidence_quote_raw).strip()
            chosen_span = None
            if evidence_quote_raw and str(source).startswith("http") and (not target_is_literal) and str(target).startswith("http"):
                chosen_span = self._select_quote_span(
                    text=text or "",
                    evidence_quote=evidence_quote_raw,
                    source_uri=str(source),
                    target_uri=str(target),
                    entities=ent_list,
                    discourse_id=discourse_id,
                )
            if chosen_span is None:
                char_start = -1
                char_end = -1
                grounding_status = "ungrounded"
                grounding_method = "none"
            else:
                char_start = int(chosen_span[0])
                char_end = int(chosen_span[1])
                grounding_status = "grounded"
                grounding_method = "evidence_quote_span"
            if len(evidence) > 500:
                evidence = evidence[:500]
            evidence_quote = evidence_quote_raw
            if len(evidence_quote) > 500:
                evidence_quote = evidence_quote[:500]

            metadata = {
                "evidence": evidence,
                "evidence_quote": evidence_quote,
                "context_source": "llm_json",
                "discourse_id": discourse_id,
                "extractor": "llm_relation_extractor",
                "char_start": int(char_start),
                "char_end": int(char_end),
                "start_offset": int(char_start),
                "end_offset": int(char_end),
                "groundingStatus": grounding_status,
                "groundingMethod": grounding_method,
                "subject_text": str(r.get("subject_text") or "").strip(),
                "subject_type": str(r.get("subject_type") or "").strip(),
                "object_text": str(r.get("object_text") or "").strip(),
                "object_type": object_type,
                "qualifiers": dict(r.get("qualifiers") or {}),
                "modality": dict(r.get("modality") or {}),
                "negated": bool(r.get("negated", False)),
                "temporal_information": dict(r.get("temporal_information") or {}),
                "quantitative_information": dict(r.get("quantitative_information") or {}),
                "predicate_text": raw_predicate,
                "predicate_mapping_status": "mapped" if normalize_predicate(raw_predicate) != "relatedTo" or rel_label == "relatedTo" and raw_predicate.lower().strip() in {"related to", "associated with"} else "local_unmapped",
                "allow_local_predicate": True,
            }

            relation_obj = Relation(
                uri=uri,
                source=source,
                relation=rel_label,
                target=target,
                confidence=confidence,
                target_literal=target_literal,
                target_datatype=str(r.get("target_datatype") or "").strip() or None,
                target_is_literal=target_is_literal,
                object_type=object_type or None,
                qualifiers=dict(r.get("qualifiers") or {}),
                predicate_text=raw_predicate or None,
                predicate_mapping_status=metadata.get("predicate_mapping_status", "mapped"),
                negated=bool(r.get("negated", False)),
                temporal_information=dict(r.get("temporal_information") or {}),
                quantitative_information=dict(r.get("quantitative_information") or {}),
                is_inferred=False,
                metadata=metadata,
            )

            results.append(relation_obj)

        return results
