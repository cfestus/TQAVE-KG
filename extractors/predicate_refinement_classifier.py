"""Narrowly-scoped, single-purpose LLM classifier that decides
whether an ACCEPTED semantic relation currently using a generic fallback
predicate (is/has/relatedTo) should be upgraded to a more specific,
semantically-entailed predicate, kept generic, or preserved as a novel
local (unmapped) predicate.

General semantic-definition prompting only -- no dataset IDs, record IDs,
entity names, or current-corpus relation examples are ever included in the
prompt. The reusable-predicate vocabulary and definitions below apply
identically to any future dataset/domain/language.
"""

import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)

DECISIONS = ("upgrade", "keep_generic", "local_specific")

REASONING_CATEGORIES = (
    "definition", "classification", "description", "identity", "equivalence",
    "attribute_assignment", "apposition", "state",
    "possession", "composition", "characteristic", "component", "output",
    "requirement", "membership",
    "affects", "uses", "causes", "associated_with", "located_in", "part_of",
    "applies_to", "derived_from", "measured_by", "references", "compares_with",
    "temporal_relation", "weak_association", "extraction_artifact", "other",
)

# Reusable existing predicate vocabulary offered to the model for the
# "upgrade" decision, with concise, general definitions -- no example
# mappings, no corpus-specific predicate/entity pairs.
_REUSABLE_PREDICATES = """
uses: the subject makes use of the object as a tool, input, or resource.
trainedOn: the subject (a model/system) was trained using the object (typically a dataset).
causes: the subject brings about or results in the object.
affects: the subject has an impact or effect on the object.
particularlyAffects: the subject has a disproportionate or disparate impact specifically on the object (typically a population/group).
partOfDomain: the subject belongs to or occurs within the object as a domain/context.
referencesSource: the subject cites or points to the object as its source document/reference/URL.
cites: the subject formally cites the object as a bibliographic reference.
includes: the subject contains the object as one of its constituent parts or members.
hasSubjectDescription: the subject's content covers or is substantively about the object, expressed as descriptive/quantitative text (not a reference to another entity).
requires: the subject mandates or necessitates the object.
mitigatedBy: the subject's risk or harm is reduced or addressed by the object.
associatedWith: the subject is meaningfully associated with the object, more specifically than a bare relation but without a stronger verb.
reflects: the subject mirrors, reveals, or is indicative of the object.
hasRisk: the subject carries or poses the object as a risk.
exhibitsRisk: the subject demonstrably shows the object as a risk or bias.
hasDescription: the subject is described, defined, or characterised by the object, which is descriptive text.
hasAttribute: the subject possesses the object as a characteristic, property, value, or state.
""".strip()

_DECISION_DEFINITIONS = """
upgrade: the evidence semantically entails one of the reusable predicates listed below -- choose the single best-fitting one as final_predicate.
keep_generic: the evidence only supports a weak, generic association -- no specific predicate is entailed. Retain the original generic predicate; final_predicate should repeat the original predicate given to you.
local_specific: the evidence clearly entails a specific relation, but none of the reusable predicates fit it -- propose a new, general, camelCase predicate name (e.g. "definesProcedureFor") as final_predicate. Never invent a predicate that is really just a paraphrase of one already listed above.
""".strip()

SYSTEM_PROMPT = f"""You are a careful ontology-typing assistant for a knowledge-graph pipeline. You will be given one accepted semantic relation (subject, generic predicate, object) plus its source context, and must decide whether the underlying evidence entails a more specific relationship than the generic predicate currently expresses.

Reusable existing predicates (prefer these over inventing anything new):
{_REUSABLE_PREDICATES}

Decision options:
{_DECISION_DEFINITIONS}

Rules (apply to any dataset, domain, or language, not specific to this text):
- The selected predicate must be ENTAILED by the source evidence -- do not choose a more specific predicate merely because it sounds plausible for entities of this general kind. If the evidence only supports a weak association, choose keep_generic rather than guessing.
- Base your decision on the actual grammatical/semantic relationship expressed between the subject and object in the evidence, not on the subject's or object's type alone.
- Prefer a reusable predicate over local_specific whenever one genuinely fits, even approximately -- only use local_specific when the evidence is clearly more specific than every reusable predicate.
- Do not use local_specific merely to rename a reusable predicate with a synonym.
- Set confidence to "confident" only when the evidence clearly and specifically supports your decision; otherwise set it to "ambiguous"."""

_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "predicate_refinement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": list(DECISIONS)},
                "final_predicate": {"type": "string"},
                "reasoning_category": {"type": "string", "enum": list(REASONING_CATEGORIES)},
                "confidence": {"type": "string", "enum": ["confident", "ambiguous"]},
            },
            "required": ["decision", "final_predicate", "reasoning_category", "confidence"],
            "additionalProperties": False,
        },
    },
}


class PredicateRefinementClassifier:
    """Single-purpose structured-output LLM classifier. Owns its own
    OpenAI client, following the same construction convention as the
    other narrowly-scoped classifiers in this pipeline."""

    def __init__(self, model: Optional[str] = None):
        Config.load_llm_api_key()
        self.client = OpenAI(api_key=getattr(Config, "LLM_API_KEY", None))
        self.model = model or Config.LLM_MODEL

    def classify(
        self,
        *,
        subject: str,
        subject_type: str,
        object_value: str,
        object_type: str,
        is_literal_object: bool,
        original_predicate: str,
        sentence: str,
        evidence: str,
    ) -> Dict[str, str]:
        """Returns {"decision", "final_predicate", "reasoning_category",
        "confidence"}. Never raises -- falls back to keep_generic /
        Ambiguous-equivalent so callers can always safely retain the
        original predicate rather than propagate an exception or
        hallucinate specificity."""
        fallback = {
            "decision": "keep_generic",
            "final_predicate": str(original_predicate or "relatedTo"),
            "reasoning_category": "other",
            "confidence": "ambiguous",
        }
        subject = str(subject or "").strip()
        object_value = str(object_value or "").strip()
        if not subject or not object_value or not original_predicate:
            return fallback

        object_kind = "a literal value" if is_literal_object else "an entity"
        user_prompt = (
            f"Subject mention: {subject!r} (type: {subject_type!r})\n"
            f"Original generic predicate: {original_predicate!r}\n"
            f"Object: {object_value!r} (type: {object_type!r}, {object_kind})\n\n"
            f"Sentence / local semantic unit:\n{sentence}\n\n"
            f"Exact evidence span:\n{evidence}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=120,
                response_format=_RESPONSE_FORMAT,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            decision = str(data.get("decision") or "").strip()
            final_predicate = str(data.get("final_predicate") or "").strip()
            reasoning_category = str(data.get("reasoning_category") or "").strip()
            confidence = str(data.get("confidence") or "").strip()
            if (
                decision not in DECISIONS
                or reasoning_category not in REASONING_CATEGORIES
                or confidence not in {"confident", "ambiguous"}
                or not final_predicate
            ):
                return fallback
            return {
                "decision": decision,
                "final_predicate": final_predicate,
                "reasoning_category": reasoning_category,
                "confidence": confidence,
            }
        except Exception:
            logger.warning("PredicateRefinementClassifier call failed; keeping generic predicate", exc_info=True)
            return fallback
