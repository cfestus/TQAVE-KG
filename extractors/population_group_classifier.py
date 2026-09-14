"""Narrowly-scoped, single-purpose LLM classifier that decides
whether a candidate mention genuinely denotes a PopulationGroup/
DemographicGroup, or is instead a Concept, an Outcome/finding/comparison
about such a group, or Ambiguous.

General semantic-definition prompting only -- no dataset IDs, record IDs,
entity names, protected-characteristic vocabulary, percentages, or example
mappings are ever included in the prompt. The class vocabulary and
definitions below apply identically to any future dataset/domain/language.
"""

import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)

ALLOWED_LABELS = (
    "DemographicGroup",
    "PopulationGroup",
    "Concept",
    "Outcome",
    "Ambiguous",
)

_CLASS_DEFINITIONS = """
PopulationGroup: A collection of people or other entities considered collectively for analysis, comparison, policy, study, or measurement. The mention itself must refer to the collection of members, not a fact about them.
DemographicGroup: A PopulationGroup whose membership is defined by a demographic or socially meaningful characteristic (e.g. an age band, a protected characteristic, a role, a shared condition). A more specific subclass of PopulationGroup, not a synonym for it.
Concept: An abstract notion, attribute, category, or generic object that is not itself a collection of members considered together.
Outcome: An observed or expected result, finding, rate, disparity, or comparison -- something that is TRUE OF a group or that happened, not the group itself.
Ambiguous: The context does not provide enough evidence to confidently distinguish between the available categories.
""".strip()

SYSTEM_PROMPT = f"""You are a careful ontology-typing assistant for a knowledge-graph pipeline. You will be given one candidate entity mention plus its source context, and must classify what that mention itself refers to, using ONLY the following closed set of general semantic categories:

{_CLASS_DEFINITIONS}

Classification rules (apply to any dataset, domain, or language, not specific to this text):
- Base your decision on the grammatical and semantic ROLE the mention plays in its own sentence, not on which words happen to appear in it.
- A mention is only PopulationGroup or DemographicGroup if it ITSELF refers to the collection of members -- not if it merely states a percentage, rate, count, disparity, finding, or comparison ABOUT such a collection. A percentage or count is a value about a group, never the group itself.
- A clause, sentence fragment, or comparison describing what happened to a group, or how one group compares to another, is Outcome -- not PopulationGroup or DemographicGroup -- even though it may contain a group-denoting noun.
- Do not require any specific protected-characteristic word, demographic label, or population-related word to be present for a confident classification, and do not withhold a confident classification merely because such a word IS present -- judge the mention's grammatical role, not its vocabulary.
- Only classify as DemographicGroup when the collection's membership is itself defined by a demographic or socially meaningful characteristic; otherwise, if it is still a genuine collection of members, prefer PopulationGroup.
- Prefer "Ambiguous" over guessing when the evidence genuinely does not distinguish between two or more categories -- do not invent certainty.
- Set confidence to "confident" only when the context provides clear, specific evidence for your chosen label; otherwise set it to "ambiguous"."""

_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "population_group_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(ALLOWED_LABELS)},
                "confidence": {"type": "string", "enum": ["confident", "ambiguous"]},
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        },
    },
}


class PopulationGroupTypeClassifier:
    """Single-purpose structured-output LLM classifier. Owns its own
    OpenAI client, following the same construction convention as
    DatasetFamilyTypeClassifier / LLMEntityExtractor."""

    def __init__(self, model: Optional[str] = None):
        Config.load_llm_api_key()
        self.client = OpenAI(api_key=getattr(Config, "LLM_API_KEY", None))
        self.model = model or Config.LLM_MODEL

    def classify(
        self,
        *,
        surface: str,
        sentence: str,
        proposed_type: str = "",
    ) -> Dict[str, str]:
        """Returns {"label": <one of ALLOWED_LABELS>, "confidence": "confident"|"ambiguous"}.

        Never raises on a malformed/failed call -- falls back to
        {"label": "Ambiguous", "confidence": "ambiguous"} so callers can
        always safely abstain rather than propagate an exception or
        silently keep a possibly-wrong type."""
        fallback = {"label": "Ambiguous", "confidence": "ambiguous"}
        surface = str(surface or "").strip()
        if not surface:
            return fallback

        user_prompt = (
            f"Entity mention: {surface!r}\n\n"
            f"Sentence / local semantic unit containing the mention:\n{sentence}\n\n"
            f"An earlier, coarser pipeline stage proposed the type {proposed_type!r} "
            f"for this mention; you may agree or disagree with it based only on the "
            f"context above."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=100,
                response_format=_RESPONSE_FORMAT,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            label = str(data.get("label") or "").strip()
            confidence = str(data.get("confidence") or "").strip()
            if label not in ALLOWED_LABELS or confidence not in {"confident", "ambiguous"}:
                return fallback
            return {"label": label, "confidence": confidence}
        except Exception:
            logger.warning("PopulationGroupTypeClassifier call failed; abstaining as Ambiguous", exc_info=True)
            return fallback
