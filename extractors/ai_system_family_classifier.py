"""Narrowly-scoped, single-purpose LLM classifier that decides
what a candidate mention currently typed AISystem/Model/System actually
denotes -- a deployed AI system, an underlying model, ordinary software, a
process/activity, an audit-specific process, an evaluation activity, an
AI-governance framework, an organisation, a concrete resource, a generic
concept, or genuinely ambiguous.

General semantic-definition prompting only -- no dataset IDs, record IDs,
entity names, AI product/model/framework names, or example mappings are
ever included in the prompt. The class vocabulary and definitions below
apply identically to any future dataset/domain/language.
"""

import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)

ALLOWED_LABELS = (
    "AISystem",
    "Model",
    "SoftwareProduct",
    "Process",
    "AuditProcess",
    "Evaluation",
    "AiGovernanceFramework",
    "Organization",
    "Resource",
    "Concept",
    "Ambiguous",
)

_CLASS_DEFINITIONS = """
AISystem: A deployed or operational system whose behaviour materially depends on AI/ML components (e.g. a named product, service, or platform used in a real decision-making or prediction context).
Model: A computational/learned artefact used within or as part of an AI system -- the component-level model or algorithm itself, not automatically the whole deployed system that embeds it.
SoftwareProduct: A software application, tool, or product that is not necessarily AI-driven.
Process: An activity, workflow, procedure, or practice.
AuditProcess: A process specifically performing structured audit/assessment activity. A more specific subclass of Process/Evaluation; use Evaluation for assessment activity that is not specifically audit-shaped.
Evaluation: An evaluative activity or assessment, not a system.
AiGovernanceFramework: A framework, methodology, or instrument for governing, evaluating, managing, or structuring AI-related practice -- not itself an AISystem unless the source clearly describes an operational AI system.
Organization: An institution, body, company, agency, or team acting as a collective entity -- not a technological system.
Resource: A concrete information-hosting or information-providing resource (e.g. a repository, platform, catalogue) that is not itself a system, model, or organisation.
Concept: An abstract notion or generic category that is not itself a system, model, process, framework, organisation, or resource.
Ambiguous: The context does not provide enough evidence to confidently distinguish between the available categories.
""".strip()

SYSTEM_PROMPT = f"""You are a careful ontology-typing assistant for a knowledge-graph pipeline. You will be given one candidate entity mention plus its source context, and must classify what that mention itself refers to, using ONLY the following closed set of general semantic categories:

{_CLASS_DEFINITIONS}

Classification rules (apply to any dataset, domain, or language, not specific to this text):
- Base your decision on the grammatical and semantic ROLE the mention plays in its own sentence, not on which words happen to appear in it (a mention is not AISystem merely because it appears in an AI-related discussion).
- A mention is only AISystem if it ITSELF denotes a deployed/operational system, not an activity performed on or around one, an organisation that operates one, or a framework used to govern/evaluate one.
- Model is the underlying computational/learned artefact, distinct from the deployed system that embeds and operates it -- do not classify a mention as Model merely because the word "model" appears in it (e.g. a process ABOUT a model, such as an evaluation or audit of a model, is Process/AuditProcess/Evaluation, not Model).
- An activity, workflow, practice, or procedure (including gerund-shaped mentions like "testing", "auditing", "red-teaming") is Process, or more specifically AuditProcess when it is itself a structured audit/compliance/assessment activity, or Evaluation when it is an assessment/evaluation activity that is not specifically audit-shaped -- never AISystem or Model, even when performed on or within an AI system.
- A framework, methodology, standard, or instrument used to guide, constrain, or evaluate AI practice is AiGovernanceFramework, not AISystem, even when it is specific to AI -- unless the source clearly describes it as itself an operational AI-driven system.
- An institution, company, agency, regulator, or team is Organization, never AISystem or Model, even when it develops, deploys, or is closely associated with one.
- Prefer "Ambiguous" over guessing when the evidence genuinely does not distinguish between two or more categories -- do not invent certainty.
- Set confidence to "confident" only when the context provides clear, specific evidence for your chosen label; otherwise set it to "ambiguous"."""

_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "ai_system_family_classification",
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


class AiSystemFamilyTypeClassifier:
    """Single-purpose structured-output LLM classifier. Owns its own
    OpenAI client, following the same construction convention as
    DatasetFamilyTypeClassifier / PopulationGroupTypeClassifier."""

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
            logger.warning("AiSystemFamilyTypeClassifier call failed; abstaining as Ambiguous", exc_info=True)
            return fallback
