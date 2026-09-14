"""Narrowly-scoped, single-purpose LLM classifier that
distinguishes a Dataset-family candidate entity from adjacent ontology
categories (ResearchStudy, Analysis, Publication, SoftwareProduct,
Resource, Concept), or abstains as Ambiguous.

General semantic-definition prompting only -- no dataset IDs, record IDs,
entity names, software/repository/research-study names, or example
mappings are ever included in the prompt. The class vocabulary and
definitions below apply identically to any future dataset/domain.
"""

import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)

ALLOWED_LABELS = (
    "Dataset",
    "ResearchStudy",
    "Analysis",
    "Publication",
    "SoftwareProduct",
    "Resource",
    "Concept",
    "Ambiguous",
)

_CLASS_DEFINITIONS = """
Dataset: A specific identifiable collection of data, observations, records, examples, measurements, images, documents or similar items that exists as a data resource. A Dataset does not need to contain a numerical count in the surrounding text to qualify.
ResearchStudy: A research investigation or study that examines a question or phenomenon.
Analysis: An analytical examination performed on some data or resource.
Publication: A paper, report, article, book, or other published/documentary research output.
SoftwareProduct: A software application, software package, program, or digital tool that is not itself the dataset being processed.
Resource: A concrete repository, platform, catalogue, portal, or information-hosting resource whose primary referent is not itself a dataset.
Concept: An abstract notion or generic data-related concept without a specific resource referent.
Ambiguous: The context does not provide enough evidence to confidently distinguish between the available categories.
""".strip()

SYSTEM_PROMPT = f"""You are a careful ontology-typing assistant for a knowledge-graph pipeline. You will be given one candidate entity mention plus its source context, and must classify what that mention itself refers to, using ONLY the following closed set of general semantic categories:

{_CLASS_DEFINITIONS}

Classification rules (apply to any dataset or domain, not specific to this text):
- Base your decision on the grammatical and semantic ROLE the mention plays in its own sentence, not merely on words that happen to appear somewhere nearby.
- A mention is only Dataset if the mention ITSELF refers to a specific data resource -- not if it merely appears near a dataset, describes a dataset, hosts/catalogues a dataset, or analyses/studies a dataset. For example: "a study using X" means the study is not X; "an analysis of X" means the analysis is not X; "a repository hosting X" means the repository is not X; "software that processes X" means the software is not X.
- Do not require the words "dataset", "data", "records", "samples", or any specific number to be present for a confident Dataset classification, when the context otherwise clearly identifies the mention as a specific data resource.
- Do not classify something as Dataset merely because a quantity, or the word "data"/"dataset", occurs somewhere in the surrounding text.
- Additional linked context from elsewhere in the source (e.g. a different list item) may only influence your answer if it clearly describes the SAME mention -- if it appears to describe a different, neighbouring item instead, ignore it.
- A concrete repository/platform/catalogue/portal that is not itself a dataset must be classified Resource, not Concept and not Dataset.
- Prefer "Ambiguous" over guessing when the evidence genuinely does not distinguish between two or more categories -- do not invent certainty.
- Set confidence to "confident" only when the context provides clear, specific evidence for your chosen label; otherwise set it to "ambiguous"."""

_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "dataset_family_classification",
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


class DatasetFamilyTypeClassifier:
    """Single-purpose structured-output LLM classifier. Owns its own
    OpenAI client, following the same construction convention as
    LLMEntityExtractor (Config.load_llm_api_key() + OpenAI(api_key=...))."""

    def __init__(self, model: Optional[str] = None):
        Config.load_llm_api_key()
        self.client = OpenAI(api_key=getattr(Config, "LLM_API_KEY", None))
        self.model = model or Config.LLM_MODEL

    def classify(
        self,
        *,
        surface: str,
        sentence: str,
        linked_context: str = "",
        proposed_type: str = "",
    ) -> Dict[str, str]:
        """Returns {"label": <one of ALLOWED_LABELS>, "confidence": "confident"|"ambiguous"}.

        Never raises on a malformed/failed call -- falls back to
        {"label": "Ambiguous", "confidence": "ambiguous"} so callers can
        always safely abstain rather than propagate an exception or
        silently default to Dataset."""
        fallback = {"label": "Ambiguous", "confidence": "ambiguous"}
        surface = str(surface or "").strip()
        if not surface:
            return fallback

        user_prompt = (
            f"Entity mention: {surface!r}\n\n"
            f"Sentence / local semantic unit containing the mention:\n{sentence}\n\n"
            f"Additional structurally linked context from the same source region "
            f"(may be empty; may describe a DIFFERENT nearby item -- use only if it "
            f"clearly describes the mention above):\n{linked_context}\n\n"
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
            logger.warning("DatasetFamilyTypeClassifier call failed; abstaining as Ambiguous", exc_info=True)
            return fallback
