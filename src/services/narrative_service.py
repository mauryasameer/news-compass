from __future__ import annotations

import logging

from meerax.llm.base import LLMProvider
from meerax.llm.prompt import PromptTemplate

logger = logging.getLogger(__name__)

NARRATIVE_PROMPT = PromptTemplate(
    "A news recommender suggested the article '{title}' (relevance score {score:.2f}). "
    "In 1-2 plain-English sentences, explain why a reader might find this relevant, "
    "based only on the title and score given."
)


def generate_narratives(recommendations: list[dict], llm: LLMProvider) -> dict[int, str]:
    narratives: dict[int, str] = {}
    for rec in recommendations:
        try:
            prompt = NARRATIVE_PROMPT.render(title=rec["title"], score=rec["score"])
            narratives[rec["item_id"]] = llm.generate(prompt, temperature=0.0).content
        except Exception:
            logger.exception("Narrative generation failed for item %s", rec["item_id"])
            narratives[rec["item_id"]] = "narrative unavailable"
    return narratives
