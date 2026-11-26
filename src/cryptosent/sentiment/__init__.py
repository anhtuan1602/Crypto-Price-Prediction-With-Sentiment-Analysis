"""Article-level enrichment: LLM sentiment + NLP features."""
from __future__ import annotations

import logging
from typing import Any, Protocol

from .lexicon import LexiconSentimentAgent
from .nlp_features import extract_features
from .types import SentimentResult

log = logging.getLogger(__name__)


class SentimentAgent(Protocol):
    backend: str

    def score(self, title: str, body: str = "") -> SentimentResult: ...


def build_agent(settings) -> SentimentAgent:
    """``auto`` -> Gemini when a key is set, otherwise the offline lexicon.
    ``finbert`` needs the optional transformers/torch extra."""
    backend = (settings.sentiment_backend or "auto").lower()
    if backend in ("auto", "gemini") and settings.gemini_api_key:
        from .gemini import GeminiSentimentAgent

        return GeminiSentimentAgent(settings.gemini_api_key, settings.gemini_model, settings.gemini_rpm)
    if backend == "gemini":
        raise ValueError("SENTIMENT_BACKEND=gemini but GEMINI_API_KEY is empty")
    if backend == "finbert":
        from .finbert import FinBertSentimentAgent

        return FinBertSentimentAgent()
    if backend == "auto":
        log.warning("GEMINI_API_KEY not set: using the offline lexicon sentiment backend")
    return LexiconSentimentAgent()


def enrich_document(agent: SentimentAgent, doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one stored article document. Falls back to the lexicon if the
    LLM call keeps failing, so one bad article never blocks the queue."""
    title, body = doc.get("title", ""), doc.get("body", "") or ""
    try:
        result = agent.score(title, body)
    except Exception as exc:  # noqa: BLE001
        log.warning("Sentiment backend %s failed for %s (%s); using lexicon", agent.backend, doc.get("_id"), exc)
        result = LexiconSentimentAgent().score(title, body)
    return result.to_doc(), extract_features(title, body)


__all__ = ["SentimentResult", "build_agent", "enrich_document", "extract_features", "LexiconSentimentAgent"]
