"""Article-level enrichment: LLM sentiment + NLP features."""
from __future__ import annotations

import logging
from typing import Any, Protocol

import requests

from .gemini import GeminiConfigError, redact
from .lexicon import LexiconSentimentAgent
from .nlp_features import extract_features
from .types import SentimentResult

log = logging.getLogger(__name__)


class SentimentAgent(Protocol):
    backend: str

    def score(self, title: str, body: str = "") -> SentimentResult: ...


def build_agent(settings, backend: str | None = None) -> SentimentAgent:
    """Pick the sentiment backend.

    ``auto``: Gemini when a key is set and works, otherwise the offline lexicon
    (with one clear ERROR line saying why). ``gemini``: Gemini or fail fast.
    ``finbert`` needs the optional transformers/torch extra."""
    backend = (backend or settings.sentiment_backend or "auto").lower()
    if backend in ("auto", "gemini") and settings.gemini_api_key:
        from .gemini import GeminiSentimentAgent

        agent = GeminiSentimentAgent(
            settings.gemini_api_key, settings.gemini_model, settings.gemini_rpm, settings.gemini_batch_size
        )
        try:
            agent.resolve_model()
        except GeminiConfigError as exc:
            if backend == "gemini":
                raise
            log.error("Gemini is not usable, falling back to the offline lexicon backend: %s", exc)
            return LexiconSentimentAgent()
        except requests.RequestException as exc:
            log.warning("Could not reach Gemini to check the model (%s); will try anyway", redact(exc))
        return agent
    if backend == "gemini":
        raise ValueError("SENTIMENT_BACKEND=gemini but GEMINI_API_KEY is empty")
    if backend == "finbert":
        from .finbert import FinBertSentimentAgent

        return FinBertSentimentAgent()
    if backend == "auto":
        log.warning("GEMINI_API_KEY not set: using the offline lexicon sentiment backend")
    return LexiconSentimentAgent()


def enrich_documents(
    agent: SentimentAgent, docs: list[dict[str, Any]]
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], GeminiConfigError | None]:
    """Score stored article documents.

    Returns ``(items, stop)`` where items are ``(article_id, sentiment, nlp)``
    for the articles that were scored. When the backend hits a configuration
    problem (bad key, unknown model, daily quota) scoring stops, ``stop`` says
    why, and the remaining articles are left unscored for a later run rather
    than silently filled with a weaker backend. A transient failure on a
    batch falls back to the lexicon for that batch only."""
    pairs = [(d.get("title", "") or "", d.get("body", "") or "") for d in docs]
    stop: GeminiConfigError | None = None
    results: list[SentimentResult | None]
    try:
        if hasattr(agent, "score_batch"):
            results = list(agent.score_batch(pairs))
        else:
            results = [agent.score(t, b) for t, b in pairs]
    except GeminiConfigError as exc:
        stop = exc
        results = []
    except Exception as exc:  # noqa: BLE001
        log.warning("Sentiment backend %s failed for a batch of %d (%s); using lexicon for it",
                    agent.backend, len(docs), redact(exc))
        lexicon = LexiconSentimentAgent()
        results = [lexicon.score(t, b) for t, b in pairs]
    items = []
    for doc, (title, body), res in zip(docs, pairs, results):
        if res is not None:
            items.append((doc["_id"], res.to_doc(), extract_features(title, body)))
    return items, stop


def enrich_document(agent: SentimentAgent, doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one article. Raises :class:`GeminiConfigError` on configuration
    problems; other failures fall back to the lexicon."""
    items, stop = enrich_documents(agent, [doc])
    if stop is not None:
        raise stop
    _, sentiment, nlp = items[0]
    return sentiment, nlp


__all__ = [
    "GeminiConfigError", "SentimentResult", "build_agent", "enrich_document", "enrich_documents",
    "extract_features", "LexiconSentimentAgent",
]
