"""Optional local transformer backend (ProsusAI/finbert).

Install with ``pip install -e .[finbert]``. It does not detect manipulation,
so it only emits bullish / bearish / neutral labels.
"""
from __future__ import annotations

from .types import SentimentResult


class FinBertSentimentAgent:
    backend = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        from transformers import pipeline  # heavy import, keep it lazy

        self.model_name = model_name
        self.pipe = pipeline("text-classification", model=model_name, top_k=None, truncation=True)

    def score(self, title: str, body: str = "") -> SentimentResult:
        text = f"{title}. {body}"[:2000]
        scores = {r["label"].lower(): r["score"] for r in self.pipe(text)[0]}
        pos, neg, neu = scores.get("positive", 0.0), scores.get("negative", 0.0), scores.get("neutral", 0.0)
        value = pos - neg
        label = "bullish" if value > 0.15 else "bearish" if value < -0.15 else "neutral"
        return SentimentResult(
            score=round(value, 4),
            label=label,
            confidence=round(max(pos, neg, neu), 4),
            reasoning=f"finbert pos={pos:.2f} neg={neg:.2f} neu={neu:.2f}",
            backend=self.backend,
            model=self.model_name,
        )
