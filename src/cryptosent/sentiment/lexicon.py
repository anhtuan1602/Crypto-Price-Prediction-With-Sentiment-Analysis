"""Crypto-aware lexicon used for (a) the NLP polarity features and (b) an
offline sentiment backend when no Gemini key is configured.

If NLTK's VADER lexicon is installed it is merged in; otherwise the built-in
finance word list below is used on its own, so nothing needs downloading.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from .types import SentimentResult

# Valence in roughly VADER's -4..4 range, tuned for crypto news.
CRYPTO_LEXICON: dict[str, float] = {
    # bullish
    "surge": 2.5, "surges": 2.5, "surged": 2.5, "soar": 2.8, "soars": 2.8, "soared": 2.8,
    "rally": 2.3, "rallies": 2.3, "rallied": 2.3, "gain": 1.6, "gains": 1.6, "gained": 1.6,
    "rise": 1.4, "rises": 1.4, "rising": 1.4, "rose": 1.4, "jump": 1.8, "jumps": 1.8, "jumped": 1.8,
    "climb": 1.5, "climbs": 1.5, "climbed": 1.5, "record": 1.5, "high": 0.8, "highs": 1.0,
    "ath": 2.5, "breakout": 2.0, "bullish": 2.6, "bull": 1.8, "bulls": 1.8, "inflows": 1.8,
    "inflow": 1.8, "adoption": 1.8, "approve": 2.0, "approves": 2.0, "approved": 2.0,
    "approval": 2.0, "accumulate": 1.4, "accumulation": 1.4, "buy": 0.8, "buying": 1.0,
    "recover": 1.6, "recovers": 1.6, "recovery": 1.6, "rebound": 1.8, "rebounds": 1.8,
    "upgrade": 1.4, "partnership": 1.4, "launch": 0.8, "launches": 0.8, "support": 0.6,
    "outperform": 1.8, "optimism": 2.0, "optimistic": 2.0, "strong": 1.2, "strength": 1.2,
    "reserve": 0.8, "institutional": 0.8, "etf": 0.4, "halving": 0.6, "uptrend": 2.0,
    # bearish
    "crash": -3.0, "crashes": -3.0, "crashed": -3.0, "plunge": -2.8, "plunges": -2.8,
    "plunged": -2.8, "plummet": -3.0, "plummets": -3.0, "drop": -1.6, "drops": -1.6,
    "dropped": -1.6, "fall": -1.5, "falls": -1.5, "fell": -1.5, "falling": -1.5,
    "decline": -1.6, "declines": -1.6, "declined": -1.6, "slump": -2.2, "slumps": -2.2,
    "dump": -2.4, "dumps": -2.4, "selloff": -2.4, "sell-off": -2.4, "sell": -0.8, "selling": -1.0,
    "bearish": -2.6, "bear": -1.8, "bears": -1.8, "outflows": -1.8, "outflow": -1.8,
    "liquidation": -2.0, "liquidations": -2.0, "liquidated": -2.0, "hack": -2.8, "hacked": -2.8,
    "exploit": -2.5, "exploited": -2.5, "fraud": -3.0, "scam": -3.0, "lawsuit": -2.0, "sues": -2.0,
    "ban": -2.4, "bans": -2.4, "banned": -2.4, "crackdown": -2.4, "reject": -2.0,
    "rejects": -2.0, "rejected": -2.0, "delay": -1.0, "delays": -1.0, "fear": -2.0,
    "fears": -2.0, "panic": -2.6, "risk": -0.8, "risks": -0.8, "weak": -1.2, "weakness": -1.2,
    "loss": -1.8, "losses": -1.8, "bankrupt": -3.0, "bankruptcy": -3.0, "insolvent": -2.8,
    "downtrend": -2.0, "correction": -1.2, "resistance": -0.5, "volatile": -0.6,
    "uncertainty": -1.4, "investigation": -1.6, "fine": -1.0, "fined": -1.6, "tumble": -2.4,
    "tumbles": -2.4, "tumbled": -2.4, "pessimism": -2.0, "capitulation": -2.6,
}

HYPE_PATTERNS = [
    r"to the moon", r"\bmoon(ing)?\b", r"\b\d{2,}x\b", r"skyrocket", r"explode[sd]?",
    r"don'?t miss", r"last chance", r"guarantee[ds]?", r"next 100x", r"get rich",
    r"death cross", r"bloodbath", r"collapse imminent", r"going to zero", r"massive crash",
    r"millionaire", r"once in a lifetime", r"act now", r"price prediction",
]
_HYPE_RE = re.compile("|".join(HYPE_PATTERNS), re.I)
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")
NEGATIONS = {"not", "no", "never", "without", "isn't", "wasn't", "don't", "doesn't", "fails", "failed"}


@lru_cache(maxsize=1)
def lexicon() -> dict[str, float]:
    merged: dict[str, float] = {}
    try:  # optional: VADER's general-purpose lexicon
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        merged.update(SentimentIntensityAnalyzer().lexicon)
    except Exception:  # noqa: BLE001 - lexicon not downloaded, fine
        pass
    merged.update(CRYPTO_LEXICON)
    return merged


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def polarity(text: str) -> dict[str, float]:
    """Positive/negative word ratios and a VADER-style compound score."""
    tokens = tokenize(text)
    lex = lexicon()
    pos = neg = 0
    total = 0.0
    for i, tok in enumerate(tokens):
        val = lex.get(tok)
        if val is None:
            continue
        if any(t in NEGATIONS for t in tokens[max(0, i - 3):i]):
            val = -0.74 * val
        total += val
        if val > 0:
            pos += 1
        elif val < 0:
            neg += 1
    n = max(len(tokens), 1)
    compound = total / math.sqrt(total * total + 15) if total else 0.0
    return {"pos_ratio": pos / n, "neg_ratio": neg / n, "compound": compound}


def hype_count(text: str) -> int:
    return len(_HYPE_RE.findall(text))


class LexiconSentimentAgent:
    """Deterministic, offline stand-in for the LLM. Weaker than Gemini but keeps
    the whole pipeline runnable (and testable) without an API key."""

    backend = "lexicon"

    def score(self, title: str, body: str = "") -> SentimentResult:
        # headlines carry most of the signal, weight them double
        pol_title = polarity(title)
        pol_body = polarity(body) if body else pol_title
        score = 0.6 * pol_title["compound"] + 0.4 * pol_body["compound"]
        text = f"{title} {body}"
        hype = hype_count(text)
        exclaim = text.count("!")
        if hype >= 1 and (exclaim >= 1 or hype >= 2 or abs(score) > 0.6):
            label = "manipulation"
        elif score > 0.15:
            label = "bullish"
        elif score < -0.15:
            label = "bearish"
        else:
            label = "neutral"
        words = len(tokenize(text))
        confidence = 0.35 + 0.35 * min(abs(score), 1.0) + 0.15 * min(words / 300, 1.0)
        if label == "manipulation":
            confidence = min(confidence, 0.6)
        return SentimentResult(
            score=round(score, 4),
            label=label,
            confidence=round(min(confidence, 0.9), 4),
            reasoning=f"lexicon compound={score:.2f}, hype_phrases={hype}",
            backend=self.backend,
            model="crypto-lexicon-v1",
        )
