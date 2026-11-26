"""Gemini prompt (report appendix 7.3.1), extended so the model returns the
numeric sentiment score s in [-1, 1] that the report's maths (Chapter 3)
needs but the original prompt never asked for."""
from __future__ import annotations

LABELS = ("manipulation", "bullish", "bearish", "neutral")

# Map the report's long label names to our short ones.
LABEL_ALIASES = {
    "bitcoin price manipulation": "manipulation",
    "manipulation": "manipulation",
    "bitcoin positive speculation": "bullish",
    "positive speculation": "bullish",
    "bullish speculation": "bullish",
    "bullish": "bullish",
    "bitcoin negative speculation": "bearish",
    "negative speculation": "bearish",
    "bearish speculation": "bearish",
    "bearish": "bearish",
    "bitcoin reporting": "neutral",
    "neutral reporting": "neutral",
    "reporting": "neutral",
    "neutral": "neutral",
}

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You are an expert cryptocurrency market analyst with 10 years of experience in
sentiment analysis and detecting market manipulation. Analyze the news article
below and judge its likely effect on the Bitcoin (BTC) price over the next hour.

Classify the article into exactly one category:

1. "manipulation": sensationalism, FOMO or FUD. Exaggerated price predictions
   ("skyrocket", "plummet", "to the moon", "death cross") without robust
   technical or fundamental backing, attempts to make readers buy or sell
   immediately, or paid promotion disguised as news.
2. "bullish": a reasonable, well-substantiated case that the BTC price may
   rise (on-chain data, institutional adoption, macro factors, ETF flows).
3. "bearish": a reasonable, well-substantiated case that the BTC price may
   fall (regulation, hacks, technical resistance, liquidations, macro).
4. "neutral": factual, objective reporting without a bullish or bearish stance.

If an article shows signs of both speculation and manipulation, choose
"manipulation" when the language is overly emotional or lacks evidence.

Also give:
- "score": a number from -1.0 (very bearish for BTC price) to 1.0 (very
  bullish). Use 0 for neutral. Manipulation articles still get a score in the
  direction they push readers (FOMO > 0, FUD < 0).
- "confidence": 0.0 to 1.0, how certain you are. Official, verifiable news
  (regulatory decisions, filings, exchange data) deserves high confidence;
  rumours and opinion pieces low confidence.
- "reasoning": one or two sentences citing specific phrases.

Return only JSON matching the schema."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "enum": list(LABELS)},
        "score": {"type": "NUMBER"},
        "confidence": {"type": "NUMBER"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["label", "score", "confidence", "reasoning"],
}


def build_prompt(title: str, body: str, max_chars: int = 6000) -> str:
    text = f"Title: {title}\n\n{body}".strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return f"{SYSTEM_PROMPT}\n\n---\nArticle to analyze:\n{text}"


def normalise_label(raw: str | None) -> str:
    if not raw:
        return "neutral"
    return LABEL_ALIASES.get(raw.strip().lower(), "neutral")
