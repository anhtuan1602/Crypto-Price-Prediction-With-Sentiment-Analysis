"""Per-article NLP features (report section 4.1.1c and appendix table 7.2):
polarity ratios, text statistics, domain keywords and named entities.

NER uses NLTK's chunker when its models are installed (the Docker image
downloads them) and otherwise a capitalised-span heuristic plus a list of
well-known crypto people and organisations.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .lexicon import hype_count, polarity, tokenize

BTC_RE = re.compile(r"\b(bitcoin|btc|xbt|sats?|satoshis?)\b", re.I)
PRICE_KEYWORDS = {
    "price", "prices", "surge", "surges", "crash", "crashes", "rally", "plunge", "plunges",
    "dump", "pump", "ath", "high", "low", "support", "resistance", "breakout", "drop",
    "drops", "soar", "soars", "slump", "rebound", "correction", "$", "usd", "percent", "%",
}
TECHNICAL_TERMS = {
    "rsi", "macd", "moving", "average", "fibonacci", "resistance", "support", "volume",
    "liquidity", "liquidation", "leverage", "futures", "options", "derivatives", "funding",
    "open", "interest", "hashrate", "hash", "mining", "on-chain", "onchain", "etf", "inflows",
    "outflows", "halving", "volatility", "short", "long", "spot", "basis", "yield",
    "treasury", "fed", "inflation", "cpi", "rate", "rates", "bollinger", "candle", "bearish",
    "bullish", "breakout", "trendline", "whale", "whales", "custody", "stablecoin", "stablecoins",
}
KNOWN_PERSONS = {
    "elon musk", "michael saylor", "gary gensler", "paul atkins", "jerome powell", "larry fink",
    "cathie wood", "changpeng zhao", "brian armstrong", "vitalik buterin", "donald trump",
    "satoshi nakamoto", "sam bankman-fried", "jack dorsey", "arthur hayes", "richard teng",
}
KNOWN_ORGS = {
    "sec", "cftc", "fed", "federal reserve", "blackrock", "fidelity", "grayscale", "coinbase",
    "binance", "kraken", "microstrategy", "strategy", "tesla", "jpmorgan", "goldman sachs",
    "tether", "circle", "ftx", "imf", "treasury", "ark invest", "bitfinex", "okx", "bybit",
    "vaneck", "nasdaq", "cme", "metaplanet", "el salvador",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CAP_SPAN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

FEATURE_NAMES = [
    "pos_ratio", "neg_ratio", "compound", "word_count", "sentence_count", "avg_sentence_length",
    "unique_word_ratio", "exclamation_count", "question_count", "btc_mention_count",
    "price_keyword_count", "technical_term_ratio", "hype_count", "ne_person_count", "ne_org_count",
]


@lru_cache(maxsize=1)
def _nltk_ner_available() -> bool:
    try:
        import nltk

        nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize("Elon Musk visited the SEC.")))
        return True
    except Exception:  # noqa: BLE001
        return False


def named_entities(text: str) -> tuple[int, int]:
    """Return (person_count, organisation_count)."""
    if _nltk_ner_available():
        import nltk

        persons = orgs = 0
        for sentence in _SENT_SPLIT.split(text)[:200]:
            tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
            for node in tree:
                label = getattr(node, "label", lambda: None)()
                if label == "PERSON":
                    persons += 1
                elif label in ("ORGANIZATION", "GPE", "FACILITY"):
                    orgs += 1
        return persons, orgs
    lower = text.lower()
    persons = sum(lower.count(p) for p in KNOWN_PERSONS)
    orgs = sum(len(re.findall(rf"\b{re.escape(o)}\b", lower)) for o in KNOWN_ORGS)
    known = KNOWN_PERSONS | KNOWN_ORGS
    spans = [s for s in _CAP_SPAN.findall(text) if s.lower() not in known]
    persons += sum(1 for s in spans if len(s.split()) == 2)  # "First Last" heuristic
    return persons, orgs


def extract_features(title: str, body: str = "") -> dict[str, float]:
    text = f"{title}. {body}" if body else title
    tokens = tokenize(text)
    n = max(len(tokens), 1)
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    pol = polarity(text)
    persons, orgs = named_entities(text)
    lowered = text.lower()
    return {
        "pos_ratio": pol["pos_ratio"],
        "neg_ratio": pol["neg_ratio"],
        "compound": pol["compound"],
        "word_count": float(len(tokens)),
        "sentence_count": float(len(sentences)),
        "avg_sentence_length": len(tokens) / max(len(sentences), 1),
        "unique_word_ratio": len(set(tokens)) / n,
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "btc_mention_count": float(len(BTC_RE.findall(text))),
        "price_keyword_count": float(
            sum(1 for t in tokens if t in PRICE_KEYWORDS) + lowered.count("$") + lowered.count("%")
        ),
        "technical_term_ratio": sum(1 for t in tokens if t in TECHNICAL_TERMS) / n,
        "hype_count": float(hype_count(text)),
        "ne_person_count": float(persons),
        "ne_org_count": float(orgs),
    }
