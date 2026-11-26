import pytest

from cryptosent.config import Settings
from cryptosent.sentiment import build_agent, enrich_document
from cryptosent.sentiment.gemini import GeminiSentimentAgent, parse_response
from cryptosent.sentiment.lexicon import LexiconSentimentAgent, polarity
from cryptosent.sentiment.nlp_features import FEATURE_NAMES, extract_features
from cryptosent.sentiment.prompt import build_prompt, normalise_label


def test_lexicon_directions():
    agent = LexiconSentimentAgent()
    assert agent.score("Bitcoin surges to record high on ETF inflows").label == "bullish"
    assert agent.score("Bitcoin plunges as exchange hack sparks panic").label == "bearish"
    assert agent.score("Bitcoin to the moon! Don't miss the next 100x!").label == "manipulation"
    assert agent.score("Developers publish protocol notes").label == "neutral"


def test_negation_flips_polarity():
    assert polarity("not a crash")["compound"] > 0 > polarity("a crash")["compound"]


def test_nlp_features_complete():
    feats = extract_features("Elon Musk says BTC price will surge!", "The SEC and BlackRock discuss ETF inflows?")
    assert set(feats) == set(FEATURE_NAMES)
    assert feats["exclamation_count"] == 1 and feats["question_count"] == 1
    assert feats["btc_mention_count"] >= 1 and feats["ne_org_count"] >= 2 and feats["ne_person_count"] >= 1


def test_gemini_response_parsing():
    payload = {"candidates": [{"content": {"parts": [{"text": '```json\n{"label": "bullish", "score": 0.7, '
                                                              '"confidence": 0.9, "reasoning": "x"}\n```'}]}}]}
    assert parse_response(payload)["label"] == "bullish"
    with pytest.raises(ValueError):
        parse_response({"candidates": []})


def test_gemini_agent_clips_and_normalises(monkeypatch):
    agent = GeminiSentimentAgent("k", rpm=10_000)

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"label": "Bitcoin Price Manipulation", '
                                                                   '"score": 3, "confidence": -1, "reasoning": "r"}'}]}}]}

    monkeypatch.setattr(agent.session, "post", lambda *a, **k: Resp())
    res = agent.score("t", "b")
    assert (res.label, res.score, res.confidence) == ("manipulation", 1.0, 0.0)


def test_label_aliases_and_prompt():
    assert normalise_label("Bitcoin Positive Speculation") == "bullish"
    assert normalise_label("weird") == "neutral"
    assert "Article to analyze" in build_prompt("Title", "x" * 10_000)


def test_build_agent_falls_back_without_key():
    assert build_agent(Settings(gemini_api_key=None, sentiment_backend="auto")).backend == "lexicon"
    with pytest.raises(ValueError):
        build_agent(Settings(gemini_api_key=None, sentiment_backend="gemini"))


def test_enrich_document_survives_backend_failure():
    class Broken:
        backend = "broken"

        def score(self, title, body=""):
            raise RuntimeError("quota")

    sentiment, nlp = enrich_document(Broken(), {"_id": "1", "title": "Bitcoin rallies", "body": ""})
    assert sentiment["backend"] == "lexicon" and "word_count" in nlp


def test_ner_models_load_once(monkeypatch):
    """nltk.ne_chunk rebuilds its model on every call in NLTK >= 3.9, which made
    scoring take seconds per article. The chunker must be built once."""
    import nltk
    import nltk.chunk
    import nltk.tag
    from nltk.tree import Tree

    from cryptosent.sentiment import nlp_features as nf

    built = {"chunker": 0, "tagger": 0}

    class FakeChunker:
        def parse(self, tagged):
            return Tree("S", [Tree("PERSON", [tagged[0]])] if tagged else [])

    class FakeTagger:
        def __init__(self):
            built["tagger"] += 1

        def tag(self, tokens):
            return [(t, "NNP") for t in tokens]

    def fake_ne_chunker(fmt="multiclass"):
        built["chunker"] += 1
        return FakeChunker()

    monkeypatch.setattr(nltk.chunk, "ne_chunker", fake_ne_chunker)
    monkeypatch.setattr(nltk.tag, "PerceptronTagger", FakeTagger)
    monkeypatch.setattr(nltk, "word_tokenize", str.split)
    nf._nltk_ner.cache_clear()
    try:
        body = "Alice met Bob. " * 100
        for _ in range(5):
            feats = extract_features("Title here", body)
        assert built == {"chunker": 1, "tagger": 1}
        assert feats["ne_person_count"] == nf.MAX_NER_SENTENCES  # capped sentences
    finally:
        nf._nltk_ner.cache_clear()


def test_enrich_pending_logs_progress(tmp_path, caplog):
    import logging
    from datetime import datetime, timedelta, timezone

    from cryptosent.collectors.news_base import Article
    from cryptosent.pipeline import enrich_pending
    from cryptosent.storage import LocalStore

    store = LocalStore(tmp_path)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    store.upsert_articles([Article(f"https://n/{i}", f"Bitcoin {i}", t0 + timedelta(minutes=i), "rss")
                           for i in range(120)])
    with caplog.at_level(logging.INFO, logger="cryptosent.pipeline"):
        assert enrich_pending(Settings(data_dir=tmp_path), store, limit=120, backend="lexicon") == 120
    progress = [r.getMessage() for r in caplog.records if "articles scored (" in r.getMessage()]
    assert len(progress) == 2 and progress[0].startswith("enrich_pending: 50/120")
    assert progress[1].startswith("enrich_pending: 100/120")
