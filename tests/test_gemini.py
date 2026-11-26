"""Gemini client behaviour: key handling, fail-fast errors, model discovery,
batching and free-tier quota handling. No network: a fake session answers.
All keys below are fake placeholders."""
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from cryptosent.collectors.news_base import Article
from cryptosent.config import Settings
from cryptosent.pipeline import enrich_pending, rescore
from cryptosent.sentiment import build_agent
from cryptosent.sentiment import gemini as gm
from cryptosent.sentiment.gemini import GeminiConfigError, GeminiSentimentAgent, pick_flash_model, redact
from cryptosent.sentiment.lexicon import LexiconSentimentAgent
from cryptosent.storage import LocalStore

KEY = "AIza" + "FAKE" * 8  # placeholder in the classic key format
KEY_NEW_FORMAT = "AQ." + "Fake0Key0" * 5  # placeholder in the newer key format


class Resp:
    def __init__(self, status, payload=None, text=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise gm.requests.HTTPError(f"HTTP {self.status_code}")


def ok(obj):
    return Resp(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]})


def err(status, message, status_name="ERR", extra=""):
    payload = {"error": {"code": status, "status": status_name, "message": message}}
    return Resp(status, payload, json.dumps(payload) + extra)


class FakeSession:
    def __init__(self, posts=(), gets=()):
        self.posts, self.gets = list(posts), list(gets)
        self.calls = []
        self.headers = {}

    def post(self, url, json=None, timeout=None, **kw):
        self.calls.append(("POST", url, json, kw))
        return self.posts.pop(0)

    def get(self, url, params=None, timeout=None, **kw):
        self.calls.append(("GET", url, params, kw))
        return self.gets.pop(0)


def make_agent(session, **kw):
    agent = GeminiSentimentAgent(KEY, rpm=100_000, **kw)
    session.headers.update(agent.session.headers)
    agent.session = session
    return agent


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)


def item(i, label="bullish", score=0.5):
    return {"id": i, "label": label, "score": score, "confidence": 0.8, "reasoning": "r"}


def test_key_goes_in_header_never_in_url():
    s = FakeSession(posts=[ok({"label": "bullish", "score": 0.4, "confidence": 0.9, "reasoning": "x"})])
    agent = make_agent(s)
    agent.score("Bitcoin up")
    _, url, _, kw = s.calls[0]
    assert "key=" not in url and KEY not in url and "params" not in kw
    assert s.headers["x-goog-api-key"] == KEY
    assert KEY not in repr(agent)


def test_redact_removes_keys():
    text = f"404 for url: https://x/models/m:generateContent?key={KEY_NEW_FORMAT} and {KEY}"
    out = redact(text)
    assert KEY not in out and KEY_NEW_FORMAT not in out and out.count("<redacted>") == 2


GP_MODELS = {"models": [{"name": f"models/{n}", "supportedGenerationMethods": ["generateContent"]} for n in [
    "gemini-2.5-flash", "gemini-2.5-flash-preview-tts", "gemini-flash-latest", "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite", "gemini-2.5-flash-image", "gemini-3-flash-preview", "gemini-3.5-flash",
    "gemini-3.5-flash-lite", "gemini-omni-1.2-flash", "gemini-3.8-flash", "gemini-3.8-flash-tts",
]]}


def test_404_falls_back_to_flash_latest():
    """A listed model can still answer 404 on generateContent (seen live)."""
    s = FakeSession(posts=[err(404, "models/gemini-2.5-flash is not found", "NOT_FOUND"),
                           ok({"label": "bullish", "score": 0.3, "confidence": 0.8, "reasoning": "x"})],
                    gets=[Resp(200, GP_MODELS)])
    agent = make_agent(s, model="gemini-2.5-flash")
    assert agent.score("t").label == "bullish"
    assert agent.model == "gemini-flash-latest"
    assert s.calls[-1][1].endswith("/models/gemini-flash-latest:generateContent")


def test_404_chain_then_fails_fast_without_retrying(caplog):
    nf = err(404, "not found", "NOT_FOUND")
    s = FakeSession(posts=[nf, nf, nf], gets=[Resp(200, GP_MODELS)] * 3)
    agent = make_agent(s, model="gemini-2.5-flash")
    with caplog.at_level(logging.WARNING), pytest.raises(GeminiConfigError) as exc:
        agent.score("t")
    posts = [c[1].split("/models/")[1] for c in s.calls if c[0] == "POST"]
    # tried: configured, the alias, then the newest full Flash model; then stop
    assert posts == ["gemini-2.5-flash:generateContent", "gemini-flash-latest:generateContent",
                     "gemini-3.8-flash:generateContent"]
    assert exc.value.kind == "model" and "retry" not in caplog.text
    n = len(s.calls)
    with pytest.raises(GeminiConfigError):  # stays disabled: no further HTTP calls
        agent.score("t")
    assert len(s.calls) == n


def test_resolve_model_switches_to_available_flash():
    models = {"models": [
        {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3-flash-image", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
    ]}
    s = FakeSession(gets=[err(404, "not found", "NOT_FOUND"), Resp(200, models)])
    agent = make_agent(s, model="gemini-2.5-flash")
    assert agent.resolve_model() == "gemini-3-flash"
    assert all("key=" not in c[1] for c in s.calls)


def test_resolve_model_bad_key():
    s = FakeSession(gets=[err(400, "API key not valid", "INVALID_ARGUMENT")])
    with pytest.raises(GeminiConfigError) as exc:
        make_agent(s).resolve_model()
    assert exc.value.kind == "auth" and "aistudio.google.com/apikey" in str(exc.value)


@pytest.mark.parametrize("names,expected", [
    (["models/gemini-flash-latest", "models/gemini-3-flash"], "gemini-flash-latest"),
    (["gemini-2.5-flash-lite", "gemini-2.5-flash"], "gemini-2.5-flash"),
    (["gemini-3-flash-preview", "gemini-2.5-flash"], "gemini-2.5-flash"),
    (["gemini-2.5-flash-001", "gemini-2.5-flash"], "gemini-2.5-flash"),
    (["gemini-2.5-pro", "gemini-2.5-flash-preview-tts"], None),
])
def test_pick_flash_model(names, expected):
    assert pick_flash_model(names) == expected


def test_per_minute_429_waits_then_succeeds(monkeypatch):
    waits = []
    monkeypatch.setattr(gm.time, "sleep", waits.append)
    s = FakeSession(posts=[err(429, "Resource exhausted", "RESOURCE_EXHAUSTED", extra=' "retryDelay": "21s"'),
                           ok({"label": "bearish", "score": -0.6, "confidence": 0.7, "reasoning": "x"})])
    assert make_agent(s).score("t").label == "bearish"
    assert 22.0 in waits


def test_daily_quota_stops_and_disables():
    s = FakeSession(posts=[err(429, "Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                               "RESOURCE_EXHAUSTED")])
    agent = make_agent(s)
    with pytest.raises(GeminiConfigError) as exc:
        agent.score("t")
    assert exc.value.kind == "quota" and len(s.calls) == 1


def test_thinking_config_dropped_when_rejected():
    s = FakeSession(posts=[err(400, "Thinking budget is not supported for this model", "INVALID_ARGUMENT"),
                           ok({"label": "neutral", "score": 0, "confidence": 0.5, "reasoning": "x"})])
    assert make_agent(s).score("t").label == "neutral"
    assert "thinkingConfig" in s.calls[0][2]["generationConfig"]
    assert "thinkingConfig" not in s.calls[1][2]["generationConfig"]


def test_score_batch_one_request_and_fills_gaps():
    s = FakeSession(posts=[ok([item(0), item(2, "bearish", -0.3)]),  # id 1 missing
                           ok({"label": "neutral", "score": 0.0, "confidence": 0.5, "reasoning": "x"})])
    agent = make_agent(s, batch_size=10)
    res = agent.score_batch([("a", ""), ("b", ""), ("c", "")])
    assert [r.label for r in res] == ["bullish", "neutral", "bearish"]
    assert len(s.calls) == 2
    prompt = s.calls[0][2]["contents"][0]["parts"][0]["text"]
    assert "<article id=2>" in prompt and s.calls[0][2]["generationConfig"]["responseSchema"]["type"] == "ARRAY"


def test_build_agent_falls_back_once_on_bad_key(monkeypatch, caplog):
    s = FakeSession(gets=[err(403, "Permission denied", "PERMISSION_DENIED")])
    monkeypatch.setattr(gm.requests, "Session", lambda: s)
    with caplog.at_level(logging.ERROR):
        agent = build_agent(Settings(gemini_api_key=KEY, sentiment_backend="auto"))
    assert isinstance(agent, LexiconSentimentAgent)
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
    assert KEY not in caplog.text
    s2 = FakeSession(gets=[err(403, "Permission denied", "PERMISSION_DENIED")])
    monkeypatch.setattr(gm.requests, "Session", lambda: s2)
    with pytest.raises(GeminiConfigError):
        build_agent(Settings(gemini_api_key=KEY, sentiment_backend="gemini"))


def _store_with_articles(tmp_path, n):
    store = LocalStore(tmp_path)
    t0 = datetime(2025, 12, 13, tzinfo=timezone.utc)
    store.upsert_articles([Article(f"https://n/{i}", f"Bitcoin story {i}", t0 + timedelta(minutes=i), "rss")
                           for i in range(n)])
    return store


def test_enrich_pending_stops_on_quota_and_keeps_rest(tmp_path, caplog):
    store = _store_with_articles(tmp_path, 25)
    quota = err(429, "GenerateRequestsPerDay quota exceeded", "RESOURCE_EXHAUSTED")
    s = FakeSession(posts=[ok([item(i) for i in range(10)]), quota])
    agent = make_agent(s, batch_size=10)
    settings = Settings(data_dir=tmp_path, gemini_api_key=KEY)
    with caplog.at_level(logging.WARNING):
        done = enrich_pending(settings, store, limit=100, agent=agent)
    assert done == 10 and len(store.articles_missing_sentiment(100)) == 15
    assert len(s.calls) == 2  # stopped immediately, no retries
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1


def test_rescore_upgrades_lexicon_scores(tmp_path, monkeypatch):
    import cryptosent.pipeline as pl

    store = _store_with_articles(tmp_path, 5)
    settings = Settings(data_dir=tmp_path)
    enrich_pending(settings, store, limit=10, agent=LexiconSentimentAgent())
    s = FakeSession(posts=[ok([item(i, "bearish", -0.9) for i in range(5)])])
    agent = make_agent(s, batch_size=10)
    monkeypatch.setattr(pl, "build_agent", lambda *a, **k: agent)
    assert rescore(settings, store, "lexicon", limit=50) == 5
    backends = set(store.get_articles()["sentiment_backend"])
    assert backends == {"gemini"}


def test_400_retries_once_with_minimal_request():
    s = FakeSession(posts=[err(400, "Invalid JSON payload received. Unknown name \"topK\"", "INVALID_ARGUMENT"),
                           ok({"label": "bullish", "score": 0.2, "confidence": 0.6, "reasoning": "x"})])
    assert make_agent(s).score("t").label == "bullish"
    first, second = s.calls[0][2]["generationConfig"], s.calls[1][2]["generationConfig"]
    assert "topK" in first and "responseSchema" in first
    assert set(second) == {"temperature", "responseMimeType"}


def test_repeated_400_fails_fast_after_two_calls():
    bad = err(400, "Request contains an invalid argument.", "INVALID_ARGUMENT")
    s = FakeSession(posts=[bad, bad])
    agent = make_agent(s)
    with pytest.raises(GeminiConfigError) as exc:
        agent.score("t")
    assert exc.value.kind == "config" and len(s.calls) == 2


def test_invalid_key_400_is_not_retried():
    s = FakeSession(posts=[err(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")])
    with pytest.raises(GeminiConfigError) as exc:
        make_agent(s).score("t")
    assert exc.value.kind == "auth" and len(s.calls) == 1


def test_enrich_pending_auto_switches_to_lexicon_on_bad_request(tmp_path, caplog):
    store = _store_with_articles(tmp_path, 25)
    bad = err(400, "Request contains an invalid argument.", "INVALID_ARGUMENT")
    s = FakeSession(posts=[bad, bad])
    agent = make_agent(s, batch_size=10)
    settings = Settings(data_dir=tmp_path, gemini_api_key=KEY, sentiment_backend="auto")
    with caplog.at_level(logging.WARNING):
        done = enrich_pending(settings, store, limit=100, agent=agent)
    assert done == 25 and not store.articles_missing_sentiment(100)
    assert set(store.get_articles()["sentiment_backend"]) == {"lexicon"}
    assert len(s.calls) == 2  # one minimal retry, then no more Gemini calls
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
    assert KEY not in caplog.text
