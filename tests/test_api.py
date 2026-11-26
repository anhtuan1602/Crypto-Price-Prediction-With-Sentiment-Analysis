import pytest
from fastapi.testclient import TestClient

from cryptosent.config import Settings
from cryptosent.pipeline import run_demo
from cryptosent.storage import LocalStore


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    mp = pytest.MonkeyPatch()
    mp.setenv("DATA_DIR", str(root / "data"))
    mp.setenv("ARTIFACTS_DIR", str(root / "artifacts"))
    mp.delenv("MONGO_URI", raising=False)
    settings = Settings()
    run_demo(settings, LocalStore(settings.data_dir), days=8, models=["ridge"])
    from cryptosent.api import app as app_module

    app_module.store.cache_clear()
    yield TestClient(app_module.app)
    mp.undo()


def test_endpoints(client):
    assert client.get("/api/health").json()["status"] == "ok"
    prices = client.get("/api/prices?hours=6").json()
    assert prices and {"t", "price"} <= set(prices[0])
    news = client.get("/api/news?limit=5").json()
    assert len(news) == 5 and news[0]["sentiment_label"] in ("bullish", "bearish", "neutral", "manipulation")
    sent = client.get("/api/sentiment?hours=6").json()
    assert sent and -1 <= sent[-1]["sent_market"] <= 1
    model = client.get("/api/model").json()
    assert model["best_model"] == "ridge" and "technical_only" in model["test"]
    assert client.get("/api/test-predictions").json()
    assert "BTC sentiment monitor" in client.get("/").text
