"""FastAPI backend for the dashboard (report: frontend/main.html)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..features.sentiment_agg import aggregate_windows
from ..pipeline import json_safe
from ..storage import Store, get_store

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Crypto Sentiment Predictor", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def store() -> Store:
    return get_store(get_settings())


def _anchor(hours: int) -> tuple[datetime, datetime]:
    """Window ending at the newest stored candle (works for live and demo data)."""
    last = store().last_price_time() or datetime.now(timezone.utc)
    end = pd.Timestamp(last).to_pydatetime() + timedelta(minutes=1)
    return end - timedelta(hours=hours), end


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    last = store().last_price_time()
    return {"status": "ok", "last_price_time": last.isoformat() if last else None}


@app.get("/api/prices")
def prices(hours: int = Query(24, ge=1, le=24 * 30)) -> list[dict[str, Any]]:
    start, end = _anchor(hours)
    df = store().get_prices(start=start, end=end)
    if df.empty:
        return []
    step = max(1, len(df) // 1500)  # keep the payload small
    df = df.iloc[::step]
    return [{"t": (ts + pd.Timedelta(minutes=1)).isoformat(), "price": float(r["close"]), "volume": float(r["volume"])}
            for ts, r in df.iterrows()]


@app.get("/api/news")
def news(limit: int = Query(50, ge=1, le=500), hours: int = Query(72, ge=1, le=24 * 30)) -> list[dict[str, Any]]:
    start, end = _anchor(hours)
    df = store().get_articles(start=start, end=end)
    if df.empty:
        return []
    df = df.sort_values("published_at", ascending=False).head(limit)
    cols = ["title", "url", "publisher", "source", "published_at", "sentiment_score", "sentiment_label",
            "sentiment_confidence", "sentiment_backend"]
    out = df[[c for c in cols if c in df]].replace({np.nan: None})
    out["published_at"] = out["published_at"].map(lambda t: t.isoformat())
    return out.to_dict(orient="records")


@app.get("/api/sentiment")
def sentiment(hours: int = Query(24, ge=1, le=24 * 30)) -> list[dict[str, Any]]:
    """Confidence-weighted market sentiment (report S_market) over a rolling
    observation window, sampled on the decision grid."""
    settings = get_settings()
    w = settings.windows
    start, end = _anchor(hours)
    arts = store().get_articles(start=start - timedelta(minutes=w.obs), end=end)
    grid = pd.date_range(pd.Timestamp(start).ceil(f"{w.step}min"), end, freq=f"{w.step}min", tz="UTC")
    if len(grid) == 0:
        return []
    agg = aggregate_windows(arts, grid, w.obs)
    return [{"t": t.isoformat(), "sent_market": float(r["sent_market"]), "n_articles": int(r["sent_n_articles"]),
             "n_manipulation": int(r["sent_n_manipulation"])} for t, r in agg.iterrows()]


@app.get("/api/predictions")
def predictions(limit: int = Query(300, ge=1, le=5000)) -> list[dict[str, Any]]:
    return json_safe(store().get_predictions(limit=limit))


@app.get("/api/model")
def model() -> dict[str, Any]:
    run = store().latest_model_run()
    if run:
        return json_safe(run)
    report = Path(get_settings().artifacts_dir) / "report.json"
    if report.exists():
        import json

        return json_safe(json.loads(report.read_text()))
    return {}


@app.get("/api/test-predictions")
def test_predictions() -> list[dict[str, Any]]:
    """Held-out test-set predictions of the last experiment (for the chart
    before live predictions accumulate)."""
    path = Path(get_settings().artifacts_dir) / "test_predictions.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, index_col=0)
    col = "pred_technical_plus_news" if "pred_technical_plus_news" in df else df.columns[-1]
    return [{"t": t, "actual": float(r["actual"]), "predicted": float(r[col])} for t, r in df.iterrows()]
