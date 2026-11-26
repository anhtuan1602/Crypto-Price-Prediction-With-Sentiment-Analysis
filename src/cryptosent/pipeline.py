"""High-level tasks shared by the CLI, the Airflow DAGs and the services."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .collectors import BinancePriceCollector, CryptoCompareCollector, PoliteSession, build_news_collectors, dedupe
from .config import Settings
from .features import TARGET, build_dataset
from .models.train import ExperimentConfig, fit_final, log_mlflow, run_experiment, save_outputs
from .sentiment import build_agent, enrich_document
from .storage import Store

log = logging.getLogger(__name__)


# -- collection -----------------------------------------------------------
def crawl_news(settings: Settings, store: Store, publish: bool = True) -> dict[str, int]:
    """Run all live news collectors once. New articles go to MongoDB and, when
    Kafka is configured, to the raw topic for the sentiment consumer."""
    articles = []
    stats: dict[str, int] = {}
    for collector in build_news_collectors(settings):
        try:
            batch = collector.crawl()
        except Exception:  # noqa: BLE001 - one broken site must not fail the run
            log.exception("Collector %s failed", collector.name)
            batch = []
        stats[collector.name] = len(batch)
        articles.extend(batch)
    articles = dedupe(articles)
    known = set()
    if hasattr(store, "db"):
        ids = [a.article_id for a in articles]
        known = {d["_id"] for d in store.db.articles.find({"_id": {"$in": ids}}, {"_id": 1})}
    new_articles = [a for a in articles if a.article_id not in known]
    stats["new"] = store.upsert_articles(new_articles)
    if publish and settings.kafka_bootstrap and new_articles:
        from .streaming import make_producer, publish_articles

        producer = make_producer(settings.kafka_bootstrap)
        stats["published"] = publish_articles(producer, settings.topic_raw, new_articles)
        producer.close()
    log.info("crawl_news: %s", stats)
    return stats


def backfill_news(settings: Settings, store: Store, days: int) -> int:
    session = PoliteSession(proxies=settings.http_proxies, timeout=settings.request_timeout)
    collector = CryptoCompareCollector(session=session, api_key=settings.cryptocompare_api_key)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    articles = collector.backfill(start)
    return store.upsert_articles(articles)


def fetch_prices(settings: Settings, store: Store, backfill_days: int | None = None) -> int:
    """Incremental 1m candle sync; ``backfill_days`` forces a deeper start."""
    collector = BinancePriceCollector(settings.binance_base_url, settings.symbol)
    now = datetime.now(timezone.utc)
    last = store.last_price_time()
    if backfill_days is not None:
        start = now - timedelta(days=backfill_days)
    elif last is not None:
        start = last - timedelta(minutes=5)
    else:
        start = now - timedelta(days=1)
    df = collector.fetch_range(start, now)
    return store.upsert_prices(df)


# -- enrichment -----------------------------------------------------------
def enrich_pending(settings: Settings, store: Store, limit: int = 200, agent=None,
                   min_age_minutes: int = 0) -> int:
    """Score articles that have no sentiment yet (used when Kafka is off, and
    as a catch-up job for anything the consumer missed). ``min_age_minutes``
    skips freshly crawled articles the Kafka consumer is still handling."""
    from .collectors.news_base import to_utc

    agent = agent or build_agent(settings)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)
    done = 0
    while True:
        docs = store.articles_missing_sentiment(limit=min(limit - done, 100))
        if min_age_minutes:
            docs = [d for d in docs if d.get("fetched_at") is None or to_utc(d["fetched_at"]) <= cutoff]
        if not docs:
            break
        items = []
        for doc in docs:
            sentiment, nlp = enrich_document(agent, doc)
            items.append((doc["_id"], sentiment, nlp))
        store.set_enrichment_many(items)
        done += len(items)
        if done >= limit:
            break
    log.info("enrich_pending: scored %d articles with %s", done, agent.backend)
    return done


# -- modelling ------------------------------------------------------------
def load_dataset(settings: Settings, store: Store, days: int | None = None, include_unlabelled: bool = False):
    start = datetime.now(timezone.utc) - timedelta(days=days) if days else None
    candles = store.get_prices(start=start)
    articles_start = None if start is None else start - timedelta(hours=7)
    articles = store.get_articles(start=articles_start)
    return build_dataset(
        candles, articles, settings.windows,
        flat_threshold_pct=settings.target_flat_threshold,
        include_unlabelled=include_unlabelled,
    )


def train(settings: Settings, store: Store, days: int | None = None, models: list[str] | None = None) -> dict[str, Any]:
    ds = load_dataset(settings, store, days)
    cfg = ExperimentConfig(random_state=settings.random_state, fee_pct=settings.fee_pct)
    if models:
        cfg.models = models
    report = run_experiment(ds, cfg)
    final = fit_final(ds, report["best_model"], cfg)
    paths = save_outputs(report, final, Path(settings.artifacts_dir))
    summary = {
        "best_model": report["best_model"],
        "test": report["test"],
        "test_baselines": report["test_baselines"],
        "backtest": report["backtest"],
        "samples": report["samples"],
        "windows": report["windows"],
        "feature_importance": report["feature_importance"],
        "sanity": {k: v for k, v in report["sanity"].items() if k != "timestamp_audit"},
    }
    store.save_model_run(summary)
    if settings.mlflow_uri:
        try:
            log_mlflow(report, paths, settings.mlflow_uri, settings.mlflow_experiment)
        except Exception:  # noqa: BLE001 - tracking must not fail training
            log.exception("MLflow logging failed")
    log.info("Saved %s", {k: str(v) for k, v in paths.items()})
    return report


def predict_latest(settings: Settings, store: Store, lookback_hours: int = 12,
                   now: datetime | None = None) -> list[dict[str, Any]]:
    """Predict for recent decision times and back-fill actuals once known."""
    model_path = Path(settings.artifacts_dir) / "model.joblib"
    if not model_path.exists():
        log.warning("No trained model at %s", model_path)
        return []
    bundle = joblib.load(model_path)
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(hours=lookback_hours)
    candles = store.get_prices(start=start - timedelta(hours=2), end=now)
    articles = store.get_articles(start=start - timedelta(hours=8), end=now)
    ds = build_dataset(candles, articles, settings.windows, include_unlabelled=True)
    df = ds.frame[ds.frame.index >= pd.Timestamp(start)]
    if df.empty:
        return []
    preds = bundle["model"].predict(df[bundle["features"]])
    rows = []
    for (t, row), p in zip(df.iterrows(), preds):
        actual = row.get(TARGET)
        rows.append({
            "decision_time": t.isoformat(),
            "target_start": row["target_start"].isoformat(),
            "target_end": row["target_end"].isoformat(),
            "predicted_pct": float(p),
            "actual_pct": None if pd.isna(actual) else float(actual),
            "price_at_t": float(row["price_at_t"]),
            "sent_market": float(row.get("sent_market", 0.0)),
            "n_articles": int(row.get("sent_n_articles", 0)),
            "model": bundle["model_name"],
            "signal": "long" if p > 2 * settings.fee_pct else "short" if p < -2 * settings.fee_pct else "flat",
        })
    store.save_predictions(rows)
    return rows


def run_demo(settings: Settings, store: Store, days: int = 30, models: list[str] | None = None) -> dict[str, Any]:
    """End-to-end offline run on synthetic data (no network needed)."""
    from .demo import DemoConfig, generate
    from .sentiment.lexicon import LexiconSentimentAgent

    candles, articles, _ = generate(DemoConfig(days=days, seed=settings.random_state))
    store.upsert_prices(candles)
    store.upsert_articles(articles)
    enrich_pending(settings, store, limit=10**9, agent=LexiconSentimentAgent())
    report = train(settings, store, models=models)
    # replay the last day as if it were live, so the dashboard has predictions
    last = store.last_price_time()
    if last is not None:
        predict_latest(settings, store, lookback_hours=24, now=last + timedelta(minutes=1))
    return report


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
