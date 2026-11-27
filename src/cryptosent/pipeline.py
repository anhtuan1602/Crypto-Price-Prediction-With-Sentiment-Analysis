"""High-level tasks shared by the CLI, the Airflow DAGs and the services."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .collectors import (
    BinancePriceCollector,
    PoliteSession,
    build_backfill_collectors,
    build_news_collectors,
    dedupe,
    fetch_full_text,
)
from .config import Settings
from .features import TARGET, build_dataset
from .models.train import ExperimentConfig, fit_final, log_mlflow, run_experiment, save_outputs
from .sentiment import LexiconSentimentAgent, build_agent, enrich_documents
from .storage import Store

log = logging.getLogger(__name__)
PROGRESS_EVERY = 50  # articles between progress log lines


# -- collection -----------------------------------------------------------
def check_sources(settings: Settings) -> list[dict[str, Any]]:
    """Fetch every live source once and report what it returns (no writes)."""
    from .collectors import CoinMarketCapCrawler, RSSCollector, feeds_from_settings

    session = PoliteSession(proxies=settings.http_proxies, timeout=settings.request_timeout, retries=1)
    rows: list[dict[str, Any]] = []
    if "coinmarketcap" in settings.news_sources:
        cmc = CoinMarketCapCrawler(session=session, fetch_bodies=False)
        api = cmc.fetch_api()
        html = [] if api else cmc.fetch_html()
        arts = api or html
        rows.append({"source": "coinmarketcap", "status": "api" if api else "html" if html else "-",
                     "entries": len(arts), "kept": len(arts),
                     "newest": max((a.published_at for a in arts), default=None),
                     "error": None if arts else "no articles from the content API or the HTML page"})
    if "rss" in settings.news_sources:
        rss = RSSCollector(session=session,
                           feeds=feeds_from_settings(settings.rss_feeds_extra, settings.rss_feeds_disabled))
        for name, url in rss.feeds.items():
            arts, rep = rss.check_feed(name, url)
            rows.append({"source": name, "status": rep["status"], "entries": rep["entries"], "kept": rep["kept"],
                         "newest": max((a.published_at for a in arts), default=None), "error": rep["error"]})
    return rows


def article_stats(store: Store, hours: int = 24) -> dict[str, Any]:
    """Counts of articles collected in the last ``hours`` (by fetched_at), per
    publisher and per day, plus scoring status."""
    df = store.get_articles()
    if df.empty:
        return {"total": 0}
    fetched = pd.to_datetime(df.get("fetched_at"), utc=True)
    df = df.assign(fetched_at=fetched.fillna(df["published_at"]))
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    recent = df[df["fetched_at"] >= since]
    backend = df.get("sentiment_backend")
    per_day = df[df["fetched_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)]
    return {
        "total": len(df),
        f"collected_last_{hours}h": len(recent),
        f"per_publisher_last_{hours}h": recent["publisher"].fillna(recent["source"]).value_counts().to_dict(),
        "per_day_utc": per_day["fetched_at"].dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict(),
        "scored_by": backend.fillna("unscored").value_counts().to_dict() if backend is not None else {},
    }

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
    known: set[str] = set()
    ids = [a.article_id for a in articles]
    if hasattr(store, "db"):
        known = {d["_id"] for d in store.db.articles.find({"_id": {"$in": ids}}, {"_id": 1})}
    elif hasattr(store, "_articles"):
        known = set(ids) & set(store._articles())
    new_articles = [a for a in articles if a.article_id not in known]
    if settings.fetch_full_text and new_articles:
        session = PoliteSession(proxies=settings.http_proxies, timeout=min(settings.request_timeout, 15), retries=1)
        stats["full_text_fetched"] = fetch_full_text(session, new_articles)
    stats["new"] = store.upsert_articles(new_articles)
    if publish and settings.kafka_bootstrap and new_articles:
        from .streaming import make_producer, publish_articles

        producer = make_producer(settings.kafka_bootstrap)
        stats["published"] = publish_articles(producer, settings.topic_raw, new_articles)
        producer.close()
    log.info("crawl_news: %s", stats)
    return stats


def backfill_news(settings: Settings, store: Store, days: int) -> int:
    """Load ``days`` of historical news from every configured backfill source.
    One failing source is logged and skipped; if all fail, the error is raised
    so the Airflow task shows it."""
    collectors = build_backfill_collectors(settings)
    if not collectors:
        raise ValueError("No news backfill source configured (NEWS_BACKFILL_SOURCES)")
    start = datetime.now(timezone.utc) - timedelta(days=days)
    stored = 0
    errors: list[Exception] = []
    for collector in collectors:
        try:
            articles = collector.backfill(start)
        except Exception as exc:  # noqa: BLE001 - try the other sources
            log.error("backfill_news: %s failed: %s", collector.name, exc)
            errors.append(exc)
            continue
        new = store.upsert_articles(articles)
        stored += new
        log.info("backfill_news: %s fetched %d articles for %d days, %d new",
                 collector.name, len(articles), days, new)
    if len(errors) == len(collectors):
        raise errors[0]
    return stored


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
                   min_age_minutes: int = 0, backend: str | None = None) -> int:
    """Score articles that have no sentiment yet (used when Kafka is off, and
    as a catch-up job for anything the consumer missed).

    ``min_age_minutes`` skips freshly crawled articles the Kafka consumer is
    still handling. ``backend`` overrides SENTIMENT_BACKEND for this run (for
    example ``lexicon`` for a large backfill on the Gemini free tier). If the
    backend reports a configuration problem or an exhausted daily quota, the
    run stops with one log line and the rest stay unscored for the next run."""
    from .collectors.news_base import to_utc

    agent = agent or build_agent(settings, backend)
    mode = (backend or settings.sentiment_backend or "auto").lower()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)
    done = 0
    started = time.monotonic()
    next_report = PROGRESS_EVERY
    log.info("enrich_pending: scoring up to %d articles with %s", limit, agent.backend)
    while done < limit:
        docs = store.articles_missing_sentiment(limit=min(limit - done, 100))
        if min_age_minutes:
            docs = [d for d in docs if d.get("fetched_at") is None or to_utc(d["fetched_at"]) <= cutoff]
        if not docs:
            break
        stop = None
        chunk = max(1, getattr(agent, "batch_size", 50))
        for i in range(0, len(docs), chunk):
            items, stop = enrich_documents(agent, docs[i:i + chunk])
            if items:
                store.set_enrichment_many(items)
                done += len(items)
            if done >= next_report:
                rate = done / max(time.monotonic() - started, 1e-6)
                log.info("enrich_pending: %d/%d articles scored (%.1f/s, backend=%s)",
                         done, limit, rate, agent.backend)
                next_report = done + PROGRESS_EVERY
            if stop is not None:
                break
        if stop is None:
            continue
        if stop.kind != "quota" and mode == "auto":
            # broken setup (bad request, key, model): one clear line, then finish
            # the run offline; `cryptosent rescore` upgrades these later
            log.error("Gemini is not usable (%s); scoring the rest of this run with the lexicon backend", stop)
            agent = LexiconSentimentAgent()
            continue
        log.error("Stopped scoring after %d articles: %s", done, stop)
        if stop.kind != "quota" and mode == "gemini":
            raise stop
        break
    log.info("enrich_pending: scored %d articles with %s in %.0fs", done, agent.backend, time.monotonic() - started)
    return done


def rescore(settings: Settings, store: Store, from_backend: str = "lexicon", limit: int = 500) -> int:
    """Re-score articles that an earlier run scored with ``from_backend`` (for
    example after a Gemini outage, or gradually upgrading a lexicon backfill
    within the free-tier quota). Newest articles first."""
    agent = build_agent(settings)
    if agent.backend == from_backend:
        raise ValueError(f"The active backend is {agent.backend}; configure a different one to rescore")
    docs = store.articles_scored_by(from_backend, limit)
    done = 0
    chunk = max(1, getattr(agent, "batch_size", 50))
    for i in range(0, len(docs), chunk):
        items, stop = enrich_documents(agent, docs[i:i + chunk])
        items = [it for it in items if it[1].get("backend") != from_backend]  # skip batch fallbacks
        if items:
            store.set_enrichment_many(items)  # overwrites; old scores stay until replaced
            done += len(items)
        if stop is not None:
            log.error("Stopped rescoring after %d articles: %s", done, stop)
            break
    return done


# -- modelling ------------------------------------------------------------
def live_news_start(settings: Settings, store: Store) -> datetime | None:
    """When live news collection began: the earliest ``fetched_at`` of an
    article from a live source (NEWS_SOURCES). None if there is none."""
    articles = store.get_articles()
    if articles.empty or "fetched_at" not in articles:
        return None
    live = articles[articles["source"].isin(settings.news_sources)]["fetched_at"].dropna()
    if live.empty:
        return None
    return pd.Timestamp(live.min()).to_pydatetime()


def training_start(settings: Settings, store: Store, days: int | None = None) -> datetime | None:
    start = datetime.now(timezone.utc) - timedelta(days=days) if days else None
    mode = (settings.train_start or "auto").strip().lower()
    if mode == "all":
        return start
    if mode == "auto":
        floor = live_news_start(settings, store)
    else:
        ts = pd.Timestamp(settings.train_start)
        floor = (ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")).to_pydatetime()
    if floor is None:
        return start
    return floor if start is None else max(start, floor)


def load_dataset(settings: Settings, store: Store, days: int | None = None, include_unlabelled: bool = False,
                 start: datetime | None = None):
    """Leakage-safe modelling table. Rows start at ``start`` (or ``days`` ago);
    prices and news are loaded a few hours earlier for indicator warm-up and
    the observation window."""
    if start is None and days:
        start = datetime.now(timezone.utc) - timedelta(days=days)
    candles = store.get_prices(start=None if start is None else start - timedelta(hours=3))
    articles = store.get_articles(start=None if start is None else start - timedelta(hours=7))
    ds = build_dataset(
        candles, articles, settings.windows,
        flat_threshold_pct=settings.target_flat_threshold,
        include_unlabelled=include_unlabelled,
    )
    if start is not None and not ds.frame.empty:
        ds.frame = ds.frame[ds.frame.index >= pd.Timestamp(start)]
    return ds


MIN_TRAIN_SAMPLES = 200  # about 33 hours of 10-minute decision times


class NotEnoughData(ValueError):
    pass


def train(settings: Settings, store: Store, days: int | None = None, models: list[str] | None = None) -> dict[str, Any]:
    start = training_start(settings, store, days)
    ds = load_dataset(settings, store, start=start)
    labelled = int(ds.frame[TARGET].notna().sum()) if TARGET in ds.frame else 0
    log.info("train: %d labelled samples from %s (TRAIN_START=%s)", labelled, start, settings.train_start)
    if labelled < MIN_TRAIN_SAMPLES:
        raise NotEnoughData(f"Only {labelled} labelled samples since {start}; need {MIN_TRAIN_SAMPLES} "
                            f"(about {MIN_TRAIN_SAMPLES * settings.windows.step / 60:.0f} hours of live collection)")
    cfg = ExperimentConfig(random_state=settings.random_state, fee_pct=settings.fee_pct)
    if labelled < 1500:
        cfg.n_splits = 3  # small live dataset: fewer, larger folds
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
