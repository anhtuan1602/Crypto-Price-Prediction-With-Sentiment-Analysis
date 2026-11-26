"""Airflow DAGs for the crypto sentiment pipeline (report section 4.1.1a).

  cmc_news_10m           every 10 min: crawl CoinMarketCap + RSS -> MongoDB + Kafka
  btc_prices_5m          every 5 min:  sync Binance 1m candles
  predict_10m            every 10 min: score the latest decision time
  sentiment_catchup_1h   hourly:       score anything the Kafka consumer missed, then
                                       upgrade some lexicon-scored articles to Gemini
  train_daily            02:00 UTC:    news backfill, retrain, evaluate, log to MLflow
  backfill_history       manual:       load N days of prices + news (run once at setup)

All heavy lifting lives in the ``cryptosent`` package; tasks are thin wrappers.
Each task builds its own store/settings so it is safe on any worker.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param

DEFAULT_ARGS = {
    "owner": "cryptosent",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=30),
}
START = datetime(2025, 1, 1)


def _ctx():
    from cryptosent.config import get_settings
    from cryptosent.storage import get_store

    settings = get_settings()
    return settings, get_store(settings)


@dag(dag_id="cmc_news_10m", schedule="*/10 * * * *", start_date=START, catchup=False,
     max_active_runs=1, default_args=DEFAULT_ARGS, tags=["crawl"])
def cmc_news_10m():
    @task
    def crawl() -> dict:
        from cryptosent.pipeline import crawl_news

        settings, store = _ctx()
        stats = crawl_news(settings, store, publish=True)
        if not settings.kafka_bootstrap:
            # no Kafka configured: score inline so the pipeline still works
            from cryptosent.pipeline import enrich_pending

            stats["scored_inline"] = enrich_pending(settings, store, limit=200)
        return stats

    crawl()


@dag(dag_id="btc_prices_5m", schedule="*/5 * * * *", start_date=START, catchup=False,
     max_active_runs=1, default_args=DEFAULT_ARGS, tags=["prices"])
def btc_prices_5m():
    @task
    def sync() -> int:
        from cryptosent.pipeline import fetch_prices

        settings, store = _ctx()
        return fetch_prices(settings, store)

    sync()


@dag(dag_id="predict_10m", schedule="1-59/10 * * * *", start_date=START, catchup=False,
     max_active_runs=1, default_args=DEFAULT_ARGS, tags=["model"])
def predict_10m():
    @task
    def refresh_prices() -> int:
        from cryptosent.pipeline import fetch_prices

        settings, store = _ctx()
        return fetch_prices(settings, store)

    @task
    def predict(_: int) -> int:
        from cryptosent.pipeline import predict_latest

        settings, store = _ctx()
        return len(predict_latest(settings, store, lookback_hours=6))

    predict(refresh_prices())


@dag(dag_id="sentiment_catchup_1h", schedule="@hourly", start_date=START, catchup=False,
     max_active_runs=1, default_args=DEFAULT_ARGS, tags=["sentiment"])
def sentiment_catchup_1h():
    @task
    def catch_up() -> int:
        from cryptosent.pipeline import enrich_pending

        settings, store = _ctx()
        return enrich_pending(settings, store, limit=1000, min_age_minutes=30)

    @task
    def upgrade_lexicon_scores(_: int) -> int:
        """Spend leftover Gemini quota on re-scoring lexicon-scored articles."""
        from cryptosent.pipeline import rescore

        settings, store = _ctx()
        if not settings.gemini_api_key or settings.sentiment_backend == "lexicon":
            return 0
        try:
            return rescore(settings, store, "lexicon", limit=settings.gemini_rescore_per_hour)
        except ValueError:  # Gemini unusable right now: build_agent fell back to lexicon
            return 0

    upgrade_lexicon_scores(catch_up())


@dag(dag_id="train_daily", schedule="0 2 * * *", start_date=START, catchup=False,
     max_active_runs=1, default_args={**DEFAULT_ARGS, "execution_timeout": timedelta(hours=3)},
     tags=["model"], params={"days": Param(180, type="integer", minimum=7)})
def train_daily():
    @task
    def backfill_recent_news() -> int:
        from cryptosent.pipeline import backfill_news

        settings, store = _ctx()
        return backfill_news(settings, store, days=2)

    @task
    def score(_: int) -> int:
        from cryptosent.pipeline import enrich_pending

        settings, store = _ctx()
        return enrich_pending(settings, store, limit=5000, min_age_minutes=30)

    @task
    def train(_: int, params=None) -> dict:
        from cryptosent.pipeline import json_safe, train

        settings, store = _ctx()
        report = train(settings, store, days=int(params["days"]))
        return json_safe({"best_model": report["best_model"], "test": report["test"]})

    train(score(backfill_recent_news()))


@dag(dag_id="backfill_history", schedule=None, start_date=START, catchup=False,
     default_args={**DEFAULT_ARGS, "execution_timeout": timedelta(hours=12)}, tags=["setup"],
     params={"days": Param(90, type="integer", minimum=1, maximum=1000),
             "score_limit": Param(50000, type="integer", minimum=0),
             # "lexicon" scores the backfill offline (no Gemini quota used);
             # sentiment_catchup_1h then upgrades it to Gemini bit by bit
             "sentiment_backend": Param("auto", enum=["auto", "gemini", "lexicon"])})
def backfill_history():
    @task
    def prices(params=None) -> int:
        from cryptosent.pipeline import fetch_prices

        settings, store = _ctx()
        return fetch_prices(settings, store, backfill_days=int(params["days"]))

    @task
    def news(params=None) -> int:
        from cryptosent.pipeline import backfill_news

        settings, store = _ctx()
        return backfill_news(settings, store, days=int(params["days"]))

    @task
    def score(_: int, params=None) -> int:
        from cryptosent.pipeline import enrich_pending

        settings, store = _ctx()
        backend = None if params["sentiment_backend"] == "auto" else params["sentiment_backend"]
        return enrich_pending(settings, store, limit=int(params["score_limit"]), backend=backend)

    prices()
    score(news())


cmc_news_10m()
btc_prices_5m()
predict_10m()
sentiment_catchup_1h()
train_daily()
backfill_history()
