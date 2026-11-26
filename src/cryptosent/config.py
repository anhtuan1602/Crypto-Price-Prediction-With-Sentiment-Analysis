"""Central configuration, read from environment variables (see .env.example).

Every setting has a sensible default so the package works offline for tests
and the synthetic demo; Docker Compose overrides them for the full stack.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = _env(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class WindowConfig:
    """Temporal framework from the report (Chapter 3.1), all in minutes.

    For a decision time ``t``:
      * observation window  [t - obs, t)        -> features (news + prices)
      * lag period          [t, t + lag)        -> never used
      * prediction window   [t + lag, t + lag + pred] -> target
    Samples are generated every ``step`` minutes.
    """

    obs: int = 30
    lag: int = 10
    pred: int = 10
    step: int = 10

    @property
    def horizon(self) -> int:
        """Minutes between decision time and the end of the target window."""
        return self.lag + self.pred

    @classmethod
    def from_env(cls) -> "WindowConfig":
        return cls(
            obs=_env_int("WINDOW_OBS_MIN", 30),
            lag=_env_int("WINDOW_LAG_MIN", 10),
            pred=_env_int("WINDOW_PRED_MIN", 10),
            step=_env_int("WINDOW_STEP_MIN", 10),
        )


@dataclass
class Settings:
    # Storage
    mongo_uri: str | None = field(default_factory=lambda: _env("MONGO_URI"))
    mongo_db: str = field(default_factory=lambda: _env("MONGO_DB", "cryptosent"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))

    # Kafka
    kafka_bootstrap: str | None = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP_SERVERS"))
    topic_raw: str = field(default_factory=lambda: _env("KAFKA_TOPIC_RAW", "news.raw"))
    topic_scored: str = field(default_factory=lambda: _env("KAFKA_TOPIC_SCORED", "news.scored"))

    # Market data
    symbol: str = field(default_factory=lambda: _env("SYMBOL", "BTCUSDT"))
    binance_base_url: str = field(
        default_factory=lambda: _env("BINANCE_BASE_URL", "https://data-api.binance.vision")
    )

    # News sources
    news_sources: list[str] = field(
        default_factory=lambda: _env_list("NEWS_SOURCES", ["coinmarketcap", "rss", "cryptocompare"])
    )
    # historical backfill; cryptocompare is used only when CRYPTOCOMPARE_API_KEY is set
    # (CoinDesk Data ended its free tier in May 2026), gdelt needs no key
    news_backfill_sources: list[str] = field(
        default_factory=lambda: _env_list("NEWS_BACKFILL_SOURCES", ["gdelt", "cryptocompare"])
    )
    cryptocompare_api_key: str | None = field(default_factory=lambda: _env("CRYPTOCOMPARE_API_KEY"))
    http_proxies: list[str] = field(default_factory=lambda: _env_list("HTTP_PROXY_POOL", []))
    request_timeout: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT", 20.0))

    # Sentiment
    sentiment_backend: str = field(default_factory=lambda: _env("SENTIMENT_BACKEND", "auto"))
    gemini_api_key: str | None = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    # Confirmed working on the free tier (2026-11). If a model answers 404 the client
    # moves on to gemini-flash-latest and then other listed Flash models (sentiment/gemini.py).
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    # free-tier friendly defaults; raise them on a paid key
    gemini_rpm: int = field(default_factory=lambda: _env_int("GEMINI_RPM", 10))
    gemini_batch_size: int = field(default_factory=lambda: _env_int("GEMINI_BATCH_SIZE", 10))
    # hourly budget for upgrading lexicon-scored articles to Gemini (0 = off)
    gemini_rescore_per_hour: int = field(default_factory=lambda: _env_int("GEMINI_RESCORE_PER_HOUR", 50))

    # Modelling
    windows: WindowConfig = field(default_factory=WindowConfig.from_env)
    target_flat_threshold: float = field(
        default_factory=lambda: _env_float("TARGET_FLAT_THRESHOLD_PCT", 0.02)
    )
    fee_pct: float = field(default_factory=lambda: _env_float("BACKTEST_FEE_PCT", 0.1))
    random_state: int = field(default_factory=lambda: _env_int("RANDOM_STATE", 42))
    mlflow_uri: str | None = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI"))
    mlflow_experiment: str = field(
        default_factory=lambda: _env("MLFLOW_EXPERIMENT", "crypto-sentiment")
    )
    artifacts_dir: Path = field(default_factory=lambda: Path(_env("ARTIFACTS_DIR", "artifacts")))


def get_settings() -> Settings:
    return Settings()
