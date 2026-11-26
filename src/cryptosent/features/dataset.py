"""Leakage-safe supervised dataset (report sections 3.1, 3.5, 4.1.1d).

For each decision time t on a ``step``-minute grid:

    features  <- news published in [t - obs, t) and candles closed by t
    (lag)        [t, t + lag)   nothing is used
    target    <- Y_t = (P(t + lag + pred) - P(t + lag)) / P(t + lag) * 100

P(T) is the close of the 1-minute candle ending at T. The builder records the
timestamps it relied on (``feature_cutoff``, ``last_article_time``,
``target_start``) so ``audit_dataset`` can prove no feature saw the future.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import WindowConfig
from .sentiment_agg import LLM_FEATURES, NLP_FEATURES, aggregate_windows
from .technical import TECHNICAL_FEATURES, price_series, technical_features

TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
FEATURE_GROUPS: dict[str, list[str]] = {
    "technical": TECHNICAL_FEATURES,
    "sentiment": LLM_FEATURES,
    "nlp": NLP_FEATURES,
    "time": TIME_FEATURES,
}
META_COLUMNS = ["feature_cutoff", "last_article_time", "target_start", "target_end", "price_at_t", "filled_ratio"]
TARGET = "target_pct"
TARGET_DIR = "target_dir"
WARMUP_MIN = 60


@dataclass
class Dataset:
    frame: pd.DataFrame
    windows: WindowConfig

    def features(self, groups: list[str] | None = None) -> list[str]:
        groups = groups or list(FEATURE_GROUPS)
        cols: list[str] = []
        for g in groups:
            cols.extend(FEATURE_GROUPS[g])
        return cols

    @property
    def y(self) -> pd.Series:
        return self.frame[TARGET]


def decision_grid(start: pd.Timestamp, end: pd.Timestamp, step_min: int) -> pd.DatetimeIndex:
    first = start.ceil(f"{step_min}min")
    if end < first:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(first, end, freq=f"{step_min}min", tz="UTC")


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + index.minute / 60
    dow = index.dayofweek
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        },
        index=index,
    )


def build_dataset(
    candles: pd.DataFrame,
    articles: pd.DataFrame,
    windows: WindowConfig,
    flat_threshold_pct: float = 0.02,
    include_unlabelled: bool = False,
    max_filled_ratio: float = 0.2,
) -> Dataset:
    """Build the modelling table.

    ``include_unlabelled=True`` keeps the most recent rows whose target is not
    known yet (used for live prediction)."""
    prices = price_series(candles)
    if prices.empty:
        return Dataset(pd.DataFrame(columns=META_COLUMNS + [TARGET]), windows)
    tech = technical_features(prices)

    start = prices.index.min() + pd.Timedelta(minutes=max(WARMUP_MIN, windows.obs))
    end = prices.index.max()
    if not include_unlabelled:
        end = end - pd.Timedelta(minutes=windows.horizon)
    grid = decision_grid(start, end, windows.step)
    if len(grid) == 0:
        return Dataset(pd.DataFrame(columns=META_COLUMNS + [TARGET]), windows)

    df = tech.reindex(grid)
    df["feature_cutoff"] = grid  # last price used is the candle closing at t
    df["price_at_t"] = prices["price"].reindex(grid).to_numpy()
    filled = prices["filled"].astype(float).rolling(windows.obs, min_periods=1).mean()
    df["filled_ratio"] = filled.reindex(grid).to_numpy()

    t_start = grid + pd.Timedelta(minutes=windows.lag)
    t_end = t_start + pd.Timedelta(minutes=windows.pred)
    p_start = prices["price"].reindex(t_start).to_numpy()
    p_end = prices["price"].reindex(t_end).to_numpy()
    df["target_start"] = t_start
    df["target_end"] = t_end
    df[TARGET] = (p_end - p_start) / p_start * 100
    df[TARGET_DIR] = np.select(
        [df[TARGET] > flat_threshold_pct, df[TARGET] < -flat_threshold_pct], [1, -1], default=0
    )
    df.loc[df[TARGET].isna(), TARGET_DIR] = np.nan

    sent = aggregate_windows(articles, grid, windows.obs)
    df = df.join(sent)
    df = df.join(time_features(grid))

    df = df[df["filled_ratio"] <= max_filled_ratio]
    df = df.dropna(subset=TECHNICAL_FEATURES)
    if not include_unlabelled:
        df = df.dropna(subset=[TARGET])
    df.index.name = "decision_time"
    return Dataset(df, windows)


class LeakageError(AssertionError):
    pass


def audit_dataset(ds: Dataset) -> dict[str, int]:
    """Report test 4, strengthened: checks the full lag structure, not only
    "article before window end". Raises ``LeakageError`` on any violation."""
    df = ds.frame
    w = ds.windows
    t = pd.Series(df.index, index=df.index)
    checks = {
        "article_after_decision_time": int((df["last_article_time"].dropna() >= t[df["last_article_time"].notna()]).sum()),
        "price_feature_after_decision_time": int((df["feature_cutoff"] > t).sum()),
        "target_starts_before_lag": int((df["target_start"] < t + pd.Timedelta(minutes=w.lag)).sum()),
        "target_window_wrong_length": int(
            ((df["target_end"] - df["target_start"]) != pd.Timedelta(minutes=w.pred)).sum()
        ),
    }
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise LeakageError(f"Look-ahead violations found: {bad}")
    return {"samples_audited": len(df), **checks}
