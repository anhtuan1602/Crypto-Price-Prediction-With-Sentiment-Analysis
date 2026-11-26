"""The most important tests: the dataset must never let a feature see the future."""
import numpy as np
import pandas as pd
import pytest

from cryptosent.config import WindowConfig
from cryptosent.features import TARGET, TECHNICAL_FEATURES, LeakageError, audit_dataset, build_dataset
from cryptosent.features.technical import price_series, technical_features

START = pd.Timestamp("2025-12-13T00:00:00Z")


def candles(minutes=600, slope=1.0):
    idx = pd.date_range(START, periods=minutes, freq="1min", tz="UTC")
    close = 1000 + slope * np.arange(1, minutes + 1)  # close of candle i = 1000 + (i+1)
    df = pd.DataFrame({"open": close - slope, "high": close + 1, "low": close - 2, "close": close,
                       "volume": 1.0, "quote_volume": 1.0, "trades": 1.0}, index=idx)
    df.index.name = "open_time"
    return df


def articles(times, scores):
    return pd.DataFrame({
        "article_id": [str(i) for i in range(len(times))],
        "published_at": pd.to_datetime(times, utc=True),
        "sentiment_score": scores,
        "sentiment_confidence": [0.9] * len(times),
        "sentiment_label": ["bullish"] * len(times),
    })


W = WindowConfig(obs=30, lag=10, pred=10, step=10)


def test_target_is_exact_future_return():
    ds = build_dataset(candles(), articles([], []), W)
    t = ds.frame.index[0]
    prices = price_series(candles())["price"]
    p0 = prices[t + pd.Timedelta(minutes=10)]
    p1 = prices[t + pd.Timedelta(minutes=20)]
    assert ds.frame.loc[t, TARGET] == pytest.approx((p1 - p0) / p0 * 100)
    # price at T is the close of the candle that *ended* at T
    assert prices[START + pd.Timedelta(minutes=1)] == 1001.0


def test_article_at_decision_time_is_excluded():
    t = START + pd.Timedelta(minutes=120)
    arts = articles([t - pd.Timedelta(seconds=1), t, t + pd.Timedelta(minutes=5)], [0.5, 1.0, -1.0])
    ds = build_dataset(candles(), arts, W)
    row = ds.frame.loc[t]
    assert row["sent_n_articles"] == 1
    assert row["sent_mean"] == pytest.approx(0.5)
    assert row["last_article_time"] < t


def test_observation_window_boundaries():
    t = START + pd.Timedelta(minutes=120)
    arts = articles([t - pd.Timedelta(minutes=30), t - pd.Timedelta(minutes=31)], [0.2, 0.9])
    row = build_dataset(candles(), arts, W).frame.loc[t]
    assert row["sent_n_articles"] == 1 and row["sent_mean"] == pytest.approx(0.2)
    assert row["sent_n_articles_6h"] == 2


def test_technical_features_are_causal():
    """Changing prices after T must not change features at T."""
    base = candles()
    altered = base.copy()
    cut = START + pd.Timedelta(minutes=300)
    altered.loc[altered.index >= cut, "close"] *= 1.5
    f1 = technical_features(price_series(base))
    f2 = technical_features(price_series(altered))
    # row T uses candles that closed by T, i.e. open_time < T
    pd.testing.assert_frame_equal(f1.loc[:cut], f2.loc[:cut])
    assert list(f1.columns) == TECHNICAL_FEATURES and len(TECHNICAL_FEATURES) == 21


def test_unlabelled_rows_only_when_requested():
    labelled = build_dataset(candles(), articles([], []), W)
    live = build_dataset(candles(), articles([], []), W, include_unlabelled=True)
    assert labelled.frame[TARGET].notna().all()
    assert live.frame[TARGET].isna().any()
    assert live.frame.index.max() > labelled.frame.index.max()


def test_audit_passes_and_detects_tampering():
    t = START + pd.Timedelta(minutes=200)
    ds = build_dataset(candles(), articles([t - pd.Timedelta(minutes=3)], [0.4]), W)
    assert audit_dataset(ds)["samples_audited"] == len(ds.frame)
    ds.frame.loc[t, "last_article_time"] = t + pd.Timedelta(minutes=1)
    with pytest.raises(LeakageError):
        audit_dataset(ds)


def test_gaps_are_dropped():
    c = candles(900)
    c = c.drop(c.index[400:600])  # 200-minute exchange outage
    ds = build_dataset(c, articles([], []), W)
    assert ds.frame["filled_ratio"].max() <= 0.2
