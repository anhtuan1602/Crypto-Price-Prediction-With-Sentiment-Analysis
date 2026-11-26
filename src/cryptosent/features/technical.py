"""21 causal technical features from 1-minute candles (report table 7.1).

Differences from the report, on purpose:
  * moving averages and Bollinger bands are expressed relative to the current
    price (``price / MA - 1``) instead of raw dollar levels, which are
    non-stationary and let tree models memorise the price regime;
  * every feature at time T uses only candles that closed at or before T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TECHNICAL_FEATURES = [
    "return_1", "return_2", "return_3", "return_5", "return_10",
    "volatility_5", "volatility_10", "volatility_20",
    "ma_5_ratio", "ma_10_ratio", "ma_20_ratio",
    "rsi_14", "macd", "macd_signal", "macd_histogram", "bb_upper_dist", "bb_lower_dist", "bb_width",
    "log_volume", "log_volume_ma5", "volume_ratio",
]


def price_series(candles: pd.DataFrame) -> pd.DataFrame:
    """Re-index 1m candles by *close* time on a gap-free minute grid.

    Row T then describes the market as known at time T: ``price`` is the close
    of the candle [T-1m, T). Gaps are forward-filled for price and zero-filled
    for volume; ``filled`` marks synthetic rows.
    """
    if candles.empty:
        return pd.DataFrame(columns=["price", "volume", "filled"])
    df = candles[["close", "volume"]].copy()
    df.index = df.index + pd.Timedelta(minutes=1)
    full = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    df = df.reindex(full)
    filled = df["close"].isna()
    out = pd.DataFrame(
        {
            "price": df["close"].ffill(),
            "volume": df["volume"].fillna(0.0),
            "filled": filled,
        },
        index=full,
    )
    out.index.name = "time"
    return out


def _rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss != 0, 100.0)


def technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute features for every minute of ``price_series`` output."""
    p = prices["price"]
    vol = prices["volume"]
    ret1 = p.pct_change() * 100
    f = pd.DataFrame(index=prices.index)
    for k in (1, 2, 3, 5, 10):
        f[f"return_{k}"] = (p / p.shift(k) - 1) * 100
    for k in (5, 10, 20):
        f[f"volatility_{k}"] = ret1.rolling(k, min_periods=k).std()
    for k in (5, 10, 20):
        f[f"ma_{k}_ratio"] = (p / p.rolling(k, min_periods=k).mean() - 1) * 100
    f["rsi_14"] = _rsi(p, 14)
    ema12 = p.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = p.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = (ema12 - ema26) / p * 100
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd"] = macd
    f["macd_signal"] = signal
    f["macd_histogram"] = macd - signal
    ma20 = p.rolling(20, min_periods=20).mean()
    sd20 = p.rolling(20, min_periods=20).std()
    upper, lower = ma20 + 2 * sd20, ma20 - 2 * sd20
    f["bb_upper_dist"] = (upper / p - 1) * 100
    f["bb_lower_dist"] = (lower / p - 1) * 100
    f["bb_width"] = (upper - lower) / ma20 * 100
    vol_ma5 = vol.rolling(5, min_periods=5).mean()
    f["log_volume"] = np.log1p(vol)
    f["log_volume_ma5"] = np.log1p(vol_ma5)
    f["volume_ratio"] = vol / vol_ma5.replace(0, np.nan)
    return f[TECHNICAL_FEATURES]
