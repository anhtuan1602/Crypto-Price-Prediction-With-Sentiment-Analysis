"""Binance 1-minute OHLCV collector.

The report used CoinGecko, whose free API cannot deliver the 1-minute candles
the technical baseline needs. Binance's public market-data mirror
(data-api.binance.vision) serves real 1-minute klines with years of history and
no API key.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .news_base import PoliteSession

log = logging.getLogger(__name__)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]
PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]


def klines_to_frame(rows: list[list]) -> pd.DataFrame:
    """Convert raw kline rows into a frame indexed by candle *open* time (UTC).

    A candle with open_time T covers [T, T+1m) and its close is the price at T+1m.
    """
    if not rows:
        return pd.DataFrame(columns=PRICE_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col])
    df = df.set_index("open_time")[PRICE_COLUMNS]
    return df[~df.index.duplicated(keep="last")].sort_index()


class BinancePriceCollector:
    name = "binance"

    def __init__(self, base_url: str = "https://data-api.binance.vision", symbol: str = "BTCUSDT",
                 session: PoliteSession | None = None):
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol
        self.session = session or PoliteSession()

    def _fetch(self, start_ms: int, end_ms: int | None = None, limit: int = 1000) -> list[list]:
        params = {"symbol": self.symbol, "interval": "1m", "startTime": start_ms, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        data = self.session.get_json(f"{self.base_url}/api/v3/klines", params=params)
        return data if isinstance(data, list) else []

    def fetch_range(self, start: datetime, end: datetime | None = None, pause: float = 0.2) -> pd.DataFrame:
        """Fetch every closed 1m candle in [start, end)."""
        end = end or datetime.now(timezone.utc)
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[list] = []
        while cursor < end_ms:
            batch = self._fetch(cursor, end_ms - 1)
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            if last_open + 60_000 <= cursor:
                break
            cursor = last_open + 60_000
            if len(batch) < 1000:
                break
            time.sleep(pause)
        df = klines_to_frame(rows)
        # drop the still-open candle so we never store an unfinished bar
        now_floor = pd.Timestamp(datetime.now(timezone.utc)).floor("min")
        df = df[df.index < now_floor]
        log.info("Binance %s: %d candles %s -> %s", self.symbol, len(df), start, end)
        return df

    def fetch_recent(self, minutes: int = 60) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        return self.fetch_range(now - timedelta(minutes=minutes), now)
