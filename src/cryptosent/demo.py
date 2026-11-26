"""Synthetic market + news generator for tests, CI and an offline demo.

Prices follow a random walk with volatility clustering. News events arrive as a
Poisson process; each carries a latent sentiment that nudges the price drift
for the following ``impact_min`` minutes, starting ``delay_min`` minutes after
publication. The injected signal is small on purpose, so the demo shows what
an honest pipeline can and cannot recover. Nothing here is real data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .collectors.news_base import Article
from .collectors.prices import PRICE_COLUMNS

BULL_HEADLINES = [
    "Bitcoin surges as spot ETF inflows hit a record",
    "BTC rallies after institutional adoption news from a major bank",
    "Bitcoin climbs as on-chain data shows strong accumulation by whales",
    "Analysts turn bullish as Bitcoin breaks key resistance",
    "Bitcoin rebounds on optimism over rate cuts",
]
BEAR_HEADLINES = [
    "Bitcoin plunges as liquidations hit leveraged traders",
    "BTC drops after regulator announces crackdown on exchanges",
    "Bitcoin falls as ETF outflows raise fears of a deeper correction",
    "Exchange hack sparks panic and a Bitcoin selloff",
    "Bitcoin slumps on weak macro data and rising uncertainty",
]
NEUTRAL_HEADLINES = [
    "Bitcoin network hashrate update published by mining pools",
    "Exchange lists new Bitcoin trading pairs",
    "Report reviews Bitcoin market structure this week",
    "Developers discuss upcoming Bitcoin protocol changes",
]
HYPE_HEADLINES = [
    "Bitcoin to the moon! Don't miss the next 100x move!",
    "Death cross warns of massive crash, Bitcoin going to zero?!",
]
BODY_FILLER = (
    "Market participants watched volume and liquidity closely while traders "
    "assessed support and resistance levels. The report cites data from several "
    "exchanges and comments from analysts at {org}."
)
ORGS = ["BlackRock", "Coinbase", "Fidelity", "Binance", "the SEC", "Grayscale"]


@dataclass
class DemoConfig:
    days: int = 30
    start: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc)
    base_price: float = 60_000.0
    vol_per_min_pct: float = 0.03
    news_per_hour: float = 4.0
    impact_pct: float = 0.25  # total drift added by a +/-1 sentiment article
    delay_min: int = 5
    impact_min: int = 30
    seed: int = 7


def generate(cfg: DemoConfig | None = None) -> tuple[pd.DataFrame, list[Article], pd.DataFrame]:
    """Return (candles, articles, ground_truth) where ground_truth lists the
    latent sentiment used to move the price."""
    cfg = cfg or DemoConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.days * 24 * 60
    index = pd.date_range(cfg.start, periods=n, freq="1min", tz="UTC")

    # news events
    n_events = rng.poisson(cfg.news_per_hour * cfg.days * 24)
    event_min = np.sort(rng.integers(0, n, n_events))
    latent = np.clip(rng.normal(0, 0.5, n_events), -1, 1)

    drift = np.zeros(n)
    for m, s in zip(event_min, latent):
        a = m + cfg.delay_min
        b = min(n, a + cfg.impact_min)
        if a < n:
            drift[a:b] += s * cfg.impact_pct / cfg.impact_min

    # GARCH-ish volatility clustering
    vol = np.empty(n)
    v = cfg.vol_per_min_pct
    shocks = rng.standard_normal(n)
    for i in range(n):
        v = 0.98 * v + 0.02 * cfg.vol_per_min_pct * (1 + abs(shocks[i - 1]) if i else 1)
        vol[i] = v
    ret = drift + vol * shocks  # percent
    close = cfg.base_price * np.exp(np.cumsum(ret / 100))
    open_ = np.concatenate([[cfg.base_price], close[:-1]])
    spread = np.abs(rng.normal(0, vol / 100, n)) * close
    candles = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.gamma(2.0, 5.0, n) * (1 + 3 * np.abs(drift) / (cfg.impact_pct / cfg.impact_min + 1e-9) * 0.1),
            "quote_volume": 0.0,
            "trades": rng.poisson(300, n).astype(float),
        },
        index=index,
    )
    candles["quote_volume"] = candles["volume"] * candles["close"]
    candles.index.name = "open_time"

    articles = []
    truth = []
    for k, (m, s) in enumerate(zip(event_min, latent)):
        if rng.random() < 0.05:
            title = HYPE_HEADLINES[int(s < 0)]
        elif s > 0.2:
            title = BULL_HEADLINES[rng.integers(len(BULL_HEADLINES))]
        elif s < -0.2:
            title = BEAR_HEADLINES[rng.integers(len(BEAR_HEADLINES))]
        else:
            title = NEUTRAL_HEADLINES[rng.integers(len(NEUTRAL_HEADLINES))]
        published = index[m] + timedelta(seconds=int(rng.integers(0, 60)))
        art = Article(
            url=f"https://example.com/demo/{k}",
            title=f"{title} ({k})",
            published_at=published.to_pydatetime(),
            source="demo",
            body=BODY_FILLER.format(org=ORGS[rng.integers(len(ORGS))]),
            publisher="demo",
        )
        articles.append(art)
        truth.append({"article_id": art.article_id, "published_at": published, "latent": s})
    return candles[PRICE_COLUMNS], articles, pd.DataFrame(truth)
