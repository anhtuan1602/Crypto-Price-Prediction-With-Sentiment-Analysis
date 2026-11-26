"""Web data collectors: news crawlers and market price feeds."""
from __future__ import annotations

from .coinmarketcap import CoinMarketCapCrawler
from .cryptocompare import CryptoCompareCollector
from .gdelt import GdeltCollector
from .news_base import Article, PoliteSession, article_hash, clean_text, dedupe, fetch_full_text, to_utc
from .prices import BinancePriceCollector
from .rss import RSSCollector

__all__ = [
    "Article", "PoliteSession", "article_hash", "clean_text", "dedupe", "fetch_full_text", "to_utc",
    "CoinMarketCapCrawler", "CryptoCompareCollector", "GdeltCollector", "RSSCollector", "BinancePriceCollector",
    "build_news_collectors", "build_backfill_collectors",
]


def build_news_collectors(settings) -> list:
    session = PoliteSession(proxies=settings.http_proxies, timeout=settings.request_timeout)
    registry = {
        "coinmarketcap": lambda: CoinMarketCapCrawler(session=session, fetch_bodies=False),  # see crawl_news
        "rss": lambda: RSSCollector(session=session),
        "cryptocompare": lambda: CryptoCompareCollector(session=session, api_key=settings.cryptocompare_api_key),
        "gdelt": lambda: GdeltCollector(session=session),
    }
    names = [n for n in settings.news_sources if n in registry]
    if not settings.cryptocompare_api_key:
        names = [n for n in names if n != "cryptocompare"]  # no free tier since May 2026
    return [registry[name]() for name in names]


def build_backfill_collectors(settings) -> list:
    """Historical news sources: GDELT (free, about 3 months) and CryptoCompare
    when a key is configured."""
    session = PoliteSession(proxies=settings.http_proxies, timeout=settings.request_timeout)
    out = []
    for name in settings.news_backfill_sources:
        if name == "gdelt":
            out.append(GdeltCollector(session=session))
        elif name == "cryptocompare" and settings.cryptocompare_api_key:
            out.append(CryptoCompareCollector(session=session, api_key=settings.cryptocompare_api_key))
    return out
