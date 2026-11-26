"""Web data collectors: news crawlers and market price feeds."""
from __future__ import annotations

from .coinmarketcap import CoinMarketCapCrawler
from .cryptocompare import CryptoCompareCollector
from .news_base import Article, PoliteSession, article_hash, clean_text, dedupe, to_utc
from .prices import BinancePriceCollector
from .rss import RSSCollector

__all__ = [
    "Article", "PoliteSession", "article_hash", "clean_text", "dedupe", "to_utc",
    "CoinMarketCapCrawler", "CryptoCompareCollector", "RSSCollector", "BinancePriceCollector",
    "build_news_collectors",
]


def build_news_collectors(settings) -> list:
    session = PoliteSession(proxies=settings.http_proxies, timeout=settings.request_timeout)
    registry = {
        "coinmarketcap": lambda: CoinMarketCapCrawler(session=session),
        "rss": lambda: RSSCollector(session=session),
        "cryptocompare": lambda: CryptoCompareCollector(session=session, api_key=settings.cryptocompare_api_key),
    }
    return [registry[name]() for name in settings.news_sources if name in registry]
