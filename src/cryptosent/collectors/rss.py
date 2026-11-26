"""RSS collectors for major crypto outlets (live news, a second source next
to CoinMarketCap so a single site outage does not empty a window)."""
from __future__ import annotations

import logging
import re

import feedparser

from .news_base import Article, PoliteSession, clean_text, dedupe, to_utc

log = logging.getLogger(__name__)

# Public feeds, no keys. Feeds that only carry a teaser get their full text
# fetched once per new article (pipeline.crawl_news -> fetch_full_text).
DEFAULT_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss/tag/bitcoin",
    "decrypt": "https://decrypt.co/feed",
    "bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
    "theblock": "https://www.theblock.co/rss.xml",
    # WordPress sites whose feeds usually include the full article (content:encoded)
    "bitcoinist": "https://bitcoinist.com/feed/",
    "newsbtc": "https://www.newsbtc.com/feed/",
    "cryptoslate": "https://cryptoslate.com/feed/",
    "cryptopotato": "https://cryptopotato.com/feed/",
    "bitcoin.com": "https://news.bitcoin.com/feed/",
    "u.today": "https://u.today/rss",
    "cryptonews": "https://cryptonews.com/news/feed/",
    "beincrypto": "https://beincrypto.com/feed/",
}

BTC_PATTERN = re.compile(r"\b(bitcoin|btc|crypto|etf|satoshi|halving|spot market|sec)\b", re.I)


def parse_feed(content: bytes | str, publisher: str, btc_only: bool = True) -> list[Article]:
    feed = feedparser.parse(content)
    out = []
    for entry in feed.entries:
        title = clean_text(entry.get("title"))
        link = entry.get("link")
        published = entry.get("published") or entry.get("updated")
        if not (title and link and published):
            continue
        body_html = ""
        if entry.get("content"):
            body_html = entry["content"][0].get("value", "")
        body = clean_text(body_html or entry.get("summary", ""))
        if btc_only and not BTC_PATTERN.search(f"{title} {body}"):
            continue
        try:
            ts = to_utc(published)
        except (TypeError, ValueError):
            continue
        out.append(
            Article(url=link, title=title, published_at=ts, source="rss", body=body, publisher=publisher)
        )
    return out


class RSSCollector:
    name = "rss"

    def __init__(self, session: PoliteSession | None = None, feeds: dict[str, str] | None = None):
        self.session = session or PoliteSession()
        self.feeds = feeds or DEFAULT_FEEDS
        self.counts: dict[str, int] = {}

    def crawl(self) -> list[Article]:
        articles: list[Article] = []
        for publisher, url in self.feeds.items():
            resp = self.session.get(url, headers={"Accept": "application/rss+xml, application/xml"})
            if resp is None:
                continue
            try:
                batch = parse_feed(resp.content, publisher)
            except Exception:  # noqa: BLE001 - one broken feed must not stop the rest
                log.exception("Failed to parse feed %s", publisher)
                continue
            self.counts[publisher] = len(batch)
            articles.extend(batch)
        articles = dedupe(articles)
        log.info("RSS: %d articles %s", len(articles), self.counts)
        return articles
