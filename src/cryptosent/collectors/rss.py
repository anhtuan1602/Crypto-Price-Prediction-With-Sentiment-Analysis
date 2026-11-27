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
    # more keyless crypto news feeds (2026-11); dead or blocked feeds are skipped
    "cryptobriefing": "https://cryptobriefing.com/feed/",
    "ambcrypto": "https://ambcrypto.com/feed/",
    "coingape": "https://coingape.com/feed/",
    "crypto.news": "https://crypto.news/feed/",
    "thecryptobasic": "https://thecryptobasic.com/feed/",
    "zycrypto": "https://zycrypto.com/feed/",
    "coinpedia": "https://coinpedia.org/feed/",
    "dailyhodl": "https://dailyhodl.com/feed/",
    "coinjournal": "https://coinjournal.net/feed/",
    "coinspeaker": "https://www.coinspeaker.com/feed/",
    "cryptodaily": "https://cryptodaily.co.uk/feed",
    "coinedition": "https://coinedition.com/feed/",
    "finbold": "https://finbold.com/feed/",
    "coincentral": "https://coincentral.com/feed/",
    "blockonomi": "https://blockonomi.com/feed/",
    "bitcoinworld": "https://bitcoinworld.co.in/feed/",
    "crypto-economy": "https://crypto-economy.com/feed/",
    "investing.com": "https://www.investing.com/rss/news_301.rss",
}

# market-moving crypto news; every feed above is crypto-focused, this only drops off-topic posts
BTC_PATTERN = re.compile(
    r"\b(bitcoin|btc|crypto\w*|etf\w*|satoshi|halving|spot market|sec|blockchain|stablecoin\w*|"
    r"altcoin\w*|ethereum|eth|binance|coinbase|microstrategy|saylor|miners?|mining|fed|digital assets?)\b",
    re.I,
)


def feeds_from_settings(extra: list[str] | None = None, disabled: list[str] | None = None) -> dict[str, str]:
    """DEFAULT_FEEDS plus RSS_FEEDS_EXTRA ("name=url" items) minus RSS_FEEDS_DISABLED."""
    feeds = dict(DEFAULT_FEEDS)
    for item in extra or []:
        name, sep, url = item.partition("=")
        if sep and url.strip().startswith("http"):
            feeds[name.strip()] = url.strip()
        else:
            log.warning("Ignoring RSS_FEEDS_EXTRA item %r (expected name=https://...)", item)
    for name in disabled or []:
        feeds.pop(name.strip(), None)
    return feeds


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

    def check_feed(self, publisher: str, url: str) -> tuple[list[Article], dict]:
        """Fetch one feed. Returns (articles, report) where report has the
        HTTP status, the number of feed entries and how many were kept."""
        report: dict = {"url": url, "status": None, "entries": 0, "kept": 0, "error": None}
        resp = self.session.get(
            url, headers={"Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8"},
            return_errors=True,
        )
        if resp is None:
            report["error"] = "no response (network error, timeout or 5xx)"
            return [], report
        report["status"] = resp.status_code
        if resp.status_code >= 400:
            report["error"] = f"HTTP {resp.status_code}"
            return [], report
        try:
            report["entries"] = len(feedparser.parse(resp.content).entries)
            batch = parse_feed(resp.content, publisher)
        except Exception as exc:  # noqa: BLE001 - one broken feed must not stop the rest
            report["error"] = f"parse error: {exc}"
            return [], report
        if report["entries"] == 0:
            # usually a bot-protection page served with HTTP 200
            report["error"] = "not a feed (0 entries; blocked or moved?)"
        report["kept"] = len(batch)
        return batch, report

    def crawl(self) -> list[Article]:
        articles: list[Article] = []
        failed = []
        for publisher, url in self.feeds.items():
            batch, report = self.check_feed(publisher, url)
            if report["error"]:
                failed.append(f"{publisher} ({report['error']})")
            self.counts[publisher] = len(batch)
            articles.extend(batch)
        articles = dedupe(articles)
        log.info("RSS: %d articles from %d/%d feeds %s", len(articles),
                 len(self.feeds) - len(failed), len(self.feeds), self.counts)
        if failed:
            log.warning("RSS feeds with no articles this run: %s", "; ".join(failed))
        return articles
