"""CoinMarketCap news crawler (Latest News + Alexandria), report section 4.1.1a.

CoinMarketCap's news pages are rendered from a public JSON content endpoint,
so we read that first and fall back to parsing the HTML page (``__NEXT_DATA__``
or ``div.article-content``) with BeautifulSoup. Both parsers are tolerant to
missing fields because the site changes its payloads without notice.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from bs4 import BeautifulSoup

from .news_base import Article, PoliteSession, clean_text, dedupe, to_utc

log = logging.getLogger(__name__)

CONTENT_API = "https://api.coinmarketcap.com/content/v3/news"
NEWS_PAGE = "https://coinmarketcap.com/currencies/bitcoin/#News"
HEADLINES_PAGE = "https://coinmarketcap.com/headlines/news/"
ALEXANDRIA_BASE = "https://coinmarketcap.com/academy/article/"
BTC_COIN_ID = 1


def _first(d: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def parse_content_item(item: dict[str, Any]) -> Article | None:
    """Turn one item of the content API (or __NEXT_DATA__) into an Article."""
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else item
    title = clean_text(_first(meta, "title", "headline"))
    published = _first(meta, "releasedAt", "createdAt", "publishedAt", "published_at")
    if not title or published is None:
        return None
    kind = (_first(meta, "type", "sourceType") or "news").lower()
    slug = _first(item, "slug") or _first(meta, "slug")
    url = _first(meta, "sourceUrl", "url", "link")
    if (kind == "alexandria" or not url) and slug:
        url = ALEXANDRIA_BASE + str(slug)
    if not url:
        return None
    body = clean_text(_first(meta, "content", "subtitle", "description", "summary") or "")
    return Article(
        url=str(url),
        title=title,
        published_at=to_utc(published),
        source="coinmarketcap",
        body=body,
        publisher=_first(meta, "sourceName", "source") or ("CoinMarketCap" if kind == "alexandria" else None),
        extra={"cmc_type": kind, "slug": slug},
    )


def parse_next_data(html_text: str) -> list[Article]:
    """Extract article lists embedded in a Next.js ``__NEXT_DATA__`` script."""
    soup = BeautifulSoup(html_text, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None or not script.string:
        return []
    try:
        payload = json.loads(script.string)
    except json.JSONDecodeError:
        return []
    found: list[Article] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if ("meta" in node and isinstance(node["meta"], dict) and "title" in node["meta"]) or (
                "title" in node and ("releasedAt" in node or "createdAt" in node)
            ):
                art = parse_content_item(node)
                if art:
                    found.append(art)
                    return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return dedupe(found)


def parse_article_body(html_text: str) -> str:
    """Isolate the article body (report: ``<div class="article-content">``)."""
    soup = BeautifulSoup(html_text, "html.parser")
    for selector in ("div.article-content", "article", "div.post-content", "main"):
        node = soup.select_one(selector)
        if node is not None:
            text = clean_text(str(node))
            if len(text) > 200:
                return text
    return ""


class CoinMarketCapCrawler:
    name = "coinmarketcap"

    def __init__(self, session: PoliteSession | None = None, page_size: int = 50, fetch_bodies: bool = True):
        self.session = session or PoliteSession()
        self.page_size = page_size
        self.fetch_bodies = fetch_bodies

    def fetch_api(self, pages: int = 1, coin_id: int = BTC_COIN_ID) -> list[Article]:
        articles: list[Article] = []
        for page in range(1, pages + 1):
            payload = self.session.get_json(
                CONTENT_API,
                params={"coins": coin_id, "page": page, "size": self.page_size},
                headers={"Accept": "application/json", "Origin": "https://coinmarketcap.com"},
            )
            items: Iterable[Any] = []
            if isinstance(payload, dict):
                items = payload.get("data") or []
            if not items:
                break
            for item in items:
                if isinstance(item, dict):
                    art = parse_content_item(item)
                    if art:
                        articles.append(art)
        return articles

    def fetch_html(self) -> list[Article]:
        articles: list[Article] = []
        for url in (HEADLINES_PAGE, NEWS_PAGE):
            resp = self.session.get(url)
            if resp is not None:
                articles.extend(parse_next_data(resp.text))
        return articles

    def enrich_bodies(self, articles: list[Article], min_len: int = 300) -> list[Article]:
        """Download the full article where we only have a teaser."""
        for art in articles:
            if len(art.body) >= min_len:
                continue
            resp = self.session.get(art.url)
            if resp is None:
                continue
            body = parse_article_body(resp.text)
            if body:
                art.body = body
        return articles

    def crawl(self, pages: int = 1) -> list[Article]:
        articles = self.fetch_api(pages=pages)
        if not articles:
            log.info("CMC content API returned nothing, falling back to HTML")
            articles = self.fetch_html()
        articles = dedupe(articles)
        if self.fetch_bodies:
            articles = self.enrich_bodies(articles)
        log.info("CoinMarketCap: %d articles", len(articles))
        return articles
