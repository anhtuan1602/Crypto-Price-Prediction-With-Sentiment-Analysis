"""CryptoCompare news API: the only free source here with deep history, used
to backfill months of articles so the model is not trained on 20 days only.

Paging works backwards with the ``lTs`` (latest timestamp) parameter.
A free API key raises the rate limit; set CRYPTOCOMPARE_API_KEY.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .news_base import Article, PoliteSession, clean_text, dedupe, to_utc

log = logging.getLogger(__name__)

NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"


def parse_items(items: list[dict]) -> list[Article]:
    out = []
    for item in items:
        title = clean_text(item.get("title"))
        url = item.get("url") or item.get("guid")
        ts = item.get("published_on")
        if not (title and url and ts):
            continue
        out.append(
            Article(
                url=url,
                title=title,
                published_at=to_utc(int(ts)),
                source="cryptocompare",
                body=clean_text(item.get("body", "")),
                publisher=(item.get("source_info") or {}).get("name") or item.get("source"),
                extra={"categories": item.get("categories", "")},
            )
        )
    return out


class CryptoCompareCollector:
    name = "cryptocompare"

    def __init__(self, session: PoliteSession | None = None, api_key: str | None = None, categories: str = "BTC"):
        self.session = session or PoliteSession()
        self.api_key = api_key
        self.categories = categories

    def _page(self, before: int | None) -> list[dict]:
        params: dict[str, str | int] = {"lang": "EN", "categories": self.categories}
        if before:
            params["lTs"] = before
        headers = {"authorization": f"Apikey {self.api_key}"} if self.api_key else None
        payload = self.session.get_json(NEWS_URL, params=params, headers=headers)
        if not isinstance(payload, dict):
            return []
        data = payload.get("Data")
        return data if isinstance(data, list) else []

    def crawl(self) -> list[Article]:
        return dedupe(parse_items(self._page(None)))

    def backfill(self, start: datetime, end: datetime | None = None, pause: float = 0.5) -> list[Article]:
        """Walk backwards from ``end`` until ``start``."""
        end = end or datetime.now(timezone.utc)
        cursor = int(end.timestamp())
        stop = int(start.timestamp())
        articles: list[Article] = []
        while cursor > stop:
            items = self._page(cursor)
            if not items:
                break
            batch = parse_items(items)
            articles.extend(a for a in batch if a.published_at.timestamp() >= stop)
            oldest = min(int(i.get("published_on", cursor)) for i in items)
            if oldest >= cursor:
                break
            cursor = oldest - 1
            log.info("CryptoCompare backfill reached %s (%d articles)", to_utc(cursor), len(articles))
            time.sleep(pause)
        return dedupe(articles)
