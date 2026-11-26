"""CryptoCompare (now CoinDesk Data) news API: the free source here with deep
history, used to backfill months of articles so the model is not trained on a
few days of RSS only.

Two endpoints are tried, newest first:

* ``data-api.cryptocompare.com/news/v1/article/list`` (current API, pages
  backwards with ``to_ts``, upper-case fields such as ``PUBLISHED_ON``)
* ``min-api.cryptocompare.com/data/v2/news/`` (legacy API, pages backwards
  with ``lTs``, lower-case fields)

Both answer errors inside a 200 or 4xx JSON body (for example a missing API
key) instead of failing loudly, so every page is checked and the reason is
logged. A backfill that cannot get a single page raises
``CryptoCompareError`` so the Airflow task fails visibly instead of storing
nothing. Set CRYPTOCOMPARE_API_KEY (free) for history and higher limits.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from .news_base import Article, PoliteSession, clean_text, dedupe, to_utc

log = logging.getLogger(__name__)

DATA_API_URL = "https://data-api.cryptocompare.com/news/v1/article/list"
NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
PAGE_LIMIT = 100


class CryptoCompareError(RuntimeError):
    pass


def _get(item: dict, *keys: str) -> Any:
    for k in keys:
        v = item.get(k)
        if v not in (None, ""):
            return v
    return None


def _categories(item: dict) -> str:
    cats = item.get("categories")
    if cats:
        return cats
    data = item.get("CATEGORY_DATA") or []
    return "|".join(str(c.get("CATEGORY") or c.get("NAME")) for c in data if isinstance(c, dict))


def parse_items(items: list[dict]) -> list[Article]:
    """Parse items from either API (lower-case legacy or upper-case current)."""
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = clean_text(_get(item, "title", "TITLE"))
        url = _get(item, "url", "URL", "guid", "GUID")
        ts = _get(item, "published_on", "PUBLISHED_ON")
        if not (title and url and ts):
            continue
        source = item.get("source_info") or item.get("SOURCE_DATA") or {}
        out.append(
            Article(
                url=url,
                title=title,
                published_at=to_utc(int(ts)),
                source="cryptocompare",
                body=clean_text(_get(item, "body", "BODY") or ""),
                publisher=source.get("name") or source.get("NAME") or item.get("source"),
                extra={"categories": _categories(item)},
            )
        )
    return out


def api_error(payload: Any) -> str | None:
    """Return the API's error message, or None for a usable page."""
    if not isinstance(payload, dict):
        return "no JSON response (network error or HTTP error page)"
    err = payload.get("Err")
    if isinstance(err, dict) and err:
        return str(err.get("message") or err)
    if str(payload.get("Response", "")).lower() == "error":
        return str(payload.get("Message") or "unknown error")
    if not isinstance(payload.get("Data"), list):
        return f"unexpected response shape (keys: {sorted(payload)[:6]})"
    return None


class CryptoCompareCollector:
    name = "cryptocompare"

    def __init__(self, session: PoliteSession | None = None, api_key: str | None = None, categories: str = "BTC"):
        self.session = session or PoliteSession()
        self.api_key = api_key
        self.categories = categories
        self.endpoint: str | None = None  # the endpoint that answered last

    def _request(self, endpoint: str, before: int | None) -> tuple[list[dict], str | None]:
        if endpoint == DATA_API_URL:
            params: dict[str, Any] = {"lang": "EN", "limit": PAGE_LIMIT, "categories": self.categories}
            if before:
                params["to_ts"] = before
        else:
            params = {"lang": "EN", "categories": self.categories}
            if before:
                params["lTs"] = before
        headers = {"authorization": f"Apikey {self.api_key}"} if self.api_key else None
        payload = self.session.get_json(endpoint, params=params, headers=headers, return_errors=True)
        error = api_error(payload)
        return ([] if error else payload["Data"]), error

    def _page(self, before: int | None) -> list[dict]:
        """One page of items, trying the endpoint that worked last first."""
        order = [DATA_API_URL, NEWS_URL]
        if self.endpoint in order:
            order.remove(self.endpoint)
            order.insert(0, self.endpoint)
        errors = []
        for endpoint in order:
            items, error = self._request(endpoint, before)
            if error is None:
                if self.endpoint != endpoint:
                    log.info("CryptoCompare: using %s", endpoint)
                self.endpoint = endpoint
                return items
            errors.append(f"{endpoint}: {error}")
        hint = "" if self.api_key else " (no CRYPTOCOMPARE_API_KEY set; a free key is usually required)"
        raise CryptoCompareError("CryptoCompare news API failed" + hint + "; " + " | ".join(errors))

    def crawl(self) -> list[Article]:
        try:
            return dedupe(parse_items(self._page(None)))
        except CryptoCompareError as exc:
            log.warning("%s", exc)
            return []

    def backfill(self, start: datetime, end: datetime | None = None, pause: float = 0.5) -> list[Article]:
        """Walk backwards from ``end`` until ``start``. Raises CryptoCompareError
        if not even the first page could be fetched."""
        end = end or datetime.now(timezone.utc)
        cursor = int(end.timestamp())
        stop = int(start.timestamp())
        articles: list[Article] = []
        pages = 0
        while cursor > stop:
            try:
                items = self._page(cursor)
            except CryptoCompareError:
                if pages == 0:
                    raise
                log.exception("CryptoCompare backfill stopped early at %s", to_utc(cursor))
                break
            pages += 1
            if not items:
                break
            batch = parse_items(items)
            articles.extend(a for a in batch if a.published_at.timestamp() >= stop)
            stamps = [int(_get(i, "published_on", "PUBLISHED_ON") or cursor) for i in items if isinstance(i, dict)]
            oldest = min(stamps, default=cursor)
            if oldest >= cursor:
                break
            cursor = oldest - 1
            if pages % 10 == 0:
                log.info("CryptoCompare backfill reached %s (%d articles, %d pages)",
                         to_utc(cursor), len(articles), pages)
            time.sleep(pause)
        reached = to_utc(max(cursor, stop))
        if cursor > stop:
            log.warning("CryptoCompare backfill only reached %s, not %s: the API returned no older items "
                        "(free tiers may limit history)", reached, to_utc(stop))
        log.info("CryptoCompare backfill: %d articles in %d pages back to %s", len(articles), pages, reached)
        return dedupe(articles)
