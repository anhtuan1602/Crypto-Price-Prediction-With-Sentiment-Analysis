"""GDELT DOC 2.0 API: free historical news search, no API key.

This is the default source for the news backfill since CoinDesk Data
(formerly CryptoCompare) ended its free API tier in May 2026.

Limits, stated honestly:

* The DOC API searches a rolling window of about the last **3 months**.
  Older dates return nothing (GDELT's raw GKG files go further back but are
  multi-GB downloads, not supported here).
* It returns **headlines and URLs only**, no article body, so sentiment is
  scored from the title.
* ``seendate`` is when GDELT's crawler saw the article, usually within
  15-60 minutes of publication. That only delays news, so it cannot leak
  future information into a sample.
* At most 250 articles per request and roughly one request every 5 seconds.
  Busy windows are split in half until they fit, so a 90-day backfill takes
  about an hour.
* GDELT matches the full text, so titles are filtered for crypto terms.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from .news_base import Article, PoliteSession, clean_text, dedupe

log = logging.getLogger(__name__)

DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
DEFAULT_QUERY = "(bitcoin OR cryptocurrency OR crypto) sourcelang:english"
RELEVANT_TITLE = re.compile(
    r"\b(bitcoin|btc|crypto\w*|blockchain|satoshi|stablecoin\w*|ethereum|binance|coinbase|"
    r"microstrategy|saylor|etf|halving|miners?|mining|defi|token\w*|altcoin\w*|digital (asset|currency))\b",
    re.I,
)
_FMT = "%Y%m%d%H%M%S"


class GdeltError(RuntimeError):
    pass


def parse_seendate(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def parse_articles(items: list[dict], relevant_only: bool = True) -> list[Article]:
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = clean_text(item.get("title"))
        url = item.get("url")
        seen = parse_seendate(item.get("seendate", ""))
        if not (title and url and seen):
            continue
        if relevant_only and not RELEVANT_TITLE.search(title):
            continue
        out.append(Article(
            url=url,
            title=title,
            published_at=seen,
            source="gdelt",
            body="",
            publisher=item.get("domain"),
            extra={"language": item.get("language"), "country": item.get("sourcecountry")},
        ))
    return out


class GdeltCollector:
    name = "gdelt"

    def __init__(self, session: PoliteSession | None = None, query: str = DEFAULT_QUERY,
                 pause: float = 5.5, min_window: timedelta = timedelta(minutes=15)):
        self.session = session or PoliteSession()
        self.query = query
        self.pause = pause
        self.min_window = min_window
        self.requests = 0
        self._last_call = 0.0

    def _fetch(self, start: datetime, end: datetime) -> list[dict]:
        params: dict[str, Any] = {
            "query": self.query, "mode": "ArtList", "format": "json", "maxrecords": MAX_RECORDS,
            "sort": "DateDesc", "startdatetime": start.strftime(_FMT), "enddatetime": end.strftime(_FMT),
        }
        for attempt in range(3):
            wait = self.pause - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            self.requests += 1
            resp = self.session.get(DOC_URL, params=params, return_errors=True)
            if resp is None:
                raise GdeltError("GDELT did not answer (network error or HTTP 5xx after retries)")
            text = resp.text.strip()
            if not text or text == "{}":
                return []  # GDELT answers {} or nothing when no article matches
            try:
                # strict=False: GDELT titles sometimes carry raw control characters
                payload = json.loads(text, strict=False)
            except ValueError:
                # errors come back as plain text, e.g. the rate-limit notice
                msg = " ".join(text[:200].split())
                if "limit requests" in msg.lower() and attempt < 2:
                    log.warning("GDELT rate limit, waiting: %s", msg)
                    time.sleep(self.pause * 2)
                    continue
                raise GdeltError(f"GDELT error: {msg}") from None
            articles = payload.get("articles") if isinstance(payload, dict) else None
            return articles if isinstance(articles, list) else []
        raise GdeltError("GDELT kept rate-limiting the requests")

    def _window(self, start: datetime, end: datetime) -> list[dict]:
        """Fetch [start, end); split the window while GDELT's 250 cap is hit."""
        items = self._fetch(start, end)
        if len(items) < MAX_RECORDS or end - start <= self.min_window:
            return items
        mid = start + (end - start) / 2
        return self._window(mid, end) + self._window(start, mid)

    def crawl(self) -> list[Article]:
        end = datetime.now(timezone.utc)
        try:
            return dedupe(parse_articles(self._fetch(end - timedelta(hours=1), end)))
        except GdeltError as exc:
            log.warning("%s", exc)
            return []

    def backfill(self, start: datetime, end: datetime | None = None,
                 step: timedelta = timedelta(hours=6)) -> list[Article]:
        """Walk backwards from ``end`` to ``start`` in ``step`` windows. Raises
        GdeltError if the first window fails, so a broken setup is visible."""
        end = (end or datetime.now(timezone.utc)).replace(microsecond=0)
        start = start.replace(microsecond=0)
        oldest_supported = end - timedelta(days=90)
        if start < oldest_supported - timedelta(days=1):
            log.warning("GDELT DOC only covers about 3 months; articles before %s will be missing",
                        oldest_supported.date())
        articles: list[Article] = []
        cursor = end
        windows = 0
        while cursor > start:
            lo = max(start, cursor - step)
            try:
                items = self._window(lo, cursor)
            except GdeltError:
                if windows == 0:
                    raise
                log.exception("GDELT backfill stopped early at %s", cursor)
                break
            windows += 1
            articles.extend(parse_articles(items))
            cursor = lo
            if windows % 20 == 0:
                log.info("GDELT backfill reached %s (%d articles, %d requests)",
                         cursor, len(articles), self.requests)
        articles = dedupe(articles)
        log.info("GDELT backfill: %d articles back to %s in %d requests", len(articles), cursor, self.requests)
        return articles
