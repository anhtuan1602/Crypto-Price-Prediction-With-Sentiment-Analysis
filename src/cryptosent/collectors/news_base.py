"""Shared article model, HTTP session with anti-ban measures, and text cleaning
(report section 4.1.1 a and b)."""
from __future__ import annotations

import hashlib
import html
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/18.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.6; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
]


@dataclass
class Article:
    url: str
    title: str
    published_at: datetime  # always timezone-aware UTC
    source: str  # collector name, e.g. "coinmarketcap"
    body: str = ""
    publisher: str | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def article_id(self) -> str:
        return article_hash(self.url, self.title)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}".strip() if self.body else self.title

    def to_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["_id"] = self.article_id
        return doc

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "Article":
        keys = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in doc.items() if k in keys}
        data["published_at"] = to_utc(data["published_at"])
        if "fetched_at" in data and data["fetched_at"] is not None:
            data["fetched_at"] = to_utc(data["fetched_at"])
        return cls(**data)


def normalize_url(url: str) -> str:
    url = url.strip()
    url = re.sub(r"[?#].*$", "", url)  # strip query/fragment (tracking params)
    return url.rstrip("/").lower()


def article_hash(url: str, title: str) -> str:
    """SHA-256 of normalised URL + title, used as the MongoDB _id for dedup."""
    key = f"{normalize_url(url)}|{clean_text(title).lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def to_utc(value: Any) -> datetime:
    """Parse ISO strings, RFC-822 dates, epoch seconds/ms or datetimes into aware UTC."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            dt = parsedate_to_datetime(text)
    else:
        raise TypeError(f"Cannot parse datetime from {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_WS = re.compile(r"\s+")
_NON_TEXT = re.compile(r"[^\w\s.,;:!?'\"%$€£()\-/&+]")


def clean_text(raw: str | None) -> str:
    """Strip HTML, scripts and noise; keep punctuation that matters for NLP
    features (! and ? counts, %, $)."""
    if not raw:
        return ""
    if "<" in raw and ">" in raw:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "svg", "figure"]):
            tag.decompose()
        raw = soup.get_text(" ")
    text = html.unescape(raw)
    text = re.sub(r"[  -​  　]", " ", text)  # unicode spaces
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("—", "-")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _NON_TEXT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def dedupe(articles: Iterable[Article]) -> list[Article]:
    seen: set[str] = set()
    out = []
    for art in articles:
        if art.article_id in seen:
            continue
        seen.add(art.article_id)
        out.append(art)
    return out


class PoliteSession:
    """requests wrapper with user-agent rotation, optional proxy pool,
    retries with exponential backoff and jitter.

    Failures return ``None`` instead of raising, so one bad source never
    breaks a DAG run (report: defensive error handling)."""

    def __init__(
        self,
        proxies: list[str] | None = None,
        timeout: float = 20.0,
        retries: int = 3,
        backoff: float = 1.5,
    ):
        self.session = requests.Session()
        self.proxies = proxies or []
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/json,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        headers.update(extra or {})
        return headers

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        return_errors: bool = False,
    ) -> requests.Response | None:
        """GET with retries on 429/5xx. A 4xx is not retried; it returns None
        unless ``return_errors`` (the caller wants the API's error body)."""
        for attempt in range(1, self.retries + 1):
            proxy = random.choice(self.proxies) if self.proxies else None
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    headers=self._headers(headers),
                    timeout=self.timeout,
                    proxies={"http": proxy, "https": proxy} if proxy else None,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                if resp.status_code >= 400:
                    log.warning("GET %s -> HTTP %s (not retrying): %s", url, resp.status_code,
                                " ".join(resp.text[:200].split()))
                    return resp if return_errors else None
                return resp
            except requests.RequestException as exc:
                wait = self.backoff**attempt + random.uniform(0, 1)
                log.warning("GET %s failed (%s), attempt %d/%d", url, exc, attempt, self.retries)
                if attempt < self.retries:
                    time.sleep(wait)
        return None

    def get_json(self, url: str, **kwargs: Any) -> Any | None:
        resp = self.get(url, **kwargs)
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError:
            log.warning("Non-JSON response from %s", url)
            return None
