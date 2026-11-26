"""Gemini 2.5 Flash sentiment agent over the REST API (no SDK needed).

Uses structured JSON output (responseSchema) so we never have to strip
markdown fences, with the generation parameters from the report
(appendix 7.3.2). Requests are rate-limited and retried.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import requests

from .prompt import PROMPT_VERSION, RESPONSE_SCHEMA, build_prompt, normalise_label
from .types import SentimentResult

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _clip(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def parse_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull the JSON object out of a generateContent response."""
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError(f"No candidates in Gemini response: {payload.get('promptFeedback')}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object in Gemini output: {text[:200]}")
    return json.loads(match.group(0))


class GeminiSentimentAgent:
    backend = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", rpm: int = 60,
                 timeout: float = 60.0, retries: int = 4):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini backend")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self._min_interval = 60.0 / max(rpm, 1)
        self._last_call = 0.0
        self._lock = threading.Lock()
        self.session = requests.Session()

    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _request_body(self, title: str, body: str) -> dict[str, Any]:
        return {
            "contents": [{"role": "user", "parts": [{"text": build_prompt(title, body)}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
                # 2.5 Flash "thinks" by default; a classification task does not need it
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

    def score(self, title: str, body: str = "") -> SentimentResult:
        url = API_URL.format(model=self.model)
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._throttle()
            try:
                resp = self.session.post(
                    url,
                    params={"key": self.api_key},
                    json=self._request_body(title, body),
                    timeout=self.timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                data = parse_response(resp.json())
                return SentimentResult(
                    score=_clip(data.get("score"), -1.0, 1.0, 0.0),
                    label=normalise_label(data.get("label")),
                    confidence=_clip(data.get("confidence"), 0.0, 1.0, 0.5),
                    reasoning=str(data.get("reasoning", ""))[:1000],
                    backend=self.backend,
                    model=f"{self.model}/{PROMPT_VERSION}",
                )
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                wait = min(60.0, 2.0**attempt)
                log.warning("Gemini call failed (%s), retry %d/%d in %.0fs", exc, attempt, self.retries, wait)
                time.sleep(wait)
        raise RuntimeError(f"Gemini scoring failed after {self.retries} attempts: {last_error}")
