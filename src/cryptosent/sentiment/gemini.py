"""Gemini sentiment agent over the REST API (no SDK needed).

* The API key travels in the ``x-goog-api-key`` header, never in the URL, and
  anything we log is passed through :func:`redact`.
* Structured JSON output (responseSchema) with the report's generation
  parameters (appendix 7.3.2).
* Several articles are scored per request (``batch_size``), which cuts the
  number of calls roughly tenfold. That matters on the free tier.
* Errors are split into retryable ones (429 per-minute limits, 5xx, network)
  and configuration problems (unknown model, rejected key, daily quota used
  up). The latter raise :class:`GeminiConfigError` immediately instead of
  retrying, so a broken setup produces one clear log line, not thousands.
* If a model answers 404 (some listed models do), the client moves on to
  ``gemini-flash-latest`` and then to the best Flash models this key lists,
  trying at most ``MAX_FALLBACKS`` models before it gives up.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import requests

from .prompt import PROMPT_VERSION, RESPONSE_SCHEMA, SYSTEM_PROMPT, build_prompt, normalise_label
from .types import SentimentResult

log = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_SECRET_RE = re.compile(
    r"(?i)(key=|x-goog-api-key['\"]?\s*[:=]\s*['\"]?)[^&\s'\"]+"  # key=... / header echoes
    r"|AIza[0-9A-Za-z_\-]{20,}"  # classic Google API keys
    r"|AQ\.[0-9A-Za-z_\-.]{20,}"  # newer Google key / token format
)
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
# models that can generate text but are not meant for this task
_EXCLUDE = ("image", "tts", "audio", "live", "embedding", "aqa", "thinking-exp", "learnlm", "robotics", "computer-use")


def redact(text: Any) -> str:
    """Remove API keys from any text before it is logged or raised."""
    return _SECRET_RE.sub(lambda m: (m.group(1) or "") + "<redacted>", str(text))


class GeminiConfigError(RuntimeError):
    """A problem retrying will not fix: bad model, rejected key, quota used up."""

    def __init__(self, message: str, kind: str = "config"):
        super().__init__(redact(message))
        self.kind = kind  # "model" | "auth" | "quota" | "config"


def _clip(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def _response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError(f"No candidates in Gemini response: {payload.get('promptFeedback')}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    return re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()


def parse_response(payload: dict[str, Any]) -> Any:
    """Pull the JSON value (object or array) out of a generateContent response."""
    text = _response_text(payload)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.S)
        if not match:
            raise ValueError(f"No JSON in Gemini output: {text[:200]}") from None
        return json.loads(match.group(1))


def pick_flash_model(names: list[str]) -> str | None:
    """Choose the best general Flash model from a ListModels result."""
    names = [n.removeprefix("models/") for n in names]
    flash = [n for n in names if "flash" in n and not any(x in n for x in _EXCLUDE)]
    if not flash:
        return None
    for alias in ("gemini-flash-latest",):
        if alias in flash:
            return alias

    def rank(name: str) -> tuple:
        version = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        return (
            "lite" not in name,  # full Flash beats Flash-Lite
            "preview" not in name and "exp" not in name,  # stable beats preview
            float(version.group(1)) if version else 0.0,  # newest generation
            not re.search(r"-\d{3}$|-\d{2}-\d{4}$", name),  # alias beats dated pin
        )

    return max(flash, key=rank)


def batch_schema(n: int) -> dict[str, Any]:
    item = {
        "type": "OBJECT",
        "properties": {"id": {"type": "INTEGER"}, **RESPONSE_SCHEMA["properties"]},
        "required": ["id", *RESPONSE_SCHEMA["required"]],
    }
    return {"type": "ARRAY", "items": item, "minItems": n, "maxItems": n}


def build_batch_prompt(items: list[tuple[str, str]], max_chars: int = 3000) -> str:
    blocks = []
    for i, (title, body) in enumerate(items):
        text = f"Title: {title}\n{body}".strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        blocks.append(f"<article id={i}>\n{text}\n</article>")
    return (
        f"{SYSTEM_PROMPT}\n\nThere are {len(items)} articles below. Judge each one independently and "
        f"return a JSON array with exactly {len(items)} objects, one per article, each carrying the "
        f"article's id.\n\n---\n" + "\n\n".join(blocks)
    )


class GeminiSentimentAgent:
    backend = "gemini"

    FALLBACK_MODEL = "gemini-flash-latest"
    MAX_FALLBACKS = 3

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite", rpm: int = 10,
                 batch_size: int = 10, timeout: float = 90.0, retries: int = 4):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini backend")
        self._api_key = api_key
        self.model = model.removeprefix("models/")
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.retries = retries
        self._min_interval = 60.0 / max(rpm, 1)
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._use_thinking_off = True
        self._minimal_config = False  # set after a 400: drop optional generation settings
        self._tried: set[str] = set()
        self.disabled: GeminiConfigError | None = None
        self.session = requests.Session()
        self.session.headers.update({"x-goog-api-key": api_key, "Content-Type": "application/json"})

    def __repr__(self) -> str:  # never show the key
        return f"GeminiSentimentAgent(model={self.model!r})"

    # -- HTTP helpers -------------------------------------------------------
    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    @staticmethod
    def _error_text(resp: requests.Response) -> str:
        try:
            err = resp.json().get("error", {})
            return redact(f"HTTP {resp.status_code} {err.get('status', '')}: {err.get('message', '')}".strip())
        except ValueError:
            return redact(f"HTTP {resp.status_code}: {resp.text[:300]}")

    def list_models(self) -> list[str]:
        names: list[str] = []
        page_token = None
        for _ in range(10):
            params = {"pageSize": 1000, **({"pageToken": page_token} if page_token else {})}
            resp = self.session.get(f"{API_BASE}/models", params=params, timeout=self.timeout)
            if resp.status_code != 200:
                raise GeminiConfigError(f"Could not list Gemini models: {self._error_text(resp)}", "auth")
            data = resp.json()
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    names.append(m["name"])
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return names

    def resolve_model(self) -> str:
        """Check the configured model once; on 404 switch to an available Flash model."""
        resp = self.session.get(f"{API_BASE}/models/{self.model}", timeout=self.timeout)
        if resp.status_code == 200:
            return self.model
        if resp.status_code in (400, 401, 403):
            raise GeminiConfigError(
                f"Gemini rejected the API key ({self._error_text(resp)}). "
                "Create a new key at https://aistudio.google.com/apikey and set GEMINI_API_KEY.", "auth")
        if resp.status_code != 404:
            resp.raise_for_status()
        return self._fall_back()

    def _fall_back(self) -> str:
        """Move on from a model that answered 404: gemini-flash-latest first,
        then the best remaining listed Flash model."""
        self._tried.add(self.model)
        previous = self.model
        if len(self._tried) >= self.MAX_FALLBACKS:
            raise GeminiConfigError(
                f"Gemini models {sorted(self._tried)} all answered 404. Set GEMINI_MODEL to a model "
                "that works for your key (test one in Google AI Studio).", "model")
        try:
            available = [n.removeprefix("models/") for n in self.list_models()]
        except (GeminiConfigError, requests.RequestException):
            available = []
        if self.FALLBACK_MODEL not in self._tried and (self.FALLBACK_MODEL in available or not available):
            choice = self.FALLBACK_MODEL
        else:
            choice = pick_flash_model([n for n in available if n not in self._tried])
        if not choice:
            raise GeminiConfigError(
                f"Model '{previous}' is not available and no other Flash model is listed for this key. "
                f"Available: {', '.join(available[:15]) or 'none'}", "model")
        log.warning("Gemini model '%s' returned 404; using '%s' instead. Set GEMINI_MODEL to choose another.",
                    previous, choice)
        self.model = choice
        return choice

    def _generate(self, prompt: str, schema: dict[str, Any], max_tokens: int) -> Any:
        if self.disabled is not None:
            raise self.disabled
        last_error: str = ""
        attempt = 0
        while attempt < self.retries:
            attempt += 1
            url = f"{API_BASE}/models/{self.model}:generateContent"
            if self._minimal_config:
                # only settings every current model accepts; the prompt still asks for JSON
                config: dict[str, Any] = {"temperature": 0.2, "responseMimeType": "application/json"}
            else:
                config = {
                    "temperature": 0.2, "topP": 0.8, "topK": 40, "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json", "responseSchema": schema,
                }
            if self._use_thinking_off and not self._minimal_config:
                # classification needs no "thinking"; older 2.5 models accept a 0 budget
                config["thinkingConfig"] = {"thinkingBudget": 0}
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
            self._throttle()
            try:
                resp = self.session.post(url, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = redact(exc)
                wait = min(60.0, 2.0 ** attempt)
            else:
                if resp.status_code == 200:
                    try:
                        return parse_response(resp.json())
                    except (ValueError, json.JSONDecodeError) as exc:
                        last_error = redact(exc)
                        wait = 2.0
                else:
                    text = self._error_text(resp)
                    last_error = text
                    if resp.status_code == 400 and self._use_thinking_off and "thinking" in text.lower():
                        self._use_thinking_off = False  # model does not accept a 0 budget
                        attempt -= 1  # not a failed attempt; bounded (happens once)
                        continue
                    if resp.status_code == 400 and not self._minimal_config and "api key" not in text.lower():
                        # a newer model may reject an optional field; retry once with a minimal request
                        log.warning("Gemini rejected the request (%s); retrying with a minimal request", text)
                        self._minimal_config = True
                        attempt -= 1
                        continue
                    if resp.status_code == 404:
                        try:
                            self._fall_back()
                        except GeminiConfigError as exc:
                            self.disabled = exc
                            raise
                        attempt -= 1  # retry with the new model; bounded by MAX_FALLBACKS
                        continue
                    if resp.status_code == 429:
                        if re.search(r"per ?day|PerDay|daily", resp.text, re.I):
                            self.disabled = GeminiConfigError(
                                f"Gemini daily quota is used up ({text}). Remaining articles stay "
                                "unscored and are picked up by the next run.", "quota")
                            raise self.disabled
                        hint = _RETRY_DELAY_RE.search(resp.text)
                        wait = float(hint.group(1)) + 1 if hint else min(60.0, 5.0 * attempt)
                    elif resp.status_code >= 500:
                        wait = min(60.0, 2.0 ** attempt)
                    else:
                        auth = resp.status_code in (401, 403) or "api key" in text.lower()
                        kind = "auth" if auth else "config"
                        self.disabled = GeminiConfigError(
                            f"Gemini request failed and will not be retried: {text} (model '{self.model}')", kind)
                        raise self.disabled
            if attempt < self.retries:
                log.warning("Gemini call failed (%s), retry %d/%d in %.0fs", last_error, attempt, self.retries, wait)
                time.sleep(wait)
        raise RuntimeError(f"Gemini scoring failed after {self.retries} attempts: {last_error}")

    # -- public API ---------------------------------------------------------
    def _result(self, data: dict[str, Any]) -> SentimentResult:
        return SentimentResult(
            score=_clip(data.get("score"), -1.0, 1.0, 0.0),
            label=normalise_label(data.get("label")),
            confidence=_clip(data.get("confidence"), 0.0, 1.0, 0.5),
            reasoning=str(data.get("reasoning", ""))[:1000],
            backend=self.backend,
            model=f"{self.model}/{PROMPT_VERSION}",
        )

    def score(self, title: str, body: str = "") -> SentimentResult:
        data = self._generate(build_prompt(title, body), RESPONSE_SCHEMA, 1024)
        if isinstance(data, list):
            data = data[0] if data else {}
        return self._result(data)

    def score_batch(self, items: list[tuple[str, str]]) -> list[SentimentResult]:
        """Score many articles, ``batch_size`` per request. Articles the model
        skipped in a batch are re-asked one by one."""
        out: list[SentimentResult | None] = [None] * len(items)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            if len(chunk) == 1:
                out[start] = self.score(*chunk[0])
                continue
            data = self._generate(build_batch_prompt(chunk), batch_schema(len(chunk)), 400 * len(chunk) + 512)
            rows = data if isinstance(data, list) else data.get("results", []) if isinstance(data, dict) else []
            for row in rows:
                try:
                    idx = int(row.get("id"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if 0 <= idx < len(chunk) and out[start + idx] is None:
                    out[start + idx] = self._result(row)
            for i in range(len(chunk)):
                if out[start + i] is None:
                    out[start + i] = self.score(*chunk[i])
        return out  # type: ignore[return-value]
