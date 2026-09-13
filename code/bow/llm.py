"""Minimal Gemini REST client with on-disk caching and token accounting.

The REST API is called directly rather than through an SDK so that the exact
``usageMetadata`` for every call is available for ``evaluation/usage_report.md``,
and so the submission has no SDK version to pin.

Responses are cached by a hash of the request payload. A full dataset run
therefore costs one pass; re-runs are free and deterministic, which is what the
challenge asks for.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = os.environ.get("BOW_MODEL", "gemini-3.5-flash-lite")
DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".cache", "llm"
)

# Published Gemini pricing (USD per 1M tokens). Used only for the cost estimate
# in the usage report; override with BOW_PRICE_IN / BOW_PRICE_OUT if it changes.
PRICING = {
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-3.7-flash": (0.30, 2.50),
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}


@dataclass
class Usage:
    """Running totals for the usage report."""

    provider: str = "Google"
    model: str = DEFAULT_MODEL
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self) -> float:
        price_in, price_out = PRICING.get(self.model, (0.30, 2.50))
        price_in = float(os.environ.get("BOW_PRICE_IN", price_in))
        price_out = float(os.environ.get("BOW_PRICE_OUT", price_out))
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


@dataclass
class Client:
    """A thin, cached JSON-mode wrapper around ``generateContent``."""

    model: str = DEFAULT_MODEL
    cache_dir: str = DEFAULT_CACHE
    api_key: str | None = None
    max_retries: int = 4
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        self.usage.model = self.model
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ---- cache -----------------------------------------------------------------

    def _cache_path(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(blob + self.model.encode()).hexdigest()[:32]
        return os.path.join(self.cache_dir, f"{digest}.json")

    # ---- request ---------------------------------------------------------------

    def generate_json(
        self,
        system: str,
        parts: list[dict],
        response_schema: dict | None = None,
        temperature: float = 0.0,
    ) -> dict | None:
        """Ask for a JSON object. Returns ``None`` when the call cannot be made."""
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        if response_schema:
            payload["generationConfig"]["responseSchema"] = _to_gemini_schema(response_schema)

        path = self._cache_path(payload)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                cached = json.load(fh)
            self.usage.cached_calls += 1
            # Cached calls still count toward the reported totals: they describe
            # the run that produced output.csv, not the billing of a re-run.
            self.usage.calls += 1
            self.usage.input_tokens += cached.get("input_tokens", 0)
            self.usage.output_tokens += cached.get("output_tokens", 0)
            return cached.get("data")

        if not self.available:
            return None

        raw = self._post(payload)
        if raw is None:
            self.usage.failures += 1
            return None

        meta = raw.get("usageMetadata", {})
        in_tok = int(meta.get("promptTokenCount", 0))
        out_tok = int(meta.get("candidatesTokenCount", 0)) + int(
            meta.get("thoughtsTokenCount", 0)
        )
        self.usage.calls += 1
        self.usage.input_tokens += in_tok
        self.usage.output_tokens += out_tok

        data = _extract_json(raw)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"data": data, "input_tokens": in_tok, "output_tokens": out_tok},
                fh,
                ensure_ascii=False,
            )
        return data

    def _post(self, payload: dict) -> dict | None:
        url = f"{API_ROOT}/{self.model}:generateContent"
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key or "",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # 429/5xx are worth retrying; anything else is a real error.
                if exc.code not in (429, 500, 502, 503, 504):
                    return None
                delay = _retry_delay(exc)
                if delay is not None:
                    # A daily quota reports a delay far beyond any sensible
                    # wait; give up rather than stall the whole run.
                    if delay > 90:
                        return None
                    time.sleep(delay + 1)
                    continue
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(min(2**attempt, 20))
        return None

    def report(self, requests_count: int) -> dict:
        u = self.usage
        return {
            **asdict(u),
            "total_tokens": u.total_tokens,
            "avg_tokens_per_request": round(u.total_tokens / max(requests_count, 1), 1),
            "estimated_cost_usd": round(u.cost_usd(), 4),
            "estimated_cost_per_request_usd": round(u.cost_usd() / max(requests_count, 1), 6),
        }


def _retry_delay(exc: urllib.error.HTTPError) -> float | None:
    """The server's own RetryInfo, in seconds, when it supplies one."""
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - the body is best-effort diagnostics
        return None
    for detail in body.get("error", {}).get("details", []):
        if detail.get("@type", "").endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def image_part(path: str) -> dict:
    with open(path, "rb") as fh:
        return {
            "inlineData": {
                "mimeType": "image/png",
                "data": base64.b64encode(fh.read()).decode("ascii"),
            }
        }


def _to_gemini_schema(schema: dict) -> dict:
    """Strip JSON-Schema keys the Gemini schema dialect rejects."""
    allowed = {"type", "properties", "items", "required", "description", "enum", "nullable"}
    out = {k: v for k, v in schema.items() if k in allowed}
    if "properties" in out:
        out["properties"] = {k: _to_gemini_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _to_gemini_schema(out["items"])
    return out


def _extract_json(raw: dict) -> dict | None:
    try:
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
