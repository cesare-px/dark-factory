"""Minimal stdlib JSON-over-HTTPS transport shared by every real adapter.

Deliberately not an SDK: keeping this to ~stdlib `urllib` is what lets every
built-in provider ship with zero extra install dependencies. Retries on
429/5xx with backoff; every exception message is redacted so a CI log never
leaks a credential.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, cast

from dark_factory.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMTransientError,
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_REDACT_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_-]{10,})"),
    re.compile(r"(AKIA[A-Z0-9]{12,})"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{10,}", re.IGNORECASE),
    re.compile(r"((?:x-api-key|authorization)\"?\s*[:=]\s*\"?)[^\s\"]{6,}", re.IGNORECASE),
]


def redact(text: str) -> str:
    """Mask API keys, bearer tokens, and auth headers in `text`."""
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(
            lambda m: m.group(1) + "…redacted…" if m.groups() else "…redacted…", text
        )
    return text


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry/backoff parameters for `post_json`."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 20.0


@dataclass(frozen=True, slots=True)
class HttpRequest:
    """One JSON-over-HTTPS request for `post_json` to send."""

    url: str
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    json_body: dict[str, Any] | None = None
    timeout_seconds: float = 120.0


def post_json(request: HttpRequest, *, retry: RetryPolicy | None = None) -> dict[str, Any]:
    """POST JSON, retrying on 429/5xx, and raise a redacted, typed error otherwise."""
    retry = retry or RetryPolicy()
    body = json.dumps(request.json_body).encode("utf-8") if request.json_body is not None else None
    headers = {"Content-Type": "application/json", **request.headers}

    last_exc: Exception | None = None
    for attempt in range(1, retry.max_attempts + 1):
        req = urllib.request.Request(request.url, data=body, headers=headers, method=request.method)
        try:
            with urllib.request.urlopen(req, timeout=request.timeout_seconds) as resp:
                return cast(dict[str, Any], json.loads(resp.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            payload = redact(exc.read().decode("utf-8", errors="replace"))
            status = exc.code
            if status == 401 or status == 403:
                raise LLMAuthError(
                    f"HTTP {status} from {_redact_url(request.url)}: {payload}"
                ) from None
            if status in _RETRYABLE_STATUS and attempt < retry.max_attempts:
                delay = _retry_delay(exc, attempt, retry)
                time.sleep(delay)
                last_exc = exc
                continue
            if status == 429:
                raise LLMRateLimitError(
                    f"HTTP 429 from {_redact_url(request.url)}: {payload}"
                ) from None
            if status in _RETRYABLE_STATUS:
                raise LLMTransientError(
                    f"HTTP {status} from {_redact_url(request.url)}: {payload}"
                ) from None
            raise LLMBadRequestError(
                f"HTTP {status} from {_redact_url(request.url)}: {payload}"
            ) from None
        except urllib.error.URLError as exc:
            if attempt < retry.max_attempts:
                time.sleep(
                    min(
                        retry.initial_backoff_seconds * (2 ** (attempt - 1)),
                        retry.max_backoff_seconds,
                    )
                )
                last_exc = exc
                continue
            raise LLMTransientError(
                f"connection error calling {_redact_url(request.url)}: {redact(str(exc))}"
            ) from None

    raise LLMTransientError(f"exhausted retries calling {_redact_url(request.url)}") from last_exc


def _retry_delay(exc: urllib.error.HTTPError, attempt: int, retry: RetryPolicy) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after is not None:
        try:
            return min(float(retry_after), retry.max_backoff_seconds)
        except ValueError:
            pass
    return min(
        float(retry.initial_backoff_seconds * (2 ** (attempt - 1))), retry.max_backoff_seconds
    )


def _redact_url(url: str) -> str:
    return re.sub(r"([?&](?:key|api_key|token)=)[^&]+", r"\1…redacted…", url)
