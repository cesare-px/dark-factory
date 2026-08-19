"""Anthropic Messages API adapter, over stdlib HTTP (no `anthropic` SDK).

Split into pure `build_payload`/`parse_usage`/`parse_response` functions plus
a thin class so the request/response mapping is unit-testable from a
recorded JSON fixture with zero network access.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from dark_factory.config.model import ResolvedLLM
from dark_factory.llm.errors import LLMAuthError
from dark_factory.llm.transport import HttpRequest, RetryPolicy, post_json
from dark_factory.llm.types import LLMRequest, LLMResponse, TokenUsage

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


def build_payload(request: LLMRequest, spec: ResolvedLLM) -> dict[str, Any]:
    """Translate a provider-agnostic `LLMRequest` into a Messages API body."""
    payload: dict[str, Any] = {
        "model": spec.model,
        "max_tokens": request.max_output_tokens,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }
    if request.system:
        payload["system"] = request.system
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    return payload


def parse_usage(raw_usage: Mapping[str, Any]) -> TokenUsage:
    """Extract token usage from a Messages API response's `usage` block."""
    return TokenUsage(
        input_tokens=raw_usage.get("input_tokens", 0),
        output_tokens=raw_usage.get("output_tokens", 0),
        cached_input_tokens=raw_usage.get("cache_read_input_tokens", 0),
        cache_write_tokens=raw_usage.get("cache_creation_input_tokens", 0),
    )


def parse_response(raw: Mapping[str, Any], spec: ResolvedLLM) -> LLMResponse:
    """Translate a raw Messages API response into an `LLMResponse`."""
    text = "".join(
        block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text"
    )
    return LLMResponse(
        text=text,
        usage=parse_usage(raw.get("usage", {})),
        provider="anthropic",
        model=raw.get("model", spec.model),
        finish_reason=raw.get("stop_reason") or "stop",
        raw=raw,
    )


class AnthropicClient:
    """LLMClient for the Anthropic Messages API."""

    provider = "anthropic"

    def __init__(self, spec: ResolvedLLM) -> None:
        """Resolve the API key from `spec.api_key_env` and set up auth headers.

        Raises:
            LLMAuthError: If the configured API key environment variable is
                unset or empty.
        """
        self.model = spec.model
        self._spec = spec
        self._base_url = spec.base_url or DEFAULT_BASE_URL
        api_key = os.environ.get(spec.api_key_env or "", "") if spec.api_key_env else ""
        if not api_key:
            raise LLMAuthError(
                f"anthropic provider: environment variable {spec.api_key_env!r} is not set or empty"
            )
        self._headers = {"x-api-key": api_key, "anthropic-version": API_VERSION}
        self._retry = RetryPolicy(max_attempts=spec.max_attempts)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send `request` to the Messages API and parse the reply."""
        raw = post_json(
            HttpRequest(
                url=f"{self._base_url}/messages",
                headers=self._headers,
                json_body=build_payload(request, self._spec),
                timeout_seconds=self._spec.timeout_seconds,
            ),
            retry=self._retry,
        )
        return parse_response(raw, self._spec)

    def close(self) -> None:
        """No persistent connection to close for this stdlib-HTTP client."""
