"""OpenAI Chat Completions adapter, over stdlib HTTP (no `openai` SDK).

Also the base class for `openai_compatible.py`: any OpenAI-shaped
`/chat/completions` endpoint (Kimi, GLM, Qwen, Ollama, vLLM, ...) reuses this
payload/response mapping and only swaps `base_url` and auth handling.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from dark_factory.config.model import ResolvedLLM
from dark_factory.llm.errors import LLMAuthError
from dark_factory.llm.transport import HttpRequest, RetryPolicy, post_json
from dark_factory.llm.types import LLMRequest, LLMResponse, TokenUsage, message_text

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def build_payload(request: LLMRequest, spec: ResolvedLLM) -> dict[str, Any]:
    """Translate a provider-agnostic `LLMRequest` into a chat/completions body."""
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.extend({"role": m.role, "content": message_text(m.content)} for m in request.messages)

    payload: dict[str, Any] = {
        "model": spec.model,
        "messages": messages,
        "max_tokens": request.max_output_tokens,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    return payload


def parse_usage(raw_usage: Mapping[str, Any]) -> TokenUsage:
    """Extract token usage from a chat/completions response's `usage` block."""
    if not raw_usage:
        return TokenUsage(is_estimated=True)
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    completion_details = raw_usage.get("completion_tokens_details") or {}
    return TokenUsage(
        input_tokens=raw_usage.get("prompt_tokens", 0),
        output_tokens=raw_usage.get("completion_tokens", 0),
        cached_input_tokens=prompt_details.get("cached_tokens", 0),
        reasoning_tokens=completion_details.get("reasoning_tokens", 0),
    )


def parse_response(raw: Mapping[str, Any], spec: ResolvedLLM, *, provider: str) -> LLMResponse:
    """Translate a raw chat/completions response into an `LLMResponse`."""
    choices = raw.get("choices") or [{}]
    message = choices[0].get("message", {})
    return LLMResponse(
        text=message.get("content") or "",
        usage=parse_usage(raw.get("usage") or {}),
        provider=provider,
        model=raw.get("model", spec.model),
        finish_reason=choices[0].get("finish_reason") or "stop",
        raw=raw,
    )


class OpenAIClient:
    """LLMClient for the OpenAI Chat Completions API."""

    provider = "openai"

    def __init__(self, spec: ResolvedLLM) -> None:
        """Resolve the API key from `spec.api_key_env` and set up auth headers.

        Raises:
            LLMAuthError: If the configured API key environment variable is
                unset or empty.
        """
        self.model = spec.model
        self._spec = spec
        self._base_url = (spec.base_url or DEFAULT_BASE_URL).rstrip("/")
        api_key = os.environ.get(spec.api_key_env or "", "") if spec.api_key_env else ""
        if not api_key:
            raise LLMAuthError(
                f"{self.provider} provider: environment variable {spec.api_key_env!r} "
                "is not set or empty"
            )
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._retry = RetryPolicy(max_attempts=spec.max_attempts)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send `request` to the chat/completions endpoint and parse the reply."""
        raw = post_json(
            HttpRequest(
                url=f"{self._base_url}/chat/completions",
                headers=self._headers,
                json_body=build_payload(request, self._spec),
                timeout_seconds=self._spec.timeout_seconds,
            ),
            retry=self._retry,
        )
        return parse_response(raw, self._spec, provider=self.provider)

    def close(self) -> None:
        """No persistent connection to close for this stdlib-HTTP client."""
