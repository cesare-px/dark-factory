"""One adapter for any OpenAI-shaped `/chat/completions` endpoint.

This is what makes Kimi (Moonshot), GLM (Zhipu), Qwen (DashScope), Ollama,
vLLM, OpenRouter, Together, and Groq all work with zero orchestrator code
changes: point `base_url` at the endpoint (or use a verified `preset`) and
everything else -- payload shape, usage parsing, budget metering -- is
inherited from the OpenAI adapter.

Presets are convenience only, never a gate: any `base_url` works.
"""

from __future__ import annotations

import os

from dark_factory.config.model import ResolvedLLM
from dark_factory.llm.providers.openai import OpenAIClient, build_payload, parse_response
from dark_factory.llm.tokens import estimate_usage_from_text
from dark_factory.llm.transport import HttpRequest, RetryPolicy, post_json
from dark_factory.llm.types import LLMRequest, LLMResponse

# Verified against each provider's official docs at implementation time.
PRESETS: dict[str, str] = {
    "moonshot": "https://api.moonshot.ai/v1",  # Kimi
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",  # GLM
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # Qwen
    "ollama": "http://localhost:11434/v1",  # local, no auth
}


class OpenAICompatibleClient(OpenAIClient):
    """LLMClient for any OpenAI-shaped `/chat/completions` endpoint."""

    provider = "openai_compatible"

    def __init__(self, spec: ResolvedLLM) -> None:
        """Resolve `base_url` from `spec.base_url` or a named preset.

        Raises:
            ValueError: If neither `base_url` nor a known `preset` is set.
        """
        base_url = spec.base_url or PRESETS.get(spec.preset or "")
        if not base_url:
            raise ValueError(
                "openai_compatible provider requires either `base_url` or a known `preset` "
                f"({', '.join(sorted(PRESETS))}) in the llm config"
            )

        # Skip OpenAIClient.__init__'s hard requirement for an API key --
        # local endpoints like Ollama take none.
        self.model = spec.model
        self._spec = spec
        self._base_url = base_url.rstrip("/")
        api_key = os.environ.get(spec.api_key_env or "", "") if spec.api_key_env else ""
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._retry = RetryPolicy(max_attempts=spec.max_attempts)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send `request` and estimate usage client-side if the reply omits it."""
        raw = post_json(
            HttpRequest(
                url=f"{self._base_url}/chat/completions",
                headers=self._headers,
                json_body=build_payload(request, self._spec),
                timeout_seconds=self._spec.timeout_seconds,
            ),
            retry=self._retry,
        )
        response = parse_response(raw, self._spec, provider=self.provider)

        if response.usage.is_estimated or (
            response.usage.input_tokens == 0 and response.usage.output_tokens == 0
        ):
            prompt_text = (request.system or "") + "".join(m.content for m in request.messages)
            response = LLMResponse(
                text=response.text,
                usage=estimate_usage_from_text(prompt_text, response.text),
                provider=response.provider,
                model=response.model,
                finish_reason=response.finish_reason,
                raw=response.raw,
            )
        return response
