"""Offline provider for exercising the pipeline's control flow without spending money.

`provider: mock` in config resolves here. Priced at $0 by default (see
pricing.FREE_BY_DEFAULT_PROVIDERS) unless a test explicitly overrides pricing
to exercise BudgetExceededError.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dark_factory.config.model import ResolvedLLM
from dark_factory.llm.errors import LLMTransientError
from dark_factory.llm.tokens import estimate_usage_from_text
from dark_factory.llm.types import LLMRequest, LLMResponse, TokenUsage


@dataclass
class MockScript:
    """Canned completions keyed by `request.metadata["agent"]`, with a "*" fallback.

    Each value is either a single string (repeated every call) or a list
    consumed one-per-call (the last entry repeats once exhausted).
    """

    responses: dict[str, str | list[str]] = field(default_factory=dict)
    fixed_usage: TokenUsage | None = None
    fail_on_call: int | None = None  # 1-indexed call number (per agent) to fail

    def response_for(self, agent: str, call_index: int) -> str:
        """Return the scripted response text for `agent`'s `call_index`-th call."""
        script = self.responses.get(agent, self.responses.get("*"))
        if script is None:
            return f"[mock response for {agent}, call {call_index + 1}]"
        if isinstance(script, str):
            return script
        return script[min(call_index, len(script) - 1)]


class MockClient:
    """LLMClient that returns scripted responses instead of calling a real API."""

    provider = "mock"

    def __init__(self, spec: ResolvedLLM, *, script: MockScript | None = None) -> None:
        """Bind an optional `MockScript`; defaults to a generic echo script."""
        self.model = spec.model
        self._script = script or MockScript()
        self._call_counts: dict[str, int] = {}

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next scripted response for this request's agent."""
        agent = request.metadata.get("agent", "*")
        call_index = self._call_counts.get(agent, 0)
        self._call_counts[agent] = call_index + 1

        if self._script.fail_on_call is not None and call_index + 1 == self._script.fail_on_call:
            raise LLMTransientError(f"mock: injected failure on {agent} call {call_index + 1}")

        text = self._script.response_for(agent, call_index)
        if self._script.fixed_usage is not None:
            usage = self._script.fixed_usage
        else:
            prompt_text = (request.system or "") + "".join(m.content for m in request.messages)
            usage = estimate_usage_from_text(prompt_text, text)

        return LLMResponse(text=text, usage=usage, provider=self.provider, model=self.model)

    def close(self) -> None:
        """No resources to release for this in-memory client."""
