"""The LLMClient protocol and the metering wrapper every agent actually calls.

MeteredLLMClient is the ONLY place spend is ever recorded against a ticket's
TokenBudgetTracker: it estimates a worst-case cost before the call (raising
BudgetExceededError up front if that would breach the ceiling) and then
charges the *actual* usage the provider reports once the call returns.
"""

from __future__ import annotations

from typing import Protocol

from dark_factory.config.model import ResolvedLLM
from dark_factory.guardrails.budget import BudgetExceededError, TokenBudgetTracker
from dark_factory.llm.registry import resolve
from dark_factory.llm.tokens import estimate_request_tokens
from dark_factory.llm.types import LLMRequest, LLMResponse, TokenUsage


class LLMClient(Protocol):
    """The minimal interface every provider adapter implements."""

    provider: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send one request and return the provider's response."""
        ...

    def close(self) -> None:
        """Release any resources held by the client (connections, etc)."""
        ...


def build_client(spec: ResolvedLLM) -> LLMClient:
    """Resolve `spec.provider` and construct its client from `spec`."""
    factory = resolve(spec.provider)
    return factory(spec)


class MeteredLLMClient:
    """Wraps a raw LLMClient: preflight-reserves budget, calls, settles actuals."""

    def __init__(self, inner: LLMClient, tracker: TokenBudgetTracker, *, agent_name: str) -> None:
        """Bind a raw client to the shared budget tracker for one agent."""
        self._inner = inner
        self._tracker = tracker
        self._agent_name = agent_name
        self.calls = 0
        self.total_usage = TokenUsage()
        self.total_cost_usd = 0.0

    @property
    def provider(self) -> str:
        """The wrapped client's provider name."""
        return self._inner.provider

    @property
    def model(self) -> str:
        """The wrapped client's model id."""
        return self._inner.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Charge a worst-case preflight estimate, call, then settle the actual cost.

        Raises:
            BudgetExceededError: If the worst-case estimate would breach the
                ticket's budget ceiling before the call is even made.
        """
        worst_case_usage = TokenUsage(
            input_tokens=estimate_request_tokens(request),
            output_tokens=request.max_output_tokens,
            is_estimated=True,
        )
        worst_case_cost = self._tracker.estimate_cost_usd(
            provider=self._inner.provider, model=self._inner.model, usage=worst_case_usage
        )
        if self._tracker.would_exceed(self._agent_name, worst_case_cost):
            raise BudgetExceededError(
                worst_case_cost,
                self._tracker.spent_usd,
                self._tracker.max_budget_usd,
                agent=self._agent_name,
            )

        response = self._inner.complete(request)

        cost = self._tracker.record(
            self._agent_name, provider=response.provider, model=response.model, usage=response.usage
        )
        self.calls += 1
        self.total_usage = self.total_usage + response.usage
        self.total_cost_usd += cost
        return response

    def close(self) -> None:
        """Close the wrapped client."""
        self._inner.close()
