"""Per-ticket financial token budgeting (blueprint §4.B), provider-agnostic.

Pricing is resolved through a PriceBook (dark_factory.pricing) rather than
hardcoded constants, so the same tracker prices an Anthropic call, an OpenAI
call, and a free local Ollama call correctly. Enforces a hard USD ceiling per
ticket to eliminate infinite-loop financial runaways.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from dark_factory.llm.types import TokenUsage
from dark_factory.pricing import PriceBook

DEFAULT_MAX_BUDGET_USD = 3.00


class BudgetExceededError(RuntimeError):
    """Raised when recording a spend would breach the hard USD ceiling."""

    def __init__(
        self, attempted_usd: float, spent_usd: float, max_budget_usd: float, *, agent: str
    ) -> None:
        """Build the error with the attempted spend, running total, and ceiling."""
        self.attempted_usd = attempted_usd
        self.spent_usd = spent_usd
        self.max_budget_usd = max_budget_usd
        self.agent = agent
        super().__init__(
            f"budget exceeded during {agent!r}: spend of ${attempted_usd:.4f} would bring total to "
            f"${spent_usd + attempted_usd:.4f}, at or over the ${max_budget_usd:.2f} ceiling"
        )


@dataclass
class _SpendRecord:
    agent_name: str
    provider: str
    model: str
    usage: TokenUsage
    cost_usd: float


@dataclass
class TokenBudgetTracker:
    """Tracks USD spend for a single ticket's pipeline run against a hard cap."""

    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD
    price_book: PriceBook = field(default_factory=PriceBook.builtin)
    max_usd_per_agent: Mapping[str, float] = field(default_factory=dict)
    on_exceed: Literal["block", "warn"] = "block"
    _ledger: list[_SpendRecord] = field(default_factory=list, repr=False)
    _spent_usd: float = field(default=0.0, repr=False)
    _spent_by_agent: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def spent_usd(self) -> float:
        """Total USD spent so far across all agents."""
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        """USD remaining before the ceiling is hit."""
        return max(self.max_budget_usd - self._spent_usd, 0.0)

    def estimate_cost_usd(self, *, provider: str, model: str, usage: TokenUsage) -> float:
        """Price `usage` for `provider`/`model` without recording a spend."""
        return self.price_book.cost_usd(provider, model, usage)

    def would_exceed(self, agent_name: str, cost_usd: float) -> bool:
        """Return whether adding `cost_usd` would breach the ticket or per-agent ceiling."""
        if self._spent_usd + cost_usd > self.max_budget_usd:
            return True
        agent_ceiling = self.max_usd_per_agent.get(agent_name)
        if agent_ceiling is not None:
            spent_so_far = self._spent_by_agent.get(agent_name, 0.0)
            if spent_so_far + cost_usd > agent_ceiling:
                return True
        return False

    def record(self, agent_name: str, *, provider: str, model: str, usage: TokenUsage) -> float:
        """Record actual usage; raises BudgetExceededError if it breaches a cap.

        A rejected spend is never appended to the ledger, so the caller can
        halt without double-charging on retry.
        """
        cost = self.estimate_cost_usd(provider=provider, model=model, usage=usage)
        if self.would_exceed(agent_name, cost) and self.on_exceed == "block":
            raise BudgetExceededError(cost, self._spent_usd, self.max_budget_usd, agent=agent_name)

        self._ledger.append(_SpendRecord(agent_name, provider, model, usage, cost))
        self._spent_usd += cost
        self._spent_by_agent[agent_name] = self._spent_by_agent.get(agent_name, 0.0) + cost
        return cost

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable summary of spend, by total and by agent."""
        estimated_calls = sum(1 for r in self._ledger if r.usage.is_estimated)
        return {
            "spent_usd": round(self._spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "max_budget_usd": self.max_budget_usd,
            "calls_recorded": len(self._ledger),
            "estimated_usage_calls": estimated_calls,
            "by_agent": {k: round(v, 6) for k, v in self._spent_by_agent.items()},
        }
