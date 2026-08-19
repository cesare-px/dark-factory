"""State machine orchestrating the 4-stage pipeline for a single ticket.

One TokenBudgetTracker is shared across all four agents for a given ticket
run, so the $-ceiling guardrail applies to the whole pipeline. Each agent
gets its OWN metered LLM client resolved from its own config entry, which is
the point of per-agent model overrides: a ticket can bill a cheap model for
the Validator and an expensive one for the Developer against one shared
ceiling.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dark_factory.agents.base import AgentDeps, AgentResult, AgentStatus, RunContext
from dark_factory.agents.developer import DeveloperAgent
from dark_factory.agents.planner import PlannerAgent
from dark_factory.agents.reviewer import ReviewerAgent
from dark_factory.agents.validator import ValidatorAgent
from dark_factory.config.model import FactoryConfig
from dark_factory.context import build_scanner
from dark_factory.guardrails.budget import BudgetExceededError, TokenBudgetTracker
from dark_factory.harness import build_harness
from dark_factory.intake.schema import FactoryTicket
from dark_factory.llm.client import MeteredLLMClient, build_client
from dark_factory.pricing import PriceBook

DepsFactory = Callable[[str, TokenBudgetTracker], AgentDeps]


@dataclass
class PipelineResult:
    """The end-to-end outcome of running all four agents against one ticket."""

    ticket_id: str
    final_status: AgentStatus
    phase_results: list[AgentResult] = field(default_factory=list)
    labels: set[str] = field(default_factory=set)
    comments: list[str] = field(default_factory=list)
    budget_summary: dict[str, Any] = field(default_factory=dict)


class DarkFactoryPipeline:
    """Runs Validator -> Planner -> Developer -> Reviewer for one ticket."""

    def __init__(
        self,
        config: FactoryConfig,
        *,
        root: Path = Path("."),
        deps_factory: DepsFactory | None = None,
    ) -> None:
        """Configure the pipeline; `deps_factory` defaults to real providers/harness."""
        self.config = config
        self.root = root
        self._deps_factory = deps_factory or self._default_deps_factory

    def _default_deps_factory(self, agent_name: str, tracker: TokenBudgetTracker) -> AgentDeps:
        spec = self.config.resolve_llm(agent_name)
        raw_client = build_client(spec)
        metered = MeteredLLMClient(raw_client, tracker, agent_name=agent_name)
        return AgentDeps(
            llm=metered,
            config=self.config,
            harness=build_harness(self.config.test_harness),
            scanner=build_scanner(self.config.context, root=self.root),
            root=self.root,
        )

    def _build_tracker(self) -> TokenBudgetTracker:
        price_book = PriceBook.builtin(on_unknown_model=self.config.pricing.on_unknown_model)
        if self.config.pricing.overrides:
            price_book = price_book.with_overrides(self.config.pricing.overrides)
        return TokenBudgetTracker(
            max_budget_usd=self.config.budget.max_usd,
            price_book=price_book,
            max_usd_per_agent=self.config.budget.max_usd_per_agent,
            on_exceed=self.config.budget.on_exceed,
        )

    def run(self, ticket: FactoryTicket) -> PipelineResult:
        """Run Validator, Planner, Developer, and Reviewer in sequence for `ticket`."""
        tracker = self._build_tracker()
        result = PipelineResult(ticket_id=ticket.ticket_id, final_status=AgentStatus.PASS)
        ctx = RunContext()

        def record_phase(phase_result: AgentResult) -> None:
            result.phase_results.append(phase_result)
            result.labels.update(self.config.label(key.value) for key in phase_result.labels)
            if phase_result.comment:
                result.comments.append(phase_result.comment)

        try:
            validation = ValidatorAgent(self._deps_factory("validator", tracker)).run(ticket, ctx)
            record_phase(validation)
            if validation.status is not AgentStatus.PASS:
                result.final_status = AgentStatus.FAIL
                return result

            planning = PlannerAgent(self._deps_factory("planner", tracker)).run(ticket, ctx)
            record_phase(planning)
            ctx = dataclasses.replace(
                ctx,
                plan_steps=tuple(planning.output.get("plan_steps", ())),
                affected_files=tuple(planning.output.get("affected_files", ())),
            )

            build = DeveloperAgent(self._deps_factory("developer", tracker)).run(ticket, ctx)
            record_phase(build)
            if build.status is not AgentStatus.PASS:
                result.final_status = build.status
                return result
            ctx = dataclasses.replace(
                ctx,
                branch=build.output.get("branch"),
                harness_history=tuple(build.output.get("history", ())),
            )

            review = ReviewerAgent(self._deps_factory("reviewer", tracker)).run(ticket, ctx)
            record_phase(review)
            result.final_status = review.status

        except BudgetExceededError as exc:
            result.final_status = AgentStatus.BLOCKED
            result.labels.add(self.config.label("blocked"))
            result.comments.append(f"Pipeline halted: {exc}")

        finally:
            result.budget_summary = tracker.summary()

        return result
