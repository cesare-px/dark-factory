"""Phase 2: Triaging & Technical Planning."""

from __future__ import annotations

from dark_factory.agents.base import AgentResult, AgentStatus, BaseAgent, RunContext
from dark_factory.context import ContextScan
from dark_factory.intake.sanitize import wrap_as_untrusted
from dark_factory.intake.schema import FactoryTicket
from dark_factory.llm.types import LLMRequest, Message

SYSTEM_PROMPT = (
    "You are the Planner agent in an autonomous software factory. Given a "
    "ticket and the repository files most likely relevant to it, write a "
    "short numbered implementation plan."
)


class PlannerAgent(BaseAgent):
    """Maps a validated ticket onto affected modules and a step-by-step plan."""

    name = "planner"

    def run(self, ticket: FactoryTicket, ctx: RunContext) -> AgentResult:
        """Scan the repo for relevant files and draft a step-by-step plan."""
        if self.deps.scanner is not None:
            scan = self.deps.scanner.scan(ticket, root=self.deps.root)
        else:
            scan = ContextScan(files=(), strategy="none", truncated=False)

        plan_steps = [
            f"Implement: {ticket.title}",
            *[f"Satisfy acceptance criterion: {c}" for c in ticket.acceptance_criteria],
            f"Run {self.deps.config.test_harness.command} and iterate until green",
        ]

        spec = self.deps.config.resolve_llm(self.name)
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            messages=(
                Message(
                    "user",
                    wrap_as_untrusted(
                        f"Ticket: {ticket.title}\n"
                        f"Candidate files ({scan.strategy}): {list(scan.files)}\n"
                        f"Draft steps: {plan_steps}"
                    ),
                ),
            ),
            max_output_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            metadata={"agent": self.name},
        )
        response = self.deps.llm.complete(request)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.PASS,
            output={
                "affected_files": scan.files,
                "plan_steps": plan_steps,
                "model_note": response.text,
            },
            usage=self.deps.llm.total_usage,
            cost_usd=self.deps.llm.total_cost_usd,
            llm_calls=self.deps.llm.calls,
            model=self.deps.llm.model,
        )
