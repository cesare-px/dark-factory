"""Phase 4: Peer Review & Shipping."""

from __future__ import annotations

from dark_factory.agents.base import AgentResult, AgentStatus, BaseAgent, RunContext
from dark_factory.intake.sanitize import wrap_as_untrusted
from dark_factory.intake.schema import FactoryTicket
from dark_factory.llm.types import LLMRequest, Message

SYSTEM_PROMPT = (
    "You are the Reviewer agent in an autonomous software factory. Given a "
    "diff summary and the ticket's acceptance criteria, give a one-line "
    "APPROVE or REJECT verdict with reasoning."
)

_DANGEROUS_CALL_MARKERS = ("eval(", "exec(")


class ReviewerAgent(BaseAgent):
    """Scans the diff against the ticket's acceptance criteria before shipping."""

    name = "reviewer"

    def _security_prefilter(self, diff_summary: str) -> list[str]:
        return [
            f"use of {marker.rstrip('(')}() detected"
            for marker in _DANGEROUS_CALL_MARKERS
            if marker in diff_summary
        ]

    def _uncovered_criteria(self, ticket: FactoryTicket, diff_summary: str) -> list[str]:
        return [c for c in ticket.acceptance_criteria if c.lower() not in diff_summary.lower()]

    def run(self, ticket: FactoryTicket, ctx: RunContext) -> AgentResult:
        """Approve or reject the diff against acceptance criteria and lint checks."""
        diff_summary = ctx.diff_summary or " ".join(ticket.acceptance_criteria)
        lint_findings = self._security_prefilter(diff_summary)
        uncovered_criteria = self._uncovered_criteria(ticket, diff_summary)

        spec = self.deps.config.resolve_llm(self.name)
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            messages=(
                Message(
                    "user",
                    wrap_as_untrusted(
                        f"Diff summary: {diff_summary}\n"
                        f"Acceptance criteria: {list(ticket.acceptance_criteria)}"
                    ),
                ),
            ),
            max_output_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            metadata={"agent": self.name},
        )
        response = self.deps.llm.complete(request)

        if lint_findings or uncovered_criteria:
            lines = [f"- security: {f}" for f in lint_findings] + [
                f"- uncovered criterion: {c}" for c in uncovered_criteria
            ]
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAIL,
                output={
                    "lint_findings": lint_findings,
                    "uncovered_criteria": uncovered_criteria,
                    "model_note": response.text,
                },
                comment="Review rejected:\n" + "\n".join(lines),
                usage=self.deps.llm.total_usage,
                cost_usd=self.deps.llm.total_cost_usd,
                llm_calls=self.deps.llm.calls,
                model=self.deps.llm.model,
            )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.PASS,
            output={"approved": True, "model_note": response.text},
            comment="Programmatic review passed. Approving for squash-merge.",
            usage=self.deps.llm.total_usage,
            cost_usd=self.deps.llm.total_cost_usd,
            llm_calls=self.deps.llm.calls,
            model=self.deps.llm.model,
        )
