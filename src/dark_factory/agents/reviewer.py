"""Phase 4: Peer Review & Shipping."""

from __future__ import annotations

from dark_factory.agents.base import AgentResult, AgentStatus, BaseAgent, RunContext
from dark_factory.intake.sanitize import wrap_as_untrusted
from dark_factory.intake.schema import FactoryTicket
from dark_factory.llm.types import LLMRequest, Message
from dark_factory.naming import render_pr_title
from dark_factory.vcs import ship

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

    def _model_approved(self, response_text: str) -> bool:
        """True only if the model's verdict clearly starts with APPROVE.

        Whether a diff satisfies a natural-language acceptance criterion is
        a semantic question -- there's no reliable string-matching
        substitute for actually reading the diff, and any such heuristic
        would just be overfit to whichever criteria phrasing motivated it.
        The model already gets asked for exactly this verdict; use it.
        Fails closed (REJECT) on anything ambiguous or missing.
        """
        return response_text.strip().upper().startswith("APPROVE")

    def run(self, ticket: FactoryTicket, ctx: RunContext) -> AgentResult:
        """Approve or reject the diff against acceptance criteria and lint checks."""
        diff_summary = ctx.diff_summary
        lint_findings = self._security_prefilter(diff_summary)

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

        if lint_findings:
            lines = [f"- security: {f}" for f in lint_findings]
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAIL,
                output={"lint_findings": lint_findings, "model_note": response.text},
                comment="Review rejected:\n" + "\n".join(lines),
                usage=self.deps.llm.total_usage,
                cost_usd=self.deps.llm.total_cost_usd,
                llm_calls=self.deps.llm.calls,
                model=self.deps.llm.model,
            )

        if not self._model_approved(response.text):
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAIL,
                output={"lint_findings": [], "model_note": response.text},
                comment=f"Review rejected:\n{response.text.strip()}",
                usage=self.deps.llm.total_usage,
                cost_usd=self.deps.llm.total_cost_usd,
                llm_calls=self.deps.llm.calls,
                model=self.deps.llm.model,
            )

        branch = ctx.branch
        if branch:
            issue_number = ticket.ticket_id.rsplit("#", 1)[-1]
            ship_result = ship(
                self.deps.root,
                branch=branch,
                base=ticket.context.branch_target,
                title=render_pr_title(self.deps.config.naming, ticket),
                body=f"Closes #{issue_number}\n\n{diff_summary}",
                repo=ticket.context.repository,
            )
            ship_note = (
                f"Opened {ship_result.pr_url}"
                if ship_result.created
                else f"No PR opened: {ship_result.reason}"
            )
        else:
            ship_result = None
            ship_note = "No PR opened: no branch computed"

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.PASS,
            output={
                "approved": True,
                "model_note": response.text,
                "shipped": ship_result.created if ship_result else False,
                "pr_url": ship_result.pr_url if ship_result else None,
            },
            comment=f"Programmatic review passed. Approving for squash-merge.\n\n{ship_note}",
            usage=self.deps.llm.total_usage,
            cost_usd=self.deps.llm.total_cost_usd,
            llm_calls=self.deps.llm.calls,
            model=self.deps.llm.model,
        )
