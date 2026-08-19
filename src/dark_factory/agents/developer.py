"""Phase 3: The Implementation Loop (Build-Test-Fix)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dark_factory.agents.base import AgentDeps, AgentResult, AgentStatus, BaseAgent, RunContext
from dark_factory.guardrails.budget import BudgetExceededError
from dark_factory.guardrails.loop_tracker import IterationLoopTracker, LoopLimitExceededError
from dark_factory.intake.sanitize import sanitize_text, wrap_as_untrusted
from dark_factory.intake.schema import FactoryTicket
from dark_factory.labels import LabelKey
from dark_factory.llm.types import LLMRequest, Message
from dark_factory.naming import render_branch

SYSTEM_PROMPT = (
    "You are the Developer agent in an autonomous software factory. Given an "
    "implementation plan and (on retries) the previous test failure, describe "
    "the patch you would apply next."
)


class DeveloperAgent(BaseAgent):
    """Runs the bounded build-test-fix loop against a technical plan."""

    name = "developer"

    def __init__(self, deps: AgentDeps) -> None:
        """Set up the build-test-fix loop tracker for this run."""
        super().__init__(deps)
        max_iterations = deps.config.max_iterations_for(self.name)
        self.loop_tracker = IterationLoopTracker(
            loop_name=f"{self.name}-btf", max_iterations=max_iterations
        )

    def _feedback_block(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return ""
        last = history[-1]
        raw = last["stderr"] or last["stdout"]
        report = sanitize_text(raw, field_name="harness_output")
        return wrap_as_untrusted(report.clean_text, label="TEST_HARNESS_OUTPUT")

    def run(self, ticket: FactoryTicket, ctx: RunContext) -> AgentResult:
        """Run the bounded build-test-fix loop until tests pass or a cap is hit."""
        branch = render_branch(self.deps.config.naming, ticket)
        history: list[dict[str, Any]] = []
        harness = self.deps.harness
        if harness is None:
            raise ValueError(f"{self.name} agent requires a test harness, but none was configured")
        spec = self.deps.config.resolve_llm(self.name)
        workdir = Path(self.deps.config.test_harness.working_dir)

        try:
            while True:
                iteration = self.loop_tracker.step()

                feedback = self._feedback_block(history)
                prompt = wrap_as_untrusted(f"Plan: {list(ctx.plan_steps)}\nAttempt: {iteration}")
                if feedback:
                    prompt = f"{prompt}\n{feedback}"

                request = LLMRequest(
                    system=SYSTEM_PROMPT,
                    messages=(Message("user", prompt),),
                    max_output_tokens=spec.max_output_tokens,
                    temperature=spec.temperature,
                    metadata={"agent": self.name},
                )
                self.deps.llm.complete(request)

                result = harness.run(iteration=iteration, workdir=workdir)
                history.append(
                    {
                        "iteration": iteration,
                        "passed": result.passed,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )

                if result.passed:
                    return AgentResult(
                        agent_name=self.name,
                        status=AgentStatus.PASS,
                        output={"branch": branch, "iterations": iteration, "history": history},
                        usage=self.deps.llm.total_usage,
                        cost_usd=self.deps.llm.total_cost_usd,
                        llm_calls=self.deps.llm.calls,
                        model=self.deps.llm.model,
                    )

        except LoopLimitExceededError as exc:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.BLOCKED,
                output={"branch": branch, "history": history, "reason": str(exc)},
                labels=(LabelKey.BLOCKED,),
                comment=(
                    f"Build-test-fix loop for {ticket.ticket_id} exceeded "
                    f"{self.loop_tracker.max_iterations} iterations without passing tests. "
                    "Escalating to a human operator."
                ),
                usage=self.deps.llm.total_usage,
                cost_usd=self.deps.llm.total_cost_usd,
                llm_calls=self.deps.llm.calls,
                model=self.deps.llm.model,
            )
        except BudgetExceededError as exc:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.BLOCKED,
                output={"branch": branch, "history": history, "reason": str(exc)},
                labels=(LabelKey.BLOCKED,),
                comment=(
                    f"Token budget for {ticket.ticket_id} was exceeded mid-implementation. "
                    "Escalating to a human operator."
                ),
                usage=self.deps.llm.total_usage,
                cost_usd=self.deps.llm.total_cost_usd,
                llm_calls=self.deps.llm.calls,
                model=self.deps.llm.model,
            )
