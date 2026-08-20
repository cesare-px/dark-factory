"""Phase 3: The Implementation Loop (Build-Test-Fix)."""

from __future__ import annotations

import json
import re
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
    "implementation plan and (on retries) the previous test failure, reply "
    "with ONLY a JSON object of the form "
    '{"files": [{"path": "relative/path.py", "content": "full file contents"}], '
    '"summary": "one-line description of the change"}'
    " -- the complete desired contents of every file that needs to exist or "
    "change to satisfy the plan. No prose, no markdown fences, just the JSON."
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_patch(text: str) -> dict[str, Any] | None:
    """Extract the `{"files": [...]}` object from a developer response.

    Tries a fenced ```json block first, then the whole response, so a model
    that wraps its answer in prose still parses. Returns None (never raises)
    on anything that isn't a well-formed `files` list, so a malformed
    response degrades to "no files written this attempt" rather than
    crashing the pipeline.
    """
    candidates = []
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1))
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("files"), list):
            return parsed
    return None


def _write_files(files: list[Any], root: Path) -> tuple[list[str], list[str]]:
    """Write each `{"path", "content"}` entry under `root`; refuse escapes.

    Returns `(written, rejected)` relative-path strings. A path that resolves
    outside `root` (via ".." or an absolute path) is skipped, not written --
    this executes LLM-authored content on disk, so escaping the sandboxed
    checkout is never acceptable regardless of what the model asked for.
    """
    written: list[str] = []
    rejected: list[str] = []
    root_resolved = root.resolve()

    for entry in files:
        if not isinstance(entry, dict):
            continue
        rel_path, content = entry.get("path"), entry.get("content")
        if not isinstance(rel_path, str) or not isinstance(content, str):
            continue

        target = (root / rel_path).resolve()
        if not target.is_relative_to(root_resolved):
            rejected.append(rel_path)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel_path)

    return written, rejected


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
        if last.get("parse_error"):
            raw = (
                "Your previous response could not be parsed as the required "
                'JSON {"files": [...]} object. Reply with ONLY that JSON -- '
                "no prose, no markdown fences."
            )
        else:
            raw = last["stderr"] or last["stdout"]
            if last.get("rejected_files"):
                raw = (
                    f"Rejected paths outside the project root: "
                    f"{last['rejected_files']}. Use only paths inside the checkout.\n{raw}"
                )
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
                response = self.deps.llm.complete(request)

                patch = _parse_patch(response.text)
                written: list[str] = []
                rejected: list[str] = []
                if patch is not None:
                    written, rejected = _write_files(patch.get("files", []), self.deps.root)

                result = harness.run(iteration=iteration, workdir=workdir)
                history.append(
                    {
                        "iteration": iteration,
                        "passed": result.passed,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "parse_error": patch is None,
                        "written_files": written,
                        "rejected_files": rejected,
                        # Otherwise a failed loop is unobservable after the
                        # fact -- nothing else records what the model said.
                        "raw_response": response.text,
                    }
                )

                if result.passed:
                    return AgentResult(
                        agent_name=self.name,
                        status=AgentStatus.PASS,
                        output={
                            "branch": branch,
                            "iterations": iteration,
                            "history": history,
                            "summary": patch.get("summary", "") if patch else "",
                        },
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
