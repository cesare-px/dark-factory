"""Phase 3: The Implementation Loop (Build-Test-Fix)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dark_factory.agents import tools
from dark_factory.agents.base import AgentDeps, AgentResult, AgentStatus, BaseAgent, RunContext
from dark_factory.agents.tool_loop import ToolLoopLimitExceededError, ToolSpec, run_tool_loop
from dark_factory.config.model import ToolUseSettings
from dark_factory.guardrails.budget import BudgetExceededError
from dark_factory.guardrails.loop_tracker import IterationLoopTracker, LoopLimitExceededError
from dark_factory.intake.sanitize import sanitize_text, wrap_as_untrusted
from dark_factory.intake.schema import FactoryTicket
from dark_factory.labels import LabelKey
from dark_factory.llm.types import ToolDefinition
from dark_factory.naming import render_branch
from dark_factory.vcs import git_diff, sender_has_write_access

SYSTEM_PROMPT = (
    "You are the Developer agent in an autonomous software factory. Given an "
    "implementation plan and (on retries) the previous test failure, use the "
    "available tools to explore the codebase and make the necessary changes. "
    "Call finish_implementation with a summary once you believe the work is "
    "complete -- the test harness decides whether it actually passes, not you."
)

FINISH_TOOL = ToolDefinition(
    name="finish_implementation",
    description="Call this once you believe your changes satisfy the plan.",
    input_schema={
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "What you changed, and why."}},
        "required": ["summary"],
    },
)


def _build_tools(root: Path, allowed_command_prefixes: tuple[str, ...]) -> tuple[ToolSpec, ...]:
    """The Developer's full tool set: read/search/write/edit plus a permission-gated run_command."""
    return (
        ToolSpec(
            definition=ToolDefinition(
                name="read_file",
                description="Read a file's contents, given a path relative to the project root.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            handler=lambda path: tools.read_file(root, path),
        ),
        ToolSpec(
            definition=ToolDefinition(
                name="list_files",
                description="List files under the project root matching a glob pattern.",
                input_schema={
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            ),
            handler=lambda pattern: tools.list_files(root, pattern),
        ),
        ToolSpec(
            definition=ToolDefinition(
                name="grep",
                description="Search file contents under the project root for a substring.",
                input_schema={
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            ),
            handler=lambda pattern: tools.grep(root, pattern),
        ),
        ToolSpec(
            definition=ToolDefinition(
                name="write_file",
                description=(
                    "Write the complete contents of a file, creating it (and any parent "
                    "directories) if it doesn't exist yet. Use edit_file instead for a "
                    "targeted change to a file that already exists."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            ),
            handler=lambda path, content: tools.write_file(root, path, content),
        ),
        ToolSpec(
            definition=ToolDefinition(
                name="edit_file",
                description=(
                    "Replace exactly one occurrence of old_string with new_string in an "
                    "existing file. old_string must match exactly once in the file -- "
                    "include enough surrounding context to make it unique."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            ),
            handler=lambda path, old_string, new_string: tools.edit_file(
                root, path, old_string, new_string
            ),
        ),
        ToolSpec(
            definition=ToolDefinition(
                name="run_command",
                description=(
                    "Run a shell command to check your work (e.g. a linter, a syntax "
                    "check, or a build step). Only a limited, pre-approved set of "
                    "commands is allowed; a rejected command is not a bug, it's a "
                    "permission this ticket didn't request."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            ),
            handler=lambda command: tools.run_command(root, command, allowed_command_prefixes),
        ),
    )


def _resolve_allowed_command_prefixes(
    ticket: FactoryTicket, cfg: ToolUseSettings
) -> tuple[str, ...]:
    """Safe defaults, plus any authorized per-ticket command family.

    A checked box is a request, never a grant: no `GITHUB_TOKEN` (a local
    run), no `sender_login`, or the sender lacking write access on the repo
    all mean the request is silently dropped, not an error -- the run
    proceeds with just the safe defaults.
    """
    prefixes = list(cfg.safe_command_prefixes)
    if not ticket.requested_permissions:
        return tuple(prefixes)

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token or not ticket.sender_login:
        return tuple(prefixes)
    if not sender_has_write_access(ticket.context.repository, ticket.sender_login, token):
        return tuple(prefixes)

    for label in ticket.requested_permissions:
        prefixes.extend(cfg.command_families.get(label, ()))
    return tuple(prefixes)


class DeveloperAgent(BaseAgent):
    """Runs the bounded build-test-fix loop, giving the model real tool access."""

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
        if last.get("tool_loop_error"):
            raw = (
                "Your previous attempt did not finish cleanly: "
                f"{last['tool_loop_error']}. Be more decisive about calling "
                "finish_implementation once you've made your changes."
            )
        else:
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
        workdir = Path(self.deps.config.test_harness.working_dir)

        tool_use_cfg = self.deps.config.tool_use
        allowed_prefixes = _resolve_allowed_command_prefixes(ticket, tool_use_cfg)
        tool_registry = _build_tools(self.deps.root, allowed_prefixes)

        try:
            while True:
                iteration = self.loop_tracker.step()

                feedback = self._feedback_block(history)
                first_message = wrap_as_untrusted(
                    f"Plan: {list(ctx.plan_steps)}\nAttempt: {iteration}"
                )
                if feedback:
                    first_message = f"{first_message}\n{feedback}"

                summary = ""
                tool_loop_error: str | None = None
                if tool_use_cfg.enabled:
                    try:
                        loop_result = run_tool_loop(
                            self.deps.llm,
                            system=SYSTEM_PROMPT,
                            first_message=first_message,
                            tools=tool_registry,
                            finish_tool=FINISH_TOOL,
                            max_iterations=tool_use_cfg.max_iterations,
                            metadata={"agent": self.name},
                        )
                        summary = str(loop_result.finish_input.get("summary", ""))
                    except ToolLoopLimitExceededError as exc:
                        tool_loop_error = str(exc)
                else:
                    tool_loop_error = "tool_use is disabled for this repo"

                harness_result = harness.run(iteration=iteration, workdir=workdir)
                history.append(
                    {
                        "iteration": iteration,
                        "passed": harness_result.passed,
                        "stdout": harness_result.stdout,
                        "stderr": harness_result.stderr,
                        "tool_loop_error": tool_loop_error,
                        "summary": summary,
                    }
                )

                if harness_result.passed:
                    return AgentResult(
                        agent_name=self.name,
                        status=AgentStatus.PASS,
                        output={
                            "branch": branch,
                            "iterations": iteration,
                            "history": history,
                            "summary": summary,
                            "diff_summary": git_diff(self.deps.root),
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
