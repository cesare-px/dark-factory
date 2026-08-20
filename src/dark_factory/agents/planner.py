"""Phase 2: Triaging & Technical Planning."""

from __future__ import annotations

from pathlib import Path

from dark_factory.agents import tools
from dark_factory.agents.base import AgentResult, AgentStatus, BaseAgent, RunContext
from dark_factory.agents.tool_loop import ToolLoopLimitExceededError, ToolSpec, run_tool_loop
from dark_factory.context import ContextScan
from dark_factory.intake.sanitize import wrap_as_untrusted
from dark_factory.intake.schema import FactoryTicket
from dark_factory.llm.types import ToolDefinition

SYSTEM_PROMPT = (
    "You are the Planner agent in an autonomous software factory. Explore "
    "the repository with the available tools to understand what needs to "
    "change, then call submit_plan with a short numbered implementation "
    "plan and the files it affects. You never write or edit files yourself "
    "-- that's the Developer agent's job."
)

SUBMIT_PLAN_TOOL = ToolDefinition(
    name="submit_plan",
    description="Call this once you've explored enough to plan the implementation.",
    input_schema={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Numbered implementation steps.",
            },
            "affected_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files this plan expects to touch.",
            },
        },
        "required": ["steps", "affected_files"],
    },
)


def _build_tools(root: Path) -> tuple[ToolSpec, ...]:
    """The Planner's read-only tool set -- it explores, it never writes."""
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
    )


class PlannerAgent(BaseAgent):
    """Explores the repo via tool use and drafts the model's own implementation plan."""

    name = "planner"

    def run(self, ticket: FactoryTicket, ctx: RunContext) -> AgentResult:
        """Let the model explore the repo, then return the plan it submits."""
        if self.deps.scanner is not None:
            scan = self.deps.scanner.scan(ticket, root=self.deps.root)
        else:
            scan = ContextScan(files=(), strategy="none", truncated=False)

        default_steps = [
            f"Implement: {ticket.title}",
            *[f"Satisfy acceptance criterion: {c}" for c in ticket.acceptance_criteria],
            f"Run {self.deps.config.test_harness.command} and iterate until green",
        ]

        first_message = wrap_as_untrusted(
            f"Ticket: {ticket.title}\n"
            f"Description: {ticket.description}\n"
            f"Acceptance criteria: {list(ticket.acceptance_criteria)}\n"
            f"Candidate files ({scan.strategy}, a starting hint -- explore further with "
            f"your tools as needed): {list(scan.files)}"
        )

        tool_use_cfg = self.deps.config.tool_use
        steps, affected_files, note = (
            default_steps,
            scan.files,
            "tool_use disabled; used the default plan",
        )

        if tool_use_cfg.enabled:
            try:
                loop_result = run_tool_loop(
                    self.deps.llm,
                    system=SYSTEM_PROMPT,
                    first_message=first_message,
                    tools=_build_tools(self.deps.root),
                    finish_tool=SUBMIT_PLAN_TOOL,
                    max_iterations=tool_use_cfg.max_iterations,
                    metadata={"agent": self.name},
                )
                submitted_steps = [str(s) for s in loop_result.finish_input.get("steps") or []]
                steps = submitted_steps or default_steps
                affected_files = tuple(
                    str(f) for f in loop_result.finish_input.get("affected_files") or []
                )
                note = "plan submitted via tool use"
            except ToolLoopLimitExceededError as exc:
                note = f"tool-use planning failed ({exc}); fell back to the default plan"

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.PASS,
            output={"affected_files": affected_files, "plan_steps": steps, "model_note": note},
            usage=self.deps.llm.total_usage,
            cost_usd=self.deps.llm.total_cost_usd,
            llm_calls=self.deps.llm.calls,
            model=self.deps.llm.model,
        )
