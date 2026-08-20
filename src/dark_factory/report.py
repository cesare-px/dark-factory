"""A stable, schema-versioned report of one pipeline run.

Includes a Markdown renderer for $GITHUB_STEP_SUMMARY and issue comments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dark_factory.agents.base import AgentStatus
from dark_factory.agents.pipeline import PipelineResult

REPORT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class RunReport:
    """A schema-versioned, serializable summary of one pipeline run."""

    schema_version: str
    ticket_id: str
    final_status: str
    labels: list[str]
    comments: list[str]
    budget: dict[str, Any]
    phases: list[dict[str, Any]]

    @classmethod
    def from_pipeline_result(cls, result: PipelineResult) -> RunReport:
        """Build a report from a completed `PipelineResult`."""
        phases = []
        for p in result.phase_results:
            phase: dict[str, Any] = {
                "agent": p.agent_name,
                "status": p.status.value,
                "model": p.model,
                "llm_calls": p.llm_calls,
                "cost_usd": round(p.cost_usd, 6),
                "usage": {
                    "input_tokens": p.usage.input_tokens,
                    "output_tokens": p.usage.output_tokens,
                    "is_estimated": p.usage.is_estimated,
                },
            }
            # The build-test-fix loop is otherwise unobservable after the
            # fact -- a failed run gives no way to see what the model tried.
            if p.agent_name == "developer" and "history" in p.output:
                phase["history"] = p.output["history"]
            phases.append(phase)
        return cls(
            schema_version=REPORT_SCHEMA_VERSION,
            ticket_id=result.ticket_id,
            final_status=result.final_status.value,
            labels=sorted(result.labels),
            comments=list(result.comments),
            budget=result.budget_summary,
            phases=phases,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation, e.g. for `--output`."""
        return {
            "schema_version": self.schema_version,
            "ticket_id": self.ticket_id,
            "final_status": self.final_status,
            "labels": self.labels,
            "comments": self.comments,
            "budget": self.budget,
            "phases": self.phases,
        }

    def to_markdown(self) -> str:
        """Render this report for `$GITHUB_STEP_SUMMARY` or an issue comment."""
        lines = [
            f"### Dark Factory run: `{self.ticket_id}`",
            "",
            f"**Result:** `{self.final_status}`",
            f"**Labels:** {', '.join(f'`{label}`' for label in self.labels) or '(none)'}",
            "",
            "| Agent | Status | Model | Calls | Cost (USD) |",
            "|---|---|---|---|---|",
        ]
        for phase in self.phases:
            lines.append(
                f"| {phase['agent']} | {phase['status']} | {phase['model'] or '-'} | "
                f"{phase['llm_calls']} | ${phase['cost_usd']:.4f} |"
            )
        lines += [
            "",
            f"**Total spend:** ${self.budget.get('spent_usd', 0):.4f} / "
            f"${self.budget.get('max_budget_usd', 0):.2f}",
        ]
        if self.comments:
            lines += ["", "**Comments:**"] + [f"- {c}" for c in self.comments]
        return "\n".join(lines)


def exit_code_for(result: PipelineResult) -> int:
    """Map a pipeline result to the CLI's stable exit-code contract."""
    if result.final_status == AgentStatus.PASS:
        return 0
    if result.final_status == AgentStatus.FAIL:
        # One phase means the Validator rejected it; four means the
        # Reviewer did.
        return 10 if len(result.phase_results) <= 1 else 30
    if result.final_status == AgentStatus.BLOCKED:
        return 20
    return 1
