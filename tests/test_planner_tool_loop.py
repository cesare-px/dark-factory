from dark_factory.agents.base import AgentDeps, AgentStatus, RunContext
from dark_factory.agents.planner import PlannerAgent
from dark_factory.config import load_config
from dark_factory.context import ScriptedScanner
from dark_factory.guardrails.budget import TokenBudgetTracker
from dark_factory.intake.schema import FactoryTicket, TicketContext
from dark_factory.llm.client import MeteredLLMClient
from dark_factory.llm.providers.mock import MockClient, MockScript, ScriptedResponse
from dark_factory.llm.types import ToolCall

_TICKET = FactoryTicket(
    ticket_id="org/project-a#1",
    title="Add a hello world greeting",
    description="As a user I want a greeting so the app says hello.",
    acceptance_criteria=("hello.txt contains 'Hello, World!'",),
    context=TicketContext(repository="org/project-a"),
)


def _deps(tmp_path, *, script: MockScript) -> AgentDeps:
    cfg = load_config()
    spec = cfg.resolve_llm("planner")
    tracker = TokenBudgetTracker(max_budget_usd=cfg.budget.max_usd)
    llm = MeteredLLMClient(MockClient(spec, script=script), tracker, agent_name="planner")
    return AgentDeps(llm=llm, config=cfg, scanner=ScriptedScanner(), root=tmp_path)


def test_planner_submits_its_own_plan_via_tool_use(tmp_path):
    (tmp_path / "greet.py").write_text("def greet(): pass\n")

    script = MockScript(
        responses={
            "planner": [
                ScriptedResponse(
                    tool_calls=(ToolCall(id="1", name="read_file", input={"path": "greet.py"}),)
                ),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(
                            id="2",
                            name="submit_plan",
                            input={
                                "steps": ["Add a docstring to greet()"],
                                "affected_files": ["greet.py"],
                            },
                        ),
                    )
                ),
            ]
        }
    )

    result = PlannerAgent(_deps(tmp_path, script=script)).run(_TICKET, RunContext())

    assert result.status == AgentStatus.PASS
    assert result.output["plan_steps"] == ["Add a docstring to greet()"]
    assert result.output["affected_files"] == ("greet.py",)
    assert result.output["model_note"] == "plan submitted via tool use"


def test_planner_falls_back_to_default_plan_when_tool_loop_fails(tmp_path):
    script = MockScript(responses={"planner": "just some prose, never calls submit_plan"})

    result = PlannerAgent(_deps(tmp_path, script=script)).run(_TICKET, RunContext())

    assert result.status == AgentStatus.PASS  # planner never blocks the pipeline
    assert result.output["plan_steps"][0] == f"Implement: {_TICKET.title}"
    assert "fell back to the default plan" in result.output["model_note"]


def test_planner_has_no_write_tools(tmp_path):
    # write_file isn't in planner's registry at all -- calling it is reported
    # to the model as an unknown tool, not executed.
    script = MockScript(
        responses={
            "planner": [
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(
                            id="1",
                            name="write_file",
                            input={"path": "evil.py", "content": "pwned"},
                        ),
                    )
                ),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(
                            id="2", name="submit_plan", input={"steps": ["x"], "affected_files": []}
                        ),
                    )
                ),
            ]
        }
    )

    result = PlannerAgent(_deps(tmp_path, script=script)).run(_TICKET, RunContext())

    assert result.status == AgentStatus.PASS
    assert not (tmp_path / "evil.py").exists()
