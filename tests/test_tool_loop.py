import pytest

from dark_factory.agents.tool_loop import ToolLoopLimitExceededError, ToolSpec, run_tool_loop
from dark_factory.config import load_config
from dark_factory.guardrails.budget import TokenBudgetTracker
from dark_factory.llm.client import MeteredLLMClient
from dark_factory.llm.providers.mock import MockClient, MockScript, ScriptedResponse
from dark_factory.llm.types import ToolCall, ToolDefinition

_FINISH_TOOL = ToolDefinition(
    name="finish_implementation",
    description="Signal the task is done.",
    input_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
)


def _llm(script: MockScript) -> MeteredLLMClient:
    cfg = load_config()
    spec = cfg.resolve_llm("developer")
    tracker = TokenBudgetTracker(max_budget_usd=cfg.budget.max_usd)
    return MeteredLLMClient(MockClient(spec, script=script), tracker, agent_name="developer")


def test_run_tool_loop_executes_a_tool_then_finishes():
    calls = []

    def read_file(path: str) -> str:
        calls.append(path)
        return "file contents"

    script = MockScript(
        responses={
            "developer": [
                ScriptedResponse(
                    tool_calls=(ToolCall(id="1", name="read_file", input={"path": "a.py"}),)
                ),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(id="2", name="finish_implementation", input={"summary": "done"}),
                    )
                ),
            ]
        }
    )
    tools = (
        ToolSpec(
            definition=ToolDefinition(name="read_file", description="Read a file", input_schema={}),
            handler=read_file,
        ),
    )

    result = run_tool_loop(
        _llm(script),
        system="you are a developer",
        first_message="implement the ticket",
        tools=tools,
        finish_tool=_FINISH_TOOL,
        max_iterations=5,
        metadata={"agent": "developer"},
    )

    assert calls == ["a.py"]
    assert result.finish_input == {"summary": "done"}
    assert result.iterations == 2


def test_run_tool_loop_feeds_back_handler_errors_without_crashing():
    def flaky_tool() -> str:
        raise ValueError("boom")

    script = MockScript(
        responses={
            "developer": [
                ScriptedResponse(tool_calls=(ToolCall(id="1", name="flaky_tool", input={}),)),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(id="2", name="finish_implementation", input={"summary": "ok"}),
                    )
                ),
            ]
        }
    )
    tools = (
        ToolSpec(
            definition=ToolDefinition(name="flaky_tool", description="", input_schema={}),
            handler=flaky_tool,
        ),
    )

    result = run_tool_loop(
        _llm(script),
        system="s",
        first_message="m",
        tools=tools,
        finish_tool=_FINISH_TOOL,
        max_iterations=5,
        metadata={"agent": "developer"},
    )

    assert result.finish_input == {"summary": "ok"}


def test_run_tool_loop_reports_unknown_tool_without_crashing():
    script = MockScript(
        responses={
            "developer": [
                ScriptedResponse(tool_calls=(ToolCall(id="1", name="does_not_exist", input={}),)),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(id="2", name="finish_implementation", input={"summary": "ok"}),
                    )
                ),
            ]
        }
    )

    result = run_tool_loop(
        _llm(script),
        system="s",
        first_message="m",
        tools=(),
        finish_tool=_FINISH_TOOL,
        max_iterations=5,
        metadata={"agent": "developer"},
    )

    assert result.finish_input == {"summary": "ok"}


def test_run_tool_loop_raises_when_model_never_calls_finish():
    script = MockScript(responses={"developer": "just some prose, no tool calls"})

    with pytest.raises(ToolLoopLimitExceededError, match="without calling the finish tool"):
        run_tool_loop(
            _llm(script),
            system="s",
            first_message="m",
            tools=(),
            finish_tool=_FINISH_TOOL,
            max_iterations=5,
            metadata={"agent": "developer"},
        )


def test_run_tool_loop_raises_when_iteration_cap_is_hit():
    def noop_tool() -> str:
        return "ok"

    script = MockScript(
        responses={
            "developer": ScriptedResponse(tool_calls=(ToolCall(id="1", name="noop", input={}),))
        }
    )
    tools = (
        ToolSpec(
            definition=ToolDefinition(name="noop", description="", input_schema={}),
            handler=noop_tool,
        ),
    )

    with pytest.raises(ToolLoopLimitExceededError, match="exceeded 2 tool-use iterations"):
        run_tool_loop(
            _llm(script),
            system="s",
            first_message="m",
            tools=tools,
            finish_tool=_FINISH_TOOL,
            max_iterations=2,
            metadata={"agent": "developer"},
        )
