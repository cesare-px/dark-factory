import subprocess

from dark_factory.agents import developer
from dark_factory.agents.base import AgentDeps, AgentStatus, RunContext
from dark_factory.agents.developer import DeveloperAgent
from dark_factory.config import load_config
from dark_factory.config.model import ToolUseSettings
from dark_factory.guardrails.budget import TokenBudgetTracker
from dark_factory.harness import SubprocessHarness
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


def _deps(tmp_path, *, script: MockScript, max_iterations: int = 3) -> AgentDeps:
    cfg = load_config(
        overrides={
            "loop": {"max_iterations": max_iterations},
            "test_harness": {"working_dir": str(tmp_path), "command": "sh check.sh"},
        }
    )
    spec = cfg.resolve_llm("developer")
    tracker = TokenBudgetTracker(max_budget_usd=cfg.budget.max_usd)
    llm = MeteredLLMClient(MockClient(spec, script=script), tracker, agent_name="developer")
    harness = SubprocessHarness(cfg.test_harness)
    return AgentDeps(llm=llm, config=cfg, harness=harness, root=tmp_path)


def test_developer_uses_tools_to_write_files_and_harness_passes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "check.sh").write_text('grep -q "Hello, World!" hello.txt\n')

    script = MockScript(
        responses={
            "developer": [
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(
                            id="1",
                            name="write_file",
                            input={"path": "hello.txt", "content": "Hello, World!\n"},
                        ),
                    )
                ),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(
                            id="2",
                            name="finish_implementation",
                            input={"summary": "Wrote hello.txt"},
                        ),
                    )
                ),
            ]
        }
    )

    agent = DeveloperAgent(_deps(tmp_path, script=script))
    result = agent.run(_TICKET, RunContext(plan_steps=("Write hello.txt",)))

    assert result.status == AgentStatus.PASS
    assert (tmp_path / "hello.txt").read_text() == "Hello, World!\n"
    assert result.output["summary"] == "Wrote hello.txt"
    assert result.output["history"][-1]["passed"] is True
    assert "hello.txt" in result.output["diff_summary"]


def test_developer_uses_edit_file_on_an_existing_file(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "check.sh").write_text('grep -q "Hello, World!" hello.txt\n')
    (tmp_path / "hello.txt").write_text("Hello, Goodbye!\n")

    script = MockScript(
        responses={
            "developer": [
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(
                            id="1",
                            name="edit_file",
                            input={
                                "path": "hello.txt",
                                "old_string": "Hello, Goodbye!",
                                "new_string": "Hello, World!",
                            },
                        ),
                    )
                ),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(
                            id="2",
                            name="finish_implementation",
                            input={"summary": "Fixed greeting"},
                        ),
                    )
                ),
            ]
        }
    )

    agent = DeveloperAgent(_deps(tmp_path, script=script))
    result = agent.run(_TICKET, RunContext(plan_steps=("Fix hello.txt",)))

    assert result.status == AgentStatus.PASS
    assert (tmp_path / "hello.txt").read_text() == "Hello, World!\n"


def test_developer_blocks_without_crashing_when_model_never_finishes(tmp_path):
    (tmp_path / "check.sh").write_text('grep -q "Hello, World!" hello.txt\n')

    script = MockScript(responses={"developer": "just some prose, no tool calls"})

    agent = DeveloperAgent(_deps(tmp_path, script=script, max_iterations=2))
    result = agent.run(_TICKET, RunContext(plan_steps=("Write hello.txt",)))

    assert result.status == AgentStatus.BLOCKED
    assert all(h["tool_loop_error"] for h in result.output["history"])
    assert not (tmp_path / "hello.txt").exists()


def test_developer_path_escaping_via_write_file_is_rejected_and_fed_back(tmp_path):
    (tmp_path / "check.sh").write_text('grep -q "Hello, World!" hello.txt\n')

    script = MockScript(
        responses={
            "developer": [
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(
                            id="1",
                            name="write_file",
                            input={"path": "../evil.txt", "content": "pwned"},
                        ),
                    )
                ),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(id="2", name="finish_implementation", input={"summary": "done"}),
                    )
                ),
            ]
        }
    )

    agent = DeveloperAgent(_deps(tmp_path, script=script, max_iterations=1))
    result = agent.run(_TICKET, RunContext(plan_steps=("Write hello.txt",)))

    assert result.status == AgentStatus.BLOCKED  # check.sh still fails: hello.txt was never written
    assert not (tmp_path.parent / "evil.txt").exists()


def test_run_command_is_restricted_to_safe_prefixes(tmp_path):
    (tmp_path / "check.sh").write_text('grep -q "Hello, World!" hello.txt\n')

    script = MockScript(
        responses={
            "developer": [
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(id="1", name="run_command", input={"command": "rm -rf /"}),
                    )
                ),
                ScriptedResponse(
                    tool_calls=(
                        ToolCall(id="2", name="finish_implementation", input={"summary": "done"}),
                    )
                ),
            ]
        }
    )

    agent = DeveloperAgent(_deps(tmp_path, script=script, max_iterations=1))
    result = agent.run(_TICKET, RunContext(plan_steps=("Write hello.txt",)))

    # The dangerous command was rejected (fed back as a tool error, not run),
    # so nothing was actually written -- confirmed indirectly via BLOCKED.
    assert result.status == AgentStatus.BLOCKED


_TOOL_USE_CFG = ToolUseSettings(
    safe_command_prefixes=("pytest",),
    command_families={"Package installation": ("pip install",)},
)


def _ticket_with_permissions(**overrides) -> FactoryTicket:
    return FactoryTicket(
        ticket_id="org/project-a#1",
        title="t",
        description="d",
        acceptance_criteria=(),
        context=TicketContext(repository="org/project-a"),
        **overrides,
    )


def test_resolve_prefixes_returns_safe_defaults_when_nothing_requested():
    ticket = _ticket_with_permissions()
    assert developer._resolve_allowed_command_prefixes(ticket, _TOOL_USE_CFG) == ("pytest",)


def test_resolve_prefixes_ignores_request_without_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    ticket = _ticket_with_permissions(
        requested_permissions=("Package installation",), sender_login="alice"
    )
    assert developer._resolve_allowed_command_prefixes(ticket, _TOOL_USE_CFG) == ("pytest",)


def test_resolve_prefixes_ignores_request_without_sender(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    ticket = _ticket_with_permissions(
        requested_permissions=("Package installation",), sender_login=None
    )
    assert developer._resolve_allowed_command_prefixes(ticket, _TOOL_USE_CFG) == ("pytest",)


def test_resolve_prefixes_ignores_request_when_sender_lacks_write_access(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setattr(developer, "sender_has_write_access", lambda *a, **k: False)
    ticket = _ticket_with_permissions(
        requested_permissions=("Package installation",), sender_login="alice"
    )
    assert developer._resolve_allowed_command_prefixes(ticket, _TOOL_USE_CFG) == ("pytest",)


def test_resolve_prefixes_grants_authorized_family(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setattr(developer, "sender_has_write_access", lambda *a, **k: True)
    ticket = _ticket_with_permissions(
        requested_permissions=("Package installation",), sender_login="alice"
    )
    assert developer._resolve_allowed_command_prefixes(ticket, _TOOL_USE_CFG) == (
        "pytest",
        "pip install",
    )


def test_resolve_prefixes_ignores_unrecognized_label(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setattr(developer, "sender_has_write_access", lambda *a, **k: True)
    ticket = _ticket_with_permissions(
        requested_permissions=("Not A Real Family",), sender_login="alice"
    )
    assert developer._resolve_allowed_command_prefixes(ticket, _TOOL_USE_CFG) == ("pytest",)
