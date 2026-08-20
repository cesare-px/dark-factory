import json
import subprocess

from dark_factory.agents.base import AgentDeps, AgentStatus, RunContext
from dark_factory.agents.developer import DeveloperAgent
from dark_factory.config import load_config
from dark_factory.guardrails.budget import TokenBudgetTracker
from dark_factory.harness import SubprocessHarness
from dark_factory.intake.schema import FactoryTicket, TicketContext
from dark_factory.llm.client import MeteredLLMClient
from dark_factory.llm.providers.mock import MockClient, MockScript

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


def test_valid_json_response_writes_files_and_harness_passes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "check.sh").write_text('grep -q "Hello, World!" hello.txt\n')
    response = json.dumps(
        {
            "files": [{"path": "hello.txt", "content": "Hello, World!\n"}],
            "summary": "Wrote hello.txt",
        }
    )

    agent = DeveloperAgent(_deps(tmp_path, script=MockScript(responses={"developer": response})))
    result = agent.run(_TICKET, RunContext(plan_steps=("Write hello.txt",)))

    assert result.status == AgentStatus.PASS
    assert (tmp_path / "hello.txt").read_text() == "Hello, World!\n"
    assert result.output["summary"] == "Wrote hello.txt"
    assert result.output["history"][-1]["passed"] is True
    assert "hello.txt" in result.output["diff_summary"]
    assert "Hello, World!" in result.output["diff_summary"]


def test_malformed_response_blocks_without_crashing(tmp_path):
    (tmp_path / "check.sh").write_text('grep -q "Hello, World!" hello.txt\n')

    agent = DeveloperAgent(
        _deps(
            tmp_path,
            script=MockScript(responses={"developer": "Sure, here is a description of the patch."}),
            max_iterations=2,
        )
    )
    result = agent.run(_TICKET, RunContext(plan_steps=("Write hello.txt",)))

    assert result.status == AgentStatus.BLOCKED
    assert all(h["parse_error"] for h in result.output["history"])
    assert not (tmp_path / "hello.txt").exists()


def test_path_escaping_write_is_rejected(tmp_path):
    (tmp_path / "check.sh").write_text('grep -q "Hello, World!" hello.txt\n')
    response = json.dumps({"files": [{"path": "../evil.txt", "content": "pwned"}]})

    agent = DeveloperAgent(
        _deps(tmp_path, script=MockScript(responses={"developer": response}), max_iterations=1)
    )
    result = agent.run(_TICKET, RunContext(plan_steps=("Write hello.txt",)))

    assert result.status == AgentStatus.BLOCKED
    assert result.output["history"][0]["rejected_files"] == ["../evil.txt"]
    assert not (tmp_path.parent / "evil.txt").exists()
