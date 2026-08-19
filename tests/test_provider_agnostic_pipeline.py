"""Proof that the pipeline behaves identically regardless of which LLM provider backs it.

Same control flow, same final status, only the billed provider/model and
cost differ. Transport is stubbed so this runs offline.
"""

import pytest

from dark_factory.agents.base import AgentDeps, AgentStatus
from dark_factory.agents.pipeline import DarkFactoryPipeline
from dark_factory.config import load_config
from dark_factory.context import ScriptedScanner
from dark_factory.harness import ScriptedHarness
from dark_factory.intake.schema import FactoryTicket, TicketContext
from dark_factory.llm.client import MeteredLLMClient, build_client


def _ticket() -> FactoryTicket:
    return FactoryTicket(
        ticket_id="org/project-a#1",
        title="Implement JWT Auth Refresh Tokens",
        description="As a user I want tokens to rotate on expiration so sessions stay secure.",
        acceptance_criteria=("Tokens must rotate on expiration", "Must pass auth.test.js"),
        context=TicketContext(repository="org/project-a"),
    )


def _fake_anthropic_response(request):
    return {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "model": request.json_body["model"],
        "stop_reason": "end_turn",
    }


def _fake_openai_response(request):
    return {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "model": request.json_body["model"],
    }


@pytest.mark.parametrize(
    "llm_overrides",
    [
        {"provider": "mock", "model": "mock-default"},
        {"provider": "anthropic", "model": "claude-sonnet-5"},
        {"provider": "openai", "model": "gpt-5"},
        {"provider": "openai_compatible", "preset": "ollama", "model": "qwen2.5-coder:32b"},
    ],
    ids=["mock", "anthropic", "openai", "openai_compatible"],
)
def test_pipeline_behaves_identically_across_providers(monkeypatch, llm_overrides):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        "dark_factory.llm.providers.anthropic.post_json",
        lambda req, retry=None: _fake_anthropic_response(req),
    )
    monkeypatch.setattr(
        "dark_factory.llm.providers.openai.post_json",
        lambda req, retry=None: _fake_openai_response(req),
    )
    monkeypatch.setattr(
        "dark_factory.llm.providers.openai_compatible.post_json",
        lambda req, retry=None: _fake_openai_response(req),
    )

    cfg = load_config(overrides={"llm": llm_overrides})

    def deps_factory(agent_name, tracker):
        spec = cfg.resolve_llm(agent_name)
        metered = MeteredLLMClient(build_client(spec), tracker, agent_name=agent_name)
        return AgentDeps(
            llm=metered,
            config=cfg,
            harness=ScriptedHarness(passes_on_attempt=1),
            scanner=ScriptedScanner(),
        )

    pipeline = DarkFactoryPipeline(cfg, deps_factory=deps_factory)
    result = pipeline.run(_ticket())

    assert result.final_status == AgentStatus.PASS
    assert [p.agent_name for p in result.phase_results] == [
        "validator",
        "planner",
        "developer",
        "reviewer",
    ]
    for phase in result.phase_results:
        assert phase.model == llm_overrides["model"]
        assert phase.llm_calls >= 1

    if llm_overrides["provider"] in ("mock", "openai_compatible"):
        assert result.budget_summary["spent_usd"] == 0.0
    else:
        assert result.budget_summary["spent_usd"] > 0.0
