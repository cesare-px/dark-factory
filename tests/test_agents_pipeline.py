import dataclasses

from dark_factory.agents.base import AgentDeps, AgentStatus
from dark_factory.agents.pipeline import DarkFactoryPipeline
from dark_factory.config import load_config
from dark_factory.context import ScriptedScanner
from dark_factory.harness import ScriptedHarness
from dark_factory.intake.schema import FactoryTicket, TicketContext
from dark_factory.llm.client import MeteredLLMClient, build_client

_DEFAULT_TICKET = FactoryTicket(
    ticket_id="org/project-a#1",
    title="Implement JWT Auth Refresh Tokens",
    description="As a user I want tokens to rotate on expiration so sessions stay secure.",
    acceptance_criteria=("Tokens must rotate on expiration", "Must pass auth.test.js"),
    context=TicketContext(repository="org/project-a"),
)


def _ticket(**overrides: object) -> FactoryTicket:
    return dataclasses.replace(_DEFAULT_TICKET, **overrides)  # type: ignore[arg-type]


def _pipeline(*, config_overrides=None, passes_on_attempt=3) -> DarkFactoryPipeline:
    """Build a pipeline wired for offline, deterministic tests.

    Uses the mock LLM provider, a scripted test harness (passes on the Nth
    build-test-fix attempt), and a scripted context scanner (no real repo
    checkout needed).
    """
    cfg = load_config(overrides=config_overrides or {})

    def deps_factory(agent_name, tracker):
        spec = cfg.resolve_llm(agent_name)
        metered = MeteredLLMClient(build_client(spec), tracker, agent_name=agent_name)
        return AgentDeps(
            llm=metered,
            config=cfg,
            harness=ScriptedHarness(passes_on_attempt=passes_on_attempt),
            scanner=ScriptedScanner(),
        )

    return DarkFactoryPipeline(cfg, deps_factory=deps_factory)


def test_pipeline_happy_path_passes_all_phases():
    pipeline = _pipeline(
        config_overrides={"budget": {"max_usd": 3.00}, "loop": {"max_iterations": 5}}
    )
    result = pipeline.run(_ticket())

    assert result.final_status == AgentStatus.PASS
    assert "spec-validated" in result.labels
    assert [p.agent_name for p in result.phase_results] == [
        "validator",
        "planner",
        "developer",
        "reviewer",
    ]
    assert result.budget_summary["calls_recorded"] > 0


def test_pipeline_fails_fast_on_incomplete_spec():
    incomplete = _ticket(description="too short", acceptance_criteria=())
    pipeline = _pipeline()
    result = pipeline.run(incomplete)

    assert result.final_status == AgentStatus.FAIL
    assert "needs-specification" in result.labels
    assert len(result.phase_results) == 1  # stopped after validator
    assert result.budget_summary["spent_usd"] == 0  # structural fail costs $0


def test_pipeline_flags_factory_blocked_when_loop_cap_hit():
    # Cap iterations below the harness's pass-on-3rd-attempt threshold.
    pipeline = _pipeline(config_overrides={"loop": {"max_iterations": 2}}, passes_on_attempt=3)
    result = pipeline.run(_ticket())

    assert result.final_status == AgentStatus.BLOCKED
    assert "factory-blocked" in result.labels
    assert any("exceeded" in c for c in result.comments)


def test_pipeline_flags_factory_blocked_when_budget_exhausted():
    # The mock provider is free by default; override its price so a real
    # ceiling actually bites, then set the ceiling to zero.
    pipeline = _pipeline(
        config_overrides={
            "budget": {"max_usd": 0.0},
            "pricing": {
                "overrides": {
                    "mock:mock-default": {"input_per_mtok_usd": 1.0, "output_per_mtok_usd": 5.0}
                }
            },
        }
    )
    result = pipeline.run(_ticket())

    assert result.final_status == AgentStatus.BLOCKED
    assert "factory-blocked" in result.labels
