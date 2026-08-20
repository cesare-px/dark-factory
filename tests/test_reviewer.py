from dark_factory.agents.base import AgentDeps, AgentStatus, RunContext
from dark_factory.agents.reviewer import ReviewerAgent
from dark_factory.config import load_config
from dark_factory.guardrails.budget import TokenBudgetTracker
from dark_factory.intake.schema import FactoryTicket, TicketContext
from dark_factory.llm.client import MeteredLLMClient
from dark_factory.llm.providers.mock import MockClient, MockScript

# Deliberately ordinary prose, no quotes or filenames -- a substring-matching
# heuristic could never approve this; only actually reading the diff (the
# model's job, not ours) can.
_TICKET = FactoryTicket(
    ticket_id="org/repo#1",
    title="Add a health check endpoint",
    description="As an operator I want to check service health.",
    acceptance_criteria=("Returns a 200 status code for valid requests",),
    context=TicketContext(repository="org/repo"),
)

_REAL_DIFF = "diff --git a/app.py b/app.py\n+def health():\n+    return 200\n"


def _deps(tmp_path, response_text: str) -> AgentDeps:
    cfg = load_config()
    spec = cfg.resolve_llm("reviewer")
    tracker = TokenBudgetTracker(max_budget_usd=cfg.budget.max_usd)
    llm = MeteredLLMClient(
        MockClient(spec, script=MockScript(responses={"reviewer": response_text})),
        tracker,
        agent_name="reviewer",
    )
    return AgentDeps(llm=llm, config=cfg, root=tmp_path)


def test_reviewer_approves_when_model_verdict_is_approve(tmp_path):
    agent = ReviewerAgent(_deps(tmp_path, "APPROVE: the health check returns 200 as required."))
    result = agent.run(_TICKET, RunContext(diff_summary=_REAL_DIFF, branch=None))

    assert result.status == AgentStatus.PASS
    assert result.output["approved"] is True


def test_reviewer_rejects_when_model_verdict_is_reject(tmp_path):
    agent = ReviewerAgent(_deps(tmp_path, "REJECT: the diff doesn't add the required endpoint."))
    result = agent.run(_TICKET, RunContext(diff_summary="unrelated diff content", branch=None))

    assert result.status == AgentStatus.FAIL
    assert "REJECT" in (result.comment or "")


def test_reviewer_rejects_ambiguous_verdict_fail_closed(tmp_path):
    # An unclear or missing verdict is not consent to ship.
    agent = ReviewerAgent(_deps(tmp_path, "Looks fine to me I guess"))
    result = agent.run(_TICKET, RunContext(diff_summary=_REAL_DIFF, branch=None))

    assert result.status == AgentStatus.FAIL


def test_reviewer_rejects_dangerous_calls_even_if_model_approves(tmp_path):
    # Defense in depth: the mechanical security prefilter is a hard veto
    # regardless of what the model says. "+eval(user_input)" here is inert
    # diff *text* the prefilter scans for the substring "eval(" -- never
    # executed by this test or by dark-factory.
    agent = ReviewerAgent(_deps(tmp_path, "APPROVE: looks good"))
    result = agent.run(_TICKET, RunContext(diff_summary="+eval(user_input)", branch=None))

    assert result.status == AgentStatus.FAIL
    assert result.output["lint_findings"]
