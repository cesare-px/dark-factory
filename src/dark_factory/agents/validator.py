"""Phase 1: Ticketing & Specification Validation."""

from __future__ import annotations

from dark_factory.agents.base import AgentResult, AgentStatus, BaseAgent, RunContext
from dark_factory.intake.sanitize import wrap_as_untrusted
from dark_factory.intake.schema import FactoryTicket
from dark_factory.labels import LabelKey
from dark_factory.llm.types import LLMRequest, Message

SYSTEM_PROMPT = (
    "You are the Validator agent in an autonomous software factory. Given a "
    "ticket's title, description, and acceptance criteria, reply with a "
    "single line: VALID, or INVALID followed by the reason."
)


class ValidatorAgent(BaseAgent):
    """Checks a ticket against required fields before it enters triage."""

    name = "validator"

    def run(self, ticket: FactoryTicket, ctx: RunContext) -> AgentResult:
        """Reject structurally incomplete tickets, else confirm with the model."""
        cfg = self.deps.config.intake
        missing: list[str] = []
        if len(ticket.description) < cfg.min_description_chars:
            missing.append("description/user-story is too short or missing")
        if not ticket.acceptance_criteria:
            missing.append("acceptance criteria section is missing")
        if ticket.is_suspect:
            missing.append(
                "content flagged by prompt-injection sanitizer: "
                + ", ".join(ticket.sanitization_flags)
            )

        if missing:
            # Structural checks short-circuit before any LLM call, so a
            # malformed ticket costs $0.
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAIL,
                output={"missing_fields": missing},
                labels=(LabelKey.NEEDS_SPECIFICATION,),
                comment=(
                    "This ticket is missing required information before it can "
                    "enter the factory pipeline:\n- " + "\n- ".join(missing)
                ),
            )

        spec = self.deps.config.resolve_llm(self.name)
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            messages=(
                Message(
                    "user",
                    wrap_as_untrusted(
                        f"Title: {ticket.title}\n"
                        f"Description: {ticket.description}\n"
                        f"Acceptance criteria: {list(ticket.acceptance_criteria)}"
                    ),
                ),
            ),
            max_output_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            metadata={"agent": self.name},
        )
        response = self.deps.llm.complete(request)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.PASS,
            output={"ticket": ticket.to_dict(), "model_note": response.text},
            labels=(LabelKey.SPEC_VALIDATED,),
            usage=self.deps.llm.total_usage,
            cost_usd=self.deps.llm.total_cost_usd,
            llm_calls=self.deps.llm.calls,
            model=self.deps.llm.model,
        )
