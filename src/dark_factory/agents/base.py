"""Shared agent scaffolding.

Every agent receives an `AgentDeps` bundle built once per pipeline run: a
metered LLM client (already bound to the shared budget tracker and this
agent's resolved model), the effective config, and the test-harness/context
seams. Agents never read config keys, env vars, or price tables themselves
-- that keeps them provider- and config-blind, which is what makes them
testable against a mock client and reusable across any provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from dark_factory.config.model import FactoryConfig
from dark_factory.context import ContextScanner
from dark_factory.harness import TestHarness
from dark_factory.intake.schema import FactoryTicket
from dark_factory.labels import LabelKey
from dark_factory.llm.client import MeteredLLMClient
from dark_factory.llm.types import TokenUsage


class AgentStatus(StrEnum):
    """Terminal status of one agent's run within the pipeline."""

    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    ERROR = "error"  # infrastructure failure (auth, transport) -- not a model judgment


@dataclass(frozen=True, slots=True)
class RunContext:
    """Typed state handed from one pipeline phase to the next."""

    plan_steps: tuple[str, ...] = ()
    affected_files: tuple[str, ...] = ()
    branch: str | None = None
    diff_summary: str = ""
    harness_history: tuple[dict[str, Any], ...] = ()


@dataclass
class AgentResult:
    """The outcome of one agent's run: status, output, labels, cost, and usage."""

    agent_name: str
    status: AgentStatus
    output: dict[str, Any] = field(default_factory=dict)
    labels: tuple[LabelKey, ...] = ()
    comment: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    llm_calls: int = 0
    model: str | None = None


@dataclass(frozen=True, slots=True)
class AgentDeps:
    """The collaborators one agent needs, built once per pipeline run."""

    llm: MeteredLLMClient
    config: FactoryConfig
    harness: TestHarness | None = None
    scanner: ContextScanner | None = None
    root: Path = Path(".")


class BaseAgent:
    """Base class all pipeline agents extend."""

    name: str = "base-agent"

    def __init__(self, deps: AgentDeps) -> None:
        """Store this agent's resolved collaborators."""
        self.deps = deps

    def run(self, ticket: FactoryTicket, ctx: RunContext) -> AgentResult:
        """Run this agent's phase of the pipeline for `ticket`."""
        raise NotImplementedError
