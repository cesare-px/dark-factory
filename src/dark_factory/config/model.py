"""Frozen dataclasses describing every configurable knob.

No pydantic: this is the only place `dependencies` beyond PyYAML would be
tempting, and plain dataclasses plus config/loader.py's validation keep the
install lean. Every field has a default, so a downstream repo with zero
config gets a fully working (mock-provider) pipeline.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

DEFAULT_SECTION_ALIASES: Mapping[str, tuple[str, ...]] = {
    "description": ("user story", "description", "summary"),
    "acceptance_criteria": ("acceptance criteria",),
}


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """One resolved or sparse LLM configuration (provider, model, credentials)."""

    provider: str = "mock"
    model: str = "mock-default"
    preset: str | None = None  # e.g. "moonshot", "zhipu", "dashscope", "ollama"
    api_key_env: str | None = "LLM_API_KEY"
    base_url: str | None = None
    max_output_tokens: int = 4096
    temperature: float | None = 0.0
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    extra: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Per-agent overrides layered on top of the top-level `llm` settings."""

    use: str | None = None  # named preset key from FactoryConfig.models
    # Sparse override: only the keys explicitly set in config, applied via
    # dataclasses.replace(base, **llm_override) so unset fields fall through
    # to `use` (or the top-level default) instead of being clobbered by
    # LLMSettings()'s own defaults.
    llm_override: Mapping[str, object] = field(default_factory=dict)
    enabled: bool = True
    max_iterations: int | None = None  # developer agent only; falls back to LoopSettings
    extra_instructions: str | None = None


@dataclass(frozen=True, slots=True)
class BudgetSettings:
    """The per-ticket USD spend ceiling and how to react when it is hit."""

    max_usd: float = 3.00
    max_usd_per_agent: Mapping[str, float] = field(default_factory=dict)
    on_exceed: Literal["block", "warn"] = "block"


@dataclass(frozen=True, slots=True)
class LoopSettings:
    """The build-test-fix retry cap and how to react when it is exhausted."""

    max_iterations: int = 5
    on_exhaust: Literal["block", "fail"] = "block"


@dataclass(frozen=True, slots=True)
class TestHarnessSettings:
    """How to run the downstream repo's test command."""

    __test__ = False  # not a pytest test class despite the name

    command: str = "./factory-test.sh"
    setup_command: str | None = None
    working_dir: str = "."
    timeout_seconds: float = 900.0
    max_output_chars: int = 8_000
    allow_shell: bool = False


@dataclass(frozen=True, slots=True)
class ContextSettings:
    """How the Planner agent scans the repo for files relevant to a ticket."""

    strategy: Literal["auto", "none", "glob", "git_grep", "scripted"] = "auto"
    include: tuple[str, ...] = ("**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.go", "**/*.rs")
    exclude: tuple[str, ...] = ("**/node_modules/**", "**/.venv/**", "**/dist/**", "**/*.lock")
    max_files: int = 25


@dataclass(frozen=True, slots=True)
class NamingSettings:
    """Templates for branch and PR names, per `dark_factory.naming`."""

    branch_template: str = "dark-factory/issue-{issue_number}"
    pr_title_template: str = "[factory] {ticket_title}"
    max_branch_length: int = 100


@dataclass(frozen=True, slots=True)
class LabelSettings:
    """The configured label strings agents attach to a ticket."""

    trigger: str | None = None
    spec_validated: str = "spec-validated"
    needs_specification: str = "needs-specification"
    blocked: str = "factory-blocked"
    shipped: str = "factory-shipped"


@dataclass(frozen=True, slots=True)
class IntakeSettings:
    """How the webhook parser extracts structured fields from an issue body."""

    section_aliases: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_SECTION_ALIASES)
    )
    min_description_chars: int = 30


@dataclass(frozen=True, slots=True)
class PricingSettings:
    """Pricing overrides and the policy for models missing from the price book."""

    on_unknown_model: Literal["error", "warn_fallback", "zero"] = "error"
    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedLLM:
    """LLMSettings fully merged from defaults -> models[use] -> agent override."""

    provider: str
    model: str
    preset: str | None
    api_key_env: str | None
    base_url: str | None
    max_output_tokens: int
    temperature: float | None
    timeout_seconds: float
    max_attempts: int
    extra: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FactoryConfig:
    """The complete, effective configuration for one pipeline run."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    models: Mapping[str, LLMSettings] = field(default_factory=dict)
    agents: Mapping[str, AgentSettings] = field(default_factory=dict)
    budget: BudgetSettings = field(default_factory=BudgetSettings)
    loop: LoopSettings = field(default_factory=LoopSettings)
    test_harness: TestHarnessSettings = field(default_factory=TestHarnessSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    naming: NamingSettings = field(default_factory=NamingSettings)
    labels: LabelSettings = field(default_factory=LabelSettings)
    intake: IntakeSettings = field(default_factory=IntakeSettings)
    pricing: PricingSettings = field(default_factory=PricingSettings)
    provenance: Mapping[str, str] = field(default_factory=dict)

    def resolve_llm(self, agent_name: str) -> ResolvedLLM:
        """Merge the top-level, preset, and per-agent LLM settings for one agent.

        Args:
            agent_name: The agent's name, e.g. "validator" or "developer".

        Returns:
            The fully merged, ready-to-use LLM configuration for that agent.
        """
        base = self.llm
        agent_cfg = self.agents.get(agent_name)

        if agent_cfg and agent_cfg.use:
            preset = self.models.get(agent_cfg.use)
            if preset is None:
                raise KeyError(
                    f"agents.{agent_name}.use = {agent_cfg.use!r} does not match any entry "
                    "in `models`"
                )
            base = preset

        if agent_cfg and agent_cfg.llm_override:
            # llm_override's keys are validated against LLMSettings' field
            # names by the config loader before construction, but mypy can't
            # verify a dynamically-keyed **Mapping unpack against the
            # dataclass's per-field types.
            base = dataclasses.replace(base, **agent_cfg.llm_override)  # type: ignore[arg-type]

        return ResolvedLLM(
            provider=base.provider,
            model=base.model,
            preset=base.preset,
            api_key_env=base.api_key_env,
            base_url=base.base_url,
            max_output_tokens=base.max_output_tokens,
            temperature=base.temperature,
            timeout_seconds=base.timeout_seconds,
            max_attempts=base.max_attempts,
            extra=base.extra,
        )

    def label(self, key: str) -> str:
        """Look up the configured label string for a `LabelSettings` field name."""
        return cast(str, getattr(self.labels, key))

    def max_iterations_for(self, agent_name: str) -> int:
        """Return the build-test-fix retry cap for one agent.

        Args:
            agent_name: The agent's name, e.g. "developer".

        Returns:
            The agent's own override if set, else `loop.max_iterations`.
        """
        agent_cfg = self.agents.get(agent_name)
        if agent_cfg and agent_cfg.max_iterations is not None:
            return agent_cfg.max_iterations
        return self.loop.max_iterations
