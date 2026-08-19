"""FactoryConfig dataclasses and the loader that builds them."""

from dark_factory.config.errors import ConfigError, ConfigIssue
from dark_factory.config.loader import find_config_file, load_config
from dark_factory.config.model import (
    AgentSettings,
    BudgetSettings,
    ContextSettings,
    FactoryConfig,
    IntakeSettings,
    LabelSettings,
    LLMSettings,
    LoopSettings,
    NamingSettings,
    PricingSettings,
    ResolvedLLM,
    TestHarnessSettings,
)

__all__ = [
    "AgentSettings",
    "BudgetSettings",
    "ConfigError",
    "ConfigIssue",
    "ContextSettings",
    "FactoryConfig",
    "IntakeSettings",
    "LLMSettings",
    "LabelSettings",
    "LoopSettings",
    "NamingSettings",
    "PricingSettings",
    "ResolvedLLM",
    "TestHarnessSettings",
    "find_config_file",
    "load_config",
]
