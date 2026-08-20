"""Load a FactoryConfig from a repo's .dark-factory.yml plus env overrides.

Precedence, lowest to highest:
  1. dataclass defaults (config/model.py)
  2. the repo's config file
  3. DARK_FACTORY_* environment variables
  4. explicit `overrides` dict (CLI --set, test kwargs)

Every layer merges as a nested dict before any dataclass is built, so file
and env overrides go through identical validation and produce identical
provenance tracking.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml

from dark_factory.config.env import parse_env_overrides
from dark_factory.config.errors import ConfigError, ConfigIssue, looks_like_secret, suggest
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
    TestHarnessSettings,
    ToolUseSettings,
)

# Raw, untyped config data as parsed from YAML/env/overrides, before any
# dataclass is built from it.
RawConfig = dict[str, Any]
RawMapping = Mapping[str, Any]

DISCOVERY_ORDER = (".dark-factory.yml", ".dark-factory.yaml", ".github/dark-factory.yml")

_DataclassT = TypeVar("_DataclassT")


def find_config_file(root: Path) -> Path | None:
    """Return the first config file found under `root` in discovery order."""
    for candidate in DISCOVERY_ORDER:
        path = root / candidate
        if path.is_file():
            return path
    return None


def _deep_merge(
    base: RawConfig, overlay: RawMapping, source: str, provenance: dict[str, str], prefix: str = ""
) -> None:
    for key, value in overlay.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value, source, provenance, path)
        else:
            base[key] = value
            provenance[path] = source


def _known_fields(dataclass_type: type) -> set[str]:
    return {f.name for f in dataclasses.fields(dataclass_type)}


def _check_unknown_keys(
    data: RawMapping, dataclass_type: type, path: str, issues: list[ConfigIssue]
) -> None:
    known = _known_fields(dataclass_type)
    for key in data:
        if key not in known:
            hint_target = suggest(str(key), sorted(known))
            hint = f"did you mean {hint_target!r}? known keys: {', '.join(sorted(known))}"
            issues.append(
                ConfigIssue(path=f"{path}.{key}", message=f"unknown key {key!r}", hint=hint)
            )


_SCALAR_TYPES = (int, float, bool, str)


def _unwrap_optional(hint: object) -> object:
    args = typing.get_args(hint)
    if typing.get_origin(hint) is typing.Union and type(None) in args:
        remaining = [a for a in args if a is not type(None)]
        return remaining[0] if len(remaining) == 1 else hint
    return hint


def _coerce_scalar(value: object, expected: type, path: str, issues: list[ConfigIssue]) -> object:
    if isinstance(value, expected) and not (expected is float and isinstance(value, bool)):
        return value
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected in (int, float) and isinstance(value, str):
        try:
            return expected(value)
        except ValueError:
            issues.append(
                ConfigIssue(
                    path=path,
                    message=f"expected a number, got a string {value!r}",
                    hint=f"write: {path.rsplit('.', 1)[-1]} = {value}  (without quotes)",
                )
            )
            return value
    if expected in (int, float, bool) and not isinstance(value, expected):
        issues.append(
            ConfigIssue(
                path=path, message=f"expected {expected.__name__}, got {type(value).__name__}"
            )
        )
    return value


def _build_dataclass(
    data: RawMapping | None,
    dataclass_type: type[_DataclassT],
    path: str,
    issues: list[ConfigIssue],
) -> _DataclassT:
    data = data or {}
    if not isinstance(data, Mapping):
        issues.append(
            ConfigIssue(path=path, message=f"expected a mapping, got {type(data).__name__}")
        )
        return dataclass_type()
    _check_unknown_keys(data, dataclass_type, path, issues)
    known = _known_fields(dataclass_type)
    hints = typing.get_type_hints(dataclass_type)
    kwargs = {k: v for k, v in data.items() if k in known}
    for key, value in list(kwargs.items()):
        field_path = f"{path}.{key}"
        if looks_like_secret(value):
            issues.append(
                ConfigIssue(
                    path=field_path,
                    message="value looks like a live secret",
                    hint="never put secrets in this file (it is committed to git) -- "
                    "use an *_env field naming an environment variable instead",
                )
            )
        expected = _unwrap_optional(hints.get(key))
        if expected in _SCALAR_TYPES and value is not None:
            kwargs[key] = _coerce_scalar(value, expected, field_path, issues)
    try:
        return dataclass_type(**kwargs)
    except TypeError as exc:
        issues.append(ConfigIssue(path=path, message=str(exc)))
        return dataclass_type()


def _build_llm(data: RawMapping | None, path: str, issues: list[ConfigIssue]) -> LLMSettings:
    return _build_dataclass(data, LLMSettings, path, issues)


def _build_models(data: RawMapping | None, issues: list[ConfigIssue]) -> dict[str, LLMSettings]:
    data = data or {}
    return {name: _build_llm(value, f"models.{name}", issues) for name, value in data.items()}


def _build_agents(data: RawMapping | None, issues: list[ConfigIssue]) -> dict[str, AgentSettings]:
    data = data or {}
    result: dict[str, AgentSettings] = {}
    for name, value in data.items():
        value = dict(value or {})
        llm_override = value.pop("llm", {}) or {}
        if not isinstance(llm_override, Mapping):
            issues.append(ConfigIssue(path=f"agents.{name}.llm", message="expected a mapping"))
            llm_override = {}
        else:
            unknown = set(llm_override) - _known_fields(LLMSettings)
            for key in unknown:
                issues.append(
                    ConfigIssue(
                        path=f"agents.{name}.llm.{key}",
                        message=f"unknown key {key!r}",
                        hint=f"known keys: {', '.join(sorted(_known_fields(LLMSettings)))}",
                    )
                )
            llm_override = {
                k: v for k, v in llm_override.items() if k in _known_fields(LLMSettings)
            }
        agent_settings = _build_dataclass(value, AgentSettings, f"agents.{name}", issues)
        result[name] = dataclasses.replace(agent_settings, llm_override=llm_override)
    return result


_TOP_LEVEL_SECTIONS = (
    "llm",
    "models",
    "agents",
    "budget",
    "loop",
    "test_harness",
    "context",
    "naming",
    "labels",
    "intake",
    "pricing",
    "tool_use",
)


def _build_config(data: RawMapping, issues: list[ConfigIssue]) -> FactoryConfig:
    for key in data:
        if key not in _TOP_LEVEL_SECTIONS:
            hint_target = suggest(str(key), list(_TOP_LEVEL_SECTIONS))
            hint = f"did you mean {hint_target!r}? known sections: {', '.join(_TOP_LEVEL_SECTIONS)}"
            issues.append(
                ConfigIssue(path=str(key), message=f"unknown top-level section {key!r}", hint=hint)
            )

    return FactoryConfig(
        llm=_build_llm(data.get("llm"), "llm", issues),
        models=_build_models(data.get("models"), issues),
        agents=_build_agents(data.get("agents"), issues),
        budget=_build_dataclass(data.get("budget"), BudgetSettings, "budget", issues),
        loop=_build_dataclass(data.get("loop"), LoopSettings, "loop", issues),
        test_harness=_build_dataclass(
            data.get("test_harness"), TestHarnessSettings, "test_harness", issues
        ),
        context=_build_dataclass(data.get("context"), ContextSettings, "context", issues),
        naming=_build_dataclass(data.get("naming"), NamingSettings, "naming", issues),
        labels=_build_dataclass(data.get("labels"), LabelSettings, "labels", issues),
        intake=_build_dataclass(data.get("intake"), IntakeSettings, "intake", issues),
        pricing=_build_dataclass(data.get("pricing"), PricingSettings, "pricing", issues),
        tool_use=_build_dataclass(data.get("tool_use"), ToolUseSettings, "tool_use", issues),
    )


def load_config(
    *,
    root: Path = Path("."),
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: RawMapping | None = None,
) -> FactoryConfig:
    """Build a `FactoryConfig` from defaults, an optional file, env vars, and overrides.

    Args:
        root: Directory to search for a config file in when `path` is unset.
        path: An explicit config file path, bypassing discovery.
        env: Environment variables to read `DARK_FACTORY_*` overrides from;
            defaults to `os.environ`.
        overrides: Explicit highest-precedence overrides (e.g. CLI `--set`).

    Returns:
        The merged, validated configuration, with per-key provenance recorded.

    Raises:
        ConfigError: If the file or merged overrides fail validation.
    """
    root = Path(root)
    config_path = path if path is not None else find_config_file(root)

    merged: RawConfig = {}
    provenance: dict[str, str] = {}

    if config_path is not None and Path(config_path).is_file():
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ConfigError(
                [ConfigIssue(path="<root>", message="top-level YAML must be a mapping")],
                source=str(config_path),
            )
        _deep_merge(merged, raw, str(config_path), provenance)

    env_data = parse_env_overrides(env if env is not None else os.environ)
    _deep_merge(merged, env_data, "environment", provenance)

    if overrides:
        _deep_merge(merged, overrides, "override", provenance)

    issues: list[ConfigIssue] = []
    config = _build_config(merged, issues)
    if issues:
        raise ConfigError(issues, source=str(config_path) if config_path else "<defaults>")

    return dataclasses.replace(config, provenance=provenance)
