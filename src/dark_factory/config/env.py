"""Environment-variable overrides for FactoryConfig.

Shaped as the same nested dict the YAML loader produces, so both sources
merge through one code path. `DARK_FACTORY_` prefix, `__` as the nesting
separator:

    DARK_FACTORY_BUDGET__MAX_USD=1.50   -> {"budget": {"max_usd": "1.50"}}
    DARK_FACTORY_LLM__PROVIDER=openai   -> {"llm": {"provider": "openai"}}

Plus short aliases for the knobs CI touches most often.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PREFIX = "DARK_FACTORY_"
NESTING_SEPARATOR = "__"

_ALIASES: dict[str, tuple[str, ...]] = {
    "DARK_FACTORY_MODEL": ("llm", "model"),
    "DARK_FACTORY_PROVIDER": ("llm", "provider"),
    "DARK_FACTORY_BUDGET_MAX_USD": ("budget", "max_usd"),
    "DARK_FACTORY_MAX_ITERATIONS": ("loop", "max_iterations"),
}

_NUMERIC_LEAVES = {
    ("budget", "max_usd"),
    ("loop", "max_iterations"),
    ("llm", "max_output_tokens"),
    ("llm", "timeout_seconds"),
    ("llm", "max_attempts"),
    ("llm", "temperature"),
}


def _coerce(path: tuple[str, ...], value: str) -> object:
    if path in _NUMERIC_LEAVES:
        try:
            return int(value) if "." not in value else float(value)
        except ValueError:
            return value
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _set_path(target: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    node = target
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def parse_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Convert `DARK_FACTORY_*` environment variables into a nested override dict."""
    result: dict[str, Any] = {}
    for name, alias_path in _ALIASES.items():
        if name in env:
            _set_path(result, alias_path, _coerce(alias_path, env[name]))

    for name, raw_value in env.items():
        if not name.startswith(PREFIX) or name in _ALIASES:
            continue
        remainder = name[len(PREFIX) :]
        if not remainder:
            continue
        path = tuple(part.lower() for part in remainder.split(NESTING_SEPARATOR))
        _set_path(result, path, _coerce(path, raw_value))

    return result
