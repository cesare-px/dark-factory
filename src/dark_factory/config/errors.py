"""Actionable configuration validation errors.

The loader accumulates every problem it finds rather than raising on the
first, so one CI run surfaces the whole list instead of a fix-one-rerun
loop -- this config file gets copy-pasted across many downstream repos.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

# Values that look like a live secret. This file gets committed to git and
# copy-pasted across repos, so a tripwire here is cheap and high value.
_SECRET_LOOKALIKE_PREFIXES = ("sk-", "AKIA", "ghp_", "gho_", "shpat_", "xox")


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """One validation problem found while building a `FactoryConfig`."""

    path: str
    message: str
    hint: str | None = None

    def render(self) -> str:
        """Render this issue as an indented, human-readable line (or two)."""
        lines = [f"  [{self.path}] {self.message}"]
        if self.hint:
            lines.append(f"      {self.hint}")
        return "\n".join(lines)


class ConfigError(ValueError):
    """All configuration problems found while loading, raised together."""

    def __init__(self, issues: list[ConfigIssue], *, source: str) -> None:
        """Build the error message from every accumulated `ConfigIssue`."""
        self.issues = issues
        self.source = source
        header = f"{source}: {len(issues)} configuration error{'s' if len(issues) != 1 else ''}\n"
        super().__init__(header + "\n".join(issue.render() for issue in issues))


def suggest(value: str, choices: list[str]) -> str | None:
    """Return the closest match to `value` in `choices`, if any is close enough."""
    matches = difflib.get_close_matches(value, choices, n=1)
    return matches[0] if matches else None


def looks_like_secret(value: object) -> bool:
    """Return whether `value` looks like a live API key or token."""
    return isinstance(value, str) and value.startswith(_SECRET_LOOKALIKE_PREFIXES)
