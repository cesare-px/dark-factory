"""Defensive sanitization for untrusted, user-authored text.

Any text in a GitHub issue is attacker-controlled: this module is the last
line of defense before that text is interpolated into an LLM prompt. We do
not try to be a perfect prompt-injection classifier (impossible) — instead we
normalize away common obfuscation tricks, flag known attack patterns for
human/agent triage, and hard-cap length to prevent context-stuffing / token
exhaustion attacks.

Flagged text is NOT silently dropped: it is neutralized (defanged) and kept,
with the flags surfaced on the ticket so downstream agents/humans can decide
whether to proceed. Silent deletion would let an attacker probe the filter;
surfacing flags keeps the pipeline auditable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_FIELD_LENGTH = 20_000

# Zero-width / invisible characters used to smuggle text past naive filters.
_ZERO_WIDTH_CHARS = re.compile("[​‌‍⁠﻿᠎⁢⁣⁤]")

# Control characters other than \n \r \t.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Patterns commonly used to hijack an agent's instructions. Case-insensitive.
_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE
    ),
    "role_hijack": re.compile(
        r"\b(you are now|act as|pretend to be|new system prompt|"
        r"disregard your (rules|guidelines))\b",
        re.IGNORECASE,
    ),
    "fake_role_marker": re.compile(
        r"^\s*(system|assistant|developer)\s*:", re.IGNORECASE | re.MULTILINE
    ),
    "delimiter_breakout": re.compile(r"```+\s*(system|end of (ticket|context))", re.IGNORECASE),
    "secret_exfiltration": re.compile(
        r"\b(print|reveal|dump|exfiltrate)\b.{0,30}"
        r"\b(api key|secret|token|credentials|env(ironment)? var)",
        re.IGNORECASE,
    ),
    "tool_command_injection": re.compile(
        r"\b(run|execute)\s+(this\s+)?(shell|bash|command)\b.{0,40}(curl|wget|rm -rf|;|\|)",
        re.IGNORECASE,
    ),
}


@dataclass(slots=True)
class SanitizationReport:
    """Result of sanitizing one untrusted text field."""

    clean_text: str
    flags: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def is_suspect(self) -> bool:
        """Return whether any injection pattern or truncation was flagged."""
        return bool(self.flags)


def sanitize_text(raw: str | None, *, field_name: str = "field") -> SanitizationReport:
    """Normalize, defang, and flag a single untrusted text field."""
    if raw is None:
        return SanitizationReport(clean_text="")

    text = unicodedata.normalize("NFKC", raw)
    text = _ZERO_WIDTH_CHARS.sub("", text)
    text = _CONTROL_CHARS.sub("", text)

    flags: list[str] = []
    for label, pattern in _INJECTION_PATTERNS.items():
        if pattern.search(text):
            flags.append(f"{field_name}:{label}")

            # Defang rather than delete: break the pattern's ability to be
            # parsed as an instruction while keeping the text auditable.
            def _tag_match(match: re.Match[str], label: str = label) -> str:
                return f"[[flagged:{label}]] {match.group(0)}"

            text = pattern.sub(_tag_match, text)

    truncated = False
    if len(text) > MAX_FIELD_LENGTH:
        text = text[:MAX_FIELD_LENGTH]
        truncated = True
        flags.append(f"{field_name}:truncated")

    return SanitizationReport(clean_text=text.strip(), flags=flags, truncated=truncated)


def wrap_as_untrusted(text: str, *, label: str = "USER_SUPPLIED_TICKET_CONTENT") -> str:
    """Wrap sanitized text in explicit untrusted-data delimiters for prompts.

    Agents must render ticket content through this wrapper (never string-
    concatenate it directly into a system/instruction prompt) so the model
    can be told, structurally, that the enclosed text is data, not commands.
    """
    return f"<{label}>\n{text}\n</{label}>"
