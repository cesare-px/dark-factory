"""Webhook intake: raw GitHub payload -> unified FactoryTicket.

This is the only code path allowed to touch a raw webhook body. Everything
downstream of parse_webhook_event() operates on the sanitized FactoryTicket
schema, never on raw payload dicts. That boundary is what makes the "swap
GitHub Issues for Linear/Notion" pivot (blueprint §4.A) a one-file change:
only this module and schema.py would need a new sibling implementation.
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from dark_factory.config.model import DEFAULT_SECTION_ALIASES
from dark_factory.intake.sanitize import sanitize_text
from dark_factory.intake.schema import FactoryTicket, TicketContext

SUPPORTED_EVENT_TYPES = frozenset({"issues"})
SUPPORTED_ACTIONS = frozenset({"opened", "edited"})

# Matches "#+ Section Name\n<body>" blocks as emitted by
# .github/ISSUE_TEMPLATE/dark-factory-task.md, at any heading level. The
# leading run of "#" is captured whole (not capped at 3) so a "####" heading
# can't be mis-split into marker "###" + heading text "# Foo".
_SECTION_PATTERN = re.compile(
    r"^#+\s+(?P<heading>.+?)\s*\n(?P<body>.*?)(?=^#+\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)

_CRITERIA_ITEM_PATTERN = re.compile(r"^\s*[-*]\s*(?:\[[ xX]\]\s*)?(?P<item>.+?)\s*$", re.MULTILINE)

# Unlike acceptance criteria (which take any bullet, checked or not), a
# permission is only "requested" if its box is actually checked -- an
# unchecked "- [ ] Package installation" must never grant anything.
_CHECKED_ITEM_PATTERN = re.compile(r"^\s*[-*]\s*\[[xX]\]\s*(?P<item>.+?)\s*$", re.MULTILINE)


class ParserError(ValueError):
    """Raised when a webhook payload cannot be safely parsed."""


def verify_github_signature(payload_body: bytes, signature_header: str | None, secret: str) -> bool:
    """Verify the `X-Hub-Signature-256` header GitHub sends with webhooks.

    Uses a constant-time comparison to avoid timing side-channels. Callers
    MUST reject the request (401/403) if this returns False -- never fall
    back to parsing an unverified payload.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), payload_body, sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)


def _extract_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for match in _SECTION_PATTERN.finditer(body or ""):
        heading = match.group("heading").strip().lower()
        sections[heading] = match.group("body").strip()
    return sections


def _find_section(sections: dict[str, str], *candidates: str) -> str:
    for heading, text in sections.items():
        for candidate in candidates:
            if candidate in heading:
                return text
    return ""


def _extract_checked_items(text: str) -> tuple[str, ...]:
    return tuple(m.group("item").strip() for m in _CHECKED_ITEM_PATTERN.finditer(text))


def _extract_acceptance_criteria(text: str) -> tuple[str, ...]:
    items = [m.group("item").strip() for m in _CRITERIA_ITEM_PATTERN.finditer(text)]
    items = [item for item in items if item]
    if items:
        return tuple(items)
    # Fall back to non-empty lines if the author didn't use a bulleted list.
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def parse_webhook_event(
    payload: dict[str, Any],
    *,
    event_type: str,
    source_event_id: str | None = None,
    section_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> FactoryTicket:
    """Parse a validated (signature-checked) GitHub webhook payload.

    Raises ParserError for structurally invalid payloads. Never raises on
    "bad content" -- content-quality checks belong to the ValidatorAgent
    (blueprint Phase 1), not the parser. The parser's job is strictly
    structural extraction + security sanitization.
    """
    if not isinstance(payload, dict):
        raise ParserError("payload must be a JSON object")

    if event_type not in SUPPORTED_EVENT_TYPES:
        raise ParserError(f"unsupported event_type: {event_type!r}")

    action = payload.get("action")
    if action not in SUPPORTED_ACTIONS:
        raise ParserError(f"unsupported action for issues event: {action!r}")

    issue = payload.get("issue")
    repository = payload.get("repository")
    if not isinstance(issue, dict) or not isinstance(repository, dict):
        raise ParserError("payload missing 'issue' or 'repository' object")

    issue_number = issue.get("number")
    repo_full_name = repository.get("full_name")
    if issue_number is None or not repo_full_name:
        raise ParserError("payload missing issue number or repository full_name")

    aliases = section_aliases or DEFAULT_SECTION_ALIASES

    raw_title = issue.get("title") or ""
    raw_body = issue.get("body") or ""

    title_report = sanitize_text(raw_title, field_name="title")
    sections = _extract_sections(raw_body)

    raw_description = _find_section(sections, *aliases.get("description", ())) or raw_body
    description_report = sanitize_text(raw_description, field_name="description")

    raw_criteria_block = _find_section(sections, *aliases.get("acceptance_criteria", ()))
    criteria_report = sanitize_text(raw_criteria_block, field_name="acceptance_criteria")
    acceptance_criteria = _extract_acceptance_criteria(criteria_report.clean_text)

    branch_target = "main"
    default_branch = repository.get("default_branch")
    if isinstance(default_branch, str) and default_branch:
        branch_target = default_branch

    all_flags = (*title_report.flags, *description_report.flags, *criteria_report.flags)

    # Raw and unauthorized -- only used as a lookup key against the
    # configured command_families catalog, never shown to an LLM prompt, so
    # it doesn't go through sanitize_text like title/description/criteria
    # do. Authorization (does `sender` actually have write access?) happens
    # later, adjacent to where it's spent -- see vcs.sender_has_write_access.
    raw_permissions_block = _find_section(sections, *aliases.get("permissions", ()))
    requested_permissions = _extract_checked_items(raw_permissions_block)

    sender = payload.get("sender")
    sender_login = sender.get("login") if isinstance(sender, dict) else None

    return FactoryTicket(
        ticket_id=f"{repo_full_name}#{issue_number}",
        title=title_report.clean_text or "(untitled)",
        description=description_report.clean_text,
        acceptance_criteria=acceptance_criteria,
        context=TicketContext(repository=repo_full_name, branch_target=branch_target),
        source_event_id=source_event_id,
        sanitization_flags=all_flags,
        requested_permissions=requested_permissions,
        sender_login=sender_login if isinstance(sender_login, str) else None,
    )
