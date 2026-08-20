"""Unified factory protocol schema.

This is the pivot-ready interface described in the blueprint (§4.A): every
upstream source (GitHub Issues today, Linear/Notion tomorrow) must be mapped
into this shape before any agent ever sees it. Agent code should never touch
a raw webhook payload directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FACTORY_PROTOCOL_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class TicketContext:
    """Which repository and branch a ticket targets."""

    repository: str
    branch_target: str = "main"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"repository": self.repository, "branch_target": self.branch_target}


@dataclass(frozen=True, slots=True)
class FactoryTicket:
    """The canonical, sanitized unit of work handed to the agent pipeline."""

    ticket_id: str
    title: str
    description: str
    acceptance_criteria: tuple[str, ...]
    context: TicketContext
    factory_protocol_version: str = FACTORY_PROTOCOL_VERSION
    source_event_id: str | None = None
    sanitization_flags: tuple[str, ...] = field(default_factory=tuple)
    # Checkbox labels checked in the issue body's permissions section (see
    # intake.parser). Raw and unauthorized -- whoever consumes this (the
    # Developer agent) must still confirm `sender_login` actually has write
    # access to the repo before honoring any of these; a checked box is a
    # *request*, never itself a grant.
    requested_permissions: tuple[str, ...] = field(default_factory=tuple)
    sender_login: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "factory_protocol_version": self.factory_protocol_version,
            "ticket_id": self.ticket_id,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": list(self.acceptance_criteria),
            "context": self.context.to_dict(),
            "source_event_id": self.source_event_id,
            "sanitization_flags": list(self.sanitization_flags),
            "requested_permissions": list(self.requested_permissions),
            "sender_login": self.sender_login,
        }

    @property
    def is_suspect(self) -> bool:
        """True if the sanitizer flagged this ticket for human triage."""
        return len(self.sanitization_flags) > 0
