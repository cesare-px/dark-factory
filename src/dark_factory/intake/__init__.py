"""Webhook intake: sanitization and parsing into the unified ticket schema."""

from dark_factory.intake.parser import ParserError, parse_webhook_event, verify_github_signature
from dark_factory.intake.sanitize import SanitizationReport, sanitize_text
from dark_factory.intake.schema import FactoryTicket, TicketContext

__all__ = [
    "FactoryTicket",
    "ParserError",
    "SanitizationReport",
    "TicketContext",
    "parse_webhook_event",
    "sanitize_text",
    "verify_github_signature",
]
