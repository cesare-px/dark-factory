"""Branch/PR naming from configurable templates, with git-ref validation.

Ticket titles are attacker-controlled (they come from an issue body) and
branch names reach `git` argv, so slug sanitization here is a security
control, not cosmetics.
"""

from __future__ import annotations

import re

from dark_factory.config.model import NamingSettings
from dark_factory.intake.schema import FactoryTicket

_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")
_INVALID_REF_CHARS = re.compile(r"[\x00-\x1f\x7f~^:?*\[\\ ]")


class InvalidRefError(ValueError):
    """A rendered branch/ref name failed git-ref safety validation."""


class InvalidTemplateError(ValueError):
    """A naming template referenced a placeholder that doesn't exist."""


def slugify(text: str, *, max_length: int = 40) -> str:
    """Lowercase, hyphenate, and truncate `text` into a safe slug."""
    slug = _SLUG_UNSAFE.sub("-", text.lower()).strip("-")
    return slug[:max_length].strip("-") or "ticket"


def _ticket_placeholders(ticket: FactoryTicket) -> dict[str, str]:
    repo = ticket.context.repository
    repo_owner, _, repo_name = repo.partition("/")
    issue_number = ticket.ticket_id.rsplit("#", 1)[-1]
    return {
        "issue_number": issue_number,
        "repo": repo,
        "repo_name": repo_name or repo,
        "repo_owner": repo_owner,
        "ticket_slug": slugify(ticket.title),
        "ticket_title": ticket.title,
    }


def render(template: str, ticket: FactoryTicket, **extra: str) -> str:
    """Fill a naming template's placeholders from `ticket` and `extra`.

    Raises:
        InvalidTemplateError: If the template references an unknown placeholder.
    """
    placeholders = {**_ticket_placeholders(ticket), **extra}
    try:
        return template.format(**placeholders)
    except KeyError as exc:
        known = ", ".join(sorted(placeholders))
        raise InvalidTemplateError(
            f"unknown placeholder {exc} in template {template!r}. known: {known}"
        ) from None


def validate_git_ref(ref: str, *, max_length: int = 100) -> str:
    """Reject unsafe git ref characters/patterns and truncate to `max_length`.

    Raises:
        InvalidRefError: If `ref` contains path traversal, control characters,
            or other characters unsafe to pass to `git` as a ref name.
    """
    if not ref or ref.startswith("/") or ref.endswith("/") or ".." in ref or ref.endswith(".lock"):
        raise InvalidRefError(f"unsafe git ref: {ref!r}")
    if "@{" in ref or _INVALID_REF_CHARS.search(ref):
        raise InvalidRefError(f"unsafe git ref: {ref!r}")
    if len(ref) > max_length:
        ref = ref[:max_length].rstrip("/-")
    return ref


def render_branch(cfg: NamingSettings, ticket: FactoryTicket) -> str:
    """Render and validate the Developer agent's branch name for `ticket`."""
    branch = render(cfg.branch_template, ticket)
    return validate_git_ref(branch, max_length=cfg.max_branch_length)


def render_pr_title(cfg: NamingSettings, ticket: FactoryTicket) -> str:
    """Render the pull request title for `ticket`."""
    return render(cfg.pr_title_template, ticket)
