import pytest

from dark_factory.config.model import NamingSettings
from dark_factory.intake.schema import FactoryTicket, TicketContext
from dark_factory.naming import (
    InvalidRefError,
    InvalidTemplateError,
    render_branch,
    render_pr_title,
    slugify,
)


def _ticket(title: str = "Add health check endpoint") -> FactoryTicket:
    return FactoryTicket(
        ticket_id="acme/widgets#42",
        title=title,
        description="d",
        acceptance_criteria=(),
        context=TicketContext(repository="acme/widgets"),
    )


def test_slugify_basic():
    assert slugify("Add health check endpoint") == "add-health-check-endpoint"


def test_render_branch_default_template():
    branch = render_branch(NamingSettings(), _ticket())
    assert branch == "dark-factory/issue-42"


def test_render_branch_with_slug_template():
    cfg = NamingSettings(branch_template="dark-factory/issue-{issue_number}-{ticket_slug}")
    branch = render_branch(cfg, _ticket())
    assert branch == "dark-factory/issue-42-add-health-check-endpoint"


def test_render_pr_title():
    title = render_pr_title(NamingSettings(), _ticket())
    assert title == "[factory] Add health check endpoint"


def test_unknown_placeholder_raises_at_render_time():
    cfg = NamingSettings(branch_template="dark-factory/{nonexistent}")
    with pytest.raises(InvalidTemplateError):
        render_branch(cfg, _ticket())


@pytest.mark.parametrize(
    "malicious_title",
    [
        "../../etc/passwd",
        "feature/../../secrets",
        "; rm -rf /",
    ],
)
def test_attacker_controlled_title_cannot_produce_unsafe_ref(malicious_title):
    cfg = NamingSettings(branch_template="dark-factory/issue-{issue_number}-{ticket_slug}")
    # slugify() strips everything unsafe before it ever reaches the ref
    # validator, so these should render as ordinary safe branch names.
    branch = render_branch(cfg, _ticket(malicious_title))
    assert ".." not in branch
    assert " " not in branch
    assert ";" not in branch


def test_validate_git_ref_rejects_dotdot_directly():
    from dark_factory.naming import validate_git_ref

    with pytest.raises(InvalidRefError):
        validate_git_ref("dark-factory/../escape")


def test_validate_git_ref_truncates_overlong_names():
    from dark_factory.naming import validate_git_ref

    ref = validate_git_ref("a" * 200, max_length=20)
    assert len(ref) <= 20
