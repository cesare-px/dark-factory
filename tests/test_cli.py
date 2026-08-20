import json
from pathlib import Path

import pytest

from dark_factory.cli import main

ISSUE_EVENT = {
    "action": "opened",
    "issue": {
        "number": 7,
        "title": "Add health check endpoint",
        "body": (
            "## User Story\nAs an operator I want a /healthz endpoint so uptime "
            "monitors can check the service.\n\n"
            "## Acceptance Criteria\n- GET /healthz returns 200\n"
        ),
    },
    "repository": {"full_name": "acme/widgets", "default_branch": "main"},
}


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(ISSUE_EVENT), encoding="utf-8")
    return tmp_path


def test_run_dry_run_exits_and_writes_report(repo: Path, capsys):
    # --dry-run's mock provider produces no real content, so the reviewer
    # correctly rejects it (nothing to actually approve) rather than
    # rubber-stamping -- this smoke-tests that the pipeline's plumbing
    # completes all four phases without crashing and without spending
    # anything, not that mock content passes review.
    output_path = repo / "run.json"
    exit_code = main(
        [
            "--root",
            str(repo),
            "run",
            "--event-path",
            str(repo / "event.json"),
            "--dry-run",
            "--output",
            str(output_path),
            "--set",
            'test_harness.command=python3 -c "pass"',
        ]
    )
    assert exit_code == 30  # reviewer rejection; see report.exit_code_for
    report = json.loads(output_path.read_text())
    assert report["final_status"] == "fail"
    assert report["ticket_id"] == "acme/widgets#7"
    assert report["budget"]["spent_usd"] == 0.0  # dry-run forces the free mock provider


def test_run_spec_fail_exits_10(repo: Path):
    incomplete_event = dict(ISSUE_EVENT)
    incomplete_event["issue"] = {**ISSUE_EVENT["issue"], "body": "too short"}  # type: ignore[dict-item]
    event_path = repo / "incomplete.json"
    event_path.write_text(json.dumps(incomplete_event), encoding="utf-8")

    exit_code = main(["--root", str(repo), "run", "--event-path", str(event_path), "--dry-run"])
    assert exit_code == 10


def test_run_bad_config_exits_2(repo: Path):
    exit_code = main(
        [
            "--root",
            str(repo),
            "run",
            "--event-path",
            str(repo / "event.json"),
            "--set",
            "llm.providerrrr=anthropic",
        ]
    )
    assert exit_code == 2


def test_run_missing_event_file_exits_2(repo: Path):
    exit_code = main(
        ["--root", str(repo), "run", "--event-path", str(repo / "does-not-exist.json"), "--dry-run"]
    )
    assert exit_code == 2


def test_doctor_reports_missing_credentials(repo: Path, capsys, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    exit_code = main(
        [
            "--root",
            str(repo),
            "doctor",
            "--set",
            "llm.provider=anthropic",
            "--set",
            "llm.model=claude-sonnet-5",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "LLM_API_KEY" in out


def test_init_scaffolds_config_and_workflow(tmp_path: Path):
    exit_code = main(["--root", str(tmp_path), "init", "--provider", "openai_compatible"])
    assert exit_code == 0
    assert (tmp_path / ".dark-factory.yml").exists()
    assert (tmp_path / ".github" / "workflows" / "caller.yml").exists()


def test_init_refuses_to_overwrite_without_force(tmp_path: Path):
    main(["--root", str(tmp_path), "init"])
    exit_code = main(["--root", str(tmp_path), "init"])
    assert exit_code == 1


def test_prices_lists_anthropic_models(capsys):
    exit_code = main(["prices", "--provider", "anthropic"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "anthropic:claude-sonnet-5" in out
