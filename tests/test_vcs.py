import subprocess
from pathlib import Path

import pytest

from dark_factory import vcs


def _init_repo_with_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)

    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "remote", "add", "origin", str(remote)], check=True)
    return work


def test_git_diff_captures_new_untracked_file(tmp_path: Path):
    work = _init_repo_with_remote(tmp_path)
    (work / "hello.txt").write_text("Hello, World!")

    diff = vcs.git_diff(work)

    assert "hello.txt" in diff
    assert "Hello, World!" in diff


def test_git_diff_returns_empty_for_non_git_dir(tmp_path: Path):
    assert vcs.git_diff(tmp_path) == ""


def test_ship_commits_pushes_and_opens_pr(tmp_path: Path, monkeypatch):
    work = _init_repo_with_remote(tmp_path)
    (work / "hello.txt").write_text("Hello, World!")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    calls = []

    def fake_pr_creator(**kwargs):
        calls.append(kwargs)
        return "https://github.com/org/repo/pull/1"

    result = vcs.ship(
        work,
        branch="dark-factory/issue-1",
        base="main",
        title="[factory] Add hello",
        body="Closes #1",
        repo="org/repo",
        pr_creator=fake_pr_creator,
    )

    assert result.created is True
    assert result.pr_url == "https://github.com/org/repo/pull/1"
    assert calls[0]["branch"] == "dark-factory/issue-1"
    assert calls[0]["token"] == "test-token"

    log = subprocess.run(
        ["git", "-C", str(tmp_path / "remote.git"), "log", "dark-factory/issue-1", "--oneline"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert log.stdout.strip() != ""


def test_ship_skips_when_no_github_token(tmp_path: Path, monkeypatch):
    work = _init_repo_with_remote(tmp_path)
    (work / "hello.txt").write_text("hi")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result = vcs.ship(work, branch="b", base="main", title="t", body="b", repo="org/repo")

    assert result.created is False
    assert result.reason == "GITHUB_TOKEN not set"


def test_ship_skips_when_not_a_git_checkout(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    result = vcs.ship(tmp_path, branch="b", base="main", title="t", body="b", repo="org/repo")

    assert result.created is False
    assert result.reason == "not a git checkout"


def test_ship_reports_nothing_to_commit_on_second_identical_run(tmp_path: Path, monkeypatch):
    work = _init_repo_with_remote(tmp_path)
    (work / "hello.txt").write_text("Hello, World!")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    creator = lambda **kw: "https://github.com/org/repo/pull/1"  # noqa: E731

    first = vcs.ship(
        work,
        branch="dark-factory/issue-1",
        base="main",
        title="t",
        body="b",
        repo="org/repo",
        pr_creator=creator,
    )
    assert first.created is True

    second = vcs.ship(
        work,
        branch="dark-factory/issue-1",
        base="main",
        title="t",
        body="b",
        repo="org/repo",
        pr_creator=creator,
    )
    assert second.created is False
    assert second.reason == "nothing to commit"


def test_default_pr_creator_reuses_existing_pr_on_422(monkeypatch):
    calls = []

    def fake_http_json(method, url, token, json_body=None):
        calls.append(method)
        if method == "POST":
            return 422, {"message": "A pull request already exists for org:branch."}
        return 200, [{"html_url": "https://github.com/org/repo/pull/5"}]

    monkeypatch.setattr(vcs, "_http_json", fake_http_json)

    url = vcs._default_pr_creator(
        repo="org/repo",
        branch="dark-factory/issue-1",
        base="main",
        title="t",
        body="b",
        token="tok",
    )

    assert url == "https://github.com/org/repo/pull/5"
    assert calls == ["POST", "GET"]


def test_default_pr_creator_raises_on_unexpected_status(monkeypatch):
    monkeypatch.setattr(vcs, "_http_json", lambda *a, **k: (500, {"message": "boom"}))

    with pytest.raises(vcs.GitOpsError):
        vcs._default_pr_creator(
            repo="org/repo", branch="b", base="main", title="t", body="b", token="tok"
        )


def test_sender_has_write_access_true_for_write_permission(monkeypatch):
    monkeypatch.setattr(vcs, "_http_json", lambda *a, **k: (200, {"permission": "write"}))
    assert vcs.sender_has_write_access("org/repo", "alice", "tok") is True


def test_sender_has_write_access_false_for_read_permission(monkeypatch):
    monkeypatch.setattr(vcs, "_http_json", lambda *a, **k: (200, {"permission": "read"}))
    assert vcs.sender_has_write_access("org/repo", "alice", "tok") is False


def test_sender_has_write_access_false_on_404(monkeypatch):
    monkeypatch.setattr(vcs, "_http_json", lambda *a, **k: (404, {"message": "Not Found"}))
    assert vcs.sender_has_write_access("org/repo", "alice", "tok") is False


def test_sender_has_write_access_false_on_network_error(monkeypatch):
    def raise_error(*a, **k):
        raise vcs.GitOpsError("network down")

    monkeypatch.setattr(vcs, "_http_json", raise_error)
    assert vcs.sender_has_write_access("org/repo", "alice", "tok") is False


def test_sender_has_write_access_false_for_empty_sender():
    assert vcs.sender_has_write_access("org/repo", "", "tok") is False
