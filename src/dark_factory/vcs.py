"""Git commit/push + GitHub PR creation -- the Reviewer agent's "shipping" step.

The only module that shells out to `git` or calls GitHub's REST API. Every
entry point is a soft no-op, never a hard failure, when there's no git
checkout or no `GITHUB_TOKEN`: a local `dark-factory run` must never require
real GitHub credentials just to validate the pipeline. PR creation is over
stdlib `urllib` (no SDK), matching `llm/transport.py`'s own philosophy, but
kept separate since that module's exceptions are LLM-error-typed.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BOT_NAME = "dark-factory[bot]"
_BOT_EMAIL = "dark-factory[bot]@users.noreply.github.com"
_GIT_TIMEOUT_SECONDS = 30
_API_TIMEOUT_SECONDS = 30


class GitOpsError(RuntimeError):
    """Raised by PR creation; `ship()` catches it and reports the reason."""


@dataclass(frozen=True, slots=True)
class ShipResult:
    """The outcome of one `ship()` call."""

    created: bool
    branch: str
    pr_url: str | None = None
    reason: str | None = None


PrCreator = Callable[..., str]


def git_diff(root: Path) -> str:
    """Return a diff of every change (including new untracked files) in `root`.

    Stages everything first (`git add -A`) -- plain `git diff` never shows
    brand-new untracked files at all, only modifications to already-tracked
    ones, which would silently miss the developer agent's common case of
    writing an entirely new file. Never raises: not being a git checkout, or
    any subprocess failure, is treated the same as "no diff to show".
    """
    if not (root / ".git").exists():
        return ""
    try:
        _run_git(["add", "-A"], root)
        proc = _run_git(["diff", "--cached"], root)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def ship(
    root: Path,
    *,
    branch: str,
    base: str,
    title: str,
    body: str,
    repo: str,
    pr_creator: PrCreator | None = None,
) -> ShipResult:
    """Commit `root`'s working tree to `branch`, push it, and open a PR.

    Reads `GITHUB_TOKEN` from the environment itself, matching how
    `AnthropicClient` reads its own key. Never raises -- any failure (no git
    checkout, no token, a git command failing, the PR API rejecting the
    request) is reported via `ShipResult.reason` instead.
    """
    if not (root / ".git").exists():
        return ShipResult(created=False, branch=branch, reason="not a git checkout")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return ShipResult(created=False, branch=branch, reason="GITHUB_TOKEN not set")

    try:
        checkout = _run_git(["checkout", "-B", branch], root)
        if checkout.returncode != 0:
            return ShipResult(
                created=False,
                branch=branch,
                reason=f"git checkout failed: {checkout.stderr.strip()}",
            )

        _run_git(["add", "-A"], root)
        staged = _run_git(["diff", "--cached", "--quiet"], root)
        if staged.returncode == 0:
            return ShipResult(created=False, branch=branch, reason="nothing to commit")

        commit = _run_git(
            [
                "-c",
                f"user.name={_BOT_NAME}",
                "-c",
                f"user.email={_BOT_EMAIL}",
                "commit",
                "-m",
                title,
            ],
            root,
        )
        if commit.returncode != 0:
            return ShipResult(
                created=False, branch=branch, reason=f"git commit failed: {commit.stderr.strip()}"
            )

        # Force is intentional and safe: this branch name is exclusively
        # bot-owned by naming convention, and every run fully regenerates
        # file contents from scratch rather than layering incremental diffs.
        push = _run_git(["push", "--force", "origin", branch], root)
        if push.returncode != 0:
            return ShipResult(
                created=False, branch=branch, reason=f"git push failed: {push.stderr.strip()}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ShipResult(created=False, branch=branch, reason=f"git operation failed: {exc}")

    creator = pr_creator or _default_pr_creator
    try:
        pr_url = creator(repo=repo, branch=branch, base=base, title=title, body=body, token=token)
    except GitOpsError as exc:
        return ShipResult(created=False, branch=branch, reason=str(exc))

    return ShipResult(created=True, branch=branch, pr_url=pr_url)


def _run_git(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
    )


def _http_json(
    method: str, url: str, token: str, json_body: dict[str, Any] | None = None
) -> tuple[int, Any]:
    body = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_API_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {"message": payload}
        return exc.code, parsed
    except urllib.error.URLError as exc:
        raise GitOpsError(f"network error calling the GitHub API: {exc}") from None


def _default_pr_creator(
    *, repo: str, branch: str, base: str, title: str, body: str, token: str
) -> str:
    """Open a PR via GitHub's REST API, reusing an existing one on a 422."""
    status, data = _http_json(
        "POST",
        f"https://api.github.com/repos/{repo}/pulls",
        token,
        {"title": title, "head": branch, "base": base, "body": body},
    )
    if status == 201:
        return str(data["html_url"])
    if status == 422:
        owner = repo.split("/", 1)[0]
        status2, existing = _http_json(
            "GET",
            f"https://api.github.com/repos/{repo}/pulls?head={owner}:{branch}&state=open",
            token,
        )
        if status2 == 200 and existing:
            return str(existing[0]["html_url"])
    raise GitOpsError(f"GitHub API error creating PR for {repo}@{branch}: HTTP {status} {data}")
