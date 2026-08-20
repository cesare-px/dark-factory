"""Tools the Planner/Developer agents can call within a tool-use loop.

Every path-taking tool shares one containment check: a tool call can never
touch anything outside the ticket's checked-out root, regardless of what an
attacker-controlled ticket asks for -- the same principle
`intake.sanitize` applies to ticket text, applied here to tool execution.
Every handler returns plain text (a tool result's content is always a
string) and raises on failure rather than returning an error sentinel --
`tool_loop.py` catches that and feeds it back to the model as an error
result, so a bad call is a correctable mistake, not a crash.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

MAX_READ_CHARS = 20_000
MAX_COMMAND_OUTPUT_CHARS = 8_000
DEFAULT_EXCLUDE_DIRS = frozenset({"node_modules", ".venv", ".git", "dist", "__pycache__"})
DEFAULT_EXCLUDE_SUFFIXES = (".lock",)


class PathEscapesRootError(ValueError):
    """A tool tried to touch a path outside the sandboxed root."""


def _resolve_safe_path(root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` against `root`; raise if it would escape `root`."""
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise PathEscapesRootError(f"path escapes the project root: {rel_path!r}")
    return target


def _is_excluded(rel_path: str) -> bool:
    """True if any path component or suffix marks `rel_path` as noise.

    Deliberately not fnmatch-pattern-based: a "**/x/**" pattern requires a
    literal "/" immediately before "x", which never matches when "x" sits
    at the checkout root -- a real gap, checking path *components* instead
    doesn't have that failure mode.
    """
    parts = Path(rel_path).parts
    if any(part in DEFAULT_EXCLUDE_DIRS for part in parts):
        return True
    return rel_path.endswith(DEFAULT_EXCLUDE_SUFFIXES)


def read_file(root: Path, path: str) -> str:
    """Return `path`'s contents, relative to `root`."""
    target = _resolve_safe_path(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text


def write_file(root: Path, path: str, content: str) -> str:
    """Write `content` to `path`, relative to `root`, creating parent dirs."""
    target = _resolve_safe_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def edit_file(root: Path, path: str, old_string: str, new_string: str) -> str:
    """Replace exactly one occurrence of `old_string` with `new_string`.

    Raises `ValueError` if `old_string` doesn't appear in `path` exactly
    once -- the same unique-match contract as Claude Code's own Edit tool,
    so an ambiguous or missing match is a correctable error fed back to the
    model, never a silently wrong (or silently no-op) edit.
    """
    target = _resolve_safe_path(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    text = target.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        raise ValueError(f"old_string not found in {path}")
    if count > 1:
        raise ValueError(f"old_string matches {count} locations in {path}; must be unique")
    target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"edited {path}"


def list_files(root: Path, pattern: str = "**/*", max_results: int = 200) -> str:
    """List files under `root` matching a glob `pattern`, one per line."""
    root_resolved = root.resolve()
    matches = []
    for candidate in root_resolved.glob(pattern):
        if not candidate.is_file():
            continue
        rel = str(candidate.relative_to(root_resolved))
        if _is_excluded(rel):
            continue
        matches.append(rel)
    matches.sort()
    truncated = len(matches) > max_results
    matches = matches[:max_results]
    text = "\n".join(matches) if matches else "(no files matched)"
    if truncated:
        text += f"\n... [truncated to {max_results} results]"
    return text


def grep(root: Path, pattern: str, max_results: int = 100) -> str:
    """Search file contents under `root` for `pattern` (plain substring, case-insensitive).

    Uses `git grep` when `root` is a git checkout (fast, respects
    `.gitignore`); falls back to a plain per-file scan otherwise.
    """
    if (root / ".git").exists():
        try:
            proc = subprocess.run(
                # --untracked: the model's own newly-written files won't be
                # `git add`ed yet, and grep should still find them. -e binds
                # `pattern` explicitly as the search expression -- `pattern`
                # is model-controlled (ultimately from ticket text), and
                # without -e a value starting with "-" would be parsed as a
                # git-grep flag instead of the search text.
                ["git", "grep", "-n", "-i", "--untracked", "--fixed-strings", "-e", pattern],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode in (0, 1):  # 1 == no matches, not an error
                lines = proc.stdout.splitlines()[:max_results]
                return "\n".join(lines) if lines else "(no matches)"
        except (OSError, subprocess.TimeoutExpired):
            pass  # fall through to the plain scan below

    root_resolved = root.resolve()
    needle = pattern.lower()
    hits: list[str] = []
    for candidate in root_resolved.rglob("*"):
        if not candidate.is_file():
            continue
        rel = str(candidate.relative_to(root_resolved))
        if _is_excluded(rel):
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                hits.append(f"{rel}:{lineno}:{line}")
                if len(hits) >= max_results:
                    break
        if len(hits) >= max_results:
            break
    return "\n".join(hits) if hits else "(no matches)"


def run_command(root: Path, command: str, allowed_prefixes: tuple[str, ...]) -> str:
    """Run `command` if it matches one of `allowed_prefixes`; reject otherwise.

    Same safety shape as `SubprocessHarness`: no shell (`shlex.split`), a
    hard timeout, and truncated output.
    """
    if not any(command.strip().startswith(prefix) for prefix in allowed_prefixes):
        raise PermissionError(
            f"command not permitted: {command!r} (allowed prefixes: {list(allowed_prefixes)})"
        )
    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[command failed: {exc}]"
    output = f"exit code: {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    if len(output) > MAX_COMMAND_OUTPUT_CHARS:
        output = output[:MAX_COMMAND_OUTPUT_CHARS] + "\n... [truncated]"
    return output
