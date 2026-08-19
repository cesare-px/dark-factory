"""Repo context scanning: which files the Planner/Developer should look at.

`strategy: auto` picks git-grep when the target checkout is a git repo,
falling back to a plain glob otherwise. `ScriptedScanner` is the offline
stand-in used by tests, replacing the old `_mock_ast_context_scan`.
"""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dark_factory.config.model import ContextSettings
from dark_factory.intake.schema import FactoryTicket


@dataclass(frozen=True, slots=True)
class ContextScan:
    """The files a scanner selected as relevant to a ticket."""

    files: tuple[str, ...]
    strategy: str
    truncated: bool


class ContextScanner(Protocol):
    """Selects the repo files most relevant to implementing a ticket."""

    def scan(self, ticket: FactoryTicket, *, root: Path) -> ContextScan:
        """Return the files this scanner considers relevant to `ticket`."""
        ...


def _matches_include_exclude(path: Path, root: Path, cfg: ContextSettings) -> bool:
    rel = str(path.relative_to(root))
    if any(fnmatch.fnmatch(rel, pattern) for pattern in cfg.exclude):
        return False
    return any(fnmatch.fnmatch(rel, pattern) for pattern in cfg.include)


class GlobScanner:
    """Selects every file under `root` matching the configured include globs."""

    def __init__(self, cfg: ContextSettings) -> None:
        """Store the include/exclude/max-files settings to scan with."""
        self._cfg = cfg

    def scan(self, ticket: FactoryTicket, *, root: Path) -> ContextScan:
        """Return every matching file under `root`, sorted and capped."""
        matches = [
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and _matches_include_exclude(p, root, self._cfg)
        ]
        matches.sort()
        truncated = len(matches) > self._cfg.max_files
        return ContextScan(
            files=tuple(matches[: self._cfg.max_files]), strategy="glob", truncated=truncated
        )


class GitGrepScanner:
    """Ranks files by hits on keywords pulled from the ticket title/criteria."""

    def __init__(self, cfg: ContextSettings) -> None:
        """Store the include/exclude/max-files settings to scan with."""
        self._cfg = cfg

    def _keywords(self, ticket: FactoryTicket) -> list[str]:
        words = ticket.title.split()
        for criterion in ticket.acceptance_criteria:
            words.extend(criterion.split())
        # Keep short, alphabetic-ish tokens; git grep on stopwords is noise.
        return sorted({w.strip(".,:;()[]\"'").lower() for w in words if len(w) > 3})[:10]

    def scan(self, ticket: FactoryTicket, *, root: Path) -> ContextScan:
        """Return files matching the ticket's keywords, ranked by hit count."""
        keywords = self._keywords(ticket)
        if not keywords:
            return ContextScan(files=(), strategy="git_grep", truncated=False)

        hit_counts: dict[str, int] = {}
        for keyword in keywords:
            try:
                proc = subprocess.run(
                    ["git", "grep", "-l", "-i", "--fixed-strings", keyword],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
            for line in proc.stdout.splitlines():
                path = Path(root / line)
                if _matches_include_exclude(path, root, self._cfg):
                    hit_counts[line] = hit_counts.get(line, 0) + 1

        ranked = sorted(hit_counts, key=lambda f: hit_counts[f], reverse=True)
        truncated = len(ranked) > self._cfg.max_files
        return ContextScan(
            files=tuple(ranked[: self._cfg.max_files]), strategy="git_grep", truncated=truncated
        )


class ScriptedScanner:
    """Deterministic offline scanner for tests and dry-runs."""

    def __init__(self, files: tuple[str, ...] = ()) -> None:
        """Store the fixed file list to return, or the built-in default if empty."""
        self._files = files

    def scan(self, ticket: FactoryTicket, *, root: Path) -> ContextScan:
        """Return the fixed file list, ignoring `ticket` and `root`."""
        files = self._files or ("src/main.py", "tests/test_main.py")
        return ContextScan(files=files, strategy="scripted", truncated=False)


def build_scanner(cfg: ContextSettings, *, root: Path) -> ContextScanner:
    """Construct the scanner implied by `cfg.strategy`."""
    strategy = cfg.strategy
    if strategy == "none":
        return ScriptedScanner(files=())
    if strategy == "scripted":
        return ScriptedScanner()
    if strategy == "glob":
        return GlobScanner(cfg)
    if strategy == "git_grep":
        return GitGrepScanner(cfg)
    # auto
    return GitGrepScanner(cfg) if (root / ".git").exists() else GlobScanner(cfg)
