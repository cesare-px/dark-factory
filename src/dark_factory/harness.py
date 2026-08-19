"""The test harness seam: the configured command that gates the Developer agent's work.

Harness stdout/stderr gets fed back into the next Developer prompt, and it
can contain repo-controlled strings or leaked environment values -- callers
MUST pass it through `intake.sanitize.sanitize_text` and
`intake.sanitize.wrap_as_untrusted` before it reaches a prompt, exactly like
issue bodies already are.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dark_factory.config.model import TestHarnessSettings


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """The outcome of one test-harness run."""

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    truncated: bool
    command: str


class TestHarness(Protocol):
    """Runs the downstream repo's test command for one build-test-fix attempt."""

    def run(self, *, iteration: int, workdir: Path) -> HarnessResult:
        """Run the harness once and return its outcome."""
        ...


class SubprocessHarness:
    """Runs the configured command as a real subprocess.

    No shell by default (`shlex.split`), a hard timeout, and truncated
    captured output -- a runaway or malicious test command can otherwise
    hang the pipeline or blow the LLM context on the next prompt.
    """

    def __init__(self, cfg: TestHarnessSettings) -> None:
        """Store the command, timeout, and output-cap settings to run with."""
        self._cfg = cfg

    @staticmethod
    def _decode(value: bytes | str | None) -> str:
        # subprocess.TimeoutExpired's stdout/stderr are typed bytes|str|None
        # regardless of the original call's text=True; at runtime they are
        # always str here since we always pass text=True.
        if value is None:
            return ""
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value

    def _run_one(self, command: str, workdir: Path) -> tuple[int, str, str, float]:
        args = command if self._cfg.allow_shell else shlex.split(command)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                shell=self._cfg.allow_shell,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self._cfg.timeout_seconds,
            )
            return proc.returncode, proc.stdout, proc.stderr, time.monotonic() - start
        except subprocess.TimeoutExpired as exc:
            return (
                124,
                self._decode(exc.stdout),
                self._decode(exc.stderr) + "\n[harness: timed out]",
                time.monotonic() - start,
            )
        except OSError as exc:
            # Command not found / not executable -- most commonly a repo
            # that hasn't created its test_harness.command script yet.
            # This is a harness failure to feed back to the Developer agent,
            # not an internal error that should crash the pipeline.
            return 127, "", f"[harness: could not run {command!r}: {exc}]", time.monotonic() - start

    def run(self, *, iteration: int, workdir: Path) -> HarnessResult:
        """Run the optional setup command, then the test command, in `workdir`."""
        if self._cfg.setup_command:
            self._run_one(self._cfg.setup_command, workdir)

        exit_code, stdout, stderr, duration = self._run_one(self._cfg.command, workdir)

        limit = self._cfg.max_output_chars
        truncated = len(stdout) > limit or len(stderr) > limit
        return HarnessResult(
            passed=exit_code == 0,
            exit_code=exit_code,
            stdout=stdout[:limit],
            stderr=stderr[:limit],
            duration_seconds=duration,
            truncated=truncated,
            command=self._cfg.command,
        )


class ScriptedHarness:
    """Deterministic offline harness: passes on the Nth attempt.

    Stands in for a real test suite in tests and local dry-runs, without
    a checked-out project or subprocess execution.
    """

    def __init__(self, *, passes_on_attempt: int = 3) -> None:
        """Set which attempt number this scripted harness starts passing on."""
        self._passes_on_attempt = passes_on_attempt

    def run(self, *, iteration: int, workdir: Path) -> HarnessResult:
        """Return a canned pass/fail result for the given attempt number."""
        passed = iteration >= self._passes_on_attempt
        return HarnessResult(
            passed=passed,
            exit_code=0 if passed else 1,
            stdout="3 passed" if passed else "",
            stderr="" if passed else f"AssertionError: attempt {iteration} failed",
            duration_seconds=0.0,
            truncated=False,
            command="<scripted>",
        )


def build_harness(cfg: TestHarnessSettings) -> TestHarness:
    """Construct the real subprocess-backed test harness for `cfg`."""
    return SubprocessHarness(cfg)
