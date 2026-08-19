"""Iteration loop capping for the Build-Test-Fix cycle (blueprint §4.B).

If the developer agent's self-repair loop exceeds a threshold (default 5),
execution must stop and the ticket flagged `factory-blocked` for a human,
rather than looping indefinitely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_ITERATIONS = 5


class LoopLimitExceededError(RuntimeError):
    """Raised when a build-test-fix loop exceeds its iteration cap."""

    def __init__(self, loop_name: str, max_iterations: int) -> None:
        """Build the error with the loop's name and its exceeded cap."""
        self.loop_name = loop_name
        self.max_iterations = max_iterations
        super().__init__(
            f"loop {loop_name!r} exceeded its cap of {max_iterations} iterations "
            "-- halting and flagging factory-blocked"
        )


@dataclass
class IterationLoopTracker:
    """Counts iterations of a bounded retry loop and enforces a hard cap."""

    loop_name: str
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    _count: int = field(default=0, repr=False)

    @property
    def count(self) -> int:
        """The number of iterations completed so far."""
        return self._count

    @property
    def is_exhausted(self) -> bool:
        """Whether the next `step()` call would exceed the cap."""
        return self._count >= self.max_iterations

    def step(self) -> int:
        """Advance the loop by one iteration.

        Raises LoopLimitExceededError once the cap is reached, so the very
        next attempt after the last allowed one is what trips the guardrail
        -- callers should call this at the top of each loop body.
        """
        if self.is_exhausted:
            raise LoopLimitExceededError(self.loop_name, self.max_iterations)
        self._count += 1
        return self._count

    def reset(self) -> None:
        """Reset the iteration count to zero."""
        self._count = 0
