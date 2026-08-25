"""The contract every nudge-writing backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class NudgeWriter(Protocol):
    """Rephrases a detected habit as a spoken sentence."""

    def write_nudge(self, habit_label: str, recent_lines: list[str] | None = None) -> str:
        """Return a candidate nudge. Callers must still validate it."""
        ...

    def close(self) -> None:
        """Release model resources. Safe to call more than once."""
        ...
