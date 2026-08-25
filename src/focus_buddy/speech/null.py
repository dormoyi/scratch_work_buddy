"""Silent speech backend, for tests and for running without audio."""

from __future__ import annotations

from pathlib import Path


class NullSpeech:
    """Never produces audio; nudges are logged only."""

    def synthesize(self, text: str, destination: Path) -> float | None:
        """Do nothing and report that no audio was written."""
        return None
