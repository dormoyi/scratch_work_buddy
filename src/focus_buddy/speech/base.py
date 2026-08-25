"""The contract every speech backend implements."""

from __future__ import annotations

import contextlib
import wave
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechBackend(Protocol):
    """Turns a sentence into a playable WAV file.

    Synthesis is deliberately separate from playback: the robot, a laptop and the
    test suite all play audio differently, but they can share one voice.
    """

    def synthesize(self, text: str, destination: Path) -> float | None:
        """Write a WAV for ``text`` and return its duration in seconds.

        Returns None when nothing was written, which callers treat as "log the
        nudge but stay silent" rather than as an error.
        """
        ...


def wav_duration_s(path: Path) -> float:
    """Read a WAV file's duration in seconds, or 0.0 if it cannot be read."""
    with contextlib.suppress(Exception):
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate:
                return handle.getnframes() / float(rate)
    return 0.0
