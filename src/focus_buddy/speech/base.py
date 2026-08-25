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


# Canonical PCM WAV header size. Only used to bound a frame count, so a file
# with extra chunks is off by a few milliseconds, which does not matter here.
_WAV_HEADER_BYTES = 44


def wav_duration_s(path: Path) -> float:
    """Read a WAV file's duration in seconds, or 0.0 if it cannot be read.

    The declared frame count is bounded by the bytes actually on disk. A WAV
    written by a streaming producer carries a placeholder length in its header
    (OpenAI's is 2147483647), and callers sleep for whatever this returns — so
    trusting the header outright parks the robot for 24 days after one nudge.
    """
    with contextlib.suppress(Exception):
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if not rate:
                return 0.0
            frames = handle.getnframes()
            bytes_per_frame = handle.getnchannels() * handle.getsampwidth()
            if bytes_per_frame:
                on_disk = max(0, path.stat().st_size - _WAV_HEADER_BYTES) // bytes_per_frame
                frames = min(frames, on_disk) if frames > 0 else on_disk
            return frames / float(rate)
    return 0.0
