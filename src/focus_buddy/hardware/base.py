"""The contract every body implements: somewhere to see from, somewhere to speak."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class Body(Protocol):
    """A camera and a speaker.

    Exists so the focus loop is identical whether it runs on a Reachy Mini or on
    a laptop webcam. Development without the robot is the common case, and it
    should exercise the same code path that ships.
    """

    def grab_frame(self) -> Image.Image | None:
        """Return the most recent camera frame as RGB, or None if unavailable."""
        ...

    def play_audio(self, wav_path: Path, duration_s: float) -> None:
        """Play a WAV file, blocking until playback finishes."""
        ...

    def close(self) -> None:
        """Release the camera and any audio resources. Safe to call twice."""
        ...
