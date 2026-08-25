"""Opt-in speech backend using the macOS `say` command.

Offline and free, but macOS only. Selected with FOCUS_BUDDY_SPEECH=macos.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .base import wav_duration_s

logger = logging.getLogger(__name__)

# `say` emits AIFF; the robot's audio pipeline wants 16-bit PCM WAV.
_AFCONVERT_ARGS = ("-f", "WAVE", "-d", "LEI16")


class MacOSSay:
    """Synthesize speech with the macOS `say` command."""

    def __init__(self, voice: str | None = None) -> None:
        """Check that the required macOS tools are present."""
        for tool in ("say", "afconvert"):
            if shutil.which(tool) is None:
                raise RuntimeError(
                    f"FOCUS_BUDDY_SPEECH=macos needs the `{tool}` command, which was not "
                    "found. This backend is macOS only; use FOCUS_BUDDY_SPEECH=openai."
                )
        self._voice = voice

    def synthesize(self, text: str, destination: Path) -> float | None:
        """Write a WAV for ``text``, returning its duration in seconds."""
        text = (text or "").strip()
        if not text:
            return None

        aiff_path = destination.with_suffix(".aiff")
        command = ["say", "-o", str(aiff_path)]
        if self._voice:
            command += ["-v", self._voice]
        command.append(text)

        try:
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run(
                ["afconvert", *_AFCONVERT_ARGS, str(aiff_path), str(destination)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.exception("`say` synthesis failed; the nudge will only be logged")
            return None
        finally:
            aiff_path.unlink(missing_ok=True)

        return wav_duration_s(destination)
