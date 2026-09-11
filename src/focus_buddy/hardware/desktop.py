"""A laptop standing in for the robot, for development without hardware."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Frames to discard before reading. OpenCV buffers frames, and with multi-second
# inference the buffered one can be several seconds stale - which shows up as the
# buddy nudging you about something you stopped doing.
_STALE_FRAMES_TO_DROP = 4


class DesktopBody:
    """Built-in webcam and system audio."""

    def __init__(self, camera_index: int = 0) -> None:
        """Open the webcam."""
        self._camera = cv2.VideoCapture(camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index}. On macOS, grant camera access to "
                "your terminal in System Settings > Privacy & Security > Camera."
            )
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._last_frame: Image.Image | None = None
        self._player = self._find_player()
        if self._player is None:
            logger.warning("No audio player found; nudges will be logged but not played")

    @staticmethod
    def _find_player() -> list[str] | None:
        """Find a command-line WAV player for this platform."""
        candidates = (
            [["afplay"]] if sys.platform == "darwin" else [["aplay", "-q"], ["paplay"], ["play"]]
        )
        for candidate in candidates:
            if shutil.which(candidate[0]):
                return candidate
        return None

    def grab_frame(self) -> Image.Image | None:
        """Return the newest webcam frame, or the previous one if the read failed."""
        for _ in range(_STALE_FRAMES_TO_DROP):
            self._camera.grab()
        ok, frame = self._camera.read()
        if not ok:
            return self._last_frame
        self._last_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self._last_frame

    def play_audio(self, wav_path: Path, duration_s: float) -> None:
        """Play a WAV through the system speakers."""
        if self._player is None:
            return
        try:
            subprocess.run([*self._player, str(wav_path)], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.exception("Audio playback failed")

    def close(self) -> None:
        """Release the webcam."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None  # type: ignore[assignment]
