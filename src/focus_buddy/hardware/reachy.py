"""The Reachy Mini as a body: its camera and its speaker."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Cap the frame we keep in memory. The face crop happens downstream, so full
# sensor resolution buys nothing and costs RAM on the robot.
_MAX_FRAME_EDGE = 1280
# play_sound returns immediately, so hold on past the clip to let it finish.
_PLAYBACK_MARGIN_S = 0.25


class ReachyBody:
    """Camera and speaker on a Reachy Mini.

    Wraps a ``ReachyMini`` handed in by the app framework rather than creating
    one: the framework owns the connection lifecycle and the media lock.
    """

    def __init__(self, reachy_mini: object, wobble: bool = True) -> None:
        """Wrap a connected ReachyMini instance."""
        self._mini = reachy_mini
        self._last_frame: Image.Image | None = None
        self._wobbling = False

        if wobble:
            # The SDK hooks the audio pipeline, so the head sways in time with
            # whatever we play. Cheaper and better-looking than a motion loop.
            try:
                self._mini.enable_wobbling()  # type: ignore[attr-defined]
                self._wobbling = True
                logger.info("Connected: camera, speaker and talk-wobble ready")
            except Exception:
                logger.warning("Connected, but head wobble is unavailable", exc_info=True)
        else:
            logger.info("Connected: camera and speaker ready")

    def grab_frame(self) -> Image.Image | None:
        """Return the newest camera frame, or the previous one if none arrived."""
        frame = self._mini.media.get_frame()  # type: ignore[attr-defined]
        if frame is None:
            return self._last_frame

        height, width = frame.shape[:2]
        longest = max(width, height)
        if longest > _MAX_FRAME_EDGE:
            scale = _MAX_FRAME_EDGE / longest
            frame = cv2.resize(
                frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
            )

        # The SDK yields BGR arrays; every model downstream expects RGB.
        self._last_frame = Image.fromarray(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))
        return self._last_frame

    def play_audio(self, wav_path: Path, duration_s: float) -> None:
        """Play a WAV through the robot's speaker."""
        try:
            self._mini.media.play_sound(str(wav_path))  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Could not play audio on the robot")
            return
        # play_sound is non-blocking and the caller deletes the file afterwards,
        # so wait for playback rather than pulling the file out from under it.
        time.sleep(max(0.0, duration_s) + _PLAYBACK_MARGIN_S)

    def close(self) -> None:
        """Stop wobbling. The connection itself belongs to the app framework."""
        if self._wobbling:
            try:
                self._mini.disable_wobbling()  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Could not disable wobbling", exc_info=True)
            self._wobbling = False
