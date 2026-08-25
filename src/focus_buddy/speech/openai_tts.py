"""Default speech backend: OpenAI text-to-speech.

Chosen as the default because it is the only option that works identically on a
laptop and on the Raspberry Pi inside a Reachy Mini wireless.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import wav_duration_s

logger = logging.getLogger(__name__)


class OpenAITTS:
    """Synthesize speech with the OpenAI audio API."""

    def __init__(self, model: str = "gpt-4o-mini-tts", voice: str = "alloy") -> None:
        """Create the backend. The client reads OPENAI_API_KEY from the environment."""
        from openai import OpenAI

        self._model = model
        self._voice = voice
        self._client = OpenAI()

    def synthesize(self, text: str, destination: Path) -> float | None:
        """Write a WAV for ``text``, returning its duration in seconds."""
        text = (text or "").strip()
        if not text:
            return None

        try:
            # WAV rather than MP3 so playback needs no decoder on the robot.
            with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=self._voice,
                input=text,
                response_format="wav",
            ) as response:
                response.stream_to_file(destination)
        except Exception:
            # Losing a nudge is not worth ending the session over.
            logger.exception("Text-to-speech failed; the nudge will only be logged")
            return None

        return wav_duration_s(destination)
