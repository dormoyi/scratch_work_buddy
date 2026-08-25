"""Speech backends: sentence in, WAV file out."""

from __future__ import annotations

import logging

from ..config import Settings, SpeechEngine
from .base import SpeechBackend, wav_duration_s
from .null import NullSpeech

logger = logging.getLogger(__name__)


def build_speech_backend(settings: Settings) -> SpeechBackend:
    """Construct the speech backend named by ``settings``.

    Falls back to silence rather than failing the run: a buddy that watches
    quietly is still useful, one that refuses to start is not.
    """
    if settings.speech is SpeechEngine.NONE:
        return NullSpeech()

    if settings.speech is SpeechEngine.MACOS:
        from .macos_say import MacOSSay

        try:
            return MacOSSay()
        except RuntimeError:
            logger.exception("Falling back to silent mode")
            return NullSpeech()

    from .openai_tts import OpenAITTS

    return OpenAITTS(model=settings.cloud_tts_model, voice=settings.cloud_tts_voice)


__all__ = ["NullSpeech", "SpeechBackend", "build_speech_backend", "wav_duration_s"]
