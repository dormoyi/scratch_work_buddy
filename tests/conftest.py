"""Fakes that let the whole loop run with no camera, no models and no network."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image

from focus_buddy.config import Settings, SpeechEngine
from focus_buddy.observations import Observation


class FakeBody:
    """A body that returns scripted frames and records what it played."""

    def __init__(self, frames: int = 10):
        self._remaining = frames
        self.played: list[Path] = []
        self.closed = False

    def grab_frame(self):
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return Image.new("RGB", (64, 48), color=(10, 20, 30))

    def play_audio(self, wav_path: Path, duration_s: float) -> None:
        self.played.append(wav_path)

    def close(self) -> None:
        self.closed = True


class FakeVision:
    """Returns a scripted sequence of observations, repeating the last one."""

    def __init__(self, observations: list[Observation]):
        self._observations = list(observations)
        self._index = 0
        self.closed = False

    def classify(self, frame) -> Observation:
        observation = self._observations[min(self._index, len(self._observations) - 1)]
        self._index += 1
        return observation

    def close(self) -> None:
        self.closed = True


class FakeSpeech:
    """Writes a tiny real WAV so the file-handling path is genuinely exercised."""

    def __init__(self):
        self.spoken: list[str] = []

    def synthesize(self, text: str, destination: Path):
        self.spoken.append(text)
        import wave

        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence
        return 0.1


class FakeNudgeWriter:
    """Returns a canned sentence, or raises if asked to."""

    def __init__(self, reply: str = "", explode: bool = False):
        self.reply = reply
        self.explode = explode
        self.calls: list[str] = []
        self.closed = False

    def write_nudge(self, habit_label: str, recent_lines=None) -> str:
        self.calls.append(habit_label)
        if self.explode:
            raise RuntimeError("model unavailable")
        return self.reply

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def settings() -> Settings:
    return Settings(
        speech=SpeechEngine.NONE,
        poll_interval_s=0.0,
        nudge_cooldown_s=0.0,
        summary_every_s=10_000.0,
    )


@pytest.fixture
def stop_event() -> threading.Event:
    return threading.Event()
