import wave

import pytest

from focus_buddy.speech.base import wav_duration_s
from focus_buddy.speech.null import NullSpeech


def write_wav(path, seconds=1.8, rate=24000, declared_frames=None):
    """Write a mono 16-bit WAV, optionally lying about its length in the header."""
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)
    if declared_frames is not None:
        # Overwrite the data-chunk size field the way a streaming writer does.
        raw = bytearray(path.read_bytes())
        raw[40:44] = (declared_frames * 2).to_bytes(4, "little")
        path.write_bytes(raw)
    return path


class TestWavDuration:
    def test_reads_an_honest_header(self, tmp_path):
        path = write_wav(tmp_path / "a.wav", seconds=1.5)
        assert wav_duration_s(path) == pytest.approx(1.5, abs=0.01)

    def test_ignores_a_streaming_placeholder_length(self, tmp_path):
        # OpenAI's streamed WAV declares 2147483647 frames regardless of content.
        # Trusting it made the robot block for 24 days after a single nudge.
        path = write_wav(tmp_path / "b.wav", seconds=1.8, declared_frames=2147483647)
        assert wav_duration_s(path) == pytest.approx(1.8, abs=0.05)

    def test_missing_file_is_zero(self, tmp_path):
        assert wav_duration_s(tmp_path / "nope.wav") == 0.0

    def test_non_wav_is_zero(self, tmp_path):
        path = tmp_path / "c.wav"
        path.write_bytes(b"not a wav at all")
        assert wav_duration_s(path) == 0.0


class TestNullSpeech:
    def test_writes_nothing_and_reports_none(self, tmp_path):
        destination = tmp_path / "d.wav"
        assert NullSpeech().synthesize("anything", destination) is None
        assert not destination.exists()
