import sys

import pytest

from focus_buddy.config import Backend, Settings, SpeechEngine


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class TestValidation:
    def test_cloud_without_a_key_is_fatal(self):
        errors = Settings(backend=Backend.CLOUD).validate()
        assert any("OPENAI_API_KEY" in error for error in errors)

    def test_openai_tts_alone_still_needs_a_key(self, monkeypatch):
        # Edge vision is local, but the default voice is not.
        settings = Settings(backend=Backend.EDGE, speech=SpeechEngine.OPENAI)
        assert any("OPENAI_API_KEY" in error for error in settings.validate())

    def test_fully_local_on_mac_needs_no_key(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        # Any real interpreter satisfies the sidecar check; using this one keeps
        # the suite from depending on a venv the developer happens to have built.
        settings = Settings(
            backend=Backend.EDGE, speech=SpeechEngine.MACOS, landmark_python=sys.executable
        )
        assert settings.validate() == []

    def test_edge_runs_anywhere_mediapipe_does(self, monkeypatch):
        """Edge is geometry now; only edge-vlm is tied to Apple Silicon."""
        monkeypatch.setattr(sys, "platform", "linux")
        settings = Settings(
            backend=Backend.EDGE, speech=SpeechEngine.NONE, landmark_python=sys.executable
        )
        assert settings.validate() == []

    def test_edge_without_a_sidecar_is_fatal(self, monkeypatch):
        """Fail before the camera opens rather than on the first frame."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        errors = Settings(backend=Backend.EDGE, landmark_python="/nope/python").validate()
        # The message must name something an installed user can actually run.
        assert any("focus-buddy-setup-sidecar" in error for error in errors)

    def test_edge_vlm_off_mac_is_fatal(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        errors = Settings(backend=Backend.EDGE_VLM).validate()
        assert any("Apple Silicon" in error for error in errors)

    def test_macos_speech_off_mac_is_fatal(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        errors = Settings(speech=SpeechEngine.MACOS).validate()
        assert any("`say`" in error for error in errors)

    def test_silent_mode_warns_but_runs(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = Settings(speech=SpeechEngine.NONE)
        assert settings.validate() == []
        assert any("Speech is disabled" in warning for warning in settings.warnings)

    def test_short_cooldown_warns(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = Settings(nudge_cooldown_s=1.0)
        settings.validate()
        assert any("talkative" in warning for warning in settings.warnings)


class TestEnvLoading:
    def test_reads_overrides(self, monkeypatch):
        monkeypatch.setenv("FOCUS_BUDDY_BACKEND", "edge")
        monkeypatch.setenv("FOCUS_BUDDY_SPEECH", "none")
        monkeypatch.setenv("FOCUS_BUDDY_NUDGE_COOLDOWN_S", "12.5")
        monkeypatch.setenv("FOCUS_BUDDY_USE_LLM_NUDGES", "true")
        settings = Settings.from_env()
        assert settings.backend is Backend.EDGE
        assert settings.speech is SpeechEngine.NONE
        assert settings.nudge_cooldown_s == 12.5
        assert settings.use_llm_nudges is True

    def test_malformed_number_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("FOCUS_BUDDY_NUDGE_COOLDOWN_S", "soon")
        assert Settings.from_env().nudge_cooldown_s == 60.0


class TestDebugProfile:
    def test_shortens_intervals(self):
        settings = Settings().for_debug()
        assert settings.nudge_cooldown_s == 15.0
        assert settings.summary_every_s == 180.0
