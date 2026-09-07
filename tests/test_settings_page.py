"""The dashboard settings page: what it saves, and what it must never hand back.

This exists because an app-store install has no .env and no way to set an
environment variable, so without a settings page Focus Buddy could not be
configured at all -- it validated, raised and exited before anyone could type a
key into it.
"""

from __future__ import annotations

import pytest

from focus_buddy import settings_page
from focus_buddy.config import Backend, Settings, SpeechEngine


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Point the settings file at a temporary home."""
    path = tmp_path / "config.env"
    monkeypatch.setattr(settings_page, "user_config_path", lambda: path)
    monkeypatch.setattr("focus_buddy.config.user_config_path", lambda: path)
    # Settings.from_env also loads a .env found from the working directory. On a
    # developer's machine that is the repo's own, complete with a real key, which
    # would make "a fresh install cannot start" quietly pass for the wrong reason.
    monkeypatch.setattr("dotenv.find_dotenv", lambda *a, **k: "")
    for name in ("OPENAI_API_KEY", "FOCUS_BUDDY_BACKEND", "FOCUS_BUDDY_SPEECH"):
        monkeypatch.delenv(name, raising=False)
    return path


class TestSaveAndRead:
    def test_round_trips_a_value(self, config_file):
        settings_page.save({"OPENAI_API_KEY": "sk-abc"}, config_file)
        assert settings_page.read_saved(config_file)["OPENAI_API_KEY"] == "sk-abc"

    def test_merges_rather_than_replaces(self, config_file):
        settings_page.save({"OPENAI_API_KEY": "sk-abc"}, config_file)
        settings_page.save({"FOCUS_BUDDY_SPEECH": "none"}, config_file)
        saved = settings_page.read_saved(config_file)
        assert saved["OPENAI_API_KEY"] == "sk-abc"
        assert saved["FOCUS_BUDDY_SPEECH"] == "none"

    def test_applies_to_the_running_process(self, config_file, monkeypatch):
        """The app re-reads settings while waiting; it must see the new value."""
        settings_page.save({"OPENAI_API_KEY": "sk-live"}, config_file)
        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-live"

    def test_empty_clears_rather_than_storing_blank(self, config_file):
        settings_page.save({"FOCUS_BUDDY_SPEECH": "none"}, config_file)
        settings_page.save({"FOCUS_BUDDY_SPEECH": ""}, config_file)
        assert "FOCUS_BUDDY_SPEECH" not in settings_page.read_saved(config_file)

    def test_only_allowlisted_keys_are_written(self, config_file):
        """A typo must not quietly create a setting nothing reads."""
        settings_page.save({"PATH": "/evil", "FOCUS_BUDDY_NONSENSE": "1"}, config_file)
        saved = settings_page.read_saved(config_file)
        assert "PATH" not in saved and "FOCUS_BUDDY_NONSENSE" not in saved

    def test_file_is_not_world_readable(self, config_file):
        """It holds an API key."""
        settings_page.save({"OPENAI_API_KEY": "sk-abc"}, config_file)
        assert config_file.stat().st_mode & 0o077 == 0

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert settings_page.read_saved(tmp_path / "absent.env") == {}

    def test_comments_and_blanks_are_ignored(self, config_file):
        config_file.write_text("# a comment\n\nFOCUS_BUDDY_SPEECH=none\nnot-a-pair\n")
        assert settings_page.read_saved(config_file) == {"FOCUS_BUDDY_SPEECH": "none"}


class TestCurrentState:
    def test_reports_why_a_fresh_install_cannot_start(self, config_file):
        state = settings_page.current_state()
        assert state["ready"] is False
        assert any("OPENAI_API_KEY" in error for error in state["errors"])

    def test_becomes_ready_once_a_key_is_saved(self, config_file):
        settings_page.save({"OPENAI_API_KEY": "sk-abc"}, config_file)
        assert settings_page.current_state()["ready"] is True

    def test_never_returns_the_key_itself(self, config_file):
        """The page is served over plain HTTP on the LAN."""
        settings_page.save({"OPENAI_API_KEY": "sk-secret-value"}, config_file)
        state = settings_page.current_state()
        assert "sk-secret-value" not in repr(state)
        assert state["has_api_key"] is True
        assert state["api_key_hint"] == "…alue"

    def test_offers_every_backend_and_speech_engine(self, config_file):
        state = settings_page.current_state()
        assert state["backends"] == [b.value for b in Backend]
        assert state["speech_engines"] == [s.value for s in SpeechEngine]


class TestConfigLoading:
    def test_settings_pick_up_the_saved_file(self, config_file, monkeypatch):
        settings_page.save({"FOCUS_BUDDY_SPEECH": "none"}, config_file)
        monkeypatch.delenv("FOCUS_BUDDY_SPEECH", raising=False)
        assert Settings.from_env().speech is SpeechEngine.NONE

    def test_a_real_environment_variable_still_wins(self, config_file, monkeypatch):
        """Saved settings fill gaps; they do not override an explicit env var."""
        settings_page.save({"FOCUS_BUDDY_BACKEND": "cloud"}, config_file)
        monkeypatch.setenv("FOCUS_BUDDY_BACKEND", "edge")
        assert Settings.from_env().backend is Backend.EDGE
