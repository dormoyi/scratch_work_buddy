"""Entry-point wiring: the flags a user types must reach the loop that runs."""

from __future__ import annotations

import logging

import pytest
from reachy_mini import ReachyMiniApp

from focus_buddy import main as main_module
from focus_buddy.config import Backend, Settings, SpeechEngine


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    """Let validate() pass without depending on the developer's real key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class FakeApp:
    """Stands in for FocusBuddy, recording how main() constructed it."""

    instances: list[FakeApp] = []
    # main() names this in the "waiting for the settings page" message, so the
    # stand-in has to carry it too.
    custom_app_url = "http://0.0.0.0:7860/"

    def __init__(self, running_on_wireless: bool = False, settings: Settings | None = None):
        self.settings = settings
        FakeApp.instances.append(self)

    def wrapped_run(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture
def captured_app(monkeypatch):
    """Replace FocusBuddy so the robot path can be exercised with no robot."""
    FakeApp.instances = []
    monkeypatch.setattr(main_module, "FocusBuddy", FakeApp)
    return FakeApp.instances


class TestRobotPathHonoursFlags:
    """Without --desktop, main() defers to the app framework.

    It must hand over the settings it already resolved: re-reading the
    environment inside the app silently drops every CLI flag.
    """

    def test_debug_flag_reaches_the_app(self, captured_app):
        assert main_module.main(["--debug"]) == 0
        settings = captured_app[0].settings
        assert settings is not None
        assert settings.nudge_cooldown_s == 15.0
        assert settings.summary_every_s == 180.0

    def test_speech_and_llm_flags_reach_the_app(self, captured_app):
        assert main_module.main(["--speech", "none", "--llm-nudges"]) == 0
        settings = captured_app[0].settings
        assert settings.speech is SpeechEngine.NONE
        assert settings.use_llm_nudges is True

    def test_backend_flag_reaches_the_app(self, captured_app):
        assert main_module.main(["--backend", "cloud"]) == 0
        assert captured_app[0].settings.backend is Backend.CLOUD

    def test_bad_configuration_still_starts_the_app(self, monkeypatch, captured_app):
        """The dashboard starts apps with `python -m focus_buddy.main`.

        Exiting here is what made a fresh install unrecoverable: the settings
        page that would supply the missing key lives inside the app, so the app
        has to come up even when it cannot yet do any work.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: cls()))
        assert main_module.main([]) == 0
        assert len(captured_app) == 1
        # No settings handed over, so the app re-reads them as the page saves.
        assert captured_app[0].settings is None

    def test_desktop_still_fails_fast_on_bad_configuration(self, monkeypatch, captured_app):
        """A person at a terminal gets an error, not a wait for a page."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: cls()))
        assert main_module.main(["--desktop"]) == 2
        assert captured_app == []


class TestDashboardPathReadsTheEnvironment:
    """The dashboard constructs the app with no arguments."""

    @pytest.fixture(autouse=True)
    def _no_daemon_probe(self, monkeypatch):
        """The SDK base class probes localhost:8000; keep the suite off the network."""
        monkeypatch.setattr(
            ReachyMiniApp, "_check_daemon_on_localhost", staticmethod(lambda *a, **k: False)
        )

    def test_defaults_to_no_injected_settings(self):
        assert main_module.FocusBuddy()._settings is None

    def test_accepts_the_frameworks_positional_argument(self):
        assert main_module.FocusBuddy(True)._settings is None


class TestSetupLogging:
    def test_silences_the_sdk_websocket_telemetry(self):
        """The SDK streams a joint-pose frame at ~50Hz; at -v it buries our lines."""
        main_module.setup_logging(verbose=True)
        for noisy in ("websockets", "httpcore", "httpx"):
            assert logging.getLogger(noisy).level == logging.WARNING
        # Child loggers are where the volume actually comes from.
        for child in ("websockets.client", "httpcore.http11", "httpcore.connection"):
            assert logging.getLogger(child).getEffectiveLevel() == logging.WARNING
