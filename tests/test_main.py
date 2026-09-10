"""Entry-point wiring: the flags a user types must reach the loop that runs."""

from __future__ import annotations

import logging
import threading
import time

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


class TestSettingsPageIsAlwaysReachable:
    """The page must exist once configured, not only while broken.

    Attaching the endpoints only when settings were missing meant the page
    worked exactly until someone used it: after that main() injected valid
    settings, the routes were never added, and the page 404ed forever.
    """

    def _routes(self, app):
        return {getattr(r, "path", None) for r in app.settings_app.routes}

    def test_endpoints_exist_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr(
            ReachyMiniApp, "_check_daemon_on_localhost", staticmethod(lambda *a, **k: False)
        )
        assert "/api/settings" in self._routes(main_module.FocusBuddy())

    def test_endpoints_exist_when_settings_were_injected(self, monkeypatch):
        monkeypatch.setattr(
            ReachyMiniApp, "_check_daemon_on_localhost", staticmethod(lambda *a, **k: False)
        )
        app = main_module.FocusBuddy(settings=Settings())
        assert "/api/settings" in self._routes(app)


class TestEitherEvent:
    """The loop takes one stop event but must stop for two different reasons."""

    def test_unset_when_neither_is_set(self):
        a, b = threading.Event(), threading.Event()
        assert main_module._EitherEvent(a, b).is_set() is False

    def test_set_when_either_is_set(self):
        a, b = threading.Event(), threading.Event()
        either = main_module._EitherEvent(a, b)
        b.set()
        assert either.is_set() is True
        b.clear()
        a.set()
        assert either.is_set() is True

    def test_wait_returns_immediately_once_set(self):
        a, b = threading.Event(), threading.Event()
        b.set()
        started = time.monotonic()
        assert main_module._EitherEvent(a, b).wait(5) is True
        assert time.monotonic() - started < 1

    def test_wait_times_out_when_neither_fires(self):
        a, b = threading.Event(), threading.Event()
        assert main_module._EitherEvent(a, b).wait(0.3) is False

    def test_wait_wakes_on_the_second_event(self):
        """Blocking on the stop event alone would sleep through a reload."""
        a, b = threading.Event(), threading.Event()
        threading.Timer(0.2, b.set).start()
        started = time.monotonic()
        assert main_module._EitherEvent(a, b).wait(5) is True
        assert time.monotonic() - started < 2


class TestSettingsArePinnedOnlyByFlags:
    """Pinning on every launch made Save kill the app.

    FocusBuddy.run treats injected settings as "the CLI chose these, do not let
    the page override them" and returns instead of rebuilding. Injecting them
    whenever the configuration merely happened to be valid meant every
    dashboard launch looked pinned, so a reload exited the process.
    """

    def test_no_flags_leaves_the_app_free_to_reload(self, captured_app):
        assert main_module.main([]) == 0
        assert captured_app[0].settings is None

    def test_flags_are_still_pinned(self, captured_app):
        assert main_module.main(["--debug"]) == 0
        assert captured_app[0].settings is not None
        assert captured_app[0].settings.nudge_cooldown_s == 15.0

    def test_has_overrides_detects_each_flag(self):
        for argv in (["--debug"], ["--backend", "cloud"], ["--speech", "none"], ["--llm-nudges"]):
            assert main_module.has_overrides(main_module.parse_args(argv)) is True

    def test_has_overrides_ignores_non_settings_flags(self):
        """-v and --desktop change how it runs, not what it is configured as."""
        assert main_module.has_overrides(main_module.parse_args(["-v"])) is False
        assert main_module.has_overrides(main_module.parse_args([])) is False
