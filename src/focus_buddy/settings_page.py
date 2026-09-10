"""The settings page the Reachy Mini dashboard shows for this app.

An app installed from the app store arrives with no ``.env`` and no way for the
user to set an environment variable, so the first run of Focus Buddy failed
configuration and exited before anyone could supply an API key. The SDK's answer
is a per-app settings page: set ``custom_app_url`` on the app class and the
framework serves ``<package>/static/index.html`` there, which the dashboard
links to.

This module owns the read/write half of that page. What the user types is saved
to :func:`~focus_buddy.config.user_config_path`, which is under the home
directory rather than in site-packages, and is picked up by
:meth:`Settings.from_env` on the next read.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from .config import Backend, Settings, SpeechEngine, user_config_path

logger = logging.getLogger(__name__)

# A frame older than this means the loop is alive but not seeing anything, which
# is a different problem from being stopped and worth showing differently.
STALE_FRAME_AFTER_S = 20.0


class Runtime:
    """What the app is actually doing, for the settings page to report.

    The page used to describe configuration only, so it could say "Ready" about
    an app that was not running and had never seen a frame. Answering "is it
    watching?" needs the running loop, not the settings.
    """

    def __init__(self) -> None:
        """Start out stopped and attached to nothing."""
        self.state = "stopped"
        self.detail = ""
        self._loop: Any = None
        self._started_at: float | None = None
        # Set by the settings page when saved values should take effect. The
        # app watches it and rebuilds its backends rather than making the user
        # stop and start the app by hand for every change.
        self.reload_requested = threading.Event()

    def waiting(self, detail: str) -> None:
        """Record that the app is up but cannot work yet."""
        self.state, self.detail, self._loop = "waiting", detail, None

    def watching(self, loop: Any) -> None:
        """Record the running focus loop, and read liveness from it."""
        self.state, self.detail, self._loop = "watching", "", loop
        self._started_at = time.time()

    def stopped(self) -> None:
        """Record that the app is shutting down."""
        self.state, self.detail, self._loop = "stopped", "", None

    def request_reload(self) -> None:
        """Ask the running app to rebuild itself with the current settings."""
        logger.info("Settings changed; asking the loop to restart")
        self.reload_requested.set()

    def snapshot(self) -> dict[str, Any]:
        """Describe the running state for the page."""
        loop = self._loop
        info: dict[str, Any] = {"state": self.state, "detail": self.detail}
        if loop is None:
            return info

        last = getattr(loop, "last_tick_at", None)
        age = None if last is None else time.time() - last
        info["frames_seen"] = getattr(loop, "frames_seen", 0)
        info["seconds_since_frame"] = None if age is None else round(age, 1)
        info["running_for_s"] = (
            None if self._started_at is None else round(time.time() - self._started_at)
        )
        # Alive but blind: the camera stopped feeding us.
        if age is not None and age > STALE_FRAME_AFTER_S:
            info["state"] = "stalled"
            info["detail"] = f"no camera frame for {age:.0f}s"
        elif info["frames_seen"] == 0:
            info["detail"] = "waiting for the first frame"

        memory = getattr(loop, "memory", None)
        if memory is not None:
            info["habits_today"] = {
                habit.label: count
                for habit, count in sorted(memory.counters.items(), key=lambda kv: kv[0].value)
            }
            info["watching_for_s"] = round(memory.elapsed_s)
        return info


RUNTIME = Runtime()

# Only these may be written from the page. An allowlist rather than "any
# FOCUS_BUDDY_* key", so a typo cannot quietly create a setting nothing reads.
WRITABLE = (
    "OPENAI_API_KEY",
    "FOCUS_BUDDY_BACKEND",
    "FOCUS_BUDDY_SPEECH",
    "FOCUS_BUDDY_NUDGE_COOLDOWN_S",
    "FOCUS_BUDDY_WOBBLE",
)


def read_saved(path: Path | None = None) -> dict[str, str]:
    """Read the saved settings file into a plain dict."""
    path = path or user_config_path()
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def save(updates: dict[str, str], path: Path | None = None) -> Path:
    """Merge ``updates`` into the saved settings and apply them to this process.

    Values are written verbatim, so a key is stored as the user typed it. Empty
    values clear the setting rather than storing a blank, which is what lets the
    page offer "use the default" without a separate control.
    """
    path = path or user_config_path()
    values = read_saved(path)
    for key, value in updates.items():
        if key not in WRITABLE:
            logger.warning("Ignoring unwritable setting %s", key)
            continue
        value = (value or "").strip()
        if value:
            values[key] = value
            os.environ[key] = value
        else:
            values.pop(key, None)
            os.environ.pop(key, None)

    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{k}={v}" for k, v in sorted(values.items()))
    path.write_text(
        "# Written by the Focus Buddy settings page. Safe to edit by hand.\n" + body + "\n",
        encoding="utf-8",
    )
    # The API key is the whole reason this file exists; keep it off the disk of
    # anyone else who can read the home directory.
    path.chmod(0o600)
    logger.info("Saved settings to %s", path)
    return path


def current_state() -> dict[str, Any]:
    """Describe the configuration for the page, without leaking the key."""
    settings = Settings.from_env()
    errors = settings.validate()
    saved = read_saved()
    key = os.getenv("OPENAI_API_KEY") or saved.get("OPENAI_API_KEY", "")
    return {
        "backend": settings.backend.value,
        "speech": settings.speech.value,
        "nudge_cooldown_s": settings.nudge_cooldown_s,
        "wobble": settings.wobble,
        # Never the key itself: this is served over plain HTTP on the LAN.
        "has_api_key": bool(key),
        "api_key_hint": f"…{key[-4:]}" if len(key) >= 4 else "",
        "runtime": RUNTIME.snapshot(),
        "ready": not errors,
        "errors": errors,
        "warnings": settings.warnings,
        "backends": [b.value for b in Backend],
        "speech_engines": [s.value for s in SpeechEngine],
    }


def attach(settings_app: Any) -> None:
    """Add the settings endpoints to the FastAPI the SDK built for this app."""
    if settings_app is None:  # pragma: no cover - only when custom_app_url is unset
        return

    @settings_app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        """Report the current configuration and why it is or is not usable."""
        return current_state()

    @settings_app.post("/api/settings")
    async def post_settings(payload: dict[str, str]) -> dict[str, Any]:
        """Save what the page submitted, then report the new state."""
        save(payload)
        return current_state()

    @settings_app.post("/api/reload")
    async def post_reload() -> dict[str, Any]:
        """Apply saved settings to the running app without a manual restart."""
        RUNTIME.request_reload()
        return {"reloading": True}
