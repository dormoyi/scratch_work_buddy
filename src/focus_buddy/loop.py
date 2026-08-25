"""The focus loop: look, decide, occasionally speak."""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from pathlib import Path

from . import nudges
from .brain import NudgeWriter
from .config import Settings
from .hardware import Body
from .memory import DayMemory
from .observations import Habit, Observation
from .perception import VisionBackend
from .speech import SpeechBackend

logger = logging.getLogger(__name__)


class FocusLoop:
    """Watches the camera and nudges the user about focus-breaking habits.

    Collaborators are injected rather than constructed here, so the loop can be
    tested end to end with fakes and no models, camera or robot.
    """

    def __init__(
        self,
        body: Body,
        vision: VisionBackend,
        speech: SpeechBackend,
        settings: Settings,
        nudge_writer: NudgeWriter | None = None,
        memory: DayMemory | None = None,
    ) -> None:
        """Wire the loop to its collaborators."""
        self._body = body
        self._vision = vision
        self._speech = speech
        self._settings = settings
        self._nudge_writer = nudge_writer
        self.memory = memory or DayMemory()

        self._last_spoke_at = 0.0
        self._last_summary_at = time.time()
        # Recent nudges, so a rephrasing model can be told not to repeat itself.
        self._spoken: list[str] = []

    def run(self, stop_event: threading.Event) -> None:
        """Run until ``stop_event`` is set.

        Every wait goes through ``stop_event.wait`` rather than ``time.sleep`` so
        the dashboard's stop button takes effect within one tick instead of
        needing the process killed.
        """
        logger.info(
            "Focus buddy watching: backend=%s speech=%s cooldown=%.0fs",
            self._settings.backend.value,
            self._settings.speech.value,
            self._settings.nudge_cooldown_s,
        )
        try:
            while not stop_event.is_set():
                self.tick()
                stop_event.wait(self._settings.poll_interval_s)
        finally:
            self.close()

    def tick(self) -> Observation | None:
        """Process a single frame. Returns the observation, or None if no frame."""
        self.memory.roll_day_if_needed()

        frame = self._body.grab_frame()
        if frame is None:
            return None

        started = time.perf_counter()
        observation = self._vision.classify(frame)
        logger.info("[%.2fs] %s", time.perf_counter() - started, observation)
        self.memory.record(observation)

        nudge = self._nudge_for(observation)
        if nudge is not None:
            self.say(nudge)

        summary = self._due_summary()
        if summary is not None:
            self.say(summary, is_nudge=False)

        return observation

    def _nudge_for(self, observation: Observation) -> str | None:
        """Decide what to say about an observation, or None to stay quiet."""
        habit = observation.primary_habit
        if habit is None:
            return None
        if time.time() - self._last_spoke_at < self._settings.nudge_cooldown_s:
            return None

        fallback = nudges.pick_template(habit)
        if not self._settings.use_llm_nudges or self._nudge_writer is None:
            return fallback
        return self._rephrase(habit, observation, fallback)

    def _rephrase(self, habit: Habit, observation: Observation, fallback: str) -> str:
        """Ask the language model to phrase the nudge, validating what comes back.

        Anything that fails validation is replaced by the template. A nudge that
        is confidently wrong about the user is worse than a repetitive one.
        """
        assert self._nudge_writer is not None
        started = time.perf_counter()
        try:
            candidate = nudges.clean(self._nudge_writer.write_nudge(habit.label, self._spoken))
        except Exception:
            logger.exception("Nudge model failed; using the template")
            return fallback

        elapsed = time.perf_counter() - started
        # Identical to the last thing said: treat as stale, not as valid.
        if self._spoken and candidate.lower() == self._spoken[-1].lower():
            candidate = ""

        reason = nudges.rejection_reason(candidate, habit, observation) if candidate else "empty"
        if reason is not None:
            logger.info("[%.2fs] rejected nudge (%s): %r", elapsed, reason, candidate)
            return fallback

        logger.info("[%.2fs] nudge: %s", elapsed, candidate)
        return candidate

    def _due_summary(self) -> str | None:
        """Return the daily status line if enough time has passed, else None."""
        if time.time() - self._last_summary_at < self._settings.summary_every_s:
            return None
        summary = self.memory.summary()
        self.memory.note_summary(summary)
        self._last_summary_at = time.time()
        return summary

    def say(self, message: str, is_nudge: bool = True) -> None:
        """Speak a message, and record that the buddy just spoke."""
        message = (message or "").strip()
        if not message:
            return

        logger.info("Saying: %s", message)
        self._last_spoke_at = time.time()
        if is_nudge:
            self._spoken.append(message)
            del self._spoken[:-10]

        with tempfile.TemporaryDirectory(prefix="focus_buddy_tts_") as tmp:
            wav_path = Path(tmp) / "nudge.wav"
            duration = self._speech.synthesize(message, wav_path)
            if duration is None or not wav_path.exists():
                return
            self._body.play_audio(wav_path, duration)

    def close(self) -> None:
        """Release every collaborator, reporting failures without masking them."""
        for name, resource in (
            ("body", self._body),
            ("vision", self._vision),
            ("nudge writer", self._nudge_writer),
        ):
            closer = getattr(resource, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.warning("Error while closing the %s", name, exc_info=True)
