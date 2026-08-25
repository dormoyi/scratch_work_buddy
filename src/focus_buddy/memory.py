"""What the buddy remembers about your day.

Deliberately small: counts of habit *episodes* since the day started, plus a
bounded window of recent observations. Episodes rather than frames, because at
one frame per second a thirty-second phone check is one distraction, not thirty.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from collections.abc import Callable
from datetime import date

from .nudges import format_duration
from .observations import HABIT_PRIORITY, Habit, Observation

logger = logging.getLogger(__name__)

# How many recent observations to keep. Bounded so a machine left running for a
# week does not grow a list of half a million strings.
RECENT_WINDOW = 120


class DayMemory:
    """Per-day habit counters and a rolling window of recent observations."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        """Create empty memory for today.

        ``clock`` is injectable so tests can move time without sleeping.
        """
        self._clock = clock
        self.recent: deque[Observation] = deque(maxlen=RECENT_WINDOW)
        self.counters: Counter[Habit] = Counter()
        self.day: date = date.today()
        self.day_started_at: float = clock()
        self.last_summary: str = ""
        # Tracks each habit's previous state so we count transitions, not frames.
        self._active: dict[Habit, bool] = dict.fromkeys(HABIT_PRIORITY, False)

    def roll_day_if_needed(self) -> bool:
        """Reset counters when the date changes. Returns True if it rolled.

        Called on every tick rather than from the summary path, so a machine left
        running overnight resets regardless of which branches ran.
        """
        today = date.today()
        if today == self.day:
            return False
        logger.info("New day (%s -> %s); resetting counters", self.day, today)
        self.recent.clear()
        self.counters = Counter()
        self._active = dict.fromkeys(HABIT_PRIORITY, False)
        self.day = today
        self.day_started_at = self._clock()
        self.last_summary = ""
        return True

    def record(self, observation: Observation) -> None:
        """Record one observation, counting any habit that just started."""
        self.roll_day_if_needed()
        self.recent.append(observation)
        for habit in HABIT_PRIORITY:
            now_active = getattr(observation, habit.value)
            if now_active and not self._active[habit]:
                self.counters[habit] += 1
            self._active[habit] = now_active

    @property
    def elapsed_s(self) -> float:
        """Seconds since the buddy started watching today."""
        return self._clock() - self.day_started_at

    def summary(self) -> str:
        """Build a spoken status line from the counters rather than from a model.

        Deterministic on purpose: the numbers are facts about the user's day and
        should not be paraphrased by anything that can hallucinate.
        """
        self.roll_day_if_needed()
        parts = [
            f"{habit.label} {self.counters[habit]} "
            f"{'time' if self.counters[habit] == 1 else 'times'}"
            for habit in HABIT_PRIORITY
        ]
        return f"You've been working {format_duration(self.elapsed_s)}. " + ", ".join(parts) + "."

    def note_summary(self, summary: str) -> None:
        """Remember the last summary spoken."""
        self.last_summary = summary
