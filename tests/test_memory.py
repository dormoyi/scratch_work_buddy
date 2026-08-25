from datetime import date, timedelta

from focus_buddy.memory import RECENT_WINDOW, DayMemory
from focus_buddy.observations import Habit, Observation

BITING = Observation(biting_nails=True)
IDLE = Observation()


class TestEpisodeCounting:
    def test_counts_episodes_not_frames(self):
        # At 1fps a 30-second habit is one episode, not thirty.
        memory = DayMemory()
        for _ in range(30):
            memory.record(BITING)
        assert memory.counters[Habit.BITING_NAILS] == 1

    def test_counts_a_new_episode_after_a_gap(self):
        memory = DayMemory()
        memory.record(BITING)
        memory.record(IDLE)
        memory.record(BITING)
        assert memory.counters[Habit.BITING_NAILS] == 2

    def test_recent_window_is_bounded(self):
        memory = DayMemory()
        for _ in range(RECENT_WINDOW * 3):
            memory.record(IDLE)
        assert len(memory.recent) == RECENT_WINDOW


class TestDayRollover:
    def test_rollover_resets_counters(self):
        memory = DayMemory()
        memory.record(BITING)
        memory.day = date.today() - timedelta(days=1)
        assert memory.roll_day_if_needed() is True
        assert memory.counters[Habit.BITING_NAILS] == 0
        assert memory.day == date.today()

    def test_no_rollover_within_the_same_day(self):
        assert DayMemory().roll_day_if_needed() is False

    def test_episode_state_resets_across_the_rollover(self):
        # Without resetting the active-habit map, a habit still in progress at
        # midnight would never be counted again on the new day.
        memory = DayMemory()
        memory.record(BITING)
        memory.day = date.today() - timedelta(days=1)
        memory.roll_day_if_needed()
        memory.record(BITING)
        assert memory.counters[Habit.BITING_NAILS] == 1


class TestSummary:
    def test_mentions_every_habit_and_the_elapsed_time(self):
        clock = iter([0.0] + [3600.0] * 20)
        memory = DayMemory(clock=lambda: next(clock))
        memory.counters[Habit.BITING_NAILS] = 3
        memory.counters[Habit.LOOKING_AT_PHONE] = 1
        summary = memory.summary()
        assert "nail-biting 3 times" in summary
        assert "phone 1 time" in summary  # singular
        assert "You've been working 1h" in summary
