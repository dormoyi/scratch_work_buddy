import threading

from conftest import FakeBody, FakeNudgeWriter, FakeSpeech, FakeVision

from focus_buddy.loop import FocusLoop
from focus_buddy.observations import Habit, Observation

WORKING = Observation()
PHONE = Observation(looking_at_phone=True, looking_at_screen=False)
NAILS = Observation(biting_nails=True)


def build(settings, observations, frames=10, nudge_writer=None):
    body, speech = FakeBody(frames=frames), FakeSpeech()
    loop = FocusLoop(
        body=body,
        vision=FakeVision(observations),
        speech=speech,
        settings=settings,
        nudge_writer=nudge_writer,
    )
    return loop, body, speech


class TestSpeaking:
    def test_stays_quiet_while_you_work(self, settings):
        loop, _, speech = build(settings, [WORKING])
        for _ in range(5):
            loop.tick()
        assert speech.spoken == []

    def test_nudges_on_a_detected_habit(self, settings):
        loop, body, speech = build(settings, [PHONE])
        loop.tick()
        assert len(speech.spoken) == 1
        assert "phone" in speech.spoken[0].lower()
        assert len(body.played) == 1

    def test_cooldown_suppresses_a_second_nudge(self, settings):
        settings.nudge_cooldown_s = 3600.0
        loop, _, speech = build(settings, [PHONE])
        loop.tick()
        loop.tick()
        assert len(speech.spoken) == 1

    def test_no_frame_means_no_work(self, settings):
        loop, _, speech = build(settings, [PHONE], frames=0)
        assert loop.tick() is None
        assert speech.spoken == []


class TestMemoryIntegration:
    def test_counts_the_episode_it_nudged_about(self, settings):
        loop, _, _ = build(settings, [NAILS])
        loop.tick()
        assert loop.memory.counters[Habit.BITING_NAILS] == 1

    def test_summary_is_spoken_when_due(self, settings):
        settings.summary_every_s = 0.0
        loop, _, speech = build(settings, [WORKING])
        loop.tick()
        assert any("You've been working" in line for line in speech.spoken)


class TestLLMNudges:
    def test_uses_a_valid_model_nudge(self, settings):
        settings.use_llm_nudges = True
        writer = FakeNudgeWriter("Hey, maybe set the phone aside now.")
        loop, _, speech = build(settings, [PHONE], nudge_writer=writer)
        loop.tick()
        assert speech.spoken == ["Hey, maybe set the phone aside now."]
        assert writer.calls == ["phone"]

    def test_falls_back_when_the_model_talks_about_the_wrong_habit(self, settings):
        settings.use_llm_nudges = True
        writer = FakeNudgeWriter("Hey, you are touching your face again.")
        loop, _, speech = build(settings, [PHONE], nudge_writer=writer)
        loop.tick()
        # Rejected, so a template about the phone is spoken instead.
        assert "phone" in speech.spoken[0].lower()
        assert speech.spoken[0] != writer.reply

    def test_falls_back_when_the_model_raises(self, settings):
        settings.use_llm_nudges = True
        writer = FakeNudgeWriter(explode=True)
        loop, _, speech = build(settings, [PHONE], nudge_writer=writer)
        loop.tick()
        assert len(speech.spoken) == 1

    def test_templates_are_used_when_llm_nudges_are_off(self, settings):
        writer = FakeNudgeWriter("Hey, maybe set the phone aside now.")
        loop, _, _ = build(settings, [PHONE], nudge_writer=writer)
        loop.tick()
        assert writer.calls == []


class TestLifecycle:
    def test_run_stops_when_the_event_is_set(self, settings):
        loop, body, _ = build(settings, [WORKING], frames=1000)
        stop = threading.Event()

        original_tick = loop.tick
        ticks = []

        def counting_tick():
            ticks.append(1)
            if len(ticks) >= 3:
                stop.set()
            return original_tick()

        loop.tick = counting_tick
        loop.run(stop)
        assert len(ticks) == 3

    def test_run_closes_everything_it_owns(self, settings):
        writer = FakeNudgeWriter()
        loop, body, _ = build(settings, [WORKING], nudge_writer=writer)
        stop = threading.Event()
        stop.set()
        loop.run(stop)
        assert body.closed and writer.closed


class TestBodyOwnership:
    """A settings reload rebuilds the backends but keeps the robot connection."""

    def _loop(self, owns_body):
        from focus_buddy.config import Settings, SpeechEngine
        from focus_buddy.loop import FocusLoop

        body, vision, speech = FakeBody(), FakeVision([Observation()]), FakeSpeech()
        loop = (
            FocusLoop(
                body=body,
                vision=vision,
                speech=speech,
                settings=Settings(speech=SpeechEngine.NONE),
            )
            if owns_body
            else FocusLoop(
                body=body,
                vision=vision,
                speech=speech,
                settings=Settings(speech=SpeechEngine.NONE),
                owns_body=False,
            )
        )
        return loop, body, vision

    def test_by_default_the_loop_closes_the_body(self):
        loop, body, vision = self._loop(True)
        loop.close()
        assert body.closed is True and vision.closed is True

    def test_a_borrowed_body_is_left_open(self):
        """Reconnecting to the robot cost ~20s and a wobble toggle for nothing."""
        loop, body, vision = self._loop(False)
        loop.close()
        assert body.closed is False
        # The things that do depend on settings are still torn down.
        assert vision.closed is True
