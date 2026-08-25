from focus_buddy import nudges
from focus_buddy.observations import Habit, Observation


class TestClean:
    def test_strips_quotes_and_keeps_first_sentence(self):
        assert nudges.clean('"Hey, put the phone down! More text."') == "Hey, put the phone down!"

    def test_handles_none(self):
        assert nudges.clean(None) == ""


class TestValidation:
    def test_accepts_a_good_phone_nudge(self):
        assert nudges.is_valid("Hey, put the phone down now.", Habit.LOOKING_AT_PHONE)

    def test_rejects_a_nudge_about_the_wrong_habit(self):
        reason = nudges.rejection_reason("Hey, you touched your face again.", Habit.BITING_NAILS)
        assert reason == "does not mention biting_nails"

    def test_rejects_too_short(self):
        assert nudges.rejection_reason("Face-touching", Habit.TOUCHING_FACE) == "too short"

    def test_rejects_too_long(self):
        long_nudge = "Hey there friend, your hand has wandered up to your face once again today"
        assert "too long" in nudges.rejection_reason(long_nudge, Habit.TOUCHING_FACE)

    def test_rejects_screen_complaint_the_frame_contradicts(self):
        # The buddy must not tell you that you are ignoring a screen you are reading.
        observed = Observation(looking_at_screen=True, touching_face=True)
        bad = "You keep touching your face and aren't looking at your screen."
        assert nudges.rejection_reason(bad, Habit.TOUCHING_FACE, observed) is not None

    def test_allows_screen_mention_when_frame_agrees(self):
        observed = Observation(looking_at_screen=False, looking_at_phone=True)
        assert nudges.is_valid(
            "Hey, look at your screen instead of the phone.", Habit.LOOKING_AT_PHONE, observed
        )

    def test_every_template_passes_its_own_validation(self):
        # A template that the validator rejects would make the fallback path
        # produce nudges the LLM path would have refused.
        for habit, templates in nudges.NUDGE_TEMPLATES.items():
            for template in templates:
                assert nudges.is_valid(template, habit), (habit, template)


class TestFormatDuration:
    def test_seconds(self):
        assert nudges.format_duration(45) == "45s"

    def test_minutes(self):
        assert nudges.format_duration(600) == "10m"

    def test_hours_and_minutes(self):
        assert nudges.format_duration(3 * 3600 + 25 * 60) == "3h 25m"

    def test_exact_hours(self):
        assert nudges.format_duration(7200) == "2h"

    def test_negative_is_clamped(self):
        assert nudges.format_duration(-5) == "0s"
