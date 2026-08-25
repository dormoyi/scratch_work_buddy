import pytest

from focus_buddy.observations import Habit, Observation, parse_observation_line


class TestHabitResolution:
    def test_nail_biting_wins_over_face_touching(self):
        # Fingers at the mouth look like both; only the specific one should surface.
        obs = Observation(biting_nails=True, touching_face=True).resolved()
        assert obs.habits == (Habit.BITING_NAILS,)
        assert obs.touching_face is False

    def test_no_habits_when_working(self):
        assert Observation().habits == ()
        assert Observation().primary_habit is None

    def test_priority_order(self):
        obs = Observation(looking_at_phone=True, touching_face=True)
        assert obs.primary_habit is Habit.LOOKING_AT_PHONE

    def test_labels(self):
        assert Habit.BITING_NAILS.label == "nail-biting"
        assert Habit.LOOKING_AT_PHONE.label == "phone"


class TestLineParsing:
    def test_round_trip(self):
        original = Observation(touching_face=True, looking_at_screen=False, straight_posture=0.4)
        assert parse_observation_line(original.to_line()) == original

    def test_quoted_and_unquoted_keys(self):
        quoted = parse_observation_line('"touching_face": True | "looking_at_phone": False')
        bare = parse_observation_line("touching_face: true, looking_at_phone: false")
        assert quoted.touching_face is True
        assert bare.touching_face is True

    def test_returns_none_without_fields(self):
        # So the caller can fall back to caption classification.
        assert parse_observation_line("The person is sitting at a desk.") is None

    def test_all_true_is_treated_as_no_signal(self):
        # A known failure mode of small quantised checkpoints on field prompts.
        line = (
            '"looking_at_screen": True | "touching_face": True | "looking_at_phone": True | '
            '"biting_nails": True | "at_keyboard": True'
        )
        assert parse_observation_line(line).habits == ()

    @pytest.mark.parametrize("token,expected", [("yes", True), ("1", True), ("no", False)])
    def test_alternative_boolean_spellings(self, token, expected):
        assert parse_observation_line(f"touching_face: {token}").touching_face is expected

    def test_posture_none(self):
        assert (
            parse_observation_line(
                '"touching_face": False | "straight_posture": None'
            ).straight_posture
            is None
        )

    def test_posture_float(self):
        assert (
            parse_observation_line(
                '"touching_face": False | "straight_posture": 0.7'
            ).straight_posture
            == 0.7
        )
