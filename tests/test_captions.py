"""Caption-classification regression tests.

Each case is something a real VLM actually said. Keep them: they are the only
thing stopping a regex tweak from silently breaking detection.
"""

import pytest

from focus_buddy.observations import Habit
from focus_buddy.perception.captions import classify_caption


class TestFaceTouch:
    def test_hands_not_visible_is_not_a_touch(self):
        obs = classify_caption(
            "The person's hands are not visible, and they are looking at the camera."
        )
        assert obs.touching_face is False
        assert obs.looking_at_phone is False
        assert obs.looking_at_screen is True

    def test_scratching_cheek(self):
        obs = classify_caption(
            "Her right hand is scratching her cheek while she looks at the screen."
        )
        assert obs.touching_face is True
        assert obs.looking_at_screen is True

    def test_hands_absent_but_touching_still_counts(self):
        # Qwen says this constantly on low desk angles: a visibility hedge followed
        # by a real detection in the same sentence.
        obs = classify_caption(
            "The person is looking at the camera. The hands are not visible, but there "
            "is a hand touching the face. The person is sitting at a desk with a laptop."
        )
        assert obs.touching_face is True

    def test_hand_on_jaw(self):
        obs = classify_caption("A palm is against the jaw and fingers are across the cheek.")
        assert obs.touching_face is True

    def test_touching_chin_with_finger(self):
        obs = classify_caption(
            "There is no hand or forearm near the head. The person is touching "
            "their chin with their finger."
        )
        assert obs.touching_face is True

    def test_touching_the_cheek(self):
        obs = classify_caption("There is a hand near the head, touching the cheek.")
        assert obs.touching_face is True

    def test_later_positive_beats_earlier_denial(self):
        obs = classify_caption(
            "There is no hand touching the face, cheek, or hair. "
            "The person is resting their head on their hand."
        )
        assert obs.touching_face is True

    def test_chin_on_sleeve_counts(self):
        obs = classify_caption("The hands are not visible. The chin is resting on a sleeve.")
        assert obs.touching_face is True

    def test_prompt_echo_denial_is_not_a_touch(self):
        # The model parrots the prompt's noun list under a leading "no". A short
        # fixed-width negation lookback misses that "no" and false-positives.
        obs = classify_caption(
            "Sentence 1: There is no visible hand, fingers, fist, forearm, wrist, "
            "or sleeve under the chin/cheek. The person is not touching anything.\n"
            "Sentence 2: The person is looking at the screen."
        )
        assert obs.touching_face is False

    def test_negated_touch_stays_false(self):
        obs = classify_caption(
            "There is no hand touching the face, cheek, or hair. "
            "The person is looking away from the camera."
        )
        assert obs.touching_face is False

    def test_bare_yes_is_not_a_detection(self):
        # 2B models often answer "Yes" and stop. That is not evidence of anything.
        obs = classify_caption("Yes")
        assert obs.touching_face is False

    @pytest.mark.parametrize(
        "caption",
        [
            "The person is propping their chin up.",
            "She is leaning on her forearm.",
            "He is touching the side of his face.",
        ],
    )
    def test_patterns_that_the_fstring_bug_disabled(self, caption):
        # These three alternatives used `.{0,24}`-style quantifiers inside an
        # f-string, so they compiled to a literal "(0, 24)" and never matched.
        assert classify_caption(caption).touching_face is True


class TestNailBiting:
    def test_finger_at_mouth(self):
        obs = classify_caption(
            "Yes, there is a finger at the person's mouth. They are looking at the camera."
        )
        assert obs.biting_nails is True
        # One gesture must not report as two habits.
        assert obs.touching_face is False
        assert obs.habits == (Habit.BITING_NAILS,)

    def test_negated_nail_biting(self):
        obs = classify_caption("There is no indication of nail-biting in the photo.")
        assert obs.biting_nails is False


class TestPhoneAndGaze:
    def test_phone_in_hands(self):
        obs = classify_caption("She is holding a phone in both hands and looking at it.")
        assert obs.looking_at_phone is True
        assert obs.looking_at_screen is False

    def test_looking_at_camera_counts_as_screen(self):
        # The webcam sits on top of the monitor, so eye contact with it is screen work.
        obs = classify_caption(
            "The person is looking at the camera. There is no hand touching the face. "
            "The person is not looking at the screen."
        )
        assert obs.looking_at_screen is True

    def test_unclear_gaze_defaults_to_working(self):
        obs = classify_caption("A person sits at a desk.")
        assert obs.looking_at_screen is True

    def test_empty_chair_is_not_at_keyboard(self):
        obs = classify_caption("The chair is empty and nobody is at the desk.")
        assert obs.at_keyboard is False


class TestLabelledShorthand:
    def test_single_line(self):
        obs = classify_caption("Face: Yes | Nails: No | Looking: screen")
        assert obs.touching_face is True
        assert obs.biting_nails is False
        assert obs.looking_at_screen is True

    def test_face_only(self):
        obs = classify_caption("Face: Yes\nNails: No\nLooking: screen")
        assert obs.touching_face is True
        assert obs.biting_nails is False

    def test_nails_only(self):
        obs = classify_caption("Face: No\nNails: Yes\nLooking: screen")
        assert obs.biting_nails is True
        assert obs.touching_face is False

    def test_looking_away_label(self):
        obs = classify_caption("Face: No | Nails: No | Looking: away")
        assert obs.looking_at_screen is False
