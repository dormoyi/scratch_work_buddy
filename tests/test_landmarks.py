"""Contact geometry: the thresholds and the distances they are applied to.

The numbers in the fixtures below are the measured ratios from a labelled set of
Reachy Mini frames, so a regression in the geometry shows up as a habit that
stops being detected rather than as an abstract arithmetic failure.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from focus_buddy.observations import Observation
from focus_buddy.perception.landmarks import (
    EYES,
    FACE_CONTACT_MAX_RATIO,
    MOUTH_CONTACT_MAX_RATIO,
    HybridVision,
    LandmarkVision,
    child_environment,
    classify_contact,
    distance_to_box,
    distance_to_point,
    encode_frame,
    face_box,
    face_width,
    iris_offsets,
    looking_at_screen,
    mouth_centre,
    observation_from_payload,
    ratios_from_landmarks,
)

# (name, d_face/face_width, d_mouth/face_width) measured from real frames.
TOUCHING = [
    ("touch00", 0.000, 0.219),
    ("touch06", 0.015, 0.329),
]
# A finger resting beside the nose lands 0.090 face widths from the mouth, inside
# the nail-biting band. Recorded rather than fixed: separating "finger by the
# nose" from "finger at the mouth" needs more than one distance, and calling it
# nail-biting costs the user a differently-worded nudge, not a false one.
AMBIGUOUS_TOUCH = ("touch03", 0.000, 0.090)
BITING = [
    ("bite00", 0.000, 0.074),
    ("bite01", 0.000, 0.119),
    ("bite07", 0.000, 0.065),
]
CLEAN = [
    ("clean00", 0.539, 0.997),
    ("clean02", 0.567, 1.069),
    ("full", 2.102, 2.754),
    ("no-hands", float("inf"), float("inf")),
]


class TestFaceBox:
    def test_box_spans_the_landmarks(self):
        points = np.array([[10.0, 20.0], [50.0, 5.0], [30.0, 40.0]])
        assert face_box(points) == (10.0, 5.0, 50.0, 40.0)

    def test_width_is_never_zero(self):
        """A degenerate box would divide every ratio by zero."""
        single = np.array([[7.0, 7.0]])
        assert face_width(face_box(single)) > 0

    def test_mouth_centre_averages_the_lip_landmarks(self):
        points = np.zeros((478, 2))
        for index in (13, 14, 78, 308):
            points[index] = [100.0, 200.0]
        assert mouth_centre(points).tolist() == [100.0, 200.0]


class TestDistances:
    def test_point_inside_the_box_is_zero(self):
        box = (0.0, 0.0, 100.0, 100.0)
        assert distance_to_box(np.array([[50.0, 50.0]]), box) == 0.0

    def test_distance_is_measured_from_the_nearest_edge(self):
        box = (0.0, 0.0, 100.0, 100.0)
        assert distance_to_box(np.array([[130.0, 50.0]]), box) == pytest.approx(30.0)

    def test_the_closest_of_several_points_wins(self):
        box = (0.0, 0.0, 10.0, 10.0)
        points = np.array([[200.0, 200.0], [12.0, 5.0]])
        assert distance_to_box(points, box) == pytest.approx(2.0)

    def test_diagonal_distance_from_a_corner(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert distance_to_box(np.array([[13.0, 14.0]]), box) == pytest.approx(5.0)

    def test_distance_to_point_takes_the_nearest(self):
        points = np.array([[0.0, 0.0], [0.0, 2.0]])
        assert distance_to_point(points, np.array([0.0, 5.0])) == pytest.approx(3.0)


class TestClassifyContact:
    @pytest.mark.parametrize("name,face,mouth", TOUCHING)
    def test_real_face_touching_frames(self, name, face, mouth):
        assert classify_contact(face, mouth) == (True, False)

    @pytest.mark.parametrize("name,face,mouth", BITING)
    def test_real_nail_biting_frames(self, name, face, mouth):
        assert classify_contact(face, mouth) == (False, True)

    @pytest.mark.parametrize("name,face,mouth", CLEAN)
    def test_real_clean_frames(self, name, face, mouth):
        assert classify_contact(face, mouth) == (False, False)

    def test_nail_biting_suppresses_face_touching(self):
        """One gesture must never be reported as two habits."""
        touching, biting = classify_contact(0.0, 0.01)
        assert (touching, biting) == (False, True)

    def test_a_finger_beside_the_nose_reads_as_nail_biting(self):
        """Known limitation, pinned so a change in behaviour is deliberate."""
        _, face, mouth = AMBIGUOUS_TOUCH
        assert classify_contact(face, mouth) == (False, True)

    def test_no_contact_ignores_mouth_proximity(self):
        """A hand near the mouth but off the face is not a habit."""
        assert classify_contact(0.9, 0.05) == (False, False)

    def test_thresholds_sit_between_the_measured_classes(self):
        """The margin is the whole reason this beats a VLM; guard it."""
        assert max(f for _, f, _ in TOUCHING + BITING) < FACE_CONTACT_MAX_RATIO
        assert FACE_CONTACT_MAX_RATIO < min(f for _, f, _ in CLEAN)
        assert max(m for _, _, m in BITING) < MOUTH_CONTACT_MAX_RATIO
        assert MOUTH_CONTACT_MAX_RATIO < min(m for _, _, m in TOUCHING)


class TestRatiosFromLandmarks:
    """The reduction the parent applies to whatever the sidecar sends back."""

    @staticmethod
    def _face(size: float = 100.0) -> list[list[float]]:
        """A square face whose mouth landmarks sit at its centre."""
        points = [[0.0, 0.0], [size, size]] + [[size / 2, size / 2]] * 400
        return points

    def test_no_face_is_unknown(self):
        assert ratios_from_landmarks(None, []) is None
        assert ratios_from_landmarks([], []) is None

    def test_face_without_hands_is_infinitely_far(self):
        """A face but no hands is a clean frame, not an unknown one."""
        face_ratio, mouth_ratio = ratios_from_landmarks(self._face(), [])
        assert face_ratio == float("inf") and mouth_ratio == float("inf")
        assert classify_contact(face_ratio, mouth_ratio) == (False, False)

    def test_ratios_are_scale_invariant(self):
        """Sitting closer must not change the verdict."""
        small = ratios_from_landmarks(self._face(100.0), [[[50.0, 50.0]] * 21])
        large = ratios_from_landmarks(self._face(400.0), [[[200.0, 200.0]] * 21])
        assert small == pytest.approx(large)

    def test_the_nearest_of_two_hands_wins(self):
        far = [[900.0, 900.0]] * 21
        near = [[50.0, 50.0]] * 21
        both = ratios_from_landmarks(self._face(), [far, near])
        assert both == ratios_from_landmarks(self._face(), [near])


class TestChildEnvironment:
    """The sidecar exists to escape this process's numpy; the env must not undo that."""

    def test_python_path_is_stripped(self, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "/somewhere/site-packages")
        monkeypatch.setenv("PYTHONHOME", "/somewhere")
        monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/.venv")
        env = child_environment()
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            assert name not in env

    def test_everything_else_survives(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/someone")
        assert child_environment()["HOME"] == "/home/someone"


class TestEncodeFrame:
    def test_large_frames_are_downscaled(self):
        data = encode_frame(Image.new("RGB", (1920, 1080)), max_edge=640)
        assert max(Image.open(io.BytesIO(data)).size) == 640

    def test_small_frames_are_left_alone(self):
        data = encode_frame(Image.new("RGB", (320, 240)), max_edge=640)
        assert Image.open(io.BytesIO(data)).size == (320, 240)

    def test_output_is_decodable_jpeg(self):
        assert Image.open(io.BytesIO(encode_frame(Image.new("RGB", (64, 48))))).format == "JPEG"


class FakeSidecar:
    """Stands in for the subprocess, returning scripted payloads."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.restarts = 0
        self.closed = False

    def landmarks(self, jpeg):
        return self._payloads.pop(0) if self._payloads else None

    def restart(self):
        self.restarts += 1

    def close(self):
        self.closed = True


def _vision(payloads):
    """A LandmarkVision wired to a fake sidecar, with no subprocess involved."""
    vision = LandmarkVision.__new__(LandmarkVision)
    vision._sidecar = FakeSidecar(payloads)
    return vision


class TestLandmarkVisionBehaviour:
    FACE = [[0.0, 0.0], [100.0, 100.0]] + [[50.0, 50.0]] * 400

    def test_contact_becomes_a_habit(self):
        vision = _vision([{"ok": True, "face": self.FACE, "hands": [[[50.0, 50.0]] * 21]}])
        assert vision.classify(Image.new("RGB", (64, 48))).biting_nails is True

    def test_no_hands_reports_no_habit(self):
        vision = _vision([{"ok": True, "face": self.FACE, "hands": []}])
        observation = vision.classify(Image.new("RGB", (64, 48)))
        assert (observation.touching_face, observation.biting_nails) == (False, False)

    def test_a_silent_sidecar_is_restarted_and_reports_nothing(self):
        """A wedged helper must not take the focus loop down, nor invent a habit."""
        vision = _vision([None])
        observation = vision.classify(Image.new("RGB", (64, 48)))
        assert (observation.touching_face, observation.biting_nails) == (False, False)
        assert vision._sidecar.restarts == 1

    def test_a_sidecar_error_reports_nothing(self):
        vision = _vision([{"ok": False, "error": "boom"}])
        observation = vision.classify(Image.new("RGB", (64, 48)))
        assert (observation.touching_face, observation.biting_nails) == (False, False)


class StubBackend:
    def __init__(self, observation):
        self._observation = observation
        self.closed = False

    def classify(self, frame):
        return self._observation

    def close(self):
        self.closed = True


class TestHybridVision:
    FACE = [[0.0, 0.0], [100.0, 100.0]] + [[50.0, 50.0]] * 400

    def test_narrowing_leaves_the_other_fields_at_their_defaults(self):
        """A model narrowed to one field must not colour in the rest."""
        primary = StubBackend(
            Observation(
                looking_at_screen=False,
                looking_at_phone=True,
                at_keyboard=False,
                straight_posture=0.1,
            )
        )
        contact = _vision([{"ok": True, "face": self.FACE, "hands": []}])
        merged = HybridVision(contact, primary, from_primary=("looking_at_phone",)).classify(
            Image.new("RGB", (64, 48))
        )
        assert merged.looking_at_phone is True
        # The model's canned gaze and posture answers are discarded.
        assert merged.looking_at_screen is True
        assert merged.at_keyboard is True
        assert merged.straight_posture is None

    def test_the_model_is_not_consulted_when_it_contributes_nothing(self):
        primary = StubBackend(Observation(looking_at_phone=True))
        contact = _vision([{"ok": True, "face": self.FACE, "hands": []}])
        merged = HybridVision(contact, primary, from_primary=()).classify(
            Image.new("RGB", (64, 48))
        )
        assert merged.looking_at_phone is False

    def test_contact_fields_cannot_be_sourced_from_the_model(self):
        with pytest.raises(ValueError, match="geometry"):
            HybridVision(_vision([]), StubBackend(Observation()), from_primary=("touching_face",))

    def test_contact_comes_from_geometry_and_the_rest_from_the_model(self):
        primary = StubBackend(
            Observation(
                looking_at_screen=False,
                looking_at_phone=True,
                touching_face=True,
                straight_posture=0.4,
            )
        )
        contact = _vision([{"ok": True, "face": self.FACE, "hands": []}])
        merged = HybridVision(contact, primary).classify(Image.new("RGB", (64, 48)))
        # Geometry overrides the model's face-touch claim...
        assert merged.touching_face is False
        # ...while gaze, phone and posture survive untouched.
        assert merged.looking_at_phone is True
        assert merged.looking_at_screen is False
        assert merged.straight_posture == 0.4

    def test_close_closes_both(self):
        primary = StubBackend(Observation())
        contact = _vision([])
        HybridVision(contact, primary).close()
        assert primary.closed and contact._sidecar.closed


def _face_with_irises(iris_h: float, iris_v: float) -> list[list[float]]:
    """A synthetic mesh whose irises sit at the requested fraction of each eye."""
    points = [[50.0, 50.0] for _ in range(478)]
    points[0] = [0.0, 0.0]  # stretch the box so face_width is sane
    points[1] = [100.0, 100.0]
    for iris, inner, outer, top, bottom in EYES:
        # Spans deliberately run opposite ways for the two eyes, as they do in
        # image coordinates, so a sign error shows up here.
        inner_x, outer_x = (20.0, 40.0) if iris == 468 else (80.0, 60.0)
        top_y, bottom_y = 30.0, 40.0
        points[inner] = [inner_x, 35.0]
        points[outer] = [outer_x, 35.0]
        points[top] = [30.0, top_y]
        points[bottom] = [30.0, bottom_y]
        points[iris] = [
            inner_x + iris_h * (outer_x - inner_x),
            top_y + iris_v * (bottom_y - top_y),
        ]
    return points


class TestIrisOffsets:
    def test_centred_irises_read_as_centred(self):
        horizontal, vertical = iris_offsets(_face_with_irises(0.5, 0.5))
        assert horizontal == pytest.approx(0.5)
        assert vertical == pytest.approx(0.5)

    def test_offset_is_recovered_for_both_eye_directions(self):
        """The two eyes' spans run opposite ways; a sign slip shows up here."""
        horizontal, _ = iris_offsets(_face_with_irises(0.8, 0.5))
        assert horizontal == pytest.approx(0.8)

    def test_a_mesh_without_iris_landmarks_is_unknown(self):
        """refine_landmarks=False yields 468 points and no irises."""
        assert iris_offsets([[0.0, 0.0]] * 468) is None


class TestLookingAtScreen:
    # Measured: eight frames of steady screen work, eight of a head turned away.
    SCREEN = [
        (0.502, 0.451),
        (0.528, 0.452),
        (0.534, 0.460),
        (0.523, 0.370),
        (0.494, 0.557),
        (0.536, 0.445),
    ]
    TURNED = [
        (0.651, 0.616),
        (0.824, 0.277),
        (0.663, 0.180),
        (0.717, 0.276),
        (0.918, 0.226),
        (0.515, 0.795),
        (0.515, 0.299),
    ]

    @pytest.mark.parametrize("offsets", SCREEN)
    def test_real_screen_frames(self, offsets):
        assert looking_at_screen(offsets) is True

    @pytest.mark.parametrize("offsets", TURNED)
    def test_real_turned_frames(self, offsets):
        assert looking_at_screen(offsets) is False

    def test_unknown_gaze_assumes_the_ordinary_case(self):
        """Reporting distraction on missing data would inflate the summary."""
        assert looking_at_screen(None) is True

    def test_the_known_miss_is_pinned(self):
        """One turned frame sits inside the box; recorded, not hidden."""
        assert looking_at_screen((0.520, 0.440)) is True


class TestObservationFromPayload:
    FACE = _face_with_irises(0.5, 0.5)

    def test_no_face_and_no_body_is_an_empty_chair(self):
        observation = observation_from_payload({"ok": True, "face": None, "pose": None})
        assert observation.at_keyboard is False
        assert observation.looking_at_screen is False

    def test_no_face_but_a_body_is_a_head_turned_away(self):
        """FaceMesh loses the face on a strong turn; the body says you're still there."""
        observation = observation_from_payload(
            {"ok": True, "face": None, "pose": [[0.0, 0.0, 0.9]] * 33}
        )
        assert observation.at_keyboard is True
        assert observation.looking_at_screen is False

    def test_a_face_gives_gaze_and_contact_together(self):
        observation = observation_from_payload(
            {"ok": True, "face": self.FACE, "hands": [[[50.0, 50.0]] * 21], "pose": None}
        )
        assert observation.at_keyboard is True
        assert observation.looking_at_screen is True
        assert observation.biting_nails is True

    def test_posture_is_never_guessed(self):
        """Neck-over-shoulder-width moves when the body rotates, so it is unused."""
        observation = observation_from_payload({"ok": True, "face": self.FACE, "hands": []})
        assert observation.straight_posture is None
