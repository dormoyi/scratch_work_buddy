"""Hand-to-face contact from landmark geometry rather than from a caption.

"Is a hand touching the face" is a distance question, and small VLMs answer it
badly. On a labelled set of sixteen Reachy Mini frames, every 2-3B checkpoint
tried scored at chance: SmolVLM2-2.2B, Qwen2-VL-2B and Qwen2.5-VL-3B each
answered "yes" on all sixteen, including the eight where the hands were nowhere
near the face. The same frames separate perfectly by distance -- 0.000-0.015
face widths while touching, 0.528-2.102 while not -- so contact is measured here
and the models are left to answer gaze, phone and posture, which they can do.

Distances are ratios of the face width, which makes every threshold independent
of how close the person sits and what resolution the camera runs at.

MediaPipe itself runs in a subprocess; see :mod:`._landmark_sidecar` for why.
"""

from __future__ import annotations

import io
import json
import logging
import os
import selectors
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..observations import Observation

logger = logging.getLogger(__name__)

# MediaPipe FaceMesh indices for the inner lips and the mouth corners. Their mean
# is a stable mouth centre that does not drift when the mouth opens.
MOUTH_LANDMARKS = (13, 14, 78, 308)
# MediaPipe Hands indices for the five fingertips.
FINGERTIP_LANDMARKS = (4, 8, 12, 16, 20)
# Iris centres, present only when the mesh is built with refine_landmarks=True.
RIGHT_IRIS, LEFT_IRIS = 468, 473
# (iris, inner corner, outer corner, upper lid, lower lid) per eye. The spans are
# signed and run opposite ways for the two eyes, which is why the code below
# divides by them rather than by their magnitude.
EYES = (
    (RIGHT_IRIS, 133, 33, 159, 145),
    (LEFT_IRIS, 362, 263, 386, 374),
)

# Measured on 16 labelled frames from a Reachy Mini camera: hand-to-face-box
# distance was 0.000-0.015 face widths while touching and 0.528-2.102 while not.
# 0.08 sits in that gap with an order of magnitude of margin on either side.
FACE_CONTACT_MAX_RATIO = 0.08
# Fingertip-to-mouth distance was 0.065-0.119 face widths across eight
# nail-biting frames, against 0.219+ for face-touching away from the mouth. A
# finger resting beside the nose falls in between and reads as nail-biting.
MOUTH_CONTACT_MAX_RATIO = 0.13

# Where the iris sits inside the eye while looking at the screen, horizontally
# and vertically. Measured over eight frames of steady screen work (h 0.494-0.536,
# v 0.370-0.557) and eight of a head turned away while seated (h up to 0.918,
# v 0.180-0.795), then widened so an ordinary glance does not read as distraction.
#
# Unlike the contact thresholds, these have no comfortable margin -- one turned
# frame sits inside the box. That is tolerable because gaze never triggers a
# nudge on its own; it only feeds the spoken daily summary.
IRIS_H_RANGE = (0.44, 0.60)
IRIS_V_RANGE = (0.32, 0.62)

# Longest edge sent to the sidecar. The thresholds are scale-invariant, so this
# only trades a little hand detail for encode, pipe and inference time.
SIDECAR_MAX_EDGE = 960
# Generous: covers graph construction on a cold start and a busy machine.
STARTUP_TIMEOUT_S = 60.0
# One frame's round trip. Measured ~35ms; anything near this means trouble.
FRAME_TIMEOUT_S = 15.0


def face_box(points: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of the face landmarks, as (x0, y0, x1, y1)."""
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def face_width(box: tuple[float, float, float, float]) -> float:
    """Width of a face box, used to normalise every distance below."""
    return max(box[2] - box[0], 1e-6)


def mouth_centre(points: np.ndarray) -> np.ndarray:
    """Mouth centre from the inner-lip and corner landmarks."""
    return points[list(MOUTH_LANDMARKS)].mean(axis=0)


def distance_to_box(points: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Shortest distance from any point to a box. Zero when a point is inside."""
    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - points[:, 0], points[:, 0] - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - points[:, 1], points[:, 1] - y1), 0.0)
    return float(np.hypot(dx, dy).min())


def distance_to_point(points: np.ndarray, target: np.ndarray) -> float:
    """Shortest distance from any point to a single target point."""
    return float(np.linalg.norm(points - target, axis=1).min())


def classify_contact(face_ratio: float, mouth_ratio: float) -> tuple[bool, bool]:
    """Turn two normalised distances into (touching_face, biting_nails).

    Mirrors :meth:`Observation.resolved`: fingers at the mouth are nail-biting,
    not face-touching, so the two are never reported together.
    """
    if face_ratio > FACE_CONTACT_MAX_RATIO:
        return False, False
    if mouth_ratio <= MOUTH_CONTACT_MAX_RATIO:
        return False, True
    return True, False


def ratios_from_landmarks(
    face_points: list[list[float]] | None, hand_points: list[list[list[float]]]
) -> tuple[float, float] | None:
    """Reduce one frame's raw landmarks to (face_ratio, mouth_ratio).

    Returns None when there is no face. Infinite ratios mean a face but no
    hands, which is a clean frame rather than an unknown one.
    """
    if not face_points:
        return None
    face = np.asarray(face_points, dtype=float)
    box = face_box(face)
    scale = face_width(box)
    mouth = mouth_centre(face)

    best_face = float("inf")
    best_mouth = float("inf")
    for hand in hand_points:
        points = np.asarray(hand, dtype=float)
        best_face = min(best_face, distance_to_box(points, box) / scale)
        fingertips = points[list(FINGERTIP_LANDMARKS)]
        best_mouth = min(best_mouth, distance_to_point(fingertips, mouth) / scale)
    return best_face, best_mouth


def iris_offsets(face_points: list[list[float]]) -> tuple[float, float] | None:
    """Mean iris position inside the eyes, as (horizontal, vertical) fractions.

    0.5 is centred. Returns None when the mesh carries no iris landmarks, which
    means it was built without refine_landmarks.
    """
    face = np.asarray(face_points, dtype=float)
    if len(face) <= LEFT_IRIS:
        return None

    horizontals: list[float] = []
    verticals: list[float] = []
    for iris, inner, outer, top, bottom in EYES:
        span_h = face[outer][0] - face[inner][0]
        span_v = face[bottom][1] - face[top][1]
        # A closed or near-profile eye collapses a span; skip it rather than
        # dividing by something arbitrarily small.
        if abs(span_h) > 1.0:
            horizontals.append((face[iris][0] - face[inner][0]) / span_h)
        if abs(span_v) > 1.0:
            verticals.append((face[iris][1] - face[top][1]) / span_v)
    if not horizontals or not verticals:
        return None
    return float(np.mean(horizontals)), float(np.mean(verticals))


def looking_at_screen(offsets: tuple[float, float] | None) -> bool:
    """Whether the eyes are pointed where the screen is.

    Unknown gaze counts as looking at the screen: that is the ordinary working
    state, and the field only colours the daily summary.
    """
    if offsets is None:
        return True
    horizontal, vertical = offsets
    return (
        IRIS_H_RANGE[0] <= horizontal <= IRIS_H_RANGE[1]
        and IRIS_V_RANGE[0] <= vertical <= IRIS_V_RANGE[1]
    )


def observation_from_payload(payload: dict[str, Any]) -> Observation:
    """Build an Observation from one sidecar response.

    Kept a plain function so every branch is testable without a subprocess.
    """
    face_points = payload.get("face")
    pose_points = payload.get("pose")

    if not face_points:
        # No face mesh at all. The body tells us whether that is a head turned
        # too far to mesh, or an empty chair.
        return Observation(
            looking_at_screen=False,
            at_keyboard=bool(pose_points),
            straight_posture=None,
        )

    ratios = ratios_from_landmarks(face_points, payload.get("hands") or [])
    touching_face, biting_nails = (
        classify_contact(*ratios) if ratios is not None else (False, False)
    )
    return Observation(
        looking_at_screen=looking_at_screen(iris_offsets(face_points)),
        touching_face=touching_face,
        biting_nails=biting_nails,
        at_keyboard=True,
        # Posture needs a scale that survives the body rotating; normalising the
        # neck length by shoulder width does not, so it is left unmeasured.
        straight_posture=None,
    ).resolved()


def default_sidecar_python() -> Path:
    """Where :mod:`tools.setup_landmark_sidecar` puts the helper interpreter."""
    return Path.home() / ".cache" / "focus_buddy" / "landmark-venv" / "bin" / "python"


def sidecar_script() -> Path:
    """Path to the sidecar module, run as a plain file by another interpreter."""
    return Path(__file__).with_name("_landmark_sidecar.py")


# Variables that would let the parent's interpreter leak into the child's import
# path. PYTHONPATH is the one that actually bites: importing reachy_mini puts this
# package's own site-packages on it for GStreamer, so an inherited environment
# silently shadows the sidecar venv with the very numpy and mediapipe it exists
# to avoid.
_INHERITED_PYTHON_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__")


def _child_environment() -> dict[str, str]:
    """Copy the parent environment, dropping anything that redirects imports."""
    return {k: v for k, v in os.environ.items() if k not in _INHERITED_PYTHON_VARS}


class SidecarError(RuntimeError):
    """The landmark subprocess could not be started or stopped responding."""


class _Sidecar:
    """A MediaPipe process, spoken to over a pipe.

    Owns only the transport. Everything it returns is raw landmark coordinates,
    so no threshold or geometry lives on the far side of the pipe.
    """

    def __init__(self, python_executable: Path, script: Path) -> None:
        """Start the helper interpreter and wait for it to report readiness."""
        self._python = python_executable
        self._script = script
        self._process: subprocess.Popen[bytes] | None = None
        self._start()

    def _start(self) -> None:
        if not self._python.exists():
            raise SidecarError(
                f"No landmark sidecar interpreter at {self._python}. Create it with:\n"
                "    python tools/setup_landmark_sidecar.py\n"
                "or point FOCUS_BUDDY_LANDMARK_PYTHON at an interpreter that has "
                "'mediapipe>=0.10.14,<0.10.30' installed."
            )
        logger.info("Starting landmark sidecar: %s", self._python)
        self._process = subprocess.Popen(
            [str(self._python), str(self._script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_child_environment(),
        )
        hello = self._read_line(STARTUP_TIMEOUT_S)
        if hello is None or not hello.get("ok"):
            detail = (hello or {}).get("error", "no response")
            self.close()
            raise SidecarError(f"Landmark sidecar failed to start: {detail}")
        logger.info("Landmark sidecar ready")

    def _read_line(self, timeout: float) -> dict[str, Any] | None:
        """Read one JSON line, or None on timeout, EOF or malformed output."""
        process = self._process
        if process is None or process.stdout is None:
            return None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(timeout):
                return None
            line = process.stdout.readline()
        finally:
            selector.close()
        if not line:
            return None
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Landmark sidecar wrote non-JSON: %r", line[:200])
            return None
        return parsed if isinstance(parsed, dict) else None

    def landmarks(self, jpeg: bytes) -> dict[str, Any] | None:
        """Send one encoded frame and return its landmark payload."""
        process = self._process
        if process is None or process.stdin is None:
            return None
        try:
            process.stdin.write(f"{len(jpeg)}\n".encode())
            process.stdin.write(jpeg)
            process.stdin.flush()
        except (BrokenPipeError, ValueError):
            return None
        return self._read_line(FRAME_TIMEOUT_S)

    def restart(self) -> None:
        """Replace a dead or wedged process with a fresh one."""
        self.close()
        self._start()

    def close(self) -> None:
        """Stop the process. Safe to call more than once."""
        process, self._process = self._process, None
        if process is None:
            return
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # pragma: no cover - best effort
                    pass
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:  # pragma: no cover - best effort
            try:
                process.kill()
            except Exception:
                pass


def encode_frame(frame: Image.Image, max_edge: int = SIDECAR_MAX_EDGE) -> bytes:
    """Downscale and JPEG-encode a frame for the pipe."""
    width, height = frame.size
    longest = max(width, height)
    if longest > max_edge:
        scale = max_edge / longest
        frame = frame.resize((int(width * scale), int(height * scale)))
    buffer = io.BytesIO()
    frame.convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class LandmarkVision:
    """Detect face-touching and nail-biting from hand and face landmarks.

    Answers only the contact habits: no amount of geometry can tell whether the
    thing in someone's hand is a phone. Compose with :class:`HybridVision` to
    keep gaze, phone and posture.
    """

    def __init__(self, python_executable: Path | None = None) -> None:
        """Start the sidecar. Raises SidecarError when it cannot be run."""
        self._sidecar = _Sidecar(python_executable or default_sidecar_python(), sidecar_script())

    def measure(self, frame: Image.Image) -> tuple[float, float] | None:
        """Return (face_ratio, mouth_ratio) for a frame, or None without a face.

        Kept alongside :meth:`classify` so thresholds can be re-derived from
        recorded frames without going through an Observation.
        """
        payload = self._request(frame)
        if payload is None:
            return None
        return ratios_from_landmarks(payload.get("face"), payload.get("hands") or [])

    def _request(self, frame: Image.Image) -> dict[str, Any] | None:
        """One round trip to the sidecar, restarting it if it stopped answering."""
        payload = self._sidecar.landmarks(encode_frame(frame))
        if payload is None:
            # A timeout or a dead process. One restart, then give up on the
            # frame: a wedged helper must not take the focus loop down with it.
            logger.warning("Landmark sidecar did not answer; restarting it")
            try:
                self._sidecar.restart()
            except SidecarError:
                logger.exception("Landmark sidecar could not be restarted")
            return None
        if not payload.get("ok"):
            logger.warning("Landmark sidecar error: %s", payload.get("error"))
            return None
        return payload

    def classify(self, frame: Image.Image) -> Observation:
        """Measure contact, gaze and presence for one frame."""
        try:
            payload = self._request(frame)
        except Exception:
            logger.exception("Landmark measurement failed; skipping this frame")
            return Observation()

        if payload is None:
            # No answer. Report nothing rather than guessing: a false nudge is
            # worse than a missed one.
            return Observation()

        observation = observation_from_payload(payload)
        logger.debug("landmarks: %s", observation)
        return observation

    def close(self) -> None:
        """Stop the sidecar. Safe to call more than once."""
        self._sidecar.close()


# Everything geometry cannot answer, and therefore everything a model may be
# asked to fill in. Narrow it per backend rather than trusting a model wholesale.
MODEL_ONLY_FIELDS = ("looking_at_screen", "looking_at_phone", "at_keyboard", "straight_posture")


class HybridVision:
    """Landmark geometry for contact habits, a model for named fields only.

    The split is by what each is good at: geometry cannot recognise a phone, and
    the small models cannot measure whether a hand reaches a face. ``from_primary``
    narrows what the model is allowed to contribute -- fields left out of it keep
    their :class:`Observation` defaults rather than taking a model's guess.
    """

    def __init__(
        self,
        contact: LandmarkVision,
        primary: Any,
        from_primary: tuple[str, ...] = MODEL_ONLY_FIELDS,
    ) -> None:
        """Wrap a landmark backend and the model answering ``from_primary``."""
        unknown = set(from_primary) - set(MODEL_ONLY_FIELDS)
        if unknown:
            raise ValueError(f"contact habits come from geometry, not the model: {sorted(unknown)}")
        self._contact = contact
        self._primary = primary
        self._from_primary = from_primary

    def classify(self, frame: Image.Image) -> Observation:
        """Measure contact, then overlay only the fields the model is trusted for."""
        contact = self._contact.classify(frame)
        if not self._from_primary:
            return contact
        primary = self._primary.classify(frame)
        overlay = {field: getattr(primary, field) for field in self._from_primary}
        return replace(contact, **overlay).resolved()

    def close(self) -> None:
        """Close both backends, reporting failures without masking them."""
        for name, backend in (("contact", self._contact), ("primary", self._primary)):
            closer = getattr(backend, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.warning("Error while closing the %s backend", name, exc_info=True)


def _cli() -> int:  # pragma: no cover - developer helper
    """Print measured ratios for image paths, for re-deriving thresholds."""
    vision = LandmarkVision()
    try:
        for path in sys.argv[1:]:
            measured = vision.measure(Image.open(path))
            if measured is None:
                print(f"{path}: no face")
            else:
                print(f"{path}: d_face={measured[0]:.3f} d_mouth={measured[1]:.3f}")
    finally:
        vision.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - developer helper
    sys.exit(_cli())
