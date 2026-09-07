r"""Landmark extractor that runs in its own interpreter.

MediaPipe cannot live in the same process as the rest of Focus Buddy. The legacy
``solutions`` graphs are the only ones that work on Apple Silicon -- the Tasks
API aborts in ``DrishtiMetalHelper`` -- and they require numpy < 2, while
reachy-mini and opencv-python both require numpy >= 2. The two constraints have
no overlap, so MediaPipe gets an interpreter of its own and speaks over a pipe.

This module is executed as a *file*, by a different Python than the rest of the
package, and must therefore import nothing from ``focus_buddy``. It deliberately
does no geometry either: it returns raw landmark coordinates and lets the parent
apply the thresholds, so there is exactly one tested copy of that logic.

Protocol, one exchange per frame:

    parent -> child   b"<byte-count>\\n" followed by that many bytes of JPEG
    child  -> parent  one line of JSON, then a flush

    {"ok": true,
     "face":  [[x, y], ...] | null,      # 478 points; 468+ are the irises
     "hands": [[[x, y], ...], ...],      # 21 points each, up to two hands
     "pose":  [[x, y, visibility], ...] | null,   # 33 points, for shoulders
     "size":  [width, height]}
    {"ok": false, "error": "..."}

Coordinates are in pixels. A null ``face`` means no face was found; an empty
``hands`` list means a face but no hands, which is a clean frame rather than an
unknown one.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any, BinaryIO


def _build() -> tuple[Any, Any, Any]:
    """Create the two MediaPipe graphs, or exit with a readable message."""
    try:
        import mediapipe as mp
    except ImportError:
        print(
            json.dumps({"ok": False, "error": "mediapipe is not installed in the sidecar venv"}),
            flush=True,
        )
        raise SystemExit(2) from None

    if not hasattr(mp, "solutions"):
        version = getattr(mp, "__version__", "?")
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"mediapipe {version} has no `solutions` API (removed between "
                        "0.10.21 and 0.10.30). Pin 'mediapipe>=0.10.14,<0.10.30'."
                    ),
                }
            ),
            flush=True,
        )
        raise SystemExit(2)

    # static_image_mode=True: frames arrive a second or more apart, so the video
    # tracking path would carry stale hand positions between them.
    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
    )
    # refine_landmarks=True appends the ten iris points (468-477). Indices below
    # 468 keep their meaning, so it costs a little time and breaks nothing.
    face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    )
    # Shoulders, for posture. Complexity 1 is the middle model: complexity 0
    # loses shoulder accuracy on a half-body desk framing.
    pose = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=1, min_detection_confidence=0.3
    )
    return hands, face, pose


def _read_exactly(stream: BinaryIO, count: int) -> bytes:
    """Read exactly ``count`` bytes, or return what arrived before EOF."""
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _to_pixels(landmarks: Any, width: int, height: int) -> list[list[float]]:
    """Convert normalised MediaPipe landmarks to pixel coordinate pairs."""
    return [[p.x * width, p.y * height] for p in landmarks.landmark]


def _to_pixels_with_visibility(landmarks: Any, width: int, height: int) -> list[list[float]]:
    """As above, keeping the visibility score pose landmarks carry."""
    return [[p.x * width, p.y * height, getattr(p, "visibility", 1.0)] for p in landmarks.landmark]


def main() -> int:
    """Serve landmark requests until stdin closes."""
    import numpy as np
    from PIL import Image

    hands, face, pose = _build()
    # Tell the parent the graphs are up before it sends a first frame.
    print(json.dumps({"ok": True, "ready": True}), flush=True)

    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return 0
        try:
            size = int(header.strip())
        except ValueError:
            print(json.dumps({"ok": False, "error": f"bad header {header!r}"}), flush=True)
            continue

        payload = _read_exactly(stdin, size)
        if len(payload) != size:
            return 0

        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            rgb = np.asarray(image)
            height, width = rgb.shape[:2]

            face_result = face.process(rgb)
            face_points = (
                _to_pixels(face_result.multi_face_landmarks[0], width, height)
                if face_result.multi_face_landmarks
                else None
            )

            hand_points: list[list[list[float]]] = []
            if face_points is not None:
                hand_result = hands.process(rgb)
                for hand in hand_result.multi_hand_landmarks or []:
                    hand_points.append(_to_pixels(hand, width, height))

            # Pose runs whether or not a face was found: a head turned far enough
            # loses the face mesh entirely, and the body is what distinguishes
            # "turned away, still at the desk" from "left the desk".
            pose_points: list[list[float]] | None = None
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                pose_points = _to_pixels_with_visibility(pose_result.pose_landmarks, width, height)

            response: dict[str, Any] = {
                "ok": True,
                "face": face_points,
                "hands": hand_points,
                "pose": pose_points,
                "size": [width, height],
            }
        except Exception as error:  # keep serving; one bad frame is not fatal
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    sys.exit(main())
