"""Cropping a webcam frame down to what the vision model should actually look at.

Small local VLMs degrade badly on a wide desk shot: the person occupies a few
percent of the pixels and hands blur into the background. Finding the face and
cropping around it, with extra room in the direction the head is turned, is what
makes a 2B model usable for habit detection.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Which way the head is turned, in image coordinates.
LOOK_LEFT = -1
LOOK_CENTRE = 0
LOOK_RIGHT = 1

_MIN_CROP_PX = 64
_MIN_FACE_PX = 48


def fallback_roi(width: int, height: int) -> tuple[int, int, int, int]:
    """Upper-centre desk framing, used when no face is detected."""
    return (
        int(width * 0.18),
        int(height * 0.10),
        int(width * 0.88),
        int(height * 0.88),
    )


def detect_face(rgb: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """Locate the most prominent face.

    Uses OpenCV Haar cascades rather than MediaPipe Tasks, which segfaults on some
    Apple Silicon setups.

    Returns (x0, y0, x1, y1, look_dir) or None. ``look_dir`` is LOOK_LEFT /
    LOOK_CENTRE / LOOK_RIGHT and acts as a cheap yaw proxy: the profile cascade is
    trained on faces turned image-right, so a hit on the mirrored image means the
    head is turned image-left.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    # (x0, y0, x1, y1, area, look_dir, is_profile)
    boxes: list[tuple[int, int, int, int, int, int, bool]] = []

    def scan(cascade: cv2.CascadeClassifier, flip: bool, look_dir: int, is_profile: bool) -> None:
        image = cv2.flip(gray, 1) if flip else gray
        for x, y, w, h in cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=4, minSize=(_MIN_FACE_PX, _MIN_FACE_PX)
        ):
            if flip:
                x = width - x - w
            boxes.append((int(x), int(y), int(x + w), int(y + h), int(w * h), look_dir, is_profile))

    for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue
        scan(cascade, flip=False, look_dir=LOOK_CENTRE, is_profile=False)
        scan(cascade, flip=True, look_dir=LOOK_CENTRE, is_profile=False)

    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    if not profile.empty():
        scan(profile, flip=False, look_dir=LOOK_RIGHT, is_profile=True)
        scan(profile, flip=True, look_dir=LOOK_LEFT, is_profile=True)

    if not boxes:
        return None

    # Prefer a profile hit when it is nearly as large as the best frontal one: the
    # yaw signal it carries is worth more than a slightly tighter box.
    best_any = max(boxes, key=lambda b: b[4])
    best_profile = max((b for b in boxes if b[6]), key=lambda b: b[4], default=None)
    chosen = (
        best_profile
        if best_profile is not None and best_profile[4] >= 0.55 * best_any[4]
        else best_any
    )
    x0, y0, x1, y1, _, look_dir, _ = chosen
    return x0, y0, x1, y1, look_dir


def crop_box_around_face(
    width: int,
    height: int,
    face: tuple[int, int, int, int],
    look_dir: int = LOOK_CENTRE,
    pad: float = 1.75,
) -> tuple[int, int, int, int]:
    """Build a crop box around a face box, biased downward and toward the gaze.

    Downward because hands, desk and phone live below the face; toward the gaze
    because that is where a phone being looked at will be.
    """
    x0, y0, x1, y1 = face
    face_w, face_h = x1 - x0, y1 - y0
    centre_x = (x0 + x1) / 2.0
    centre_y = (y0 + y1) / 2.0 + 0.28 * face_h

    half = max(face_w, face_h) * (1.0 + pad) / 2.0
    lead = 0.55 * max(face_w, face_h)
    if look_dir == LOOK_LEFT:
        left_extra, right_extra = lead, 0.12 * lead
    elif look_dir == LOOK_RIGHT:
        left_extra, right_extra = 0.12 * lead, lead
    else:
        left_extra = right_extra = 0.35 * lead

    return (
        int(max(0, centre_x - half - left_extra)),
        int(max(0, centre_y - half * 0.85)),
        int(min(width, centre_x + half + right_extra)),
        int(min(height, centre_y + half * 1.15)),
    )


def crop_to_subject(frame: Image.Image, pad: float = 1.75) -> tuple[Image.Image, bool, int]:
    """Crop a frame around the subject.

    Returns (cropped_image, face_was_found, look_dir).
    """
    rgb = np.asarray(frame.convert("RGB"))
    height, width = rgb.shape[:2]

    face = detect_face(rgb)
    if face is None:
        return frame.crop(fallback_roi(width, height)), False, LOOK_CENTRE

    look_dir = face[4]
    box = crop_box_around_face(width, height, face[:4], look_dir=look_dir, pad=pad)
    if box[2] - box[0] < _MIN_CROP_PX or box[3] - box[1] < _MIN_CROP_PX:
        return frame.crop(fallback_roi(width, height)), False, LOOK_CENTRE
    return frame.crop(box), True, look_dir


def downscale(frame: Image.Image, max_edge: int = 640) -> Image.Image:
    """Cap the longest edge, for inference speed and memory headroom."""
    width, height = frame.size
    longest = max(width, height)
    if longest <= max_edge:
        return frame
    scale = max_edge / longest
    return frame.resize((int(width * scale), int(height * scale)))


def prepare_for_edge(
    frame: Image.Image, max_edge: int = 640, pad: float = 1.75
) -> tuple[Image.Image, bool, int]:
    """Crop then downscale a frame for a local VLM."""
    cropped, found_face, look_dir = crop_to_subject(frame, pad=pad)
    return downscale(cropped, max_edge=max_edge), found_face, look_dir
