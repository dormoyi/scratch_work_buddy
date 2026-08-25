import base64
import io
import os
import re
import tempfile

from dotenv import load_dotenv

load_dotenv()


VLM_PROMPT = """
You are a vision classifier for a focus assistant. The image comes from a webcam mounted at the
top of the user's screen, so the camera sees the user from the screen's point of view.

Classify each field using these definitions:
- looking_at_screen: because the camera sits at the top of the screen, a user looking at the
  screen appears to look at the camera or BELOW it. A downward gaze with the face pointed at the
  camera is the normal appearance of screen-reading: classify it True. Classify False only when
  the head is clearly turned to the side, or the user is looking at a phone in their hands.
- touching_face: a hand is in contact with the face anywhere except the mouth (cheek, chin,
  forehead, eyes, hair).
- looking_at_phone: the user is holding a phone and their gaze is directed at it.
- biting_nails: fingertips are at or in the mouth. If fingers are at the mouth, prefer
  biting_nails=True over touching_face.
- at_keyboard: the user is present and seated at their desk. False only if the seat is empty or
  the user is clearly walking away.
- straight_posture: a score from 0 to 1. 1 means sitting fully upright, 0.5 means noticeably
  hunched toward the screen, 0 means fully slouched or collapsed over the desk. Scoring posture
  requires seeing the chest and both shoulders; if the frame is cropped tighter than that (head
  and hair only, or one shoulder), answer None instead of guessing.

Judge each field independently from what is visible. If a field is genuinely ambiguous, choose the
most common working state (looking_at_screen=True, others False).

Respond with exactly one line in exactly this format, with no extra text before or after:
"looking_at_screen": <True or False> | "touching_face": <True or False> | "looking_at_phone": <True or False> | "biting_nails": <True or False> | "at_keyboard": <True or False> | "straight_posture": <number between 0 and 1, or None>
"""

EDGE_FIELDS = [
    "looking_at_screen",
    "touching_face",
    "looking_at_phone",
    "biting_nails",
    "at_keyboard",
]

# Matches both "looking_at_screen": True and looking_at_screen: True
_FIELD_RE = re.compile(
    r'["\']?(looking_at_screen|touching_face|looking_at_phone|biting_nails|at_keyboard'
    r'|straight_posture)["\']?\s*:\s*([^\s|]+)',
    re.IGNORECASE,
)

# Prefer LOCAL_VISION_MODEL=mlx-community/Qwen2-VL-2B-Instruct-4bit for edge.
# Ask about face contact first — "hands not visible" is a stuck prior when a sleeve hides fingers.
VLM_PROMPT_EDGE = """
Webcam close-up. Reply in exactly two short sentences.
Sentence 1 (required): is anything supporting or touching the face — hand, fingers, fist,
forearm, wrist, or sleeve under the chin/cheek? Or finger at/in the mouth? Or nothing touching?
Do not say "hands not visible" if a sleeve or arm is under the chin.
Sentence 2: where is the person looking — screen, camera, phone, or away?
"""


def _parse_bool(token):
    token = token.strip().strip(",.").lower()
    if token in ("true", "yes", "1"):
        return True
    if token in ("false", "no", "0"):
        return False
    return None


def _format_vision_line(
    looking_at_screen,
    touching_face,
    looking_at_phone,
    biting_nails,
    at_keyboard=True,
    straight_posture=None,
):
    return (
        f'"looking_at_screen": {looking_at_screen} | '
        f'"touching_face": {touching_face} | '
        f'"looking_at_phone": {looking_at_phone} | '
        f'"biting_nails": {biting_nails} | '
        f'"at_keyboard": {at_keyboard} | '
        f'"straight_posture": {straight_posture}'
    )


def classify_from_description(description):
    # Map a free-form VLM caption onto the shared detection line.
    d = description.lower()

    def negated(span_start):
        # Whole current sentence only — a 40-char lookback misses "no … or sleeve under
        # the chin" when the model echoes the long prompt noun list.
        chunk = d[:span_start]
        last_stop = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
        window_start = last_stop + 1 if last_stop >= 0 else 0
        window = d[window_start:span_start]
        # "hands are not visible, but …" — ignore visibility-only negations so a later
        # positive in the same sentence can still fire. Keep "no visible X under chin"
        # as a real denial ("no" stays).
        window = re.sub(
            r"\b(not|n't|never)\s+(visible|in\s+(the\s+)?frame|shown|seen)\b",
            " ",
            window,
        )
        return bool(re.search(r"\b(no|not|n't|without|never)\b", window))

    # Structured edge labels (preferred when the model follows the prompt).
    labeled_face = re.search(r"\bface(?:-?\s*touch)?\s*:\s*(yes|no)\b", d)
    labeled_nails = re.search(
        r"\b(?:nails?|nail-?\s*bite|nailbite)\s*:\s*(yes|no)\b", d
    )

    biting_nails = False
    if labeled_nails:
        biting_nails = labeled_nails.group(1) == "yes"
    else:
        for match in re.finditer(
            r"biting|nails?.{0,24}mouth|mouth.{0,24}(finger|nail|hand)|"
            r"fingers?\s+(are\s+)?(at|in|inside|near|touching)\s+(the\s+)?mouth|"
            r"finger\s+at\s+(the\s+)?(person'?s\s+)?mouth|"
            r"hand\s+at\s+(the\s+)?mouth",
            d,
        ):
            if not negated(match.start()):
                biting_nails = True
                break

    # Positive contact language wins even if an earlier sentence said "no hand touching".
    # Include inverse phrasing: "resting their head on their hand".
    # Do NOT treat a bare "Yes" as face-touch — the 2B model often stops there.
    face_parts = r"(face|cheek|chin|jaw|forehead|hair|temple)"
    touching_face = False
    if not biting_nails:
        if labeled_face:
            touching_face = labeled_face.group(1) == "yes"
        if not touching_face:
            for match in re.finditer(
                rf"scratch|"
                rf"touch(ing)?\s+(her\s+|his\s+|their\s+|the\s+)?{face_parts}|"
                rf"hand\s+(is\s+)?(touching|on|against|resting on)\s+"
                rf"(her\s+|his\s+|their\s+|the\s+)?{face_parts}|"
                rf"palm\s+(is\s+)?(on|against|under)\s+(her\s+|his\s+|their\s+|the\s+)?"
                rf"{face_parts}|"
                rf"on\s+(her\s+|his\s+|their\s+|the\s+)?{face_parts}|"
                rf"fingers?\s+(are\s+)?(on|across|against|at)\s+"
                rf"(her\s+|his\s+|their\s+|the\s+)?{face_parts}|"
                rf"fingers?\s+in\s+(her\s+|his\s+|their\s+|the\s+)?hair|"
                rf"resting\s+on\s+(her\s+|his\s+|their\s+|the\s+)?{face_parts}|"
                rf"covering\s+(her\s+|his\s+|their\s+|the\s+)?(face|mouth|eyes)|"
                rf"hand\s+to\s+(her\s+|his\s+|their\s+|the\s+)?{face_parts}|"
                rf"touching\s+.{0,24}{face_parts}|"
                rf"(head|chin|jaw|cheek|face)\s+on\s+(her\s+|his\s+|their\s+|a\s+|the\s+)?hand|"
                rf"resting\s+(her\s+|his\s+|their\s+|the\s+)?(head|chin)\s+on\s+"
                rf"(her\s+|his\s+|their\s+|a\s+|the\s+)?(hand|arm|forearm|wrist|sleeve)|"
                rf"(head|chin|jaw|cheek|face|lip)\s+(is\s+)?(resting\s+)?on\s+"
                rf"(her\s+|his\s+|their\s+|a\s+|the\s+)?(hand|arm|forearm|wrist|sleeve|fist)|"
                rf"leaning\s+.{0,24}(on|into)\s+(her\s+|his\s+|their\s+|a\s+|the\s+)?"
                rf"(hand|arm|forearm|sleeve)|"
                rf"hand\s+under\s+(her\s+|his\s+|their\s+|the\s+)?(chin|cheek|jaw|face)|"
                rf"(arm|forearm|sleeve)\s+under\s+(her\s+|his\s+|their\s+|the\s+)?"
                rf"(chin|cheek|jaw|face)|"
                rf"propping\s+.{0,16}(chin|face|cheek)|"
                rf"supported\s+by\s+(her\s+|his\s+|their\s+|a\s+|the\s+)?(hand|arm|sleeve)",
                d,
            ):
                if not negated(match.start()):
                    touching_face = True
                    break

    looking_at_phone = False
    for match in re.finditer(r"\bphone\b", d):
        if not negated(match.start()):
            looking_at_phone = True
            break

    looking_away = bool(
        re.search(
            r"looking away|turned (to the )?side|head turned|looking off|"
            r"looking elsewhere",
            d,
        )
    )
    # Desk cam: "looking away from the camera" often still means screen work.
    looking_away_from_camera_only = bool(
        re.search(r"looking away from (the )?camera", d)
    )
    # Webcam at top of screen: "looking at the camera" IS screen work.
    looking_at_camera = bool(
        re.search(r"looking at (the )?camera", d)
    ) and not bool(re.search(r"not looking at (the )?camera", d))
    # Structured "Looking: screen" line from the edge prompt.
    labeled_looking = re.search(
        r"\blooking\s*:\s*(screen|camera|phone|away)\b", d
    )
    explicit_screen = looking_at_phone is False and (
        (labeled_looking is not None and labeled_looking.group(1) == "screen")
        or bool(
            re.search(
                r"looking (at|toward|towards|down).{0,20}(screen|monitor|laptop)|"
                r"looking at (the )?screen|"
                r"seated in front of a laptop|"
                r"in front of (a |the )?(screen|monitor|laptop)",
                d,
            )
        )
    ) and not bool(re.search(r"not looking at (the )?(screen|monitor)", d))
    if labeled_looking and labeled_looking.group(1) == "phone":
        looking_at_phone = True
        explicit_screen = False
    looking_at_screen = looking_at_camera or explicit_screen
    # Default to screen only when gaze is unclear (not when truly looking away).
    truly_away = looking_away and not looking_away_from_camera_only and not explicit_screen
    if labeled_looking and labeled_looking.group(1) == "away":
        truly_away = True
        looking_at_screen = False
    if not truly_away and not looking_at_phone and not looking_at_screen:
        looking_at_screen = True

    at_keyboard = not bool(
        re.search(r"empty (chair|seat)|walking away|stood up|not at (the )?desk", d)
    )

    return _format_vision_line(
        looking_at_screen=looking_at_screen,
        touching_face=touching_face,
        looking_at_phone=looking_at_phone,
        biting_nails=biting_nails,
        at_keyboard=at_keyboard,
        straight_posture=None,
    )


def parse_edge_answers(raw_text):
    # If the model already emitted a field line, normalize it; else treat as a caption.
    values = {}
    for match in _FIELD_RE.finditer(raw_text):
        field, token = match.group(1), match.group(2)
        if field == "straight_posture":
            token_l = token.strip().strip(",.").lower()
            if token_l in ("none", "null"):
                values[field] = None
            else:
                try:
                    values[field] = float(token_l)
                except ValueError:
                    values[field] = None
        else:
            parsed = _parse_bool(token)
            if parsed is not None:
                values[field] = parsed

    bool_values = [values.get(field) for field in EDGE_FIELDS]
    # This 4-bit checkpoint's failure mode on format prompts is "everything True".
    if bool_values and all(v is True for v in bool_values):
        return _format_vision_line(True, False, False, False, True, None)

    if any(v is not None for v in bool_values):
        parts = [f'"{field}": {values.get(field, False)}' for field in EDGE_FIELDS]
        parts.append(f'"straight_posture": {values.get("straight_posture", None)}')
        return " | ".join(parts)

    return classify_from_description(raw_text)


def _face_roi_box(w, h):
    # Fallback when no face is detected: slightly wider upper-center desk framing.
    return (
        int(w * 0.18),
        int(h * 0.10),
        int(w * 0.88),
        int(h * 0.88),
    )


def _detect_face_xyxy(rgb_array):
    # OpenCV Haar (MediaPipe Tasks crashes on some Apple Silicon setups).
    # Returns (x0, y0, x1, y1, look_dir) where look_dir is -1 / 0 / +1 in image
    # coords: which way the head is turned (proxy for gaze / phone side).
    # Profile cascade is trained on faces looking image-right; flipped detect → left.
    import cv2

    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    frontal_names = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
    )
    boxes = []  # (x0, y0, x1, y1, area, look_dir, is_profile)

    for name in frontal_names:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue
        for flip in (False, True):
            img = cv2.flip(gray, 1) if flip else gray
            faces = cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)
            )
            for x, y, fw, fh in faces:
                if flip:
                    x = w - x - fw
                boxes.append(
                    (int(x), int(y), int(x + fw), int(y + fh), fw * fh, 0, False)
                )

    profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )
    if not profile.empty():
        for flip, look_dir in ((False, 1), (True, -1)):
            img = cv2.flip(gray, 1) if flip else gray
            faces = profile.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)
            )
            for x, y, fw, fh in faces:
                if flip:
                    x = w - x - fw
                boxes.append(
                    (
                        int(x),
                        int(y),
                        int(x + fw),
                        int(y + fh),
                        fw * fh,
                        look_dir,
                        True,
                    )
                )

    if not boxes:
        return None

    # Prefer a clear profile hit (yaw signal) when area is competitive; else largest.
    best_profile = max(
        (b for b in boxes if b[6]), key=lambda b: b[4], default=None
    )
    best_any = max(boxes, key=lambda b: b[4])
    if best_profile is not None and best_profile[4] >= 0.55 * best_any[4]:
        chosen = best_profile
    else:
        chosen = best_any
    x0, y0, x1, y1, _, look_dir, _ = chosen
    return x0, y0, x1, y1, look_dir


def _asymmetric_face_crop_box(w, h, box, look_dir=0, pad=1.75):
    # Wider than the old tight head crop; shift extra room toward look_dir and down
    # so a phone in the hands stays inside the VLM frame.
    x0, y0, x1, y1 = box[:4]
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    cy = cy + 0.28 * bh  # desk / hands / phone usually below the face
    side = max(bw, bh) * (1.0 + pad)
    half = side / 2.0
    # Extra lead room in the look direction (~0.55× face box); slight opposite room.
    lead = 0.55 * max(bw, bh)
    left_extra = lead if look_dir < 0 else (0.12 * lead if look_dir > 0 else 0.35 * lead)
    right_extra = lead if look_dir > 0 else (0.12 * lead if look_dir < 0 else 0.35 * lead)
    left = int(max(0, cx - half - left_extra))
    right = int(min(w, cx + half + right_extra))
    top = int(max(0, cy - half * 0.85))
    bottom = int(min(h, cy + half * 1.15))
    return left, top, right, bottom


def _crop_around_face(camera_frame, pad=1.75):
    # pad=1.75 → looser framing so phone / desk context survives the face crop.
    import numpy as np

    rgb = np.asarray(camera_frame.convert("RGB"))
    h, w = rgb.shape[:2]
    detected = _detect_face_xyxy(rgb)
    if detected is None:
        left, top, right, bottom = _face_roi_box(w, h)
        return camera_frame.crop((left, top, right, bottom)), False, 0

    look_dir = detected[4]
    left, top, right, bottom = _asymmetric_face_crop_box(
        w, h, detected, look_dir=look_dir, pad=pad
    )
    if right - left < 64 or bottom - top < 64:
        left, top, right, bottom = _face_roi_box(w, h)
        return camera_frame.crop((left, top, right, bottom)), False, 0
    return camera_frame.crop((left, top, right, bottom)), True, look_dir


def _shrink_for_edge(camera_frame, max_edge=640):
    # Cap for speed/RAM with Reachy GStreamer + VLM on 8GB.
    w, h = camera_frame.size
    longest = max(w, h)
    if longest <= max_edge:
        return camera_frame
    scale = max_edge / longest
    return camera_frame.resize((int(w * scale), int(h * scale)))


def _prepare_edge_frame(camera_frame, max_edge=640):
    cropped, found_face, look_dir = _crop_around_face(camera_frame)
    shrunk = _shrink_for_edge(cropped, max_edge=max_edge)
    return shrunk, found_face, look_dir


class VLM_EDGE:
    # On Apple Silicon, bitsandbytes 4-bit is CUDA-only, so we load the MLX 4-bit convert
    # (~1.5GB) instead of the full bf16 2.2B checkpoint that thrashes an 8GB Mac.
    def __init__(self):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model_id = os.getenv(
            "LOCAL_VISION_MODEL",
            "smdesai/SmolVLM2-2.2B-Instruct-4bit",
        )
        self.vlm_model, self.processor = load(model_id)
        self.config = load_config(model_id)
        # Detect "stuck" captions: same text across different frame hashes.
        self._recent_captions = []

    def get_vision_response(self, camera_frame):
        # camera_frame is a PIL Image; mlx-vlm generate wants a path/URL list
        import hashlib

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        frame, found_face, look_dir = _prepare_edge_frame(camera_frame)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            frame.save(tmp, format="JPEG", quality=85)
            image_path = tmp.name
            # Also snapshot what the model saw this frame (overwrites each call).
            frame.save("debug_frame.jpg", format="JPEG", quality=85)

        frame_hash = hashlib.md5(open(image_path, "rb").read()).hexdigest()[:8]

        try:
            formatted_prompt = apply_chat_template(
                self.processor,
                self.config,
                VLM_PROMPT_EDGE,
                num_images=1,
            )
            output = generate(
                self.vlm_model,
                self.processor,
                formatted_prompt,
                [image_path],
                max_tokens=60,
                temperature=0.0,
                verbose=False,
            )
        finally:
            os.unlink(image_path)
            try:
                import gc

                import mlx.core as mx

                mx.clear_cache()
                gc.collect()
            except Exception:
                pass

        # Newer mlx-vlm returns a GenerationResult; older versions returned a plain string.
        raw = output.text if hasattr(output, "text") else str(output)
        raw = raw.strip()
        face_tag = "face" if found_face else "noface"
        look_tag = {-1: "L", 0: "C", 1: "R"}.get(look_dir, "?")
        # frame=xxxxxxxx changing => camera ok; same hash + same caption => stuck frame;
        # changing hash + same caption => model ignoring the image / stuck prior.
        # look=L/C/R => crop biased left / center / right from profile-face yaw.
        print(f"[edge raw {face_tag} look={look_tag} frame={frame_hash}] {raw!r}")

        norm = re.sub(r"\s+", " ", raw.lower())
        self._recent_captions.append((frame_hash, norm))
        self._recent_captions = self._recent_captions[-6:]
        hashes = {h for h, _ in self._recent_captions}
        captions = {c for _, c in self._recent_captions}
        if len(self._recent_captions) >= 4 and len(hashes) >= 3 and len(captions) == 1:
            print(
                "[vlm warn] same caption across different frames — "
                "model may be stuck on a prior; check debug_frame.jpg"
            )

        self.vision_response = parse_edge_answers(raw)
        return self.vision_response


class VLM_CLOUD:
    def __init__(self):
        from openai import OpenAI

        self.vlm_model = "gpt-4o-mini"  # gpt-realtime is voice-only, so we use a vision-capable model here
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment

    def get_vision_response(self, camera_frame):
        # camera_frame is a PIL Image; the API wants it as a base64 data URL
        buffer = io.BytesIO()
        camera_frame.save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = self.client.responses.create(
            model=self.vlm_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": VLM_PROMPT},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                }
            ],
        )
        self.vision_response = response.output_text
        return self.vision_response
