"""Prompts for the vision backends."""

# Cloud models follow a strict output format reliably, so ask for the fields directly.
CLOUD_VISION_PROMPT = """
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

# Local 2B models answer a description far more reliably than a field template, so
# ask for prose and classify it in `captions.py`. Face contact is asked first
# because "hands not visible" is a sticky prior once the model commits to it.
EDGE_VISION_PROMPT = """
Webcam close-up. Reply in exactly two short sentences.
Sentence 1 (required): is anything supporting or touching the face — hand, fingers, fist,
forearm, wrist, or sleeve under the chin/cheek? Or finger at/in the mouth? Or nothing touching?
Do not say "hands not visible" if a sleeve or arm is under the chin.
Sentence 2: where is the person looking — screen, camera, phone, or away?
"""
