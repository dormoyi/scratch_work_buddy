"""Turning a free-form VLM caption into a typed :class:`Observation`.

Small local vision models are far better at describing a scene than at filling in
a rigid field template, so the edge backend asks for two plain sentences and this
module does the classification. Every pattern here exists because a real model
said something that the previous version got wrong; the tests in
``tests/test_captions.py`` are that history.
"""

from __future__ import annotations

import re

from ..observations import Observation

# Parts of the head that count as "face" for contact purposes.
_FACE = r"(face|cheek|chin|jaw|forehead|hair|temple)"
# Possessive determiners models sprinkle in: "her cheek", "the chin", "their face".
_DET = r"(her\s+|his\s+|their\s+|the\s+|a\s+)?"
# Things that can prop up a face.
_SUPPORT = r"(hand|arm|forearm|wrist|sleeve|fist)"


def _expand(pattern: str) -> str:
    """Substitute the shared sub-patterns.

    Deliberately not an f-string: regex quantifiers like ``.{0,24}`` collide with
    f-string replacement fields and silently compile to a literal ``(0, 24)``.
    """
    return (
        pattern.replace("<FACE>", _FACE).replace("<DET>", _DET).replace("<SUPPORT>", _SUPPORT)
    )


# Models often hedge with "hands are not visible" and then describe a hand anyway.
# Strip visibility-only hedges so a genuine detection later in the same sentence
# still counts, while real denials ("no hand under the chin") keep their "no".
_VISIBILITY_HEDGE_RE = re.compile(r"\b(not|n't|never)\s+(visible|in\s+(the\s+)?frame|shown|seen)\b")
_NEGATION_RE = re.compile(r"\b(no|not|n't|without|never)\b")

_NAILS_RE = re.compile(
    r"biting|nails?.{0,24}mouth|mouth.{0,24}(finger|nail|hand)|"
    r"fingers?\s+(are\s+)?(at|in|inside|near|touching)\s+(the\s+)?mouth|"
    r"finger\s+at\s+(the\s+)?(person'?s\s+)?mouth|"
    r"hand\s+at\s+(the\s+)?mouth"
)

_FACE_TOUCH_RE = re.compile(
    _expand(
        r"scratch|"
        r"touch(ing)?\s+<DET><FACE>|"
        r"touching\s+.{0,24}<FACE>|"
        r"hand\s+(is\s+)?(touching|on|against|resting on)\s+<DET><FACE>|"
        r"palm\s+(is\s+)?(on|against|under)\s+<DET><FACE>|"
        r"on\s+<DET><FACE>|"
        r"fingers?\s+(are\s+)?(on|across|against|at)\s+<DET><FACE>|"
        r"fingers?\s+in\s+<DET>hair|"
        r"resting\s+on\s+<DET><FACE>|"
        r"covering\s+<DET>(face|mouth|eyes)|"
        r"hand\s+to\s+<DET><FACE>|"
        r"(head|chin|jaw|cheek|face)\s+on\s+<DET>hand|"
        r"resting\s+<DET>(head|chin)\s+on\s+<DET><SUPPORT>|"
        r"(head|chin|jaw|cheek|face|lip)\s+(is\s+)?(resting\s+)?on\s+<DET><SUPPORT>|"
        r"leaning\s+.{0,24}(on|into)\s+<DET><SUPPORT>|"
        r"hand\s+under\s+<DET>(chin|cheek|jaw|face)|"
        r"(arm|forearm|sleeve)\s+under\s+<DET>(chin|cheek|jaw|face)|"
        r"propping\s+.{0,16}(chin|face|cheek)|"
        r"supported\s+by\s+<DET><SUPPORT>"
    )
)

# The structured shorthand the edge prompt asks for, e.g. "Face: Yes | Looking: screen".
_LABELLED_FACE_RE = re.compile(r"\bface(?:-?\s*touch)?\s*:\s*(yes|no)\b")
_LABELLED_NAILS_RE = re.compile(r"\b(?:nails?|nail-?\s*bite|nailbite)\s*:\s*(yes|no)\b")
_LABELLED_LOOKING_RE = re.compile(r"\blooking\s*:\s*(screen|camera|phone|away)\b")

_LOOKING_AWAY_RE = re.compile(
    r"looking away|turned (to the )?side|head turned|looking off|looking elsewhere"
)
# A desk camera sits on the screen, so "looking away from the camera" is usually
# still screen work, not distraction.
_AWAY_FROM_CAMERA_RE = re.compile(r"looking away from (the )?camera")
_AT_CAMERA_RE = re.compile(r"looking at (the )?camera")
_NOT_AT_CAMERA_RE = re.compile(r"not looking at (the )?camera")
_AT_SCREEN_RE = re.compile(
    r"looking (at|toward|towards|down).{0,20}(screen|monitor|laptop)|"
    r"looking at (the )?screen|"
    r"seated in front of a laptop|"
    r"in front of (a |the )?(screen|monitor|laptop)"
)
_NOT_AT_SCREEN_RE = re.compile(r"not looking at (the )?(screen|monitor)")
_AWAY_FROM_DESK_RE = re.compile(
    r"empty (chair|seat)|(chair|seat) (is|appears|looks) empty|"
    r"walking away|stood up|not at (the )?desk|\b(nobody|no one)\b"
)


def _is_negated(text: str, match_start: int) -> bool:
    """Whether a match sits inside a negated clause.

    Scoped to the current sentence. A fixed-width lookback is not enough: models
    echo the prompt's noun list, so the "no" can be 60+ characters before the word
    that matched.
    """
    preceding = text[:match_start]
    last_stop = max(preceding.rfind("."), preceding.rfind("!"), preceding.rfind("?"))
    window = text[last_stop + 1 : match_start] if last_stop >= 0 else preceding
    window = _VISIBILITY_HEDGE_RE.sub(" ", window)
    return bool(_NEGATION_RE.search(window))


def _found_unnegated(pattern: re.Pattern[str], text: str) -> bool:
    """True if the pattern matches anywhere outside a negated clause."""
    return any(not _is_negated(text, m.start()) for m in pattern.finditer(text))


def classify_caption(caption: str) -> Observation:
    """Classify a free-form caption into an :class:`Observation`."""
    text = caption.lower()

    labelled_nails = _LABELLED_NAILS_RE.search(text)
    if labelled_nails:
        biting_nails = labelled_nails.group(1) == "yes"
    else:
        biting_nails = _found_unnegated(_NAILS_RE, text)

    # Fingers at the mouth are nail-biting, not face-touching; skip the second
    # test entirely so one gesture never reports as two habits.
    touching_face = False
    if not biting_nails:
        labelled_face = _LABELLED_FACE_RE.search(text)
        if labelled_face:
            touching_face = labelled_face.group(1) == "yes"
        if not touching_face:
            touching_face = _found_unnegated(_FACE_TOUCH_RE, text)

    looking_at_phone = _found_unnegated(re.compile(r"\bphone\b"), text)

    labelled_looking = _LABELLED_LOOKING_RE.search(text)
    labelled_gaze = labelled_looking.group(1) if labelled_looking else None

    at_camera = bool(_AT_CAMERA_RE.search(text)) and not _NOT_AT_CAMERA_RE.search(text)
    explicit_screen = (
        not looking_at_phone
        and (labelled_gaze == "screen" or bool(_AT_SCREEN_RE.search(text)))
        and not _NOT_AT_SCREEN_RE.search(text)
    )

    if labelled_gaze == "phone":
        looking_at_phone = True
        explicit_screen = False

    looking_at_screen = at_camera or explicit_screen
    truly_away = (
        bool(_LOOKING_AWAY_RE.search(text))
        and not _AWAY_FROM_CAMERA_RE.search(text)
        and not explicit_screen
    )
    if labelled_gaze == "away":
        truly_away = True
        looking_at_screen = False

    # When gaze is merely unclear, assume the ordinary case rather than reporting
    # distraction. Only an explicit "looking away" overrides this.
    if not truly_away and not looking_at_phone and not looking_at_screen:
        looking_at_screen = True

    return Observation(
        looking_at_screen=looking_at_screen,
        touching_face=touching_face,
        looking_at_phone=looking_at_phone,
        biting_nails=biting_nails,
        at_keyboard=not bool(_AWAY_FROM_DESK_RE.search(text)),
        straight_posture=None,
    ).resolved()
