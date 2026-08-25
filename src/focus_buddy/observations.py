"""The single fact the whole app is built around: what the camera just saw.

Every vision backend returns an :class:`Observation`; every downstream consumer
(memory, nudges, logging) reads typed fields off it. Backends differ only in how
they *produce* one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum


class Habit(str, Enum):
    """A focus-breaking habit the buddy is willing to speak up about."""

    TOUCHING_FACE = "touching_face"
    LOOKING_AT_PHONE = "looking_at_phone"
    BITING_NAILS = "biting_nails"

    @property
    def label(self) -> str:
        """Short human-readable name, used in spoken summaries."""
        return _HABIT_LABELS[self]


_HABIT_LABELS = {
    Habit.TOUCHING_FACE: "face-touching",
    Habit.LOOKING_AT_PHONE: "phone",
    Habit.BITING_NAILS: "nail-biting",
}

# Priority order when several habits fire on the same frame. Fingers at the mouth
# also look like a hand on the face, so nail-biting must win over face-touching.
HABIT_PRIORITY = (Habit.BITING_NAILS, Habit.LOOKING_AT_PHONE, Habit.TOUCHING_FACE)


@dataclass(frozen=True)
class Observation:
    """One classified camera frame.

    Defaults describe the ordinary working state, so a backend that can only
    determine some fields still produces a sane observation.
    """

    looking_at_screen: bool = True
    touching_face: bool = False
    looking_at_phone: bool = False
    biting_nails: bool = False
    at_keyboard: bool = True
    straight_posture: float | None = None

    @property
    def habits(self) -> tuple[Habit, ...]:
        """Detected habits, highest priority first."""
        return tuple(h for h in HABIT_PRIORITY if getattr(self, h.value))

    @property
    def primary_habit(self) -> Habit | None:
        """The one habit worth speaking about on this frame, if any."""
        habits = self.habits
        return habits[0] if habits else None

    def resolved(self) -> Observation:
        """Drop lower-priority habits that are really the same gesture.

        Fingers at the mouth register as both nail-biting and face-touching; only
        the more specific one should reach the user.
        """
        if self.biting_nails and self.touching_face:
            return replace(self, touching_face=False)
        return self

    def to_line(self) -> str:
        """Render the flat one-line form used in logs and VLM prompts."""
        return " | ".join(
            [
                f'"looking_at_screen": {self.looking_at_screen}',
                f'"touching_face": {self.touching_face}',
                f'"looking_at_phone": {self.looking_at_phone}',
                f'"biting_nails": {self.biting_nails}',
                f'"at_keyboard": {self.at_keyboard}',
                f'"straight_posture": {self.straight_posture}',
            ]
        )

    def __str__(self) -> str:
        """Use the flat line form when interpolated into log messages."""
        return self.to_line()


# Accepts both `"touching_face": True` and `touching_face: true`.
_FIELD_RE = re.compile(
    r'["\']?(looking_at_screen|touching_face|looking_at_phone|biting_nails'
    r'|at_keyboard|straight_posture)["\']?\s*:\s*([^\s|,]+)',
    re.IGNORECASE,
)

BOOL_FIELDS = (
    "looking_at_screen",
    "touching_face",
    "looking_at_phone",
    "biting_nails",
    "at_keyboard",
)


def parse_bool(token: str) -> bool | None:
    """Parse a model-emitted boolean token, or None if it isn't one."""
    token = token.strip().strip(",.").lower()
    if token in ("true", "yes", "1"):
        return True
    if token in ("false", "no", "0"):
        return False
    return None


def _parse_posture(token: str) -> float | None:
    token = token.strip().strip(",.").lower()
    if token in ("none", "null", "n/a"):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_observation_line(text: str) -> Observation | None:
    """Parse a field line emitted by a VLM.

    Returns None when the text carries no usable fields, so callers can fall back
    to free-form caption parsing.
    """
    values: dict[str, object] = {}
    for match in _FIELD_RE.finditer(text):
        field, token = match.group(1).lower(), match.group(2)
        if field == "straight_posture":
            values[field] = _parse_posture(token)
        else:
            parsed = parse_bool(token)
            if parsed is not None:
                values[field] = parsed

    if not any(field in values for field in BOOL_FIELDS):
        return None

    # Small quantised checkpoints fail this prompt by answering True to everything.
    # Treat an all-True line as "no signal" rather than three simultaneous habits.
    if all(values.get(field) is True for field in BOOL_FIELDS):
        return Observation()

    return Observation(
        looking_at_screen=bool(values.get("looking_at_screen", True)),
        touching_face=bool(values.get("touching_face", False)),
        looking_at_phone=bool(values.get("looking_at_phone", False)),
        biting_nails=bool(values.get("biting_nails", False)),
        at_keyboard=bool(values.get("at_keyboard", True)),
        straight_posture=values.get("straight_posture"),  # type: ignore[arg-type]
    ).resolved()
