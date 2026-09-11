"""What the buddy actually says, and the guard rails on saying it.

Template nudges are the default and the fallback. A language model can rephrase
them, but anything it produces must pass :func:`rejection_reason` before it
reaches a speaker: a small model that drifts off-topic is worse than a fixed
sentence, because the user cannot tell it is wrong about them.
"""

from __future__ import annotations

import random
import re

from .observations import Habit, Observation

# Several phrasings per habit so a repeat offender does not hear the same line
# four times in a row.
NUDGE_TEMPLATES: dict[Habit, tuple[str, ...]] = {
    Habit.BITING_NAILS: (
        "Hey, you're biting your nails again.",
        "Oops - ease up on the nail biting.",
        "Hey, give those nails a break.",
        "Caught you biting your nails - easy does it.",
    ),
    Habit.TOUCHING_FACE: (
        "Hey, you're touching your face again.",
        "Oops - hands off your face for a bit.",
        "Hey, try not to touch your face.",
        "Face touch! Maybe keep your hands down for a sec.",
    ),
    Habit.LOOKING_AT_PHONE: (
        "Hey, maybe put the phone down for now.",
        "Oops - phone can wait a minute.",
        "Hey, slide the phone aside when you can.",
        "Phone check - try putting it down for a bit.",
    ),
}

# A valid nudge has to mention the habit it is about. Without this check a small
# model happily nudges you about your posture when you were on your phone.
HABIT_KEYWORDS: dict[Habit, tuple[str, ...]] = {
    Habit.BITING_NAILS: ("nail", "biting", "mouth", "finger", "nibble", "nibbling"),
    Habit.TOUCHING_FACE: ("face", "cheek", "chin", "scratch", "hand", "hands"),
    Habit.LOOKING_AT_PHONE: ("phone",),
}

MIN_WORDS = 4
MAX_WORDS = 12

# Small models love to append a scolding "...and you're not even looking at your
# screen", which is often flatly untrue. Reject it when the frame says otherwise.
_SCREEN_COMPLAINT_RE = re.compile(
    r"\b(not|n't|haven'?t|aren'?t|wasn'?t)\b.{0,24}\blook(ing)?\b.{0,16}\bscreen\b|"
    r"\blook\s+at\s+(your\s+)?screen\b|"
    r"\bfocus\s+on\s+(the\s+|your\s+)?screen\b",
    re.IGNORECASE,
)


def pick_template(habit: Habit, rng: random.Random | None = None) -> str:
    """Pick a template nudge for a habit."""
    options = NUDGE_TEMPLATES.get(habit)
    if not options:
        return "Hey, gentle reminder."
    return (rng or random).choice(options)


def clean(text: str | None) -> str:
    """Normalise raw model output down to a single spoken sentence."""
    text = (text or "").strip().strip('"').strip("'")
    return re.split(r"(?<=[.!?])\s+", text)[0].strip()


def rejection_reason(text: str, habit: Habit, observation: Observation | None = None) -> str | None:
    """Explain why a candidate nudge is unusable, or None if it is fine.

    Returning the reason rather than a bool keeps the logs diagnosable: over a
    long run the distribution of rejection reasons tells you what to fix in the
    prompt.
    """
    words = (text or "").split()
    if len(words) < MIN_WORDS:
        return "too short"
    if len(words) > MAX_WORDS:
        return f"too long ({len(words)} words)"
    if not any(keyword in text.lower() for keyword in HABIT_KEYWORDS[habit]):
        return f"does not mention {habit.value}"
    if observation is not None and observation.looking_at_screen:
        if _SCREEN_COMPLAINT_RE.search(text):
            return "complains about screen attention that the frame contradicts"
    return None


def is_valid(text: str, habit: Habit, observation: Observation | None = None) -> bool:
    """Whether a candidate nudge is safe to speak."""
    return rejection_reason(text, habit, observation) is None


def format_duration(seconds: float) -> str:
    """Render a duration the way a person would say it out loud."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"
