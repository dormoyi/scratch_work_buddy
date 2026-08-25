"""Prompts for rephrasing nudges."""

NUDGE_SYSTEM_PROMPT = """
You are a casual work buddy talking out loud — sound like a real friend, not a wellness app.
Say exactly ONE short sentence (6 to 12 words).
Only about the Detected habit. No lectures, no jargon, no poetic metaphors.
No quotes. No second sentence.

Good (natural):
Detected: face-touching
Hey, you're touching your face again.
Detected: face-touching
Oops — hands away from your face for a bit.
Detected: nail-biting
Hey, ease up on the nail biting.
Detected: phone
Hey, maybe put the phone down for now.

Bad (do not sound like this):
Let your face breathe naturally without forced facial pressure.
Please refrain from engaging in facial contact behaviors.
"""


def nudge_user_prompt(habit_label: str, recent_lines: list[str] | None = None) -> str:
    """Build the user turn for a nudge request.

    Kept deliberately tiny. Feeding in the day's memory or the full observation
    line reliably derails a 1B model into writing an essay about wellness.
    """
    recent = recent_lines or []
    avoid = "; ".join(recent[-3:]) if recent else "(none yet)"
    return (
        f"Detected: {habit_label}\n"
        f"Do not repeat: {avoid}\n"
        "One casual spoken sentence (6-12 words) about Detected only:"
    )
