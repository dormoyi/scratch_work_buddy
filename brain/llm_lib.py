import os
import random
import re

from dotenv import load_dotenv

load_dotenv()


PROMPT_CONVERSATION = """
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

PROMPT_SUMMARIZING = """
Write one short spoken status line from the stats. Max 20 words.
Example: You've been working 2 hours. Nail-biting 3 times, phone 1 time.
"""

# Keywords we expect in a valid nudge for each habit (for fallback checks).
HABIT_KEYWORDS = {
    "biting_nails": ("nail", "biting", "mouth", "finger", "nibble", "nibbling"),
    "touching_face": ("face", "cheek", "chin", "scratch", "hand", "hands"),
    "looking_at_phone": ("phone",),
}

# Tiny LLMs often add a false "not looking at screen" clause.
_SCREEN_NEGATION_RE = re.compile(
    r"\b(not|n't|haven'?t|aren'?t|wasn'?t)\b.{0,24}\blook(ing)?\b.{0,16}\bscreen\b|"
    r"\blook\s+at\s+(your\s+)?screen\b|"
    r"\bfocus\s+on\s+(the\s+|your\s+)?screen\b",
    re.IGNORECASE,
)


def _nudge_user_prompt(detected_labels, vlm_response, memory, recent_lines=None):
    # Keep this tiny: day memory + full VLM lines derail Llama-3.2-1B into essays.
    detected = ", ".join(detected_labels) if detected_labels else "none"
    recent = recent_lines or []
    avoid = "; ".join(recent[-3:]) if recent else "(none yet)"
    return (
        f"Detected: {detected}\n"
        f"Do not repeat: {avoid}\n"
        "One casual spoken sentence (6-12 words) about Detected only:"
    )


class LLM_EDGE:
    # Gemma-3-1B at bf16 was ~2GB+ and fought the VLM for the 8GB Mac's RAM.
    # Llama-3.2-1B 4-bit via MLX is ~0.7GB and shares the Metal stack with the VLM.
    def __init__(self):
        from mlx_lm import load

        model_id = os.getenv(
            "LOCAL_LLM_MODEL",
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
        )
        self.llm_model, self.tokenizer = load(model_id)

    def _generate(self, system_prompt, user_text, max_tokens=40, temp=0.85):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return generate(
            self.llm_model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temp, top_p=0.9),
            verbose=False,
        ).strip()

    def get_llm_response(
        self, vlm_response, memory, detected_labels=None, recent_lines=None
    ):
        return self._generate(
            PROMPT_CONVERSATION,
            _nudge_user_prompt(
                detected_labels or [], vlm_response, memory, recent_lines
            ),
            max_tokens=18,
            temp=0.6,
        )

    def get_summarizing_response(self, short_term_memory, long_term_memory):
        return self._generate(
            PROMPT_SUMMARIZING,
            f"Previous summary:\n{long_term_memory}\n\n"
            f"Recent detections:\n{short_term_memory}",
            max_tokens=40,
            temp=0.3,
        )


class LLM_CLOUD:
    def __init__(self):
        from openai import OpenAI

        self.llm_model = "gpt-4o-mini"
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment

    def _generate(self, system_prompt, user_text):
        response = self.client.responses.create(
            model=self.llm_model,
            instructions=system_prompt,
            input=user_text,
            temperature=0.9,
        )
        return response.output_text

    def get_llm_response(
        self, vlm_response, memory, detected_labels=None, recent_lines=None
    ):
        return self._generate(
            PROMPT_CONVERSATION,
            _nudge_user_prompt(
                detected_labels or [], vlm_response, memory, recent_lines
            ),
        )

    def get_summarizing_response(self, short_term_memory, long_term_memory):
        return self._generate(
            PROMPT_SUMMARIZING,
            f"Previous summary:\n{long_term_memory}\n\n"
            f"Recent detections:\n{short_term_memory}",
        )


def clean_nudge(text):
    text = (text or "").strip().strip('"').strip("'")
    # Keep first sentence only.
    text = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    return text


def nudge_reject_reason(text, habit_keys, vlm_response=None):
    # Why a candidate failed validation (for logs).
    words = (text or "").split()
    if len(words) < 4:
        return "too short"
    if len(words) > 12:
        return f"too long ({len(words)} words)"
    lower = (text or "").lower()
    primary = habit_keys[0]
    if not any(k in lower for k in HABIT_KEYWORDS[primary]):
        return f"missing habit words for {primary}"
    if vlm_response and '"looking_at_screen": True' in vlm_response:
        if _SCREEN_NEGATION_RE.search(text or ""):
            return "false screen complaint"
    return None


def nudge_looks_valid(text, habit_keys, vlm_response=None):
    return nudge_reject_reason(text, habit_keys, vlm_response) is None


def pick_fallback_nudge(habit_key, options_by_habit):
    choices = options_by_habit.get(habit_key) or ["Hey, gentle reminder."]
    if isinstance(choices, str):
        return choices
    return random.choice(choices)
