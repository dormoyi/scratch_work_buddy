"""Cloud nudge writer backed by an OpenAI text model."""

from __future__ import annotations

import logging

from .prompts import NUDGE_SYSTEM_PROMPT, nudge_user_prompt

logger = logging.getLogger(__name__)


class OpenAINudgeWriter:
    """Rephrase nudges with an OpenAI model."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.9) -> None:
        """Create the backend. The client reads OPENAI_API_KEY from the environment."""
        from openai import OpenAI

        self._model = model
        self._temperature = temperature
        self._client = OpenAI()

    def write_nudge(self, habit_label: str, recent_lines: list[str] | None = None) -> str:
        """Return a candidate nudge sentence."""
        response = self._client.responses.create(
            model=self._model,
            instructions=NUDGE_SYSTEM_PROMPT,
            input=nudge_user_prompt(habit_label, recent_lines),
            temperature=self._temperature,
        )
        return response.output_text or ""

    def close(self) -> None:
        """Nothing to release for the HTTP client."""
