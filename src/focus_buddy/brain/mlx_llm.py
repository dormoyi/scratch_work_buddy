"""Local nudge writer running on Apple Silicon through MLX.

Apple Silicon only; see README > Edge mode.
"""

from __future__ import annotations

import logging

from .prompts import NUDGE_SYSTEM_PROMPT, nudge_user_prompt

logger = logging.getLogger(__name__)


class MLXNudgeWriter:
    """Rephrase nudges with a small local model.

    Llama-3.2-1B at 4-bit is ~0.7GB. Gemma-3-1B at bf16 was 2GB+ and competed
    with the vision model for unified memory on an 8GB machine.
    """

    def __init__(self, model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit") -> None:
        """Load the model. Expect a download on first use."""
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "Edge mode needs the optional MLX dependencies. Install them with:\n"
                "    pip install 'focus_buddy[edge]'\n"
                "They are available on Apple Silicon only."
            ) from exc

        logger.info("Loading local nudge model %s", model_id)
        self._model, self._tokenizer = load(model_id)

    def write_nudge(self, habit_label: str, recent_lines: list[str] | None = None) -> str:
        """Return a candidate nudge sentence."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [
            {"role": "system", "content": NUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": nudge_user_prompt(habit_label, recent_lines)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 18 tokens is roughly the 12-word ceiling the validator enforces; letting
        # it run longer only produces text that gets truncated anyway.
        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=18,
            sampler=make_sampler(temp=0.6, top_p=0.9),
            verbose=False,
        ).strip()

    def close(self) -> None:
        """Drop the model and free its memory."""
        self._model = None
        self._tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Could not clear the MLX cache", exc_info=True)
