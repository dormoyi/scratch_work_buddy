"""Nudge-writing backends."""

from __future__ import annotations

import logging

from ..config import Backend, Settings
from .base import NudgeWriter

logger = logging.getLogger(__name__)


def build_nudge_writer(settings: Settings) -> NudgeWriter | None:
    """Construct the nudge writer named by ``settings``, importing it lazily.

    Returns None when nudges should stay as the fixed phrasings. A local backend
    never falls back to the cloud: picking an offline backend and switching on
    rewriting used to route to OpenAI without saying so, which is the opposite
    of what the setting implies.
    """
    if settings.backend is Backend.CLOUD:
        from .openai_llm import OpenAINudgeWriter

        return OpenAINudgeWriter(model=settings.cloud_llm_model)

    if settings.local_nudge_model_available:
        from .mlx_llm import MLXNudgeWriter

        return MLXNudgeWriter(model_id=settings.edge_llm_model)

    logger.warning(
        "No local language model for nudges on backend=%s; using the fixed phrasings. "
        "Install the edge-vlm extra on Apple Silicon, or use backend=cloud.",
        settings.backend.value,
    )
    return None


__all__ = ["NudgeWriter", "build_nudge_writer"]
