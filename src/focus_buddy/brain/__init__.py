"""Nudge-writing backends."""

from __future__ import annotations

from ..config import Backend, Settings
from .base import NudgeWriter


def build_nudge_writer(settings: Settings) -> NudgeWriter:
    """Construct the nudge writer named by ``settings``, importing it lazily."""
    if settings.backend is Backend.EDGE:
        from .mlx_llm import MLXNudgeWriter

        return MLXNudgeWriter(model_id=settings.edge_llm_model)

    from .openai_llm import OpenAINudgeWriter

    return OpenAINudgeWriter(model=settings.cloud_llm_model)


__all__ = ["NudgeWriter", "build_nudge_writer"]
