"""Vision backends: camera frame in, :class:`~focus_buddy.observations.Observation` out."""

from __future__ import annotations

from pathlib import Path

from ..config import Backend, Settings
from .base import VisionBackend


def build_vision_backend(settings: Settings) -> VisionBackend:
    """Construct the vision backend named by ``settings``.

    Imports the implementation lazily so a cloud-only install never needs MLX,
    and the module-load cost of an unused backend is never paid.
    """
    if settings.backend is Backend.EDGE_VLM:
        from .mlx_vision import MLXVision

        return MLXVision(
            model_id=settings.edge_vision_model,
            debug_frame_path=settings.debug_frame_path,
        )

    if settings.uses_landmarks:
        from .landmarks import LandmarkVision

        return LandmarkVision(
            Path(settings.landmark_python).expanduser() if settings.landmark_python else None
        )

    from .openai_vision import OpenAIVision

    return OpenAIVision(model=settings.cloud_vision_model)


__all__ = ["VisionBackend", "build_vision_backend"]
