"""Cloud vision backend: OpenAI multimodal model, one frame per call."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, cast

from PIL import Image

from ..observations import Observation, parse_observation_line
from .captions import classify_caption
from .prompts import CLOUD_VISION_PROMPT

logger = logging.getLogger(__name__)


class OpenAIVision:
    """Classify frames with an OpenAI vision model."""

    def __init__(self, model: str = "gpt-4o-mini", jpeg_quality: int = 85) -> None:
        """Create the backend. The client reads OPENAI_API_KEY from the environment."""
        from openai import OpenAI

        self._model = model
        self._jpeg_quality = jpeg_quality
        self._client = OpenAI()

    def classify(self, frame: Image.Image) -> Observation:
        """Classify one frame, falling back to a neutral observation on failure."""
        try:
            # Cast: the SDK types this as a union of ~30 TypedDicts that a plain
            # dict literal cannot be checked against.
            request_input = cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": CLOUD_VISION_PROMPT},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{self._encode(frame)}",
                            },
                        ],
                    }
                ],
            )
            response = self._client.responses.create(model=self._model, input=request_input)
            raw = (response.output_text or "").strip()
        except Exception:
            # A dropped connection or rate limit must not end the session; skip
            # this frame and try again on the next tick.
            logger.exception("Vision request failed; treating this frame as unremarkable")
            return Observation()

        logger.debug("cloud vision raw: %r", raw)
        # The model is asked for the field line, but occasionally narrates instead.
        return parse_observation_line(raw) or classify_caption(raw)

    def _encode(self, frame: Image.Image) -> str:
        buffer = io.BytesIO()
        frame.convert("RGB").save(buffer, format="JPEG", quality=self._jpeg_quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def close(self) -> None:
        """Nothing to release for the HTTP client."""
