"""Local vision backend: a quantised VLM running on Apple Silicon through MLX.

Apple Silicon only. `bitsandbytes` 4-bit is CUDA-only and MLX publishes no builds
for ARM Linux, so this cannot run on a Reachy Mini wireless. See README > Edge mode.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image

from ..observations import Observation, parse_observation_line
from .captions import classify_caption
from .framing import LOOK_CENTRE, LOOK_LEFT, LOOK_RIGHT, prepare_for_edge
from .prompts import EDGE_VISION_PROMPT

logger = logging.getLogger(__name__)

_LOOK_TAGS = {LOOK_LEFT: "L", LOOK_CENTRE: "C", LOOK_RIGHT: "R"}

# How many recent frames to inspect when deciding the model is stuck.
_STUCK_WINDOW = 6
_STUCK_MIN_SAMPLES = 4
_STUCK_MIN_DISTINCT_FRAMES = 3


class MLXVision:
    """Classify frames with a local MLX vision model.

    The 4-bit SmolVLM2 convert is ~1.5GB, against ~5GB for the bf16 checkpoint,
    which is what makes this fit alongside everything else on an 8GB Mac.
    """

    def __init__(
        self,
        model_id: str = "smdesai/SmolVLM2-2.2B-Instruct-4bit",
        max_tokens: int = 60,
        debug_frame_path: str | Path | None = None,
    ) -> None:
        """Load the model. Expect a multi-gigabyte download on first use."""
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "Edge mode needs the optional MLX dependencies. Install them with:\n"
                "    pip install 'focus_buddy[edge]'\n"
                "They are available on Apple Silicon only."
            ) from exc

        logger.info("Loading local vision model %s (first run downloads it)", model_id)
        self._model, self._processor = load(model_id)
        self._config = load_config(model_id)
        self._max_tokens = max_tokens
        self._debug_frame_path = Path(debug_frame_path) if debug_frame_path else None
        # (frame_hash, normalised_caption) pairs, for stuck-model detection.
        self._recent: list[tuple[str, str]] = []

    def classify(self, frame: Image.Image) -> Observation:
        """Crop, classify, and parse one frame."""
        prepared, found_face, look_dir = prepare_for_edge(frame)

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            tmp.close()
            prepared.save(tmp.name, format="JPEG", quality=85)
            if self._debug_frame_path is not None:
                self._debug_frame_path.parent.mkdir(parents=True, exist_ok=True)
                prepared.save(self._debug_frame_path, format="JPEG", quality=85)
            frame_hash = hashlib.md5(Path(tmp.name).read_bytes()).hexdigest()[:8]
            raw = self._generate(tmp.name)
        except Exception:
            logger.exception("Local vision inference failed; skipping this frame")
            return Observation()
        finally:
            os.unlink(tmp.name)
            self._free_metal_memory()

        logger.debug(
            "edge vision [%s look=%s frame=%s] %r",
            "face" if found_face else "noface",
            _LOOK_TAGS.get(look_dir, "?"),
            frame_hash,
            raw,
        )
        self._warn_if_stuck(frame_hash, raw)

        return parse_observation_line(raw) or classify_caption(raw)

    def _generate(self, image_path: str) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            self._processor, self._config, EDGE_VISION_PROMPT, num_images=1
        )
        output = generate(
            self._model,
            self._processor,
            prompt,
            [image_path],
            max_tokens=self._max_tokens,
            temperature=0.0,
            verbose=False,
        )
        # Newer mlx-vlm returns a GenerationResult; older versions returned a string.
        return (output.text if hasattr(output, "text") else str(output)).strip()

    def _warn_if_stuck(self, frame_hash: str, raw: str) -> None:
        """Warn when the model repeats itself across genuinely different frames.

        Identical captions on changing frames mean the model has latched onto a
        prior and is no longer reading the image, which otherwise looks like a
        camera fault.
        """
        normalised = " ".join(raw.lower().split())
        self._recent.append((frame_hash, normalised))
        self._recent = self._recent[-_STUCK_WINDOW:]

        if len(self._recent) < _STUCK_MIN_SAMPLES:
            return
        distinct_frames = {h for h, _ in self._recent}
        distinct_captions = {c for _, c in self._recent}
        if len(distinct_frames) >= _STUCK_MIN_DISTINCT_FRAMES and len(distinct_captions) == 1:
            logger.warning(
                "Vision model returned the same caption for %d different frames; "
                "it may be ignoring the image. Set FOCUS_BUDDY_DEBUG_FRAME to inspect the crop.",
                len(distinct_frames),
            )

    @staticmethod
    def _free_metal_memory() -> None:
        """Return the Metal allocator's cache between frames.

        Without this the VLM and the nudge model fight over unified memory on an
        8GB machine and inference latency climbs steadily.
        """
        try:
            import mlx.core as mx

            mx.clear_cache()
            gc.collect()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Could not clear the MLX cache", exc_info=True)

    def close(self) -> None:
        """Drop the model and free its memory."""
        self._model = None
        self._processor = None
        self._free_metal_memory()
