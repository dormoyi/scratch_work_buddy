"""Compare local VLM checkpoints on the same image and prompt.

A development tool, not part of the installed package. Run it when picking or
re-evaluating the edge vision model; it prints load time, inference time and the
raw caption for each candidate so you can weigh quality against latency.

Usage:
    python tools/vlm_bakeoff.py --image crop.jpg
    python tools/vlm_bakeoff.py --models mlx-community/Qwen2-VL-2B-Instruct-4bit

Requires the edge extra (Apple Silicon only):
    pip install -e '.[edge]'
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from focus_buddy.perception.captions import classify_caption
from focus_buddy.perception.prompts import EDGE_VISION_PROMPT

# 4-bit converts plus the higher-precision 2.2B and a much smaller 500M, so the
# quality/latency curve is visible rather than guessed at.
CANDIDATES = (
    "smdesai/SmolVLM2-2.2B-Instruct-4bit",
    "mlx-community/SmolVLM2-2.2B-Instruct-mlx",
    "mlx-community/SmolVLM2-500M-Video-Instruct-mlx",
    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
)


def run_one(model_id: str, image_path: str, max_tokens: int) -> dict[str, Any]:
    """Load one checkpoint, caption one image, and free the memory again."""
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"\n=== {model_id} ===")
    started = time.perf_counter()
    model, processor = load(model_id)
    config = load_config(model_id)
    load_s = time.perf_counter() - started
    print(f"load:  {load_s:.1f}s")

    prompt = apply_chat_template(processor, config, EDGE_VISION_PROMPT, num_images=1)
    started = time.perf_counter()
    output = generate(
        model,
        processor,
        prompt,
        [image_path],
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    infer_s = time.perf_counter() - started
    caption = (output.text if hasattr(output, "text") else str(output)).strip()

    print(f"infer: {infer_s:.2f}s")
    print(f"raw:   {caption!r}")
    # The caption alone is not the thing being judged; what matters is what the
    # classifier makes of it, so show that too.
    print(f"parsed:{classify_caption(caption)}")

    del model, processor
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass

    return {"model": model_id, "load_s": load_s, "infer_s": infer_s, "caption": caption}


def main(argv: list[str] | None = None) -> int:
    """Run the bake-off."""
    parser = argparse.ArgumentParser(description="Compare local MLX VLM checkpoints.")
    parser.add_argument("--image", required=True, help="path to a test frame")
    parser.add_argument("--models", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--max-tokens", type=int, default=60)
    args = parser.parse_args(argv)

    results = []
    for model_id in args.models:
        try:
            results.append(run_one(model_id, args.image, args.max_tokens))
        except Exception as error:
            print(f"FAILED: {error}")
            results.append({"model": model_id, "error": str(error)})

    print("\n======== SUMMARY ========")
    for result in results:
        if "error" in result:
            print(f"- {result['model']}: ERROR {result['error'][:120]}")
        else:
            print(
                f"- {result['model']}: load {result['load_s']:.1f}s, "
                f"infer {result['infer_s']:.2f}s\n    {result['caption']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
