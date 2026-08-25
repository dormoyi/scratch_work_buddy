"""Compare edge VLM checkpoints on the same image + prompt.

Usage:
  .venv/bin/python vision/vlm_bakeoff.py
  .venv/bin/python vision/vlm_bakeoff.py --image debug_frame.jpg
"""

import argparse
import time

from dotenv import load_dotenv

load_dotenv()

# Mix of 4-bit converts + the official higher-precision MLX 2.2B + smaller 500M.
CANDIDATES = [
    "smdesai/SmolVLM2-2.2B-Instruct-4bit",
    "mlx-community/SmolVLM2-2.2B-Instruct-mlx",
    "mlx-community/SmolVLM2-500M-Video-Instruct-mlx",
    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
]

PROMPT = (
    "This is a webcam photo from the top of a computer monitor. "
    "In one or two short sentences, describe only: (1) where the person's hands are, "
    "(2) where they are looking. If hands are out of frame, say they are not visible. "
    "Do not use the words True or False."
)


def run_one(model_id, image_path):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"\n=== {model_id} ===")
    t0 = time.perf_counter()
    model, processor = load(model_id)
    config = load_config(model_id)
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f}s")

    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
    t1 = time.perf_counter()
    out = generate(
        model,
        processor,
        prompt,
        [image_path],
        max_tokens=60,
        temperature=0.0,
        verbose=False,
    )
    infer_s = time.perf_counter() - t1
    text = (out.text if hasattr(out, "text") else str(out)).strip()
    print(f"infer: {infer_s:.2f}s")
    print(f"raw: {text!r}")

    # Free Metal memory before the next candidate.
    del model, processor
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except Exception:
        pass
    return {"model": model_id, "load_s": load_s, "infer_s": infer_s, "text": text}


def main():
    parser = argparse.ArgumentParser(description="Bake off MLX VLM checkpoints")
    parser.add_argument("--image", default="debug_frame.jpg")
    parser.add_argument(
        "--models",
        nargs="*",
        default=CANDIDATES,
        help="HF model IDs to try (default: built-in shortlist)",
    )
    args = parser.parse_args()

    results = []
    for model_id in args.models:
        try:
            results.append(run_one(model_id, args.image))
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({"model": model_id, "error": str(exc)})

    print("\n======== SUMMARY ========")
    for r in results:
        if "error" in r:
            print(f"- {r['model']}: ERROR {r['error'][:120]}")
        else:
            print(
                f"- {r['model']}: load {r['load_s']:.1f}s, infer {r['infer_s']:.2f}s\n"
                f"    {r['text']}"
            )


if __name__ == "__main__":
    main()
