"""Create the virtualenv that the landmark sidecar runs in.

Ships with the package rather than living in ``tools/`` because ``edge`` is
unusable without it: an installed wheel carries no ``tools`` directory, so a
user who installed Focus Buddy from an app store had no way to run the setup
the error message told them to run.

    focus-buddy-setup-sidecar
    python -m focus_buddy.setup_sidecar --recreate

MediaPipe cannot share an interpreter with the rest of Focus Buddy: the only
API that works on Apple Silicon is the legacy ``solutions`` one, which needs
numpy < 2, while reachy-mini needs numpy >= 2.2.5. This builds a small separate
environment for MediaPipe alone. Nothing in it is imported by Focus Buddy,
which talks to it over a pipe.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Exact pins, not ranges. The legacy `solutions` graphs are gone by 0.10.30,
# leaving only the Tasks API, which aborts in DrishtiMetalHelper on Apple
# Silicon -- and 0.10.21, the newest that still has `solutions`, ships a wheel
# whose _framework_bindings fail to initialise. 0.10.14 is what actually runs.
# numpy is pinned because that wheel was built against the 1.x ABI.
REQUIREMENTS = ("mediapipe==0.10.14", "numpy<2", "pillow>=10.0")


def sidecar_python(venv_dir: Path) -> Path:
    """Locate the interpreter inside a sidecar venv, on either layout."""
    posix = venv_dir / "bin" / "python"
    windows = venv_dir / "Scripts" / "python.exe"
    if posix.exists():
        return posix
    if windows.exists():
        return windows
    return windows if os.name == "nt" else posix


def build(path: Path, recreate: bool = False) -> Path:
    """Create the venv and install MediaPipe into it. Returns the interpreter.

    Every subprocess runs with a scrubbed environment. Importing reachy_mini puts
    this package's own site-packages on PYTHONPATH for GStreamer, and a child
    interpreter that inherits it resolves the parent's numpy and mediapipe --
    exactly the pair the sidecar exists to escape.
    """
    from .perception.landmarks import child_environment

    env = child_environment()
    if path.exists() and recreate:
        print(f"removing {path}")
        shutil.rmtree(path)

    if not path.exists():
        print(f"creating venv at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(path)], check=True, env=env)

    python = sidecar_python(path)
    if not python.exists():
        raise SystemExit(f"no interpreter under {path}")

    print(f"installing {', '.join(REQUIREMENTS)}")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True, env=env
    )
    # Not --quiet: a partial install here surfaces as a hang on the first frame,
    # which is far harder to diagnose than a noisy pip.
    subprocess.run([str(python), "-m", "pip", "install", *REQUIREMENTS], check=True, env=env)
    return python


def verify(python: Path) -> None:
    """Fail loudly here rather than at the first camera frame."""
    from .perception.landmarks import child_environment

    probe = (
        "import mediapipe as mp, numpy;"
        "assert hasattr(mp, 'solutions'), 'no solutions API in mediapipe ' + mp.__version__;"
        "mp.solutions.hands.Hands(static_image_mode=True).close();"
        "print('mediapipe', mp.__version__, 'numpy', numpy.__version__)"
    )
    result = subprocess.run(
        [str(python), "-c", probe], capture_output=True, text=True, env=child_environment()
    )
    if result.returncode != 0:
        raise SystemExit(f"sidecar verification failed:\n{result.stderr.strip()}")
    print(f"verified: {result.stdout.strip()}")


def main(argv: list[str] | None = None) -> int:
    """Build and verify the sidecar environment."""
    # Imported here so --help works without pulling in numpy and Pillow.
    from .perception.landmarks import default_sidecar_dir

    parser = argparse.ArgumentParser(description="Create the landmark sidecar environment.")
    parser.add_argument(
        "--path", type=Path, default=None, help="where to build it (default: the cache directory)"
    )
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild")
    args = parser.parse_args(argv)

    path = (args.path or default_sidecar_dir()).expanduser()
    python = build(path, recreate=args.recreate)
    verify(python)
    print(
        f"\nDone. Focus Buddy finds this automatically; to use a different one set\n"
        f"    FOCUS_BUDDY_LANDMARK_PYTHON={python}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
