"""Create the virtualenv that the landmark sidecar runs in.

MediaPipe's working API needs numpy < 2 while reachy-mini and opencv-python need
numpy >= 2, so the two cannot share an interpreter. This builds a small separate
environment for MediaPipe alone; nothing in it is imported by Focus Buddy, which
talks to it over a pipe.

Usage:
    python tools/setup_landmark_sidecar.py            # default location
    python tools/setup_landmark_sidecar.py --path ~/somewhere/venv
    python tools/setup_landmark_sidecar.py --recreate # rebuild from scratch
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Exact pin, not a range. The legacy `solutions` graphs are gone by 0.10.30,
# leaving only the Tasks API, which aborts in DrishtiMetalHelper on Apple
# Silicon -- and 0.10.21, the newest that still has `solutions`, ships a wheel
# whose _framework_bindings fail to initialise here. 0.10.14 is what actually
# runs. numpy is pinned because that wheel was built against the 1.x ABI.
REQUIREMENTS = ("mediapipe==0.10.14", "numpy<2", "pillow>=10.0")


def default_path() -> Path:
    """Default sidecar location, matching perception.landmarks."""
    return Path.home() / ".cache" / "focus_buddy" / "landmark-venv"


def build(path: Path, recreate: bool = False) -> Path:
    """Create the venv and install MediaPipe into it. Returns the interpreter."""
    if path.exists() and recreate:
        print(f"removing {path}")
        shutil.rmtree(path)

    if not path.exists():
        print(f"creating venv at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)

    python = path / "bin" / "python"
    if not python.exists():  # Windows layout
        python = path / "Scripts" / "python.exe"
    if not python.exists():
        raise SystemExit(f"no interpreter under {path}")

    print(f"installing {', '.join(REQUIREMENTS)}")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", *REQUIREMENTS], check=True)
    return python


def verify(python: Path) -> None:
    """Fail loudly here rather than at the first camera frame."""
    probe = (
        "import mediapipe as mp, numpy;"
        "assert hasattr(mp, 'solutions'), 'no solutions API in mediapipe ' + mp.__version__;"
        "mp.solutions.hands.Hands(static_image_mode=True).close();"
        "print('mediapipe', mp.__version__, 'numpy', numpy.__version__)"
    )
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"sidecar verification failed:\n{result.stderr.strip()}")
    print(f"verified: {result.stdout.strip()}")


def main(argv: list[str] | None = None) -> int:
    """Build and verify the sidecar environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=default_path())
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)

    python = build(args.path.expanduser(), recreate=args.recreate)
    verify(python)
    print(
        "\nDone. Focus Buddy finds this automatically; to use a different one set\n"
        f"    FOCUS_BUDDY_LANDMARK_PYTHON={python}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
