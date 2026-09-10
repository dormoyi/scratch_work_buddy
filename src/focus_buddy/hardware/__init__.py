"""Bodies: a camera to see with and a speaker to talk through.

Imported lazily. Eagerly importing both bodies pulled OpenCV into every run,
including on the robot, where the laptop webcam class is never constructed --
and this app has to be frugal: it shares an 8GB machine with the Reachy daemon
and its GStreamer pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Body

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .desktop import DesktopBody
    from .reachy import ReachyBody

__all__ = ["Body", "DesktopBody", "ReachyBody"]


def __getattr__(name: str) -> Any:
    """Import a body only when something actually asks for it."""
    if name == "DesktopBody":
        from .desktop import DesktopBody

        return DesktopBody
    if name == "ReachyBody":
        from .reachy import ReachyBody

        return ReachyBody
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
