"""Bodies: a camera to see with and a speaker to talk through."""

from .base import Body
from .desktop import DesktopBody
from .reachy import ReachyBody

__all__ = ["Body", "DesktopBody", "ReachyBody"]
