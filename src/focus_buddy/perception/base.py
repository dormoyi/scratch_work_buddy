"""The contract every vision backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from ..observations import Observation


@runtime_checkable
class VisionBackend(Protocol):
    """Classifies a camera frame into an :class:`Observation`."""

    def classify(self, frame: Image.Image) -> Observation:
        """Classify one frame.

        Implementations should return a default :class:`Observation` rather than
        raise on a recoverable model failure: one bad frame must not end the run.
        """
        ...

    def close(self) -> None:
        """Release model resources. Safe to call more than once."""
        ...
