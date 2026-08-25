"""Focus Buddy: a Reachy Mini desk companion that nudges you about focus-breaking habits."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("focus_buddy")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
