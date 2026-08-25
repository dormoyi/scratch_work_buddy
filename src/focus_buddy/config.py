"""Runtime configuration, resolved once at startup from env vars and CLI flags."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum


class Backend(str, Enum):
    """Where the vision and language models run."""

    CLOUD = "cloud"
    EDGE = "edge"


class SpeechEngine(str, Enum):
    """How spoken nudges are synthesised."""

    OPENAI = "openai"
    MACOS = "macos"
    NONE = "none"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Everything tunable about a run.

    Built once in :func:`focus_buddy.main.main` (or in the app entry point) and
    passed down explicitly, so no module reads the environment at import time.
    """

    backend: Backend = Backend.CLOUD
    speech: SpeechEngine = SpeechEngine.OPENAI

    # How often to classify a frame.
    poll_interval_s: float = 1.0
    # Minimum quiet time between two spoken nudges.
    nudge_cooldown_s: float = 60.0
    # How often to speak the running daily summary.
    summary_every_s: float = 3600.0

    # Let a language model phrase the nudge. Off by default: template nudges are
    # instant, free, and never say anything strange to the user.
    use_llm_nudges: bool = False

    # Model identifiers.
    cloud_vision_model: str = "gpt-4o-mini"
    cloud_llm_model: str = "gpt-4o-mini"
    cloud_tts_model: str = "gpt-4o-mini-tts"
    cloud_tts_voice: str = "alloy"
    edge_vision_model: str = "smdesai/SmolVLM2-2.2B-Instruct-4bit"
    edge_llm_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"

    # Write the exact crop the edge VLM saw to this path, for debugging framing.
    debug_frame_path: str | None = None

    # Populated by validate(); non-fatal notes worth showing the user.
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables (and a .env file if present)."""
        from dotenv import load_dotenv

        load_dotenv()

        backend = Backend(os.getenv("FOCUS_BUDDY_BACKEND", Backend.CLOUD.value).lower())
        speech = SpeechEngine(os.getenv("FOCUS_BUDDY_SPEECH", SpeechEngine.OPENAI.value).lower())

        return cls(
            backend=backend,
            speech=speech,
            poll_interval_s=_env_float("FOCUS_BUDDY_POLL_INTERVAL_S", 1.0),
            nudge_cooldown_s=_env_float("FOCUS_BUDDY_NUDGE_COOLDOWN_S", 60.0),
            summary_every_s=_env_float("FOCUS_BUDDY_SUMMARY_EVERY_S", 3600.0),
            use_llm_nudges=_env_bool("FOCUS_BUDDY_USE_LLM_NUDGES", False),
            cloud_vision_model=os.getenv("FOCUS_BUDDY_CLOUD_VISION_MODEL", "gpt-4o-mini"),
            cloud_llm_model=os.getenv("FOCUS_BUDDY_CLOUD_LLM_MODEL", "gpt-4o-mini"),
            cloud_tts_model=os.getenv("FOCUS_BUDDY_TTS_MODEL", "gpt-4o-mini-tts"),
            cloud_tts_voice=os.getenv("FOCUS_BUDDY_TTS_VOICE", "alloy"),
            edge_vision_model=os.getenv(
                "FOCUS_BUDDY_EDGE_VISION_MODEL", "smdesai/SmolVLM2-2.2B-Instruct-4bit"
            ),
            edge_llm_model=os.getenv(
                "FOCUS_BUDDY_EDGE_LLM_MODEL", "mlx-community/Llama-3.2-1B-Instruct-4bit"
            ),
            debug_frame_path=os.getenv("FOCUS_BUDDY_DEBUG_FRAME") or None,
        )

    def for_debug(self) -> Settings:
        """Shorten the intervals so a demo shows all behaviour within a minute."""
        self.nudge_cooldown_s = 15.0
        self.summary_every_s = 180.0
        return self

    @property
    def needs_openai(self) -> bool:
        """Whether this configuration will call the OpenAI API."""
        return (
            self.backend is Backend.CLOUD
            or self.speech is SpeechEngine.OPENAI
            or (self.use_llm_nudges and self.backend is Backend.CLOUD)
        )

    def validate(self) -> list[str]:
        """Return fatal configuration errors; also records non-fatal warnings.

        Checked before any model is loaded so misconfiguration fails in under a
        second rather than after a multi-gigabyte download.
        """
        errors: list[str] = []
        self.warnings = []

        if self.needs_openai and not os.getenv("OPENAI_API_KEY"):
            errors.append(
                "OPENAI_API_KEY is not set, but this configuration needs the OpenAI API "
                f"(backend={self.backend.value}, speech={self.speech.value}). "
                "Copy .env.example to .env and add your key."
            )

        if self.backend is Backend.EDGE and sys.platform != "darwin":
            errors.append(
                "Edge mode requires macOS on Apple Silicon: it runs the local models "
                "through MLX, which has no builds for Linux or Windows (including the "
                "Raspberry Pi inside a Reachy Mini wireless). "
                "Use FOCUS_BUDDY_BACKEND=cloud instead."
            )

        if self.speech is SpeechEngine.MACOS and sys.platform != "darwin":
            errors.append(
                "FOCUS_BUDDY_SPEECH=macos needs the macOS `say` command. "
                "Use FOCUS_BUDDY_SPEECH=openai instead."
            )

        if self.speech is SpeechEngine.NONE:
            self.warnings.append("Speech is disabled; nudges will only be logged.")

        if self.nudge_cooldown_s < 5:
            self.warnings.append(
                f"nudge_cooldown_s={self.nudge_cooldown_s}s is very short; "
                "the buddy will be talkative."
            )

        return errors
