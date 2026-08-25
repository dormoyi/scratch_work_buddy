"""Entry points for Focus Buddy.

Two ways in, one loop:

* :class:`FocusBuddy` is the Reachy Mini app, launched from the robot dashboard.
* :func:`main` is the ``focus-buddy`` console script, for running on a laptop.

Heavy imports live inside the functions that need them. The robot dashboard
imports this module just to enumerate installed apps, and that should not pull in
OpenCV or a vision model.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from reachy_mini import ReachyMini, ReachyMiniApp

from .config import Backend, Settings, SpeechEngine

logger = logging.getLogger("focus_buddy")


def setup_logging(verbose: bool = False) -> None:
    """Configure log output for a standalone run."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries log a request line per frame at INFO, which buries ours.
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_loop(body: object, settings: Settings) -> object:
    """Assemble a :class:`~focus_buddy.loop.FocusLoop` around an existing body."""
    from .brain import build_nudge_writer
    from .loop import FocusLoop
    from .perception import build_vision_backend
    from .speech import build_speech_backend

    return FocusLoop(
        body=body,  # type: ignore[arg-type]
        vision=build_vision_backend(settings),
        speech=build_speech_backend(settings),
        settings=settings,
        # Only pay to load a language model when nudges will actually use one.
        nudge_writer=build_nudge_writer(settings) if settings.use_llm_nudges else None,
    )


def _report(settings: Settings) -> list[str]:
    """Validate settings, logging any warnings. Returns fatal errors."""
    errors = settings.validate()
    for warning in settings.warnings:
        logger.warning(warning)
    return errors


class FocusBuddy(ReachyMiniApp):
    """Reachy Mini app that watches for focus-breaking habits and nudges you."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the focus loop until the dashboard stops the app."""
        from .hardware import ReachyBody

        settings = Settings.from_env()
        errors = _report(settings)
        if errors:
            # Surfaced by the dashboard through ReachyMiniApp.error.
            raise RuntimeError("Focus Buddy is not configured correctly:\n- " + "\n- ".join(errors))

        body = ReachyBody(reachy_mini)
        loop = build_loop(body, settings)
        loop.run(stop_event)  # type: ignore[attr-defined]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the standalone runner."""
    parser = argparse.ArgumentParser(
        prog="focus-buddy",
        description="A desk buddy that notices focus-breaking habits and nudges you.",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="use the computer's webcam and speakers instead of a Reachy Mini",
    )
    parser.add_argument(
        "--backend",
        choices=[b.value for b in Backend],
        help="where the models run (default: cloud; edge is macOS on Apple Silicon only)",
    )
    parser.add_argument(
        "--speech",
        choices=[s.value for s in SpeechEngine],
        help="how nudges are spoken (default: openai)",
    )
    parser.add_argument(
        "--llm-nudges",
        action="store_true",
        help="let a language model rephrase nudges instead of using fixed templates",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="short intervals for demos: 15s nudge cooldown, summary every 3 minutes",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG level")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build settings from the environment, then apply CLI overrides.

    Flags win over environment variables so a one-off run never needs the .env
    file edited and then edited back.
    """
    settings = Settings.from_env()
    if args.backend:
        settings.backend = Backend(args.backend)
    if args.speech:
        settings.speech = SpeechEngine(args.speech)
    if args.llm_nudges:
        settings.use_llm_nudges = True
    if args.debug:
        settings.for_debug()
    return settings


def main(argv: list[str] | None = None) -> int:
    """Run Focus Buddy from the command line."""
    args = parse_args(argv)
    setup_logging(args.verbose)

    settings = settings_from_args(args)
    errors = _report(settings)
    if errors:
        for error in errors:
            logger.error(error)
        return 2

    if not args.desktop:
        # Without --desktop, defer to the app framework so the robot connection,
        # media lock and cleanup are handled exactly as the dashboard does it.
        app = FocusBuddy()
        try:
            app.wrapped_run()
        except KeyboardInterrupt:
            app.stop()
        return 0

    from .hardware import DesktopBody

    stop_event = threading.Event()
    try:
        body = DesktopBody()
    except RuntimeError as error:
        logger.error("%s", error)
        return 1

    loop = build_loop(body, settings)
    try:
        loop.run(stop_event)  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        logger.info("Stopping.")
        stop_event.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
