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
import time

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
    # These libraries bury our lines under per-frame chatter: one request line
    # per frame from the HTTP clients, and a joint-pose frame at ~50Hz from the
    # robot SDK's websocket, which at -v is thousands of lines a minute.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_loop(body: object, settings: Settings, owns_body: bool = True) -> object:
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
        owns_body=owns_body,
    )


def _report(settings: Settings) -> list[str]:
    """Validate settings, logging any warnings. Returns fatal errors."""
    errors = settings.validate()
    for warning in settings.warnings:
        logger.warning(warning)
    return errors


# How often to re-read the configuration while waiting for the settings page.
_CONFIG_POLL_S = 2.0


class _EitherEvent:
    """Reads as set when either of two events is set.

    The focus loop takes a single stop event, and it should stop both when the
    dashboard says so and when saved settings need applying. Rather than teach
    the loop about reloading, give it something that is true for either reason.
    """

    def __init__(self, *events: threading.Event) -> None:
        """Wrap the events to watch."""
        self._events = events

    def is_set(self) -> bool:
        """Whether any wrapped event is set."""
        return any(event.is_set() for event in self._events)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout``, returning as soon as any event is set.

        Polls rather than blocking on one event, because blocking on the stop
        event would ignore a reload request for a whole poll interval.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            remaining = 0.2 if deadline is None else min(0.2, deadline - time.monotonic())
            self._events[0].wait(max(0.0, remaining))
        return True


class FocusBuddy(ReachyMiniApp):
    """Reachy Mini app that watches for focus-breaking habits and nudges you."""

    # Serving this makes the dashboard show a settings page for the app. Without
    # it there is no way to supply an API key to an app-store install, which is
    # what made the first version exit immediately on a fresh robot.
    custom_app_url = "http://0.0.0.0:7860/"

    def __init__(self, running_on_wireless: bool = False, settings: Settings | None = None) -> None:
        """Create the app.

        The dashboard constructs this with no arguments, so ``settings`` defaults
        to the environment. The CLI passes its own already-validated settings in,
        which is what makes flags like ``--debug`` apply on the robot and not
        only under ``--desktop``.
        """
        super().__init__(running_on_wireless)
        self._settings = settings
        # Always, not just when the configuration is broken. Attaching only in
        # the unconfigured case meant the settings page worked exactly until it
        # had been used once: afterwards main() injected valid settings, the
        # endpoints were never added, and the page sat on "Loading..." forever
        # because its first fetch 404ed.
        from .settings_page import attach

        attach(self.settings_app)

    def _await_configuration(self, stop_event: threading.Event) -> Settings | None:
        """Block until the settings page yields a usable configuration.

        Returns None if the app is stopped first. Waiting rather than raising is
        the whole point: an unconfigured app has to stay alive long enough for
        someone to open its settings page and type a key into it.
        """
        announced = False
        while not stop_event.is_set():
            settings = Settings.from_env()
            errors = _report(settings)
            if not errors:
                return settings
            if not announced:
                from .settings_page import RUNTIME

                RUNTIME.waiting("; ".join(errors))
                # Also surfaced by the dashboard through ReachyMiniApp.error.
                self.error = "Focus Buddy needs configuring:\n- " + "\n- ".join(errors)
                logger.error(
                    "%s\nWaiting for the settings page at %s", self.error, self.custom_app_url
                )
                announced = True
            stop_event.wait(_CONFIG_POLL_S)
        return None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the focus loop until the dashboard stops the app."""
        from .hardware import ReachyBody

        settings = self._settings
        if settings is None:
            # Dashboard launch: nothing has read the environment for us yet.
            settings = self._await_configuration(stop_event)
            if settings is None:
                return
            self.error = ""

        from .settings_page import RUNTIME

        # Rebuilt whenever the settings page asks for it, so changing the
        # backend or the voice does not mean stopping and starting the app by
        # hand. The dashboard's stop event ends the outer loop for good.
        body = None
        body_wobble: bool | None = None
        try:
            while not stop_event.is_set():
                RUNTIME.reload_requested.clear()
                # The robot connection outlives a settings change. Only wobble
                # is fixed at construction, so only wobble forces a reconnect.
                if body is None or body_wobble != settings.wobble:
                    if body is not None:
                        body.close()
                    RUNTIME.waiting("connecting to the robot")
                    body = ReachyBody(reachy_mini, wobble=settings.wobble)
                    body_wobble = settings.wobble

                loop = build_loop(body, settings, owns_body=False)
                RUNTIME.watching(loop)
                try:
                    loop.run(_EitherEvent(stop_event, RUNTIME.reload_requested))  # type: ignore[attr-defined]
                finally:
                    RUNTIME.stopped()

                if stop_event.is_set() or self._settings is not None:
                    # Stopped for real, or running with settings the CLI pinned,
                    # which the page has no business overriding.
                    return
                logger.info("Applying new settings")
                settings = self._await_configuration(stop_event)
                if settings is None:
                    return
        finally:
            if body is not None:
                body.close()


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


def has_overrides(args: argparse.Namespace) -> bool:
    """Whether the user actually asked for something on the command line.

    Only then are settings pinned. Handing resolved settings over regardless
    made the app treat every dashboard launch as CLI-pinned, so a reload
    request ended the process instead of rebuilding -- the app exited whenever
    someone pressed Save.
    """
    return bool(args.backend or args.speech or args.llm_nudges or args.debug)


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

    if not args.desktop:
        # Without --desktop, defer to the app framework so the robot connection,
        # media lock and cleanup are handled exactly as the dashboard does it.
        #
        # This is also the path the dashboard itself takes: it starts an app with
        # `python -m <entry point module>`, which for us is this file. So it must
        # survive a bad configuration rather than exiting -- an app that exits
        # cannot serve the settings page that would fix it, which is exactly how
        # a fresh install used to fail with code 2 and no way forward. Passing no
        # settings makes FocusBuddy wait and re-read them as the page saves.
        if errors:
            for error in errors:
                logger.error(error)
            logger.error("Waiting for the settings page at %s", FocusBuddy.custom_app_url)
        # Hand settings over only when flags were given, so they survive; with
        # none, leave the app free to re-read them as the settings page saves.
        pinned = settings if has_overrides(args) and not errors else None
        app = FocusBuddy(settings=pinned)
        try:
            app.wrapped_run()
        except KeyboardInterrupt:
            app.stop()
        return 0

    # --desktop is a person at a terminal with flags and a shell, so it fails
    # fast instead of waiting for a page nobody is going to open.
    if errors:
        for error in errors:
            logger.error(error)
        return 2

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
