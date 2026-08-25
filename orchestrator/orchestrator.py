import subprocess
import tempfile
import time
import wave
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
from PIL import Image

from brain.llm_lib import (
    LLM_CLOUD,
    LLM_EDGE,
    clean_nudge,
    nudge_looks_valid,
    nudge_reject_reason,
    pick_fallback_nudge,
)
from vision.VLM_lib import VLM_CLOUD, VLM_EDGE

# Prod vs debug intervals (seconds).
SPEAK_COOLDOWN_S = 15  # shortened for faster debugging (was 60)
SPEAK_COOLDOWN_DEBUG_S = 15
SUMMARY_EVERY_S = 60 * 60  # once an hour
SUMMARY_EVERY_DEBUG_S = 3 * 60  # every few minutes while debugging


def _mac_say_to_wav(message: str, wav_path: Path) -> float:
    """Synthesize speech with macOS `say`, return duration in seconds."""
    aiff_path = wav_path.with_suffix(".aiff")
    subprocess.run(["say", "-o", str(aiff_path), message], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff_path), str(wav_path)],
        check=True,
    )
    aiff_path.unlink(missing_ok=True)
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


class Robot:
    """Reachy Mini via the official SDK: camera + speaker."""

    def __init__(self):
        from reachy_mini import ReachyMini

        self.last_frame = None
        print("[robot] connecting to Reachy Mini (needs SDK≈daemon version)…")
        self.mini = ReachyMini()
        # Same idea as reachy_mini_conversation_app's HeadWobbler: audio → head sway.
        # SDK hooks play_sound / push_audio so nudges wiggle without a custom motion loop.
        try:
            self.mini.enable_wobbling()
            print("[robot] connected — camera + speaker + talk-wiggle ready")
        except Exception as exc:
            print(f"[robot] connected — wobbling unavailable ({exc})")

    def get_last_frame(self):
        frame = self.mini.media.get_frame()
        if frame is None:
            return self.last_frame
        # Downscale lightly for RAM only — face ROI crop happens in VLM_EDGE.
        h, w = frame.shape[:2]
        longest = max(w, h)
        if longest > 1280:
            scale = 1280 / longest
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        # SDK returns BGR; VLMs expect RGB PIL.
        self.last_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.last_frame

    def use_speaker(self, message):
        print(message)
        if not message or not str(message).strip():
            return
        with tempfile.TemporaryDirectory(prefix="work_buddy_tts_") as tmp:
            wav_path = Path(tmp) / "nudge.wav"
            try:
                duration_s = _mac_say_to_wav(str(message).strip(), wav_path)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"[robot] TTS failed ({exc}); message printed only")
                return
            self.mini.media.play_sound(str(wav_path))
            # play_sound is non-blocking; keep the file until playback finishes.
            time.sleep(duration_s + 0.25)

    def release(self):
        try:
            if hasattr(self.mini, "disable_wobbling"):
                self.mini.disable_wobbling()
        except Exception:
            pass
        try:
            self.mini.__exit__(None, None, None)
        except Exception as exc:
            print(f"[robot] release warning: {exc}")


class NoRobot:
    # The alternative for debugging with the MAC camera and MAC speaker.
    def __init__(self):
        self.last_frame = None
        self.camera = cv2.VideoCapture(0)  # 0 = the default (built-in) camera
        if not self.camera.isOpened():
            raise RuntimeError("Could not open the Mac camera (check camera permissions for your terminal/IDE)")
        # Prefer the newest frame; default buffers return stale images after slow inference.
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_last_frame(self):
        # OpenCV keeps a buffer of old frames; with ~4s VLM latency that buffer goes stale.
        # Grab-and-discard, then read once so we classify the newest frame.
        for _ in range(4):
            self.camera.grab()
        ok, frame = self.camera.read()
        if not ok:
            return self.last_frame  # keep the previous frame if the read failed
        # OpenCV gives BGR numpy arrays; the VLMs expect a PIL image in RGB
        self.last_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.last_frame

    def use_speaker(self, message):
        print(message)
        subprocess.run(["say", message])  # macOS built-in text-to-speech

    def release(self):
        self.camera.release()


class VLM:
    # Returns the VLM classification whether the person is looking at the screen, touching their face, looking at their phone, biting their nails, at the keyboard, or standing straight.
    def __init__(self, edge=False):
        if edge:
            self.vlm_model = VLM_EDGE()
        else:
            self.vlm_model = VLM_CLOUD()

    def get_vision_response(self, camera_frame):
        t0 = time.perf_counter()
        response = self.vlm_model.get_vision_response(camera_frame)
        elapsed = time.perf_counter() - t0
        print(f"[vlm {elapsed:.2f}s] {response}")
        return response


# VLM flags that warrant a nudge from the buddy.
BAD_BEHAVIORS = ["touching_face", "looking_at_phone", "biting_nails"]

BEHAVIOR_LABELS = {
    "touching_face": "face-touching",
    "looking_at_phone": "phone",
    "biting_nails": "nail-biting",
}

# Warm spoken nudges when the LLM fails validation — pick at random for variety.
NUDGE_SENTENCES = {
    "biting_nails": [
        "Hey, you're biting your nails again.",
        "Oops — ease up on the nail biting.",
        "Hey, give those nails a break.",
        "Caught you biting your nails — easy does it.",
    ],
    "touching_face": [
        "Hey, you're touching your face again.",
        "Oops — hands off your face for a bit.",
        "Hey, try not to touch your face.",
        "Face touch! Maybe keep your hands down for a sec.",
    ],
    "looking_at_phone": [
        "Hey, maybe put the phone down for now.",
        "Oops — phone can wait a minute.",
        "Hey, slide the phone aside when you can.",
        "Phone check — try putting it down for a bit.",
    ],
}


def detected(behavior, vlm_response):
    return f'"{behavior}": True' in vlm_response


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


class MemoryManager:
    # Short-term: recent raw VLM lines (cleared on summary).
    # Long-term: last spoken/printed status string for the day.
    # Counters: rising-edge episodes per bad behavior since day start (not per-frame).
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = ""
        self.event_counters = Counter()
        self._prev_bad = {b: False for b in BAD_BEHAVIORS}
        self.day = date.today()
        self.day_started_at = time.time()

    def maybe_roll_day(self):
        # Call every loop so midnight reset does not depend on a specific code path.
        today = date.today()
        if today == self.day:
            return False
        print(f"[memory] new day {self.day} -> {today}; resetting counters")
        self.short_term_memory = []
        self.long_term_memory = ""
        self.event_counters = Counter()
        self._prev_bad = {b: False for b in BAD_BEHAVIORS}
        self.day = today
        self.day_started_at = time.time()
        return True

    def add_memory(self, memory):
        self.maybe_roll_day()
        self.short_term_memory.append(memory)
        # Count an episode when a bad behavior turns on (False -> True), not every frame.
        for behavior in BAD_BEHAVIORS:
            now = detected(behavior, memory)
            if now and not self._prev_bad[behavior]:
                self.event_counters[behavior] += 1
            self._prev_bad[behavior] = now

    def get_short_term_memory(self):
        return self.short_term_memory

    def get_event_counters(self):
        return dict(self.event_counters)

    def hours_working(self):
        self.maybe_roll_day()
        return (time.time() - self.day_started_at) / 3600.0

    def format_status_summary(self):
        # Deterministic spoken/printed daily status — no LLM needed.
        self.maybe_roll_day()
        worked = _format_duration(time.time() - self.day_started_at)
        parts = [
            f"{BEHAVIOR_LABELS[b]} {self.event_counters[b]} "
            f"{'time' if self.event_counters[b] == 1 else 'times'}"
            for b in BAD_BEHAVIORS
        ]
        return f"You've been working {worked}. " + ", ".join(parts) + "."

    def get_long_term_memory(self):
        self.maybe_roll_day()
        status = self.format_status_summary()
        if self.long_term_memory:
            return f"{self.long_term_memory}\n{status}"
        return status

    def set_long_term_memory(self, summary):
        self.long_term_memory = summary
        self.short_term_memory = []


class LLMBrain:
    # Demo: template nudges only — skip loading the local/cloud LLM.
    USE_LLM = False

    def __init__(
        self,
        memory_manager,
        edge=False,
        speak_cooldown_s=SPEAK_COOLDOWN_S,
        summary_every_s=SUMMARY_EVERY_S,
    ):
        # Lazy-load LLM on first nudge so VLM can start alone; watch [llm Xs] vs [vlm Xs]
        # for thrashing when both share Metal RAM.
        self._edge = edge
        self.llm_model = None
        self.memory_manager = memory_manager
        self.conversation = []
        self.last_spoke = 0.0
        self.last_refreshed = time.time()
        self.speak_cooldown_s = speak_cooldown_s
        self.summary_every_s = summary_every_s

    def _ensure_llm(self):
        if self.llm_model is None:
            print("[llm] loading model (first nudge)…")
            t0 = time.perf_counter()
            self.llm_model = LLM_EDGE() if self._edge else LLM_CLOUD()
            print(f"[llm] loaded in {time.perf_counter() - t0:.1f}s")
        return self.llm_model

    def should_talk(self, vlm_response):
        # Speak only on a bad-behavior detection, and only after the cooldown.
        if time.time() - self.last_spoke < self.speak_cooldown_s:
            return False
        return any(detected(behavior, vlm_response) for behavior in BAD_BEHAVIORS)

    def maybe_talk(self, vlm_response):
        # Returns the message to speak, or None if the buddy should stay quiet.
        if not self.should_talk(vlm_response):
            return None

        habits = [b for b in BAD_BEHAVIORS if detected(b, vlm_response)]
        # Prefer nail-biting over face-touch when both somehow appear.
        if "biting_nails" in habits and "touching_face" in habits:
            habits = [b for b in habits if b != "touching_face"]
        labels = [BEHAVIOR_LABELS[h] for h in habits]
        fallback = pick_fallback_nudge(habits[0], NUDGE_SENTENCES)

        if not self.USE_LLM:
            print(f"[nudge template] {fallback}")
            self.conversation.append(fallback)
            self.last_spoke = time.time()
            return fallback

        memory = self.memory_manager.get_long_term_memory()
        t0 = time.perf_counter()
        try:
            raw = self._ensure_llm().get_llm_response(
                vlm_response,
                memory,
                labels,
                recent_lines=self.conversation,
            )
            message = clean_nudge(raw)
            # Same wording as last nudge → treat as stale and fall back.
            if self.conversation and message.lower() == self.conversation[-1].lower():
                message = ""
            if not nudge_looks_valid(message, habits, vlm_response):
                reason = nudge_reject_reason(message, habits, vlm_response) or "invalid"
                print(
                    f"[llm {time.perf_counter() - t0:.2f}s fallback:{reason}] "
                    f"{raw!r} -> {fallback}"
                )
                message = fallback
            else:
                print(f"[llm {time.perf_counter() - t0:.2f}s] {message}")
        except Exception as exc:
            print(f"[llm error {time.perf_counter() - t0:.2f}s] {exc}; using fallback")
            message = fallback

        self.conversation.append(message)
        self.last_spoke = time.time()
        return message

    def get_conversation(self):
        return self.conversation

    def refresh_long_term_memory(self):
        # Periodic status line from counters (not an LLM essay).
        # Returns the new summary string when one was just produced, else None.
        self.memory_manager.maybe_roll_day()
        if time.time() - self.last_refreshed < self.summary_every_s:
            return None

        summary = self.memory_manager.format_status_summary()
        print(f"[summary] {summary}")
        self.memory_manager.set_long_term_memory(summary)
        self.last_refreshed = time.time()
        return summary


class OrchestratorLoop:
    def __init__(self, use_robot=False, edge=False, debug=False):
        self.robot = Robot() if use_robot else NoRobot()
        self.vlm = VLM(edge=edge)
        self.memory_manager = MemoryManager()
        speak_cooldown = SPEAK_COOLDOWN_DEBUG_S if debug else SPEAK_COOLDOWN_S
        summary_every = SUMMARY_EVERY_DEBUG_S if debug else SUMMARY_EVERY_S
        if debug:
            print(
                f"[debug] speak cooldown {speak_cooldown}s, "
                f"summary every {summary_every}s"
            )
        self.llm_brain = LLMBrain(
            self.memory_manager,
            edge=edge,
            speak_cooldown_s=speak_cooldown,
            summary_every_s=summary_every,
        )
        if not LLMBrain.USE_LLM:
            print("[demo] LLM off — using template nudges only")

    def run(self, loop_seconds=1):
        try:
            while True:
                self.memory_manager.maybe_roll_day()

                camera_frame = self.robot.get_last_frame()
                if camera_frame is None:
                    time.sleep(loop_seconds)
                    continue

                vlm_response = self.vlm.get_vision_response(camera_frame)
                self.memory_manager.add_memory(vlm_response)

                message = self.llm_brain.maybe_talk(vlm_response)
                if message is not None:
                    self.robot.use_speaker(message)

                summary = self.llm_brain.refresh_long_term_memory()
                if summary is not None:
                    self.robot.use_speaker(summary)

                time.sleep(loop_seconds)
        finally:
            release = getattr(self.robot, "release", None)
            if callable(release):
                release()
