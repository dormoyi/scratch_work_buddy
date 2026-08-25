---
title: Focus Buddy
emoji: 👀
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
short_description: A Reachy Mini that notices when you touch your face and says something.
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Focus Buddy

A [Reachy Mini](https://huggingface.co/pollen-robotics) app that sits on your desk, watches
for the small habits that break your focus — touching your face, biting your nails, drifting
onto your phone — and says something about it. Once an hour it tells you how the day is going.

It is deliberately not a productivity dashboard. There is no score, nothing is uploaded, and
the only record is a per-day tally that resets at midnight.

```
camera frame ──► vision backend ──► Observation ──► memory (episode counts)
                                          │
                                          └──► nudge (template or LLM) ──► speech ──► speaker
```

## Quick start

```bash
git clone https://github.com/dormoyi/focus_buddy
cd focus_buddy
pip install -e .

cp .env.example .env      # then add your OPENAI_API_KEY
focus-buddy --desktop     # run on your laptop webcam, no robot needed
```

With a Reachy Mini connected:

```bash
focus-buddy               # connects to the robot through the SDK
```

Or install it on the robot and launch **Focus Buddy** from the Reachy Mini dashboard.

## What it watches for

| Habit | It says something when |
|---|---|
| `touching_face` | a hand is in contact with your face, anywhere but the mouth |
| `biting_nails` | your fingertips are at or in your mouth |
| `looking_at_phone` | you are holding a phone and looking at it |

It also tracks `looking_at_screen`, `at_keyboard` and a `straight_posture` score, which feed
the daily summary but never trigger a nudge on their own.

Nudges are rate-limited (one per minute by default) and counted as **episodes**, not frames:
a thirty-second phone check is one distraction, not thirty.

## Configuration

Everything is set through environment variables, usually via `.env`. See
[`.env.example`](.env.example) for the full list. The ones that matter:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required unless you run fully local on a Mac |
| `FOCUS_BUDDY_BACKEND` | `cloud` | `cloud` or `edge` (see below) |
| `FOCUS_BUDDY_SPEECH` | `openai` | `openai`, `macos`, or `none` |
| `FOCUS_BUDDY_NUDGE_COOLDOWN_S` | `60` | minimum quiet time between nudges |
| `FOCUS_BUDDY_USE_LLM_NUDGES` | `false` | let a model rephrase nudges |

Command-line flags override the environment for a single run:

```bash
focus-buddy --desktop --debug          # 15s cooldown, summary every 3 minutes
focus-buddy --desktop --speech none    # watch silently, log nudges
focus-buddy --desktop --llm-nudges -v  # model-phrased nudges, debug logging
```

### Speech

`openai` (default) uses the OpenAI text-to-speech API. It is the default because it is the
only option that behaves identically on a laptop and on the Raspberry Pi inside a Reachy Mini
wireless.

`macos` uses the built-in `say` command: offline, free, and **macOS only**. Choose it
explicitly with `FOCUS_BUDDY_SPEECH=macos` when you are running on a Mac and would rather not
spend API calls on speech.

`none` keeps the buddy watching but silent; nudges are written to the log.

### Cloud vs edge

> [!IMPORTANT]
> **Edge mode requires macOS on Apple Silicon. There is no supported way to run it anywhere
> else** — not on Linux, not on Windows, and *not on the Raspberry Pi inside a Reachy Mini
> wireless*. It runs quantised models through [MLX](https://github.com/ml-explore/mlx), which
> is built for Apple GPUs and publishes no ARM-Linux or x86-Linux builds. If you are running
> the app on the robot itself, you want `cloud`.

**`cloud`** (default) sends each frame to an OpenAI vision model. Works on every platform,
needs a network connection and an API key, and costs roughly a fraction of a cent per frame.

**`edge`** runs everything locally on your Mac: a 4-bit SmolVLM2-2.2B for vision, and
optionally a 4-bit Llama-3.2-1B to phrase nudges. Nothing leaves the machine and there is no
per-frame cost. To use it:

```bash
pip install -e '.[edge]'
FOCUS_BUDDY_BACKEND=edge focus-buddy --desktop --speech macos
```

The first run downloads several gigabytes of weights. Set `HF_HOME` to control where they
land. On an 8GB Mac, expect a few seconds per frame and keep `FOCUS_BUDDY_USE_LLM_NUDGES=false`
unless you have memory to spare — the two models compete for unified memory.

Edge mode does not send the whole frame to the model. It finds your face with an OpenCV Haar
cascade and crops around it, biased downward and toward the direction your head is turned, so
your hands and any phone stay in frame. Small models are far more accurate on that crop than
on a wide desk shot. Set `FOCUS_BUDDY_DEBUG_FRAME=debug/frame.jpg` to see exactly what the
model is being shown; if detection is behaving strangely, look there first.

## How it is put together

| Module | Responsibility |
|---|---|
| `observations.py` | `Observation` and `Habit` — the typed result every backend returns |
| `perception/` | frame → `Observation`. `framing` crops, `captions` classifies prose |
| `brain/` | optional LLM rephrasing of a nudge |
| `nudges.py` | nudge templates and the validation a model's output must pass |
| `memory.py` | per-day episode counters and the spoken summary |
| `speech/` | sentence → WAV file |
| `hardware/` | `ReachyBody` and `DesktopBody`: a camera and a speaker |
| `loop.py` | the tick: look, decide, occasionally speak |
| `main.py` | `FocusBuddy` (the Reachy Mini app) and the `focus-buddy` CLI |

Each of the four backend families sits behind a `Protocol` and is built by a small factory
from `Settings`. That is what lets the whole loop be tested with fakes, and what makes adding
a new speech engine or vision model a single new file rather than a new `if` branch.

Two design choices worth knowing about:

**Nudges are templates by default.** A language model can rephrase them
(`FOCUS_BUDDY_USE_LLM_NUDGES=true`), but anything it produces has to pass `nudges.is_valid`
first — right length, actually mentions the habit it is supposed to be about, and does not
contradict the frame. Whatever fails is replaced by the template. A buddy that is confidently
wrong about you is worse than one that repeats itself.

**The summary is computed, not generated.** The numbers in "you've been working 2h,
nail-biting 3 times" are facts about your day, so they are formatted from counters rather than
paraphrased by a model that could get them wrong.

## Development

```bash
pip install -e . --group dev
pytest                    # 80 tests, no network, no camera, no models
ruff check . && ruff format --check .
mypy src/
```

The test suite runs the entire loop against fakes (`tests/conftest.py`), so
`pytest` exercises the real decision logic in under a second.

`tests/test_captions.py` deserves a note: every case in it is something a real vision model
actually said. When you tune a regex in `perception/captions.py`, those tests are the only
thing standing between you and silently breaking detection for a phrasing you forgot about.

To compare local vision checkpoints:

```bash
python tools/vlm_bakeoff.py --image debug/frame.jpg
```

## Privacy

Camera frames are classified and discarded. Nothing is written to disk unless you set
`FOCUS_BUDDY_DEBUG_FRAME`. In `edge` mode no image ever leaves your machine; in `cloud` mode
each frame is sent to OpenAI for classification and is subject to their data policy. Habit
counts live in memory only and reset at midnight.

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
