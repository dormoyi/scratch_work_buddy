# Contributing to Focus Buddy

Thanks for taking a look. Issues and pull requests are both welcome.

## Getting set up

```bash
git clone https://github.com/dormoyi/focus_buddy
cd focus_buddy
python -m venv .venv && source .venv/bin/activate
pip install -e . --group dev
cp .env.example .env      # add your OPENAI_API_KEY
```

Check that everything works:

```bash
pytest                          # no network, camera or models needed
ruff check . && ruff format --check .
mypy src/
```

## Running it

```bash
focus-buddy --desktop --debug   # laptop webcam, short intervals
focus-buddy                     # connect to a Reachy Mini
```

`--debug` shortens the nudge cooldown to 15 seconds and the summary interval to three
minutes, which makes a full demo take a minute rather than an hour.

## The shape of the code

The loop is one tick: grab a frame, classify it into an `Observation`, record it, and
sometimes speak. Four things are pluggable, each behind a `Protocol` with a small factory
that reads `Settings`:

| Thing | Protocol | Factory |
|---|---|---|
| Vision | `perception.base.VisionBackend` | `perception.build_vision_backend` |
| Nudge phrasing | `brain.base.NudgeWriter` | `brain.build_nudge_writer` |
| Speech | `speech.base.SpeechBackend` | `speech.build_speech_backend` |
| Camera + speaker | `hardware.base.Body` | constructed in `main` |

Adding a backend means adding a file and one branch in the relevant factory. It should not
mean touching `loop.py`.

Two conventions worth respecting:

- **Import heavy dependencies inside functions.** The robot dashboard imports `main.py` just
  to list installed apps; it should not pull in OpenCV or a model as a side effect.
- **Never read the environment at import time.** Configuration is resolved once, into
  `Settings`, and passed down. This is what keeps the tests hermetic.

## Tests

`tests/conftest.py` has fakes for every collaborator, so the whole loop can run without a
camera, a robot, a model or a network. New behaviour in `loop.py` should be testable there.

`tests/test_captions.py` is a regression suite: every case is something a real vision model
actually said. If you tune a pattern in `perception/captions.py`, run it - those tests are
the only thing that catches a change that silently breaks detection for a phrasing you
were not thinking about. When you fix a new misclassification, add the caption that caused
it.

## Pull requests

- Keep `ruff`, `mypy` and `pytest` green.
- One concern per PR.
- If you change what the buddy says or when it says it, say so in the description - that is
  the part users actually experience.

## Code of conduct

Be decent to each other. Harassment or personal attacks are not welcome here, and
maintainers will remove contributions and contributors that involve them.
