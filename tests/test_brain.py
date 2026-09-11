"""Which nudge writer a configuration gets, and which it must never get.

Choosing an offline backend and then turning nudge rewriting on used to build
an OpenAI writer without saying so. The setting reads like "let a model phrase
this"; it should not also read like "and start paying for an API".
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from focus_buddy.brain import build_nudge_writer
from focus_buddy.config import Backend, Settings, SpeechEngine


def _settings(backend, **kwargs):
    return Settings(backend=backend, speech=SpeechEngine.NONE, use_llm_nudges=True, **kwargs)


def _with_local_model(settings, available):
    """Pretend MLX is or is not installed, without importing it."""
    return mock.patch.object(
        type(settings), "local_nudge_model_available", property(lambda self: available)
    )


class TestNeedsOpenAI:
    @pytest.mark.parametrize("backend", [Backend.EDGE, Backend.EDGE_VLM])
    def test_a_local_backend_never_needs_a_key_for_nudges(self, backend):
        assert _settings(backend).needs_openai is False

    def test_cloud_still_does(self):
        assert _settings(Backend.CLOUD).needs_openai is True

    def test_openai_speech_still_does(self):
        settings = Settings(backend=Backend.EDGE, speech=SpeechEngine.OPENAI)
        assert settings.needs_openai is True


class TestBuildNudgeWriter:
    def test_cloud_gets_the_hosted_writer(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        writer = build_nudge_writer(_settings(Backend.CLOUD))
        assert type(writer).__name__ == "OpenAINudgeWriter"

    def test_a_local_backend_without_a_local_model_gets_nothing(self):
        """Nothing means the fixed phrasings, not a quiet call to a paid API."""
        settings = _settings(Backend.EDGE)
        with _with_local_model(settings, False):
            assert build_nudge_writer(settings) is None

    def test_a_local_backend_with_a_local_model_gets_it(self):
        settings = _settings(Backend.EDGE)
        stub = mock.Mock(name="MLXNudgeWriter")
        module = mock.Mock(MLXNudgeWriter=stub)
        with _with_local_model(settings, True):
            with mock.patch.dict(sys.modules, {"focus_buddy.brain.mlx_llm": module}):
                writer = build_nudge_writer(settings)
        assert writer is stub.return_value
        stub.assert_called_once_with(model_id=settings.edge_llm_model)

    @pytest.mark.parametrize("backend", [Backend.EDGE, Backend.EDGE_VLM])
    def test_no_local_backend_ever_builds_the_hosted_writer(self, backend, monkeypatch):
        """The regression this file exists for."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _settings(backend)
        with _with_local_model(settings, False):
            writer = build_nudge_writer(settings)
        assert writer is None


class TestValidationWarning:
    def test_warns_when_the_rewrite_will_not_happen(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _settings(Backend.EDGE)
        with _with_local_model(settings, False):
            settings.validate()
        assert any("fixed phrasings" in warning for warning in settings.warnings)

    def test_silent_when_a_local_model_is_there(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _settings(Backend.EDGE)
        with _with_local_model(settings, True):
            settings.validate()
        assert not any("fixed phrasings" in warning for warning in settings.warnings)

    def test_silent_on_cloud(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _settings(Backend.CLOUD)
        settings.validate()
        assert not any("fixed phrasings" in warning for warning in settings.warnings)
