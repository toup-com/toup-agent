"""Regression test for TKT-LAT-012 — TTS streaming flag + [PERF] tags."""

from __future__ import annotations

import inspect

from app.agent import voice_handler


def test_tts_streaming_enabled_flag_present():
    from app.config import settings
    assert hasattr(settings, "tts_streaming_enabled")
    assert settings.tts_streaming_enabled is False


def test_synthesize_speech_emits_buffered_perf_tag():
    src = inspect.getsource(voice_handler.synthesize_speech)
    assert "[PERF] tts=buffered" in src


def test_stream_tts_elevenlabs_emits_streaming_perf_tag():
    src = inspect.getsource(voice_handler.stream_tts_elevenlabs)
    assert "[PERF] tts=streaming" in src
    assert "first_byte_ms" in src


def test_streaming_function_remains_async_generator():
    assert inspect.isasyncgenfunction(voice_handler.stream_tts_elevenlabs)
