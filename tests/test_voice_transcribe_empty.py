"""A transcript is speech, or it is nothing — never a sentence about itself.

Why this file exists
--------------------
2026-09-15. `voice_handler.transcribe_voice` answered a phrase Whisper heard
nothing in with the literal string `"(empty transcription)"`, returned in the
same channel as real speech. `POST /api/voice/transcribe` passed it through
(it rejected only bodies beginning "ERROR:"), `api.transcribeAudio` returned
`data.text || ''`, and the app's `useDictation` appends every non-empty
`heard` to the settled transcript. The sentinel was appended to the end of a
user's Persian sentence, in her own composer, in her own words' place.

A sentinel in the content channel is indistinguishable from content, so every
consumer downstream has to know a magic string — and the one that did not is
the one the user saw.

The client half is pinned by `scripts/check-dictation-transcript.js` in the
app repo, which executes `usableTranscript`.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. python -m pytest -q \
        tests/test_voice_transcribe_empty.py
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, Optional

import pytest

from app.agent import voice_handler


# ── A fake Whisper ────────────────────────────────────────────────────
class _Resp:
    def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *a, **kw):
        return _Resp(self._payload)


def _patch_whisper(mp, payload: Dict[str, Any]) -> None:
    import httpx

    mp.setattr(httpx, "AsyncClient", lambda *a, **kw: _Client(payload))


def _transcribe(mp, payload: Dict[str, Any], tmp_path) -> str:
    _patch_whisper(mp, payload)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00" * 64)
    return asyncio.run(voice_handler.transcribe_voice(str(audio), api_key="k"))


# ══════════════════════════════════════════════════════════════════════

def test_an_empty_transcription_is_the_empty_string(monkeypatch, tmp_path):
    assert _transcribe(monkeypatch, {"text": ""}, tmp_path) == ""


def test_whitespace_only_is_the_empty_string(monkeypatch, tmp_path):
    assert _transcribe(monkeypatch, {"text": "   \n "}, tmp_path) == ""


def test_the_sentinel_string_is_gone_from_the_source():
    """Not just unreachable — absent. A sentinel that still exists is one
    `or` away from being returned again, and there is no type that stops it."""
    code = "\n".join(
        line for line in inspect.getsource(voice_handler).splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "(empty transcription)" not in code
    assert "return text\n" in inspect.getsource(voice_handler.transcribe_voice)


def test_real_speech_is_returned_untouched(monkeypatch, tmp_path):
    """Including non-ASCII: the incident's own transcript was Persian, and a
    filter that is really a charset filter would be worse than the bug."""
    for spoken in ("hello there", "سلام، حالت چطوره؟", "  padded  "):
        out = _transcribe(monkeypatch, {"text": spoken}, tmp_path)
        assert out == spoken.strip(), out


def test_transcription_log_carries_a_length_and_a_hash_never_the_words(
    monkeypatch, tmp_path, caplog,
):
    """`logger.info(f"... ({len(text)} chars): {text[:100]}")` shipped up to
    100 characters of the user's speech to Loki on every dictated phrase."""
    secret = "قرار ملاقات با دکتر ساعت چهار"
    with caplog.at_level(logging.INFO):
        out = _transcribe(monkeypatch, {"text": secret}, tmp_path)
    assert out == secret
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in blob, blob
    for word in secret.split():
        assert word not in blob, f"{word!r} leaked: {blob}"
    assert "chars=" in blob and "h=" in blob, blob


def test_the_hash_distinguishes_two_transcripts():
    """The whole debugging use of the old line was telling two runs apart."""
    a = voice_handler._text_fingerprint("one")
    b = voice_handler._text_fingerprint("two")
    assert a != b
    assert a == voice_handler._text_fingerprint("one")
    assert len(a) == 8


def test_error_strings_are_unchanged(monkeypatch, tmp_path):
    """`ERROR:` is the route's 400 contract and is NOT a sentinel in the same
    sense: it is a failure, not a result, and voice.py rejects it explicitly."""
    audio = tmp_path / "missing.wav"
    out = asyncio.run(voice_handler.transcribe_voice(str(audio), api_key="k"))
    assert out.startswith("ERROR:")


# ── The route ─────────────────────────────────────────────────────────

def test_route_reports_emptiness_in_its_own_field():
    """`{text, empty}`: the state becomes machine-readable and unspeakable.
    ADDITIVE — an old client reads `text` and now gets "" (falsy) instead of
    a sentence, which is the behaviour it always should have had."""
    from app.api import voice as voice_route

    src = inspect.getsource(voice_route.transcribe_audio)
    assert '"empty": not text' in src
    assert 'return {"text": text}' not in src
    # The ERROR: → 400 branch must survive: a failure is not an empty result.
    assert 'text.startswith("ERROR:")' in src


def test_no_caller_in_the_repo_still_expects_the_sentinel():
    """The Telegram voice-note path forwarded `transcription` to the agent as
    if it were speech; with the sentinel gone it would have forwarded "". It
    must answer the user instead."""
    # Read, not imported: `python-telegram-bot` is not in the platform test
    # environment, and what is under test is the source, not the runtime.
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "app/agent/telegram_bot.py"
    # Comments may (and do) NAME the sentinel; only code may not use it.
    code = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "(empty transcription)" not in code
    assert "if not transcription.strip():" in code
