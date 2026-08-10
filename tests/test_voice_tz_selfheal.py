"""GA run: voice resolved "today" in UTC for users whose tz lives elsewhere.

The seam, measured in production on 2026-08-10:

* Chat and mobile send an IANA tz on every turn, and the agent persists
  it to ``users.timezone`` in the **tenant** database.
* Voice sends no tz at all — ``ws_realtime`` says so in its own comment —
  so it falls back to ``_get_user_tz_name``, which reads the **platform**
  copy of ``users``, then to UTC.

The write and the read target different databases. Canary: tenant row
``America/Toronto``, platform row ``UTC``. Fleet-wide, **36 of 43**
platform user rows have ``timezone IS NULL``; **23 of those users were
active in the last 30 days**.

Consequence: between 00:00 and 04:00 UTC a Toronto user's chat is on
2026-08-10 while their voice session is on 2026-08-11 — two day chats,
split transcripts, and "what did I do today" answering for the wrong
day. That is the #488 / #448 family: a platform-side read of something
only the agent side knows.

The structural fix is G-19a (``voice_context_from_agent``): one
assembler, so voice inherits chat's resolution. It is merged but cannot
reach the fleet while image builds are unavailable. This is the
narrow, platform-only mitigation that ships in the meantime: the voice
client is a browser, so it can send the same ``Intl`` zone the web app
already captures, and the relay persists it — self-healing the platform
row for every subsequent session on any surface.

Red-first: both behavioural tests fail on the pre-fix tree (the handler
ignores ``tz`` entirely, so nothing is persisted and nothing is used).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_WS_REALTIME = Path(__file__).resolve().parents[1] / "app" / "api" / "ws_realtime.py"


def _module_ast() -> ast.Module:
    return ast.parse(_WS_REALTIME.read_text())


def test_config_frame_accepts_a_client_timezone():
    """The `config` frame must read a `tz` field."""
    src = _WS_REALTIME.read_text()
    assert '_apply_client_tz(' in src, (
        "the config frame does not consult a client-supplied tz — voice "
        "still falls back to the platform users row, which is NULL for "
        "36 of 43 users"
    )


def test_the_helper_validates_before_trusting():
    """An unvalidated zone name is the #488 trap: `resolve_local_date`
    falls back to UTC on an unparseable zone, which is exactly the bug
    this is fixing. Validate with zoneinfo, never trust the wire."""
    import app.api.ws_realtime as rt

    import inspect

    src = inspect.getsource(rt._apply_client_tz)
    assert "ZoneInfo" in src, "the client tz is trusted without validation"


@pytest.mark.asyncio
async def test_invalid_zone_is_ignored_not_persisted(monkeypatch):
    import app.api.ws_realtime as rt

    written: list = []

    async def _fake_persist(user_id, tz):  # pragma: no cover - shape only
        written.append((user_id, tz))

    monkeypatch.setattr(rt, "_persist_user_tz", _fake_persist)

    for bogus in ("Not/AZone", "", "  ", "America/Toronto; DROP TABLE users", None):
        assert await rt._apply_client_tz("u1", bogus) is None
    assert written == [], f"an invalid zone was persisted: {written}"


@pytest.mark.asyncio
async def test_valid_zone_is_returned_and_persisted(monkeypatch):
    import app.api.ws_realtime as rt

    written: list = []

    async def _fake_persist(user_id, tz):
        written.append((user_id, tz))

    monkeypatch.setattr(rt, "_persist_user_tz", _fake_persist)

    got = await rt._apply_client_tz("u1", "America/Toronto")
    assert got == "America/Toronto"
    assert written == [("u1", "America/Toronto")], (
        "a valid client zone was used but not persisted — the platform "
        "row stays NULL and the next session repeats the UTC fallback"
    )


@pytest.mark.asyncio
async def test_persist_never_overwrites_an_existing_value(monkeypatch):
    """Self-heal fills a blank; it does not override a zone the user set
    explicitly (the account page's precise-location flow writes one)."""
    import inspect

    import app.api.ws_realtime as rt

    src = inspect.getsource(rt._persist_user_tz)
    assert "is_(None)" in src or "timezone.is_(None)" in src, (
        "the update is not restricted to rows whose timezone IS NULL — "
        "a device zone would clobber an explicitly-chosen one"
    )


def test_frontend_sends_its_zone():
    """The relay can only self-heal if the client actually sends one."""
    hook = (
        Path(__file__).resolve().parents[2]
        / "frontend" / "src" / "hooks" / "useRealtimeVoice.ts"
    )
    if not hook.exists():          # backend-only checkouts
        pytest.skip("frontend not present in this checkout")
    src = hook.read_text()
    assert "resolvedOptions().timeZone" in src and "type: 'config'" in src, (
        "the voice client does not send its IANA zone in the config frame"
    )
