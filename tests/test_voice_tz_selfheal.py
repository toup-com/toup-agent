"""GA run: voice resolved "today" in UTC for users whose tz lives elsewhere.

The seam, measured in production on 2026-08-10:

* Chat and mobile send an IANA tz on every turn, and the agent persists
  it to ``users.timezone`` in the **tenant** database.
* Voice sends no tz at all, so it falls back to ``_get_user_tz_name``,
  which reads the **platform** copy of ``users``, then to UTC.

The write and the read target different databases. Canary: tenant row
``America/Toronto``, platform row ``UTC``. Fleet-wide, **36 of 43**
platform user rows have ``timezone IS NULL``; **23 of those users were
active in the last 30 days**.

Consequence: between 00:00 and 04:00 UTC a Toronto user's chat is on
2026-08-10 while their voice session is on 2026-08-11 — two day chats,
split transcripts, and "what did I do today" answering for the wrong
day. That is the #488 / #448 family: a platform-side read of something
only the agent side knows.

SCOPE — this mitigation covers the WEB voice client only. The mobile
app has its own first-class realtime voice client hitting the same
relay, and it sends NO tz in its config frame; nothing platform-side
can conjure the device's IANA zone. Worse, the exposed population is
precisely mobile: every web page load already PATCHes the platform
timezone via ``captureTimezone()``, so web voice users were mostly
healthy before this fix — while all 11 Sign-in-with-Apple (iOS-only)
users have a NULL platform row. Closing that requires the mobile app
to send its zone (a separate repo and an App Store release) or the
structural fix, G-19a ``voice_context_from_agent``, which makes voice
inherit the agent-side resolution. Documented in GA_LEDGER.

These tests assert BEHAVIOUR, not the presence of a diff. The first
version of this file was six source-text greps; deleting the fix's one
call site left all six green (the ``async def _apply_client_tz`` line
satisfied the substring check). Each test below fails against that
mutation or against the pre-fix tree.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

_WS_REALTIME = Path(__file__).resolve().parents[1] / "app" / "api" / "ws_realtime.py"


# ── The call site must exist, as a CALL ──────────────────────────────


def test_config_frame_actually_calls_the_helper():
    """An `ast.Call` to `_apply_client_tz` outside its own definition.

    A substring grep is satisfied by the `async def` line itself, which
    is how a mutation that deleted the call site — restoring the bug in
    full — passed the previous suite.
    """
    tree = ast.parse(_WS_REALTIME.read_text())

    calls = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_apply_client_tz":
            continue
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", getattr(fn, "attr", None))
            if name == "_apply_client_tz":
                calls.append(node)
    assert calls, (
        "_apply_client_tz is defined but never CALLED — the config frame "
        "ignores the client tz and voice falls back to the platform users "
        "row, which is NULL for 36 of 43 users"
    )


# ── The helper's behaviour ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_zone_is_ignored_not_persisted(monkeypatch):
    """An unvalidated zone name is the #488 trap: `resolve_local_date`
    falls back to UTC on an unparseable zone, which is exactly the bug
    this is fixing. Validate with zoneinfo, never trust the wire."""
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


# ── The persist path, against a real users table ─────────────────────


@pytest.mark.asyncio
async def test_persist_fills_a_null_row(test_user_id):
    from sqlalchemy import select

    import app.api.ws_realtime as rt
    from app.db import User, async_session_maker

    await rt._persist_user_tz(test_user_id, "America/Toronto")

    async with async_session_maker() as db:
        row = (await db.execute(select(User).where(User.id == test_user_id))).scalar_one()
    assert row.timezone == "America/Toronto", (
        f"users.timezone is {row.timezone!r} after self-heal — the write "
        "did not land, so every later session repeats the UTC fallback"
    )


@pytest.mark.asyncio
async def test_persist_never_overwrites_an_existing_value(test_user_id):
    """Self-heal fills a blank; it does not override a zone the user set
    explicitly (the account page's precise-location flow writes one)."""
    from sqlalchemy import select, update

    import app.api.ws_realtime as rt
    from app.db import User, async_session_maker

    async with async_session_maker() as db:
        await db.execute(
            update(User).where(User.id == test_user_id).values(timezone="Asia/Tokyo")
        )
        await db.commit()

    await rt._persist_user_tz(test_user_id, "America/Toronto")

    async with async_session_maker() as db:
        row = (await db.execute(select(User).where(User.id == test_user_id))).scalar_one()
    assert row.timezone == "Asia/Tokyo", (
        "a device-detected zone clobbered one the user chose deliberately"
    )


# ── The consequence: the day actually resolves differently ───────────


def test_day_resolution_differs_with_and_without_tz():
    """The bug's observable: 23:30Z→01:10Z straddles UTC midnight but is
    19:30→21:10 in Toronto — the SAME local day. With tz=None the relay
    declines session reuse there and mints a fresh 'Voice Session',
    splitting the user's transcript at 8pm local."""
    import app.api.ws_realtime as rt

    started = datetime(2026, 8, 10, 23, 30, tzinfo=timezone.utc)
    now = datetime(2026, 8, 11, 1, 10, tzinfo=timezone.utc)

    assert rt._same_local_day(started, now, "America/Toronto") is True
    assert rt._same_local_day(started, now, None) is False
    # And an invalid zone falls back to UTC rather than raising mid-relay.
    assert rt._same_local_day(started, now, "Not/AZone") is False


# ── The web client sends one (the mobile client does NOT — see module
#    docstring; that gap is closed in the app repo, not here) ─────────


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
