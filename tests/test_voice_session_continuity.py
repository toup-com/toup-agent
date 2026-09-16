"""One voice conversation per LOCAL DAY, not one per socket.

The relay has always announced the Conversation it bound a call to
(`{"type":"session_id"}`) and the mobile client has never had a handler for it —
so no client ever supplied one back, and every call AND every mid-call reconnect
minted a fresh `channel='voice'` row. Two consequences, both user-visible:

  * a durable task started thirty seconds earlier became unaddressable after a
    reconnect ("This task is not available in this conversation"), because task
    scope was keyed on the raw conversation id;
  * a day's calls fanned out into a dozen Conversations nobody can name.

The server half is a create-on-absence default: with no usable id, reuse the
user's own same-local-day voice Conversation before making another. Pure/faked —
no DB, no network; platform sweep.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest


def _install(monkeypatch, *, get_session=None, listed=None, created_id="new-id",
             tz_name="America/Toronto"):
    """Fake the three things `_get_or_create_voice_session` touches."""
    from app.api import ws_realtime

    calls: list[tuple] = []

    async def _fake_vps_api(agent_url, key, method, path, params=None, json_body=None, **kw):
        calls.append((method, path, params, json_body))
        if method == "GET" and path.startswith("/api/sessions/"):
            return get_session
        if method == "GET" and path == "/api/sessions":
            return {"sessions": listed or [], "total_count": len(listed or [])}
        if method == "POST" and path == "/api/sessions":
            return {"id": created_id}
        return None

    async def _fake_info(_uid):
        return ("https://agent.test", "key")

    async def _fake_tz(_uid):
        return tz_name

    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_info)
    monkeypatch.setattr(ws_realtime, "_get_user_tz_name", _fake_tz)
    return calls


def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=None).isoformat()


@pytest.mark.asyncio
async def test_a_client_supplied_id_from_today_is_reused(monkeypatch):
    from app.api.ws_realtime import _get_or_create_voice_session

    now = datetime.now(timezone.utc)
    calls = _install(monkeypatch, get_session={"id": "sess-a", "started_at": _iso(now)})

    assert await _get_or_create_voice_session("u1", "sess-a") == "sess-a"
    assert not any(c[0] == "POST" for c in calls), "reuse must not create a row"


@pytest.mark.asyncio
async def test_with_no_id_todays_voice_conversation_is_reused(monkeypatch):
    """THE FIX. A reconnect that supplies nothing must land on the same row."""
    from app.api.ws_realtime import _get_or_create_voice_session

    now = datetime.now(timezone.utc)
    # `_iso(now)`, not `now - 5 min`: the reuse rule is SAME LOCAL DAY in
    # America/Toronto (which `_install` pins), so a five-minute-old row is
    # yesterday for the five minutes after local midnight and this test failed
    # for that window every day. Age is not what it is about.
    calls = _install(monkeypatch, listed=[
        {"id": "today-voice", "started_at": _iso(now)},
    ])

    assert await _get_or_create_voice_session("u1", None) == "today-voice"
    assert not any(c[0] == "POST" for c in calls)
    listed = [c for c in calls if c[1] == "/api/sessions"]
    assert listed and listed[0][2]["channel"] == "voice", "the lookup must be channel-scoped"


@pytest.mark.asyncio
async def test_yesterdays_voice_conversation_is_not_reused(monkeypatch):
    """A day is the unit. Reusing yesterday's would file today's call into it."""
    from app.api.ws_realtime import _get_or_create_voice_session

    now = datetime.now(timezone.utc)
    _install(monkeypatch, listed=[
        {"id": "old-voice", "started_at": _iso(now - timedelta(days=2))},
    ], created_id="fresh")

    assert await _get_or_create_voice_session("u1", None) == "fresh"


@pytest.mark.asyncio
async def test_a_lookup_failure_never_costs_the_user_a_call(monkeypatch):
    """Degrading to "create one" is exactly the pre-R46 behaviour; failing the
    call would be a new way to lose a conversation."""
    from app.api import ws_realtime

    async def _boom(*a, **kw):
        if a[2] == "GET" and a[3] == "/api/sessions":
            raise RuntimeError("tenant down")
        if a[2] == "POST":
            return {"id": "fallback"}
        return None

    async def _fake_info(_uid):
        return ("https://agent.test", "key")

    async def _fake_tz(_uid):
        return "UTC"

    monkeypatch.setattr(ws_realtime, "_vps_api", _boom)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_info)
    monkeypatch.setattr(ws_realtime, "_get_user_tz_name", _fake_tz)

    assert await ws_realtime._get_or_create_voice_session("u1", None) == "fallback"


@pytest.mark.asyncio
async def test_a_stale_client_id_falls_through_to_todays_row(monkeypatch):
    """A phone that cached yesterday's id must not resurrect yesterday's row."""
    from app.api.ws_realtime import _get_or_create_voice_session

    now = datetime.now(timezone.utc)
    _install(
        monkeypatch,
        get_session={"id": "stale", "started_at": _iso(now - timedelta(days=3))},
        listed=[{"id": "today-voice", "started_at": _iso(now)}],
    )
    assert await _get_or_create_voice_session("u1", "stale") == "today-voice"


# ── Identity ──────────────────────────────────────────────────────────────

def test_voice_client_msg_id_is_deterministic_and_session_scoped():
    from app.api.ws_realtime import voice_client_msg_id

    a = voice_client_msg_id("sess-1", "item:abc")
    assert a == voice_client_msg_id("sess-1", "item:abc"), "a replay must collide"
    assert a != voice_client_msg_id("sess-2", "item:abc"), "sessions must not collide"
    assert a != voice_client_msg_id("sess-1", "item:def")
    uuid.UUID(a)  # a real uuid, not a concatenation


def test_voice_client_msg_id_separates_the_two_halves_of_one_turn():
    """The question and the answer are two rows and must never share a key."""
    from app.api.ws_realtime import voice_client_msg_id

    assert (voice_client_msg_id("s", "item:1")
            != voice_client_msg_id("s", "response:1"))
