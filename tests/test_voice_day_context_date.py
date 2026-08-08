"""ITEM 14 — the voice model's own context must not call a previous day "today".

Five of six surfaces share one day-as-chat context because they all run
`AgentRunner.run`, which resolves ONE day_chat_id from the user's local
*now* (`agent_runner.py:1109` → `resolve_day_chat_id_for_now`) and loads
it through the single loader `load_day_context`
(`agent_runner.py:1114`). Voice is the exception: the Realtime model's
OWN context — what it actually SPEAKS from — is assembled on
platform-api by `build_realtime_instructions`, a different path.

That path asked the agent for `/api/day-chats?limit=1`, which is ordered
`local_date DESC` (`app/api/day_chats.py:205-207`), and printed whatever
came back under:

    "# Today's Full Conversation History (N messages across all channels)"

`dc_list[0]` is the NEWEST day chat, not today's. On a voice-first
morning — the user has said nothing yet today — the newest day chat is a
previous day, so the model was handed yesterday's transcript labelled as
today's and narrated it as "earlier today".

`settings.voice_day_context_date_guard` (DEFAULT OFF — it edits the
prompt a live call speaks from) makes two changes and only two:
  1. the header names the day's real date when it is not today;
  2. the direct-date fallback asks for the user's LOCAL date instead of
     the relay's UTC date.
No message is ever dropped from the model's context by either.

The flag-OFF cases below are the anti-vacuity controls: they must stay
GREEN when the production change is reverted.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.config import settings
import app.api.ws_realtime as rt


TORONTO = "America/Toronto"
# Kiritimati is UTC+14: 21:00 UTC on the 3rd is already the 4th there.
KIRITIMATI = "Pacific/Kiritimati"

# 2026-08-03 21:00 UTC — 17:00 local in Toronto (same date), 11:00 on
# 2026-08-04 in Kiritimati (a DIFFERENT date from the relay's UTC date).
FROZEN_UTC = datetime(2026, 8, 3, 21, 0, 0, tzinfo=timezone.utc)


def _freeze(monkeypatch, frozen_utc: datetime = FROZEN_UTC):
    """Freeze ws_realtime's module-level `datetime`. Subclassing keeps
    strftime/fromisoformat working for every other helper in the module."""

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102
            return frozen_utc.astimezone(tz) if tz else frozen_utc.replace(tzinfo=None)

    monkeypatch.setattr(rt, "datetime", _Frozen)


def _wire(monkeypatch, *, newest_day: str, messages: list[dict], tz: str | None,
          guard: bool):
    """Stub every VPS read `build_realtime_instructions` makes.

    Returns the list of (method, path) tuples the builder requested so a
    test can assert WHICH date was asked for.
    """
    monkeypatch.setattr(settings, "voice_day_context_date_guard", guard)
    _freeze(monkeypatch)

    calls: list[tuple[str, str]] = []

    async def fake_get_vps_info(_uid):
        return ("http://agent.test", "fake-agent-key-not-a-secret")

    async def fake_load_identities_local(_uid):
        return None

    async def fake_get_user_tz_name(_uid):
        return tz

    async def fake_get_agent_name(_uid):
        return None

    async def fake_vps_api(url, key, method, path, params=None, json_body=None,
                           timeout=15.0):
        calls.append((method, path))
        if path == "/api/day-chats":
            return [{"id": "dc-1", "local_date": newest_day}] if newest_day else []
        if path.startswith("/api/day-chats/") and path.endswith("/messages"):
            asked = path.split("/")[3]
            return list(messages) if asked == newest_day else []
        if path == "/api/memories":
            return {"memories": []}
        return None

    monkeypatch.setattr(rt, "_get_vps_info", fake_get_vps_info)
    monkeypatch.setattr(rt, "_vps_api", fake_vps_api)
    monkeypatch.setattr(rt, "_load_identities_local", fake_load_identities_local)
    monkeypatch.setattr(rt, "_get_user_tz_name", fake_get_user_tz_name)
    monkeypatch.setattr(rt, "_get_agent_name", fake_get_agent_name)
    return calls


YESTERDAY_MSGS = [
    {"role": "user", "content": "book the dentist for me", "channel": "web",
     "created_at": "2026-08-02T14:00:00"},
    {"role": "assistant", "content": "Booked for Thursday.", "channel": "web",
     "created_at": "2026-08-02T14:00:05"},
]

_TODAY_HEADER = "# Today's Full Conversation History"


# ── The defect ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_guard_on_previous_day_is_not_labelled_today(monkeypatch):
    """Newest day chat is 2026-08-02, the user's local today is
    2026-08-03: the block must NOT claim to be today's conversation."""
    _wire(monkeypatch, newest_day="2026-08-02", messages=YESTERDAY_MSGS,
          tz=TORONTO, guard=True)

    instr = await rt.build_realtime_instructions(str(uuid.uuid4()))

    assert _TODAY_HEADER not in instr
    assert "Conversation from 2026-08-02" in instr
    assert "This is NOT today: today is 2026-08-03" in instr
    # Nothing is dropped — the recall content still reaches the model.
    assert "book the dentist for me" in instr


@pytest.mark.asyncio
async def test_guard_off_previous_day_still_claims_today(monkeypatch):
    """ANTI-VACUITY CONTROL for the header change. With the flag off the
    historical (defective) wording is produced byte-for-byte — this is
    what proves the assertion above is reading the flag, not the weather.
    Stays GREEN when the production change is reverted."""
    _wire(monkeypatch, newest_day="2026-08-02", messages=YESTERDAY_MSGS,
          tz=TORONTO, guard=False)

    instr = await rt.build_realtime_instructions(str(uuid.uuid4()))

    assert f"{_TODAY_HEADER} (2 messages across all channels)" in instr
    assert "book the dentist for me" in instr


@pytest.mark.asyncio
async def test_guard_on_today_is_byte_identical_to_guard_off(monkeypatch):
    """The ordinary case — the user HAS spoken today. The guard must be a
    no-op here; anything else is a live-voice regression."""
    today_msgs = [
        {"role": "user", "content": "morning", "channel": "voice",
         "created_at": "2026-08-03T13:00:00"},
    ]
    _wire(monkeypatch, newest_day="2026-08-03", messages=today_msgs,
          tz=TORONTO, guard=True)
    on = await rt.build_realtime_instructions("11111111-1111-1111-1111-111111111111")

    _wire(monkeypatch, newest_day="2026-08-03", messages=today_msgs,
          tz=TORONTO, guard=False)
    off = await rt.build_realtime_instructions("11111111-1111-1111-1111-111111111111")

    assert on == off
    assert f"{_TODAY_HEADER} (1 messages across all channels)" in on


@pytest.mark.asyncio
async def test_guard_on_unresolvable_tz_keeps_historical_wording(monkeypatch):
    """No User.timezone on the relay ⇒ the guard must stay OFF, not fall
    back to UTC.

    This is the guard's own failure mode, inverted: a Toronto user at
    21:00 local is already on the NEXT UTC date, so a UTC "today" would
    label TODAY's conversation as a previous day. The day chat here is
    dated 2026-08-02 and the relay's UTC date is 2026-08-03 — a UTC
    fallback WOULD fire, so this case only passes if the code genuinely
    refuses to guess."""
    _wire(monkeypatch, newest_day="2026-08-02", messages=YESTERDAY_MSGS,
          tz=None, guard=True)

    instr = await rt.build_realtime_instructions(str(uuid.uuid4()))

    assert _TODAY_HEADER in instr
    assert "This is NOT today" not in instr


@pytest.mark.asyncio
async def test_guard_on_unresolvable_tz_keeps_utc_fallback_date(monkeypatch):
    """Same reason on the fetch side: with no tz there is no local date to
    ask for, so the direct-date fallback must stay on the UTC date."""
    calls = _wire(monkeypatch, newest_day="", messages=[], tz=None, guard=True)

    await rt.build_realtime_instructions(str(uuid.uuid4()))

    asked = [p for _m, p in calls if p.startswith("/api/day-chats/")]
    assert asked == ["/api/day-chats/2026-08-03/messages"], asked


# ── Direct-date fallback: local, not the relay's UTC date ─────────────

@pytest.mark.asyncio
async def test_fallback_asks_for_local_date_not_utc_date(monkeypatch):
    """No day chats at all (empty list) → the builder falls through to the
    direct-date fetch. In Kiritimati (UTC+14) 21:00 UTC on the 3rd is the
    4th locally, so asking the relay's UTC date requests the WRONG day."""
    calls = _wire(monkeypatch, newest_day="", messages=[],
                  tz=KIRITIMATI, guard=True)

    await rt.build_realtime_instructions(str(uuid.uuid4()))

    asked = [p for _m, p in calls if p.startswith("/api/day-chats/")]
    assert asked == ["/api/day-chats/2026-08-04/messages"], asked


@pytest.mark.asyncio
async def test_fallback_guard_off_still_asks_for_utc_date(monkeypatch):
    """ANTI-VACUITY CONTROL for the fallback change. Flag off keeps the
    UTC date verbatim — stays GREEN when the production change is
    reverted."""
    calls = _wire(monkeypatch, newest_day="", messages=[],
                  tz=KIRITIMATI, guard=False)

    await rt.build_realtime_instructions(str(uuid.uuid4()))

    asked = [p for _m, p in calls if p.startswith("/api/day-chats/")]
    assert asked == ["/api/day-chats/2026-08-03/messages"], asked


# ── Header helper, in isolation ───────────────────────────────────────

def test_header_helper_matrix():
    h = rt._day_history_header
    # Guard off (local_today=None) → historical wording, always.
    assert h(3, "2026-01-01", None) == (
        "# Today's Full Conversation History (3 messages across all channels)"
    )
    # Guard on, same day → historical wording, byte-for-byte.
    assert h(3, "2026-08-03", "2026-08-03") == (
        "# Today's Full Conversation History (3 messages across all channels)"
    )
    # Guard on, unknown source date → cannot contradict it, so unchanged.
    assert h(3, None, "2026-08-03").startswith("# Today's Full Conversation History")
    # Guard on, different day → dated, and says so.
    out = h(3, "2026-08-02", "2026-08-03")
    assert "Today's Full Conversation History" not in out
    assert "2026-08-02" in out and "2026-08-03" in out


def test_local_today_str_uses_user_tz(monkeypatch):
    _freeze(monkeypatch)
    assert rt._local_today_str(TORONTO) == "2026-08-03"
    assert rt._local_today_str(KIRITIMATI) == "2026-08-04"
    # Invalid / missing tz degrades to UTC instead of raising.
    assert rt._local_today_str("Not/AZone") == "2026-08-03"
    assert rt._local_today_str(None) == "2026-08-03"
