"""A corrected row has to reach an OPEN thread (R48, A4-4/A4-8).

`POST /api/sessions/{id}/messages` already upserts on
(conversation_id, client_msg_id, role) — the server side of revise-in-place was
finished; the live-delivery half never was. `message_frame` accepts `revision`
and documents exactly this failure ("without a monotonic marker the CORRECTED
answer is the one that is discarded"), and the route never passed it. That was
latent while no producer repeated a key. R48's Live path revises one utterance
as its fragments settle, so a repeated key is now the normal case and the
consequence is a thread stuck on the first two words of every sentence until a
manual refresh.

Three properties are pinned here, plus the provenance field the same round
adds:

* the broadcast carries the revision it was given;
* a rewrite does not erase what the first write established — a revision
  carries provenance, not the media card;
* one key produces one row, and the keyed write is serialised against a
  concurrent one (A4-8) rather than racing it into two.

Needs RUN_MODE=agent: conversations / messages / day_chats are AGENT_ONLY and
this drives the REAL writer. See COVERAGE_DEBT.txt.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from app.config import settings
from app.db import async_session_maker
from app.db.models import Conversation, Message, User
from app.schemas import SessionMessageCreate

AGENT_KEY = "test-agent-key-revision"


class _Req:
    def __init__(self, headers: dict | None = None):
        self.headers = dict(headers or {})


@pytest.fixture(autouse=True)
def _agent_key(monkeypatch):
    monkeypatch.setattr(settings, "agent_api_key", AGENT_KEY, raising=False)
    yield


async def _seed():
    user_id, conv_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"rev-{user_id[:8]}@example.test",
                    hashed_password="x", name="Revision Test"))
        db.add(Conversation(id=conv_id, user_id=user_id, title="Voice",
                            channel="voice", message_count=0, total_tokens=0))
        await db.commit()
    return user_id, conv_id


async def _post(conv_id, user_id, body, frames):
    from app.api import sessions as sessions_mod
    import app.api.ws_chat as ws_chat

    async def _capture(uid, frame):
        frames.append(frame)

    orig = ws_chat.broadcast_to_user
    ws_chat.broadcast_to_user = _capture
    try:
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            out = await sessions_mod.create_session_message(
                session_id=conv_id, body=body,
                request=_Req({"x-agent-key": AGENT_KEY}),
                current_user=user, db=db,
            )
        # The broadcast is fire-and-forget by design; give it its tick.
        await asyncio.sleep(0)
        return out
    finally:
        ws_chat.broadcast_to_user = orig


# ── A4-4: the correction has to be deliverable ──────────────────────────

@pytest.mark.asyncio
async def test_the_route_broadcasts_the_revision_it_was_given():
    user_id, conv_id = await _seed()
    frames: list = []
    key = str(uuid.uuid4())

    await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="بر", client_msg_id=key, revision=0,
    ), frames)
    await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="برام یه هتل پیدا کن", client_msg_id=key, revision=1,
    ), frames)

    assert len(frames) == 2, frames
    assert frames[0]["revision"] == 0
    assert frames[1]["revision"] == 1
    # Same id both times, which is what makes the marker necessary: the client
    # de-dupes by id and has no update path without it.
    assert frames[0]["id"] == frames[1]["id"]
    assert frames[1]["content"] == "برام یه هتل پیدا کن"


@pytest.mark.asyncio
async def test_a_write_with_no_revision_is_byte_identical_to_today():
    """Absent must stay indistinguishable from before: a history refetch
    returns rows with no revision and the clients correctly read that as
    'not a correction'."""
    user_id, conv_id = await _seed()
    frames: list = []
    await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="hello", client_msg_id=str(uuid.uuid4()),
    ), frames)
    assert "revision" not in frames[0], frames[0]


@pytest.mark.asyncio
async def test_a_revision_does_not_erase_media_or_tool_events():
    """A revision carries provenance and a longer sentence. It must not NULL
    the media card and the run rail the first write established — and the
    frame it broadcasts must still describe the row that exists, or an open
    thread loses the card until the next refetch."""
    user_id, conv_id = await _seed()
    frames: list = []
    key = str(uuid.uuid4())

    await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Playing it now.", client_msg_id=key,
        revision=0,
        media={"type": "youtube", "video_id": "v1", "title": "HUMBLE."},
        tool_events=[{"tool": "play_media", "started_at_ms": 1}],
    ), frames)

    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Playing HUMBLE. now.", client_msg_id=key,
        revision=1, voice={"source": "delegated", "delegation_id": "d1"},
    ), frames)

    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        meta = json.loads(row.metadata_json or "{}")
    assert meta["media"]["video_id"] == "v1", meta
    assert meta["tool_events"], meta
    assert meta["voice"] == {"source": "delegated", "delegation_id": "d1"}

    assert frames[1]["media"]["video_id"] == "v1", frames[1]


@pytest.mark.asyncio
async def test_voice_provenance_is_allowlisted_and_bounded():
    """The dict is request-body input that ends up inside a row every client
    renders. Unknown keys and prose values are dropped, not truncated."""
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Done.", client_msg_id=str(uuid.uuid4()),
        voice={
            "source": "live_spoken", "played_ms": 1200, "interrupted": True,
            "note": "x" * 5000,          # unknown key
            "model": "y" * 500,           # known key, absurd value
        },
    ), frames)

    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        voice = json.loads(row.metadata_json or "{}")["voice"]
    assert voice == {"source": "live_spoken", "played_ms": 1200,
                     "interrupted": True}, voice


def test_a_dropped_provenance_key_is_logged_by_NAME(caplog):
    """Review L3-R9. The allowlist is the right default for request-body
    input, but the producer here is our OWN relay: the failure that matters is
    "L1 added a key and it vanished", and a silent drop makes that
    indistinguishable from a relay that never sent it.

    Names and counts only — a provenance VALUE never reaches the log.

    The unknown key is deliberately SYNTHETIC. This test used `turn_id` as its
    "plausible future L1 key", and the future arrived: the relay's user rows
    carry `turn_id` and the allowlist admits it by design (causal identity), so
    the test went red on a correct allowlist. A name no producer will ever
    mint cannot go stale that way, and the premise is asserted first so any
    collision fails at the premise, not as a confusing mismatch below.
    """
    import logging

    from app.api.sessions import _VOICE_KEYS, _clean_voice

    unknown = "zz_synthetic_unknown_voice_key"
    assert unknown not in _VOICE_KEYS, (
        "the synthetic 'unknown' key is now allowlisted — pick another one"
    )

    with caplog.at_level(logging.WARNING, logger="app.api.sessions"):
        out = _clean_voice({
            "source": "delegated",
            unknown: "t-9",                # a key the allowlist does not know
            "model": "m" * 500,            # known key, absurd value
        })
    assert out == {"source": "delegated"}
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert unknown in blob, blob
    assert "model" in blob, blob
    # The VALUES are not in it.
    assert "t-9" not in blob and "mmmm" not in blob, blob


def test_the_relay_s_user_turn_provenance_survives_the_allowlist(caplog):
    """The other half of the change above. `turn_id` was this file's example
    of an UNKNOWN key; it is now the user row's causal identity, stamped by the
    relay's `user_voice` together with the turn's provider-clock span. Making
    it unknown again to satisfy an old example would silently strip every
    user row of the id its task and reply are parented to."""
    import logging

    from app.api.sessions import _clean_voice

    user_voice = {"source": "live_user", "turn_id": "live-utt:ps-1:7",
                  "start_ms": 1200, "end_ms": 2400, "clock": "provider"}
    with caplog.at_level(logging.WARNING, logger="app.api.sessions"):
        out = _clean_voice(dict(user_voice))
    assert out == user_voice, out
    assert not [r for r in caplog.records if "voice provenance" in r.getMessage()]


def test_bounded_request_turn_ids_survive_without_accepting_an_unbounded_list():
    from app.api.sessions import _VOICE_REQUEST_TURN_IDS_MAX, _clean_voice
    from app.services.live_voice_protocol import REQUEST_TURN_IDS_MAX

    assert _VOICE_REQUEST_TURN_IDS_MAX == REQUEST_TURN_IDS_MAX == 16
    ids = [f"live-utt:session:{i}" for i in range(3)]
    voice = {"source": "delegated", "parent_user_turn_id": ids[-1],
             "request_turn_ids": ids}
    assert _clean_voice(voice) == voice
    for invalid in ([], ids + ids[:1], [""], ["x" * 121],
                    [f"turn-{i}" for i in range(17)], [1]):
        assert "request_turn_ids" not in (_clean_voice({"request_turn_ids": invalid}) or {})


def test_a_clean_provenance_dict_logs_nothing(caplog):
    """One line per write, bounded by the allowlist — and none at all on the
    ordinary path, which is every write the relay makes today."""
    import logging

    from app.api.sessions import _clean_voice

    with caplog.at_level(logging.WARNING, logger="app.api.sessions"):
        out = _clean_voice({"source": "delegated", "delegation_id": "dlg-1",
                            "played_ms": 900, "interrupted": False})
    assert out["delegation_id"] == "dlg-1"
    assert not [r for r in caplog.records if "voice provenance" in r.getMessage()]


def test_the_delegation_id_is_stored_at_the_SAME_bound_the_wire_uses():
    """`delegation_id` is the one provenance value that is COMPARED, by
    `voice_jobs.VoiceTurnJob._row_is_proof`, against a copy of itself that
    reached the agent bounded to 64 (relay body `[:64]`, then
    `ChatRequest(max_length=64)`). The id is provider-supplied and unbounded.

    Stored at a different length the two can never match and a delivered
    answer closes its card `cancelled` — silent, and in the direction that
    looks correct. So this key is truncated to the same 64 rather than kept
    long (the `_VOICE_STR_MAX` path) or dropped as oversize.
    """
    from app.api.sessions import _VOICE_ID_MAX, _clean_voice

    long_id = "dlg-" + "a" * 200
    out = _clean_voice({"source": "delegated", "delegation_id": long_id})
    assert out["delegation_id"] == long_id[:_VOICE_ID_MAX]
    assert _VOICE_ID_MAX == 64
    # Ordinary ids are untouched — the bound is a ceiling, not a rewrite.
    assert _clean_voice({"delegation_id": "dlg-1"})["delegation_id"] == "dlg-1"


# ── A4-8: one key, one row ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_two_writes_of_the_same_key_produce_one_row():
    user_id, conv_id = await _seed()
    frames: list = []
    key = str(uuid.uuid4())
    for i, text in enumerate(["I", "I need", "I need a hotel"]):
        await _post(conv_id, user_id, SessionMessageCreate(
            role="user", content=text, client_msg_id=key, revision=i,
        ), frames)

    from sqlalchemy import select
    async with async_session_maker() as db:
        rows = (await db.execute(select(Message).where(
            Message.conversation_id == conv_id))).scalars().all()
        conv = await db.get(Conversation, conv_id)
    assert len(rows) == 1, [r.content for r in rows]
    assert rows[0].content == "I need a hotel"
    assert conv.message_count == 1, (
        "a rewrite is not a new message — counting it inflates the thread"
    )


@pytest.mark.asyncio
async def test_the_keyed_lookup_takes_a_row_lock_on_postgres():
    """The belt-and-braces half of A4-8, asserted at the SOURCE because the
    race it closes cannot be reproduced against SQLite.

    `client_msg_id` is deliberately non-unique (a chat turn stamps one value on
    both of its rows), so there is no constraint to catch a double insert — two
    overlapping POSTs of one key would both SELECT nothing and both INSERT. The
    lock is taken on the CONVERSATION row, which the write updates anyway, and
    ONLY for a keyed write, so an ordinary insert is unchanged.
    """
    import inspect
    from app.api import sessions as sessions_mod

    src = inspect.getsource(sessions_mod.create_session_message)
    i = src.find("with_for_update()")
    assert i != -1, "the keyed upsert must lock the row it reads"
    # ORDER is the property: the lock has to be taken before the lookup it
    # protects, or it is decoration.
    j = src.find("Message.client_msg_id == client_msg_id")
    assert j != -1 and i < j, (
        "the lock is taken AFTER the lookup it exists to protect — the "
        "window it closes is still open"
    )
    # …and only for a keyed write.
    head = src[max(0, i - 200):i]
    assert "client_msg_id" in head, head
