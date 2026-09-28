"""`message.voice` stops being write-only (Blocker B, contract v0.2).

The relay minted the object, `_message_payload` put it on the wire,
`SessionMessageCreate.voice` accepted it, `sessions._clean_voice` allowlisted it
and `sessions._build_metadata` persisted it into `messages.metadata_json` — and
then NOTHING read it back. Not one REST reader, not the live WS frame. So the
field the wire contract describes was unobservable from the app at its very
first read: the phone could not tell a spoken paraphrase from the complete task
result, could not see that a reply had been cut off, and could not see how much
of it the caller had actually heard.

`grep -rn '"voice"' backend/app/api` matched only the WRITE path before this
round, which is exactly why nothing was red. This file is the pin for both
halves:

  * the allowlist still ADMITS the v0.2 record keys (`version`, `record_kind`,
    `heard_chars`) and still DROPS — loudly — anything it does not know, because
    `voice` is request-body input written into a row every client renders;
  * every read surface returns the object: the three REST readers
    (`day_chats` both arms, `sessions`, `messages_recover`) and the live
    `message` frame, which derives it from the COMMITTED row.

…and the legacy leg of each, because a row written before any of this existed
must serialize exactly as it does today.

Lane: RUN_MODE=agent. `day_chats`/`conversations`/`messages` are AGENT_ONLY and
all four real readers SELECT them, so the route-driven tests cannot run in the
platform sweep (measured on 16ee4958: 24/24 under agent, 9 failed under
platform with `no such table: day_chats` / `conversations`). The allowlist,
schema, merge and frame tests are pure and would run anywhere; they live here
because splitting one contract across two files is how a half of it stops being
maintained. This file therefore needs its `# agent-mode` line in
tests/COVERAGE_DEBT.txt — it shipped without it once and failed the platform
sweep (CI run 35792648976); test_ci_coverage_ratchet.py now checks the "Lane:"
line above against that list.

Local run (from backend/):
    RUN_MODE=agent PYTHONPATH=. .venv312/bin/python -m pytest \
      tests/test_voice_record_metadata.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


# A full v0.2 provenance object for the row that carries the COMPLETE backend
# answer. D2: the playback numbers are deliberately absent — they describe the
# spoken paraphrase, which is a different and shorter string, and a row that
# carried both would be claiming playback facts about text it is not.
TASK_RESULT_VOICE = {
    "version": 1,
    "record_kind": "task_result",
    "source": "delegated",
    "delegation_id": "dlg-7",
    "task_id": "dlg-7",
    "assistant_turn_id": "live:ps-1:3",
    "parent_user_turn_id": "live-utt:ps-1:7",
    "model": "gpt-live",
    "spoken": True,
}

# …and for the sibling row that records what the caller actually heard.
TRANSCRIPT_VOICE = {
    "version": 1,
    "record_kind": "assistant_transcript",
    "source": "live_spoken",
    "assistant_turn_id": "live:ps-1:3",
    "parent_user_turn_id": "live-utt:ps-1:7",
    "epoch": 3,
    "played_ms": 1840,
    "interrupted": True,
    "heard_chars": 15,
    "generated_chars": 64,
    "clock": "provider",
}


# ── The allowlist ────────────────────────────────────────────────────────

def test_the_v0_2_record_keys_survive_the_allowlist(caplog):
    """`version` / `record_kind` / `heard_chars` are the three keys the split
    is made of. Dropped here they would be minted by the relay, believed by the
    relay, and stored nowhere — the silent class this allowlist logs for."""
    from app.api.sessions import _VOICE_KEYS, _clean_voice

    assert {"version", "record_kind", "heard_chars"} <= _VOICE_KEYS

    with caplog.at_level(logging.WARNING, logger="app.api.sessions"):
        out = _clean_voice(dict(TRANSCRIPT_VOICE))
    assert out == TRANSCRIPT_VOICE, out
    # The ordinary path logs nothing — one line per write, and only for a drop.
    assert not [r for r in caplog.records if "voice provenance" in r.getMessage()]

    assert _clean_voice(dict(TASK_RESULT_VOICE)) == TASK_RESULT_VOICE


def test_nothing_was_heard_is_a_zero_not_an_absence():
    """An interrupted epoch the caller heard NOTHING of records
    `heard_chars: 0`, and `0` is falsy in every language this pipeline is
    written in. `_clean_voice` type-tests rather than truth-tests for exactly
    this reason; a drop here would turn "heard nothing" into "cannot say",
    which is the one distinction the field exists to make."""
    from app.api.sessions import _build_metadata, _clean_voice

    out = _clean_voice({
        "version": 1, "record_kind": "assistant_transcript",
        "heard_chars": 0, "interrupted": True, "played_ms": 0,
    })
    assert out["heard_chars"] == 0
    assert out["played_ms"] == 0
    assert out["interrupted"] is True
    # …and it survives the persist step, whose `if voice:` guard is a
    # non-empty-DICT test, not a test of the values inside it.
    assert json.loads(_build_metadata(None, None, None, out))["voice"] == out


def test_an_unknown_key_is_still_dropped_AND_logged(caplog):
    """The allowlist must not become a passthrough because the relay grew a
    field. `voice` is request-body input that lands inside a row every client
    renders, so an open-ended blob there is an unbounded write — and the drop
    is LOGGED by name, because "L1 added a key and it vanished" is otherwise
    indistinguishable from a relay that never sent it.

    Names and counts only. The VALUE never reaches the log: `heard_text` is the
    words the caller heard, and this round's rule is that nothing logs content.
    """
    from app.api.sessions import _VOICE_KEYS, _clean_voice

    # Deliberate: the one v0.2 wire field that must NEVER be persisted under a
    # provenance key. The heard prefix is the row's own `content`; what the
    # provenance object carries is its LENGTH.
    assert "heard_text" not in _VOICE_KEYS

    with caplog.at_level(logging.WARNING, logger="app.api.sessions"):
        out = _clean_voice({
            "version": 1,
            "record_kind": "assistant_transcript",
            "heard_text": "First sentence.",
            "not_a_provenance_key": 1,
        })
    assert out == {"version": 1, "record_kind": "assistant_transcript"}, out

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "heard_text" in blob, blob
    assert "not_a_provenance_key" in blob, blob
    assert "First sentence" not in blob, blob


def test_prose_under_a_known_record_key_is_dropped_not_truncated():
    """A provenance value is an id, a count or a flag. `record_kind` is a short
    enum; a caller that sends a paragraph under it is not sending provenance,
    and truncating would store a prefix of a paragraph as if it were one."""
    from app.api.sessions import _VOICE_STR_MAX, _clean_voice

    out = _clean_voice({
        "source": "delegated",
        "record_kind": "x" * (_VOICE_STR_MAX + 1),
    })
    assert out == {"source": "delegated"}, out
    # The bound is a ceiling, not a rewrite.
    assert _clean_voice({"record_kind": "x" * _VOICE_STR_MAX})["record_kind"]


# ── The schema (declare-or-dropped) ──────────────────────────────────────

def test_the_response_model_would_have_dropped_the_field_undeclared():
    """`ChatMessageResponse` ignores undeclared keys (pydantic 2.x `extra`
    defaults to `ignore`). That is why adding `voice` to four serializers and
    not to this model would have changed NOTHING observable — the dict would be
    built and then silently discarded on the way out. Both halves are asserted
    so the mechanism is visible next to the fix."""
    from app.schemas import ChatMessageResponse

    assert "voice" in ChatMessageResponse.model_fields

    row = ChatMessageResponse(
        id="m1", role="assistant", content="hi",
        created_at=datetime(2026, 9, 22, 9, 0, 0),
        voice=dict(TRANSCRIPT_VOICE),
        # An undeclared neighbour: what `voice` itself was until this round.
        some_undeclared_key={"still": "dropped"},
    )
    assert row.voice == TRANSCRIPT_VOICE
    dumped = row.model_dump()
    assert dumped["voice"]["record_kind"] == "assistant_transcript"
    assert dumped["voice"]["heard_chars"] == 15
    assert "some_undeclared_key" not in dumped

    # Legacy row: the key is PRESENT and null, like every other optional dict
    # on this model — a missing key is a third state the clients have no
    # branch for.
    legacy = ChatMessageResponse(
        id="m2", role="assistant", content="hi",
        created_at=datetime(2026, 9, 22, 9, 0, 0),
    )
    assert legacy.voice is None
    assert "voice" in legacy.model_dump()


# ── The rewrite merge ────────────────────────────────────────────────────

def test_a_voice_only_rewrite_keeps_its_siblings_and_REPLACES_voice():
    """`_merge_metadata` is top-level, and for `voice` that is the decision,
    not an accident.

    D2 requires the delegated row's playback numbers to be GONE once it is
    re-stamped `task_result`: they describe the paraphrase, and the complete
    answer is a different string. An additive merge cannot express a removal,
    so it would resurrect exactly the keys the record kind promises are absent.
    The producer therefore writes the FULL object every time (contract v0.2),
    and this pins both halves of that bargain.
    """
    from app.api.sessions import _build_metadata, _merge_metadata

    first = _build_metadata(
        {"type": "youtube", "video_id": "v1"},
        [{"tool": "web_search", "started_at_ms": 1}],
        None,
        dict(TRANSCRIPT_VOICE),
    )
    rewrite = _build_metadata(None, None, None, dict(TASK_RESULT_VOICE))
    merged = json.loads(_merge_metadata(first, rewrite))

    # Siblings a revision is silent about survive — the R48 rule this helper
    # was written for.
    assert merged["media"]["video_id"] == "v1"
    assert merged["tool_events"][0]["tool"] == "web_search"
    # …and `voice` does NOT accumulate: the playback numbers of the paraphrase
    # are not on the row that carries the complete answer.
    assert merged["voice"] == TASK_RESULT_VOICE
    for stale in ("played_ms", "interrupted", "epoch", "heard_chars"):
        assert stale not in merged["voice"], stale


# ── The live frame ───────────────────────────────────────────────────────

def _frame_row(**kw):
    base = dict(
        id="m1", role="assistant", content="hello",
        created_at=datetime(2026, 9, 22, 9, 0, 0),
        channel="voice", day_chat_id="day-1",
        client_msg_id=None, occurred_at=None, metadata_json=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_live_frame_carries_the_committed_provenance():
    """The frame must agree with the readers, or an open thread and the same
    thread after a refetch disagree about which row is the complete answer."""
    from app.api.message_frames import message_frame

    f = message_frame(_frame_row(
        metadata_json=json.dumps({"voice": TRANSCRIPT_VOICE}),
    ))
    assert f["voice"] == TRANSCRIPT_VOICE
    # A copy, not the row's own dict — the frame is handed to a broadcast task.
    assert f["voice"] is not TRANSCRIPT_VOICE


def test_the_frame_reads_the_row_not_the_request_body():
    """`media`/`tool_events` are passed in because only the caller holds their
    wire form. `voice` has none: it is the allowlisted object already written
    and already reconciled by `_merge_metadata`, so deriving it here is what
    makes the frame describe the COMMITTED row on a rewrite — and what gives
    the second writer (`voice_tasks._persist_message`) the field for free
    without a new argument at its call site."""
    import inspect

    from app.api.message_frames import message_frame

    assert "voice" not in inspect.signature(message_frame).parameters

    # Half-written / hand-edited rows degrade to "no provenance" rather than
    # raising on a path that is fire-and-forget live delivery.
    for bad in (None, "", "{not json", json.dumps([1, 2]),
                json.dumps({"voice": "a string"}), json.dumps({"voice": {}})):
        assert "voice" not in message_frame(_frame_row(metadata_json=bad)), bad


def test_a_legacy_row_produces_a_frame_byte_identical_to_todays():
    """An old app receiving the new field must be unaffected — and a row with
    no provenance must not grow a key at all, because absence is the state
    every shipped client already handles."""
    from app.api.message_frames import message_frame

    assert "voice" not in message_frame(_frame_row())
    assert "voice" not in message_frame(_frame_row(
        metadata_json=json.dumps({"media": {"type": "youtube"}}),
    ))


# ── The read surfaces, driven for real ───────────────────────────────────
# (asyncio_mode = auto in pytest.ini — the async tests below need no mark, and
# a module-level `pytestmark` would wrongly claim the pure tests above.)


async def _seed_day(specs, *, with_day_chat: bool):
    """A one-day voice thread holding exactly `specs`, in the order given.

    `specs` is a list of `(suffix, role, content, voice_or_None)`; each row
    lands at `f"{uid}-{suffix}"` one second after the last, so thread order is
    the list order on every reader.

    `with_day_chat` picks which arm of `day_chats.get_day_chat_messages` runs:
    a real DayChat row takes the fast path, its absence takes the date-range
    scan. Both build their own row dict, so a key added to one and not the
    other — or a projection applied to one and not the other — is invisible on
    whichever path a given tenant happens to take.
    """
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    uid = str(uuid.uuid4())
    conv = f"{uid}-conv"
    dc = f"{uid}-dc"
    today = datetime.now(timezone.utc).date()
    now = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)

    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"bb-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="BB", timezone="UTC"))
        if with_day_chat:
            db.add(DayChat(id=dc, user_id=uid, local_date=today, timezone="UTC"))
            await db.flush()
        db.add(Conversation(id=conv, user_id=uid, channel="voice",
                            day_chat_id=dc if with_day_chat else None,
                            started_at=now))
        await db.flush()
        for i, (suffix, role, content, voice) in enumerate(specs):
            db.add(Message(
                id=f"{uid}-{suffix}", conversation_id=conv,
                day_chat_id=dc if with_day_chat else None,
                channel="voice", role=role, content=content,
                created_at=now + timedelta(seconds=i),
                metadata_json=(
                    json.dumps({"voice": voice}) if voice is not None else None
                ),
            ))
        await db.commit()
    return uid, conv, today


async def _seed(*, with_day_chat: bool):
    """A day holding one voice row WITH provenance and one legacy row without."""
    return await _seed_day(
        [
            ("legacy", "user", "How are admissions?", None),
            ("voice", "assistant", "First sentence.", TRANSCRIPT_VOICE),
        ],
        with_day_chat=with_day_chat,
    )


async def _user(uid: str):
    from app.db.database import async_session_maker
    from app.db.models.user import User

    async with async_session_maker() as db:
        return await db.get(User, uid)


def _rows(out):
    return out if isinstance(out, list) else json.loads(bytes(out.body).decode())


def _split(rows, uid):
    voiced = next(r for r in rows if r["id"] == f"{uid}-voice")
    legacy = next(r for r in rows if r["id"] == f"{uid}-legacy")
    return voiced, legacy


def _assert_both_legs(voiced, legacy, where: str):
    """One assertion pair, four readers. The provenance comes back intact, and
    a row that never had any is PRESENT-and-null — the same shape `media` and
    `tool_events` use, never a missing key."""
    assert voiced.get("voice") == TRANSCRIPT_VOICE, f"{where}: {voiced}"
    assert voiced["voice"]["record_kind"] == "assistant_transcript", where
    assert voiced["voice"]["heard_chars"] == 15, where
    assert "voice" in legacy, f"{where}: key must be present, not missing"
    assert legacy["voice"] is None, f"{where}: {legacy['voice']!r}"


@pytest.mark.parametrize("with_day_chat", [True, False])
async def test_day_chats_returns_the_provenance_on_both_arms(
    monkeypatch, with_day_chat,
):
    """THE primary history fetch — every client asks it first. It builds two
    separate row dicts (the DayChat fast path and the date-range scan), so both
    are driven."""
    import app.api.day_chats as api
    from app.db.database import async_session_maker

    async def _no_proxy(*a, **k):
        return None

    monkeypatch.setattr(api, "_get_agent_proxy_info", _no_proxy)
    uid, conv, day = await _seed(with_day_chat=with_day_chat)
    user = await _user(uid)
    async with async_session_maker() as db:
        rows = _rows(await api.get_day_chat_messages(
            date_str=day.isoformat(), limit=500, current_user=user, db=db,
        ))
    voiced, legacy = _split(rows, uid)
    _assert_both_legs(voiced, legacy, f"day_chats(day_chat={with_day_chat})")


async def test_the_session_serializer_returns_the_provenance():
    """`/api/sessions/...` is the FALLBACK the mobile client takes whenever
    day-chats fails. A field emitted by only one serializer disappears exactly
    when the app is already degraded — the bug class every neighbour key in
    this serializer was added for."""
    from sqlalchemy import select

    from app.api.sessions import _message_to_response
    from app.db.database import async_session_maker
    from app.db.models import Message

    uid, conv, day = await _seed(with_day_chat=True)
    async with async_session_maker() as db:
        msgs = (await db.execute(
            select(Message).where(Message.conversation_id == conv)
            .order_by(Message.created_at.asc())
        )).scalars().all()
        out = [_message_to_response(m, None, None, {conv: "voice"}) for m in msgs]

    legacy, voiced = out
    assert voiced.voice == TRANSCRIPT_VOICE, voiced.voice
    assert legacy.voice is None
    # Through the response MODEL, not just the dict it was built from: an
    # undeclared field would be dropped right here.
    assert voiced.model_dump()["voice"]["record_kind"] == "assistant_transcript"
    assert "voice" in legacy.model_dump()


async def test_the_by_date_fallback_route_returns_the_provenance():
    """The fourth reader — `/api/sessions/by-date/{date}/messages`, the route
    the client actually calls when day-chats is down."""
    from app.api.sessions import get_messages_by_date
    from app.db.database import async_session_maker

    uid, conv, day = await _seed(with_day_chat=True)
    user = await _user(uid)
    async with async_session_maker() as db:
        rows = _rows(await get_messages_by_date(
            day.isoformat(), limit=200, tz_offset=0, current_user=user, db=db,
        ))
    voiced, legacy = _split(rows, uid)
    _assert_both_legs(voiced, legacy, "sessions.by_date")


async def test_messages_since_returns_the_provenance(monkeypatch):
    """The WS-reconnect backstop. It returns precisely the rows the socket
    never delivered — and a voice turn cut short mid-call is the row that most
    needs to say it was cut short."""
    import app.api.messages_recover as api
    from app.db.database import async_session_maker

    async def _no_proxy(*a, **k):
        return None

    monkeypatch.setattr(api, "_get_agent_proxy_info", _no_proxy)
    uid, conv, day = await _seed(with_day_chat=True)
    user = await _user(uid)
    async with async_session_maker() as db:
        rows = _rows(await api.messages_since(
            f"{uid}-legacy", limit=100, current_user=user, db=db,
        ))
    # The seed row itself is excluded; the voice row is the tail.
    voiced = next(r for r in rows if r["id"] == f"{uid}-voice")
    assert voiced["voice"] == TRANSCRIPT_VOICE, voiced


async def test_a_row_with_malformed_provenance_degrades_instead_of_500ing():
    """A hand-edited or half-written row must not take a whole history load
    down with it — the same defensive posture `day_chats._metadata` takes."""
    from app.api.day_chats import _serialize_meta_card

    for raw in (None, "", "{not json", json.dumps([1]),
                json.dumps({"voice": "a string"}), json.dumps({"voice": {}}),
                json.dumps({"voice": None})):
        msg = SimpleNamespace(id="m1", metadata_json=raw)
        assert _serialize_meta_card(msg, "voice") is None, raw


# ── The nothing-heard rule, READ layer ───────────────────────────────────
#
# A validated receipt of the empty string means the caller heard NOTHING of
# that output epoch. The relay cannot write that honestly: the persistence API
# refuses an empty body (`ws_realtime._save_voice_messages`, and this very
# route 400s on empty content), so the row keeps the GENERATED text and says
# `heard_chars: 0` beside it. Left alone, the live surface shows nothing while
# the saved chat shows the whole answer — the exact live/saved divergence this
# repair exists to remove.
#
# SAVE never presents unheard words as heard; READ — below — projects such a
# row's content to the empty string on EVERY surface; RENDER draws nothing.
# These tests are the READ half, driven through the real serializers rather
# than through the helper alone, because the defect was never in the predicate
# — it was in a surface that forgot to ask it.

#: What the relay generated for one epoch, and what the caller never heard.
SILENT_EPOCH_TEXT = (
    "Admissions close on the fourteenth of March, and the deposit is due "
    "two weeks after that."
)
#: The complete backend answer. A `task_result` row is NEVER projected and
#: never clipped: it is not a playback claim, and the whole point of splitting
#: the two rows is that the answer survives a call that played none of it.
TASK_RESULT_TEXT = (
    "Admissions close on 14 March. The deposit is due on 28 March, and late "
    "deposits are accepted until 4 April with a fee."
)

#: A receipt for zero characters — the caller heard none of this epoch.
NOTHING_HEARD_VOICE = {
    "version": 1,
    "record_kind": "assistant_transcript",
    "source": "live_spoken",
    "assistant_turn_id": "live:ps-1:4",
    "epoch": 4,
    "played_ms": 0,
    "interrupted": True,
    "heard_chars": 0,
    "generated_chars": len(SILENT_EPOCH_TEXT),
    "clock": "provider",
}
#: The relay's other way of saying it: a row already written with text, later
#: contradicted by a validated empty receipt. It cannot be un-written and it
#: cannot be rewritten to empty content, so it is re-stamped with the flag —
#: and the flag alone retracts the row, `heard_chars` notwithstanding.
RETRACTED_VOICE = {
    "version": 1,
    "record_kind": "assistant_transcript",
    "source": "live_spoken",
    "assistant_turn_id": "live:ps-1:5",
    "epoch": 5,
    "played_ms": 640,
    "interrupted": True,
    "heard_chars": 12,
    "transcript_retracted": True,
    "generated_chars": len(SILENT_EPOCH_TEXT),
    "clock": "provider",
}

#: One day, five rows, one expectation per row — the table every surface below
#: is measured against. `id suffix -> (stored content, served content)`.
_HEARD_SPECS = [
    # A legacy row: no voice provenance at all. Every pre-relay row and every
    # typed turn. Byte-identical to today, or the projection has started
    # blanking ordinary chat history.
    ("legacy", "user", "How are admissions?", None),
    # A transcript row with a real receipt: the heard PREFIX is what the
    # caller heard, and it is served unchanged. Truncation is not the rule —
    # "serve only what was heard" is, and here that is a non-empty string.
    ("heard", "assistant", "First sentence.", TRANSCRIPT_VOICE),
    # Heard nothing: stored full, served empty.
    ("silent", "assistant", SILENT_EPOCH_TEXT, NOTHING_HEARD_VOICE),
    # Retracted: stored full, served empty, even though its receipt counted 12.
    ("retracted", "assistant", SILENT_EPOCH_TEXT, RETRACTED_VOICE),
    # The complete backend answer, in the same conversation as the two rows
    # above. Full text, always.
    ("result", "assistant", TASK_RESULT_TEXT, TASK_RESULT_VOICE),
]

_HEARD_EXPECT = {
    "legacy": "How are admissions?",
    "heard": "First sentence.",
    "silent": "",
    "retracted": "",
    "result": TASK_RESULT_TEXT,
}


async def _every_read_surface(with_day_chat: bool, monkeypatch):
    """`{surface name: {id suffix: row dict}}` for one seeded day.

    Every client-facing message serializer in the backend, driven for real:
    both arms of the primary day-chats fetch, the session serializer the
    clients fall back to, the by-date route that wraps it, the WS-reconnect
    backstop, and the live `{"type":"message"}` frame builder. A rule that
    holds on four of the five is a rule that fails whenever a client takes the
    fifth path — which, for the fallbacks, is exactly when it is already
    degraded.
    """
    from sqlalchemy import select

    import app.api.day_chats as day_chats_api
    import app.api.messages_recover as recover_api
    from app.api.message_frames import message_frame
    from app.api.sessions import _message_to_response, get_messages_by_date
    from app.db.database import async_session_maker
    from app.db.models import Message

    async def _no_proxy(*a, **k):
        return None

    monkeypatch.setattr(day_chats_api, "_get_agent_proxy_info", _no_proxy)
    monkeypatch.setattr(recover_api, "_get_agent_proxy_info", _no_proxy)

    uid, conv, day = await _seed_day(_HEARD_SPECS, with_day_chat=with_day_chat)
    user = await _user(uid)
    out: dict = {}

    def _by_suffix(rows):
        return {r["id"].rsplit("-", 1)[-1]: r for r in rows}

    async with async_session_maker() as db:
        out["day_chats"] = _by_suffix(_rows(
            await day_chats_api.get_day_chat_messages(
                date_str=day.isoformat(), limit=500, current_user=user, db=db,
            )
        ))
    async with async_session_maker() as db:
        out["sessions.by_date"] = _by_suffix(_rows(
            await get_messages_by_date(
                day.isoformat(), limit=200, tz_offset=0,
                current_user=user, db=db,
            )
        ))
    async with async_session_maker() as db:
        out["messages_since"] = _by_suffix(_rows(
            await recover_api.messages_since(
                f"{uid}-legacy", limit=100, current_user=user, db=db,
            )
        ))
    async with async_session_maker() as db:
        msgs = (await db.execute(
            select(Message).where(Message.conversation_id == conv)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )).scalars().all()
        out["sessions.serializer"] = _by_suffix([
            _message_to_response(m, None, None, {conv: "voice"}).model_dump()
            for m in msgs
        ])
        out["message_frame"] = _by_suffix([
            message_frame(m, channel="voice") for m in msgs
        ])
    return uid, out


@pytest.mark.parametrize("with_day_chat", [True, False])
async def test_no_surface_serves_words_the_caller_never_heard(
    monkeypatch, with_day_chat,
):
    """The whole rule, on every surface at once.

    Stored content is unchanged on disk — the relay could not write it empty
    and this layer does not try to. What changes is what leaves the process:
    an `assistant_transcript` row the caller heard nothing of serves the empty
    string, and nothing else moves.
    """
    uid, surfaces = await _every_read_surface(with_day_chat, monkeypatch)
    for name, rows in surfaces.items():
        for suffix, expected in _HEARD_EXPECT.items():
            if suffix == "legacy" and name == "messages_since":
                continue  # the seed row itself; this route excludes it
            assert suffix in rows, f"{name}: row {suffix!r} missing"
            assert rows[suffix]["content"] == expected, (
                f"{name}/{suffix}: served {rows[suffix]['content']!r}, "
                f"expected {expected!r}"
            )


@pytest.mark.parametrize("with_day_chat", [True, False])
async def test_a_projected_row_keeps_every_bit_of_its_provenance(
    monkeypatch, with_day_chat,
):
    """Blanking the body must not blank the explanation.

    `heard_chars`, `interrupted` and `record_kind` are how a client renders
    the empty row HONESTLY — "you interrupted before anything played" is a
    real event in the thread. Strip them with the text and the row becomes an
    unexplained blank bubble, which is a different lie from the one we fixed.
    """
    uid, surfaces = await _every_read_surface(with_day_chat, monkeypatch)
    for name, rows in surfaces.items():
        for suffix, expected in (
            ("silent", NOTHING_HEARD_VOICE),
            ("retracted", RETRACTED_VOICE),
            ("result", TASK_RESULT_VOICE),
        ):
            assert rows[suffix].get("voice") == expected, f"{name}/{suffix}"
        # …and the legacy row still has the key present-and-null on the REST
        # readers (the live frame omits it entirely — its only-when-present
        # rule, unchanged).
        if name != "message_frame" and "legacy" in rows:
            assert "voice" in rows["legacy"], name
            assert rows["legacy"]["voice"] is None, name
        if name == "message_frame":
            assert "voice" not in rows["legacy"], name


async def test_the_task_result_row_is_never_projected_or_clipped():
    """The backend answer is not a playback claim.

    A delegated turn's `task_result` row carries the complete answer and no
    playback numbers, precisely because the numbers describe the spoken
    paraphrase. Even handed the exact provenance that silences a transcript
    row — a zero receipt AND the retraction flag — it is served whole: if the
    caller heard none of the paraphrase, the written answer is the only thing
    left of the turn, and clipping it would delete the work as well as the
    words.
    """
    from app.schemas import (
        VOICE_KIND_TASK_RESULT, VOICE_KIND_TRANSCRIPT,
        heard_nothing, public_heard_text,
    )

    # The two kinds this rule turns on, pinned to the contract's own names
    # rather than to two string literals that can drift apart.
    assert TASK_RESULT_VOICE["record_kind"] == VOICE_KIND_TASK_RESULT
    assert TRANSCRIPT_VOICE["record_kind"] == VOICE_KIND_TRANSCRIPT
    assert NOTHING_HEARD_VOICE["record_kind"] == VOICE_KIND_TRANSCRIPT
    assert RETRACTED_VOICE["record_kind"] == VOICE_KIND_TRANSCRIPT

    for extra in ({"heard_chars": 0}, {"transcript_retracted": True},
                  {"heard_chars": 0, "transcript_retracted": True}):
        voice = {**TASK_RESULT_VOICE, **extra}
        assert heard_nothing(voice) is False, extra
        assert public_heard_text(TASK_RESULT_TEXT, voice) == TASK_RESULT_TEXT


def test_absence_of_a_receipt_is_not_a_receipt_for_zero():
    """The predicate, at its edges. Every False here is a row that would
    otherwise be blanked in production for no reason.

    `heard_chars` is OMITTED, never sent as None, when no receipt arrived
    (`live_voice_protocol.voice_record_fields`) — exactly so "this client
    never told us" cannot be read as "this client heard nothing". And a row
    with `voice` but no `record_kind` is the tenant-rollout window before
    `_VOICE_KEYS` shipped these keys: legacy semantics, untouched.
    """
    from app.schemas import heard_nothing

    T = "assistant_transcript"
    silences = [
        {"record_kind": T, "heard_chars": 0},
        {"record_kind": T, "heard_chars": 0.0},
        {"record_kind": T, "heard_chars": 7, "transcript_retracted": True},
        {"record_kind": T, "transcript_retracted": True},
    ]
    for voice in silences:
        assert heard_nothing(voice) is True, voice

    speaks = [
        None, {}, "not a dict", ["not a dict"],
        # No provenance shape at all, or the pre-repair one.
        {"source": "live_spoken", "heard_chars": 0},
        {"version": 1, "source": "live_spoken", "played_ms": 0},
        # A receipt that never arrived is not a receipt for zero.
        {"record_kind": T},
        {"record_kind": T, "heard_chars": None},
        # A count is a number. `False == 0` in Python, and a flag smuggled
        # into a count field must not silence a row.
        {"record_kind": T, "heard_chars": False},
        {"record_kind": T, "heard_chars": "0"},
        # A real receipt, however short.
        {"record_kind": T, "heard_chars": 1},
        # A flag is a bool or a number — never a string, or "false" retracts.
        {"record_kind": T, "heard_chars": 9, "transcript_retracted": False},
        {"record_kind": T, "heard_chars": 9, "transcript_retracted": "false"},
        {"record_kind": T, "heard_chars": 9, "transcript_retracted": 0},
        # Another kind entirely.
        {"record_kind": "task_result", "heard_chars": 0},
        {"record_kind": "something_new", "heard_chars": 0},
    ]
    for voice in speaks:
        assert heard_nothing(voice) is False, voice


def test_the_projection_is_pure_and_composes_with_the_role_guard():
    """It never mutates, and it is the SECOND of two guards, not a rival.

    `public_text` answers "may this ROLE's body be rendered at all";
    `public_heard_text` answers "did the caller hear any of it". A serializer
    composes them; neither one subsumes the other, and a row that is both a
    marker and a silenced transcript must come out blank either way.
    """
    from app.api.message_cards import public_text
    from app.schemas import public_heard_text

    voice = dict(NOTHING_HEARD_VOICE)
    before = json.dumps(voice, sort_keys=True)
    assert public_heard_text(SILENT_EPOCH_TEXT, voice) == ""
    assert json.dumps(voice, sort_keys=True) == before, "voice was mutated"

    # Idempotent: projecting an already-projected body is still empty, and
    # projecting an untouched body twice changes nothing.
    assert public_heard_text("", voice) == ""
    once = public_heard_text(TASK_RESULT_TEXT, TASK_RESULT_VOICE)
    assert public_heard_text(once, TASK_RESULT_VOICE) == once

    # None content is the empty string, never None: `ChatMessageResponse.
    # content` is a required `str` and a None there is a 500 on a history load.
    assert public_heard_text(None, None) == ""
    assert public_heard_text(None, NOTHING_HEARD_VOICE) == ""

    # Composed, in the order the serializers use it.
    assert public_heard_text(public_text("job", "{\"job_id\": \"x\"}"), None) == ""
    assert public_heard_text(
        public_text("assistant", SILENT_EPOCH_TEXT), NOTHING_HEARD_VOICE,
    ) == ""


def test_the_retraction_flag_survives_the_allowlist(caplog):
    """`transcript_retracted` is provenance, and provenance is allowlisted.

    Dropped on the way in, the retraction never reaches a row and every reader
    keeps serving words that were never played — the flag would be a fix that
    exists only in the relay's memory.
    """
    from app.api.sessions import _clean_voice

    with caplog.at_level(logging.WARNING):
        out = _clean_voice(dict(RETRACTED_VOICE))
    assert out == RETRACTED_VOICE
    assert out["transcript_retracted"] is True
    assert "dropped" not in caplog.text.lower(), caplog.text


# ── The enumeration: a new surface cannot silently skip the rule ─────────
#
# The nothing-heard rule has one structural guarantee available to it.
# `public_text` is ALREADY the universal funnel for a client-facing message
# BODY — api/message_cards.py exists because three of four readers once
# returned a raw job marker as the message, and the fix was to make every
# serializer go through one function. So the invariant is exact and
# mechanically checkable: **every call site of `public_text` either composes
# `public_heard_text` around it, or is a named, reasoned non-surface.**
#
# A serializer added next quarter appears in neither list and turns this test
# red, which is the whole point — the previous four parity defects in this
# file's neighbourhood (`tool_events`, `admin_notice`, `channel`, `voice`
# itself) were all "shipped to one reader, forgotten on the fallback".

#: Modules that serialize a stored message for a client. Every `public_text`
#: call in these must be wrapped.
_PROJECTING_MODULES = {
    "app/api/sessions.py",
    "app/api/day_chats.py",
    "app/api/messages_recover.py",
    "app/api/message_frames.py",
}

#: …and the call sites that are NOT message-history surfaces, each with the
#: reason it is out of scope. These are REPORTED GAPS, not blessings: an entry
#: here is a claim about the call site's inputs, and the test below fails if
#: one of them quietly becomes projected (delete the entry) or if a new
#: unprojected site appears anywhere (fix it, or justify it here).
_NOT_A_HISTORY_SURFACE = {
    "app/agent/channel_echo.py": (
        "Echoes a turn that arrived from WhatsApp/Telegram/Discord/Slack "
        "(OFF_APP_ORIGIN_CHANNELS) back to an open app socket, and builds the "
        "frame from the TEXT IT WAS HANDED, not from a stored row — it never "
        "reads metadata_json and a voice row never reaches it."
    ),
    "app/api/chat.py": (
        "GET /api/chat/sessions/{id}/messages — a legacy reader that predates "
        "every metadata key on ChatMessageResponse: it serializes no `voice`, "
        "no `media`, no `tool_events`, no `channel`. It CAN serve a voice "
        "conversation's rows, so it is a real gap; fixing it means teaching "
        "it the whole metadata contract, not just this projection. Handed off "
        "rather than half-fixed here."
    ),
}


def _callee_name(node) -> str:
    import ast as _ast

    func = node.func
    if isinstance(func, _ast.Name):
        return func.id
    if isinstance(func, _ast.Attribute):
        return func.attr
    return ""


def _unprojected_public_text_sites(path):
    """Line numbers of `public_text(...)` calls NOT wrapped in
    `public_heard_text(...)` in one file.

    Structural, not textual: the check is that the `public_text` Call node IS
    the first argument of a `public_heard_text` Call node, which is the shape
    a serializer must use and the one a copy-pasted neighbour will inherit.
    """
    import ast as _ast

    tree = _ast.parse(path.read_text(encoding="utf-8"))
    wrapped = set()
    calls = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        name = _callee_name(node)
        if name == "public_heard_text" and node.args:
            wrapped.add(id(node.args[0]))
        elif name == "public_text":
            calls.append(node)
    return sorted(n.lineno for n in calls if id(n) not in wrapped)


def _public_text_call_sites():
    """`{repo-relative path: [line, ...]}` for every `public_text` CALL under
    `app/`. The definition in api/message_cards.py is a FunctionDef, not a
    call, and `ws_realtime._public_text` is a different name — neither is
    picked up."""
    import ast as _ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    out = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = _ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        lines = [
            n.lineno for n in _ast.walk(tree)
            if isinstance(n, _ast.Call) and _callee_name(n) == "public_text"
        ]
        if lines:
            out[f"app/{path.relative_to(root).as_posix()}"] = sorted(lines)
    return out


def test_every_message_serializer_routes_through_the_projection():
    """No surface may serve a message body without asking whether it was
    heard."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sites = _public_text_call_sites()

    # The funnel still exists and still has call sites. Without this, a rename
    # of `public_text` would make every assertion below vacuously true.
    assert sites, "no public_text call sites found — was the funnel renamed?"
    missing = _PROJECTING_MODULES - set(sites)
    assert not missing, f"expected these to serialize message bodies: {missing}"

    unprojected = {
        rel: _unprojected_public_text_sites(root / rel) for rel in sites
    }
    unprojected = {rel: lines for rel, lines in unprojected.items() if lines}

    for rel in sorted(_PROJECTING_MODULES):
        assert not unprojected.get(rel), (
            f"{rel}:{unprojected[rel]} serializes a message body without "
            "schemas.public_heard_text — an unheard voice transcript would "
            "reach the client as words the caller never heard. Compose it: "
            "public_heard_text(public_text(role, content), voice)"
        )

    strays = set(unprojected) - _PROJECTING_MODULES - set(_NOT_A_HISTORY_SURFACE)
    assert not strays, (
        f"new unprojected public_text call site(s): "
        f"{ {k: unprojected[k] for k in sorted(strays)} }. Either compose "
        "schemas.public_heard_text, or add the file to "
        "_NOT_A_HISTORY_SURFACE with the reason it cannot serve a stored "
        "voice row."
    )

    stale = sorted(set(_NOT_A_HISTORY_SURFACE) - set(unprojected))
    assert not stale, (
        f"{stale} no longer has an unprojected public_text call — the gap is "
        "closed; delete the entry from _NOT_A_HISTORY_SURFACE so the list "
        "keeps meaning what it says."
    )


# ── R2 AF6: the task's NAME survives the write and every read ────────────
#
# The day chat headed a voice run with the nearest user row above it, and on a
# fragmenting relay that row IS a fragment — the recorded cards «نیست», «بگو».
# The app now reads `voice.task_title`: the same string the live `delegation`
# frame carried as `title`, i.e. what the call's own card was headed with. It
# is the one provenance key that carries words, so it is pinned here on both
# sides of the rule it is an exception to: it gets through, and nothing else
# about the allowlist loosens.

#: A request long enough to be cut. The title is built by the relay's OWN
#: function — the one `send_delegation` calls for `delegation.title` — rather
#: than typed as a literal that could drift from it. The leading word carries a
#: ZWNJ, which is not whitespace and must survive every hop.
AF6_REQUEST = (
    "می‌خوام استاد رباتیک دانشگاه تورنتو رو پیدا کنی که ایرانی باشه و "
    "توی پردیس مرکزی کار می‌کنه، و بگو الان کجا درس میده"
)


def _af6_title() -> str:
    from app.services.live_voice_protocol import request_title

    title = request_title(AF6_REQUEST)
    # The fixture is only worth something if it exercises the cut and the ZWNJ.
    assert title.endswith("…") and "‌" in title and len(title) <= 80, title
    return title


def _titled_result_voice() -> dict:
    return {**TASK_RESULT_VOICE, "task_title": _af6_title()}


def test_the_task_title_survives_the_allowlist(caplog):
    from app.api.sessions import _VOICE_KEYS, _clean_voice

    assert "task_title" in _VOICE_KEYS
    sent = _titled_result_voice()
    with caplog.at_level(logging.WARNING, logger="app.api.sessions"):
        out = _clean_voice(dict(sent))
    assert out == sent, out
    assert out["task_title"] == _af6_title()
    assert not [r for r in caplog.records if "voice provenance" in r.getMessage()]


def test_a_task_title_is_one_line_bounded_and_never_logged(caplog):
    """Sanitized like the card title it is: one line, no control characters,
    bounded by the provenance ceiling — and an oversize or empty one is DROPPED,
    not truncated (a half-cut name is a new fragment; without a name the app
    keeps its own title). The drop is logged by key, never by value."""
    from app.api.sessions import _VOICE_STR_MAX, _clean_voice

    out = _clean_voice({"source": "delegated", "task_title": "  find\nthe\x00 professor\t "})
    assert out == {"source": "delegated", "task_title": "find the professor"}, out
    # A ZWNJ is a letter-joiner, not whitespace: it is kept exactly.
    assert _clean_voice({"task_title": "می‌خوام"})["task_title"] == "می‌خوام"
    # At the ceiling it is kept, whole.
    assert _clean_voice({"task_title": "a" * _VOICE_STR_MAX})["task_title"] == "a" * _VOICE_STR_MAX

    secret = "SECRET-WORDS " * 12
    with caplog.at_level(logging.WARNING, logger="app.api.sessions"):
        for bad in (secret, "y" * 100_000, "", "   \n\t ", 42, None, ["x"], {"t": "x"}):
            assert _clean_voice({"source": "delegated", "task_title": bad}) == {
                "source": "delegated",
            }, bad
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "task_title:oversize" in blob and "task_title:empty" in blob, blob
    assert "SECRET-WORDS" not in blob, blob


@pytest.mark.parametrize("with_day_chat", [True, False])
async def test_day_chats_returns_the_task_title_on_both_arms(monkeypatch, with_day_chat):
    """Both arms of the primary history fetch build their own row dict."""
    import app.api.day_chats as api
    from app.db.database import async_session_maker

    async def _no_proxy(*a, **k):
        return None

    monkeypatch.setattr(api, "_get_agent_proxy_info", _no_proxy)
    voice = _titled_result_voice()
    uid, conv, day = await _seed_day(
        [
            ("legacy", "user", "How are admissions?", None),
            ("voice", "assistant", TASK_RESULT_TEXT, voice),
        ],
        with_day_chat=with_day_chat,
    )
    user = await _user(uid)
    async with async_session_maker() as db:
        rows = _rows(await api.get_day_chat_messages(
            date_str=day.isoformat(), limit=500, current_user=user, db=db,
        ))
    voiced, legacy = _split(rows, uid)
    assert voiced["voice"] == voice, voiced
    assert voiced["voice"]["task_title"] == _af6_title()
    assert legacy["voice"] is None


async def test_the_task_title_survives_the_real_write_and_every_read_surface(monkeypatch):
    """The whole hop, driven for real: the relay's body (`_message_payload`,
    the function `_save_voice_messages` calls) → `POST /api/sessions/{id}/
    messages` with the agent key → `metadata_json` → the live frame the route
    broadcasts, the day chat, the session serializer, the by-date fallback and
    the reconnect backstop. Twice: the relay REVISES the delegated row in place
    (the spoken flag, the heard fold) and `_merge_metadata` REPLACES `voice` on
    a rewrite, so the name must ride every revision — which it does because the
    relay copies the whole dict — and must still be there after the second."""
    import asyncio as _asyncio

    from sqlalchemy import select

    import app.api.day_chats as day_chats_api
    import app.api.messages_recover as recover_api
    import app.api.sessions as sessions_mod
    import app.api.ws_chat as ws_chat
    from app.api.ws_realtime import _message_payload, voice_client_msg_id
    from app.config import settings
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.user import User
    from app.schemas import SessionMessageCreate

    agent_key = "test-agent-key-af6"
    monkeypatch.setattr(settings, "agent_api_key", agent_key, raising=False)

    async def _no_proxy(*a, **k):
        return None

    monkeypatch.setattr(day_chats_api, "_get_agent_proxy_info", _no_proxy)
    monkeypatch.setattr(recover_api, "_get_agent_proxy_info", _no_proxy)
    frames: list = []

    async def _capture(_uid, frame):
        frames.append(frame)

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _capture)

    uid, conv = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"af6-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="AF6", timezone="UTC"))
        db.add(Conversation(id=conv, user_id=uid, title="Voice", channel="voice",
                            message_count=0, total_tokens=0))
        await db.commit()

    class _Req:
        headers = {"x-agent-key": agent_key}

    async def _post(body: dict):
        async with async_session_maker() as db:
            user = await db.get(User, uid)
            out = await sessions_mod.create_session_message(
                session_id=conv, body=SessionMessageCreate(**body),
                request=_Req(), current_user=user, db=db,
            )
        await _asyncio.sleep(0)
        return out

    title = _af6_title()
    ask, _ = _message_payload("user", AF6_REQUEST)
    await _post(ask)
    ref = "live-delegation:ps-af6:dlg-7"
    first_voice = {**TASK_RESULT_VOICE, "task_title": title}
    first_voice.pop("spoken")
    for revision, voice in ((1, first_voice), (2, {**first_voice, "spoken": True})):
        body, params = _message_payload(
            "assistant", TASK_RESULT_TEXT, "gpt-live",
            tool_events=[{"tool": "web_search", "started_at_ms": 1}],
            client_msg_id=voice_client_msg_id(conv, ref),
            voice=dict(voice), revision=revision,
        )
        assert params is None and body["voice"]["task_title"] == title
        await _post(body)

    answer_frames = [f for f in frames if f.get("role") == "assistant"]
    assert len(answer_frames) == 2, frames
    for f in answer_frames:
        assert f["voice"]["task_title"] == title, f
    assert answer_frames[-1]["voice"]["spoken"] is True

    async with async_session_maker() as db:
        msgs = (await db.execute(
            select(Message).where(Message.conversation_id == conv)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )).scalars().all()
        assert [m.role for m in msgs] == ["user", "assistant"], "one row, revised in place"
        ask_row, answer_row = msgs
        stored = json.loads(answer_row.metadata_json)
        assert stored["voice"]["task_title"] == title
        assert stored["tool_events"][0]["tool"] == "web_search"
        served = sessions_mod._message_to_response(answer_row, None, None, {conv: "voice"})
        assert served.model_dump()["voice"]["task_title"] == title
        day = (answer_row.occurred_at or answer_row.created_at).date()
        ask_id, answer_id = ask_row.id, answer_row.id

    user = await _user(uid)
    async with async_session_maker() as db:
        day_rows = _rows(await day_chats_api.get_day_chat_messages(
            date_str=day.isoformat(), limit=500, current_user=user, db=db,
        ))
    async with async_session_maker() as db:
        by_date = _rows(await sessions_mod.get_messages_by_date(
            day.isoformat(), limit=200, tz_offset=0, current_user=user, db=db,
        ))
    async with async_session_maker() as db:
        since = _rows(await recover_api.messages_since(
            ask_id, limit=100, current_user=user, db=db,
        ))
    for where, rows in (("day_chats", day_rows), ("sessions.by_date", by_date),
                        ("messages_since", since)):
        row = next((r for r in rows if r["id"] == answer_id), None)
        assert row is not None, f"{where}: the answer row is missing"
        assert row["voice"]["task_title"] == title, f"{where}: {row['voice']}"
