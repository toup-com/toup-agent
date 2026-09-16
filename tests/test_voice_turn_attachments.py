"""A picture made during a VOICE turn has to reach the thread.

Before round 46 it reached nowhere. `/internal/agent-turn` runs with
`save=False` (the realtime relay persists the turn itself), so no Message row
was written by the runner; `ChatResponse` had no `attachments` field, and
`SessionMessageCreate` had none either. An image the agent generated or edited
while the user was talking existed only in storage and in the workspace: no
card, no row, nothing on reopening the app. `media` and `tool_events` were each
added to that same schema after exactly this bug — their own docstrings say so.

Two hops, both driven here:

  1. `/internal/agent-turn` returns the turn's attachments on `ChatResponse`,
     read off `AgentResponse.persisted` (the documented echo of what the turn
     wrote, or of what the caller must write when it owns persistence).
  2. `POST /api/sessions/{id}/messages` persists them onto the
     `Message.attachments` COLUMN — not into `metadata_json`. The column is
     what `GET /api/files/{message_id}/{aid}` authorizes against and what every
     history serializer already reads; in metadata_json the picture would be
     just as invisible as before.

Needs RUN_MODE=agent: conversations / messages are AGENT_ONLY. See
COVERAGE_DEBT.txt.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.asyncio


def _attachment(att_id: str = "a" * 32) -> dict:
    return {"id": att_id, "filename": "edited_1a2b3c4d.png",
            "mime_type": "image/png", "size_bytes": 4096,
            "storage_path": f"u/{att_id}_edited.png",
            "created_at": "2026-09-15T00:00:00Z",
            "width": 1024, "height": 1536, "has_thumb": True}


class _Req:
    """Only the header the route reads."""

    def __init__(self, key: str):
        self.headers = {"X-Agent-Key": key}


class _Resp:
    """The shape AgentRunner.run returns, with only what the route reads."""

    def __init__(self, persisted):
        self.text = "Here it is."
        self.session_id = "s1"
        self.tokens_input = 1
        self.tokens_output = 1
        self.tokens_total = 2
        self.model = "m"
        self.tool_calls = []
        self.processing_time_ms = 5
        self.persisted = persisted


async def test_agent_turn_returns_the_picture_the_voice_turn_made(monkeypatch):
    """Hop 1. save=False writes no row, so the caller must be TOLD."""
    from app.api import api_v1
    from app.config import settings

    monkeypatch.setattr(settings, "user_id", str(uuid.uuid4()), raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "k-test", raising=False)

    att = _attachment()
    captured = {}

    class _Runner:
        async def run(self, **kw):
            captured.update(kw)
            return _Resp({"attachments": [att]})

    monkeypatch.setattr(api_v1, "_agent_runner", _Runner(), raising=False)

    out = await api_v1.internal_agent_turn(
        api_v1.ChatRequest(message="edit that photo", session_id="s1", save=False),
        _Req("k-test"),
    )

    assert captured["save_assistant_message"] is False, "the relay owns persistence"
    assert out.attachments == [att], (
        "the one channel a voice-made picture has back to the caller"
    )
    assert out.attachments[0]["id"] == att["id"]


async def test_no_attachments_is_an_empty_list_not_a_missing_field(monkeypatch):
    """A relay on an older build must see today's shape, never a null."""
    from app.api import api_v1
    from app.config import settings

    monkeypatch.setattr(settings, "user_id", str(uuid.uuid4()), raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "k-test", raising=False)

    class _Runner:
        async def run(self, **kw):
            return _Resp({})

    monkeypatch.setattr(api_v1, "_agent_runner", _Runner(), raising=False)
    out = await api_v1.internal_agent_turn(
        api_v1.ChatRequest(message="hello", session_id="s1", save=False),
        _Req("k-test"))
    assert out.attachments == []


async def _seed_session() -> tuple[object, str]:
    from app.db import async_session_maker, User
    from app.db.models import Conversation

    uid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@test.local",
                    hashed_password="x" * 60, name="R46"))
        db.add(Conversation(id=sid, user_id=uid, channel="voice", is_active=True))
        await db.commit()
        user = await db.get(User, uid)
        db.expunge(user)
    return user, sid


async def test_the_voice_persist_writes_the_attachments_column(monkeypatch):
    """Hop 2. The COLUMN — metadata_json would be invisible to every reader."""
    from app.api.sessions import create_session_message
    from app.db import async_session_maker
    from app.db.models import Message
    from app.schemas import SessionMessageCreate
    from sqlalchemy import select

    user, sid = await _seed_session()
    att = _attachment("b" * 32)

    async with async_session_maker() as db:
        await create_session_message(
            session_id=sid,
            body=SessionMessageCreate(role="assistant", content="Here it is.",
                                      attachments=[att]),
            current_user=user, db=db,
        )

    async with async_session_maker() as db:
        row = (await db.execute(select(Message).where(
            Message.conversation_id == sid))).scalars().one()

    assert row.attachments and row.attachments[0]["id"] == att["id"]
    assert row.attachments[0]["mime_type"] == "image/png"
    # The card lays out at the real shape on first paint.
    assert row.attachments[0]["width"] == 1024
    # `metadata_json` is a JSON STRING; `"attachments" not in "<a dict-shaped
    # string>"` was true for the wrong reason and stayed true either way.
    import json as _json
    assert "attachments" not in _json.loads(row.metadata_json or "{}"), (
        "attachments has a column; metadata_json is for the fields that do not"
    )


async def test_a_turn_with_no_picture_writes_no_attachments(monkeypatch):
    from app.api.sessions import create_session_message
    from app.db import async_session_maker
    from app.db.models import Message
    from app.schemas import SessionMessageCreate
    from sqlalchemy import select

    user, sid = await _seed_session()
    async with async_session_maker() as db:
        await create_session_message(
            session_id=sid,
            body=SessionMessageCreate(role="assistant", content="Sure."),
            current_user=user, db=db,
        )
    async with async_session_maker() as db:
        row = (await db.execute(select(Message).where(
            Message.conversation_id == sid))).scalars().one()
    assert not row.attachments


async def test_the_STREAM_done_frame_carries_them_too(monkeypatch):
    """Hop 1 on the path production voice actually takes.

    `ws_realtime._think` tries `/internal/agent-turn/stream` FIRST and falls
    back to the blocking POST only on a transport error, and
    `voice_realtime_tool_events` is enabled in production — so the only branch
    the previous test covered is the one for old agent images. The keys are
    added with a conditional dict-unpack, a shape easy to lose in a merge.
    """
    from app.api import api_v1
    from app.config import settings

    monkeypatch.setattr(settings, "run_mode", "agent", raising=False)
    monkeypatch.setattr(settings, "user_id", str(uuid.uuid4()), raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "k-test", raising=False)

    att = _attachment("c" * 32)

    class _Runner:
        async def run(self, **kw):
            return _Resp({"attachments": [att], "app_artifact": {"slug": "budget"}})

    monkeypatch.setattr(api_v1, "_agent_runner", _Runner(), raising=False)

    resp = await api_v1.internal_agent_turn_stream(
        api_v1.ChatRequest(message="edit that photo", session_id="s1", save=False),
        _Req("k-test"),
    )
    done = await _collect_done(resp)
    assert done["attachments"][0]["id"] == att["id"], (
        "the streaming path — the one voice takes — dropped the turn's file"
    )
    assert done["app_artifact"] == {"slug": "budget"}


async def test_the_stream_omits_the_keys_entirely_when_nothing_was_produced(monkeypatch):
    """Absent, not null and not an empty list: an older relay's parse must be
    unchanged by a turn that produced nothing."""
    from app.api import api_v1
    from app.config import settings

    monkeypatch.setattr(settings, "run_mode", "agent", raising=False)
    monkeypatch.setattr(settings, "user_id", str(uuid.uuid4()), raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "k-test", raising=False)

    class _Runner:
        async def run(self, **kw):
            return _Resp({})

    monkeypatch.setattr(api_v1, "_agent_runner", _Runner(), raising=False)
    resp = await api_v1.internal_agent_turn_stream(
        api_v1.ChatRequest(message="hello", session_id="s1", save=False),
        _Req("k-test"))
    done = await _collect_done(resp)
    assert "attachments" not in done and "app_artifact" not in done


async def _collect_done(resp) -> dict:
    """Read the SSE body and return the terminal `done` frame."""
    import json as _json

    frames = []
    async for chunk in resp.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        for line in chunk.splitlines():
            if line.startswith("data: "):
                frames.append(_json.loads(line[6:]))
    done = [f for f in frames if f.get("type") == "done"]
    assert done, f"the stream never terminated with a done frame: {frames}"
    return done[-1]
