"""What a voice turn PRODUCED has to reach the thread, exactly once.

`POST /api/sessions/{id}/messages` is the only writer for a spoken turn (the
agent runs it with save=False). It carried text, media and tool_events and
nothing else — so a PDF generated during a call had no Message row for
`GET /api/files/{message_id}/{aid}` to authorize against, and a `present_app`
made by voice was unreachable. `attachments` goes to the COLUMN (which every
history serializer already reads), `app_artifact` to metadata_json (the key
`AgentRunner._save_messages` writes).

The other half is identity: `client_msg_id` makes a replayed persist an UPSERT
rather than the user's sentence twice in their thread, and `occurred_at` is when
the utterance happened rather than when the database got round to it.

Both identity columns are NULLABLE. They used to be gated here by a
module-level `hasattr` probe, which meant that if the model hunk were reverted
or lost in a merge this file would report GREEN while asserting nothing about
identity — six of its tests skipped, and a skipped test that reads green guards
nothing. They land in THIS change (OWNERSHIP A12), so their presence is now a
test rather than a condition.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import pytest

from app.config import settings
from app.db import async_session_maker
from app.db.models import Conversation, Message, User
from app.schemas import SessionMessageCreate

AGENT_KEY = "test-agent-key-for-this-module"


class _Req:
    """The handler reads exactly one thing off the request: whether it carries
    the agent key. Everything else about it is FastAPI's business."""

    def __init__(self, headers: dict | None = None):
        self.headers = dict(headers or {})


@pytest.fixture(autouse=True)
def _agent_key(monkeypatch):
    """`attachments[].storage_path` is a capability and is accepted only from
    the platform voice relay, which authenticates with this key."""
    monkeypatch.setattr(settings, "agent_api_key", AGENT_KEY, raising=False)
    yield


def test_the_identity_columns_exist():
    """Not a skipif. A12 lands `client_msg_id` and `occurred_at` in this same
    change; if the model hunk is lost, every identity assertion below must go
    RED, not quietly absent."""
    assert hasattr(Message, "client_msg_id")
    assert hasattr(Message, "occurred_at")


async def _seed():
    user_id, conv_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        user = User(
            id=user_id, email=f"att-{user_id[:8]}@example.test",
            hashed_password="x", name="Attachment Test",
        )
        db.add(user)
        db.add(Conversation(
            id=conv_id, user_id=user_id, title="Voice", channel="voice",
            message_count=0, total_tokens=0,
        ))
        await db.commit()
    return user_id, conv_id


async def _post(
    conv_id: str, user_id: str, body: SessionMessageCreate, frames: list,
    *, trusted: bool = True,
):
    """Call the route function with a real session, capturing the live frame.

    `trusted` models the X-Agent-Key the platform voice relay sends; without it
    the caller is an ordinary JWT holder.
    """
    from app.api import sessions as sessions_mod
    import app.api.ws_chat as ws_chat

    async def _capture(uid, frame):
        frames.append((uid, frame))

    orig = ws_chat.broadcast_to_user
    ws_chat.broadcast_to_user = _capture
    try:
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            return await sessions_mod.create_session_message(
                session_id=conv_id, body=body,
                request=_Req({"x-agent-key": AGENT_KEY} if trusted else {}),
                current_user=user, db=db,
            )
    finally:
        ws_chat.broadcast_to_user = orig


@pytest.mark.asyncio
async def test_attachments_land_on_the_column_not_in_metadata():
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Here is the report.",
        attachments=[{"id": "a1", "filename": "report.pdf", "mime_type": "application/pdf"}],
    ), frames)

    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        assert row.attachments and row.attachments[0]["id"] == "a1", (
            "the file route authorizes against this COLUMN; metadata_json is invisible to it"
        )
        assert "attachments" not in json.loads(row.metadata_json or "{}")


@pytest.mark.asyncio
async def test_app_artifact_lands_in_metadata_under_the_key_clients_read():
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Built it.", app_artifact={"slug": "snake", "revision": 3},
    ), frames)

    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        meta = json.loads(row.metadata_json or "{}")
        assert meta["app_artifact"]["slug"] == "snake"
        assert meta["app_artifact"]["revision"] == 3


@pytest.mark.asyncio
async def test_the_live_frame_carries_both():
    user_id, conv_id = await _seed()
    frames: list = []
    await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Here.",
        attachments=[{"id": "a1"}], app_artifact={"slug": "snake"},
        media={"type": "youtube", "video_id": "v"},
        tool_events=[{"tool": "web_search", "started_at_ms": 1}],
    ), frames)

    assert frames, "a committed row must be announced, not left for the next refetch"
    _uid, frame = frames[0]
    assert frame["type"] == "message"
    assert frame["attachments"][0]["id"] == "a1"
    assert frame["app_artifact"]["slug"] == "snake"
    assert frame["media"]["video_id"] == "v"
    assert frame["tool_events"]
    assert frame["channel"] == "voice"


@pytest.mark.asyncio
async def test_a_turn_that_produced_nothing_looks_exactly_as_it_did():
    """An older platform build and this one must be indistinguishable for a
    plain spoken turn."""
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="hello",
    ), frames)

    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        assert not row.attachments
        assert row.metadata_json is None
    _uid, frame = frames[0]
    for k in ("attachments", "app_artifact", "media", "tool_events"):
        assert k not in frame


@pytest.mark.asyncio
async def test_the_live_frame_carries_the_wire_form_of_an_attachment():
    """The frame used to ship the RAW body dicts: no download_url, so a file
    produced during a voice turn rendered as a dead card until the next history
    refetch — and `storage_path`, the internal object key (and the tenant id
    inside it), went out with it. Both REST serializers enrich and strip; this
    is the one surface that bypasses both."""
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Here is the deck.",
        attachments=[{
            "id": "a1", "filename": "deck.pptx", "size_bytes": 10,
            "mime_type":
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "storage_path": f"{user_id}/a1_deck.pptx",
        }],
    ), frames)

    _uid, frame = frames[0]
    att = frame["attachments"][0]
    assert att["download_url"].endswith(f"/files/{resp.id}/a1"), (
        "AttachmentCard reads download_url; without it the card is dead"
    )
    assert att["preview_url"], "one preview policy — PPTX is previewable"
    assert att["kind"] == "pptx" and att["role"] == "final"
    assert "storage_path" not in att, "the internal object key reached the client"
    # …and the COLUMN keeps it, because files.py authorizes against it.
    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        assert row.attachments[0]["storage_path"].endswith("a1_deck.pptx")


@pytest.mark.asyncio
async def test_the_upsert_key_includes_the_role():
    """`client_msg_id` is deliberately NOT unique: a chat turn stamps the same
    value on its user row AND its assistant row (db/models/conversation.py).
    Keyed on (conversation, client_msg_id) alone the second POST overwrote the
    first row's role and content instead of appending — the user's message,
    silently replaced by the answer to it."""
    user_id, conv_id = await _seed()
    frames: list = []
    u = await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="what is my balance", client_msg_id="turn-1",
    ), frames)
    a = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Forty dollars.", client_msg_id="turn-1",
    ), frames)

    assert u.id != a.id, "one turn's two halves collapsed onto one row"
    async with async_session_maker() as db:
        user_row = await db.get(Message, u.id)
        assert user_row.role == "user"
        assert user_row.content == "what is my balance"
        conv = await db.get(Conversation, conv_id)
        assert conv.message_count == 2


@pytest.mark.asyncio
async def test_an_implausible_occurred_at_is_treated_as_unknown():
    """`occurred_at` is unvalidated client input that drives the thread's sort
    key. A far-future stamp pins a row to the bottom of the day permanently;
    C1 already defines null as 'unknown', which falls back to created_at."""
    user_id, conv_id = await _seed()
    frames: list = []
    far = await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="hello", client_msg_id="cm-far",
        occurred_at=datetime.utcnow() + timedelta(days=400),
    ), frames)
    old = await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="hello again", client_msg_id="cm-old",
        occurred_at=datetime.utcnow() - timedelta(days=400),
    ), frames)
    near = await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="and again", client_msg_id="cm-near",
        occurred_at=datetime.utcnow() - timedelta(minutes=2),
    ), frames)

    async with async_session_maker() as db:
        assert (await db.get(Message, far.id)).occurred_at is None
        assert (await db.get(Message, old.id)).occurred_at is None
        assert (await db.get(Message, near.id)).occurred_at is not None


class _RaisesOnIdentityLookup:
    """A tenant DB whose self-heal has not added the column yet: the SELECT on
    `client_msg_id` raises and everything else works."""

    def __init__(self, db):
        self._db = db
        self.raised = False

    def __getattr__(self, name):
        return getattr(self._db, name)

    async def execute(self, stmt, *args, **kwargs):
        if not self.raised and "client_msg_id" in str(stmt):
            self.raised = True
            raise RuntimeError(
                'column messages.client_msg_id does not exist'
            )
        return await self._db.execute(stmt, *args, **kwargs)


class _RaisesOnFirstCommit:
    """The INSERT itself is rejected: the same shape ws_chat retries around."""

    def __init__(self, db):
        self._db = db
        self.raised = False

    def __getattr__(self, name):
        return getattr(self._db, name)

    async def commit(self):
        if not self.raised:
            self.raised = True
            raise RuntimeError(
                'column "client_msg_id" of relation "messages" does not exist'
            )
        return await self._db.commit()


@pytest.mark.asyncio
async def test_a_tenant_without_the_column_degrades_instead_of_500ing():
    """Both degradation paths `await db.rollback()` first, and rollback EXPIRES
    every instance in the session — so the very next read of
    `session.message_count` raised MissingGreenlet and the handler 500'd
    instead of performing the plain insert it was written to perform."""
    from app.api import sessions as sessions_mod
    import app.api.ws_chat as ws_chat

    async def _capture(uid, frame):
        return None

    orig = ws_chat.broadcast_to_user
    ws_chat.broadcast_to_user = _capture
    try:
        for wrapper in (_RaisesOnIdentityLookup, _RaisesOnFirstCommit):
            user_id, conv_id = await _seed()
            async with async_session_maker() as db:
                user = await db.get(User, user_id)
                resp = await sessions_mod.create_session_message(
                    session_id=conv_id,
                    body=SessionMessageCreate(
                        role="user", content="what is my balance",
                        client_msg_id="cm-degraded",
                    ),
                    current_user=user, db=wrapper(db),
                )
            async with async_session_maker() as db:
                row = await db.get(Message, resp.id)
                assert row is not None and row.content == "what is my balance", (
                    f"{wrapper.__name__}: the transcript was lost"
                )
                conv = await db.get(Conversation, conv_id)
                assert conv.message_count == 1, wrapper.__name__
    finally:
        ws_chat.broadcast_to_user = orig


@pytest.mark.asyncio
async def test_a_replay_upserts_instead_of_duplicating_the_turn():
    user_id, conv_id = await _seed()
    frames: list = []
    body = SessionMessageCreate(
        role="user", content="what is my balance",
        client_msg_id="voice-cm-1",
        # Relative, never a literal date: `occurred_at` is now rejected as
        # implausible more than a day from now, so a baked stamp would turn this
        # file red at a wall-clock boundary rather than on a code change.
        occurred_at=datetime.utcnow() - timedelta(minutes=1),
    )
    first = await _post(conv_id, user_id, body, frames)
    second = await _post(conv_id, user_id, body, frames)
    assert first.id == second.id, "a replayed persist must land on the row it already wrote"

    async with async_session_maker() as db:
        conv = await db.get(Conversation, conv_id)
        assert conv.message_count == 1, "the count must not advance on an upsert"


@pytest.mark.asyncio
async def test_occurred_at_orders_the_two_halves_of_one_turn():
    user_id, conv_id = await _seed()
    frames: list = []
    base = datetime.utcnow() - timedelta(minutes=5)
    # Written in the WRONG order on purpose — the provider can finish a response
    # before the transcription of the utterance that caused it.
    a = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Forty dollars.",
        client_msg_id="cm-a", occurred_at=base + timedelta(milliseconds=1),
    ), frames)
    u = await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="what is my balance",
        client_msg_id="cm-u", occurred_at=base,
    ), frames)

    async with async_session_maker() as db:
        rows = {r.id: r for r in [await db.get(Message, a.id), await db.get(Message, u.id)]}
        assert rows[u.id].occurred_at < rows[a.id].occurred_at
        assert rows[u.id].created_at > rows[a.id].created_at, (
            "insert order is the thing occurred_at exists to override"
        )


# ── `attachments` is untrusted input with a capability inside it ────────────

@pytest.mark.asyncio
async def test_an_ordinary_caller_cannot_name_a_storage_key():
    """`storage_path` is the object key `files.py` streams — the ONE field in
    this body that grants a read rather than describing one. Until C8 opened
    the column to a request body it had only server-controlled writers; a JWT
    holder naming an arbitrary key under the workspace would mint itself a
    download of a file it never produced, including, on a recycled pool slot, a
    file a previous tenant left behind."""
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Here.",
        attachments=[
            # Both keys would pass the SCOPE check — they are inside the
            # caller's own namespace — so only the "did this come from the
            # relay" test can reject them. Written this way on purpose: with a
            # foreign prefix here the scope check would mask the trust check
            # and a mutation of either would survive.
            {"id": "a1", "filename": "not-mine.pdf",
             "storage_path": f"{user_id}/abc_secret.pdf"},
            {"id": "a2", "filename": "also-not-mine.pdf",
             "storage_path": "shared/abc_secret.pdf"},
        ],
    ), frames, trusted=False)

    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        assert [a["id"] for a in row.attachments] == ["a1", "a2"], (
            "the row still records what was produced"
        )
        for att in row.attachments:
            assert "storage_path" not in att, (
                "a caller that is not the relay granted itself a download"
            )


@pytest.mark.asyncio
async def test_even_the_relay_cannot_name_a_key_outside_the_user_scope():
    """The relay forwards what the agent returned. A capability field has to
    survive a confused producer, not only a hostile caller."""
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Here.",
        attachments=[
            {"id": "a1", "storage_path": "../../etc/passwd"},
            {"id": "a2", "storage_path": "another-tenant/abc_x.pdf"},
            {"id": "a3", "storage_path": f"{user_id}/abc_mine.pdf"},
        ],
    ), frames)

    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        by_id = {a["id"]: a for a in row.attachments}
        assert "storage_path" not in by_id["a1"]
        assert "storage_path" not in by_id["a2"]
        assert by_id["a3"]["storage_path"].endswith("abc_mine.pdf"), (
            "the user's own key must survive or every voice-produced file is dead"
        )


@pytest.mark.asyncio
async def test_unknown_attachment_fields_are_dropped_and_the_list_is_capped():
    """Key-allowlisted like its sibling `_clean_tool_events`, and bounded by
    A13's MAX_ATTACHMENTS_PER_TURN — an unbounded list is an unbounded row."""
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Here.",
        attachments=[{"id": f"a{i}", "filename": "x.pdf", "evil": "x" * 100}
                     for i in range(20)],
    ), frames)

    async with async_session_maker() as db:
        row = await db.get(Message, resp.id)
        assert len(row.attachments) == 8
        assert all("evil" not in a for a in row.attachments)
        assert row.attachments[0]["filename"] == "x.pdf"


@pytest.mark.asyncio
async def test_a_role_the_clients_cannot_render_is_refused():
    """`role` is written to the row, is half of the UPSERT key and is stamped
    on the live frame. The pydantic pattern is only a gate while nothing can
    route round it — the handler used to accept an unvalidated query value
    whenever the body omitted the field."""
    from fastapi import HTTPException

    user_id, conv_id = await _seed()
    frames: list = []
    with pytest.raises(HTTPException) as exc:
        # `model_copy(update=…)` skips validation — exactly what a caller that
        # routed round the pydantic pattern hands the handler.
        await _post(conv_id, user_id, SessionMessageCreate(
            content="x",
        ).model_copy(update={"role": "system"}), frames)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_the_query_form_is_gone_from_the_signature():
    """The sender stopped putting the transcript in the URL; the RECEIVER has
    to stop accepting it, or the leak channel is open for anything that still
    uses the old form (a staged rollout, an out-of-tree caller)."""
    import inspect
    from app.api import sessions as sessions_mod

    params = inspect.signature(sessions_mod.create_session_message).parameters
    for gone in ("content", "role", "model_used", "day_chat_id"):
        assert gone not in params, (
            f"`{gone}` is a bare scalar with a default, i.e. a QUERY PARAMETER — "
            "a spoken sentence sent that way lands in the access log"
        )


# ── The day the row belongs to must still EXIST after a degradation ────────

@pytest.mark.asyncio
async def test_a_degraded_insert_still_points_at_a_real_day():
    """`get_or_create_day_chat` INSERTs the DayChat and only flushes it, so the
    `await db.rollback()` in each degradation path destroys a row the Message's
    `day_chat_id` foreign key still names. On Postgres the follow-up commit
    then raises a FK violation neither handler expects — a 500 on the path
    written to degrade, and a lost voice turn."""
    from app.db.models import DayChat

    for wrapper in (_RaisesOnIdentityLookup, _RaisesOnFirstCommit):
        user_id, conv_id = await _seed()
        import app.api.ws_chat as ws_chat
        from app.api import sessions as sessions_mod

        async def _capture(uid, frame):
            return None

        orig = ws_chat.broadcast_to_user
        ws_chat.broadcast_to_user = _capture
        try:
            async with async_session_maker() as db:
                user = await db.get(User, user_id)
                resp = await sessions_mod.create_session_message(
                    session_id=conv_id,
                    body=SessionMessageCreate(
                        role="user", content="hello", client_msg_id="cm-day",
                    ),
                    request=_Req({"x-agent-key": AGENT_KEY}),
                    current_user=user, db=wrapper(db),
                )
        finally:
            ws_chat.broadcast_to_user = orig

        async with async_session_maker() as db:
            row = await db.get(Message, resp.id)
            assert row.day_chat_id, f"{wrapper.__name__}: the row lost its day"
            assert await db.get(DayChat, row.day_chat_id) is not None, (
                f"{wrapper.__name__}: day_chat_id names a DayChat the rollback "
                "destroyed — a dangling foreign key"
            )


@pytest.mark.asyncio
async def test_an_upsert_with_nothing_to_say_does_not_erase_the_card():
    """A replayed assistant persist after a relay reconnect carries no media
    and no tool_events by construction (both are per-socket state). The upsert
    copied every field, so the second write NULLed the media card and the run
    rail off a row that already had them — while `attachments` degraded the
    opposite way, because it is only written when non-empty."""
    user_id, conv_id = await _seed()
    frames: list = []
    first = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Playing it.", client_msg_id="cm-replay",
        media={"type": "youtube", "video_id": "v"},
        tool_events=[{"tool": "play_media", "started_at_ms": 1}],
    ), frames)
    second = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Playing it.", client_msg_id="cm-replay",
    ), frames)
    assert first.id == second.id

    async with async_session_maker() as db:
        meta = json.loads((await db.get(Message, first.id)).metadata_json or "{}")
        assert meta.get("media", {}).get("video_id") == "v", "the card was erased by a replay"
        assert meta.get("tool_events"), "the run rail was erased by a replay"
