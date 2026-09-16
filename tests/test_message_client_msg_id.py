"""The turn's identity, persisted — on BOTH rows, and null-tolerant forever.

Round 46, A12. Before this there was exactly ONE durable record of a
`client_msg_id` in the whole system — `ProcessedMessage`, a dedup ledger that
is never served — so the app's only way to recognise its own message was role +
byte-equal content within ±10 s. On 2026-09-15 a WhatsApp "Hi" persisted at
14:50:29 was a legal twin for an in-app "Hi" sent at ~14:49:48: the optimistic
bubble was adopted by the WhatsApp row, then orphaned by the next resync when
the real in-app row arrived, and the user watched their own message disappear
and reappear lower down the thread.

Two properties, and the second is the one that will actually bite:

  * both rows of a turn carry the id, so a client can pair with either; and
  * NULL means UNKNOWN, never "not mine". Every pre-existing row is null, and
    so is every row written by a container still on image 1cd801aacb11 — for a
    full rollout cycle some tenants stamp the field and some do not, in the
    same user's day thread.

`conversations` / `messages` / `processed_messages` are AGENT_ONLY
(app/db/models/base.py), which is why this runs in the agent lane.

Local run (from backend/):
    RUN_MODE=agent PYTHONPATH=. pytest tests/test_message_client_msg_id.py \
      -q -p no:cacheprovider
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio

USER = "00000000-0000-4000-8000-000000000001"  # synthetic (IMPL_RULES §5)


@pytest_asyncio.fixture
async def requires_message_tables():
    from sqlalchemy import inspect as _inspect
    from app.db.database import engine

    async with engine.connect() as conn:
        present = await conn.run_sync(
            lambda sc: {n: _inspect(sc).has_table(n)
                        for n in ("conversations", "messages", "processed_messages")}
        )
    missing = [n for n, ok in present.items() if not ok]
    if missing:
        pytest.skip(f"requires AGENT_ONLY table(s) {missing} — run with RUN_MODE=agent")


async def _conversation(db, cid: str):
    from app.db.models import Conversation

    db.add(Conversation(id=cid, user_id=USER, channel="mobile"))
    await db.flush()


def test_the_self_heal_really_creates_the_columns_and_their_indexes():
    """The self-heal path IS the migration here — agent DBs have no alembic.

    This asserts the STATEMENTS, because the live-table test below cannot:
    that DB is built by `create_all` from the SQLAlchemy models, so deleting
    all four lines from `_alter_statements` left the whole file green
    (mutation run, round 46 review). On Postgres the models are not consulted
    at boot at all — `init_db` runs these ALTERs and nothing else — and the
    `_reconcile_model_columns` backstop can add a COLUMN but never an INDEX,
    which is what makes the receipt lookup and the day-thread sort affordable.
    If these stop running, every SELECT that names them 500s on every chat
    turn (the 2026-06-29 `users.apple_sub` class)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app/db/database.py").read_text()
    i = src.index("_alter_statements = [")
    block = src[i:src.index("\n    ]", i)]
    for stmt in (
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS client_msg_id VARCHAR(100)",
        "CREATE INDEX IF NOT EXISTS ix_messages_client_msg_id ON messages (client_msg_id)",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS occurred_at TIMESTAMP",
        "CREATE INDEX IF NOT EXISTS ix_messages_occurred_at ON messages (occurred_at)",
    ):
        assert stmt in block, f"missing from _alter_statements: {stmt}"
    # VARCHAR(100) is load-bearing: ws_chat bounds the wire value against it.
    from app.api.ws_chat import _MAX_CLIENT_MSG_ID_LEN

    assert _MAX_CLIENT_MSG_ID_LEN == 100


async def test_the_columns_exist_on_the_live_table(requires_message_tables):
    """The model side of the same fact. Under sqlite this is `create_all` from
    the models, so it proves the MODEL carries the columns — never that the
    self-heal runs; that is the test above."""
    from sqlalchemy import inspect as _inspect
    from app.db.database import engine

    async with engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in _inspect(sc).get_columns("messages")}
        )
    assert "client_msg_id" in cols
    assert "occurred_at" in cols


async def test_both_rows_of_a_turn_carry_the_id(requires_message_tables):
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models import Message

    cid = str(uuid.uuid4())
    cmid = "cm-" + uuid.uuid4().hex[:8]
    async with async_session_maker() as db:
        await _conversation(db, cid)
        db.add(Message(id=str(uuid.uuid4()), conversation_id=cid, role="user",
                       content="Hi", channel="mobile", client_msg_id=cmid))
        db.add(Message(id=str(uuid.uuid4()), conversation_id=cid, role="assistant",
                       content="Hello", channel="mobile", client_msg_id=cmid))
        await db.commit()

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Message).where(Message.client_msg_id == cmid)
        )).scalars().all()
    assert {r.role for r in rows} == {"user", "assistant"}, "a client must be able to pair with either"


async def test_the_column_is_not_unique(requires_message_tables):
    """Two rows sharing it is the DESIGN, not a collision. A unique index here
    would make the assistant insert fail on every turn."""
    from app.db.database import async_session_maker
    from app.db.models import Message

    cid = str(uuid.uuid4())
    cmid = "cm-" + uuid.uuid4().hex[:8]
    async with async_session_maker() as db:
        await _conversation(db, cid)
        for role in ("user", "assistant"):
            db.add(Message(id=str(uuid.uuid4()), conversation_id=cid, role=role,
                           content="x", client_msg_id=cmid))
        await db.commit()   # must not raise


async def test_a_row_with_no_id_is_still_writable(requires_message_tables):
    """Every pre-existing row, every channel turn, every routine. The column is
    nullable forever."""
    from app.db.database import async_session_maker
    from app.db.models import Message

    cid = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _conversation(db, cid)
        m = Message(id=str(uuid.uuid4()), conversation_id=cid, role="user", content="x")
        db.add(m)
        await db.commit()
        assert m.client_msg_id is None
        assert m.occurred_at is None


async def test_occurred_at_is_when_it_HAPPENED_not_when_it_was_written(requires_message_tables):
    """A voice utterance or a WhatsApp message can be persisted long after the
    fact, and a day thread sorted by write time puts it in the wrong place."""
    from sqlalchemy import select, func
    from app.db.database import async_session_maker
    from app.db.models import Message

    cid = str(uuid.uuid4())
    now = datetime.utcnow()
    async with async_session_maker() as db:
        await _conversation(db, cid)
        # Written LAST, happened FIRST.
        db.add(Message(id="b-late", conversation_id=cid, role="user", content="spoken",
                       created_at=now + timedelta(minutes=10),
                       occurred_at=now - timedelta(minutes=5)))
        db.add(Message(id="a-typed", conversation_id=cid, role="user", content="typed",
                       created_at=now))
        await db.commit()

    async with async_session_maker() as db:
        ordered = (await db.execute(
            select(Message.id)
            .where(Message.conversation_id == cid)
            .order_by(func.coalesce(Message.occurred_at, Message.created_at).asc(),
                      Message.id.asc())
        )).scalars().all()
    assert ordered == ["b-late", "a-typed"], ordered


async def test_the_sort_key_is_todays_order_when_nothing_stamps_occurred_at(requires_message_tables):
    """The mid-rollout guarantee: with every `occurred_at` null, COALESCE is
    `created_at` and the thread reads exactly as it does today."""
    from sqlalchemy import select, func
    from app.db.database import async_session_maker
    from app.db.models import Message

    cid = str(uuid.uuid4())
    now = datetime.utcnow()
    async with async_session_maker() as db:
        await _conversation(db, cid)
        for i in range(4):
            db.add(Message(id=f"m{i}", conversation_id=cid, role="user", content=str(i),
                           created_at=now + timedelta(seconds=i)))
        await db.commit()

    async with async_session_maker() as db:
        by_coalesce = (await db.execute(
            select(Message.id).where(Message.conversation_id == cid)
            .order_by(func.coalesce(Message.occurred_at, Message.created_at).asc(),
                      Message.id.asc())
        )).scalars().all()
        by_created = (await db.execute(
            select(Message.id).where(Message.conversation_id == cid)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )).scalars().all()
    assert by_coalesce == by_created == ["m0", "m1", "m2", "m3"]


async def test_the_receipt_reads_the_ledger_and_the_answer_row(requires_message_tables):
    """`turn_receipt` (C2) is the durable half of the turn contract: the client
    asks "did MY message get answered?" and the answer comes from the
    exactly-once ledger plus the assistant row, not from a wall clock."""
    from app.db.database import async_session_maker
    from app.db.models import Message, ProcessedMessage
    from app.api import ws_chat

    cid = str(uuid.uuid4())
    cmid = "cm-" + uuid.uuid4().hex[:8]
    pm_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"toup-msg:{USER}:{cmid}"))

    # 1. Never seen here.
    r = await ws_chat._turn_receipt(USER, cmid)
    assert r["status"] == "unknown", r

    # 2. Claimed by the ledger, no answer row yet. NOT `failed` — that is also
    #    what a turn answered before this image shipped looks like.
    async with async_session_maker() as db:
        await _conversation(db, cid)
        db.add(ProcessedMessage(id=pm_id, user_id=USER, client_msg_id=cmid, session_id=cid))
        await db.commit()
    r = await ws_chat._turn_receipt(USER, cmid)
    assert r["status"] == "unknown" and r.get("code") == "no_answer_persisted", r

    # 3. The answer exists.
    async with async_session_maker() as db:
        db.add(Message(id="asst-1", conversation_id=cid, role="assistant",
                       content="Hello", client_msg_id=cmid))
        await db.commit()
    r = await ws_chat._turn_receipt(USER, cmid)
    assert r["status"] == "completed"
    assert r["message_id"] == "asst-1"
    assert r["client_msg_id"] == cmid


async def test_a_running_turn_beats_the_ledger(requires_message_tables):
    """The common case — the user reconnects while their turn is still running
    — must not depend on a DB that may be exactly what is broken."""
    import time
    from app.api import ws_chat

    cmid = "cm-" + uuid.uuid4().hex[:8]
    ws_chat._set_active_turn(USER, mission_id="chatturn:live", stage="thinking",
                             started_at=time.time(), client_msg_id=cmid)
    try:
        r = await ws_chat._turn_receipt(USER, cmid)
        assert r["status"] == "running"
        assert r["mission_id"] == "chatturn:live"
    finally:
        ws_chat._clear_active_turn(USER, "chatturn:live")


async def test_the_receipt_carries_no_content(requires_message_tables):
    """It answers a question about a turn; it is not a second delivery channel
    for the reply."""
    from app.db.database import async_session_maker
    from app.db.models import Message, ProcessedMessage
    from app.api import ws_chat

    cid = str(uuid.uuid4())
    cmid = "cm-" + uuid.uuid4().hex[:8]
    async with async_session_maker() as db:
        await _conversation(db, cid)
        db.add(ProcessedMessage(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"toup-msg:{USER}:{cmid}")),
            user_id=USER, client_msg_id=cmid, session_id=cid))
        db.add(Message(id="asst-2", conversation_id=cid, role="assistant",
                       content="a secret answer", client_msg_id=cmid))
        await db.commit()

    r = await ws_chat._turn_receipt(USER, cmid)
    assert "a secret answer" not in str(r)
    assert set(r) <= {"type", "client_msg_id", "status", "message_id", "mission_id", "code"}


async def test_the_answer_never_sorts_ABOVE_the_question(requires_message_tables):
    """`occurred_at` is the TURN's receipt time. Stamped on the assistant row
    as well as the user's, both rows of one turn carry the identical sort key
    and the day-chat `ORDER BY COALESCE(occurred_at, created_at), id` then
    tie-breaks on a random uuid — so the answer renders above the question
    about half the time, on every history read, for every turn. (Measured at
    48.9% over 2000 draws before the fix.) Executed against the REAL
    `_save_messages`, over enough id draws that a coin flip cannot pass."""
    import types

    from sqlalchemy import func, select
    from app.agent.agent_runner import AgentRunner
    from app.db.database import async_session_maker
    from app.db.models import Message

    # `_save_messages` touches `self` only for the tool sidecar state.
    fake_self = types.SimpleNamespace(
        tools=types.SimpleNamespace(
            _last_media=None, _last_pending_action=None, pending_attachments=[],
        )
    )
    received_at = 1_757_950_000.0

    for i in range(40):
        cid = str(uuid.uuid4())
        cmid = f"cm-{uuid.uuid4().hex[:12]}"
        async with async_session_maker() as db:
            await _conversation(db, cid)
            await db.commit()
        async with async_session_maker() as db:
            await AgentRunner._save_messages(
                fake_self, db, cid, USER,
                user_message="Hi", assistant_response="Hello",
                tokens_input=1, tokens_output=1, model="m", processing_time_ms=1,
                channel="mobile", client_msg_id=cmid, occurred_at=received_at,
            )
            await db.commit()

        async with async_session_maker() as db:
            rows = (await db.execute(
                select(Message)
                .where(Message.conversation_id == cid)
                .order_by(
                    func.coalesce(Message.occurred_at, Message.created_at).asc(),
                    Message.id.asc(),
                )
            )).scalars().all()
        assert [r.role for r in rows] == ["user", "assistant"], (
            f"draw {i}: the answer sorted above the question — {[r.id for r in rows]}"
        )
        # The identity itself is on BOTH rows; only the timestamp is the user's.
        assert all(r.client_msg_id == cmid for r in rows)
        assert rows[0].occurred_at is not None
        assert rows[1].occurred_at is None, (
            "the assistant row carries the turn's receipt time — that is the bug"
        )


async def test_the_receipt_is_scoped_to_the_caller(requires_message_tables):
    """`client_msg_id` comes straight off the wire and `messages` has no
    user_id column, so an unjoined lookup answers about ANY row in the
    database carrying the id. The preceding ProcessedMessage gate does not
    stop that: a caller mints their own gate by sending one message with the
    id they want to probe. Single-tenant in production; app/main.py mounts
    this same router against a shared database."""
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message, ProcessedMessage
    from app.api import ws_chat

    other = "00000000-0000-4000-8000-0000000000ff"
    their_cid = str(uuid.uuid4())
    cmid = "cm-" + uuid.uuid4().hex[:8]

    async with async_session_maker() as db:
        db.add(Conversation(id=their_cid, user_id=other, channel="mobile"))
        await db.flush()
        db.add(Message(id="their-answer", conversation_id=their_cid,
                       role="assistant", content="not yours", client_msg_id=cmid))
        # Our own ledger claim for the same id — the gate we can mint ourselves.
        db.add(ProcessedMessage(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"toup-msg:{USER}:{cmid}")),
            user_id=USER, client_msg_id=cmid, session_id=their_cid))
        await db.commit()

    r = await ws_chat._turn_receipt(USER, cmid)
    assert r.get("message_id") != "their-answer", r
    assert r["status"] == "unknown", r
