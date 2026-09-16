"""A brand-new tenant's FIRST message was not persisted and was never acked.

Observed live on 2026-09-07 at 15:00:13.4Z on a throwaway signup
(scratchpad/e2e-findings.md, finding 2):

    [WS] Failed to pre-save user message: ForeignKeyViolation
         conversation_id … not present in "conversations"

`messages.conversation_id` is NOT NULL with an FK onto `conversations.id`, and
ws_chat's pre-save stamped it with the CLIENT's `session_id` while only ever
SELECTing that row — never creating it. The mobile client persists its last
session id in SecureStore and NEVER clears it (`sessionStore.ts`'s
`clearSessionId` has no callers, verified in the installed client), so the
first message after a signup, a re-install or an account switch carries an id
this tenant has no row for. The write failed, no `user_message_persisted` ack
went out, the client's ledger kept the message pending, and it re-sent on the
next reconnect — the `Duplicate dropped … replay/double-send` line in the same
run.

These run against a real Postgres (`TEST_PG_URL`, default
`postgresql+asyncpg://localhost/toup_lane_agent`); FK enforcement is the whole
subject, so sqlite would prove nothing. Skipped when no such database (or no
asyncpg driver) exists — EXCEPT under `TEST_PG_REQUIRED=1`, which the
`pytest-postgres` lane sets: there a skip would mean the one place this suite
is meant to run had quietly proven nothing, so it fails instead.
"""
from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

PG_URL = os.environ.get(
    "TEST_PG_URL", "postgresql+asyncpg://localhost/toup_lane_agent"
)
PG_REQUIRED = bool(os.environ.get("TEST_PG_REQUIRED"))


def _no_postgres(reason: str):
    if PG_REQUIRED:
        pytest.fail(f"TEST_PG_REQUIRED is set but {reason}", pytrace=False)
    pytest.skip(reason)


async def _engine():
    from sqlalchemy import text

    try:
        # Inside the try on purpose: create_async_engine imports the driver,
        # and the platform sweep (no asyncpg installed) died here with
        # ModuleNotFoundError instead of skipping — CI run 34146791609.
        from sqlalchemy.ext.asyncio import create_async_engine
        eng = create_async_engine(PG_URL)
    except Exception as e:  # noqa: BLE001
        _no_postgres(f"no asyncpg driver for {PG_URL}: {e}")
    try:
        async with eng.begin() as c:
            await c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as e:  # noqa: BLE001
        await eng.dispose()
        _no_postgres(f"no scratch Postgres at {PG_URL}: {e}")
    return eng


async def _fresh_tenant():
    """A tenant DB with the four tables the pre-save touches and one user —
    i.e. a container that has just been bound and has no conversations."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.db.models.base import Base
    from app.db.models import User

    eng = await _engine()
    names = ("users", "day_chats", "conversations", "messages")
    tables = [Base.metadata.tables[n] for n in names]
    async with eng.begin() as c:
        await c.run_sync(lambda s: Base.metadata.drop_all(s, tables=tables[::-1], checkfirst=True))
        await c.run_sync(lambda s: Base.metadata.create_all(s, tables=tables, checkfirst=True))
    Session = async_sessionmaker(eng, expire_on_commit=False)
    uid = str(uuid.uuid4())
    async with Session() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@toup.ai", name="E2E",
                    hashed_password="x"))
        await db.commit()
    return eng, Session, uid


# ── The defect, reproduced ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_bare_insert_the_presave_used_to_do_violates_the_fk():
    """Documents the mechanism. Passes before AND after the fix — it is the
    confirmation of the finding, not the falsifier."""
    from sqlalchemy.exc import IntegrityError
    from app.db.models import Message as DbMessage

    eng, Session, uid = await _fresh_tenant()
    try:
        async with Session() as db:
            db.add(DbMessage(id=str(uuid.uuid4()),
                             conversation_id=str(uuid.uuid4()),
                             role="user", content="hi"))
            with pytest.raises(IntegrityError) as ei:
                await db.commit()
        assert "ForeignKeyViolation" in str(ei.value)
    finally:
        await eng.dispose()


# ── The falsifier ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_stale_session_id_gets_a_conversation_and_the_message_persists():
    from app.api.ws_chat import _ensure_presave_conversation
    from app.db.models import Message as DbMessage, Conversation
    from sqlalchemy import select

    eng, Session, uid = await _fresh_tenant()
    session_id = str(uuid.uuid4())  # the SecureStore leftover
    try:
        async with Session() as db:
            conv = await _ensure_presave_conversation(db, uid, session_id, "mobile")
            assert conv is not None and conv.id == session_id
            db.add(DbMessage(id=str(uuid.uuid4()), conversation_id=session_id,
                             role="user", content="hi"))
            await db.commit()  # this is what raised ForeignKeyViolation before

        async with Session() as db:
            rows = (await db.execute(select(DbMessage))).scalars().all()
            assert len(rows) == 1 and rows[0].content == "hi"
            convs = (await db.execute(select(Conversation))).scalars().all()
            assert [c.id for c in convs] == [session_id]
            assert convs[0].user_id == uid and convs[0].channel == "mobile"
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_the_run_reuses_the_conversation_the_presave_created():
    """`id=session_id` is the whole point: the runner looks a session up by
    `(id, user_id)`, so it must find OUR row rather than mint a second one.
    Minting a second one would put the user's message and the agent's reply in
    different threads."""
    from app.agent.agent_runner import AgentRunner
    from app.api.ws_chat import _ensure_presave_conversation
    from app.db.models import Conversation
    from sqlalchemy import select

    eng, Session, uid = await _fresh_tenant()
    session_id = str(uuid.uuid4())
    try:
        for channel in ("mobile", "app", "web", "extension", None):
            async with Session() as db:
                sid = str(uuid.uuid4())
                made = await _ensure_presave_conversation(db, uid, sid, channel)
                await db.commit()
                found, created = await AgentRunner._get_or_create_session(
                    None, db, uid, sid, None, channel=channel,
                )
                assert created is False, f"{channel}: the run minted a NEW session"
                assert found.id == made.id == sid, f"{channel}: different row"

        async with Session() as db:
            convs = (await db.execute(select(Conversation))).scalars().all()
            assert len(convs) == 5, "the run created extra conversations"
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_an_existing_conversation_is_returned_untouched():
    from app.api.ws_chat import _ensure_presave_conversation
    from app.db.models import Conversation
    from sqlalchemy import select

    eng, Session, uid = await _fresh_tenant()
    session_id = str(uuid.uuid4())
    try:
        async with Session() as db:
            db.add(Conversation(id=session_id, user_id=uid, channel="web",
                                is_active=True, title="mine"))
            await db.commit()
        async with Session() as db:
            conv = await _ensure_presave_conversation(db, uid, session_id, "mobile")
            assert conv.title == "mine" and conv.channel == "web", \
                "an existing thread must not be rewritten by the pre-save"
            await db.commit()
        async with Session() as db:
            assert len((await db.execute(select(Conversation))).scalars().all()) == 1
    finally:
        await eng.dispose()


# ── The deferrals: cases the pre-save must NOT invent a row for ───────────

@pytest.mark.asyncio
async def test_the_bridge_app_shim_session_id_is_deferred_not_invented():
    """`apps_proxy` sends the literal `app-{app_id}`; the runner normalises it
    to None on purpose. Creating a row for it would fight that shim — and
    `conversations.id` is String(36) besides."""
    from app.api.ws_chat import _ensure_presave_conversation, _PresaveWithoutConversation

    eng, Session, uid = await _fresh_tenant()
    try:
        async with Session() as db:
            with pytest.raises(_PresaveWithoutConversation):
                await _ensure_presave_conversation(db, uid, "app-abc123", "app")
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_an_index_governed_channel_is_deferred_to_its_own_resolver():
    """`routine`/`trigger`/`api`/`digest`/`admin` rows are governed by the
    partial unique index `ix_conversations_system_channel_per_day` and have ONE
    writer, `resolve_or_create_day_conversation`. A blind insert here becomes a
    second writer."""
    from app.api.ws_chat import _ensure_presave_conversation, _PresaveWithoutConversation
    from app.agent.conversation_resolver import INDEXED_SYSTEM_CHANNELS

    eng, Session, uid = await _fresh_tenant()
    try:
        for ch in INDEXED_SYSTEM_CHANNELS:
            async with Session() as db:
                with pytest.raises(_PresaveWithoutConversation):
                    await _ensure_presave_conversation(
                        db, uid, str(uuid.uuid4()), ch,
                    )
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_a_tenant_with_no_user_row_defers_instead_of_raising_the_fk():
    """The other FK on `conversations` is onto `users`. A container bound
    milliseconds before the first message may not have the row yet; that is a
    deferral, not a crash."""
    from app.api.ws_chat import _ensure_presave_conversation, _PresaveWithoutConversation

    eng, Session, uid = await _fresh_tenant()
    try:
        async with Session() as db:
            with pytest.raises(_PresaveWithoutConversation) as ei:
                await _ensure_presave_conversation(
                    db, str(uuid.uuid4()), str(uuid.uuid4()), "mobile",
                )
        # It must RECOGNISE the unready tenant, not merely survive the write
        # blowing up: `resolve_day_chat_id_for_now` swallows its own flush
        # error, so by the time the INSERT fails the transaction is already
        # aborted and the log line an operator reads says nothing useful.
        assert "users row" in str(ei.value), str(ei.value)
    finally:
        await eng.dispose()


# ── The handler wiring ────────────────────────────────────────────────────

def test_the_handler_ensures_the_conversation_before_it_stamps_the_message():
    """Order is the fix: a SELECT after the message was staged proves nothing."""
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat.ws_chat)
    ensure_at = src.find("_ensure_presave_conversation(")
    stamp_at = src.find("conversation_id=session_id,\n                                day_chat_id=_presave_dc_id")
    assert ensure_at > 0, "the pre-save no longer ensures the conversation"
    assert stamp_at > ensure_at, "the message is stamped before the row exists"
    # The old shape — a bare lookup with no create — must be gone.
    assert "select(Conversation).where(Conversation.id == session_id)\n" \
        not in src[ensure_at:stamp_at + 2000]


def test_a_deferral_is_logged_as_expected_not_as_a_failed_write():
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat.ws_chat)
    at = src.find("except _PresaveWithoutConversation")
    assert at > 0, "the deferral is not caught separately"
    assert at < src.find("Failed to pre-save user message"), \
        "the generic handler shadows the deferral"


def test_the_retry_branch_re_establishes_the_conversation_after_its_rollback():
    """The `reply_to_message_id` fallback rolls back — which also undoes the
    conversation this pre-save flushed but had not committed. Without a
    re-ensure the retry hits the very FK this fix exists for."""
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat.ws_chat)
    # Round 46: the retry drops whichever of the three optional columns the
    # tenant is missing (reply_to_message_id / client_msg_id / occurred_at),
    # so the anchor is the pop LOOP rather than the single-column pop.
    at = src.find("for _c in _missing_cols:")
    assert at > 0
    assert "_msg_kwargs.pop(_c, None)" in src[at:at + 200]
    before = src[max(0, at - 900):at]
    assert "_presave_db.rollback()" in before
    assert "_ensure_presave_conversation(" in before
