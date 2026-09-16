"""Attachment identity and the durable per-file status, against a real
Postgres.

Two properties need a real engine rather than sqlite:

  * `Message.attachments` is a JSON column, and the per-file `ingest` status
    the user and the model both read rides INSIDE it. A JSON round trip
    through asyncpg is not the same code path as sqlite's TEXT column, and the
    status is the whole point of this round's attachment work — a file that
    fails must still be legible in history.
  * Two simultaneous uploads of the same bytes must produce ONE identity. The
    guarantee is content-addressing plus an in-process lock (the container is
    the single writer for a tenant); this file exercises it while the same
    process is also talking to a real DB.

CI: named in the "Run pool / reclaim / proxy / pre-save suites on Postgres"
step of .github/workflows/test-backend.yml. Being named there also removes it
from the sqlite sweep, which has no asyncpg.

Local run:
    TEST_PG_URL='postgresql+asyncpg://postgres:test@127.0.0.1:5432/toup_presave' \
    TEST_PG_REQUIRED=1 PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_attachment_dedup_postgres.py -v --tb=short -p no:cacheprovider
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
import attachments as fx  # noqa: E402

PG_URL = os.environ.get("TEST_PG_URL", "postgresql+asyncpg://localhost/toup_lane_agent")
PG_REQUIRED = bool(os.environ.get("TEST_PG_REQUIRED"))

# context_budget_logs is included only so the drop below can run: it FKs
# onto conversations, and this database is shared with the rebucket suite.
_TABLES = ("users", "day_chats", "conversations", "messages", "context_budget_logs")


def _no_postgres(reason: str):
    if PG_REQUIRED:
        pytest.fail(f"TEST_PG_REQUIRED is set but {reason}", pytrace=False)
    pytest.skip(reason)


async def _engine():
    from sqlalchemy import text

    try:
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
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.db.models import User
    from app.db.models.base import Base

    eng = await _engine()
    tables = [Base.metadata.tables[n] for n in _TABLES]
    async with eng.begin() as c:
        await c.run_sync(lambda s: Base.metadata.drop_all(s, tables=tables[::-1], checkfirst=True))
        await c.run_sync(lambda s: Base.metadata.create_all(s, tables=tables, checkfirst=True))
    Session = async_sessionmaker(eng, expire_on_commit=False)
    uid = str(uuid.uuid4())
    async with Session() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@toup.ai", name="Attach", hashed_password="x"))
        await db.commit()
    return eng, Session, uid


@pytest.mark.asyncio
async def test_two_concurrent_uploads_of_one_file_make_one_attachment(tmp_path):
    from app.api import chat_attachments as CA

    eng, Session, uid = await _fresh_tenant()
    try:
        fx.local_storage(tmp_path)
        CA._inflight.clear()
        data = fx.jpeg_image(600, 400)
        a, b, c = await asyncio.gather(
            CA.ingest_and_store(data, "p.jpg", "image/jpeg", uid),
            CA.ingest_and_store(data, "p.jpg", "image/jpeg", uid),
            CA.ingest_and_store(data, "p.jpg", "image/jpeg", uid),
        )
        ids = {a["attachment_id"], b["attachment_id"], c["attachment_id"]}
        assert len(ids) == 1, "the same bytes produced more than one identity"
        assert [a["_deduped"], b["_deduped"], c["_deduped"]].count(False) == 1
    finally:
        from app.services import file_storage
        file_storage._backend = None
        await eng.dispose()


@pytest.mark.asyncio
async def test_the_per_file_ingest_status_survives_a_real_json_round_trip(tmp_path):
    """A password-protected PDF must still be legible in history: the user has
    to be able to see WHICH file the agent could not read, days later."""
    if not fx.have("pypdf"):
        pytest.skip("pypdf not installed here")
    from app.api import chat_attachments as CA
    from app.db.models import Conversation, Message
    from sqlalchemy import select

    eng, Session, uid = await _fresh_tenant()
    try:
        fx.local_storage(tmp_path)
        CA._inflight.clear()
        good = await CA.ingest_and_store(
            fx.long_text_pdf("PLAINTEXT ", 3000), "ok.pdf", "application/pdf", uid
        )
        locked = await CA.ingest_and_store(
            fx.encrypted_pdf(), "locked.pdf", "application/pdf", uid
        )
        assert locked["ingest"]["status"] == "password_protected"

        rows = []
        for rec in (good, locked):
            att = dict(rec["attachment"])
            att["ingest"] = rec["ingest"]
            rows.append(att)

        async with Session() as db:
            # naive UTC: the columns are TIMESTAMP WITHOUT TIME ZONE
            now = datetime.utcnow()
            db.add(Conversation(id="C1", user_id=uid, channel="mobile", started_at=now))
            await db.flush()
            db.add(Message(id="M1", conversation_id="C1", channel="mobile", role="user",
                           content="read these", attachments=rows, created_at=now))
            await db.commit()

        async with Session() as db:
            row = (await db.execute(select(Message).where(Message.id == "M1"))).scalar_one()
            back = row.attachments
        assert len(back) == 2
        assert back[0]["ingest"]["status"] in ("ok", "truncated")
        assert back[1]["ingest"]["status"] == "password_protected"
        assert back[1]["filename"] == "locked.pdf"
        # order is the order the user attached them
        assert back[0]["filename"] == "ok.pdf"
    finally:
        from app.services import file_storage
        file_storage._backend = None
        await eng.dispose()


@pytest.mark.asyncio
async def test_a_resend_of_the_same_turn_does_not_duplicate_the_attachments(tmp_path):
    """The warm-up loop re-sends the same frame with the same client_msg_id up
    to 15 times. With ids in the frame the resend costs one small JSON body and
    resolves to the SAME stored attachment — nothing is written twice."""
    from app.api import chat_attachments as CA

    eng, Session, uid = await _fresh_tenant()
    try:
        fx.local_storage(tmp_path)
        CA._inflight.clear()
        rec = await CA.ingest_and_store(fx.png_image(), "a.png", "image/png", uid)
        aid = rec["attachment_id"]
        for _ in range(15):
            turn = CA.build_turn_record(uid, aid)
            assert turn is not None and turn["attachment"]["id"] == rec["attachment"]["id"]
    finally:
        from app.services import file_storage
        file_storage._backend = None
        await eng.dispose()
