"""Regression locks for the 2026-06 new-user first-message fixes.

Covers:
  - Issue 4 (duplicate messages): the session-independent in-process dedup
    guard in ws_chat — drops a reconnect-replay of the FIRST message
    (session_id=null bypasses the DB idempotency block).
  - Issue 3 (greets "Agent Owner"): the soul-sync request now carries the
    owner's real name/email so a (re)provisioned container self-corrects.
"""
from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "test")


def test_dedup_guard_drops_replay_regardless_of_session():
    """First sight records + proceeds (False); an identical key (the reconnect
    replay) is reported as a duplicate (True); a distinct key proceeds."""
    from app.api.ws_chat import _dedup_seen_client_msg, _recent_client_msgs
    _recent_client_msgs.clear()
    key = "user-abc:cmid-123"
    assert _dedup_seen_client_msg(key) is False, "first dispatch must proceed"
    assert _dedup_seen_client_msg(key) is True, "replay must be flagged duplicate"
    assert _dedup_seen_client_msg("user-abc:cmid-456") is False, "distinct msg ok"
    # Distinct users with same client_msg_id must NOT collide.
    assert _dedup_seen_client_msg("user-xyz:cmid-123") is False


def test_dedup_guard_is_bounded():
    """The guard self-prunes so it can't grow unbounded under load."""
    from app.api import ws_chat
    ws_chat._recent_client_msgs.clear()
    # Stuff well past the cap with already-expired entries, then trigger a prune.
    import time as _t
    past = _t.monotonic() - 10_000
    for i in range(ws_chat._RECENT_MSG_MAX + 50):
        ws_chat._recent_client_msgs[f"k{i}"] = past
    # A fresh insert triggers the prune branch (len > cap) which drops expired.
    assert ws_chat._dedup_seen_client_msg("fresh-key") is False
    assert len(ws_chat._recent_client_msgs) < ws_chat._RECENT_MSG_MAX + 50


def test_soul_sync_request_carries_owner_identity():
    """The /api/soul/sync contract accepts owner_name/owner_email so the
    post-provision sync can correct the agent's local User row name."""
    from app.api.soul import SoulSyncRequest
    r = SoulSyncRequest(
        user_id="u1", name="Aria", compiled_text="...",
        owner_name="Nariman", owner_email="nariman@gmail.com",
    )
    assert r.owner_name == "Nariman"
    assert r.owner_email == "nariman@gmail.com"
    # Backward-compatible: older platform callers omit them.
    r2 = SoulSyncRequest(user_id="u1", name="Aria", compiled_text="...")
    assert r2.owner_name is None and r2.owner_email is None


# ── Bug B: durable exactly-once ledger (replay-loop credit burn) ──────────

def _derive_ledger_id(user_id: str, client_msg_id: str) -> str:
    """Mirror the derivation in ws_chat's ProcessedMessage claim."""
    import uuid as _uuid
    return str(_uuid.uuid5(_uuid.NAMESPACE_OID, f"toup-msg:{user_id}:{client_msg_id}"))


def test_processed_message_is_agent_only_partitioned():
    """The exactly-once ledger lives in the tenant (agent) DB and is created
    by init_db's create_all — NOT alembic — so recycled pool DBs provision it
    on boot. It must be registered on the metadata AND partitioned agent-only,
    or it would never be created on the agent containers that need it."""
    from app.db.models import ProcessedMessage
    from app.db.models.base import Base, AGENT_ONLY_TABLES
    assert ProcessedMessage.__tablename__ == "processed_messages"
    assert "processed_messages" in AGENT_ONLY_TABLES, (
        "must be agent-only so init_db create_all builds it in the tenant DB"
    )
    assert "processed_messages" in Base.metadata.tables, (
        "must be on Base.metadata or create_all can't build it"
    )
    # The primary key is the (user_id, client_msg_id) derivation — that is the
    # whole dedup mechanism.
    pk_cols = [c.name for c in ProcessedMessage.__table__.primary_key.columns]
    assert pk_cols == ["id"]


def test_processed_message_pk_collision_is_exactly_once():
    """A replay of the same (user_id, client_msg_id) derives the SAME primary
    key and collides on INSERT — so the second dispatch can never reach the
    LLM or the credit gate. This is the durable guarantee the in-process guard
    cannot give (it's lost when the container is swapped mid-boot). Distinct
    messages from the same user do NOT collide."""
    import asyncio
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.models import ProcessedMessage

    async def _run():
        # Self-contained engine (StaticPool → one shared in-memory DB across
        # connections) so this is independent of the platform-mode test schema
        # that excludes agent-only tables.
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:", poolclass=StaticPool
        )
        async with engine.begin() as conn:
            await conn.run_sync(ProcessedMessage.__table__.create)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        uid, cmid = "user-1", "cmsg-abc"
        pid = _derive_ledger_id(uid, cmid)

        async with Session() as s:  # first claim → wins
            s.add(ProcessedMessage(id=pid, user_id=uid, client_msg_id=cmid))
            await s.commit()

        collided = False
        async with Session() as s:  # replay → PK collision → dropped
            s.add(ProcessedMessage(id=pid, user_id=uid, client_msg_id=cmid))
            try:
                await s.commit()
            except IntegrityError:
                collided = True
                await s.rollback()

        # A genuinely different message from the same user is NOT blocked.
        other_id = _derive_ledger_id(uid, "cmsg-different")
        distinct_ok = False
        async with Session() as s:
            s.add(ProcessedMessage(id=other_id, user_id=uid, client_msg_id="cmsg-different"))
            await s.commit()
            distinct_ok = True

        await engine.dispose()
        return collided, distinct_ok, pid, other_id

    collided, distinct_ok, pid, other_id = asyncio.run(_run())
    assert collided is True, "replay must collide on the derived primary key"
    assert distinct_ok is True, "a distinct message must still be accepted"
    assert pid != other_id


def test_sync_soul_to_vps_payload_carries_owner_identity(monkeypatch):
    """The SENDER half of the owner-identity fix: the user-initiated soul save
    (which reaches the agent — the agent name updates) must ALSO carry
    owner_name/owner_email so the agent stops greeting "Hey Agent Owner",
    even when the bridge doesn't forward user_name into /admin/bind."""
    import asyncio
    import app.services.agent_http as agent_http
    from app.api import soul as soul_mod

    captured: dict = {}

    class _FakeResp:
        status_code = 200
        text = "ok"

    class _FakeClient:
        async def put(self, url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return _FakeResp()

    monkeypatch.setattr(agent_http, "get_agent_http_client", lambda: _FakeClient())

    ok = asyncio.run(soul_mod._sync_soul_to_vps(
        agent_url="https://agent", agent_api_key="k",
        user_id="u1", name="AHMAD", compiled_text="...",
        owner_name="NARIMAN", owner_email="nariman@gmail.com",
    ))
    assert ok is True
    assert captured["json"]["owner_name"] == "NARIMAN"
    assert captured["json"]["owner_email"] == "nariman@gmail.com"

    # Backward-compatible: omitting owner identity sends explicit nulls (the
    # receiver no-ops on them), never a KeyError.
    captured.clear()
    asyncio.run(soul_mod._sync_soul_to_vps(
        agent_url="https://agent", agent_api_key="k",
        user_id="u1", name="AHMAD", compiled_text="...",
    ))
    assert captured["json"]["owner_name"] is None
    assert captured["json"]["owner_email"] is None
