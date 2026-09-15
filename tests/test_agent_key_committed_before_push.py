"""The agent key is COMMITTED before the bridge pushes it (D-5).

The claim path minted `agent_api_key` with a compare-and-set on the CALLER's
transaction and then called the bridge — which reaches the agent's
`/api/admin/bind`, whose handler calls `ensure_mcp_initialized` SYNCHRONOUSLY,
firing `POST /api/mcp/mcp` immediately. `MCPAuthMiddleware` resolves
`X-Agent-Key` with a fresh, deliberately UNCACHED DB read
(`mcp_auth.py:192-207`, and its docstring says caching was not added on
purpose), sees an uncommitted row, and 401s.

Measured on 2026-09-12: key pushed 19:12:38.70, committed ~19:12:40.3, first
MCP request 19:12:40.142 — inside the window. `mcp_tools_cache` fell back to an
empty "stale" list and the agent ran with ZERO connector tools until 19:14:02,
82 s later. ~1.7 s per claim in the healthy case, and hit EVERY time, because
the bind handler fires the request immediately.

The fix is a SEPARATE short transaction, not a reordered commit. The claim path
holds `pg_advisory_xact_lock('pool_claim:<uid>')`, which is transaction-scoped:
committing the caller's transaction to make the key visible would release the
lock before the bridge call, and that lock is what makes two near-simultaneous
claims produce one bind rather than two containers holding different keys (the
2026-06-30 / 2026-07-01 double-bind incidents). A second session commits the
key while the first keeps the lock.

Run:
    cd backend && PYTHONPATH=. python -m pytest -q \
        tests/test_agent_key_committed_before_push.py
"""

from __future__ import annotations

import ast
import asyncio
import os
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("ENVIRONMENT", "test")


async def _seed_config(key=None):
    from app.db import async_session_maker
    from app.db.models import User, AgentConfig

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                    name="T", is_active=True))
        await db.flush()
        db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                           agent_api_key=key))
        await db.commit()
    return uid


async def _stored_key(uid):
    """Read through a FRESH session — the only read that proves a commit."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from sqlalchemy import select
    async with async_session_maker() as db:
        return (await db.execute(
            select(AgentConfig.agent_api_key).where(AgentConfig.user_id == uid)
        )).scalar_one_or_none()


@pytest.mark.asyncio
async def test_the_key_is_visible_to_another_session_immediately():
    """THE property. `MCPAuthMiddleware` opens its own session per request, so
    "visible to another session" is exactly what it needs and exactly what an
    uncommitted CAS did not give it."""
    from app.services.pool_service import ensure_agent_api_key

    uid = await _seed_config()
    key = await ensure_agent_api_key(uid)

    assert key
    assert await _stored_key(uid) == key, (
        "the key must be committed, not merely assigned"
    )


@pytest.mark.asyncio
async def test_it_is_idempotent_and_never_overwrites_an_existing_key():
    """A rotated key that this path re-minted would 401 every chat on a live
    tenant. The CAS is `WHERE agent_api_key IS NULL` for that reason."""
    from app.services.pool_service import ensure_agent_api_key

    uid = await _seed_config(key="already-set")
    assert await ensure_agent_api_key(uid) == "already-set"
    assert await ensure_agent_api_key(uid) == "already-set"
    assert await _stored_key(uid) == "already-set"


@pytest.mark.asyncio
async def test_concurrent_mints_agree_on_one_key():
    """Two replicas each reading NULL and minting a DIFFERENT random key is
    the 2026-06-30 shape: the platform keeps one while Caddy routes to the
    container bound with the other, and every chat 4001s. Every racer must
    read back the SAME winning value — which is why this re-reads rather than
    returning its own candidate."""
    from app.services.pool_service import ensure_agent_api_key

    uid = await _seed_config()
    keys = await asyncio.gather(*[ensure_agent_api_key(uid) for _ in range(5)])
    assert len(set(keys)) == 1, keys
    assert keys[0] == await _stored_key(uid)


@pytest.mark.asyncio
async def test_a_failed_mint_answers_none_so_the_caller_can_fall_back():
    """Any failure degrades to today's behaviour — an in-transaction CAS —
    rather than leaving the claim without a key."""
    from app.services import pool_service as ps

    uid = await _seed_config()

    def _boom(*a, **k):
        raise RuntimeError("pool exhausted")

    import app.db.database as dbmod
    real = dbmod.async_session_maker
    dbmod.async_session_maker = _boom
    try:
        assert await ps.ensure_agent_api_key(uid) is None
    finally:
        dbmod.async_session_maker = real


@pytest.mark.asyncio
async def test_build_bind_payload_uses_the_committed_key(monkeypatch):
    """End to end: the payload the bridge receives carries a key another
    session can already see."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from sqlalchemy import select
    from app.services.pool_service import _build_bind_payload

    uid = await _seed_config()
    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == uid)
        )).scalar_one()
        payload = await _build_bind_payload(db, uid, cfg)
        # Deliberately NOT committing `db` — the point is that the key is
        # already durable without it, which is the whole defect.

    assert payload["agent_api_key"]
    assert await _stored_key(uid) == payload["agent_api_key"], (
        "the bridge is about to push this key; MCPAuthMiddleware reads it from "
        "its own session milliseconds later"
    )


@pytest.mark.asyncio
async def test_the_fallback_still_produces_a_key(monkeypatch):
    """With the separate-session mint unavailable the payload must still carry
    a key — late-visible beats absent."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from sqlalchemy import select
    from app.services import pool_service as ps

    uid = await _seed_config()

    async def _unavailable(_uid):
        return None

    monkeypatch.setattr(ps, "ensure_agent_api_key", _unavailable)

    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == uid)
        )).scalar_one()
        payload = await ps._build_bind_payload(db, uid, cfg)
        assert payload["agent_api_key"]
        await db.commit()
    assert await _stored_key(uid) == payload["agent_api_key"]


# ── Structural: the advisory lock is NOT weakened ─────────────────────

def _pool_src() -> str:
    return (Path(__file__).resolve().parents[1]
            / "app" / "services" / "pool_service.py").read_text()


def test_the_claim_path_still_takes_the_per_user_advisory_lock():
    src = _pool_src()
    assert "pg_advisory_xact_lock(hashtext(:k)::bigint)" in src
    assert "_claim_drive_key(user_id)" in src


def test_the_mint_does_not_commit_the_callers_session():
    """The one way this change could be unsafe. `claim_for_user`'s transaction
    holds the xact-scoped advisory lock; a commit there releases it before the
    bridge call and re-opens the double-bind window."""
    src = _pool_src()
    body = src[src.index("async def ensure_agent_api_key"):
               src.index("async def _build_bind_payload")]
    tree = ast.parse(body.strip())
    fn = tree.body[0]
    # The only session it commits is the one it opened itself.
    commits = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and n.attr == "commit"
    ]
    assert commits, "it must commit something — that is the point"
    for c in commits:
        assert isinstance(c.value, ast.Name) and c.value.id == "kdb", (
            "ensure_agent_api_key may only commit its OWN session"
        )
    # …and it never takes the caller's session as an argument at all.
    args = [a.arg for a in fn.args.args]
    assert args == ["user_id"], args


def test_the_mint_bounds_its_lock_wait():
    """Deadlock-proof by construction rather than by inspection: every current
    caller reaches it with no uncommitted write on the row, and "no caller ever
    will" is not a property a future edit preserves."""
    src = _pool_src()
    body = src[src.index("async def ensure_agent_api_key"):
               src.index("async def _build_bind_payload")]
    assert "lock_timeout" in body
    assert "postgresql" in body, "the timeout must be dialect-guarded"


def test_the_cas_still_guards_on_null():
    src = _pool_src()
    assert src.count("AgentConfig.agent_api_key.is_(None)") >= 2, (
        "both the committed mint and the fallback must be idempotent"
    )
