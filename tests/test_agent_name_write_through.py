"""agent_name write-through — the platform/tenant split, closed at the source.

The `agent_configs.agent_name` split (GA ledger, Last Mile L-1): the Soul
page wrote the platform copy; the tenant copy — the one text chat
(`agent_runner.py`) and the W-6 voice assembler read — was carried only by
two silent-failure paths:

  1. `_save_soul_impl` pushed the rename to the tenant via
     `_sync_soul_to_vps` and IGNORED the result: a dead or unreachable
     tenant meant a 200 to the user and a platform-only save. Silent
     drift, one rename at a time.
  2. `_sync_soul_after_start` — the ONLY carrier for names chosen during
     onboarding (before a container exists) — gave up after a 30s health
     gate. Heavy tenants boot in ~2 minutes, so the seed never landed.
     Same too-short-gate class as `agent_key_rotation_verify_timeout_s`
     (fixed at 180s in #596, which named this site as the unfixed twin).

These tests pin the fail-closed contract:
  * a bound-tenant rename that cannot reach the tenant DOES NOT succeed
    (no silent platform-only save);
  * the sync payload actually carries the name;
  * the receiver actually writes the tenant row, and REPORTS failure
    instead of warning-and-200;
  * the post-start seed gate clears a heavy boot.

Every failure-mode test here was proven RED against the pre-fix tree.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select


SOUL_BODY = {
    "name": "Renamed Agent",
    "color": "#A855F7",
    "pronouns": "they",
    "style": "casual",
    "traits": [],
    "custom_instructions": "",
    "defer_onboarding_complete": True,
}


@pytest_asyncio.fixture(autouse=True)
async def _soul_tables():
    """Create the AGENT_ONLY tables this file needs.

    `soul_configs` and `identities` are declared AGENT_ONLY
    (app/db/models/base.py), so the platform-mode sweep's `init_db` skips
    them — even though `app/api/soul.py`, which owns the Soul page, is
    mounted in `platform_main` and both reads and writes `soul_configs`.
    That mismatch is recorded as a finding; it is NOT re-litigated here,
    because changing a table's partition affects every lane.

    Creating them here — rather than parking this file in COVERAGE_DEBT —
    keeps these pins in the default sweep. Same pattern and rationale as
    `test_shared_day_context_invariants.day_tables`.

    Autouse and module-level, so it runs AFTER conftest's autouse database
    reset and the tables survive to the test.
    """
    from app.db.database import engine
    from app.db.models.base import Base

    async with engine.begin() as conn:
        for name in ("soul_configs", "identities"):
            await conn.run_sync(Base.metadata.tables[name].create, checkfirst=True)


@pytest_asyncio.fixture
async def soul_client() -> AsyncIterator:
    """Client with the /soul and /agent-setup routers mounted (the default
    `client` fixture doesn't include them — same shape as
    test_prewarm_phase1.soul_client)."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.auth import router as auth_router
    from app.api.agent_setup import router as agent_setup_router
    from app.api.soul import router as soul_router
    from app.config import settings

    app = FastAPI()
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(agent_setup_router, prefix=settings.api_prefix)
    app.include_router(soul_router, prefix=settings.api_prefix)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, app


@pytest_asyncio.fixture
async def bound_agent_config(test_user_id: str) -> str:
    """AgentConfig for a BOUND tenant: agent_url + agent_api_key present,
    so the soul save is required to write through to the tenant."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=test_user_id,
            hosting_mode="managed",
            agent_name="Before Rename",
            agent_color="#000000",
            agent_url="https://tenant.test.invalid",
            agent_api_key="test-tenant-key",
            bundle_status="none",
        ))
        await db.commit()
    return test_user_id


@pytest_asyncio.fixture
async def unbound_agent_config(test_user_id: str) -> str:
    """AgentConfig for an UNBOUND user: no container yet, nothing to write
    through to — the platform-only save is the correct outcome, and the
    tenant copy is seeded later by `_sync_soul_after_start`."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=test_user_id,
            hosting_mode="managed",
            agent_name=None,
            agent_url=None,
            agent_api_key=None,
            bundle_status="none",
        ))
        await db.commit()
    return test_user_id


@pytest_asyncio.fixture
async def receiver_key() -> AsyncIterator[str]:
    """Give the receiver endpoint a known X-Agent-Key for the test."""
    from app.config import settings
    prev = settings.agent_api_key
    settings.agent_api_key = "test-receiver-key"
    try:
        yield "test-receiver-key"
    finally:
        settings.agent_api_key = prev


# ── 1. Fail-closed: unreachable tenant ⇒ the rename does NOT succeed ──


@pytest.mark.asyncio
async def test_bound_rename_fails_closed_when_tenant_sync_fails(
    soul_client, auth_headers, bound_agent_config
):
    """MUTATION: revert soul.py to ignore the sync result → this returns
    200 and commits the platform copy → red. Proven red on the pre-fix
    tree (silent platform-only success was the live behavior)."""
    ac, _ = soul_client
    with patch(
        "app.api.soul._sync_soul_to_vps", new_callable=AsyncMock
    ) as mock_sync:
        mock_sync.return_value = False  # tenant unreachable / rejected
        resp = await ac.put("/api/soul", headers=auth_headers, json=SOUL_BODY)

    assert resp.status_code == 502, (
        f"expected fail-closed 502 when the tenant write fails, got "
        f"{resp.status_code}: a rename must never silently succeed "
        f"platform-only — that is the exact drift that blanked agent "
        f"names fleet-wide"
    )

    # And NOTHING was saved: neither the SoulConfig nor the platform
    # AgentConfig carries the new name.
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from app.db.models.soul_config import SoulConfig

    async with async_session_maker() as db:
        soul = (await db.execute(
            select(SoulConfig).where(SoulConfig.user_id == bound_agent_config)
        )).scalar_one_or_none()
        assert soul is None or soul.name != SOUL_BODY["name"], (
            "platform SoulConfig kept the failed rename"
        )
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == bound_agent_config)
        )).scalar_one()
        assert cfg.agent_name == "Before Rename", (
            "platform AgentConfig kept the failed rename"
        )


# ── 2. The sync payload carries the name (the write-through contract) ──


@pytest.mark.asyncio
async def test_bound_rename_carries_agent_name_to_tenant_sync(
    soul_client, auth_headers, bound_agent_config
):
    ac, _ = soul_client
    with patch(
        "app.api.soul._sync_soul_to_vps", new_callable=AsyncMock
    ) as mock_sync:
        mock_sync.return_value = True
        resp = await ac.put("/api/soul", headers=auth_headers, json=SOUL_BODY)

    assert resp.status_code == 200
    mock_sync.assert_called_once()
    updates = mock_sync.call_args.kwargs["agent_config_updates"]
    assert updates["agent_name"] == SOUL_BODY["name"]

    # Platform copies committed once the tenant write succeeded.
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from app.db.models.soul_config import SoulConfig

    async with async_session_maker() as db:
        soul = (await db.execute(
            select(SoulConfig).where(SoulConfig.user_id == bound_agent_config)
        )).scalar_one()
        assert soul.name == SOUL_BODY["name"]
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == bound_agent_config)
        )).scalar_one()
        assert cfg.agent_name == SOUL_BODY["name"]


# ── 3. Unbound user: platform-only save is correct, and succeeds ──


@pytest.mark.asyncio
async def test_unbound_save_is_platform_only(
    soul_client, auth_headers, unbound_agent_config
):
    ac, _ = soul_client
    with patch(
        "app.api.soul._sync_soul_to_vps", new_callable=AsyncMock
    ) as mock_sync:
        resp = await ac.put("/api/soul", headers=auth_headers, json=SOUL_BODY)

    assert resp.status_code == 200
    mock_sync.assert_not_called()

    from app.db import async_session_maker
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == unbound_agent_config)
        )).scalar_one()
        assert cfg.agent_name == SOUL_BODY["name"]


# ── 4. The post-start seed gate clears a heavy boot ──


def test_post_start_seed_gate_clears_a_heavy_boot():
    """The founder-class boot (WhatsApp sidecar + Telegram + workspace
    restore) takes ~2 minutes to answer /agent/health. A 30s gate meant
    the ONLY carrier of onboarding-chosen names never fired for exactly
    the tenants users care most about. Same class as
    agent_key_rotation_verify_timeout_s (#596). MUTATION: drop the
    setting or shrink it below 120 → red."""
    import inspect

    from app.config import settings
    from app.services import docker_host_service

    assert settings.soul_sync_health_timeout_s >= 120

    src = inspect.getsource(docker_host_service._sync_soul_after_start)
    assert "soul_sync_health_timeout_s" in src, (
        "_sync_soul_after_start does not honor the configured gate — "
        "a hardcoded poll count reintroduces the 30s ceiling"
    )


# ── 5. The receiver writes the tenant row ──


@pytest.mark.asyncio
async def test_sync_receiver_writes_agent_name_to_tenant_row(
    soul_client, test_user_id, receiver_key
):
    """agent_config_updates must land in the (tenant-side) agent_configs
    row — creating it when absent, as pool-era containers lack the row."""
    ac, _ = soul_client
    resp = await ac.put(
        "/api/soul/sync",
        headers={"X-Agent-Key": receiver_key},
        json={
            "user_id": test_user_id,
            "agent_config_updates": {
                "agent_name": "Zed",
                "agent_color": "#ffffff",
            },
        },
    )
    assert resp.status_code == 200

    from app.db import async_session_maker
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == test_user_id)
        )).scalar_one()
        assert cfg.agent_name == "Zed"


# ── 6. The receiver REPORTS an agent_config write failure ──


@pytest.mark.asyncio
async def test_sync_receiver_reports_agent_config_write_failure(
    soul_client, test_user_id, receiver_key
):
    """MUTATION: restore the warn-and-200 except branch → red. Proven red
    on the pre-fix tree (the receiver logged a warning and returned 200,
    so the sender's fail-closed check could never see the failure)."""
    from app.db import get_db
    from app.db import async_session_maker

    ac, app = soul_client

    class _BrokenNested:
        """Session proxy: begin_nested raises, everything else delegates."""

        def __init__(self, real):
            self._real = real

        def begin_nested(self):
            raise RuntimeError("forced agent_config write failure")

        def __getattr__(self, name):
            return getattr(self._real, name)

    async def _broken_db():
        async with async_session_maker() as real:
            yield _BrokenNested(real)

    app.dependency_overrides[get_db] = _broken_db
    try:
        resp = await ac.put(
            "/api/soul/sync",
            headers={"X-Agent-Key": receiver_key},
            json={
                "user_id": test_user_id,
                "agent_config_updates": {"agent_name": "Zed"},
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 502, (
        f"expected 502 when the tenant agent_config write fails, got "
        f"{resp.status_code} — a warn-and-200 receiver hides the failure "
        f"from the sender's fail-closed check"
    )

# ── 7. The SECOND platform-only writer: PUT /api/agent-setup/config ──


@pytest.mark.asyncio
async def test_agent_setup_config_rename_fails_closed_too(
    soul_client, auth_headers, bound_agent_config
):
    """agent_name is an accepted field of AgentConfigUpdate but was NOT in
    _BRIDGE_FIELDS and had no tenant write-through at all — a rename via
    the settings route drifted the copies silently. Same contract as the
    Soul save. MUTATION: drop the write-through in update_config → red.
    Proven red on the pre-fix tree."""
    ac, _ = soul_client
    with patch(
        "app.api.soul._sync_soul_to_vps", new_callable=AsyncMock
    ) as mock_sync:
        mock_sync.return_value = False
        resp = await ac.put(
            "/api/agent-setup/config",
            headers=auth_headers,
            json={"agent_name": "Settings Rename"},
        )
    assert resp.status_code == 502, (
        f"expected fail-closed 502, got {resp.status_code}"
    )

    from app.db import async_session_maker
    from app.db.models import AgentConfig

    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == bound_agent_config)
        )).scalar_one()
        assert cfg.agent_name == "Before Rename", (
            "platform AgentConfig kept the failed settings-route rename"
        )


@pytest.mark.asyncio
async def test_agent_setup_config_rename_writes_through_on_success(
    soul_client, auth_headers, bound_agent_config
):
    ac, _ = soul_client
    with patch(
        "app.api.soul._sync_soul_to_vps", new_callable=AsyncMock
    ) as mock_sync:
        mock_sync.return_value = True
        resp = await ac.put(
            "/api/agent-setup/config",
            headers=auth_headers,
            json={"agent_name": "Settings Rename"},
        )
    assert resp.status_code == 200
    mock_sync.assert_called_once()
    updates = mock_sync.call_args.kwargs["agent_config_updates"]
    assert updates == {"agent_name": "Settings Rename"}

# ── 8. The tenant-schema drift that silently ate the backfill ──


def test_tenant_agent_model_not_null_drift_is_reconciled():
    """Tenant agent_configs tables created from old snapshots have
    NOT NULL on agent_model while the ORM says nullable=True. The
    receiver's row-CREATE (identity fields only — it must never invent
    a model, R-6) hit NotNullViolationError on every tenant without a
    pre-existing row; warn-and-200 swallowed it; the L-1 backfill
    silently no-opped for exactly those tenants (live: 3134fece log,
    2026-08-12). The reconcile list must carry the DROP NOT NULL.
    MUTATION: remove the ALTER line → red."""
    import inspect

    from app.db import database

    src = inspect.getsource(database)
    assert (
        "ALTER TABLE agent_configs ALTER COLUMN agent_model DROP NOT NULL"
        in src
    ), (
        "the tenant migrator no longer drops the legacy NOT NULL on "
        "agent_configs.agent_model — row-creates for identity fields "
        "will fail on old tenant schemas"
    )

    from app.db.models.agent import AgentConfig
    assert AgentConfig.__table__.c.agent_model.nullable is True, (
        "ORM contract changed: agent_model is no longer nullable — "
        "revisit the reconcile ALTER and the receiver's row-CREATE"
    )


# ── 9. The retry window must actually cover a blue-green swap ──────────


@pytest.mark.asyncio
async def test_a_tenant_that_returns_within_the_drain_window_still_saves(
    soul_client, auth_headers, bound_agent_config
):
    """A blue-green swap makes the tenant refuse connections for a few
    seconds. `_sync_soul_to_vps` carries a 15s timeout but returns False
    in MILLISECONDS on a refused connection, so a fixed "one retry,
    sleep 2s" gave up about two seconds into an outage it was written to
    ride out — and the user's save 502'd for a tenant that was seconds
    from being back.

    Here the tenant refuses for the first four attempts and answers on
    the fifth. Attempt schedule is ~0s, 1s, 3s, 7s, 11s against a 12s
    deadline, so the save must survive.

    On the pre-fix tree this test errors rather than fails, because that
    version has no clock reference to patch at all — so the old
    behaviour is demonstrated inline below rather than merely asserted
    about.
    """
    # The retired schedule, reproduced: one attempt, one 2s sleep, one
    # more attempt. Against a tenant that returns at t≈4s it is already
    # out of attempts, which is the whole defect.
    old_attempt_times = [0.0, 2.0]
    assert max(old_attempt_times) < 4.0, (
        "the fixed two-attempt schedule gave up before a blue-green swap "
        "could plausibly finish — this is what the deadline replaces"
    )

    ac, _ = soul_client
    calls = {"n": 0}

    async def _refuse_then_answer(*_a, **_kw):
        calls["n"] += 1
        return calls["n"] >= 5  # first four "connection refused"

    # Real sleeps would make this a 12-second test; the schedule is what
    # matters, not the wall clock, so the deadline clock is advanced by
    # exactly what each sleep would have cost.
    clock = {"t": 0.0}

    async def _fake_sleep(seconds):
        clock["t"] += seconds

    with patch(
        "app.api.soul._sync_soul_to_vps", new=_refuse_then_answer
    ), patch(
        "app.api.soul.asyncio.sleep", new=_fake_sleep
    ), patch(
        "app.api.soul._time.monotonic", new=lambda: clock["t"]
    ):
        resp = await ac.put("/api/soul", headers=auth_headers, json=SOUL_BODY)

    assert calls["n"] == 5, (
        f"expected the retry loop to keep trying across the refusal "
        f"window (attempts at ~0s, 1s, 3s, 7s, 11s within a 12s "
        f"deadline); it made {calls['n']} attempt(s)"
    )
    assert resp.status_code == 200, (
        f"a tenant that came back inside the drain window still failed "
        f"the user's save: {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_a_slow_tenant_does_not_hold_the_transaction_open_forever(
    soul_client, auth_headers, bound_agent_config
):
    """The other half of the deadline: against a tenant that is merely
    SLOW (each attempt burning its full 15s timeout), the loop must stop
    after the deadline rather than multiplying 15s attempts. One retry
    at most, so the open transaction is bounded.

    MUTATION: swap the wall-clock deadline for an attempt COUNT of 5 →
    five 15s attempts → red.
    """
    ac, _ = soul_client
    calls = {"n": 0}
    clock = {"t": 0.0}

    async def _slow_failure(*_a, **_kw):
        calls["n"] += 1
        clock["t"] += 15.0  # each attempt burns the full client timeout
        return False

    async def _fake_sleep(seconds):
        clock["t"] += seconds

    with patch(
        "app.api.soul._sync_soul_to_vps", new=_slow_failure
    ), patch(
        "app.api.soul.asyncio.sleep", new=_fake_sleep
    ), patch(
        "app.api.soul._time.monotonic", new=lambda: clock["t"]
    ):
        resp = await ac.put("/api/soul", headers=auth_headers, json=SOUL_BODY)

    assert resp.status_code == 502
    assert calls["n"] == 1, (
        f"a slow tenant must not be retried past the deadline — the "
        f"transaction is open the whole time; made {calls['n']} attempts"
    )
