"""
T5c — Health-probe scheduler tests.

Smoke matrix:

  H1. run_once on empty registry → no work, returns 0/0.
  H2. ok probe bumps last_used_at + emits 'ok' metric.
  H3. fail probe doesn't flip status before threshold.
  H4. N consecutive failures → identity flips to provider_down.
  H5. ok after fail clears the failure counter (won't flip on later
      isolated failures).
  H6. concurrency cap honoured.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import ClassVar

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from sqlalchemy import select

from app.config import settings
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.db.models import ConnectorIdentity, User
from app.services import connector_metrics as _m
from app.services import connector_vault as vault
from app.services.connector_health_probe import HealthProbeScheduler
from app.services.connector_registry import (
    ConnectorEntry,
    ConnectorManifest,
    ConnectorTool,
    ChannelPolicy,
    HealthSpec,
    OAuthSpec,
    get_registry,
    reset_registry_for_tests,
)
from app.services.credential_crypto import _multi_fernet


class _ConfigurableProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "hstub"

    def __init__(self):
        self.probe_calls = 0
        self.results: list[HealthResult] = [HealthResult(ok=True)]
        self.delay_s = 0.0
        self.in_flight = 0
        self.max_in_flight = 0

    async def execute(self, tool_name, tool_input, ctx):
        return ConnectorOk(content="ok")

    async def revoke(self, user_id, access_token, refresh_token=None):
        return None

    async def refresh(self, refresh_token, *, scopes=None):
        return RefreshResult(
            access_token="a", refresh_token=refresh_token,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        self.probe_calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            idx = min(self.probe_calls - 1, len(self.results) - 1)
            return self.results[idx]
        finally:
            self.in_flight -= 1


def _install_hstub() -> _ConfigurableProvider:
    reset_registry_for_tests()
    manifest = ConnectorManifest(
        manifest_version=1,
        id="hstub",
        name="H Stub",
        short_description="t5c harness",
        status="experimental",
        category="test",
        oauth=OAuthSpec(
            provider_app="stub_provider_app", scopes=[], pkce=True, refresh=True,
        ),
        health=HealthSpec(probe="hstub__do"),
        tools=[
            ConnectorTool(
                name="hstub__do",
                description="x",
                input_schema={"type": "object", "properties": {}},
                mutates=False,
                elevation=False,
                output_redaction=[],
                channel_policy=ChannelPolicy(default="allow", deny=[]),
            )
        ],
    )
    provider = _ConfigurableProvider()
    reg = get_registry()
    reg._entries["hstub"] = ConnectorEntry(manifest=manifest, provider=provider)
    for tool in manifest.tools:
        reg._tool_index[tool.name] = manifest.id
    return provider


@pytest.fixture(autouse=True)
def _crypto():
    prev = settings.platform_encryption_key
    prev_prev = settings.platform_encryption_key_previous
    settings.platform_encryption_key = Fernet.generate_key().decode()
    settings.platform_encryption_key_previous = ""
    _multi_fernet.cache_clear()
    try:
        yield
    finally:
        settings.platform_encryption_key = prev
        settings.platform_encryption_key_previous = prev_prev
        _multi_fernet.cache_clear()


@pytest.fixture(autouse=True)
def _reset_metrics():
    _m.reset_for_tests()
    yield


async def _seed_user_with_identity() -> str:
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="x"))
        await db.commit()
    async with async_session_maker() as db:
        await vault.put(
            db, uid, "hstub",
            access_token="seed",
            refresh_token="rt",
            access_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
    return uid


# ─── H1: empty registry ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_h1_empty_registry_no_work():
    reset_registry_for_tests()
    sched = HealthProbeScheduler()
    summary = await sched.run_once()
    assert summary["sweeps"] == 0


# ─── H2: ok probe bumps last_used_at ────────────────────────────────────


@pytest.mark.asyncio
async def test_h2_ok_probe_bumps_last_used_at_and_emits_metric():
    provider = _install_hstub()
    uid = await _seed_user_with_identity()

    sched = HealthProbeScheduler()
    summary = await sched.run_once()
    assert summary["ok"] == 1
    assert summary["fail"] == 0
    assert provider.probe_calls == 1

    # Identity unchanged status, last_used_at bumped.
    async with async_session_maker() as db:
        ident = (await db.execute(
            select(ConnectorIdentity).where(ConnectorIdentity.user_id == uid)
        )).scalar_one()
    assert ident.status == "active"
    assert ident.last_used_at is not None

    # Metric counter incremented.
    rendered = _m.render()
    assert "connector_health_probes_total" in rendered
    assert 'outcome="ok"' in rendered


# ─── H3 + H4: failure threshold flip ────────────────────────────────────


@pytest.mark.asyncio
async def test_h3_h4_failure_threshold_flips_to_provider_down():
    provider = _install_hstub()
    provider.results = [HealthResult(ok=False, detail="boom")]  # always fail
    uid = await _seed_user_with_identity()

    sched = HealthProbeScheduler(failure_threshold=3)

    # Sweep 1 — fail count = 1, status still active.
    await sched.run_once()
    async with async_session_maker() as db:
        ident = (await db.execute(
            select(ConnectorIdentity).where(ConnectorIdentity.user_id == uid)
        )).scalar_one()
    assert ident.status == "active"

    # Sweep 2 — fail count = 2, status still active.
    await sched.run_once()
    async with async_session_maker() as db:
        ident = (await db.execute(
            select(ConnectorIdentity).where(ConnectorIdentity.user_id == uid)
        )).scalar_one()
    assert ident.status == "active"

    # Sweep 3 — fail count = 3, threshold reached → provider_down.
    await sched.run_once()
    # The flip is fire-and-forget — let the background task settle.
    await asyncio.sleep(0.1)
    async with async_session_maker() as db:
        ident = (await db.execute(
            select(ConnectorIdentity).where(ConnectorIdentity.user_id == uid)
        )).scalar_one()
    assert ident.status == "provider_down"


# ─── H5: ok-after-fail resets the counter ──────────────────────────────


@pytest.mark.asyncio
async def test_h5_ok_after_fail_resets_counter():
    """fail, fail, ok, fail, fail → counter reset by ok, no flip."""
    provider = _install_hstub()
    provider.results = [
        HealthResult(ok=False, detail="1"),
        HealthResult(ok=False, detail="2"),
        HealthResult(ok=True),
        HealthResult(ok=False, detail="4"),
        HealthResult(ok=False, detail="5"),
    ]
    uid = await _seed_user_with_identity()

    sched = HealthProbeScheduler(failure_threshold=3)
    for _ in range(5):
        await sched.run_once()
    await asyncio.sleep(0.1)

    async with async_session_maker() as db:
        ident = (await db.execute(
            select(ConnectorIdentity).where(ConnectorIdentity.user_id == uid)
        )).scalar_one()
    # 2 failures, then ok (reset), then 2 more failures — never hit 3.
    assert ident.status == "active"


# ─── H6: concurrency cap ───────────────────────────────────────────────


def _is_sqlite() -> bool:
    import os
    return os.environ.get("DATABASE_URL", "").startswith("sqlite")


@pytest.mark.asyncio
async def test_h6_per_connector_concurrency_cap_honoured():
    if _is_sqlite():
        # User.is_canary partial unique index trips the second seed
        # under SQLite — same trap as test_mcp_auth's db_tenants
        # fixture. Postgres CI runs this; skip locally.
        pytest.skip("requires Postgres (User.is_canary partial unique index)")
    provider = _install_hstub()
    provider.delay_s = 0.05  # so probes overlap
    # Seed 10 users (10 identities for the same connector).
    uids = []
    for _ in range(10):
        uids.append(await _seed_user_with_identity())

    sched = HealthProbeScheduler(connector_concurrency=3)
    await sched.run_once()
    await asyncio.sleep(0.1)

    assert provider.probe_calls == 10
    assert provider.max_in_flight <= 3, (
        f"concurrency cap violated: max_in_flight={provider.max_in_flight}"
    )
