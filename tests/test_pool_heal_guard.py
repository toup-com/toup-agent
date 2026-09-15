"""Containment pins for the September 12 pool/named split incident."""

import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("ENVIRONMENT", "test")


def test_healthy_claim_is_observed_without_mutation():
    from app.services import pool_service as ps

    with patch(
        "app.services.prewarm_service._is_agent_actually_healthy",
        new=AsyncMock(return_value=True),
    ) as health, patch(
        "app.services.docker_host_service.restart_container", new=AsyncMock()
    ) as restart, patch(
        "app.services.docker_host_service.provision_container", new=AsyncMock()
    ) as provision:
        asyncio.run(
            ps._verify_and_heal_pool_claim(
                "u1234567", budget_s=0.1, interval_s=0.01
            )
        )

    assert health.await_count == 1
    restart.assert_not_awaited()
    provision.assert_not_awaited()


def test_readiness_timeout_never_restarts_or_cold_swaps():
    """Slow DNS/TLS/boot is uncertainty, even for a seconds-old signup."""
    from app.services import pool_service as ps

    with patch(
        "app.services.prewarm_service._is_agent_actually_healthy",
        new=AsyncMock(return_value=False),
    ), patch(
        "app.services.docker_host_service.restart_container", new=AsyncMock()
    ) as restart, patch(
        "app.services.docker_host_service.provision_container", new=AsyncMock()
    ) as provision:
        asyncio.run(
            ps._verify_and_heal_pool_claim(
                "u1234567", budget_s=0.01, interval_s=0.02
            )
        )

    restart.assert_not_awaited()
    provision.assert_not_awaited()


def test_late_health_within_observation_window_still_never_mutates():
    from app.services import pool_service as ps

    with patch(
        "app.services.prewarm_service._is_agent_actually_healthy",
        new=AsyncMock(side_effect=[False, True]),
    ) as health, patch(
        "app.services.docker_host_service.restart_container", new=AsyncMock()
    ) as restart, patch(
        "app.services.docker_host_service.provision_container", new=AsyncMock()
    ) as provision:
        asyncio.run(
            ps._verify_and_heal_pool_claim(
                "u1234567", budget_s=0.1, interval_s=0
            )
        )

    assert health.await_count == 2
    restart.assert_not_awaited()
    provision.assert_not_awaited()


def test_guard_never_raises():
    from app.services import pool_service as ps

    with patch(
        "app.services.prewarm_service._is_agent_actually_healthy",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        asyncio.run(
            ps._verify_and_heal_pool_claim(
                "u1234567", budget_s=0.1, interval_s=0
            )
        )


# ── D-2's PRECONDITION, not just its door (L10 §5c) ───────────────────
#
# Wave 1 made `provision_container` refuse a pool→named swap for any
# `recreate`. That closes the door. But a pool row only ever REACHES the
# named path because something first knocked it out of running/provisioning,
# and there are exactly two realistic writers that do: `container_monitor`
# marking an unhealthy container `error`, and `stop_container` marking it
# `stopped`. Neither had a pool exemption. A pool member is repaired IN PLACE
# (the bridge re-applies its persisted bind and route after a restart), so
# neither status was ever the right word for one.

import uuid as _uuid

import pytest as _pytest


@_pytest.mark.asyncio
async def test_stop_container_refuses_to_park_a_pool_member():
    from app.db import async_session_maker
    from app.db.models import User, ManagedContainer
    from app.services.docker_host_service import stop_container

    uid = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                    name="T"))
        await db.flush()
        db.add(ManagedContainer(
            id=str(_uuid.uuid4()), user_id=uid,
            container_name="toup-agent-pool-42", host_port=9411,
            db_name="pool_42", status="running",
        ))
        await db.commit()

    async with async_session_maker() as db:
        row = await stop_container(db, uid)
    assert row is not None
    assert row.status == "running", (
        "a stopped pool row is the state that walks the next provision into "
        "the named path and onto an empty database"
    )


@_pytest.mark.asyncio
async def test_stop_container_still_stops_a_named_tenant():
    """The exemption is for pool members only — a named tenant's container IS
    the user's, and stopping it must keep working."""
    from app.db import async_session_maker
    from app.db.models import User, ManagedContainer
    from app.services.docker_host_service import stop_container

    uid = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                    name="T"))
        await db.flush()
        db.add(ManagedContainer(
            id=str(_uuid.uuid4()), user_id=uid,
            container_name=f"toup-agent-{uid[:8]}", host_port=9412,
            db_name=f"toup_agent_{uid[:8]}", status="running",
        ))
        await db.commit()

    async with async_session_maker() as db:
        row = await stop_container(db, uid)
    assert row is not None and row.status == "stopped"


def test_container_monitor_never_writes_error_on_a_pool_row():
    """Source probe: the exemption must sit ABOVE the `status = "error"`
    write, not beside it. (This loop no-ops in production for want of
    DOCKER_HOST_IP, which makes the fix cheaper now, not less important.)"""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "container_monitor.py").read_text()
    guard = src.index('if (container.container_name or "").startswith(')
    write = src.index('container.status = "error"', guard)
    assert guard < write
    # …and the guard's branch leaves the loop before that write.
    assert "continue" in src[guard:write]
