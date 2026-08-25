"""Row ↔ docker truth sync (R30, task_4ae52334 — pre-ramp blocker).

A blue-green re-randomises the host port and nothing wrote it back;
the stale row starved the running-tenant census and the canary gate
twice on 2026-08-25. This pins: the sync writes port/image/status
drift back to the row, keeps AgentConfig.agent_url in step, leaves
missing containers to the re-provision legs, and a dead bridge skips
instead of failing.
"""

import uuid
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import AgentConfig, ManagedContainer, User


async def _seed(port=9055, image_tag="oldtag", status="running"):
    uid = str(uuid.uuid4())
    name = f"toup-agent-{uid[:8]}"
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Sync"))
        db.add(ManagedContainer(
            user_id=uid, container_name=name, host_port=port,
            image_tag=image_tag, status=status, db_name=f"db_{uid[:8]}",
        ))
        db.add(AgentConfig(
            user_id=uid, agent_url=f"http://144.126.138.62:{port}",
            agent_api_key="k", deploy_status="active",
        ))
        await db.commit()
    return uid, name


def _bridge_stub(monkeypatch, containers):
    class _Resp:
        status_code = 200

        def json(self):
            return {"containers": containers}

    class _Client:
        async def get(self, path):
            assert path == "/v1/pool/tenant-truth"
            return _Resp()

    @asynccontextmanager
    async def _client(timeout_s=15):
        yield _Client()

    monkeypatch.setattr(
        "app.services.docker_host_service._bridge_client", _client,
    )


@pytest.mark.asyncio
async def test_port_image_status_drift_written_back(monkeypatch):
    from app.services.docker_host_service import reconcile_managed_rows
    uid, name = await _seed(port=9055, image_tag="oldtag")
    _bridge_stub(monkeypatch, [{
        "name": name, "image": "ghcr.io/toup-com/toup-agent:b1b9153c7695",
        "running": True, "status": "Up 2 hours", "host_port": 9123,
    }])
    async with async_session_maker() as db:
        out = await reconcile_managed_rows(db)
    assert out["fixed"] == 1, out
    async with async_session_maker() as db:
        row = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == uid)
        )).scalar_one()
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == uid)
        )).scalar_one()
        assert row.host_port == 9123
        assert row.image_tag == "b1b9153c7695"
        assert row.status == "running"
        assert cfg.agent_url.endswith(":9123")


@pytest.mark.asyncio
async def test_stopped_container_flips_row_and_missing_left_alone(
    monkeypatch,
):
    from app.services.docker_host_service import reconcile_managed_rows
    uid1, name1 = await _seed(port=9201)
    uid2, _name2 = await _seed(port=9202)
    _bridge_stub(monkeypatch, [{
        "name": name1, "image": "x:tag", "running": False,
        "status": "Exited (0)", "host_port": None,
    }])
    async with async_session_maker() as db:
        out = await reconcile_managed_rows(db)
    assert out["missing"] >= 1
    async with async_session_maker() as db:
        row1 = (await db.execute(
            select(ManagedContainer).where(
                ManagedContainer.user_id == uid1)
        )).scalar_one()
        row2 = (await db.execute(
            select(ManagedContainer).where(
                ManagedContainer.user_id == uid2)
        )).scalar_one()
        assert row1.status == "stopped"
        # Missing container: the row is untouched here (the
        # re-provision legs own it).
        assert row2.status == "running"


@pytest.mark.asyncio
async def test_dead_bridge_skips_never_fails(monkeypatch):
    from app.services.docker_host_service import reconcile_managed_rows

    @asynccontextmanager
    async def _client(timeout_s=15):
        raise ConnectionError("bridge down")
        yield  # pragma: no cover

    monkeypatch.setattr(
        "app.services.docker_host_service._bridge_client", _client,
    )
    async with async_session_maker() as db:
        out = await reconcile_managed_rows(db)
    assert "skipped" in out
