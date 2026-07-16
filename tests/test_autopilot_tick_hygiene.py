"""Autopilot ticks are engine internals, not user tasks.

Each ~5-min mission heartbeat mints a BuildJob (full job lifecycle for
retry/observability), but task surfaces must not show them — the
MISSION is the user-visible unit. Founder bug 2026-07-16: anonymous
'Routine fire: autopilot <date>' rows in Agent Tasks while the mission
itself appeared nowhere.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient


async def _mk_user() -> str:
    from app.db import User, async_session_maker
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"tick-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="Tick Test",
        ))
        await db.commit()
    return user_id


async def _mk_job(user_id: str, *, job_type: str, title: str) -> str:
    from app.db import async_session_maker
    from app.db.models import BuildJob

    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=user_id, title=title, prompt="p",
            job_type=job_type, status="completed",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return job_id


@pytest.mark.asyncio
async def test_jobs_listing_hides_autopilot_ticks(monkeypatch):
    from fastapi import FastAPI
    from app.config import settings
    from app.api.apps import router as apps_router

    user_id = await _mk_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    task_id = await _mk_job(
        user_id, job_type="agent_task", title="Top 10 CRM comparison",
    )
    tick_id = await _mk_job(
        user_id, job_type="autopilot_tick",
        title="Autopilot: UI/UX design tools research — tick 3",
    )

    app = FastAPI()
    app.include_router(apps_router, prefix="/api")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        default = await ac.get("/api/apps/jobs/")
        assert default.status_code == 200, default.text
        ids = [j["id"] for j in default.json()]
        assert task_id in ids
        assert tick_id not in ids, "ticks must be hidden by default"

        debug = await ac.get("/api/apps/jobs/?include_ticks=true")
        ids = [j["id"] for j in debug.json()]
        assert task_id in ids and tick_id in ids
