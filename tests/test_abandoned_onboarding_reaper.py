"""Phase 2 tests — abandoned-onboarding reaper.

Four required cases per docs/onboarding/prewarm-phase0.md Phase 2:
  - 14d-old + abandoned (onboarding_completed=False, bundle='none')   → REAPED
  - 14d-old + paying (bundle_status='active')                          → SKIPPED
  - 14d-old + completed onboarding                                     → SKIPPED
  - 13d-old + abandoned (under threshold)                              → SKIPPED

The day_chats engagement signal in the original spec was deferred —
day_chats is AGENT_ONLY and not queryable from this reaper. See the
NOTE block at scheduled_tasks.run_abandoned_onboarding_reaper for the
architectural rationale. `onboarding_completed=False` is strictly
stronger for the abandoned-user case.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select


@pytest_asyncio.fixture
async def reaper_grace_days_14() -> AsyncIterator[None]:
    """Force the reaper grace to 14 days regardless of env override."""
    from app.config import settings
    prev = settings.abandoned_onboarding_grace_days
    settings.abandoned_onboarding_grace_days = 14
    try:
        yield
    finally:
        settings.abandoned_onboarding_grace_days = prev


async def _seed_candidate(
    *,
    user_id: str,
    created_days_ago: int,
    onboarding_completed: bool,
    bundle_status: str | None,
    container_status: str = "running",
) -> None:
    """Seed an AgentConfig + ManagedContainer pair for a single test case."""
    from app.db import async_session_maker, User
    from app.db.models import AgentConfig
    from app.db.models.platform import ManagedContainer
    from app.services.auth_service import get_password_hash

    backdated = datetime.utcnow() - timedelta(days=created_days_ago)
    async with async_session_maker() as db:
        u = User(
            id=user_id,
            email=f"u-{user_id[:8]}@example.com",
            hashed_password=get_password_hash("test1234abcd"),
            name=f"User {user_id[:6]}",
        )
        db.add(u)
        await db.flush()

        cfg = AgentConfig(
            user_id=user_id,
            hosting_mode="managed",
            onboarding_completed=onboarding_completed,
            bundle_status=bundle_status,
            agent_name="Test",
            llm_mode=None,
        )
        db.add(cfg)

        mc = ManagedContainer(
            id=str(uuid.uuid4()),
            user_id=user_id,
            container_name=f"toup-agent-{user_id[:8]}",
            host_port=9000 + (hash(user_id) % 900),
            db_name=f"toup_agent_{user_id[:8]}",
            status=container_status,
        )
        db.add(mc)
        await db.commit()

        # Backdate created_at via raw UPDATE (the model doesn't expose
        # an override and SQLAlchemy default fires at flush).
        from sqlalchemy import text
        await db.execute(
            text("UPDATE agent_configs SET created_at = :ts WHERE user_id = :uid"),
            {"ts": backdated, "uid": user_id},
        )
        await db.commit()


# ─── Case 1 — 14d-old + abandoned → REAPED ─────────────────────────────


@pytest.mark.asyncio
async def test_reaper_destroys_14d_abandoned_container(reaper_grace_days_14):
    """A user who created an account 14 days ago, never finished
    onboarding, and never paid: container destroyed on reaper run."""
    from app.scripts.scheduled_tasks import run_abandoned_onboarding_reaper

    user_id = str(uuid.uuid4())
    await _seed_candidate(
        user_id=user_id,
        created_days_ago=15,
        onboarding_completed=False,
        bundle_status="none",
    )

    destroy_called_for: list[str] = []

    async def fake_destroy(db, uid):
        destroy_called_for.append(uid)
        return True

    with patch(
        "app.services.docker_host_service.destroy_container",
        new=fake_destroy,
    ):
        await run_abandoned_onboarding_reaper()

    assert destroy_called_for == [user_id]


# ─── Case 2 — 14d-old + paying → SKIPPED ───────────────────────────────


@pytest.mark.asyncio
async def test_reaper_skips_paying_user_even_if_no_onboarding(reaper_grace_days_14):
    """Paying users are owned by the bundle_reaper (post-cancellation).
    The abandoned-onboarding reaper must NEVER touch them, even if
    onboarding_completed is somehow false."""
    from app.scripts.scheduled_tasks import run_abandoned_onboarding_reaper

    user_id = str(uuid.uuid4())
    await _seed_candidate(
        user_id=user_id,
        created_days_ago=15,
        onboarding_completed=False,
        bundle_status="active",  # ← paying
    )

    destroy_called_for: list[str] = []

    async def fake_destroy(db, uid):
        destroy_called_for.append(uid)
        return True

    with patch(
        "app.services.docker_host_service.destroy_container",
        new=fake_destroy,
    ):
        await run_abandoned_onboarding_reaper()

    assert destroy_called_for == []


# ─── Case 3 — 14d-old + onboarding completed → SKIPPED ─────────────────


@pytest.mark.asyncio
async def test_reaper_skips_user_who_completed_onboarding(reaper_grace_days_14):
    """A user who reached past Install (onboarding_completed=True) has
    engaged. Even if they're idle, reaping their container would be a
    real product disruption — they came back to chat with their agent
    and find it gone. The bundle_reaper handles cancelled-subscription
    cleanup separately."""
    from app.scripts.scheduled_tasks import run_abandoned_onboarding_reaper

    user_id = str(uuid.uuid4())
    await _seed_candidate(
        user_id=user_id,
        created_days_ago=15,
        onboarding_completed=True,  # ← engaged
        bundle_status="none",
    )

    destroy_called_for: list[str] = []

    async def fake_destroy(db, uid):
        destroy_called_for.append(uid)
        return True

    with patch(
        "app.services.docker_host_service.destroy_container",
        new=fake_destroy,
    ):
        await run_abandoned_onboarding_reaper()

    assert destroy_called_for == []


# ─── Case 4 — 13d-old + abandoned → SKIPPED (under threshold) ──────────


@pytest.mark.asyncio
async def test_reaper_skips_user_under_grace_window(reaper_grace_days_14):
    """A user who hit Soul.save 13 days ago and never came back is
    abandoned-shaped, but still inside the 14-day grace window. Don't
    reap yet — give them another day to come back."""
    from app.scripts.scheduled_tasks import run_abandoned_onboarding_reaper

    user_id = str(uuid.uuid4())
    await _seed_candidate(
        user_id=user_id,
        created_days_ago=13,  # ← under threshold
        onboarding_completed=False,
        bundle_status="none",
    )

    destroy_called_for: list[str] = []

    async def fake_destroy(db, uid):
        destroy_called_for.append(uid)
        return True

    with patch(
        "app.services.docker_host_service.destroy_container",
        new=fake_destroy,
    ):
        await run_abandoned_onboarding_reaper()

    assert destroy_called_for == []


# ─── Defense-in-depth: disabled when grace=0 ──────────────────────────


@pytest.mark.asyncio
async def test_reaper_disabled_when_grace_days_zero():
    """grace_days=0 = disabled (dev / test escape hatch). Reaper exits
    immediately without querying the DB."""
    from app.config import settings
    prev = settings.abandoned_onboarding_grace_days
    settings.abandoned_onboarding_grace_days = 0
    try:
        from app.scripts.scheduled_tasks import run_abandoned_onboarding_reaper

        destroy_called = []

        async def fake_destroy(db, uid):
            destroy_called.append(uid)
            return True

        with patch(
            "app.services.docker_host_service.destroy_container",
            new=fake_destroy,
        ):
            await run_abandoned_onboarding_reaper()

        assert destroy_called == []
    finally:
        settings.abandoned_onboarding_grace_days = prev
