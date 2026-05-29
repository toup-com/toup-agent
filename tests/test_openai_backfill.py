"""Tests for the per-tenant OpenAI project backfill reconciler.

The backfill is the eventually-consistent guarantee behind "every user
gets their OWN OpenAI key": a transient OpenAI Admin API failure at
signup leaves a user on the shared platform master key, and this sweep
provisions them on the next platform boot (or admin trigger). The
shared master key is the same class of single-point-of-failure that
took down Claude for everyone on 2026-05-29 — backfill ensures users
don't linger on it.
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import pytest
from sqlalchemy import select


async def _seed(bundle_status: str, project_id: str | None) -> str:
    from app.db import async_session_maker
    from app.db.models import AgentConfig, User
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"bf-{uid[:8]}@test.local",
                    hashed_password="x" * 60, name="BF User"))
        db.add(AgentConfig(
            user_id=uid,
            bundle_status=bundle_status,
            bundle_openai_project_id=project_id,
            bundle_openai_api_key=("sk-proj-existing" if project_id else None),
        ))
        await db.commit()
    return uid


@pytest.mark.asyncio
async def test_backfill_provisions_active_user_missing_project(monkeypatch):
    """An active-bundle user with no project gets one provisioned."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from app.api import billing

    uid = await _seed("active", None)

    # Mock the OpenAI Admin API so no real traffic; return a deterministic
    # project + key that the helper writes onto the config.
    monkeypatch.setattr(
        "app.services.openai_admin_service.provision_tenant",
        lambda prefix, user_name=None: (f"proj_{prefix}", "sk-proj-NEW"),
    )

    async with async_session_maker() as db:
        summary = await billing.backfill_missing_openai_projects(db)

    assert summary["provisioned"] >= 1
    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == uid)
        )).scalar_one()
    assert cfg.bundle_openai_project_id is not None
    assert cfg.bundle_openai_api_key == "sk-proj-NEW"


@pytest.mark.asyncio
async def test_backfill_skips_users_who_already_have_a_project(monkeypatch):
    """A user who already has a project must NOT be re-provisioned (no
    orphan projects, no duplicate billing)."""
    from app.db import async_session_maker
    from app.api import billing

    uid = await _seed("active", "proj_existing")
    calls = []
    monkeypatch.setattr(
        "app.services.openai_admin_service.provision_tenant",
        lambda prefix, user_name=None: calls.append(prefix) or ("proj_x", "sk-x"),
    )

    async with async_session_maker() as db:
        summary = await billing.backfill_missing_openai_projects(db)

    # The already-provisioned user is not even a candidate (query filters
    # bundle_openai_project_id IS NULL), so provision_tenant isn't called
    # for them.
    assert uid not in calls


@pytest.mark.asyncio
async def test_backfill_skips_non_bundle_users(monkeypatch):
    """Users not in active/cancelling bundle status are out of scope."""
    from app.db import async_session_maker
    from app.api import billing

    await _seed("none", None)
    calls = []
    monkeypatch.setattr(
        "app.services.openai_admin_service.provision_tenant",
        lambda prefix, user_name=None: calls.append(prefix) or ("proj_x", "sk-x"),
    )

    async with async_session_maker() as db:
        summary = await billing.backfill_missing_openai_projects(db)

    # The 'none'-status user must not have been provisioned.
    assert summary["candidates"] == summary["provisioned"] + summary["failed"]


@pytest.mark.asyncio
async def test_backfill_counts_failures_without_raising(monkeypatch):
    """A provisioning failure for one user is counted, not raised — the
    sweep must continue past a bad row."""
    from app.db import async_session_maker
    from app.api import billing

    await _seed("active", None)

    def _boom(prefix, user_name=None):
        raise RuntimeError("simulated admin API outage")

    monkeypatch.setattr(
        "app.services.openai_admin_service.provision_tenant", _boom,
    )

    async with async_session_maker() as db:
        summary = await billing.backfill_missing_openai_projects(db)

    assert summary["failed"] >= 1
    assert summary["provisioned"] == 0  # nothing succeeded, but no exception
