"""
Tests for `docker_host_service` — specifically the image-tag resolution
helper used during first-time tenant provisioning.

Background: 2026-04-26 incident — a first-time bundle subscriber's
provision_container call hit the bridge with `toup-agent:latest` (the
fresh-install sentinel from settings.docker_agent_image), which 404'd
because the rollout pipeline only publishes SHA-tagged images. Fix:
resolve the image_tag from the latest successfully-completed Rollout row
at provision time, falling back to the sentinel only on a fresh platform
install with zero rollout history.

These tests pin that resolution: the helper must (1) return the most
recent `complete` rollout's tag, (2) ignore `aborted_canary_failed` /
`cancelled` / in-flight rows even when more recent, (3) return None on
empty so the caller's fallback fires.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.db.database import async_session_maker
from app.db.models import Rollout
from app.services.docker_host_service import _latest_known_good_image_tag


def _rollout_row(image_tag: str, status: str, completed_at: datetime | None) -> Rollout:
    return Rollout(
        image_tag=image_tag,
        status=status,
        trigger="ci",
        canary_wait_minutes=10,
        started_at=(completed_at or datetime.utcnow()) - timedelta(minutes=5),
        completed_at=completed_at,
    )


@pytest.mark.asyncio
async def test_latest_known_good_returns_most_recent_complete():
    """Two `complete` rollouts → helper picks the one with later completed_at."""
    older = datetime.utcnow() - timedelta(days=2)
    newer = datetime.utcnow() - timedelta(hours=1)

    async with async_session_maker() as db:
        db.add(_rollout_row("ghcr.io/toup-com/toup-agent:older", "complete", older))
        db.add(_rollout_row("ghcr.io/toup-com/toup-agent:newer", "complete", newer))
        await db.commit()

        tag = await _latest_known_good_image_tag(db)

    assert tag == "ghcr.io/toup-com/toup-agent:newer"


@pytest.mark.asyncio
async def test_latest_known_good_ignores_aborted_canary_failed():
    """A more-recent `aborted_canary_failed` rollout must not shadow the older `complete`."""
    older_complete = datetime.utcnow() - timedelta(days=2)
    newer_aborted = datetime.utcnow() - timedelta(hours=1)

    async with async_session_maker() as db:
        db.add(_rollout_row("ghcr.io/toup-com/toup-agent:older_good", "complete", older_complete))
        db.add(_rollout_row("ghcr.io/toup-com/toup-agent:newer_bad", "aborted_canary_failed", newer_aborted))
        await db.commit()

        tag = await _latest_known_good_image_tag(db)

    assert tag == "ghcr.io/toup-com/toup-agent:older_good"


@pytest.mark.asyncio
async def test_latest_known_good_ignores_cancelled_and_in_flight():
    """`cancelled`, `pending`, and `running` rollouts must not be selected even if newest."""
    base = datetime.utcnow()

    async with async_session_maker() as db:
        db.add(_rollout_row("ghcr.io/toup-com/toup-agent:complete_old", "complete", base - timedelta(days=3)))
        db.add(_rollout_row("ghcr.io/toup-com/toup-agent:cancelled_recent", "cancelled", base - timedelta(hours=2)))
        # in-flight rollouts have no completed_at
        db.add(_rollout_row("ghcr.io/toup-com/toup-agent:pending_recent", "pending", None))
        db.add(_rollout_row("ghcr.io/toup-com/toup-agent:running_recent", "running", None))
        await db.commit()

        tag = await _latest_known_good_image_tag(db)

    assert tag == "ghcr.io/toup-com/toup-agent:complete_old"


@pytest.mark.asyncio
async def test_latest_known_good_returns_none_when_no_rollouts():
    """Fresh platform install — caller falls back to settings.docker_agent_image."""
    async with async_session_maker() as db:
        tag = await _latest_known_good_image_tag(db)

    assert tag is None


@pytest.mark.asyncio
async def test_latest_known_good_returns_none_when_no_complete_rollouts():
    """All rollouts failed canary → helper returns None; caller falls back."""
    async with async_session_maker() as db:
        db.add(_rollout_row(
            "ghcr.io/toup-com/toup-agent:failed1", "aborted_canary_failed",
            datetime.utcnow() - timedelta(days=1),
        ))
        db.add(_rollout_row(
            "ghcr.io/toup-com/toup-agent:failed2", "aborted_canary_failed",
            datetime.utcnow() - timedelta(hours=2),
        ))
        await db.commit()

        tag = await _latest_known_good_image_tag(db)

    assert tag is None
