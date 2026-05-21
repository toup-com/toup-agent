"""Production-fix tests for the one-active-per-kind contract.

Background: 2026-05-14 a user reported two identical Gmail briefings
showing up in their day-chat. Root cause was that the in-Python guard
at `app/api/routines.py:create_routine` used `.scalar_one_or_none()`
which raises `MultipleResultsFound` when ≥2 rows match — i.e., the
check crashed in exactly the state it was meant to prevent. Two
enabled `email_briefing` routines slipped in (race / pre-guard window)
and both fired on their respective crons.

Migration 041 adds a partial UNIQUE index on
`(user_id, kind) WHERE enabled = true AND kind != 'agent_task'`
(Postgres-only — load-bearing) and a cleanup pass that disables all
but the freshest. The Python guard now uses
`app.services.routine_uniqueness.find_conflicting_enabled_routine_ids`
which returns a LIST (handles multi-row), not `.scalar_one_or_none()`
which crashes.

This file pins three contracts at the helper-function layer:
  1. Empty list when no enabled routine of this kind exists.
  2. ONE-element list when one enabled sibling exists.
  3. MULTI-element list when the legacy duplicate state exists —
     the function MUST NOT crash; it must return all the ids.
  4. `agent_task` is exempt — always returns empty list.
  5. `exclude_routine_id` excludes self (the update-enable path).

Tests target the service layer directly because the FastAPI router
init is broken in this test env (pre-existing Starlette version skew).
The DB partial UNIQUE is verified separately via a Postgres-only test
that skips on SQLite.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest


# ── Helpers ─────────────────────────────────────────────────────────


async def _make_user() -> str:
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            User(
                id=user_id, email=f"{user_id}@oneperkind.test",
                hashed_password="x", name="Test", timezone="UTC",
            )
        )
        await db.commit()
    return user_id


async def _seed_routine(
    user_id: str,
    *,
    kind: str = "email_briefing",
    enabled: bool = True,
    cron: str = "58 10 * * *",
) -> str:
    """Direct-insert a routine row (bypass the API). Used to set up
    legacy duplicate state for the regression tests."""
    from app.db import async_session_maker
    from app.db.models import Routine

    rid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            Routine(
                id=rid, user_id=user_id, kind=kind, enabled=enabled,
                schedule_cron_local=cron, last_status="never_run",
                config_json={"delivery_channels": ["website"]},
            )
        )
        await db.commit()
    return rid


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_list_when_no_enabled_routine_exists():
    """Zero matching enabled routines → empty list. The create path
    treats this as "go ahead, no conflict"."""
    from app.db import async_session_maker
    from app.services.routine_uniqueness import find_conflicting_enabled_routine_ids

    user_id = await _make_user()
    async with async_session_maker() as db:
        result = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind="email_briefing",
        )
    assert result == []


@pytest.mark.asyncio
async def test_disabled_routines_do_not_count_as_conflict():
    """A disabled `email_briefing` doesn't block a new one from being
    created — only enabled rows compete for the one-per-kind slot."""
    from app.db import async_session_maker
    from app.services.routine_uniqueness import find_conflicting_enabled_routine_ids

    user_id = await _make_user()
    await _seed_routine(user_id, kind="email_briefing", enabled=False)

    async with async_session_maker() as db:
        result = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind="email_briefing",
        )
    assert result == []


@pytest.mark.asyncio
async def test_single_enabled_routine_returns_one_id():
    """Standard case: one existing enabled routine → 409 with its id."""
    from app.db import async_session_maker
    from app.services.routine_uniqueness import find_conflicting_enabled_routine_ids

    user_id = await _make_user()
    rid = await _seed_routine(user_id, kind="email_briefing")

    async with async_session_maker() as db:
        result = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind="email_briefing",
        )
    assert result == [rid]


@pytest.mark.asyncio
async def test_legacy_duplicate_state_returns_all_ids_no_crash():
    """The original 2026-05-14 bug: when 2+ enabled rows already exist
    (legacy DB state, before migration 041 cleaned them up), the helper
    MUST return all the ids without crashing. The pre-fix code used
    `.scalar_one_or_none()` which raises `MultipleResultsFound` here."""
    from app.db import async_session_maker
    from app.services.routine_uniqueness import find_conflicting_enabled_routine_ids

    user_id = await _make_user()
    rid1 = await _seed_routine(user_id, kind="email_briefing")
    rid2 = await _seed_routine(user_id, kind="email_briefing")

    async with async_session_maker() as db:
        result = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind="email_briefing",
        )
    assert set(result) == {rid1, rid2}


@pytest.mark.asyncio
async def test_agent_task_is_exempt_returns_empty_even_with_duplicates():
    """`agent_task` is the generic kind — many enabled routines are
    valid. The helper short-circuits to [] even when many exist."""
    from app.db import async_session_maker
    from app.services.routine_uniqueness import find_conflicting_enabled_routine_ids

    user_id = await _make_user()
    await _seed_routine(user_id, kind="agent_task")
    await _seed_routine(user_id, kind="agent_task")
    await _seed_routine(user_id, kind="agent_task")

    async with async_session_maker() as db:
        result = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind="agent_task",
        )
    assert result == []


@pytest.mark.asyncio
async def test_reminder_is_exempt_returns_empty_even_with_duplicates():
    """`reminder` is the second non-singleton kind. The
    `routines__remind` skill creates ONE reminder per "remind me to X"
    request — users have many ("call mom at 6", "water plants Tuesday",
    "standup ping at 9:30"). Without this exemption, the second
    reminder a user creates 409s with `An enabled 'reminder' routine
    already exists` and the agent surfaces a confusing error.
    Regression test for the 2026-05-21 production bug where the user
    asked "remind me in 2 minutes to test reminders" and got told the
    reminder system was wired to Telegram."""
    from app.db import async_session_maker
    from app.services.routine_uniqueness import find_conflicting_enabled_routine_ids

    user_id = await _make_user()
    await _seed_routine(user_id, kind="reminder")
    await _seed_routine(user_id, kind="reminder")
    await _seed_routine(user_id, kind="reminder")

    async with async_session_maker() as db:
        result = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind="reminder",
        )
    assert result == []


@pytest.mark.asyncio
async def test_exclude_routine_id_excludes_self_for_update_enable_path():
    """When the update path re-enables a previously-disabled routine,
    it passes its OWN id as exclude_routine_id so a self-enable
    doesn't false-positive against itself."""
    from app.db import async_session_maker
    from app.services.routine_uniqueness import find_conflicting_enabled_routine_ids

    user_id = await _make_user()
    self_rid = await _seed_routine(user_id, kind="email_briefing", enabled=False)
    # Manually flip enabled to True to simulate the update path mid-flight.
    from app.db.models import Routine
    async with async_session_maker() as db:
        r = await db.get(Routine, self_rid)
        r.enabled = True
        await db.commit()

    async with async_session_maker() as db:
        result = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind="email_briefing",
            exclude_routine_id=self_rid,
        )
    # Only sibling routines count. With no sibling, the "self-enable"
    # is conflict-free.
    assert result == []


@pytest.mark.asyncio
async def test_exclude_routine_id_still_flags_other_enabled_siblings():
    """Exclude self, but a separate enabled sibling still trips the
    409 path. Test pins that exclude_routine_id is precise."""
    from app.db import async_session_maker
    from app.services.routine_uniqueness import find_conflicting_enabled_routine_ids

    user_id = await _make_user()
    sibling = await _seed_routine(user_id, kind="email_briefing")
    self_rid = await _seed_routine(user_id, kind="email_briefing")

    async with async_session_maker() as db:
        result = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind="email_briefing",
            exclude_routine_id=self_rid,
        )
    assert result == [sibling]
