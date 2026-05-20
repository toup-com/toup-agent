"""PR #49 — ``RoutineRunner._build_run_shim`` populates the row-shaped
object handlers receive as the ``run`` parameter of
``handler.execute(routine, run, db)``.

With the legacy ``RoutineRun`` ORM row gone after the cutover, the
runner builds a ``types.SimpleNamespace`` shim from the BuildJob
fields so the handler-side contract stays sane for future handler
authors. Today no handler reads any field off ``run`` (audit was
done in PR #47 before the cutover), but having a defined shape
prevents a future handler from quietly seeing ``AttributeError``
for a benign access.

What this file pins:

  1. ``id`` mirrors ``BuildJob.id`` (and equals the ``run_id`` passed
     into ``_run_with_retry``).
  2. ``routine_id`` mirrors ``BuildJob.source_id``.
  3. ``user_id`` mirrors ``BuildJob.user_id``.
  4. ``fire_instant`` mirrors ``BuildJob.fire_instant`` (mig 051
     column, populated by ``_fire``).
  5. ``scheduled_for_local_date`` is parsed from
     ``BuildJob.idempotency_key`` (``_fire`` sets it to
     ``str(local_date)``).
  6. A malformed/missing ``idempotency_key`` leaves
     ``scheduled_for_local_date=None`` rather than raising.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime
from types import SimpleNamespace

import pytest


os.environ.setdefault("AGENT_API_KEY", "test-key-shim")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000sh001")


USER_ID = "00000000-0000-0000-0000-0000000sh001"
ROUTINE_ID = "00000000-0000-0000-0000-0000000sh100"


def _fake_job_row(**overrides):
    """A small stand-in for a SQLAlchemy BuildJob row — the shim
    builder only reads attributes off it; no DB session required."""
    defaults = {
        "id": str(uuid.uuid4()),
        "user_id": USER_ID,
        "source_id": ROUTINE_ID,
        "idempotency_key": "2026-05-20",
        "fire_instant": datetime(2026, 5, 20, 8, 0, 0),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_routine(**overrides):
    defaults = {
        "id": ROUTINE_ID,
        "user_id": USER_ID,
        "kind": "email_briefing",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_shim_mirrors_buildjob_id():
    from app.agent.routines.runner import RoutineRunner

    job = _fake_job_row()
    shim = RoutineRunner._build_run_shim(job, _fake_routine())
    assert shim.id == job.id


def test_shim_routine_id_from_source_id():
    from app.agent.routines.runner import RoutineRunner

    job = _fake_job_row(source_id=ROUTINE_ID)
    shim = RoutineRunner._build_run_shim(job, _fake_routine())
    assert shim.routine_id == ROUTINE_ID


def test_shim_user_id_from_job():
    from app.agent.routines.runner import RoutineRunner

    job = _fake_job_row(user_id=USER_ID)
    shim = RoutineRunner._build_run_shim(job, _fake_routine())
    assert shim.user_id == USER_ID


def test_shim_fire_instant_from_job():
    from app.agent.routines.runner import RoutineRunner

    fi = datetime(2026, 5, 20, 7, 30, 15)
    job = _fake_job_row(fire_instant=fi)
    shim = RoutineRunner._build_run_shim(job, _fake_routine())
    assert shim.fire_instant == fi


def test_shim_scheduled_local_date_from_idempotency_key():
    """``_fire`` sets ``idempotency_key=str(local_date)``; the shim
    parses that back into a ``date`` instance for handler ergonomics."""
    from app.agent.routines.runner import RoutineRunner

    job = _fake_job_row(idempotency_key="2026-05-20")
    shim = RoutineRunner._build_run_shim(job, _fake_routine())
    assert shim.scheduled_for_local_date == date(2026, 5, 20)


def test_shim_scheduled_local_date_none_on_unparseable_key():
    """A future kind that uses a non-date idempotency_key shape
    (e.g., an hour bucket like ``2026-05-20-09``) leaves
    ``scheduled_for_local_date=None`` rather than raising. The
    handler contract stays stable."""
    from app.agent.routines.runner import RoutineRunner

    job = _fake_job_row(idempotency_key="2026-05-20-09")
    shim = RoutineRunner._build_run_shim(job, _fake_routine())
    assert shim.scheduled_for_local_date is None


def test_shim_scheduled_local_date_none_on_missing_key():
    """No idempotency_key (defensive — should not happen in
    production) leaves the field None."""
    from app.agent.routines.runner import RoutineRunner

    job = _fake_job_row(idempotency_key=None)
    shim = RoutineRunner._build_run_shim(job, _fake_routine())
    assert shim.scheduled_for_local_date is None


def test_shim_falls_back_to_routine_when_job_missing_attrs():
    """If the job row is missing ``user_id`` / ``source_id``
    (extreme defence-in-depth), the shim falls back to the
    routine's values so we never hand the handler ``None``
    where a real id is expected."""
    from app.agent.routines.runner import RoutineRunner

    job = SimpleNamespace(
        id="job-id",
        # No user_id, no source_id, no fire_instant, no idempotency_key.
    )
    routine = _fake_routine(id="r-id", user_id="u-id")
    shim = RoutineRunner._build_run_shim(job, routine)
    assert shim.id == "job-id"
    assert shim.routine_id == "r-id"
    assert shim.user_id == "u-id"
    assert shim.scheduled_for_local_date is None
    assert shim.fire_instant is None
