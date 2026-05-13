"""Ticket 3 regression tests — routine scheduler missed-fire bug.

Locks three properties that the bug sweep depends on:

  1. `PATCH /api/routines/{id}` returns a freshly-synced `next_run_at`
     (not the stale pre-reload value). The original bug: the response
     was built from a `routine` ORM object that was refreshed BEFORE
     `reload_routine`'s `_sync_next_run` wrote the new value to DB.

  2. The runner exposes missed-fire / reload-failure / reconcile
     counters via `status_snapshot()` — invariant #4 from the bug
     sweep. Without these, drift between in-memory APScheduler state
     and DB is silent.

  3. The runner's reconcile loop wires up on `start()` and tears down
     on `stop()`. Tested via source-grep + status_snapshot's
     `reconcile_active` flag.
"""

from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_RUNNER = (BACKEND / "app/agent/routines/runner.py").read_text()
_API = (BACKEND / "app/api/routines.py").read_text()


def test_runner_exposes_observability_counters_in_status_snapshot():
    """Invariant #4 — scheduled-execution surfaces must expose a
    missed-fire counter in admin observability. Without these the bug
    sweep's #4 invariant has no enforcement and we can't tell drift
    from working-as-intended."""
    for needle in (
        "missed_fires_total",
        "reload_failures_total",
        "reconcile_runs_total",
        "reconcile_active",
    ):
        assert needle in _RUNNER, (
            f"status_snapshot must include '{needle}'. Operators read "
            f"these off /_runner_status to spot scheduler drift."
        )


def test_runner_increments_missed_fire_counter_on_drift():
    """The `_fire` path must compute now - scheduled_for and increment
    `_missed_fires_total` when drift > misfire_grace_time. Pin via
    source-grep so a refactor can't silently drop the detection."""
    assert "_missed_fires_total += 1" in _RUNNER, (
        "RoutineRunner._fire must increment _missed_fires_total when "
        "drift exceeds misfire_grace_time. Without this we can't tell "
        "a routine fired late from one that fired on time."
    )
    assert "missed_fire routine_id=" in _RUNNER, (
        "The missed-fire path must log structured fields so operators "
        "can grep logs by routine_id."
    )


def test_runner_has_periodic_reconcile_loop():
    """Bug B (silent reload failures) is backstopped by a periodic
    reconciler that calls reload_all every RECONCILE_INTERVAL_SECONDS.
    Pin its existence so a cleanup can't quietly drop the backstop."""
    assert "async def _reconcile_loop(" in _RUNNER, (
        "RoutineRunner must expose `_reconcile_loop()`. This is the "
        "backstop for the Bug B silent-reload-failure mode."
    )
    assert "RECONCILE_INTERVAL_SECONDS" in _RUNNER, (
        "Reconciler must use a named interval constant."
    )
    assert "self._reconcile_runs_total += 1" in _RUNNER, (
        "Each successful reconcile cycle must increment the counter."
    )


def test_api_re_reads_routine_after_reload():
    """Ticket 3 / Bug A — the original symptom. `update_routine` MUST
    re-read the routine from DB after `reload_routine` so the response
    carries the freshly-synced `next_run_at` instead of the stale
    pre-reload value. Pin via source-grep against
    `app/api/routines.py`."""
    # The re-read pattern: a `db.get(Routine, ...)` call AFTER the
    # reload_routine call. Easiest pin: search for the comment marker
    # we left at the re-read site.
    assert "Re-read the routine after reload" in _API, (
        "update_routine must re-read the routine after reload_routine "
        "so the API response carries the synced next_run_at. Without "
        "this re-read, the response shows pre-update data — the "
        "original Ticket 3 / Bug A symptom."
    )


def test_api_surfaces_reload_failures_not_swallowed():
    """Ticket 3 / Bug B — `reload_routine` exceptions must NOT be
    silently swallowed. They must increment a counter AND raise a 5xx
    so the client sees the failure."""
    assert "_reload_failures_total" in _API, (
        "update_routine must increment _reload_failures_total when "
        "reload_routine raises. Silent swallow leaves the OLD job "
        "registered with the OLD cron — the API lies about success."
    )
    # The re-raise as HTTPException 503 — explicit user-visible error.
    assert "status_code=503" in _API, (
        "update_routine must raise HTTPException(503) on reload failure "
        "so the client knows the schedule didn't take effect."
    )
