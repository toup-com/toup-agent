"""GA run D-7: memory decay was armed but could not fire.

`memory_decay` was registered with `IntervalTrigger(hours=6)`. An
interval trigger's first fire is six hours **after the scheduler
starts** — and the agent fleet is recreated on every merge to `main`
under `backend/**`. Measured 2026-08-10: **178 such commits in 14 days,
median gap 0.3h**, with only 13 of 177 gaps reaching six hours. Every
recreation restarts the scheduler and resets the timer, so the job's
first fire kept being cancelled by the next deploy.

Observed directly: `toup-agent-871bac24` reported
`StartedAt=2026-08-10T18:45:35Z`, then `StartedAt=2026-08-10T23:34:40Z`
— recreated inside the hour.

The evidence that it never ran, three ways:

* `memories.last_decayed_at` is NULL for **every** row on all three
  tenants checked (135, 8, and 12 memories).
* No `DECAYED` memory_event on two of them; the 12 on the founder's
  tenant predate both the column (#490, 2026-08-06) and the flag flip
  (#549, 2026-08-10).
* `agent_memory_maintenance_enabled` only became True on 2026-08-10, so
  100% NULL is exactly what "never ran since the stamp existed" looks
  like.

The fix is the one its neighbour already uses: `memory_consolidation`
sits four lines below with `CronTrigger(hour=3)`, which is wall-clock and
therefore survives restarts. Decay keeps its ~6-hourly cadence as fixed
clock times instead of an elapsed-time interval.

Red-first: both assertions fail against the pre-fix registration.
"""
from __future__ import annotations

import inspect
import re

import pytest


def _registration_src() -> str:
    from app.scripts import scheduled_tasks

    return inspect.getsource(scheduled_tasks)


def _job_block(src: str, job_id: str) -> str:
    """The add_job(...) call that registers `job_id`."""
    m = re.search(
        r"scheduler\.add_job\((?:[^()]|\([^()]*\))*?id=\"" + re.escape(job_id) + r"\"",
        src,
        re.S,
    )
    assert m, f"no add_job registration found for id={job_id!r}"
    return m.group(0)


def test_decay_uses_a_wallclock_trigger():
    """An IntervalTrigger's first fire is N hours after scheduler start;
    the fleet restarts far more often than that."""
    block = _job_block(_registration_src(), "memory_decay")
    assert "CronTrigger" in block, (
        "memory_decay still uses an elapsed-time trigger. The fleet is "
        "recreated every ~18 minutes (median gap between backend merges), "
        "so a 6-hour interval resets before it ever fires — which is why "
        "last_decayed_at is NULL on every memory in production."
    )
    assert "IntervalTrigger" not in block


def test_no_memory_maintenance_job_relies_on_a_long_interval():
    """The general property, so this cannot come back on a sibling job.

    Anything whose cadence is longer than the fleet's restart cadence
    must be wall-clock. Short intervals (health check, minutes) are fine
    — they fire long before a redeploy.
    """
    src = _registration_src()
    offenders = []
    for job_id in (
        "memory_decay",
        "memory_consolidation",
        "retrieval_feedback_analysis",
        "day_archival",
    ):
        try:
            block = _job_block(src, job_id)
        except AssertionError:
            continue          # not every job exists in every build
        if "IntervalTrigger" in block and "hours=" in block:
            offenders.append(job_id)
    assert not offenders, (
        f"these jobs use an hours-scale IntervalTrigger and will not "
        f"survive the fleet's restart cadence: {offenders}"
    )


def test_decay_keeps_roughly_its_configured_cadence():
    """The flag said 'every 6 hours'; the fix must not quietly become
    'once a day'."""
    block = _job_block(_registration_src(), "memory_decay")
    hours = re.search(r'hour\s*=\s*"([^"]+)"', block)
    assert hours, f"decay's CronTrigger has no explicit hour spec: {block}"
    slots = [h for h in hours.group(1).split(",") if h.strip()]
    assert len(slots) >= 4, (
        f"decay dropped from ~4 runs/day to {len(slots)} — the 6-hourly "
        "cadence in decay_interval_hours should be preserved"
    )


def test_decay_runs_before_consolidation():
    """Decay re-weights; consolidation summarises. Order matters."""
    src = _registration_src()
    decay = _job_block(src, "memory_decay")
    consolidation = _job_block(src, "memory_consolidation")

    decay_hours = re.search(r'hour\s*=\s*"([^"]+)"', decay)
    cons_hour = re.search(r"hour\s*=\s*(\d+)", consolidation)
    assert decay_hours and cons_hour

    cons = int(cons_hour.group(1))
    assert any(int(h) < cons for h in decay_hours.group(1).split(",")), (
        "no decay slot runs before consolidation's hour; consolidation "
        "would summarise strengths decay has not yet updated"
    )
