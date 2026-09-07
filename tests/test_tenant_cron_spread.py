"""The fleet's wall-clock crons must not all name the same second.

2026-09-07, measured on the production host: `day_archival` fired at
15:00:00.023 / .028 / .093 / .438 UTC on four independent tenant containers,
because every one of them registered `CronTrigger(minute=0)`. ~95 containers
share one 16-core VPS and one pgbouncer, so that is a fleet-wide burst every
hour. At 15:00:00 that hour it stalled a brand-new tenant's FIRST turn: the
container's own scheduler logged `Run time of job "Memory Maintenance:
day_archival" was missed by 0:00:05.77`, the platform saw a 7 s WebSocket
accept and two HTTP ReadTimeouts, and the client's 12 s silent-turn grace
expired and reconnected.

These tests pin the three properties the fix has to have and the one it must
NOT lose (a wall-clock trigger — an IntervalTrigger's first fire is measured
from scheduler start and this fleet restarts more often than hourly).
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys

os.environ.setdefault("ENVIRONMENT", "test")

from app.scripts import scheduled_tasks as st


SYNTHETIC = [f"toup_agent_feed{n:04d}" for n in range(1, 101)]


def _minute(identity: str, job_id: str = "day_archival") -> int:
    trigger = st.spread_hourly_cron(job_id, identity=identity)
    return int(str(trigger.fields[trigger.FIELD_NAMES.index("minute")]))


# ── The defect: every tenant on the same second ───────────────────────────

def test_a_hundred_tenants_do_not_share_one_minute():
    """No single minute of the hour may hold more than 10 % of the fleet.

    This is the falsifier for the old code, where 100 % of the fleet sat in
    minute 0.
    """
    buckets: dict[int, int] = {}
    for ident in SYNTHETIC:
        m = _minute(ident)
        assert 0 <= m <= 59
        buckets[m] = buckets.get(m, 0) + 1
    worst = max(buckets.values())
    assert worst <= 10, f"one minute holds {worst} of 100 tenants: {buckets}"
    # And the spread is real, not two clumps: at least half the hour is used.
    assert len(buckets) >= 30, f"only {len(buckets)} distinct minutes used"


def test_the_second_within_the_minute_is_spread_too():
    """Sixty containers landing on second 0 of sixty different minutes would
    still be a once-a-minute stampede on a shared pgbouncer."""
    seconds = set()
    for ident in SYNTHETIC:
        t = st.spread_hourly_cron("day_archival", identity=ident)
        seconds.add(int(str(t.fields[t.FIELD_NAMES.index("second")])))
    assert len(seconds) >= 30, f"only {len(seconds)} distinct seconds used"


# ── Stability: the same tenant must keep its slot ─────────────────────────

def test_the_same_identity_gets_the_same_slot_twice():
    for ident in SYNTHETIC[:20]:
        assert _minute(ident) == _minute(ident)
        assert st.cron_slot_seconds("day_archival", 3600, ident) == \
            st.cron_slot_seconds("day_archival", 3600, ident)


def test_the_slot_survives_a_different_process_hash_seed():
    """`hash()` is randomised per process (PYTHONHASHSEED). A builtin-hash
    implementation looks perfectly stable inside one interpreter and hands the
    same container a different slot on every restart — which is the whole
    property this fix exists to provide. Two child interpreters with hostile,
    different seeds must agree with each other and with us."""
    code = (
        "import os;os.environ.setdefault('ENVIRONMENT','test');"
        "from app.scripts import scheduled_tasks as st;"
        "print(st.cron_slot_seconds('day_archival',3600,'toup_agent_feed0047'))"
    )
    outs = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=os.getcwd())
        outs.append(subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env=env, cwd=os.getcwd(),
        ).stdout.strip())
    assert len(set(outs)) == 1, f"slot moved with PYTHONHASHSEED: {outs}"
    assert outs[0] == str(
        st.cron_slot_seconds("day_archival", 3600, "toup_agent_feed0047")
    )


def test_two_jobs_on_one_tenant_do_not_share_a_second():
    """Salted with the job id: a tenant's hourly jobs must not stack either."""
    ident = "toup_agent_feed0047"
    a = st.cron_slot_seconds("day_archival", 3600, ident)
    b = st.cron_slot_seconds("current_context_rollover", 3600, ident)
    assert a != b


# ── The daily window stays inside its hour ────────────────────────────────

def test_a_daily_job_stays_in_the_hour_it_was_configured_for():
    for ident in SYNTHETIC:
        t = st.spread_daily_cron("memory_consolidation", hour=3, identity=ident)
        assert int(str(t.fields[t.FIELD_NAMES.index("hour")])) == 3
        assert 0 <= int(str(t.fields[t.FIELD_NAMES.index("minute")])) <= 59
    # A window wider than an hour is clamped, never allowed to escape.
    t = st.spread_daily_cron("x", hour=3, window_minutes=600, identity="a")
    assert int(str(t.fields[t.FIELD_NAMES.index("hour")])) == 3


# ── Identity resolution ───────────────────────────────────────────────────

def test_identity_prefers_the_tenant_database_name():
    """A pool slot keeps its DATABASE_URL across bind, restart, blue-green and
    recreate; the bound user id only appears at bind. Preferring the user id
    would move a slot's cron slot the first time it restarted after a claim."""
    prev = os.environ.pop(st._CRON_SPREAD_ID_ENV, None)
    try:
        ident = st.tenant_cron_identity()
        assert ident, "identity must never be empty"
        os.environ[st._CRON_SPREAD_ID_ENV] = "explicit-override"
        assert st.tenant_cron_identity() == "explicit-override"
    finally:
        os.environ.pop(st._CRON_SPREAD_ID_ENV, None)
        if prev is not None:
            os.environ[st._CRON_SPREAD_ID_ENV] = prev


# ── Source probes: the registrations actually use it ──────────────────────

def test_the_platform_day_archival_registration_is_spread():
    src = inspect.getsource(st.setup_scheduler)
    at = src.find('id="day_archival"')
    assert at > 0
    block = src[max(0, at - 700):at]
    assert 'spread_hourly_cron("day_archival")' in block, \
        "day_archival still registers a fleet-wide fixed trigger"
    assert "CronTrigger(minute=0)" not in block


def test_the_agent_registrations_are_spread_and_still_wall_clock():
    """The three memory-maintenance jobs run in EVERY tenant container — they
    are the ones that actually burst. And they must stay CRON: an
    IntervalTrigger resets on every fleet recreate and never fires."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("agent_main.py").read_text()
    at = src.find("_mm_jobs = []")
    assert at > 0, "the memory-maintenance block moved"
    end = src.find("_mm_registered = []", at)
    assert end > at
    block = src[at:end]
    for job in ("day_archival", "current_context_rollover"):
        assert f'_mm_hourly_cron("{job}")' in block, f"{job} is not spread"
    assert '_mm_daily_cron(\n                            "memory_consolidation"' in block \
        or '_mm_daily_cron("memory_consolidation"' in block
    assert "_MMCron(minute=0)" not in block and "_MMCron(minute=5)" not in block
    # The wall clock is the invariant the spread must not trade away.
    assert "IntervalTrigger" not in block and "_MMInterval" not in block


def test_a_missed_run_is_no_longer_dropped_on_the_floor():
    """APScheduler's default misfire grace is ONE second, so the 5.77 s miss
    of 2026-09-07 did not run late — it did not run at all."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("agent_main.py").read_text()
    at = src.find('name=f"Memory Maintenance: {_mm_id}"')
    assert at > 0
    tail = src[at:at + 800]
    assert "misfire_grace_time" in tail
