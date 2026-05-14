"""Regression tests for two 2026-05-12 issues:

  1. `Routine.next_run_at` was permanently null because the column
     was added to the schema (with a placeholder comment) but the
     sync from APScheduler.next_run_time → DB was never wired.
     Symptom: the routines dashboard showed "next run: —" forever,
     even for healthy daily routines. Source-grep that the sync
     call is wired in both code paths (registration + post-terminal).

  2. `refresh_tenant_tools` had a 3 s timeout with a single attempt
     and best-effort error handling. When a tenant agent was
     warming up (image pull, app startup) the notify silently
     failed → the user's MCP tools cache stayed empty for up to
     60 s after a connector reconnect → the agent told the user
     "the Gmail tool isn't exposed to me right now" even though
     they'd just connected it.

Both fixes are simple wiring; pin them source-grep so a future
cleanup can't quietly drop the sync call or shrink the retry budget.
"""
from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_RUNNER = (BACKEND / "app/agent/routines/runner.py").read_text()
_REFRESH = (BACKEND / "app/services/docker_host_service.py").read_text()


# ── next_run_at sync ────────────────────────────────────────────────


def test_runner_exposes_sync_next_run_helper():
    """The sync helper must be its own method so both code paths
    (register + post-terminal) share the same behaviour. Inlining
    it twice invites drift — one path will quietly diverge."""
    assert "async def _sync_next_run(" in _RUNNER, (
        "RoutineRunner must expose `_sync_next_run(routine_id)`. "
        "Without a shared helper, the register and post-terminal "
        "paths drift and one of them will silently stop writing "
        "next_run_at after a refactor."
    )


def test_register_path_syncs_next_run_at():
    """After `scheduler.add_job(...)`, APScheduler knows the next
    fire time. The DB column must be populated immediately so the
    dashboard renders 'next run: tomorrow 8:00 AM' instead of '—'."""
    # The sync must come AFTER add_job (otherwise the job doesn't
    # exist yet). Spot-check by ensuring the call appears in the
    # function body of _register_trigger_for. The body slice is
    # generous (5000) so adding observability / comments to the
    # method can't accidentally push _sync_next_run out of the window.
    idx = _RUNNER.index("async def _register_trigger_for(")
    body = _RUNNER[idx:idx + 5000]
    assert "_sync_next_run(" in body, (
        "_register_trigger_for must call _sync_next_run after the "
        "scheduler.add_job call. Drop this and next_run_at stays "
        "null forever for newly-registered routines."
    )


def test_post_terminal_resyncs_next_run_at():
    """After each fire, APScheduler advances the job's
    next_run_time to the next cron tick. Without re-syncing,
    next_run_at sticks at the FIRST fire time forever — which is
    in the past for every subsequent dashboard load."""
    idx = _RUNNER.index("async def _post_terminal(")
    # Generous body window (10000) — _post_terminal grew with Ticket 2.1
    # (outcome derivation) and Ticket 2.2 (nudge fan-out + dedupe). The
    # claim under test is presence, not position.
    body = _RUNNER[idx:idx + 10000]
    assert "_sync_next_run(" in body, (
        "_post_terminal must re-sync next_run_at after the run row "
        "is finalised. Without it, the dashboard shows yesterday's "
        "fire time forever — until the container restarts and "
        "_register_trigger_for repopulates it."
    )


def test_sync_next_run_handles_naive_datetime():
    """`Routine.next_run_at` is a tz-naive DateTime column.
    APScheduler hands back a tz-aware datetime. We must convert to
    UTC then strip tzinfo — otherwise the DB driver either rejects
    the value OR stores an 8-hour-off timestamp for Toronto users."""
    idx = _RUNNER.index("async def _sync_next_run(")
    body = _RUNNER[idx:idx + 1500]
    # The conversion sequence: astimezone(timezone.utc) then
    # replace(tzinfo=None). Either swap or skip and the timestamp
    # drifts.
    assert "astimezone(timezone.utc)" in body, (
        "_sync_next_run must convert to UTC before stripping tzinfo. "
        "Without it, Toronto/Tokyo/etc users see next_run_at offset "
        "by hours."
    )
    assert "replace(tzinfo=None)" in body, (
        "_sync_next_run must strip tzinfo before persisting — the "
        "column is a naive DateTime."
    )


def test_sync_next_run_is_fail_soft():
    """A bug in the sync path MUST NOT take down the scheduler. The
    user-visible impact of a null next_run_at is a missing dashboard
    badge; APScheduler still fires on schedule from in-memory state.
    Don't trade a visual bug for a hard scheduler failure."""
    idx = _RUNNER.index("async def _sync_next_run(")
    body = _RUNNER[idx:idx + 1500]
    assert "except Exception" in body, (
        "_sync_next_run must catch Exception. A DB blip during sync "
        "shouldn't crash the scheduler loop."
    )


# ── refresh_tenant_tools durability ─────────────────────────────────


def test_refresh_tools_retries_on_transport_failure():
    """A single 3 s attempt was too brittle — a tenant container
    warming up (image pull, slow startup) routinely took 4-6 s for
    its first request. The notify failed silently, the cache stayed
    empty, the user saw 'Gmail tool not exposed' for up to 60 s
    after reconnect. Add a second attempt with a longer timeout."""
    idx = _REFRESH.index("async def refresh_tenant_tools(")
    body = _REFRESH[idx:idx + 3500]
    # Two-pass loop: ensure both 3.0 and 10.0 are referenced and
    # there's a retry counter / loop construct.
    assert "3.0" in body and "10.0" in body, (
        "refresh_tenant_tools must try the fast 3 s timeout AND a "
        "longer 10 s fallback. Drop either and the original "
        "'silent miss on cold agent' regression returns."
    )
    # The retry must be wrapped in a loop (so the same code path
    # handles both attempts) — not duplicated try/except blocks.
    assert ("for attempt" in body) or ("for _ in" in body) or ("range(" in body), (
        "refresh_tenant_tools must use a loop for the retry — "
        "duplicating the try/except invites drift."
    )


def test_refresh_tools_does_not_retry_on_non_2xx():
    """Retrying a 401 or 404 won't help — the agent reachable but
    rejecting. Bail to the TTL safety net instead of hammering the
    container. Pin this so a future 'always retry' simplification
    doesn't accidentally chew through retry budget on a config bug."""
    idx = _REFRESH.index("async def refresh_tenant_tools(")
    body = _REFRESH[idx:idx + 3500]
    # The non-2xx branch returns False immediately (no continue/loop
    # re-entry). Look for the early return inside the non-2xx
    # branch.
    # We can't grep the exact return placement, so spot-check that
    # the non-2xx path mentions "TTL is the safety net" — the
    # docstring pattern of "we don't retry, we wait."
    assert body.count("TTL is the safety net") >= 1, (
        "refresh_tenant_tools must have at least one early-return "
        "branch that bails to the TTL safety net (rather than "
        "retrying) — non-2xx is the natural case."
    )
