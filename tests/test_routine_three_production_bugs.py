"""Regression tests for three 2026-05-12 production bugs.

  1. **Prompt leak.** When an `agent_task` routine fired, its
     `prompt_text` ("Every day at 1:21 PM America/Toronto time, fetch
     the user's latest 5 Gmail messages…") appeared in the day-chat
     as if the USER had typed it. Root cause: the agent_task handler
     called `AgentRunner.run(user_message=prompt_text)` which writes a
     synthetic Message with role="user" by default.

  2. **Idempotency gate blocks mid-day schedule change.** The
     `(routine_id, scheduled_for_local_date)` UNIQUE constraint
     prevented a second fire on the same day, EVEN when the user
     moved the schedule from 8 AM to 1:21 PM after the 8 AM run
     failed. Same-day retry was permanently blocked until midnight.

  3. **Silent UTC fallback when User.timezone is missing.** The
     `_resolve_tz` helper logs a WARN and returns ZoneInfo("UTC"),
     so a routine for a tz-less user fires at UTC-interpreted times
     (1:21 PM UTC = 9:21 AM Toronto during EDT). The auth-side
     auto-capture commits (f62ae3a, 2d1e14c) cover new users but
     legacy users without a captured tz silently misfire.

Source-grep so a future refactor can't quietly drop the fixes.
"""
from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_AGENT_TASK = (BACKEND / "app/agent/routines/agent_task_handler.py").read_text()
_RUNNER = (BACKEND / "app/agent/routines/runner.py").read_text()
_API = (BACKEND / "app/api/routines.py").read_text()


# ── 1. Prompt leak ──────────────────────────────────────────────────


def test_agent_task_handler_does_not_save_user_message():
    """The routine's prompt_text is a SYSTEM instruction, not user
    speech. Persisting it with role="user" pollutes the chat history
    with the LLM-generated wrapper text the user never typed."""
    # The handler MUST pass save_user_message=False to runner.run.
    # Grep the literal kwarg — a future refactor that drops it would
    # re-leak the prompt to chat.
    idx = _AGENT_TASK.index("runner.run(")
    body = _AGENT_TASK[idx:idx + 1500]
    assert "save_user_message=False" in body, (
        "agent_task_handler.runner.run(…) must pass "
        "save_user_message=False. Drop this and the routine's "
        "prompt_text re-appears in the day-chat as a synthetic "
        "user message — the 2026-05-12 leak comes back."
    )


# ── 2. Idempotency gate after schedule change ──────────────────────


def test_update_routine_clears_failed_runs_when_schedule_changes():
    """When the user moves a routine's schedule mid-day, today's
    FAILED / SKIPPED run rows must be cleared so the new schedule
    can claim a fresh row on its next fire. Without this, the user
    is locked out of same-day retries until midnight."""
    idx = _API.index("async def update_routine(")
    body = _API[idx:idx + 4000]
    # The schedule-change branch must contain a DELETE against
    # RoutineRun filtered on today + failed/skipped statuses.
    assert "schedule_changed" in body, (
        "update_routine must detect when schedule_cron_local actually "
        "changed (vs unchanged updates like name/enabled) so we only "
        "clear runs on real reschedules."
    )
    assert "delete(RoutineRun)" in body or "delete(routine_run)" in body.lower(), (
        "update_routine must DELETE today's failed/skipped RoutineRun "
        "rows when schedule changes. Drop this and the user can't "
        "retry today after a failed morning fire."
    )
    # And the delete MUST be scoped to recoverable statuses — never
    # touch success rows (don't double-deliver).
    assert (
        '"failed"' in body and '"skipped_reauth"' in body
    ), (
        "the run-clearance MUST be scoped to failed + skipped_reauth. "
        "Clearing success rows would risk double-delivering to "
        "Telegram / WhatsApp."
    )


def test_force_run_drops_prior_failed_row_before_firing():
    """The dashboard's 'Run now' button must work as a retry for
    today's failed run. Without bypassing the idempotency gate for
    recoverable terminal states, the user gets a 409 every time
    they click — the button is dead-on-arrival after a morning miss."""
    idx = _API.index("async def force_run(")
    body = _API[idx:idx + 3500]
    # Prior row in failed / skipped_reauth → delete + commit before
    # the new fire. Prior row in success → still 409 (don't double-
    # deliver). Prior row in running → also 409 (race-safe).
    assert "prior.status in" in body or "prior.status ==" in body, (
        "force_run must branch on the prior run's status. Without it "
        "we either always-409 (current bug) or always-overwrite (which "
        "double-delivers successful runs)."
    )
    assert (
        '"failed"' in body and '"skipped_reauth"' in body
    ), (
        "force_run's retry branch must cover failed AND skipped_reauth. "
        "Both are recoverable terminal states the user should be able "
        "to retry."
    )


# ── 3. Timezone hardening ──────────────────────────────────────────


def test_create_routine_rejects_when_user_timezone_missing():
    """A routine for a tz-less user fires at UTC-interpreted times —
    a 'fire at 1:21 PM' becomes 'fire at 1:21 PM UTC' = 9:21 AM
    Toronto during EDT. Reject creation early so the user fixes the
    tz BEFORE the routine starts misfiring."""
    idx = _API.index("async def create_routine(")
    body = _API[idx:idx + 4000]
    # Must check User.timezone and raise on missing.
    assert "missing_timezone" in body or "timezone isn't set" in body.lower(), (
        "create_routine must reject when User.timezone is missing. "
        "Silently accepting the create and falling back to UTC inside "
        "the scheduler is the 2026-05-12 misfire bug."
    )
    # Should be a 400, not a 500 — the user can recover by setting
    # their tz.
    assert "status_code=400" in body, (
        "missing-timezone rejection should be a 400 (user-recoverable) "
        "not a 500 (server-error implication)."
    )


def test_register_trigger_skips_when_user_timezone_missing():
    """Belt-and-braces: even if a routine somehow exists in DB without
    a tz (legacy data, manual SQL), the scheduler must SKIP it rather
    than silently registering it in UTC. Skipping logs an error the
    operator can grep; misfiring is invisible."""
    idx = _RUNNER.index("async def _register_trigger_for(")
    body = _RUNNER[idx:idx + 2500]
    assert "missing_tz" in body, (
        "_register_trigger_for must have a 'missing_tz' early-return "
        "for tz-less users. Without it the routine fires at "
        "UTC-interpreted times silently — the user's 1 PM briefing "
        "becomes a 9 AM briefing."
    )
    # And the skip must log at ERROR (not WARN) so it surfaces in
    # operator dashboards. WARN gets buried under noise.
    assert "logger.error" in body, (
        "missing-tz skip must log at ERROR. WARN doesn't surface in "
        "production alerting; ERROR pages the operator."
    )
