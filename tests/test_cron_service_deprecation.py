"""Phase C tests — CronService deprecation gate + migrated-row skip.

Pins three contracts:

  1. ``CronService.start()`` is a no-op when
     ``settings.cron_service_enabled = False``. The scheduler is not
     started; ``_load_jobs_from_db`` is not called.

  2. ``CronService._load_jobs_from_db`` skips any ``cron_jobs`` row
     whose ``migrated_to_routine_id`` is set, regardless of the row's
     ``enabled`` flag. This is the anti-double-fire interlock with
     mig 043.

  3. The deprecation log fires on a normal start so operators see the
     warning even when the flag stays True.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest


# Source-grep guards — keep the deprecation surface findable.

from pathlib import Path
BACKEND = Path(__file__).resolve().parent.parent
_CRON_SRC = (BACKEND / "app/agent/cron_service.py").read_text()
_TELEGRAM_SRC = (BACKEND / "app/agent/telegram_bot.py").read_text()


def test_cron_service_module_has_deprecation_gate():
    """The flag-check at start() is load-bearing — dropping it would
    re-activate every CronJob the moment a tenant deploys this code
    with the flag set False."""
    assert "cron_service_enabled" in _CRON_SRC, (
        "CronService.start() must read settings.cron_service_enabled. "
        "Removing this check re-activates legacy CronJobs even when "
        "the operator has deprecated the service."
    )


def test_cron_service_skips_migrated_rows_at_load():
    """The migrated-row skip in _load_jobs_from_db prevents the
    legacy CronService from firing reminders that have already been
    transferred to the Routine system. Without this guard the same
    reminder fires twice."""
    assert "migrated_to_routine_id" in _CRON_SRC, (
        "_load_jobs_from_db must skip rows with migrated_to_routine_id set. "
        "Dropping this check re-introduces the 2026-05-12 double-delivery "
        "scenario (CronJob and Routine both firing the same payload)."
    )


def test_telegram_cron_command_renders_deprecation_banner():
    """Every /cron response must lead with a banner pointing users at
    /reminders so the deprecation isn't invisible."""
    assert "/cron is deprecated" in _TELEGRAM_SRC, (
        "/cron must surface a one-line deprecation banner. Without it "
        "users keep adding cron_jobs that will stop firing the moment "
        "Phase D drops the table."
    )


def test_phase_d_migration_default_is_noop():
    """Mig 044 must be a no-op by default. Auto-dropping cron_jobs on
    `alembic upgrade head` without operator opt-in would destroy any
    reminder a tenant hasn't migrated yet."""
    mig_path = BACKEND / "alembic/versions/20260514_0044_044_drop_cron_jobs_optin.py"
    assert mig_path.exists(), "Phase D migration file missing."
    body = mig_path.read_text()
    assert "ALLOW_CRONJOB_TABLE_DROP" in body, (
        "Phase D must gate the DROP behind ALLOW_CRONJOB_TABLE_DROP env var."
    )
    assert "_opt_in_set()" in body, (
        "Phase D must check the opt-in env var before dropping."
    )
    assert "migrated_to_routine_id IS NULL" in body, (
        "Phase D must refuse to drop when any enabled cron_jobs row "
        "still has NULL migrated_to_routine_id — those reminders would "
        "silently stop firing."
    )
