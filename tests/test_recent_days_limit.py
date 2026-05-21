"""Regression test for TKT-LAT-009 — hard row cap on get_recent_day_summaries.

Verifies the SELECT carries an explicit ``.limit(DAY_CHATS_ROW_HARD_CAP)``
so a stale local_date column + an inflated ``limit_days`` override can't
trigger a full per-user table scan.
"""

from __future__ import annotations

import inspect

from app.services import recent_days_service


def test_default_limit_days_is_two():
    assert recent_days_service.DEFAULT_LIMIT_DAYS == 2


def test_row_hard_cap_constant_present():
    assert recent_days_service.DAY_CHATS_ROW_HARD_CAP == 30


def test_get_recent_day_summaries_emits_limit_clause():
    """Source-level guardrail: the query body must reference the hard cap."""
    src = inspect.getsource(recent_days_service.get_recent_day_summaries)
    assert ".limit(DAY_CHATS_ROW_HARD_CAP)" in src, (
        "TKT-LAT-009: recent-days SELECT must carry an explicit LIMIT "
        "as defense against unbounded scans."
    )
