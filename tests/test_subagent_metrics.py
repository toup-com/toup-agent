"""Sub-agent metrics — Phase 6 observability.

Pin:
  - Spawn-attempted counter increments with the right labels
    (channel, parent_depth).
  - Spawn-rejected counter labels match the closed REJECTION_CODES
    set.
  - Duration histogram observation lands in a bucket keyed by
    outcome.
  - Credit-spent histogram scales credits → thousandths so the
    existing ms-based buckets work.
  - Runaway detection: > 10 spawns in 5 min emits the runaway log
    event; 10 or fewer doesn't.
  - The Prometheus exposition renders the new metric names so a
    /metrics scraper sees them.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_metrics_for_test():
    """Connector_metrics is process-global. Reset before each test so
    counter snapshots don't accumulate across tests."""
    from app.services.connector_metrics import reset_for_tests
    from app.agent import subagent_metrics
    reset_for_tests()
    subagent_metrics.reset_runaway_state_for_tests()
    yield
    reset_for_tests()
    subagent_metrics.reset_runaway_state_for_tests()


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────


def test_metric_names_are_stable():
    """Dashboards key on these names. Pin them so a rename in one
    site lights up here, not in a Grafana alert that silently went
    flat."""
    from app.agent.subagent_metrics import (
        M_SPAWN_ATTEMPTED, M_SPAWN_REJECTED,
        M_LANE_ACQUIRED, M_LANE_RELEASED,
        M_RUN_DURATION_MS, M_RUN_CREDIT_SPENT_THOUSANDTHS,
    )
    assert M_SPAWN_ATTEMPTED == "subagent_spawn_attempted_total"
    assert M_SPAWN_REJECTED == "subagent_spawn_rejected_total"
    assert M_LANE_ACQUIRED == "subagent_lane_acquired_total"
    assert M_LANE_RELEASED == "subagent_lane_released_total"
    assert M_RUN_DURATION_MS == "subagent_run_duration_ms"
    assert M_RUN_CREDIT_SPENT_THOUSANDTHS == "subagent_run_credit_spent_thousandths"


# ──────────────────────────────────────────────────────────────────────
# Counters
# ──────────────────────────────────────────────────────────────────────


def test_spawn_attempted_counter_with_labels():
    from app.agent.subagent_metrics import record_spawn_attempted
    from app.services.connector_metrics import render

    record_spawn_attempted(channel="web", parent_depth=0)
    record_spawn_attempted(channel="telegram", parent_depth=0)
    record_spawn_attempted(channel="web", parent_depth=1)

    out = render()
    # Each unique label combo gets its own counter line.
    assert 'subagent_spawn_attempted_total{channel="web",parent_depth="0"} 1' in out
    assert 'subagent_spawn_attempted_total{channel="telegram",parent_depth="0"} 1' in out
    assert 'subagent_spawn_attempted_total{channel="web",parent_depth="1"} 1' in out


def test_spawn_rejected_counter_per_reason():
    """Every value in REJECTION_CODES is a valid label value for
    the reject counter. A typo in either side surfaces here."""
    from app.agent.subagent_dispatcher import REJECTION_CODES
    from app.agent.subagent_metrics import record_spawn_rejected
    from app.services.connector_metrics import render

    for code in REJECTION_CODES.all():
        record_spawn_rejected(reason=code)

    out = render()
    for code in REJECTION_CODES.all():
        assert f'subagent_spawn_rejected_total{{reason="{code}"}} 1' in out


def test_lane_acquired_released_pair_gives_implicit_gauge():
    """acquired - released = currently-held. Operator dashboards
    derive concurrent count this way."""
    from app.agent.subagent_metrics import (
        record_lane_acquired, record_lane_released,
    )
    from app.services.connector_metrics import render

    record_lane_acquired()
    record_lane_acquired()
    record_lane_acquired()
    record_lane_released(outcome="success")

    out = render()
    assert "subagent_lane_acquired_total 3" in out
    assert 'subagent_lane_released_total{outcome="success"} 1' in out


# ──────────────────────────────────────────────────────────────────────
# Histograms
# ──────────────────────────────────────────────────────────────────────


def test_duration_histogram_records_per_outcome():
    from app.agent.subagent_metrics import record_run_duration
    from app.services.connector_metrics import render

    record_run_duration(duration_ms=120, outcome="success")
    record_run_duration(duration_ms=15_000, outcome="success")
    record_run_duration(duration_ms=300_000, outcome="timeout")

    out = render()
    # Counts land in buckets keyed by outcome label.
    assert 'subagent_run_duration_ms_count{outcome="success"} 2' in out
    assert 'subagent_run_duration_ms_count{outcome="timeout"} 1' in out


def test_credit_spent_histogram_scales_to_thousandths():
    """connector_metrics histogram buckets are integer ms (50, 100,
    250, ...). We encode credits as thousandths so 0.001 credits =
    1 unit, lands in the 50-unit bucket; 0.5 credits = 500 units,
    lands in the 500-unit bucket."""
    from app.agent.subagent_metrics import record_credit_spent
    from app.services.connector_metrics import render

    record_credit_spent(credit=0.001, outcome="success")
    record_credit_spent(credit=0.5, outcome="success")
    record_credit_spent(credit=2.0, outcome="failed")

    out = render()
    assert 'subagent_run_credit_spent_thousandths_count{outcome="success"} 2' in out
    assert 'subagent_run_credit_spent_thousandths_count{outcome="failed"} 1' in out


def test_credit_spent_zero_or_none_skipped():
    """0 credit (or None) doesn't pollute the histogram — those
    aren't observations worth bucketing."""
    from app.agent.subagent_metrics import record_credit_spent
    from app.services.connector_metrics import render

    record_credit_spent(credit=0.0, outcome="success")
    record_credit_spent(credit=None, outcome="success")  # type: ignore[arg-type]

    out = render()
    assert "subagent_run_credit_spent_thousandths_count" not in out


# ──────────────────────────────────────────────────────────────────────
# Runaway detection
# ──────────────────────────────────────────────────────────────────────


def test_runaway_not_triggered_under_threshold(caplog):
    """10 spawns or fewer in 5 minutes is fine — no alert log."""
    import logging
    from app.agent.subagent_metrics import record_spawn_attempt

    caplog.set_level(logging.WARNING, logger="app.agent.subagent_metrics")
    for _ in range(10):
        assert record_spawn_attempt("user-A") is False
    assert "subagent.runaway_detected" not in caplog.text


def test_runaway_triggered_at_eleventh_spawn(caplog):
    """The 11th spawn in the window triggers the alert log."""
    import logging
    from app.agent.subagent_metrics import record_spawn_attempt

    caplog.set_level(logging.WARNING, logger="app.agent.subagent_metrics")
    for _ in range(10):
        record_spawn_attempt("user-B")
    runaway = record_spawn_attempt("user-B")
    assert runaway is True
    assert "subagent.runaway_detected" in caplog.text
    assert "user_id=user-B" in caplog.text


def test_runaway_is_per_user_not_global(caplog):
    """User A's burst shouldn't trigger the alert for user B."""
    import logging
    from app.agent.subagent_metrics import record_spawn_attempt

    caplog.set_level(logging.WARNING, logger="app.agent.subagent_metrics")
    for _ in range(11):
        record_spawn_attempt("user-C")  # triggers
    caplog.clear()
    # user-D's first spawn shouldn't immediately trigger
    assert record_spawn_attempt("user-D") is False
    assert "subagent.runaway_detected" not in caplog.text


# ──────────────────────────────────────────────────────────────────────
# Structured-log helper
# ──────────────────────────────────────────────────────────────────────


def test_log_lifecycle_emits_stable_key_value_shape(caplog):
    import logging
    from app.agent.subagent_metrics import log_lifecycle

    caplog.set_level(logging.INFO, logger="app.agent.subagent_metrics")
    log_lifecycle(
        event="completed",
        user_id="user-X",
        job_id="job-Y",
        parent_job_id="parent-Z",
        depth=1,
        outcome="success",
        duration_ms=1234.5,
        credit_spent=0.1234,
    )
    msg = caplog.text
    assert "subagent.completed" in msg
    assert "user_id=user-X" in msg
    assert "job_id=job-Y" in msg
    assert "parent_job_id=parent-Z" in msg
    assert "depth=1" in msg
    assert "outcome=success" in msg
    assert "duration_ms=1234.5" in msg
    assert "credit_spent=0.1234" in msg


def test_log_lifecycle_omits_unset_optional_fields(caplog):
    import logging
    from app.agent.subagent_metrics import log_lifecycle

    caplog.set_level(logging.INFO, logger="app.agent.subagent_metrics")
    log_lifecycle(
        event="rejected",
        user_id="u",
        job_id="-",
        reason="SUBAGENT_DISABLED",
    )
    msg = caplog.text
    assert "reason=SUBAGENT_DISABLED" in msg
    # Optional fields not set: must not appear with junk values
    assert "duration_ms=" not in msg
    assert "credit_spent=" not in msg
    assert "outcome=" not in msg
