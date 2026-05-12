"""Unit tests for the trigger rate limiter + coalescer.

Pure-function logic — no DB, no async. The limiter is the production
protection against:
  - Mailing-list bursts producing N×LLM-calls (rate limit per-hour
    and per-minute caps).
  - Back-to-back inbox arrivals producing N duplicate summaries
    (coalesce window folds siblings into the in-flight handler's
    batch).

Tests pin every decision path:
  - Bare fire (no limits hit) → action="fire".
  - Coalesce when in-flight + within window.
  - Coalesce takes precedence over rate-limit (so a burst doesn't
    flip to rate_limit halfway through).
  - per_minute limit boundary.
  - per_hour limit boundary.
  - release() clears in-flight, allowing the next event to fire.
  - drain_coalesced() returns the siblings and clears the list.
  - warmup() seeds history from a restart.

Time is controlled by monkey-patching `time.time` to a settable
mock; that makes the test deterministic and lets us collapse a
1-hour window into milliseconds.
"""

from __future__ import annotations

import pytest

from app.agent.triggers.rate_limiter import (
    RateLimitConfig,
    TriggerRateLimiter,
    parse_rate_limit_config,
)


class _FakeClock:
    """Drop-in for `time.time`. The limiter calls `time.time()` —
    we monkey-patch the module-level reference rather than every
    call site, since the limiter only has one reference."""

    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


@pytest.fixture
def clock(monkeypatch):
    fc = _FakeClock()
    monkeypatch.setattr("app.agent.triggers.rate_limiter._now", fc)
    return fc


# ── parse_rate_limit_config ──────────────────────────────────────────


def test_parse_defaults_on_missing():
    cfg = parse_rate_limit_config(None)
    assert cfg.per_hour == 30
    assert cfg.per_minute == 5
    assert cfg.coalesce_window_sec == 120


def test_parse_overrides_present_fields():
    cfg = parse_rate_limit_config({"per_hour": 10, "per_minute": 2, "coalesce_window_sec": 30})
    assert cfg.per_hour == 10
    assert cfg.per_minute == 2
    assert cfg.coalesce_window_sec == 30


def test_parse_falls_back_to_default_on_garbage():
    cfg = parse_rate_limit_config(
        {"per_hour": -5, "per_minute": "nope", "coalesce_window_sec": None}
    )
    assert cfg.per_hour == 30
    assert cfg.per_minute == 5
    assert cfg.coalesce_window_sec == 120


# ── Gate decisions ──────────────────────────────────────────────────


def test_first_event_fires(clock):
    l = TriggerRateLimiter()
    cfg = RateLimitConfig()
    d = l.gate("t1", "e1", cfg)
    assert d.action == "fire"


def test_second_event_during_inflight_coalesces(clock):
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(coalesce_window_sec=120)
    l.gate("t1", "e1", cfg)
    # clock didn't advance; in-flight is e1; e2 arrives.
    d = l.gate("t1", "e2", cfg)
    assert d.action == "coalesce"
    assert d.parent_event_id == "e1"


def test_third_event_also_coalesces_into_same_parent(clock):
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(coalesce_window_sec=120)
    l.gate("t1", "e1", cfg)
    l.gate("t1", "e2", cfg)
    d = l.gate("t1", "e3", cfg)
    assert d.action == "coalesce"
    assert d.parent_event_id == "e1"


def test_after_coalesce_window_passes_event_rate_limits_normally(clock):
    """When the in-flight handler's wallclock exceeds
    coalesce_window_sec, new events stop folding and go through the
    rate-limit gate."""
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(coalesce_window_sec=60)
    l.gate("t1", "e1", cfg)
    clock.advance(120)  # past coalesce window
    d = l.gate("t1", "e2", cfg)
    # Not coalesced. The rate budget allows it (only 1 fire so far),
    # but we never released e1 — so e2 is treated as a fresh fire
    # (the in-flight check is purely time-window).
    assert d.action == "fire"


def test_per_minute_rate_limit_kicks_in(clock):
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(per_minute=2, per_hour=100, coalesce_window_sec=0)
    # 2 fires within the same second
    for i in range(2):
        d = l.gate("t1", f"e{i}", cfg)
        assert d.action == "fire", f"event {i} should fire"
        l.release("t1")
    # 3rd in the same minute → rate_limit
    d = l.gate("t1", "e3", cfg)
    assert d.action == "rate_limit"
    assert d.reason == "per_minute"


def test_per_minute_rate_limit_resets_after_60s(clock):
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(per_minute=2, per_hour=100, coalesce_window_sec=0)
    for i in range(2):
        l.gate("t1", f"e{i}", cfg)
        l.release("t1")
    clock.advance(70)
    d = l.gate("t1", "e3", cfg)
    assert d.action == "fire"


def test_per_hour_rate_limit_kicks_in(clock):
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(per_minute=100, per_hour=3, coalesce_window_sec=0)
    for i in range(3):
        l.gate("t1", f"e{i}", cfg)
        l.release("t1")
        # Move past per-minute window so per-minute doesn't trip first.
        clock.advance(61)
    d = l.gate("t1", "e4", cfg)
    assert d.action == "rate_limit"
    assert d.reason == "per_hour"


def test_per_hour_window_slides(clock):
    """Fires older than 1 h drop out of the count."""
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(per_minute=100, per_hour=2, coalesce_window_sec=0)
    l.gate("t1", "e1", cfg)
    l.release("t1")
    clock.advance(70)
    l.gate("t1", "e2", cfg)
    l.release("t1")
    # 2/2 used; this third would be rate_limited
    d = l.gate("t1", "e3", cfg)
    assert d.action == "rate_limit"
    # Advance past first fire's age
    clock.advance(3600)
    d = l.gate("t1", "e4", cfg)
    assert d.action == "fire"


def test_release_clears_inflight(clock):
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(per_minute=10, coalesce_window_sec=120)
    l.gate("t1", "e1", cfg)
    l.release("t1")
    # No in-flight → e2 fires (not coalesce)
    d = l.gate("t1", "e2", cfg)
    assert d.action == "fire"


def test_drain_coalesced_returns_and_clears(clock):
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(coalesce_window_sec=120)
    l.gate("t1", "e1", cfg)
    l.gate("t1", "e2", cfg)
    l.gate("t1", "e3", cfg)
    siblings = l.drain_coalesced("t1")
    assert siblings == ["e2", "e3"]
    # Second drain returns empty.
    assert l.drain_coalesced("t1") == []


def test_coalesce_takes_precedence_over_rate_limit(clock):
    """Critical invariant: a burst within the coalesce window MUST
    fold into the parent batch, even if the user is technically
    rate-limited. Otherwise the user gets nothing instead of one
    digest of N emails."""
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(per_minute=1, per_hour=1, coalesce_window_sec=120)
    l.gate("t1", "e1", cfg)
    # Hit the per_minute=1 cap. But e1 is still in-flight → coalesce.
    d = l.gate("t1", "e2", cfg)
    assert d.action == "coalesce"
    assert d.parent_event_id == "e1"


def test_warmup_seeds_history(clock):
    """Container restart: limiter is fresh in-memory but the DB has
    recent fires. `warmup` reads them in so a hot rate-limited
    trigger doesn't immediately reset its budget."""
    l = TriggerRateLimiter()
    now = clock.t
    # 30 fires in the last hour → should be at the per_hour cap.
    fires = [now - i * 50 for i in range(30)]
    l.warmup("t1", fires)
    cfg = RateLimitConfig(per_minute=100, per_hour=30, coalesce_window_sec=0)
    d = l.gate("t1", "e_after_restart", cfg)
    assert d.action == "rate_limit"
    assert d.reason == "per_hour"


def test_warmup_drops_old_entries(clock):
    """A 90-minute-old fire is outside the 1-h window — must not
    count against the budget after warmup."""
    l = TriggerRateLimiter()
    now = clock.t
    l.warmup("t1", [now - 5400])  # 90 min ago
    cfg = RateLimitConfig(per_minute=100, per_hour=1, coalesce_window_sec=0)
    d = l.gate("t1", "e_after_restart", cfg)
    assert d.action == "fire"  # window pruned the stale fire


def test_isolated_per_trigger(clock):
    """Trigger A's rate limit must not affect trigger B."""
    l = TriggerRateLimiter()
    cfg = RateLimitConfig(per_minute=1, coalesce_window_sec=0)
    l.gate("tA", "e1", cfg)
    l.release("tA")
    # tA exhausted; tB has its own budget.
    d = l.gate("tB", "e1", cfg)
    assert d.action == "fire"
