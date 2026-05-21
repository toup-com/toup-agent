"""Regression tests for TKT-LAT-004 — User.timezone TTL cache.

Exercises the cache helpers and their TTL/invalidation semantics. The
helpers live in ``app.agent._user_tz_cache`` (config-free) so this test
module doesn't transitively boot Settings.
"""

from __future__ import annotations

import time

from app.agent import _user_tz_cache as tz_cache


def _reset_cache() -> None:
    tz_cache._USER_TZ_CACHE.clear()


def test_cache_miss_returns_none():
    _reset_cache()
    assert tz_cache.get_cached_user_tz("nobody") is None


def test_set_then_get_returns_cached_value():
    _reset_cache()
    tz_cache.set_cached_user_tz("u1", "America/Toronto")
    assert tz_cache.get_cached_user_tz("u1") == "America/Toronto"


def test_set_none_is_no_op():
    _reset_cache()
    tz_cache.set_cached_user_tz("u1", None)
    assert tz_cache.get_cached_user_tz("u1") is None


def test_invalidate_drops_entry():
    _reset_cache()
    tz_cache.set_cached_user_tz("u1", "Europe/Berlin")
    assert tz_cache.get_cached_user_tz("u1") == "Europe/Berlin"
    tz_cache.invalidate_cached_user_tz("u1")
    assert tz_cache.get_cached_user_tz("u1") is None


def test_invalidate_unknown_user_is_safe():
    _reset_cache()
    tz_cache.invalidate_cached_user_tz("never-cached")  # must not raise


def test_expired_entry_is_dropped_lazily(monkeypatch):
    _reset_cache()
    tz_cache.set_cached_user_tz("u1", "UTC")
    fake_now = time.monotonic() + tz_cache._USER_TZ_TTL_S + 1.0
    monkeypatch.setattr(tz_cache.time, "monotonic", lambda: fake_now)
    assert tz_cache.get_cached_user_tz("u1") is None
    assert "u1" not in tz_cache._USER_TZ_CACHE


def test_cache_is_per_user():
    _reset_cache()
    tz_cache.set_cached_user_tz("u1", "America/Toronto")
    tz_cache.set_cached_user_tz("u2", "Europe/Berlin")
    assert tz_cache.get_cached_user_tz("u1") == "America/Toronto"
    assert tz_cache.get_cached_user_tz("u2") == "Europe/Berlin"
    tz_cache.invalidate_cached_user_tz("u1")
    assert tz_cache.get_cached_user_tz("u1") is None
    assert tz_cache.get_cached_user_tz("u2") == "Europe/Berlin"


def test_agent_runner_reexports_match():
    """The agent_runner private re-exports must point at the same callables."""
    from app.agent import agent_runner

    assert agent_runner._get_cached_user_tz is tz_cache.get_cached_user_tz
    assert agent_runner._set_cached_user_tz is tz_cache.set_cached_user_tz
    assert agent_runner._invalidate_cached_user_tz is tz_cache.invalidate_cached_user_tz
