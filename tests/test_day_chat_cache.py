"""Regression tests for TKT-LAT-011 — day_chat_id (user_id, local_date) cache."""

from __future__ import annotations

from datetime import date

from app.agent import _day_chat_cache as dcc


def _reset():
    dcc._CACHE.clear()


def test_miss_returns_none():
    _reset()
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 21)) is None


def test_set_then_get_returns_id():
    _reset()
    dcc.set_cached_day_chat_id("u1", date(2026, 5, 21), "dc-abc")
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 21)) == "dc-abc"


def test_set_empty_id_is_no_op():
    _reset()
    dcc.set_cached_day_chat_id("u1", date(2026, 5, 21), "")
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 21)) is None


def test_different_date_is_different_key():
    _reset()
    dcc.set_cached_day_chat_id("u1", date(2026, 5, 21), "dc-21")
    dcc.set_cached_day_chat_id("u1", date(2026, 5, 22), "dc-22")
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 21)) == "dc-21"
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 22)) == "dc-22"


def test_invalidate_single_entry():
    _reset()
    dcc.set_cached_day_chat_id("u1", date(2026, 5, 21), "dc-21")
    dcc.invalidate_cached_day_chat_id("u1", date(2026, 5, 21))
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 21)) is None


def test_invalidate_user_drops_all_dates():
    _reset()
    dcc.set_cached_day_chat_id("u1", date(2026, 5, 21), "a")
    dcc.set_cached_day_chat_id("u1", date(2026, 5, 22), "b")
    dcc.set_cached_day_chat_id("u2", date(2026, 5, 21), "c")
    dcc.invalidate_user("u1")
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 21)) is None
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 22)) is None
    # Other users untouched
    assert dcc.get_cached_day_chat_id("u2", date(2026, 5, 21)) == "c"


def test_lru_eviction_at_capacity(monkeypatch):
    _reset()
    monkeypatch.setattr(dcc, "_MAX_ENTRIES", 3)
    dcc.set_cached_day_chat_id("u1", date(2026, 1, 1), "1")
    dcc.set_cached_day_chat_id("u1", date(2026, 1, 2), "2")
    dcc.set_cached_day_chat_id("u1", date(2026, 1, 3), "3")
    dcc.set_cached_day_chat_id("u1", date(2026, 1, 4), "4")
    # Oldest evicted
    assert dcc.get_cached_day_chat_id("u1", date(2026, 1, 1)) is None
    assert dcc.get_cached_day_chat_id("u1", date(2026, 1, 4)) == "4"


def test_combined_tz_invalidate_drops_day_chat_cache():
    """TKT-LAT-011: tz update sites that call the combined hook must
    also clear the user's day_chat entries (tz change can shift
    local_date)."""
    from app.agent import _user_tz_cache as tz_cache
    _reset()
    tz_cache._USER_TZ_CACHE.clear()
    tz_cache.set_cached_user_tz("u1", "America/Toronto")
    dcc.set_cached_day_chat_id("u1", date(2026, 5, 21), "dc-21")
    tz_cache.invalidate_cached_user_tz_with_day_chat("u1")
    assert tz_cache.get_cached_user_tz("u1") is None
    assert dcc.get_cached_day_chat_id("u1", date(2026, 5, 21)) is None
