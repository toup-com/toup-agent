"""Regression test for TKT-LAT-008 — portrait cache TTL bump."""

from __future__ import annotations

import inspect

from app.services import user_portrait_service as ups


def test_portrait_cache_ttl_is_six_hours():
    assert ups._CACHE_TTL_SECONDS == 6 * 3600


def test_get_or_build_portrait_emits_cache_metric():
    """[PERF] portrait_cache=... must be logged on all three branches."""
    src = inspect.getsource(ups.UserPortraitService.get_or_build_portrait)
    assert "portrait_cache=hit" in src
    assert "portrait_cache=stale" in src
    assert "portrait_cache=miss" in src


def test_get_or_build_portrait_never_awaits_llm_directly():
    """The hot path must NOT call build_portrait / synthesis directly."""
    src = inspect.getsource(ups.UserPortraitService.get_or_build_portrait)
    assert "await self.build_portrait" not in src
    assert "await self._synthesize" not in src
    # Rebuild scheduling pattern is preserved.
    assert "_schedule_background_rebuild" in src
