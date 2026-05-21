"""Regression test for TKT-LAT-013 — page-context readable_content trim.

The render function should:
  * Default to the compact cap (2000 chars) when settings flag absent.
  * Honor the flag if a settings module overrides it at runtime.
"""

from __future__ import annotations

import sys

from app.services import page_context_render as pcr


def test_compact_cap_is_two_thousand():
    assert pcr._LIMITS_COMPACT["readable_content"] == 2000


def test_verbose_cap_preserved_for_opt_in():
    assert pcr._LIMITS_VERBOSE["readable_content"] == 8000


def test_render_truncates_to_compact_cap_by_default():
    payload = {
        "url": "https://example.com",
        "title": "x",
        "readable_content": "a" * 9000,
    }
    out = pcr.render_page_context(payload)
    # Compact cap is 2000 chars; ellipsis = 1 char, so body line length
    # at-or-under cap.
    body_line = next((line for line in out.splitlines() if line.startswith("  a")), None)
    assert body_line is not None
    # The truncated body is at most cap + ellipsis chars.
    assert len(body_line) <= 2000 + 2


def test_render_honors_verbose_flag_when_set(monkeypatch):
    # Stand up a tiny Settings stub on sys.modules['app.config'] so the
    # function's runtime import resolves to our override.
    class _FakeSettings:
        extension_page_context_compact = False

    import app.config as real_config
    monkeypatch.setattr(real_config, "settings", _FakeSettings(), raising=False)
    payload = {
        "url": "https://example.com",
        "title": "x",
        "readable_content": "b" * 9000,
    }
    out = pcr.render_page_context(payload)
    body_lines = [line for line in out.splitlines() if line.startswith("  b")]
    # In verbose mode the cap is 8000.
    assert sum(len(l) for l in body_lines) > 2000
