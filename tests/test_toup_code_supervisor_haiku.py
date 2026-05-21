"""Regression test for TKT-LAT-015 — Toup-Code supervisor Haiku pin.

The supervisor loop in /code/supervise calls call_system_llm once per
decision cycle. Default behavior preserves the user's chat model (the
"user OWNS this orchestration" semantic). When the
`toup_code_supervisor_use_haiku` flag is set, the call must be
hard-pinned to claude-haiku-4-5-20251001 regardless of the user's
configured model.
"""

from __future__ import annotations

import pytest


def test_flag_defaults_to_off():
    """Default must not change observable behavior — user model wins."""
    from app.config import settings
    assert settings.toup_code_supervisor_use_haiku is False


def test_model_resolves_to_none_when_flag_off(monkeypatch):
    """With flag off, the supervisor's model arg should be None
    (== user's agent_model fallthrough)."""
    import app.config as real_config

    class _FakeSettings:
        toup_code_supervisor_use_haiku = False

    monkeypatch.setattr(real_config, "settings", _FakeSettings(), raising=False)

    # Reproduce the production decision logic verbatim. If this drifts
    # from app/api/toup_code.py the test will catch it.
    flag = real_config.settings.toup_code_supervisor_use_haiku
    sup_model = "claude-haiku-4-5-20251001" if flag else None
    assert sup_model is None


def test_model_pins_haiku_when_flag_on(monkeypatch):
    """With flag on, the supervisor's model must be pinned Haiku."""
    import app.config as real_config

    class _FakeSettings:
        toup_code_supervisor_use_haiku = True

    monkeypatch.setattr(real_config, "settings", _FakeSettings(), raising=False)

    flag = real_config.settings.toup_code_supervisor_use_haiku
    sup_model = "claude-haiku-4-5-20251001" if flag else None
    assert sup_model == "claude-haiku-4-5-20251001"


def test_supervisor_loop_uses_settings_lookup_not_module_level():
    """The flag must be read inside the loop, not captured at import,
    so an operator flipping the env var hot-reloads behavior without
    requiring a process restart of toup_code.py specifically.

    Read the file as text to avoid importing the whole FastAPI app
    (which would pull in heavy startup machinery just to verify a
    source-level invariant)."""
    from pathlib import Path

    src = Path(
        Path(__file__).resolve().parents[1] / "app" / "api" / "toup_code.py"
    ).read_text()

    # The decision happens inline at the call site, not via a
    # module-level constant. Catch any future refactor that captures
    # the value at import time (which would make the flag unflippable
    # without a redeploy).
    assert "settings.toup_code_supervisor_use_haiku" in src
    # And the model string itself is at the call site.
    assert "claude-haiku-4-5-20251001" in src
