"""The tunnel's settings hot-reload must reach import-time references.

Pinned incident (2026-08-25, R29-D): the platform's config push delivered
AUTOMATIONS_ENABLED=true to the founder tenant, `_reload_settings` logged
success — and `/api/automations` kept answering "Feature not available"
until a full process restart. The old reload did `get_settings.cache_clear()`
then rebound `app.config.settings` to a NEW object: every module that did
`from app.config import settings` at import (nearly all of them, including
the automations feature gate) still held the OLD object and never saw a
single reloaded value.

The fix mutates the ONE canonical Settings object in place, so every held
reference — and get_settings()'s cache — observe the new values with no
identity fork. These tests drive the REAL gate (`_flag_or_404`) through the
REAL reload entrypoint.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("AGENT_API_KEY", "test-key-hot-reload")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000000bb")


@pytest.fixture()
def _reload_env(tmp_path, monkeypatch):
    """Point _reload_settings at a controlled .env and guarantee the
    canonical settings return to their pre-test state."""
    import app.config as cfg

    env_file = tmp_path / ".env"
    env_file.write_text("AUTOMATIONS_ENABLED=true\n")
    monkeypatch.chdir(tmp_path)
    # load_dotenv(override=True) writes into os.environ; route it
    # through monkeypatch so teardown restores the var.
    monkeypatch.setenv("AUTOMATIONS_ENABLED", "false")

    before = dict(cfg.settings.__dict__)
    yield
    cfg.settings.__dict__.clear()
    cfg.settings.__dict__.update(before)


def _reload():
    from app.agent.tunnel_client import AgentTunnelClient
    # _reload_settings touches no instance state — skip __init__.
    AgentTunnelClient.__new__(AgentTunnelClient)._reload_settings()


def test_import_time_reference_sees_reloaded_value(_reload_env):
    import app.config as cfg
    from app.config import settings as held_before_reload

    object.__setattr__(cfg.settings, "automations_enabled", False)
    assert held_before_reload.automations_enabled is False

    _reload()

    assert held_before_reload.automations_enabled is True, (
        "a reference captured at import time never saw the reloaded "
        "value — the reload rebound the module attribute instead of "
        "mutating the canonical object (the founder-tenant incident)"
    )
    assert cfg.settings is held_before_reload, (
        "reload forked the settings identity — module attribute and "
        "held references now diverge on every future update"
    )
    assert cfg.get_settings() is held_before_reload, (
        "get_settings() serves a different object than import-time "
        "references hold"
    )


def test_the_actual_automations_gate_opens_on_reload(_reload_env):
    """The exact symptom: /api/automations' _flag_or_404 kept raising
    after a successful-looking reload."""
    from fastapi import HTTPException
    import app.config as cfg
    from app.api.automations import _flag_or_404

    object.__setattr__(cfg.settings, "automations_enabled", False)
    with pytest.raises(HTTPException):
        _flag_or_404()

    _reload()

    _flag_or_404()  # must NOT raise — the gate sees the pushed flag
