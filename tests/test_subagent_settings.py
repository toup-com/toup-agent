"""Sub-agent spawning settings.

Phase 1 of the sub-agent spawning arc. The settings are read by the
Phase 2 dispatcher and Phase 6 metrics, so a default-value drift here
silently changes spawn behavior on every deploy. Pin them.

What we pin:

  1. Spawning is OFF by default (kill switch). Operator must flip
     ``SUBAGENT_SPAWNING_ENABLED=true`` per environment after the
     smoke matrix passes.
  2. Default caps match the v1 plan (depth=1, 5 per parent,
     3 concurrent per user, 50/24h).
  3. Credit multiplier is 1.0 (no implicit upcharge or discount).
  4. Timeout defaults are within the ceiling
     (300s default, 900s ceiling).
  5. Orphan-sweep threshold matches the triggers/runner.py
     ORPHAN_THRESHOLD shape (minutes).
  6. Every setting can be overridden via env vars (the deployment
     surface — flipping the kill switch via SUBAGENT_SPAWNING_ENABLED
     must work without code edit).
"""
from __future__ import annotations

import importlib
import sys


def _reload_settings(monkeypatch, **env: str):
    """Reload app.config under a fresh env so pydantic-settings
    re-reads SUBAGENT_* env vars. Pattern matches the existing
    test_migration_*.py reload dance."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-subagent")
    monkeypatch.setenv("ENCRYPTION_KEY", "test-32-byte-encryption-key--x12")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy_subagent")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])
    from app.config import get_settings
    get_settings.cache_clear()
    return get_settings()


def test_spawning_disabled_by_default(monkeypatch):
    s = _reload_settings(monkeypatch)
    assert s.subagent_spawning_enabled is False, (
        "kill switch must default OFF — operator flips per environment"
    )


def test_default_caps_match_v1_plan(monkeypatch):
    s = _reload_settings(monkeypatch)
    assert s.subagent_max_depth == 1
    assert s.subagent_max_children_per_parent == 5
    assert s.subagent_max_per_user_concurrent == 3
    assert s.subagent_max_per_user_24h == 50


def test_credit_multiplier_is_unity_by_default(monkeypatch):
    s = _reload_settings(monkeypatch)
    assert s.subagent_credit_multiplier == 1.0


def test_timeout_defaults_within_ceiling(monkeypatch):
    s = _reload_settings(monkeypatch)
    assert s.subagent_default_timeout_seconds == 300
    assert s.subagent_max_timeout_seconds == 900
    assert (
        s.subagent_default_timeout_seconds <= s.subagent_max_timeout_seconds
    ), "default must be ≤ ceiling"


def test_orphan_sweep_threshold_matches_minutes_convention(monkeypatch):
    s = _reload_settings(monkeypatch)
    assert s.subagent_orphan_sweep_threshold_minutes == 10


def test_kill_switch_can_be_flipped_via_env(monkeypatch):
    s_on = _reload_settings(monkeypatch, SUBAGENT_SPAWNING_ENABLED="true")
    assert s_on.subagent_spawning_enabled is True

    s_off = _reload_settings(monkeypatch, SUBAGENT_SPAWNING_ENABLED="false")
    assert s_off.subagent_spawning_enabled is False


def test_caps_can_be_overridden_via_env(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        SUBAGENT_MAX_DEPTH="3",
        SUBAGENT_MAX_CHILDREN_PER_PARENT="7",
        SUBAGENT_MAX_PER_USER_CONCURRENT="2",
        SUBAGENT_MAX_PER_USER_24H="100",
        SUBAGENT_CREDIT_MULTIPLIER="1.5",
        SUBAGENT_DEFAULT_TIMEOUT_SECONDS="120",
        SUBAGENT_MAX_TIMEOUT_SECONDS="600",
        SUBAGENT_ORPHAN_SWEEP_THRESHOLD_MINUTES="15",
    )
    assert s.subagent_max_depth == 3
    assert s.subagent_max_children_per_parent == 7
    assert s.subagent_max_per_user_concurrent == 2
    assert s.subagent_max_per_user_24h == 100
    assert s.subagent_credit_multiplier == 1.5
    assert s.subagent_default_timeout_seconds == 120
    assert s.subagent_max_timeout_seconds == 600
    assert s.subagent_orphan_sweep_threshold_minutes == 15


def test_supervisor_defaults(monkeypatch):
    """D-P6: the supervisor is off image-wide, staged by allowlist; executor
    and supervisor models default to the user's own model; 3 workers / 240 s."""
    s = _reload_settings(monkeypatch)
    assert s.supervisor_enabled is False
    assert s.supervisor_user_allowlist == ""
    assert s.subagent_model is None and s.supervisor_model is None
    assert s.supervisor_max_workers == 3
    assert s.supervisor_timeout_seconds == 240
    assert s.tool_timeout_overrides["delegate"] > s.supervisor_timeout_seconds


def test_supervisor_settings_hot_reload():
    from app.agent.config_reload import RELOADABLE_FIELDS
    assert {"supervisor_enabled", "supervisor_user_allowlist", "subagent_model",
            "supervisor_model", "supervisor_max_workers",
            "supervisor_timeout_seconds"} <= RELOADABLE_FIELDS


def _dockerfile_allowlist(path):
    import re
    from pathlib import Path

    text = Path(path).read_text()
    found = re.findall(r'^ENV SUPERVISOR_USER_ALLOWLIST="([^"]*)"\s*$', text, re.M)
    return found


def test_the_relay_and_the_tenant_image_carry_one_supervisor_allowlist():
    """Voice round 2026-10-06, D-T2. The gate is read in TWO processes: the
    tenant image (may this turn call `delegate`?) and platform-api (the voice
    relay applies a tenant's `supervisor_model` only when the gate is on for
    it). Only the agent image carried the allowlist, so the relay's gate was
    OFF for exactly the accounts whose turns could delegate. Both Dockerfiles
    carry the line exactly once and byte-identical."""
    from pathlib import Path

    backend = Path(__file__).resolve().parent.parent
    agent = _dockerfile_allowlist(backend / "Dockerfile.agent")
    platform = _dockerfile_allowlist(backend.parent / "Dockerfile")
    assert len(agent) == 1 and len(platform) == 1, (agent, platform)
    assert agent == platform
    ids = [u for u in agent[0].split(",") if u]
    assert "871bac24-c366-42b5-b224-8802c73aef3a" in ids
    assert "c47c5b4b-16d2-498c-b231-15d53d7454b9" in ids
