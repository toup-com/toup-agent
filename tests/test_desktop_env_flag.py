"""Connections rollout reaches config transports without enabling other tenants."""
from types import SimpleNamespace
from unittest.mock import patch
from app.config import Settings, settings
from app.api.agent_setup import _build_env, _desktop_env_flag
from app.db.database import async_session_maker
from app.db.models import AgentConfig, User
from app.services.feature_flags import FLAGS, set_allowlist, set_rollout_pct
from app.services.ssh_deploy_service import generate_env_content
from test_env_render_automations import _cfg


def test_env_delivery_is_opt_in_and_preserves_dark_render():
    off = generate_env_content("owner", "key")
    assert generate_env_content("owner", "key", desktop_relay_enabled=False) == off
    on = generate_env_content("owner", "key", desktop_relay_enabled=True)
    assert on.replace("\n# --- Mac Connections ---\nDESKTOP_RELAY_ENABLED=true\n", "") == off
    assert "DESKTOP_RELAY_ENABLED=true" in _build_env(_cfg(), "owner", desktop_relay_enabled=True)
    assert "DESKTOP" not in _build_env(_cfg(), "other")
    assert Settings.model_fields["desktop_relay_enabled"].default is False
    assert Settings.model_fields["desktop_connections_rollout_pct"].default == 0
    assert FLAGS["desktop_connections"].salt == "desktop_connections"


async def test_allowlist_reaches_pool_bind_and_removes_access():
    from app.services.pool_service import _build_bind_payload
    from app.services.runtime_identity import _PAYLOAD_TO_SETTING
    from app.api.admin_pool import _BIND_FIELDS
    async with async_session_maker() as db:
        for uid in ("listed", "other"):
            db.add(User(id=uid, email=f"{uid}@example.com", hashed_password=""))
            db.add(AgentConfig(user_id=uid, agent_api_key=f"key-{uid}"))
        await db.commit()
        await set_rollout_pct(db, "desktop_connections", 0)
        await set_allowlist(db, "desktop_connections", ["listed"])
        for uid, expected in (("listed", True), ("other", False)):
            assert await _desktop_env_flag(uid) is expected
            payload = await _build_bind_payload(db, uid, SimpleNamespace(agent_api_key=f"key-{uid}"))
            assert payload["desktop_relay_enabled"] is expected
        await set_allowlist(db, "desktop_connections", [])
        payload = await _build_bind_payload(db, "listed", SimpleNamespace(agent_api_key="key-listed"))
        assert payload["desktop_relay_enabled"] is False
    assert "desktop_relay_enabled" in _BIND_FIELDS
    assert _PAYLOAD_TO_SETTING["desktop_relay_enabled"] == "desktop_relay_enabled"


async def test_flag_lookup_failure_does_not_enable_access():
    with patch("app.services.feature_flags.is_enabled", side_effect=RuntimeError("down")):
        assert await _desktop_env_flag("owner") is False


def test_pool_runtime_flag_supports_enable_and_disable(monkeypatch):
    from app.services.runtime_identity import apply_to_settings
    monkeypatch.setattr(settings, "desktop_relay_enabled", False)
    monkeypatch.setenv("DESKTOP_RELAY_ENABLED", "false")
    apply_to_settings({"desktop_relay_enabled": True})
    assert settings.desktop_relay_enabled is True
    apply_to_settings({"desktop_relay_enabled": False})
    assert settings.desktop_relay_enabled is False
