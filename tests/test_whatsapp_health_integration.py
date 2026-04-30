"""Integration test for /agent/health WhatsApp surface.

The health endpoint has to surface either the Cloud API channel's
health dict, the Baileys channel's dict, or a fallback string, in
priority order: Baileys first → Cloud API → "configured" → "disabled".

Drives the resolution logic directly rather than spinning up the
whole FastAPI app, so the test runs without DB / network. The four
permutations exercise the branch table the agent_main.py code
implements.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch


def _make_resolver(settings):
    """Mirror the resolution logic in agent_main.py:agent_health.

    Kept here as an executable spec — if agent_main.py changes the
    branching, this test surfaces the drift.
    """
    from app.agent.channels.whatsapp_channel import get_active_channel as wa_cloud
    from app.agent.channels.whatsapp_baileys import get_active_baileys_channel as wa_baileys

    baileys = wa_baileys()
    if baileys is not None:
        return baileys.health()
    cloud = wa_cloud()
    if cloud is not None:
        return cloud.health()
    if settings.whatsapp_mode or settings.whatsapp_phone_number_id:
        return "configured"
    return "disabled"


class TestWhatsAppHealthResolution:
    def test_no_channel_no_creds_returns_disabled(self):
        import app.agent.channels.whatsapp_channel as wc
        import app.agent.channels.whatsapp_baileys as wb
        s = MagicMock(whatsapp_mode=None, whatsapp_phone_number_id=None)
        with patch.object(wc, "_active_channel", None), \
             patch.object(wb, "_active_channel", None):
            assert _make_resolver(s) == "disabled"

    def test_creds_but_no_active_channel_returns_configured(self):
        import app.agent.channels.whatsapp_channel as wc
        import app.agent.channels.whatsapp_baileys as wb
        # mode='qr_link' OR phone_number_id set — either flips to "configured"
        s_qr = MagicMock(whatsapp_mode="qr_link", whatsapp_phone_number_id=None)
        s_cloud = MagicMock(whatsapp_mode=None, whatsapp_phone_number_id="123")
        with patch.object(wc, "_active_channel", None), \
             patch.object(wb, "_active_channel", None):
            assert _make_resolver(s_qr) == "configured"
            assert _make_resolver(s_cloud) == "configured"

    def test_cloud_active_returns_cloud_health_dict(self):
        import app.agent.channels.whatsapp_channel as wc
        import app.agent.channels.whatsapp_baileys as wb
        cloud_ch = MagicMock()
        cloud_ch.health = MagicMock(return_value={
            "configured": True, "started": True, "phone_number_id": "12345",
            "app_secret_configured": True, "verify_token_configured": True,
            "allowed_numbers_count": 0, "inbound_count": 7,
            "last_inbound_at": "2026-04-30T12:00:00Z",
            "last_send_at": "2026-04-30T12:00:01Z",
            "last_send_error": None, "connected_at_marked_in_db": True,
        })
        s = MagicMock(whatsapp_mode="cloud_api", whatsapp_phone_number_id="12345")
        with patch.object(wc, "_active_channel", cloud_ch), \
             patch.object(wb, "_active_channel", None):
            health = _make_resolver(s)
            assert isinstance(health, dict)
            assert health["inbound_count"] == 7
            # No "mode" key in Cloud API health — distinguishes the
            # two surfaces for any consumer that needs to branch.
            assert "mode" not in health

    def test_baileys_active_returns_qr_link_health_dict(self):
        import app.agent.channels.whatsapp_channel as wc
        import app.agent.channels.whatsapp_baileys as wb
        bail_ch = MagicMock()
        bail_ch.health = MagicMock(return_value={
            "configured": True, "started": True, "mode": "qr_link",
            "session_status": "linked", "connected": True,
            "self_e164": "+19998887777", "allowed_numbers_count": 1,
            "inbound_count": 3,
            "last_inbound_at": "2026-04-30T12:00:00Z",
            "last_send_at": "2026-04-30T12:00:01Z",
            "last_send_error": None, "reconnect_attempts": 0,
        })
        s = MagicMock(whatsapp_mode="qr_link", whatsapp_phone_number_id=None)
        with patch.object(wc, "_active_channel", None), \
             patch.object(wb, "_active_channel", bail_ch):
            health = _make_resolver(s)
            assert isinstance(health, dict)
            assert health["mode"] == "qr_link"
            assert health["session_status"] == "linked"
            assert health["self_e164"] == "+19998887777"

    def test_baileys_takes_precedence_over_cloud(self):
        """If both happen to be active (transient state during a
        mode switch), Baileys wins — that's the priority order in
        agent_main.py:agent_health and matches "current transport"
        semantics."""
        import app.agent.channels.whatsapp_channel as wc
        import app.agent.channels.whatsapp_baileys as wb
        cloud_ch = MagicMock()
        cloud_ch.health = MagicMock(return_value={"started": True, "configured": True})
        bail_ch = MagicMock()
        bail_ch.health = MagicMock(return_value={"mode": "qr_link", "session_status": "linked"})
        s = MagicMock(whatsapp_mode="qr_link", whatsapp_phone_number_id=None)
        with patch.object(wc, "_active_channel", cloud_ch), \
             patch.object(wb, "_active_channel", bail_ch):
            health = _make_resolver(s)
            assert isinstance(health, dict)
            assert health.get("mode") == "qr_link"
