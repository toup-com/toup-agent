"""Unit tests for app.api.whatsapp_qr (agent-side QR endpoints).

Verifies the three thin routes correctly:
* surface 503 when no Baileys channel is active (so the UI can guide
  the user to save whatsapp_mode='qr_link' first);
* delegate to the live channel when present;
* /qr/start with a previous logged_out state force-logouts before
  re-pairing.

Pytest-fixture-free so it works under both real pytest and the
direct-execution harness used by the other whatsapp test files.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, HTTPException


def _agent_router_handlers():
    """Build a FastAPI app, mount the QR router, return a dict keyed
    by path → handler (async callable)."""
    from app.api.whatsapp_qr import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    out = {}
    for r in app.routes:
        path = getattr(r, "path", None)
        endpoint = getattr(r, "endpoint", None)
        if path and endpoint:
            out[path] = endpoint
    return out


def _fake_user():
    return MagicMock(id="u-test")


def _mock_channel(session_status: str = "not_linked", **status_overrides):
    ch = MagicMock()
    ch.health = MagicMock(return_value={"session_status": session_status})
    ch.get_pairing_status = MagicMock(return_value={
        "session_status": session_status,
        "connected": status_overrides.get("connected", False),
        "self_e164": status_overrides.get("self_e164"),
        "qr_data_url": status_overrides.get("qr_data_url"),
        "qr_emitted_at": status_overrides.get("qr_emitted_at"),
    })
    ch.force_logout = AsyncMock()
    return ch


# ── 503 path: no active Baileys channel ──────────────────────────


class TestNoActiveChannel:
    async def test_start_503(self):
        import app.agent.channels.whatsapp_baileys as wb
        with patch.object(wb, "_active_channel", None):
            handlers = _agent_router_handlers()
            try:
                await handlers["/api/whatsapp/qr/start"](_user=_fake_user())
                assert False, "expected HTTPException"
            except HTTPException as exc:
                assert exc.status_code == 503
                assert "not active" in exc.detail.lower()

    async def test_status_503(self):
        import app.agent.channels.whatsapp_baileys as wb
        with patch.object(wb, "_active_channel", None):
            handlers = _agent_router_handlers()
            try:
                await handlers["/api/whatsapp/qr/status"](_user=_fake_user())
                assert False, "expected HTTPException"
            except HTTPException as exc:
                assert exc.status_code == 503

    async def test_logout_503(self):
        import app.agent.channels.whatsapp_baileys as wb
        with patch.object(wb, "_active_channel", None):
            handlers = _agent_router_handlers()
            try:
                await handlers["/api/whatsapp/qr/logout"](_user=_fake_user())
                assert False, "expected HTTPException"
            except HTTPException as exc:
                assert exc.status_code == 503


# ── Active channel: delegates correctly ──────────────────────────


class TestActiveChannel:
    async def test_status_returns_snapshot(self):
        import app.agent.channels.whatsapp_baileys as wb
        ch = _mock_channel(
            session_status="linking",
            qr_data_url="data:image/png;base64,FAKE",
        )
        with patch.object(wb, "_active_channel", ch):
            handlers = _agent_router_handlers()
            res = await handlers["/api/whatsapp/qr/status"](_user=_fake_user())
            assert res["session_status"] == "linking"
            assert res["qr_data_url"] == "data:image/png;base64,FAKE"
            ch.get_pairing_status.assert_called_once()

    async def test_logout_calls_force_logout(self):
        import app.agent.channels.whatsapp_baileys as wb
        ch = _mock_channel()
        with patch.object(wb, "_active_channel", ch):
            handlers = _agent_router_handlers()
            res = await handlers["/api/whatsapp/qr/logout"](_user=_fake_user())
            assert res["ok"] is True
            assert res["session_status"] == "not_linked"
            ch.force_logout.assert_awaited_once()

    async def test_start_idempotent_when_idle(self):
        """Starting from not_linked just returns ok — supervisor is
        already running and will emit a QR shortly."""
        import app.agent.channels.whatsapp_baileys as wb
        ch = _mock_channel(session_status="not_linked")
        with patch.object(wb, "_active_channel", ch):
            handlers = _agent_router_handlers()
            res = await handlers["/api/whatsapp/qr/start"](_user=_fake_user())
            assert res["ok"] is True
            ch.force_logout.assert_not_called()

    async def test_start_force_logouts_when_logged_out(self):
        """If the previous session terminated as logged_out, start()
        clears it before re-pairing — otherwise neonize would refuse
        to emit a fresh QR."""
        import app.agent.channels.whatsapp_baileys as wb
        ch = _mock_channel(session_status="logged_out")
        with patch.object(wb, "_active_channel", ch):
            handlers = _agent_router_handlers()
            res = await handlers["/api/whatsapp/qr/start"](_user=_fake_user())
            assert res["ok"] is True
            ch.force_logout.assert_awaited_once()
