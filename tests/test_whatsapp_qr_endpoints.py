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
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

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
    """Autospec'd against the REAL channel, not a bare MagicMock.

    A bare `MagicMock()` invents any attribute you ask for, so
    `await channel.kick_pair()` returned a MagicMock and blew up with
    `TypeError: object MagicMock can't be used in 'await' expression` — the
    endpoint had grown a coroutine call the fixture never heard of. That is a
    mock drifting from its subject, and it fails as a confusing TypeError
    rather than "you forgot to stub kick_pair".

    `create_autospec` takes the surface FROM `BaileysWhatsAppChannel`: every
    method exists, coroutines come back as AsyncMock automatically, and calling
    something the class does not define is an AttributeError at the call — so
    the next method the endpoint adds cannot silently pass here.
    """
    from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel

    ch = create_autospec(BaileysWhatsAppChannel, instance=True)
    ch.health.return_value = {"session_status": session_status}
    ch.get_pairing_status.return_value = {
        "session_status": session_status,
        "connected": status_overrides.get("connected", False),
        "self_e164": status_overrides.get("self_e164"),
        "qr_data_url": status_overrides.get("qr_data_url"),
        "qr_emitted_at": status_overrides.get("qr_emitted_at"),
    }
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

    async def test_start_repairs_from_any_prior_session_state(self):
        """/start must kick a fresh pair no matter how the last session ended.

        This used to assert `force_logout()` was awaited first, with the
        reasoning "otherwise neonize would refuse to emit a fresh QR". Commit
        056eaf25 replaced neonize with the Baileys sidecar, and the teardown
        moved INTO `kick_pair()` — it POSTs /pair/start, which wipes auth and
        starts a fresh QR flow (see its docstring). So the separate
        force_logout step is gone, deliberately, and the endpoint is correct.

        The property the old test cared about is still worth pinning: a
        `logged_out` session must not be a dead end. Assert that, rather than
        the intermediate call that no longer exists.
        """
        import app.agent.channels.whatsapp_baileys as wb
        for prior in ("logged_out", "not_linked", "linked", "linking"):
            ch = _mock_channel(session_status=prior)
            with patch.object(wb, "_active_channel", ch):
                handlers = _agent_router_handlers()
                res = await handlers["/api/whatsapp/qr/start"](_user=_fake_user())
                assert res["ok"] is True, f"start failed from {prior!r}"
                ch.kick_pair.assert_awaited_once()
