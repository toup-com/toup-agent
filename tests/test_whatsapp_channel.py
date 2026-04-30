"""Unit tests for app.agent.channels.whatsapp_channel.

Exercises the bits of the channel that don't need a real Meta account
or a database:

* HMAC verification path (valid, invalid, missing app_secret).
* One-shot warning when an unsigned webhook is accepted.
* Health surface lifecycle (pre-start, post-start, after a send).
* Active-channel reference identity-aware ``stop()``.
* Outbound retry + send-error tracker.

End-to-end inbound (claim→dispatch→reply) is covered indirectly by
the message-handler tests; here we focus on the channel's own
responsibilities.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI


def _capture_warnings():
    """Return (buf, attach) — buf collects warnings, attach plugs the
    handler into the channel module's logger.
    """
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.WARNING)
    logging.getLogger("app.agent.channels.whatsapp_channel").addHandler(h)
    logging.getLogger("app.agent.channels.whatsapp_channel").setLevel(logging.WARNING)
    return buf, h


def _detach(handler):
    logging.getLogger("app.agent.channels.whatsapp_channel").removeHandler(handler)


def _post_handler(ch):
    """Locate the POST /api/whatsapp/webhook handler registered by ch."""
    app = FastAPI()
    ch.register_routes(app)
    for route in app.routes:
        if (
            getattr(route, "path", "") == "/api/whatsapp/webhook"
            and "POST" in getattr(route, "methods", set())
        ):
            return route.endpoint
    raise AssertionError("POST /api/whatsapp/webhook not registered")


# ── HMAC verification ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hmac_valid_signature_passes():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(
        phone_number_id="ph", access_token="tok", verify_token="v",
        app_secret="sekrit",
    )
    handler = _post_handler(ch)

    body = json.dumps({"entry": []}).encode()
    sig = "sha256=" + hmac.new(b"sekrit", body, hashlib.sha256).hexdigest()
    req = MagicMock()
    req.body = AsyncMock(return_value=body)
    req.headers = {"X-Hub-Signature-256": sig}

    resp = await handler(req)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_hmac_invalid_signature_rejected_and_warns():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(
        phone_number_id="ph", access_token="tok", verify_token="v",
        app_secret="sekrit",
    )
    handler = _post_handler(ch)

    buf, h = _capture_warnings()
    try:
        req = MagicMock()
        req.body = AsyncMock(return_value=b'{"entry":[]}')
        req.headers = {"X-Hub-Signature-256": "sha256=garbage"}
        resp = await handler(req)
        assert resp.status_code == 403
        assert "webhook.signature_invalid" in buf.getvalue()
    finally:
        _detach(h)


@pytest.mark.asyncio
async def test_hmac_missing_header_rejected_when_secret_set():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(
        phone_number_id="ph", access_token="tok", verify_token="v",
        app_secret="sekrit",
    )
    handler = _post_handler(ch)

    req = MagicMock()
    req.body = AsyncMock(return_value=b'{"entry":[]}')
    req.headers = {}  # no signature
    resp = await handler(req)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_unsigned_webhook_warns_once_then_silent():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    # No app_secret => unsigned path.
    ch = WhatsAppChannel(
        phone_number_id="ph", access_token="tok", verify_token="v",
    )
    handler = _post_handler(ch)

    buf, h = _capture_warnings()
    try:
        for _ in range(3):
            req = MagicMock()
            req.body = AsyncMock(return_value=b'{"entry":[]}')
            req.headers = {}
            resp = await handler(req)
            assert resp.status_code == 200
        assert buf.getvalue().count("webhook.unsigned_accepted") == 1
    finally:
        _detach(h)


# ── GET handshake ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_handshake_correct_token_returns_challenge():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(
        phone_number_id="ph", access_token="tok", verify_token="my-verify-token",
    )
    app = FastAPI()
    ch.register_routes(app)
    get_route = next(
        r for r in app.routes
        if getattr(r, "path", "") == "/api/whatsapp/webhook"
        and "GET" in getattr(r, "methods", set())
    )

    req = MagicMock()
    req.query_params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "my-verify-token",
        "hub.challenge": "OPAQUE_VALUE",
    }
    resp = await get_route.endpoint(req)
    assert resp.status_code == 200
    assert resp.body == b"OPAQUE_VALUE"


@pytest.mark.asyncio
async def test_get_handshake_wrong_token_returns_403():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(
        phone_number_id="ph", access_token="tok", verify_token="correct",
    )
    app = FastAPI()
    ch.register_routes(app)
    get_route = next(
        r for r in app.routes
        if getattr(r, "path", "") == "/api/whatsapp/webhook"
        and "GET" in getattr(r, "methods", set())
    )

    req = MagicMock()
    req.query_params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "wrong",
        "hub.challenge": "OPAQUE",
    }
    resp = await get_route.endpoint(req)
    assert resp.status_code == 403


# ── health() lifecycle ───────────────────────────────────────────


def test_health_pre_start_shape():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(phone_number_id="ph", access_token="tok", verify_token="v")
    h = ch.health()
    assert h["configured"] is True
    assert h["started"] is False
    assert h["app_secret_configured"] is False
    assert h["verify_token_configured"] is True
    assert h["inbound_count"] == 0
    assert h["last_inbound_at"] is None
    assert h["last_send_at"] is None
    assert h["last_send_error"] is None
    assert h["connected_at_marked_in_db"] is False


@pytest.mark.asyncio
async def test_health_post_start_and_active_ref():
    from app.agent.channels.whatsapp_channel import (
        WhatsAppChannel,
        get_active_channel,
    )

    ch = WhatsAppChannel(phone_number_id="ph", access_token="tok", verify_token="v")
    await ch.start()
    try:
        assert get_active_channel() is ch
        h = ch.health()
        assert h["started"] is True
    finally:
        # Properly mock aclose so stop() doesn't raise.
        ch._http = MagicMock()
        ch._http.aclose = AsyncMock()
        await ch.stop()
    assert get_active_channel() is None


@pytest.mark.asyncio
async def test_stop_only_clears_self_not_other_channel():
    """If two channels exist (one stopped, one started later), stopping
    the first must not clear the active reference owned by the second.
    """
    from app.agent.channels.whatsapp_channel import (
        WhatsAppChannel,
        get_active_channel,
    )

    ch1 = WhatsAppChannel(phone_number_id="ph1", access_token="tok", verify_token="v")
    await ch1.start()
    ch1._http = MagicMock(); ch1._http.aclose = AsyncMock()

    ch2 = WhatsAppChannel(phone_number_id="ph2", access_token="tok", verify_token="v")
    await ch2.start()  # now ch2 is the active one
    ch2._http = MagicMock(); ch2._http.aclose = AsyncMock()

    await ch1.stop()
    assert get_active_channel() is ch2
    await ch2.stop()
    assert get_active_channel() is None


# ── Outbound retry + tracker ─────────────────────────────────────


@pytest.mark.asyncio
async def test_send_400_records_last_send_error_no_retry():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(phone_number_id="ph", access_token="tok", verify_token="v")
    fake_400 = MagicMock(status_code=400, text='{"error":{"message":"bad"}}')
    ch._http = MagicMock()
    ch._http.post = AsyncMock(return_value=fake_400)

    await ch.send_text("+14155552671", "hello")
    assert ch._http.post.await_count == 1  # no retry on 400
    err = ch.health()["last_send_error"]
    assert err is not None
    assert err["status"] == 400
    assert "bad" in err["message_excerpt"]


@pytest.mark.asyncio
async def test_send_429_then_200_clears_error():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(phone_number_id="ph", access_token="tok", verify_token="v")
    fake_seq = [
        MagicMock(status_code=429, text="rate limited"),
        MagicMock(status_code=200, text=""),
    ]
    ch._http = MagicMock()
    ch._http.post = AsyncMock(side_effect=fake_seq)

    await ch.send_text("+14155552671", "hello")
    assert ch._http.post.await_count == 2
    h = ch.health()
    assert h["last_send_error"] is None
    assert h["last_send_at"] is not None


@pytest.mark.asyncio
async def test_send_long_text_chunked_and_markdown_converted():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(phone_number_id="ph", access_token="tok", verify_token="v")
    fake_200 = MagicMock(status_code=200, text="")
    ch._http = MagicMock()
    ch._http.post = AsyncMock(return_value=fake_200)

    long_text = "**bold** part. " + ("x" * 5000)
    await ch.send_text("+14155552671", long_text)
    assert ch._http.post.await_count >= 2  # was chunked

    # First chunk's body must be WhatsApp-flavored markdown.
    first_body = ch._http.post.await_args_list[0].kwargs["json"]["text"]["body"]
    assert "*bold*" in first_body
    assert "**bold**" not in first_body


@pytest.mark.asyncio
async def test_send_empty_text_does_not_call_meta():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    ch = WhatsAppChannel(phone_number_id="ph", access_token="tok", verify_token="v")
    ch._http = MagicMock()
    ch._http.post = AsyncMock()

    await ch.send_text("+14155552671", "")
    assert ch._http.post.await_count == 0


# ── Boot security warning ───────────────────────────────────────


@pytest.mark.asyncio
async def test_start_emits_security_warning_when_app_secret_unset():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    buf, h = _capture_warnings()
    try:
        ch = WhatsAppChannel(
            phone_number_id="ph", access_token="tok", verify_token="v",
        )
        await ch.start()
        ch._http = MagicMock(); ch._http.aclose = AsyncMock()
        try:
            assert "SECURITY app_secret not configured" in buf.getvalue()
        finally:
            await ch.stop()
    finally:
        _detach(h)


@pytest.mark.asyncio
async def test_start_does_not_warn_when_app_secret_set():
    from app.agent.channels.whatsapp_channel import WhatsAppChannel

    buf, h = _capture_warnings()
    try:
        ch = WhatsAppChannel(
            phone_number_id="ph", access_token="tok",
            verify_token="v", app_secret="sekrit",
        )
        await ch.start()
        ch._http = MagicMock(); ch._http.aclose = AsyncMock()
        try:
            assert "SECURITY app_secret not configured" not in buf.getvalue()
        finally:
            await ch.stop()
    finally:
        _detach(h)
