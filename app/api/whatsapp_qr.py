"""Agent-side QR pairing endpoints for the Baileys WhatsApp transport.

Three thin routes that delegate into the running
``BaileysWhatsAppChannel`` instance:

* ``POST /api/whatsapp/qr/start``  — tear down any existing session,
  trigger a fresh pairing flow. neonize emits the QR string within
  ~1 second; the client polls ``/qr/status`` for the rendered PNG.
* ``GET  /api/whatsapp/qr/status`` — snapshot of pairing state. Polled
  every 1–2 s by the Settings modal.
* ``POST /api/whatsapp/qr/logout`` — force-logout, wipe the on-disk
  session, mark "not_linked".

These routes are mounted ONLY on the agent's FastAPI app
(``agent_main.py``). The platform reaches them through
``/api/agent-setup/whatsapp/qr-*`` proxy endpoints that forward via
``X-Agent-Key`` over Caddy TLS.

Auth: ``get_current_user`` accepts both JWT and ``X-Agent-Key`` (see
``app/api/auth.py``), so the platform's proxy hits these directly with
the agent's API key — no separate auth pipeline needed.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp/qr", tags=["WhatsApp QR Pairing"])

# How long a pairing request may wait for a restarting adapter.
#
# The budget that matters is WAIT + WORK, not the wait alone: this route waits
# here and only THEN calls the sidecar, whose own mint is capped at ~8 s
# (`requestPairingCodeWhenReady`, bounded by `_SIDECAR_HTTP_TIMEOUT_S` = 10 s).
# 6 + 10 = 16 s, so the two platform call sites that can now wait
# (`/whatsapp/qr-start`, `/whatsapp/pair-code`) pass `timeout_s=18.0` instead
# of `_agent_qr_proxy`'s 10 s default — otherwise the platform 504s on a
# request this route is still holding and a valid code exists on the sidecar
# that nobody reads. Move one number and move the other.
_ADAPTER_WAIT_BUDGET_S = 6.0


def _require_active_channel():
    """Fetch the live ``BaileysWhatsAppChannel`` or raise 503.

    Returns 503 (not 404) because the channel exists in the codebase
    but isn't currently active — the right thing for the UI to do is
    surface "WhatsApp not configured for QR mode yet" and let the
    user fix it via Settings, not silently treat it as a missing
    feature.
    """
    from app.agent.channels.whatsapp_baileys import get_active_baileys_channel

    channel = get_active_baileys_channel()
    if channel is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "WhatsApp QR-link mode is not active on this agent. "
                "Save whatsapp_mode='qr_link' in agent settings and "
                "wait for the container to restart."
            ),
        )
    return channel


async def _await_active_channel(budget_s: float = _ADAPTER_WAIT_BUDGET_S):
    """The live adapter, waiting out a restart in flight (R46 F2).

    A restart is not atomic with registration: between `stop()` and the new
    `register()` the registry is empty, and this route used to answer 503
    within milliseconds of looking — which is how a 2.6 s internal transition
    reached the user as two failures the client had to paper over with a blind
    1800 ms retry (incident 2026-09-15, 14:48:20.990 / 14:48:23.367). The
    restart path publishes an event; this awaits it. It is NOT a poll and NOT
    the fix for the incident — F1 removes the restart itself — it is the
    bounded safety net for the restarts that remain real (mode change, sick
    sidecar, boot).
    """
    from app.agent.channels.whatsapp_baileys import (
        adapter_restart_in_flight, await_active_baileys_channel,
        get_active_baileys_channel,
    )

    # No adapter and nothing starting one is the PRE-EXISTING condition —
    # QR-link mode is not configured on this agent. Answer it exactly as
    # before, immediately: waiting 6 s to say "not configured" would be a new
    # kind of slow, and the platform proxy has its own copy for this case.
    if get_active_baileys_channel() is None and not adapter_restart_in_flight():
        return _require_active_channel()

    channel = await await_active_baileys_channel(budget_s)
    if channel is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "whatsapp_adapter_starting",
                "message": (
                    "The WhatsApp channel is still starting on your agent. "
                    "Try again in a moment."
                ),
                # The client honours this instead of guessing a sleep.
                "retry_after_s": 3,
            },
            headers={"Retry-After": "3"},
        )
    return channel


@router.post("/start")
async def qr_start(_user=Depends(get_current_user)):
    """Trigger a fresh QR pairing.

    Always wipes any existing auth state and tells the Baileys
    sidecar to spin up a brand new socket. Calling while a pairing
    is already in flight cancels it and starts a new one — exactly
    what the user wants when they click "Connect via QR" again.
    """
    channel = await _await_active_channel()
    await channel.kick_pair()
    return {"ok": True}


@router.get("/status")
async def qr_status(_user=Depends(get_current_user)):
    """Return the current pairing snapshot.

    Shape::

        {
          "session_status": "not_linked" | "linking" | "linked" | "logged_out",
          "connected": bool,
          "self_e164": str | null,
          "qr_data_url": "data:image/png;base64,..." | null,
          "qr_emitted_at": iso8601 | null
        }

    The frontend modal polls this every ~1.5 s while ``session_status``
    is ``"linking"``; flips to a success state when ``"linked"`` and
    closes; surfaces an error when ``"logged_out"``.
    """
    channel = _require_active_channel()
    return await channel.get_pairing_status()


class PairCodeRequest(BaseModel):
    """Body for ``POST /api/whatsapp/qr/pair-code``."""

    phone: str = Field(..., min_length=8, max_length=20, description="User's WhatsApp number in E.164 (with or without leading '+').")


@router.post("/pair-code")
async def qr_pair_code(
    payload: PairCodeRequest,
    _user=Depends(get_current_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Mint an 8-character WhatsApp pairing code for single-device linking.

    The mobile app cannot scan its own QR, so we use Baileys'
    ``requestPairingCode`` path instead. The returned code is what the
    user types into WhatsApp → Settings → Linked Devices → "Link a
    Device" → "Link with phone number instead." After they confirm,
    ``/qr/status`` reports ``session_status="linked"`` exactly as the
    QR flow would.
    """
    channel = await _await_active_channel()
    try:
        code = await channel.request_pairing_code(
            payload.phone, idempotency_key=idempotency_key,
        )
    except Exception as exc:
        # Class name, no traceback: the exception here is built from the
        # sidecar's own error body, which can echo the E.164 being paired —
        # and `RedactUserIdentifiers` masks `record.getMessage()` only, never
        # `record.exc_info`, so a `logger.exception` would ship the number to
        # Loki unmasked.
        logger.warning("[WHATSAPP-QR] pair_code.failed err=%s", type(exc).__name__)
        raise HTTPException(status_code=502, detail=f"Pairing-code request failed: {str(exc)[:200]}")
    return {"ok": True, "pairing_code": code, "phone": payload.phone}


@router.post("/logout")
async def qr_logout(_user=Depends(get_current_user)):
    """Force-logout the WhatsApp session and wipe on-disk state.

    Used when the user clicks "Disconnect" in Settings or wants to
    move the agent to a different phone number. After this returns,
    ``session_status`` is ``"not_linked"`` and the next ``/qr/start``
    will request a fresh QR.
    """
    channel = _require_active_channel()
    await channel.force_logout()
    return {"ok": True, "session_status": "not_linked"}
