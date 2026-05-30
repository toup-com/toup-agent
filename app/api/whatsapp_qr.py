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

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp/qr", tags=["WhatsApp QR Pairing"])


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


@router.post("/start")
async def qr_start(_user=Depends(get_current_user)):
    """Trigger a fresh QR pairing.

    Always wipes any existing auth state and tells the Baileys
    sidecar to spin up a brand new socket. Calling while a pairing
    is already in flight cancels it and starts a new one — exactly
    what the user wants when they click "Connect via QR" again.
    """
    channel = _require_active_channel()
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
    return channel.get_pairing_status()


class PairCodeRequest(BaseModel):
    """Body for ``POST /api/whatsapp/qr/pair-code``."""

    phone: str = Field(..., min_length=8, max_length=20, description="User's WhatsApp number in E.164 (with or without leading '+').")


@router.post("/pair-code")
async def qr_pair_code(payload: PairCodeRequest, _user=Depends(get_current_user)):
    """Mint an 8-character WhatsApp pairing code for single-device linking.

    The mobile app cannot scan its own QR, so we use Baileys'
    ``requestPairingCode`` path instead. The returned code is what the
    user types into WhatsApp → Settings → Linked Devices → "Link a
    Device" → "Link with phone number instead." After they confirm,
    ``/qr/status`` reports ``session_status="linked"`` exactly as the
    QR flow would.
    """
    channel = _require_active_channel()
    try:
        code = await channel.request_pairing_code(payload.phone)
    except Exception as exc:
        logger.exception("[WHATSAPP-QR] pair_code.failed")
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
