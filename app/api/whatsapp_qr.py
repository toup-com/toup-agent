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

    Idempotent — calling it while a pairing is already in flight is
    safe; neonize's connect loop simply continues, and the next
    ``/qr/status`` poll returns the latest QR. Calling it after a
    successful pair re-emits a QR (intended UX for "I want to relink
    on a different phone").
    """
    channel = _require_active_channel()
    # If there's a logged_out terminal state, force_logout() clears
    # it so the next supervisor iteration re-pairs from scratch.
    if channel.health().get("session_status") == "logged_out":
        await channel.force_logout()
    # Otherwise the supervisor is already running; QR will appear in
    # the next status poll. Return ok and let the client poll.
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
