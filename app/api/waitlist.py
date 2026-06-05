"""Public waitlist / invite-request endpoint.

Unauthenticated: a visitor on the closed-beta sign-in screen leaves their
email and a short note, and we notify the admin so they can follow up.

Delivery channel: **Telegram is primary** because Railway blocks outbound SMTP
(port 587 times out from the container), so raw email can't be sent from prod.
The admin-alert Telegram bot reaches api.telegram.org over HTTPS, which is not
blocked. Email is attempted best-effort as a secondary channel — it is a no-op
today (no HTTP email provider configured) but auto-upgrades the moment one is
(e.g. RESEND_API_KEY). The request succeeds if *either* channel delivers.

There is no DB model — this is a notification, not a stored entity. The only
recipients are the fixed admin Telegram chat and admin email, so the route
cannot be used as an open relay; email reply-to is the requester.
"""
from __future__ import annotations

import html as html_lib
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from app.config import settings
from app.services.email_service import send_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/waitlist", tags=["Waitlist"])

# Where invite requests land. Kept local (not a setting) because it is the
# human who triages beta access, not a configurable system address.
ADMIN_NOTIFY = "mrhx@toup.ai"

# email_service returns success=True with these providers when it only LOGGED
# the message (no real send) — treat them as "not actually delivered".
_LOGGED_ONLY = {"logged", "smtp-logged", "resend-logged", "ses-logged"}


class WaitlistRequest(BaseModel):
    email: EmailStr
    name: Optional[str] = Field(default=None, max_length=120)
    note: Optional[str] = Field(default=None, max_length=2000)
    source: Optional[str] = Field(default="mobile", max_length=40)


class WaitlistResponse(BaseModel):
    ok: bool


async def _notify_telegram(text_html: str) -> bool:
    """Best-effort admin ping via the shared admin-alert Telegram bot.

    Reuses ADMIN_ALERT_TELEGRAM_{TOKEN,CHAT_ID} (already configured in prod).
    Returns True only on a 2xx from Telegram. Never raises.
    """
    token = (settings.admin_alert_telegram_token or "").strip()
    chat_id = (settings.admin_alert_telegram_chat_id or "").strip()
    if not token or not chat_id:
        logger.warning("[waitlist] no Telegram admin-alert config — skipping")
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text_html,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
        if r.status_code >= 400:
            logger.warning("[waitlist] telegram HTTP %d: %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        logger.warning("[waitlist] telegram send failed: %s", e)
        return False


@router.post("", response_model=WaitlistResponse)
@router.post("/", response_model=WaitlistResponse)
async def request_invite(payload: WaitlistRequest) -> WaitlistResponse:
    e = html_lib.escape
    name = (payload.name or "").strip()
    note = (payload.note or "").strip()
    rows = [
        ("Email", str(payload.email)),
        ("Name", name or "—"),
        ("Source", payload.source or "—"),
        ("Note", note or "—"),
    ]

    # ── Primary: Telegram (works on Railway) ──
    tg_text = "<b>🎟️ New Toup invite request</b>\n\n" + "\n".join(
        f"<b>{e(k)}:</b> {e(v)}" for k, v in rows
    )
    telegram_ok = await _notify_telegram(tg_text)

    # ── Secondary: email (best-effort; no-op until an HTTP provider is set) ──
    html_body = (
        "<h2 style='font-family:system-ui,sans-serif'>New Toup invite request</h2>"
        "<table cellpadding='6' style='font-family:system-ui,sans-serif;font-size:14px;border-collapse:collapse'>"
        + "".join(
            f"<tr><td style='color:#666'><b>{e(k)}</b></td><td>{e(v)}</td></tr>"
            for k, v in rows
        )
        + "</table>"
    )
    text_body = "New Toup invite request\n\n" + "\n".join(f"{k}: {v}" for k, v in rows)
    email_res = await send_email(
        to=ADMIN_NOTIFY,
        subject=f"Toup invite request — {payload.email}",
        html=html_body,
        text=text_body,
        reply_to=str(payload.email),
    )
    email_ok = bool(getattr(email_res, "success", False)) and \
        getattr(email_res, "provider", "") not in _LOGGED_ONLY

    if not telegram_ok and not email_ok:
        logger.error(
            "[waitlist] all channels failed for %s (telegram=%s email=%s)",
            payload.email, telegram_ok, email_ok,
        )
        raise HTTPException(status_code=502, detail="Could not submit request. Please try again.")

    logger.info(
        "[waitlist] invite request from %s (telegram=%s email=%s)",
        payload.email, telegram_ok, email_ok,
    )
    return WaitlistResponse(ok=True)
