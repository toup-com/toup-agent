"""Public waitlist / invite-request endpoint.

Unauthenticated: a visitor on the closed-beta sign-in screen leaves their
email and a short note, and we email the admin (``mrhx@toup.ai``) so they can
follow up. There is no DB model — this is a notification, not a stored entity.
Best-effort send: a provider failure surfaces as 502 so the app can ask the
user to retry. The only recipient is the fixed admin address, so the route
cannot be used as an open relay; reply-to is set to the requester so the admin
can answer straight from their inbox.
"""
from __future__ import annotations

import html as html_lib
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from app.services.email_service import send_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/waitlist", tags=["Waitlist"])

# Where invite requests land. Kept local (not a setting) because it is the
# human who triages beta access, not a configurable system address.
ADMIN_NOTIFY = "mrhx@toup.ai"


class WaitlistRequest(BaseModel):
    email: EmailStr
    name: Optional[str] = Field(default=None, max_length=120)
    note: Optional[str] = Field(default=None, max_length=2000)
    source: Optional[str] = Field(default="mobile", max_length=40)


class WaitlistResponse(BaseModel):
    ok: bool


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

    result = await send_email(
        to=ADMIN_NOTIFY,
        subject=f"Toup invite request — {payload.email}",
        html=html_body,
        text=text_body,
        reply_to=str(payload.email),  # admin replies go straight to the requester
    )
    if not result.success:
        logger.error("[waitlist] email send failed for %s", payload.email)
        raise HTTPException(status_code=502, detail="Could not submit request. Please try again.")

    logger.info("[waitlist] invite request from %s", payload.email)
    return WaitlistResponse(ok=True)
