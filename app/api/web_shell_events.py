"""
Mobile-web-shell telemetry ingest.

  POST /api/web-shell/events   (auth; the SPA emits rollout events)

One endpoint with a discriminated body rather than three paths, because these
three events share a lifecycle: they are added and removed together as the ramp
runs, and a new one should not need a new route, a new client method and a new
mount.

Server-side, for the same reason the onboarding funnel is server-side: `user_id`
comes from the token and not from the client, and the funnel lands in one place
the admin dashboard can graph without correlating two clocks.

Returns 204 and NEVER 4xx for a payload it merely does not recognise — an
unknown event name is dropped and counted, not rejected. A telemetry endpoint
that errors makes a deployed client retry, and a client retrying telemetry is a
worse outage than the missing data.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from app.db import User
from app.api.auth import get_current_user
from app.services import web_shell_events as ev

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web-shell", tags=["Web shell"])


class WebShellEvent(BaseModel):
    event: Literal["shell_rendered", "drawer_opened", "shell_disabled"]
    # shell_rendered
    shell: Optional[Literal["mobile", "legacy"]] = None
    viewport_width: Optional[int] = Field(default=None, ge=0, le=20_000)
    standalone: bool = False
    # drawer_opened
    via: Optional[Literal["button", "swipe"]] = None
    # shell_disabled
    reason: Optional[str] = Field(default=None, max_length=64)


@router.post("/events", status_code=204, response_class=Response)
async def post_web_shell_event(
    body: WebShellEvent,
    user: User = Depends(get_current_user),
) -> Response:
    uid = str(user.id)
    if body.event == "shell_rendered":
        ev.emit_shell_rendered(
            user_id=uid,
            # A client that omits `shell` is a client we cannot attribute, and
            # attributing it to the new shell would inflate the numerator of the
            # one metric this exists to measure.
            shell=body.shell or "unknown",
            viewport=ev.bucket_for_width(body.viewport_width or 0),
            standalone=body.standalone,
        )
    elif body.event == "drawer_opened":
        ev.emit_drawer_opened(user_id=uid, via=body.via or "button")
    elif body.event == "shell_disabled":
        ev.emit_shell_disabled(user_id=uid, reason=body.reason)
    return Response(status_code=204)
