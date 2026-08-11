"""T3c — Google Calendar connector provider.

Calendar v3 REST. Same shared `_google_base` helpers as Gmail. The
agent's user-scoped token is the only auth path; service accounts
are not supported.
"""

from __future__ import annotations

import json
from typing import ClassVar, Optional

from app.connectors._google_base import (
    _GoogleConnectorError,
    google_liveness,
    google_refresh,
    google_request,
    google_revoke,
)
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.services import connector_vault as _vault

CAL_API_BASE = "https://www.googleapis.com/calendar/v3"


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "calendar")
    if ident is None or not ident.access_token:
        raise _GoogleConnectorError(
            ConnectorToolError(message="No active Calendar identity", retryable=False),
        )
    return ident.access_token


def _retarget_reauth(result: ConnectorResult) -> ConnectorResult:
    if isinstance(result, ConnectorReauthRequired):
        return ConnectorReauthRequired(reauth_url="/agent/integrations/calendar")
    return result


class CalendarProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "calendar"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _GoogleConnectorError as e:
            return _retarget_reauth(e.result)

        try:
            if tool_name == "calendar__list_events":
                params = {
                    "maxResults": int(tool_input.get("max_results", 10)),
                    "singleEvents": "true",
                    "orderBy": "startTime",
                }
                if tool_input.get("time_min"):
                    params["timeMin"] = tool_input["time_min"]
                if tool_input.get("time_max"):
                    params["timeMax"] = tool_input["time_max"]
                if tool_input.get("query"):
                    params["q"] = tool_input["query"]
                result = await google_request(
                    "GET",
                    f"{CAL_API_BASE}/calendars/primary/events",
                    access_token=access_token,
                    params=params,
                    scope_hint="calendar.events",
                )
                events = []
                for ev in (result.get("items") or [])[:int(tool_input.get("max_results", 10))]:
                    events.append({
                        "id": ev.get("id"),
                        "summary": ev.get("summary"),
                        "start": ev.get("start"),
                        "end": ev.get("end"),
                        "location": ev.get("location"),
                        "htmlLink": ev.get("htmlLink"),
                        "attendee_count": len(ev.get("attendees") or []),
                    })
                return ConnectorOk(content=json.dumps({"events": events}))

            if tool_name == "calendar__create_event":
                summary = tool_input.get("summary")
                start = tool_input.get("start")
                end = tool_input.get("end")
                if not (summary and start and end):
                    return ConnectorToolError(
                        message="summary/start/end required",
                        retryable=False,
                    )
                body: dict = {
                    "summary": summary,
                    "start": {"dateTime": start},
                    "end": {"dateTime": end},
                }
                if tool_input.get("description"):
                    body["description"] = tool_input["description"]
                if tool_input.get("location"):
                    body["location"] = tool_input["location"]
                if tool_input.get("attendees"):
                    body["attendees"] = [
                        {"email": e} for e in tool_input["attendees"] if e
                    ]
                result = await google_request(
                    "POST",
                    f"{CAL_API_BASE}/calendars/primary/events",
                    access_token=access_token,
                    json_body=body,
                    params={"sendUpdates": "all"},
                    scope_hint="calendar.events",
                )
                return ConnectorOk(content=json.dumps({
                    "id": result.get("id"),
                    "htmlLink": result.get("htmlLink"),
                    "summary": result.get("summary"),
                    "start": result.get("start"),
                    "end": result.get("end"),
                }))

            if tool_name == "calendar__check_availability":
                tmin = tool_input.get("time_min")
                tmax = tool_input.get("time_max")
                if not (tmin and tmax):
                    return ConnectorToolError(
                        message="time_min/time_max required",
                        retryable=False,
                    )
                cals = tool_input.get("calendars") or ["primary"]
                body = {
                    "timeMin": tmin,
                    "timeMax": tmax,
                    "items": [{"id": c} for c in cals],
                }
                result = await google_request(
                    "POST",
                    f"{CAL_API_BASE}/freeBusy",
                    access_token=access_token,
                    json_body=body,
                    scope_hint="calendar.events",
                )
                return ConnectorOk(content=json.dumps(result))

            if tool_name == "calendar__delete_event":
                eid = tool_input.get("event_id")
                if not eid:
                    return ConnectorToolError(message="event_id required", retryable=False)
                await google_request(
                    "DELETE",
                    f"{CAL_API_BASE}/calendars/primary/events/{eid}",
                    access_token=access_token,
                    params={"sendUpdates": "all"},
                    scope_hint="calendar.events",
                )
                return ConnectorOk(content=json.dumps({
                    "deleted": True,
                    "event_id": eid,
                }))

            return ConnectorToolError(
                message=f"unknown calendar tool {tool_name!r}",
                retryable=False,
            )
        except _GoogleConnectorError as e:
            return _retarget_reauth(e.result)

    async def revoke(self, user_id, access_token, refresh_token=None):
        await google_revoke(access_token)

    async def refresh(
        self,
        refresh_token: str,
        *,
        scopes: Optional[list[str]] = None,
    ) -> RefreshResult:
        return await google_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """Probe the SAME surface the tools use — `events.list` on the
        primary calendar.

        This used to call `users/me/calendarList`, which requires
        `calendar.readonly`. That scope is `scopes_optional` in the
        manifest and `oauth.py` only ever requests `oauth.scopes`, so
        NO user has ever held it. Every probe returned 403
        insufficientPermissions, three sweeps flipped the identity, and
        Calendar read "Provider down" for every user on the platform
        while all four of its tools worked perfectly. Verified against
        a live grant on 2026-08-07.

        The rule this cost us: a health probe must exercise a scope the
        connector actually asks for. Probing a wider surface than the
        grant turns a working connector into a permanently dead one.
        """
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _GoogleConnectorError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")

        ok, detail = await google_liveness(
            f"{CAL_API_BASE}/calendars/primary/events?maxResults=1",
            access_token=access_token,
        )
        return HealthResult(ok=ok, detail=detail)
