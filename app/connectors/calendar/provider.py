"""T3c — Google Calendar connector provider.

Calendar v3 REST. Same shared `_google_base` helpers as Gmail. The
agent's user-scoped token is the only auth path; service accounts
are not supported.
"""

from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Optional

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

# events.list' own ceiling for one page. The post-filters below (owner,
# agenda, response) run over what came back, so a narrowed read has to
# ask for more than it will return — and this is how much more it may.
_MAX_FETCH = 250

# `within_hours` is a window the PROVIDER resolves against its own
# clock, which is the only way a poll event — whose `poll_args` are
# static by construction — can say "the next 24 hours". 90 days is the
# far end of anything worth calling upcoming.
_MAX_WITHIN_HOURS = 24 * 90


def _calendar_id(raw: Any) -> str:
    """The calendar to act on, escaped for one URL path segment.

    Calendar ids are email addresses (`me@example.com`,
    `…@group.calendar.google.com`) and the literal `primary`, so `@` and
    `.` go over the wire as themselves. Everything else is escaped, `/`
    included: this arrives as an LLM tool argument, and `quote()`'s
    default `safe='/'` would let one walk out of its segment onto a
    different Calendar resource.
    """
    cid = str(raw or "").strip() or "primary"
    return urllib.parse.quote(cid, safe="@.")


def _clamp(raw: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(n, hi))


def _agenda(ev: dict) -> str:
    """What this meeting has to read before it, as a WORD, not the text.

    "No agenda yet" only ever needs to know whether there is one, and a
    meeting description is the most sensitive field on the row — so the
    row carries the SHAPE ("description", "attachment") and never the
    body. An empty string is the honest "nothing to read", and it is a
    string rather than a boolean because `false` is indistinguishable
    from "set" to the filter vocabulary that reads it.
    """
    parts = []
    if str(ev.get("description") or "").strip():
        parts.append("description")
    if ev.get("attachments"):
        parts.append("attachment")
    return "+".join(parts)


def _my_response(ev: dict) -> str:
    """The signed-in user's own responseStatus, or "".

    Google marks the caller's own attendee row with `self: true`, so
    this needs no identity lookup and no extra scope. An event with no
    attendees (a hold, a personal block) has no response and answers "".
    """
    for att in (ev.get("attendees") or []):
        if isinstance(att, dict) and att.get("self"):
            return str(att.get("responseStatus") or "")
    return ""


def _role(ev: dict) -> str:
    """"organizer" | "attendee" | "" — who owns this meeting.

    `organizer.self` is Google's own marker for the caller, so this is a
    fact rather than a name comparison. "" is returned when neither can
    be established, and every narrowing that reads this keeps a row it
    cannot judge.
    """
    org = ev.get("organizer")
    if isinstance(org, dict) and org.get("self"):
        return "organizer"
    creator = ev.get("creator")
    if isinstance(creator, dict) and creator.get("self"):
        return "organizer"
    if _my_response(ev) or ev.get("attendees"):
        return "attendee"
    return ""


def _event_row(ev: dict) -> dict:
    """One events.list item → the row every reader of this connector
    sees.

    The three fields R43 §6 and §7 turn on — who owns it, whether it has
    an agenda, and what the user answered — are computed here and
    nowhere else, so a chip, an instant trigger and the brief cannot
    disagree about the same meeting.
    """
    org = ev.get("organizer") if isinstance(ev.get("organizer"), dict) else {}
    return {
        "id": ev.get("id"),
        "summary": ev.get("summary"),
        "start": ev.get("start"),
        "end": ev.get("end"),
        "location": ev.get("location"),
        "htmlLink": ev.get("htmlLink"),
        "attendee_count": len(ev.get("attendees") or []),
        "organizer_email": org.get("email"),
        "role": _role(ev),
        "my_response": _my_response(ev),
        "agenda": _agenda(ev),
        "status": ev.get("status"),
    }


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
                want = _clamp(tool_input.get("max_results", 10), 10, 1, 50)
                mine = bool(tool_input.get("organized_by_me"))
                bare = bool(tool_input.get("without_agenda"))
                waiting = bool(tool_input.get("awaiting_my_response"))
                narrowed = mine or bare or waiting
                params = {
                    # Google has no predicate for any of the three
                    # narrowings below, so they run over the page that
                    # comes back — which means a narrowed read has to
                    # ask for more than it will return, or "4 of my
                    # meetings" turns into whichever of the first 10 on
                    # the calendar happened to be mine.
                    "maxResults": min(_MAX_FETCH, want * 5) if narrowed else want,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                }
                if tool_input.get("time_min"):
                    params["timeMin"] = tool_input["time_min"]
                if tool_input.get("time_max"):
                    params["timeMax"] = tool_input["time_max"]
                hours = tool_input.get("within_hours")
                if hours is not None:
                    # An explicit bound the caller wrote outranks the
                    # window; `within_hours` only ever fills a side that
                    # was left open.
                    h = _clamp(hours, 24, 1, _MAX_WITHIN_HOURS)
                    now = datetime.now(timezone.utc).replace(microsecond=0)
                    params.setdefault("timeMin", now.isoformat())
                    params.setdefault(
                        "timeMax",
                        (now + timedelta(hours=h)).replace(
                            microsecond=0).isoformat(),
                    )
                if tool_input.get("query"):
                    params["q"] = tool_input["query"]
                result = await google_request(
                    "GET",
                    f"{CAL_API_BASE}/calendars/"
                    f"{_calendar_id(tool_input.get('calendar_id'))}/events",
                    access_token=access_token,
                    params=params,
                    scope_hint="calendar.events",
                )
                rows = [_event_row(ev) for ev in (result.get("items") or [])
                        if isinstance(ev, dict)]
                if mine:
                    rows = [r for r in rows if r["role"] == "organizer"]
                if bare:
                    rows = [r for r in rows if not r["agenda"]]
                if waiting:
                    # An invitation is one someone ELSE sent that is
                    # still unanswered; an organiser is never awaiting
                    # their own reply.
                    rows = [r for r in rows
                            if r["my_response"] == "needsAction"
                            and r["role"] != "organizer"]
                payload: dict[str, Any] = {"events": rows[:want]}
                if narrowed:
                    # Say what was asked for, so a lit chip and an empty
                    # list are not the same sentence.
                    payload["narrowed_by"] = [
                        n for n, on in (
                            ("organized_by_me", mine),
                            ("without_agenda", bare),
                            ("awaiting_my_response", waiting),
                        ) if on
                    ]
                return ConnectorOk(content=json.dumps(payload))

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
                attendees = [e for e in (tool_input.get("attendees") or []) if e]
                if attendees:
                    body["attendees"] = [{"email": e} for e in attendees]
                result = await google_request(
                    "POST",
                    f"{CAL_API_BASE}/calendars/"
                    f"{_calendar_id(tool_input.get('calendar_id'))}/events",
                    access_token=access_token,
                    json_body=body,
                    # `all` mails every attendee. An event with none —
                    # a hold the user put on their own calendar to read
                    # something — has nobody to tell, and `all` on it
                    # still asks Google to run the notification path.
                    # The parameter follows the guest list.
                    params={"sendUpdates": "all" if attendees else "none"},
                    scope_hint="calendar.events",
                )
                return ConnectorOk(content=json.dumps({
                    "id": result.get("id"),
                    "htmlLink": result.get("htmlLink"),
                    "summary": result.get("summary"),
                    "start": result.get("start"),
                    "end": result.get("end"),
                    "calendar_id": str(
                        tool_input.get("calendar_id") or "primary"),
                    "attendee_count": len(attendees),
                }))

            if tool_name == "calendar__check_availability":
                tmin = tool_input.get("time_min")
                tmax = tool_input.get("time_max")
                if not (tmin and tmax):
                    return ConnectorToolError(
                        message="time_min/time_max required",
                        retryable=False,
                    )
                # Busy blocks are derived from events.list, NOT from
                # /freeBusy. freebusy.query does not accept
                # `calendar.events` — Google's reference lists only
                # calendar, calendar.readonly, calendar.freebusy and
                # calendar.events.freebusy. This connector requests
                # `calendar.events` and nothing else (calendar.readonly
                # is in `scopes_optional`, which `_build_authorize_url`
                # never sends), so the old /freeBusy call 403'd for
                # EVERY user, always — a tool that shipped and could not
                # once have succeeded.
                #
                # events.list on `calendar.events` returns the same
                # information: `singleEvents` expands recurrences, so
                # each item's start/end IS a busy block. Deriving it
                # here keeps availability working without adding a
                # scope to the consent screen — which would otherwise
                # mean re-doing the Google verification submission.
                params = {
                    "timeMin": tmin,
                    "timeMax": tmax,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "maxResults": "250",
                }
                result = await google_request(
                    "GET",
                    f"{CAL_API_BASE}/calendars/primary/events",
                    access_token=access_token,
                    params=params,
                    scope_hint="calendar.events",
                )
                busy = []
                for ev in (result.get("items") or []):
                    # `transparent` means "free" on the user's calendar
                    # (Google's own wording for events that don't block
                    # time). Cancelled events linger in the list when
                    # syncToken-style reads are used; neither is busy.
                    if ev.get("transparency") == "transparent":
                        continue
                    if ev.get("status") == "cancelled":
                        continue
                    start = (ev.get("start") or {})
                    end = (ev.get("end") or {})
                    # All-day events carry `date`; timed ones `dateTime`.
                    s = start.get("dateTime") or start.get("date")
                    e = end.get("dateTime") or end.get("date")
                    if s and e:
                        busy.append({"start": s, "end": e})
                return ConnectorOk(content=json.dumps({
                    # Same envelope freebusy.query returned, so any
                    # caller that already parsed this keeps working.
                    "calendars": {"primary": {"busy": busy}},
                    "timeMin": tmin,
                    "timeMax": tmax,
                }))

            if tool_name == "calendar__delete_event":
                eid = tool_input.get("event_id")
                if not eid:
                    return ConnectorToolError(message="event_id required", retryable=False)
                await google_request(
                    "DELETE",
                    f"{CAL_API_BASE}/calendars/"
                    f"{_calendar_id(tool_input.get('calendar_id'))}/events/"
                    f"{urllib.parse.quote(str(eid), safe='')}",
                    access_token=access_token,
                    params={"sendUpdates": "all"},
                    scope_hint="calendar.events",
                )
                return ConnectorOk(content=json.dumps({
                    "deleted": True,
                    "event_id": eid,
                    "calendar_id": str(
                        tool_input.get("calendar_id") or "primary"),
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
