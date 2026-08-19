"""Screenshot reports → the Admin thread (platform-only).

A user files a report from the app (``POST /support/issues`` — a note, a
severity, the screen they were on, the build — then ``POST
/support/issues/{id}/attachment`` with the screenshot). Until now that reached
the operator by email alone. This module is the SECOND destination: the same
report opens as a message in the user's Admin thread, so it lands in the
Conversations pane with its screenshot and severity, and the operator answers
it there — through the reply plumbing that already exists — instead of by
email.

What is deliberately NOT here:

  * No new thread. ``admin_thread_messages`` has no thread entity (D3, Unit 1:
    "starting a conversation with a user who already has one must not create a
    second"), and both clients render ONE Admin inbox per user. A report is one
    row of ``kind='report'`` in that thread; a second report is a second row —
    its own card, its own severity, its own screenshot — never merged into the
    first. Per-report threads would need a thread id on every route and both
    clients to learn it, which is a redesign of Conversations, not a
    destination for reports.
  * No second picture table. The screenshot is an ``admin_thread_attachments``
    row (093) on the report row; the support card keeps its own copy in
    ``support_attachments`` because email and the Support tab read it there.
    Bounded (3 MB, images only, one per report from the app) and rare.
  * No email change. ``support.py`` still sends the card email; the thread row
    is written beside it, and a failure here never fails intake.

The report row's id is a uuid5 of the support issue id — the same trick the
fan-out uses for (dispatch, user). The screenshot arrives in a SECOND request
and has to find its message; a deterministic id does that without a lookup
column, and a replayed intake collides on the PK instead of filing twice.

"Answered" is structural, not a flag: a report is OPEN while no operator
``out`` row has been written after it. The Conversations list badges the
highest open severity per user (critical loudest); ``reply_in_thread`` uses
the same predicate to decide that its reply answers a report — and delivers
THAT reply into the user's chat as a persistent card with a Reply action,
through the ordinary dispatch fan-out. One definition (``report_state_for_users``)
serves both, so the badge and the delivery can never disagree.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AdminDispatch,
    AdminThreadAttachment,
    AdminThreadMessage,
    DISPATCH_AUDIENCE_ALL,
    REPORT_SEVERITY_MEDIUM,
    REPORT_SEVERITY_RANK,
    THREAD_IN,
    THREAD_KIND_REPORT,
    THREAD_OUT,
)

logger = logging.getLogger(__name__)

# The card title an operator's answer to a report wears in the user's chat.
# The dispatch model requires a title and the operator typed none; this names
# the card for what it is rather than repeating the sender's name.
REPORT_REPLY_TITLE = "Reply to your report"

# What the context block may carry, and in what order the panel shows it. The
# app's SupportReportOverlay writes `Screen / App / Device / Platform` lines
# into free-text `repro_info`; a future client can send these structured as
# `context`. Anything outside this list is dropped, and every value is capped:
# this is display data on the operator's screen, never a place to smuggle a
# few KB per report into a JSON column.
CONTEXT_KEYS: tuple[str, ...] = ("screen", "app_version", "build", "platform", "device", "os")
_CONTEXT_VALUE_MAX = 200
_CONTEXT_RAW_MAX = 2000

_APP_RE = re.compile(r"^\s*(?P<version>[^\s(]+)\s*(?:\((?P<build>[^)]*)\))?\s*$")


def report_message_id(issue_id: str) -> str:
    """The thread row a support issue opens. Deterministic — see the module
    docstring for why."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"admin-thread-report:{issue_id}"))


def _clip(v: Any, n: int = _CONTEXT_VALUE_MAX) -> Optional[str]:
    s = str(v).strip() if v is not None else ""
    return s[:n] if s else None


def parse_report_context(
    repro_info: Optional[str],
    explicit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Structure the report's context, best-effort.

    Reads the mobile overlay's free-text block::

        Screen: Chat
        App: 1.2.0 (40)
        Device: iPhone 15 Pro · iOS 18.5
        Platform: ios

    into ``{screen, app_version, build, platform, device, os}``, keeps the
    original text under ``raw`` (capped) so nothing the user's device said is
    lost to the parser, and lets an ``explicit`` structured block win over
    anything parsed. Every value is allow-listed and capped; unknown keys are
    dropped, and a line that does not parse only leaves its field NULL.
    """
    out: Dict[str, Optional[str]] = {k: None for k in CONTEXT_KEYS}
    raw = (repro_info or "").strip()

    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        if key == "screen":
            out["screen"] = _clip(value)
        elif key == "app":
            m = _APP_RE.match(value)
            if m:
                out["app_version"] = _clip(m.group("version"))
                out["build"] = _clip(m.group("build")) if m.group("build") else None
            else:
                out["app_version"] = _clip(value)
        elif key == "device":
            # "iPhone 15 Pro · iOS 18.5" — the overlay joins model and OS with
            # a middle dot; older text may use " - " or "," which we accept too.
            parts = [p.strip() for p in re.split(r"\s+·\s+|\s+-\s+|,\s*", value, maxsplit=1)]
            out["device"] = _clip(parts[0])
            if len(parts) > 1:
                out["os"] = _clip(parts[1])
        elif key == "os":
            out["os"] = _clip(value)
        elif key == "platform":
            out["platform"] = _clip(value.lower())
        elif key in ("build", "app_version"):
            out[key] = _clip(value)

    for key, value in (explicit or {}).items():
        k = str(key).strip().lower()
        if k in CONTEXT_KEYS:
            v = _clip(value)
            if v is not None:
                out[k] = v.lower() if k == "platform" else v

    result: Dict[str, Optional[str]] = dict(out)
    result["raw"] = raw[:_CONTEXT_RAW_MAX] if raw else None
    return result


def build_report_json(
    *,
    support_issue_id: str,
    channel: Optional[str],
    repro_info: Optional[str],
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The `report_json` block a report row carries. ONE construction site so
    the panel, the user's inbox and the tests read the same keys."""
    return {
        "support_issue_id": support_issue_id,
        "channel": _clip(channel, 32),
        "context": parse_report_context(repro_info, context),
    }


async def open_report_in_thread(
    db: AsyncSession,
    *,
    user_id: str,
    issue_id: str,
    note: str,
    severity: Optional[str],
    channel: Optional[str],
    repro_info: Optional[str],
    context: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
) -> tuple[AdminThreadMessage, bool]:
    """Write the report as an `in` row of the user's Admin thread. Commits.

    Returns ``(row, created)``. Idempotent on the uuid5 id: a replayed intake
    (or a second worker) collides on the PK and gets the existing row back.

    The body is the NOTE and only the note — this row renders as the user's
    own bubble in their inbox, where a "[critical] Report:" prefix would read
    as something they typed. Severity and context ride the columns; the panel
    builds the card header from those.
    """
    now = now or datetime.utcnow()
    row_id = report_message_id(issue_id)
    existing = await db.get(AdminThreadMessage, row_id)
    if existing is not None:
        return existing, False

    sev = (severity or "").strip().lower()
    if sev not in REPORT_SEVERITY_RANK:
        # The support intake validates against its enum before we are called;
        # this is the seam for a caller that did not. Never refuse the report
        # over its label — file it at the default and let the operator judge.
        sev = REPORT_SEVERITY_MEDIUM

    msg = AdminThreadMessage(
        id=row_id,
        user_id=user_id,
        dispatch_id=None,
        direction=THREAD_IN,
        body=note,
        author_admin_id=None,
        sender_name=None,
        created_at=now,
        kind=THREAD_KIND_REPORT,
        severity=sev,
        report_json=build_report_json(
            support_issue_id=issue_id, channel=channel,
            repro_info=repro_info, context=context,
        ),
    )
    db.add(msg)
    try:
        await db.commit()
    except IntegrityError:
        # Two intakes for one issue racing on two replicas — the PK wins.
        await db.rollback()
        existing = await db.get(AdminThreadMessage, row_id)
        if existing is None:  # pragma: no cover — the row we just collided on
            raise
        return existing, False

    logger.info(
        "[report-thread] opened report %s for user %s severity=%s channel=%s",
        issue_id[:8], user_id[:8], sev, channel,
    )
    return msg, True


async def attach_report_screenshot(
    db: AsyncSession,
    *,
    issue_id: str,
    data: bytes,
    mime_type: str,
    sha256: Optional[str],
    uploaded_by_user_id: Optional[str],
    now: Optional[datetime] = None,
) -> Optional[AdminThreadAttachment]:
    """Hang the screenshot off the report row, if the row exists. Commits.

    Returns None — and writes nothing — when there is no report row for this
    issue (the issue predates this feature, or was filed by a path that never
    opened a thread), or when the row has been deleted for everyone: 093's
    purge already destroyed that message's pictures, and a late upload must
    not resurrect one on a tombstone.

    Idempotent per (message, sha256): the app uploads once, but a client that
    retries after a dropped connection must not give the operator the same
    screenshot twice.
    """
    now = now or datetime.utcnow()
    row_id = report_message_id(issue_id)
    row = (await db.execute(
        select(AdminThreadMessage.id, AdminThreadMessage.deleted_at)
        .where(AdminThreadMessage.id == row_id)
    )).first()
    if row is None or row.deleted_at is not None:
        return None

    if sha256:
        dup = (await db.execute(
            select(AdminThreadAttachment.id).where(
                AdminThreadAttachment.message_id == row_id,
                AdminThreadAttachment.sha256 == sha256,
            )
        )).scalar_one_or_none()
        if dup is not None:
            return await db.get(AdminThreadAttachment, dup)

    att = AdminThreadAttachment(
        id=str(uuid.uuid4()),
        message_id=row_id,
        data=data,
        mime_type=mime_type,
        size_bytes=len(data),
        sha256=sha256,
        uploaded_by_user_id=uploaded_by_user_id,
        created_at=now,
    )
    db.add(att)
    await db.commit()
    logger.info(
        "[report-thread] attached %s (%d bytes) to report %s",
        mime_type, len(data), issue_id[:8],
    )
    return att


class ReportState(NamedTuple):
    """What the Conversations list and the reply route both need to know
    about one user's reports. See ``report_state_for_users``."""
    report_count: int
    open_count: int
    # The badge: the highest OPEN severity, else the latest report's. None
    # when the user has never filed one.
    severity: Optional[str]
    latest_severity: Optional[str]
    latest_report_id: Optional[str]
    # The loudest report no operator has answered — the one the badge names.
    open_report_id: Optional[str]
    open_severity: Optional[str]
    # Every unanswered report, oldest first — a reply answers all of them, and
    # the thread view marks each one open or answered off this list rather
    # than re-deriving the predicate client-side (where a broadcast's row
    # would look like an answer).
    open_report_ids: tuple = ()


EMPTY_REPORT_STATE = ReportState(0, 0, None, None, None, None, None, ())


async def report_state_for_users(
    db: AsyncSession, user_ids: Iterable[str],
) -> Dict[str, ReportState]:
    """Per-user report facts, in two queries over the given ids.

    A report is OPEN while no operator ``out`` row ADDRESSED TO THAT USER
    exists after it — a typed thread reply, or a dispatch sent to them alone;
    deleted or not, since a tombstone is still a turn the operator took. A
    BROADCAST's thread row does not count: "maintenance tonight" to everyone
    is not an answer to anyone's report, and letting it close every open report
    on the platform would send the operator's real answer thread-only, with no
    card. Reports deleted for everyone are tombstones and no longer count.
    Reports merely hidden from the user still do — the operator can still see
    them, and they are the operator's queue.

    Called with a PAGE of user ids (the list route's ≤500), never the whole
    table; and by the reply route with one. Both share this so the badge that
    says "open" and the reply that answers it can never disagree.
    """
    ids = [u for u in dict.fromkeys(user_ids) if u]
    if not ids:
        return {}

    broadcast_ids = select(AdminDispatch.id).where(
        AdminDispatch.audience == DISPATCH_AUDIENCE_ALL,
    )
    last_out_rows = (await db.execute(
        select(AdminThreadMessage.user_id, func.max(AdminThreadMessage.created_at))
        .where(
            AdminThreadMessage.user_id.in_(ids),
            AdminThreadMessage.direction == THREAD_OUT,
            # `IS NULL OR NOT IN`, spelled out: a typed reply has no dispatch
            # id, and in SQL `NULL NOT IN (...)` is NULL — false — so a bare
            # NOT IN would drop every typed reply and nothing would ever
            # count as an answer.
            or_(
                AdminThreadMessage.dispatch_id.is_(None),
                AdminThreadMessage.dispatch_id.not_in(broadcast_ids),
            ),
        )
        .group_by(AdminThreadMessage.user_id)
    )).all()
    last_out: Dict[str, datetime] = {r[0]: r[1] for r in last_out_rows}

    report_rows = (await db.execute(
        select(
            AdminThreadMessage.id,
            AdminThreadMessage.user_id,
            AdminThreadMessage.severity,
            AdminThreadMessage.created_at,
        )
        .where(
            AdminThreadMessage.user_id.in_(ids),
            AdminThreadMessage.kind == THREAD_KIND_REPORT,
            AdminThreadMessage.deleted_at.is_(None),
        )
        .order_by(AdminThreadMessage.created_at.asc())
    )).all()

    per_user: Dict[str, list] = {}
    for r in report_rows:
        per_user.setdefault(r.user_id, []).append(r)

    out: Dict[str, ReportState] = {}
    for uid, rows in per_user.items():
        answered_before = last_out.get(uid)
        open_rows = [
            r for r in rows
            if answered_before is None or r.created_at > answered_before
        ]
        latest = rows[-1]
        open_severity: Optional[str] = None
        open_report_id: Optional[str] = None
        if open_rows:
            # Loudest first; among equals, the newest. A reply answers EVERY
            # open report at once (none is open after it), so the id here is
            # the one the badge names, for the composer hint and the audit.
            loudest = max(
                open_rows,
                key=lambda r: (REPORT_SEVERITY_RANK.get(r.severity or "", -1), r.created_at),
            )
            open_severity = loudest.severity
            open_report_id = loudest.id
        out[uid] = ReportState(
            report_count=len(rows),
            open_count=len(open_rows),
            severity=open_severity or latest.severity,
            latest_severity=latest.severity,
            latest_report_id=latest.id,
            open_report_id=open_report_id,
            open_severity=open_severity,
            open_report_ids=tuple(r.id for r in open_rows),
        )
    return out


async def open_report_for_user(db: AsyncSession, user_id: str) -> ReportState:
    """The single-user form the reply route uses."""
    return (await report_state_for_users(db, [user_id])).get(user_id, EMPTY_REPORT_STATE)
