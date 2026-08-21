"""The job-card row: one projection of it, and the guard that keeps its
marker out of the message stream.

A job card is stored as an ordinary ``messages`` row whose ``role`` is
``job`` and whose ``content`` is not prose but a MARKER —
``{"job_id": …, "job_name": …}`` — written by three producers
(``ws_chat``'s two writers and ``voice_jobs._write_chat_card``). The row is
a POINTER; the card itself is rendered from the ``BuildJob`` it points at.

Exactly one of the four history readers knew that. ``api/sessions.py``
blanked the content and projected the job fields. ``api/day_chats.py``
(both of its branches) and ``api/messages_recover.py`` returned
``content`` verbatim — and ``/api/day-chats/{date}/messages`` is the
PRIMARY fetch every client makes; the session routes are only its
fallback. So on the path that actually runs, the marker WAS the message.
A client that does not special-case ``role='job'`` — the mobile app maps
every non-user role to ``assistant`` and renders ``content`` — printed it
as an agent bubble::

    {"job_id": "d89c6987-17ff-420b-bf1b-419d29ce8217",
     "job_name": "\\u06a9\\u0627\\u0631\\u0628\\u0631 \\u062f\\u0631\\u0628…"}

an internal UUID plus the user's own Persian sentence in ``\\uXXXX``
escapes (``json.dumps`` defaults to ``ensure_ascii=True``), attributed to
the agent as if it had said it. Round 13 gave voice turns real job cards,
which is what made these rows common enough to notice; the leak itself is
as old as the row type and hit typed app-builder runs identically.

Two rules live here, and every reader gets both:

1. **The projection** (:func:`job_card_fields`) — ONE implementation of
   "a ``role='job'`` row becomes a card", including the step list, so the
   card a voice turn leaves in the thread is the card a typed run leaves,
   field for field. Drift between serializers is the bug class this
   module exists to end (``tool_events``, ``admin_notice`` and ``channel``
   each shipped to one serializer first and vanished on the fallback).

2. **The guard** (:func:`public_text`) — nothing that looks like an
   internal marker is ever emitted as user-visible text, whatever wrote
   it. A new marker-shaped row type invented next quarter cannot repeat
   this leak by forgetting to teach four readers about itself: it fails
   closed, as an empty body, instead of open, as a JSON blob with a UUID
   in it.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "JOB_MARKER_KEYS",
    "attach_run_to_cards",
    "collect_job_ids",
    "is_internal_marker",
    "job_card_fields",
    "job_marker_content",
    "load_build_jobs",
    "parse_job_marker",
    "public_text",
]


# Every key a job marker is allowed to carry. The guard fires only when a
# body is EXACTLY one of these objects — an object whose keys are all in
# this set and nothing else. That precision is deliberate: a user who
# pastes JSON containing the word "job_id" alongside anything else is
# writing a real message and must still see it, while a body that is
# nothing but marker keys can only have come from a mis-routed writer.
JOB_MARKER_KEYS: frozenset = frozenset({
    "job_id", "job_name", "job_status", "job_type", "job_step",
})

# A bare identifier as an entire message body. Accepts the `job-<uuid>`
# row-id form too, because that is the other internal token that has ever
# been within one bug of being rendered.
_BARE_ID_RE = re.compile(
    r"^(?:job-|urn:uuid:)?"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def job_marker_content(
    job_id: str,
    job_name: str,
    job_type: Optional[str] = None,
) -> str:
    """The marker string a job row stores in ``content``.

    One writer for all three producers so the shape cannot drift.
    ``ensure_ascii=False`` is not cosmetic: it is the difference between a
    leak that reads as a Persian sentence and one that reads as
    ``\\u06a9\\u0627…``. The row is not supposed to be rendered at all,
    but a marker that stays legible when something renders it anyway is
    strictly the better failure.
    """
    body: Dict[str, Any] = {"job_id": job_id, "job_name": job_name}
    if job_type:
        body["job_type"] = job_type
    return json.dumps(body, ensure_ascii=False)


def parse_job_marker(content: Optional[str]) -> Dict[str, Any]:
    """The marker's fields, or ``{}`` for a missing / malformed body."""
    if not content:
        return {}
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def is_internal_marker(text: Optional[str], *, role: str = "assistant") -> bool:
    """True when ``text`` is a machine token rather than something a person
    or the agent said.

    Two shapes, and the role decides whether the second one counts:

    * a JSON object whose keys are ALL marker keys — always internal, from
      any role, because no human types one;
    * a bare UUID as the whole body — internal from a non-user role. A
      USER row is the user's own words; echoing back a UUID they typed is
      harmless, and blanking it would delete a real message. An assistant
      or job row that is nothing but an identifier is always a leak.
    """
    if not text:
        return False
    s = text.strip()
    if not s:
        return False
    if role != "user" and _BARE_ID_RE.match(s):
        return True
    if not (s.startswith("{") and s.endswith("}")):
        return False
    parsed = parse_job_marker(s)
    if not parsed:
        return False
    return bool(parsed) and set(parsed).issubset(JOB_MARKER_KEYS)


def public_text(role: Optional[str], content: Optional[str]) -> str:
    """The body a client may render, given the row's role.

    ``role='job'`` is ALWAYS blank — the row is a pointer, and its card is
    built from the projected fields. Every other role passes through
    unless :func:`is_internal_marker` recognises the body as a machine
    token.
    """
    if (role or "") == "job":
        return ""
    if is_internal_marker(content, role=role or "assistant"):
        # Loud on purpose. Reaching here means a writer put a marker on a
        # renderable role — the guard held, and someone should still fix
        # the writer.
        logger.warning(
            "[message_cards] blanked an internal marker on a role=%r row", role,
        )
        return ""
    return content or ""


def collect_job_ids(messages: Iterable[Any]) -> List[str]:
    """The job ids referenced by the ``role='job'`` rows in ``messages``."""
    out: List[str] = []
    for m in messages:
        if getattr(m, "role", None) != "job":
            continue
        jid = parse_job_marker(getattr(m, "content", None)).get("job_id")
        if jid and jid not in out:
            out.append(jid)
    return out


async def load_build_jobs(db, messages: Iterable[Any]) -> Dict[str, Any]:
    """``{job_id: BuildJob}`` for every card row in ``messages``.

    Never raises. ``build_jobs`` lives in the tenant DB, and these
    serializers are mounted in platform_main too — where the table is
    absent and the SELECT would 500 the whole history load for the sake of
    one card's step count. An empty map degrades to the marker's own
    ``job_name`` / ``job_status``, which is what every reader did before
    the table existed.
    """
    job_ids = collect_job_ids(messages)
    if not job_ids:
        return {}
    try:
        from sqlalchemy import select
        from app.db.models import BuildJob
        rows = (await db.execute(
            select(BuildJob).where(BuildJob.id.in_(job_ids))
        )).scalars().all()
        return {bj.id: bj for bj in rows}
    except Exception as exc:  # noqa: BLE001
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "[message_cards] BuildJob lookup skipped (%s)", str(exc)[:120],
        )
        return {}


def job_card_fields(
    message: Any,
    build_jobs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The card fields for a ``role='job'`` row. ``{}`` for any other role.

    Includes ``job_steps`` — the step list itself, not just its two
    counts. A card rebuilt from history without it can only draw a
    progress bar, which is why a reopened thread showed a bare title where
    the live card had shown the run's steps. The clients already have a
    renderer for exactly this list (it is what the live ``job_update``
    frames carry); giving history the same list is what makes the two the
    same card rather than two designs for one job.
    """
    if getattr(message, "role", None) != "job":
        return {}

    meta = parse_job_marker(getattr(message, "content", None))
    job_id = meta.get("job_id", "") or ""
    out: Dict[str, Any] = {
        "job_id": job_id,
        # The title is the card. It is generated for voice by
        # `job_titles.derive_job_title` and by the model for a typed run,
        # both through one normaliser — so the marker's copy is already
        # the string the live card showed, in the user's own language and
        # script. "App Build" is the legacy app-builder default and only
        # applies to rows that never carried a title.
        "job_name": meta.get("job_name") or "App Build",
        "job_type": meta.get("job_type"),
        # A pointer with no row behind it is a finished job whose
        # BuildJob was reaped, or a platform-mounted read with no
        # `build_jobs` table — 'completed' both times, never 'running',
        # or the card spins forever on every reload.
        "job_status": meta.get("job_status", "completed"),
    }

    bj = (build_jobs or {}).get(job_id) if job_id else None
    if bj is None:
        return out

    out["job_status"] = bj.status
    out["job_app_id"] = getattr(bj, "app_id", None)
    if out["job_name"] == "App Build" and getattr(bj, "title", None):
        out["job_name"] = bj.title.replace("Build: ", "")
    if not out.get("job_type"):
        cfg = getattr(bj, "config_json", None)
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except (json.JSONDecodeError, TypeError, ValueError):
                cfg = None
        if isinstance(cfg, dict) and cfg.get("job_type"):
            out["job_type"] = cfg["job_type"]

    try:
        steps = json.loads(bj.steps_json) if getattr(bj, "steps_json", None) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        steps = []
    if not isinstance(steps, list):
        steps = []
    steps = [s for s in steps if isinstance(s, dict)]
    out["job_steps"] = steps or None
    out["job_total_steps"] = len(steps)
    # `done`, not `completed`. A STEP is pending / running / done / failed
    # — `job_steps.finish_all_steps`, `_tool_create_job`, `apps.py` and the
    # app-builder skill all write `done`, and nothing has ever written
    # `completed` on a step; `completed` is the JOB's status, one level up.
    # Counting the wrong word is what re-rendered every finished card in
    # history as "0/3 steps" (fixed in f31d2854). Both spellings are
    # accepted so a hand-edited or future row cannot regress it the other
    # way.
    out["job_completed_steps"] = sum(
        1 for s in steps if s.get("status") in ("done", "completed")
    )
    _running = next((s for s in steps if s.get("status") == "running"), None)
    if _running and _running.get("label"):
        out["job_step"] = _running["label"]
    return out


def attach_run_to_cards(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold each run's tool records onto the card that owns them.

    The runner stamps ``job_id`` onto every ``tool_events`` record it
    persists (``StepTracker.event_fields``, Round 13 — voice records carry
    it too). Those records hold the run's SOURCES: ``domains``, ``urls``
    and the ``{title, url, domain}`` list the voice overlay draws with a
    favicon, a site and a headline. They ride the ANSWER row, because the
    answer is what the runner was writing when it had them.

    So the card knew the steps and the answer knew the sources, and the
    thread showed a titled card next to a separate row of action chips for
    the same three seconds of work. Copying the records onto the card row
    gives one artifact that carries both. It is a COPY, not a move: the
    answer row keeps its records, so a client that renders chips and not
    cards is exactly as it was and cannot regress while it ships. A client
    that renders the card decides which of the two to draw.

    Rows are the serialized rows in thread order — dicts from the
    day-chats readers, ``ChatMessageResponse`` models from the session
    ones. Both are accepted rather than forcing one shape on four call
    sites; a serializer that has to reshape its output to use the shared
    helper is a serializer that will stop using it. Pure and idempotent.
    """
    cards = {
        _get(r, "job_id"): r for r in rows
        if _get(r, "role") == "job" and _get(r, "job_id")
    }
    if not cards:
        return rows
    for row in rows:
        if _get(row, "role") == "job":
            continue
        events = _get(row, "tool_events")
        if not isinstance(events, list):
            continue
        for ev in events:
            if not isinstance(ev, dict):
                continue
            card = cards.get(ev.get("job_id"))
            if card is None:
                continue
            existing = _get(card, "tool_events")
            if not isinstance(existing, list):
                existing = []
                _set(card, "tool_events", existing)
            if not any(
                e.get("call_id") == ev.get("call_id")
                and e.get("started_at_ms") == ev.get("started_at_ms")
                for e in existing
            ):
                existing.append(ev)
    return rows


def _get(row: Any, key: str) -> Any:
    return row.get(key) if isinstance(row, dict) else getattr(row, key, None)


def _set(row: Any, key: str, value: Any) -> None:
    if isinstance(row, dict):
        row[key] = value
    else:
        setattr(row, key, value)
