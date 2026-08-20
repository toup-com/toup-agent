"""Current context — the six-layer replacement for "Working on" (v3 §6).

`you/current-context` is one of the three system files and the only one whose
body is prose rather than bullets. It answers "what is going on with this
person right now", in six layers that age downward:

    ## Today (Wed, Aug 20 — America/Toronto)
    ## Yesterday
    ## Last 2 days
    ## This week
    ## This month
    ## Past 12 months
    ### Aug 2026 …

This module owns the two things that write it, and NOTHING else writes it:

* `refresh_after_turn` — the post-turn updater. Off the response path,
  debounced, one small model call that rewrites ONLY the `Today` layer.
* `run_context_rollover` — the hourly agent-side job that ages the layers
  when the user's LOCAL date advances.

Why it is not the curator's job. The curator is the ONE writer of memory
files, and this is deliberately outside that rule: Current context is not
memory. It is a situation report that is true for a few hours, it rewrites
itself every ten minutes, and it holds nothing that would still be worth
knowing in six weeks — the exact test the curator's durability rules apply.
Routing it through `curate_turn` would put a transient paragraph under a
prompt whose whole job is to refuse transient content, and would file a
`memory_file_changes` row every ten minutes into a log that means "what the
writer changed about what it knows about you". `memory_file_ops`'
`FileState.has_prose` is the other half of the boundary: every bullet op on
this file is refused, so the curator cannot flatten the layers on its way
past, and this module writes the row directly so no change line is emitted.

Three invariants, each of which is a test:

1. **A turn never waits on this and never fails because of it.**
   `refresh_after_turn` opens its own session, swallows everything, and is
   spawned with a strong reference (a bare `create_task` can be collected
   mid-await).
2. **The debounce cursor lives in `pinned_meta_json`, not in memory.** An
   in-process cursor is reset by every container restart, and this fleet's
   median redeploy gap was 0.3 h at the 2026-08 audit — an in-memory
   debounce is no debounce at all.
3. **Rollover is idempotent per user-local day and MONOTONIC.** Three runs
   in an hour do nothing after the first; a clock that goes backwards (DST
   fall-back, a corrected timezone, a replica with a skewed clock) never
   rolls the file backwards and never duplicates a layer. One model failure
   leaves the file and the cursor untouched and retries next hour — never a
   half-rolled body.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import MemoryFile
from app.memory_files import (
    CURRENT_CONTEXT_SLUG,
    LAYER_BUDGETS,
    LAYER_LAST_2_DAYS,
    LAYER_PAST_12_MONTHS,
    LAYER_THIS_MONTH,
    LAYER_THIS_WEEK,
    LAYER_TODAY,
    LAYER_YESTERDAY,
    MAX_MONTH_PARAGRAPHS,
    MONTH_PARAGRAPH_MAX,
    PROSE_LAYERS,
    CurrentContext,
    clamp_prose,
    month_key,
    parse_current_context,
    parse_month_key,
    render_current_context,
)
from app.services import memory_file_ops as ops

logger = logging.getLogger(__name__)

#: At most one rewrite per ten minutes, unless something material changed.
REFRESH_DEBOUNCE = timedelta(minutes=10)

#: How many scheduled items and running jobs the Today prompt may name. A
#: situation report that lists twelve things is a calendar, not a report.
MAX_SCHEDULED_LINES = 6
MAX_JOB_LINES = 4

#: Archival day summaries fed to one month compression, newest first.
MAX_ARCHIVAL_DAYS = 31
ARCHIVAL_DAY_CHARS = 320

#: Distinct months one rollover may compress. Two is the structural maximum
#: (the `This month` layer's month, and the `This week` layer's when its
#: Monday fell in the month before), so three is a defensive ceiling.
MAX_MONTH_COMPRESSIONS = 3

_META_LAST_REFRESH = "last_refresh_at"
_META_LAST_ROLLOVER = "last_rollover_date"


# ── The row and its cursor ────────────────────────────────────────────

async def _context_row(db: AsyncSession, user_id: str) -> MemoryFile:
    await ops.ensure_system_files(db, user_id)
    await db.commit()
    row = (await db.execute(
        select(MemoryFile).where(and_(
            MemoryFile.user_id == user_id,
            MemoryFile.slug == CURRENT_CONTEXT_SLUG,
        ))
    )).scalar_one()
    return row


def _meta(row: MemoryFile) -> Dict[str, Any]:
    try:
        parsed = json.loads(row.pinned_meta_json) if row.pinned_meta_json else {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        # A corrupt cursor must not wedge the feature forever. Starting over
        # costs one extra rewrite; refusing to parse costs the whole file.
        logger.warning("[current_context] unreadable pinned_meta, starting over")
        return {}


async def _save(
    db: AsyncSession,
    row: MemoryFile,
    *,
    body: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist body and/or cursor. Writes NO `memory_file_changes` row.

    A context refresh is not a memory change. The Memory log is "what the
    writer changed about what it knows about you"; a file that rewrites
    itself every ten minutes would drown every real line in it.
    """
    if body is not None:
        row.body_md = body
        row.updated_at = datetime.utcnow()
    if meta is not None:
        row.pinned_meta_json = json.dumps(meta, ensure_ascii=False)
    await db.commit()


def _local_date(when: datetime, tz_name: Optional[str]) -> date:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            return when.astimezone(ZoneInfo(tz_name)).date()
        except Exception:
            pass
    return when.astimezone(timezone.utc).date()


def _local_time_label(when: Optional[datetime], tz_name: Optional[str]) -> str:
    """`5:20 PM` in the user's zone, or '' when there is no time to name."""
    if when is None:
        return ""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            when = when.astimezone(ZoneInfo(tz_name))
        except Exception:
            when = when.astimezone(timezone.utc)
    else:
        when = when.astimezone(timezone.utc)
    hour = when.hour % 12 or 12
    return f"{hour}:{when.minute:02d} {'AM' if when.hour < 12 else 'PM'}"


def today_note(when: date, tz_name: Optional[str]) -> str:
    """The `## Today (…)` parenthetical — "Wed, Aug 20 — America/Toronto".

    Parentheses are structurally forbidden inside it: the clients split a
    trailing `(…)` off a heading with `^(.*?)\\s*\\(([^()]*)\\)\\s*$`, so a
    nested pair makes the note unparseable and the layer lose its title.
    An IANA zone name never contains one.
    """
    stamp = f"{when.strftime('%a')}, {when.strftime('%b')} {when.day}"
    return f"{stamp} — {(tz_name or 'UTC').replace('(', '').replace(')', '')}"


# ══ The post-turn updater (§6) ════════════════════════════════════════

def _clean_model_prose(text: str) -> str:
    """Prose, from a model that may have reached for markdown anyway."""
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line[:2] in ("- ", "* ", "• "):
            line = line[2:].strip()
        lines.append(line)
    return " ".join(lines).strip()


def _prose_problem(text: str) -> Optional[str]:
    """Why this paragraph may not be written, or None.

    The Today layer is assembled from the day's rolling summary, which is
    itself derived from the conversation — so the never-store tier has to be
    applied here too, not only to the curator's bullets. The id and
    parameter screens come along for the same ride: `max_results=1` in a
    situation report is a tool call leaking into the product.
    """
    from app.memory_files import _HEX_ID_RE, _PARAM_RE, _UUID_RE

    if _UUID_RE.search(text) or _HEX_ID_RE.search(text):
        return "internal id in the context paragraph"
    if _PARAM_RE.search(text):
        return "tool parameter in the context paragraph"
    try:
        from app.services.memory_secrets import sensitive_content_reason

        reason = sensitive_content_reason(text)
    except Exception:  # pragma: no cover - the gate module is always present
        reason = None
    return reason or None


def build_today_prompt(
    *,
    day_summary: str,
    scheduled: Sequence[str],
    jobs: Sequence[str],
    previous: str,
    today_label: str,
) -> str:
    """The one small call that rewrites the `Today` layer.

    Two rules in here were paid for. It SUMMARISES STATE and never recaps
    the transcript: the rolling summary is deliberately not injected when
    the day's history is complete, and pasting it into Current context
    re-creates exactly the duplication that decision removed. And a routine
    is REFERENCED, never owned — the founder's acceptance criterion is that
    their 5:06 PM quote routine appears at most once, as a one-line
    reference, so the scheduled block below carries a name and a time and
    nothing else: no cron expression, no prompt text, no id.
    """
    def block(title: str, lines: Sequence[str], empty: str) -> str:
        body = "\n".join(f"- {ln}" for ln in lines) if lines else empty
        return f"{title}\n{body}\n\n"

    return (
        f"Today is {today_label}.\n\n"
        "You keep ONE short paragraph up to date: what is going on in this "
        "person's life right now. It is read at the top of every reply, so "
        "it has to be current, factual and brief.\n\n"
        f"WHAT TODAY LOOKS LIKE SO FAR:\n"
        f"{(day_summary or '').strip() or '(nothing recorded yet)'}\n\n"
        + block(
            "SCHEDULED TODAY (reference each ONCE, never copy its internals):",
            scheduled, "(nothing scheduled)",
        )
        + block("RUNNING RIGHT NOW:", jobs, "(nothing running)")
        + block(
            "WHAT THE PARAGRAPH SAYS AT THE MOMENT:",
            [previous] if (previous or "").strip() else [],
            "(nothing yet)",
        )
        + "Rewrite it. Reply with the paragraph and nothing else — no "
          "heading, no bullets, no markdown, no preamble.\n\n"
        "- 2 to 5 connected sentences, at most "
        f"{LAYER_BUDGETS[LAYER_TODAY]} characters.\n"
        "- Summarise the STATE of the day. Never recap the conversation: no "
        "\"asked about\", no \"you replied\", no quoting either side.\n"
        "- A scheduled item is one clause, mentioned ONCE: \"has soccer at "
        "5:20 PM\". Never its schedule expression, its instructions or its "
        "id, and never twice.\n"
        "- Subjectless third person, the same voice the memory files use: "
        "\"has soccer at 5:20 PM\", never \"You have soccer\".\n"
        "- Keep what is still true, drop what has finished, add what is new.\n"
        "- Nothing durable belongs here — that lives in the memory files. "
        "This paragraph is only about right now.\n"
        "- If there is genuinely nothing to say, reply with the single word "
        "NOTHING."
    )


async def _day_summary(db: AsyncSession, user_id: str, local: date) -> str:
    from app.db.models.day_chat import DayChat

    row = (await db.execute(
        select(DayChat.rolling_summary).where(and_(
            DayChat.user_id == user_id, DayChat.local_date == local,
        ))
    )).scalar_one_or_none()
    return (row or "").strip()


async def _scheduled_today(
    db: AsyncSession, user_id: str, local: date, tz_name: Optional[str]
) -> List[str]:
    """Today's routines and reminders as ONE reference line each.

    Read from the `routines` table and reduced to a name and a local time
    before anything else sees them. What is deliberately NOT read: the cron
    expression, `prompt_text`, `config_json`, `last_state_json` and the id.
    Deduped by routine id, so one arrangement can never be announced twice.
    """
    from app.db.models.routine import Routine

    rows = (await db.execute(
        select(Routine).where(and_(
            Routine.user_id == user_id, Routine.enabled.is_(True),
        ))
    )).scalars().all()

    out: List[Tuple[datetime, str]] = []
    seen: set = set()
    for row in rows:
        when = row.next_run_at or row.last_run_at
        if when is None or row.id in seen:
            continue
        if _local_date(when, tz_name) != local:
            continue
        label = ((row.reminder_text if row.kind == "reminder" else row.name)
                 or row.name or row.kind or "").strip()
        if not label:
            continue
        seen.add(row.id)
        stamp = _local_time_label(when, tz_name)
        out.append((when, f"{label[:80]}{f' at {stamp}' if stamp else ''}"))
    out.sort(key=lambda pair: pair[0])
    return [line for _, line in out[:MAX_SCHEDULED_LINES]]


async def _running_jobs(db: AsyncSession, user_id: str) -> List[str]:
    from app.db.models.app import BuildJob

    rows = (await db.execute(
        select(BuildJob.title, BuildJob.status).where(and_(
            BuildJob.user_id == user_id,
            BuildJob.status.in_(["running", "queued"]),
        )).order_by(BuildJob.created_at.desc()).limit(MAX_JOB_LINES)
    )).all()
    return [f"{(t or 'a task')[:80]} ({s})" for t, s in rows]


async def _routine_fired_since(
    db: AsyncSession, user_id: str, since: Optional[datetime]
) -> bool:
    """True when a routine or reminder has fired since the last refresh.

    This is the only way "a reminder fired" can ever be a material signal:
    the fire itself runs with `disable_post_processing=True`, so it produces
    no turn this updater is allowed to run on. The next real turn notices it
    here instead.
    """
    if since is None:
        return False
    from app.db.models.routine import Routine

    naive = since.astimezone(timezone.utc).replace(tzinfo=None)
    hit = (await db.execute(
        select(Routine.id).where(and_(
            Routine.user_id == user_id, Routine.last_run_at > naive,
        )).limit(1)
    )).scalar_one_or_none()
    return hit is not None


async def refresh_today(
    db: AsyncSession,
    user_id: str,
    *,
    material: bool = False,
    api_key: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Rewrite the `Today` layer, at most once per `REFRESH_DEBOUNCE`.

    `material=True` bypasses the debounce; the first turn of a new local day
    and a routine that fired since the last refresh do so on their own.

    Returns `{"skipped": <reason>}` when nothing was written. Raises only on
    a genuine failure — the caller (`refresh_after_turn`) is what makes this
    unable to break a turn.
    """
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    row = await _context_row(db, user_id)
    meta = _meta(row)
    tz_name = await ops.resolve_user_tz(db, user_id)
    local = _local_date(now_utc, tz_name)

    last_at = _parse_iso(meta.get(_META_LAST_REFRESH))
    if last_at is None or _local_date(last_at, tz_name) != local:
        material = True          # first turn of a new local day
    elif not material:
        material = await _routine_fired_since(db, user_id, last_at)

    if not material and last_at is not None and now_utc - last_at < REFRESH_DEBOUNCE:
        return {"skipped": "debounced", "written": False}

    previous = parse_current_context(row.body_md).get(LAYER_TODAY)

    # CLAIM before the model call, not after. Two turns can land inside the
    # same second on this path, and the cursor is the only thing that stops
    # both of them paying for a rewrite. It also bounds the cost of a failing
    # provider to one call per debounce window instead of one per turn.
    meta[_META_LAST_REFRESH] = now_utc.astimezone(timezone.utc).isoformat()
    await _save(db, row, meta=meta)

    day_summary = await _day_summary(db, user_id, local)
    scheduled = await _scheduled_today(db, user_id, local, tz_name)
    jobs = await _running_jobs(db, user_id)

    if not (day_summary or scheduled or jobs or previous):
        return {"skipped": "nothing_to_say", "written": False}

    prompt = build_today_prompt(
        day_summary=day_summary,
        scheduled=scheduled,
        jobs=jobs,
        previous=previous,
        today_label=today_note(local, tz_name),
    )
    # Release the pooled connection before the round trip — same #407
    # reasoning as `curate_turn`: this runs fire-and-forget after the reply,
    # on a path that dies routinely (a voice caller hanging up cancels the
    # parent), and a death mid-call used to leak the connection.
    await db.commit()

    from app.services.memory_curator import _llm

    response = await _llm(api_key).complete(
        messages=[{"role": "user", "content": prompt}],
        model=settings.memory_extraction_model,
        temperature=0.2,
    )
    text = _clean_model_prose(getattr(response, "content", "") or "")
    if not text or text.strip().upper().rstrip(".") == "NOTHING":
        return {"skipped": "model_said_nothing", "written": False}

    problem = _prose_problem(text)
    if problem:
        logger.warning("[current_context] refusing the rewrite: %s", problem)
        return {"skipped": problem, "written": False}

    row = await _context_row(db, user_id)
    ctx = parse_current_context(row.body_md)
    ctx.today_note = today_note(local, tz_name)
    ctx.set(LAYER_TODAY, text)
    # `meta=None`: the cursor was already claimed above, and re-writing the
    # copy read before the model call would clobber anything the hourly
    # rollover wrote while this call was in flight.
    await _save(db, row, body=render_current_context(ctx))
    logger.info(
        "[current_context] today rewritten for user=%s (%d chars, %d scheduled, "
        "%d running)", str(user_id)[:8], len(ctx.get(LAYER_TODAY)),
        len(scheduled), len(jobs),
    )
    return {"skipped": None, "written": True, "chars": len(ctx.get(LAYER_TODAY))}


# ── Fire and forget ───────────────────────────────────────────────────
# The event loop holds only a WEAK reference to a running task, so a bare
# `create_task` whose result nobody stores can be collected mid-await — the
# same hazard `agent_runner._spawn_background` exists for.

_tasks: set = set()


def spawn_refresh(session_maker, user_id: str, *, material: bool = False) -> None:
    """Start a refresh that the caller neither waits for nor can be hurt by."""
    task = asyncio.create_task(
        refresh_after_turn(session_maker, user_id, material=material)
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def refresh_after_turn(
    session_maker, user_id: str, *, material: bool = False
) -> None:
    """`refresh_today` in its own session, unable to raise.

    Off the response path by construction: a turn must never wait on the
    context paragraph and must never fail because of it.
    """
    try:
        async with session_maker() as db:
            try:
                await refresh_today(db, user_id, material=material)
            except Exception as exc:  # noqa: BLE001
                await db.rollback()
                logger.warning(
                    "[current_context] refresh failed (non-fatal): %s: %s",
                    type(exc).__name__, str(exc)[:200],
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[current_context] refresh session error: %s", exc)


# ══ Rollover (§6) ═════════════════════════════════════════════════════

def _representative_dates(prev: date) -> List[Tuple[str, date]]:
    """The date each layer's content is ABOUT, as of the last rollover.

    A layer holds a range, so its representative is a day inside that range:
    the Monday of `This week`, the 1st of `This month`. That is what lets
    the re-bucketing below be one pass of arithmetic rather than a per-day
    loop with a model call at every month boundary.
    """
    return [
        (LAYER_TODAY, prev),
        (LAYER_YESTERDAY, prev - timedelta(days=1)),
        (LAYER_LAST_2_DAYS, prev - timedelta(days=2)),
        (LAYER_THIS_WEEK, prev - timedelta(days=prev.weekday())),
        (LAYER_THIS_MONTH, prev.replace(day=1)),
    ]


def _destination(rep: date, new_date: date, source_index: int) -> int:
    """Which layer content written about `rep` belongs to on `new_date`.

    Returns an index into `PROSE_LAYERS`, or `len(PROSE_LAYERS)` for "out of
    the layers entirely — it is a month paragraph now".

    `max(..., source_index)` is what makes the rollover MONOTONIC. Without
    it, `This week` content whose Monday is one day before the new date
    classifies as `Yesterday` and the file rolls BACKWARDS — content
    getting younger, which is the one thing a rollover can never do.
    """
    delta = (new_date - rep).days
    if delta <= 0:
        by_date = 0
    elif delta == 1:
        by_date = 1
    elif delta <= 3:
        by_date = 2
    elif rep.isocalendar()[:2] == new_date.isocalendar()[:2]:
        by_date = 3
    elif (rep.year, rep.month) == (new_date.year, new_date.month):
        by_date = 4
    else:
        by_date = len(PROSE_LAYERS)
    return max(by_date, source_index)


def plan_rollover(
    ctx: CurrentContext, prev_date: date, new_date: date
) -> Tuple[CurrentContext, List[Tuple[str, str]]]:
    """Age the layers from `prev_date` to `new_date`, in ONE pass.

    Returns the new context and `[(month_label, text_to_compress)]` — the
    months whose paragraph still has to be written by a model. The caller
    applies those and ONLY then persists, so one model failure leaves the
    file exactly as it was.

    A multi-day gap is ordinary (a user away for a week) and costs the same
    single pass: every layer's content is re-bucketed against the new date
    directly, so nothing shifts one slot at a time and `Yesterday` can never
    end up back in `Today`.
    """
    rolled = CurrentContext(months=list(ctx.months), today_note=None)
    buckets: Dict[int, List[str]] = {}
    pending_months: Dict[str, List[str]] = {}

    for index, (layer, rep) in enumerate(_representative_dates(prev_date)):
        text = (ctx.get(layer) or "").strip()
        if not text:
            continue
        dest = _destination(rep, new_date, index)
        if dest >= len(PROSE_LAYERS):
            pending_months.setdefault(month_key(rep), []).append(text)
        else:
            buckets.setdefault(dest, []).append(text)

    for dest, texts in buckets.items():
        # Newest first: the sources were walked in layer order, which is
        # already newest to oldest.
        rolled.set(PROSE_LAYERS[dest], " ".join(texts))

    # Anything the writer invented outside the canonical five rides along
    # unchanged rather than being silently deleted by a rollover.
    for layer, text in ctx.layers.items():
        if layer not in PROSE_LAYERS and layer != LAYER_PAST_12_MONTHS:
            rolled.layers[layer] = text

    todo = sorted(
        ((label, " ".join(parts)) for label, parts in pending_months.items()),
        key=lambda pair: parse_month_key(pair[0]) or (0, 0),
        reverse=True,
    )
    if len(todo) > MAX_MONTH_COMPRESSIONS:
        # Structurally unreachable: every source's representative date comes
        # from `prev_date`, its month, or its ISO Monday, which span at most
        # two months. Say so rather than dropping text in silence.
        logger.warning(
            "[current_context] %d months to compress in one pass, keeping %d: %s",
            len(todo), MAX_MONTH_COMPRESSIONS, [label for label, _ in todo],
        )
    return rolled, todo[:MAX_MONTH_COMPRESSIONS]


def merge_month(ctx: CurrentContext, label: str, paragraph: str) -> None:
    """File a month paragraph, newest first, at most twelve."""
    paragraph = clamp_prose(paragraph, MONTH_PARAGRAPH_MAX)
    if not paragraph:
        return
    existing = {lbl: text for lbl, text in ctx.months}
    if label in existing:
        paragraph = clamp_prose(
            f"{existing[label]} {paragraph}", MONTH_PARAGRAPH_MAX
        )
    existing[label] = paragraph
    ctx.months = sorted(
        existing.items(),
        key=lambda pair: parse_month_key(pair[0]) or (0, 0),
        reverse=True,
    )[:MAX_MONTH_PARAGRAPHS]


def build_month_prompt(label: str, source: str, day_summaries: Sequence[str]) -> str:
    days = "\n".join(f"- {s}" for s in day_summaries) if day_summaries else "(none)"
    return (
        f"Write ONE short paragraph summarising {label} for this person, for "
        "the long-term section of their situation report.\n\n"
        f"WHAT THE MONTH'S CONTEXT PARAGRAPH SAID:\n{source.strip() or '(nothing)'}\n\n"
        f"DAY SUMMARIES FROM {label.upper()} (newest first):\n{days}\n\n"
        f"Reply with the paragraph and nothing else — no heading, no bullets, "
        f"no markdown. At most {MONTH_PARAGRAPH_MAX} characters. Name what "
        "happened and where things stand, in subjectless third person "
        "(\"sat the IELTS exam on Aug 30\", never \"You sat\"). Leave out "
        "anything that was only true for a day, and leave out ids, "
        "parameters and tool names entirely."
    )


async def _archival_days(
    db: AsyncSession, user_id: str, label: str
) -> List[str]:
    parsed = parse_month_key(label)
    if parsed is None:
        return []
    year, month = parsed
    first = date(year, month, 1)
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    from app.db.models.day_chat import DayChat

    rows = (await db.execute(
        select(DayChat.local_date, DayChat.archival_summary).where(and_(
            DayChat.user_id == user_id,
            DayChat.local_date >= first,
            DayChat.local_date < nxt,
        )).order_by(DayChat.local_date.desc()).limit(MAX_ARCHIVAL_DAYS)
    )).all()
    return [
        f"{when.isoformat()}: {summary.strip()[:ARCHIVAL_DAY_CHARS]}"
        for when, summary in rows if (summary or "").strip()
    ]


async def _compress_month(
    db: AsyncSession, user_id: str, label: str, source: str,
    api_key: Optional[str],
) -> str:
    """One model call per month crossed. Raises — the caller aborts on it."""
    from app.services.memory_curator import _llm

    prompt = build_month_prompt(label, source, await _archival_days(db, user_id, label))
    response = await _llm(api_key).complete(
        messages=[{"role": "user", "content": prompt}],
        model=settings.memory_extraction_model,
        temperature=0.2,
    )
    text = _clean_model_prose(getattr(response, "content", "") or "")
    if not text:
        raise ValueError(f"empty month summary for {label}")
    if _prose_problem(text):
        # Not a failure to retry — the model answered, the answer is not
        # storable. Fall back to the source prose, clamped.
        logger.warning("[current_context] month %s summary refused, using source", label)
        return clamp_prose(source, MONTH_PARAGRAPH_MAX)
    return clamp_prose(text, MONTH_PARAGRAPH_MAX)


async def roll_over_user(
    db: AsyncSession,
    user_id: str,
    *,
    api_key: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Advance one user's Current context to their local today.

    Idempotent per local day and monotonic. Everything is computed before
    anything is written, so a model failure during a month compression
    leaves both the body and the cursor exactly as they were and the next
    hourly pass tries again — there is no half-rolled state to recover from.
    """
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    row = await _context_row(db, user_id)
    meta = _meta(row)
    tz_name = await ops.resolve_user_tz(db, user_id)
    local = _local_date(now_utc, tz_name)

    cursor = _parse_date(meta.get(_META_LAST_ROLLOVER))
    if cursor is None:
        # First sight of this user. Plant the cursor; there is nothing to
        # age, because nothing has aged yet.
        meta[_META_LAST_ROLLOVER] = local.isoformat()
        await _save(db, row, meta=meta)
        return {"rolled": False, "reason": "cursor_planted", "day": local.isoformat()}

    if local <= cursor:
        # Same day (idempotent), or a clock that went backwards — DST
        # fall-back, a corrected timezone, a replica with a skewed clock.
        # Never roll backwards and never re-roll a day.
        return {"rolled": False, "reason": "not_a_new_day", "day": cursor.isoformat()}

    ctx = parse_current_context(row.body_md)
    rolled, todo = plan_rollover(ctx, cursor, local)

    for label, source in todo:
        merge_month(rolled, label, await _compress_month(
            db, user_id, label, source, api_key,
        ))

    rolled.today_note = today_note(local, tz_name)
    # Re-read the cursor: a post-turn refresh may have claimed
    # `last_refresh_at` while the month compressions above were in flight,
    # and this is the write that would otherwise put the stale copy back.
    await db.refresh(row)
    await _save(
        db, row,
        body=render_current_context(rolled),
        meta={**_meta(row), _META_LAST_ROLLOVER: local.isoformat()},
    )
    logger.info(
        "[current_context] rolled user=%s %s → %s (%d day(s), %d month(s) written)",
        str(user_id)[:8], cursor.isoformat(), local.isoformat(),
        (local - cursor).days, len(todo),
    )
    return {
        "rolled": True, "from": cursor.isoformat(), "to": local.isoformat(),
        "days": (local - cursor).days, "months_written": len(todo),
    }


async def run_context_rollover() -> Dict[str, Any]:
    """The hourly agent-side job. Single tenant, cheap, idempotent.

    CRON, never interval (rebuild-2026-08 RC3.1): an interval trigger's
    first fire is measured from scheduler start, and this fleet is recreated
    more often than an hourly interval would ever fire.
    """
    user_id = getattr(settings, "user_id", "") or ""
    if not user_id:
        return {"skipped": "no tenant user"}

    from app.db.database import async_session_maker

    async with async_session_maker() as db:
        try:
            return await roll_over_user(db, user_id)
        except Exception as exc:  # noqa: BLE001
            await db.rollback()
            logger.warning(
                "[current_context] rollover failed, retrying next hour: %s: %s",
                type(exc).__name__, str(exc)[:200],
            )
            return {"skipped": f"{type(exc).__name__}"}


# ── Small helpers ─────────────────────────────────────────────────────

def _parse_iso(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_date(raw: Any) -> Optional[date]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None
