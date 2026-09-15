"""Day-chat rebucketing — a pure planner and a transactional applier.

Why a service and not thirty lines inside `GET /api/day-chats`: the same
work has three callers (the list endpoint's self-heal, the ws_chat
timezone-learn hook, and an operator running
`scripts/rebucket_day_chats.py`), and the version that lived inside the
endpoint got every hard rule wrong at once.

The rules that are load-bearing, each of which was a defect on
2026-09-14:

* Targets are PER MESSAGE, never "today" and never one date for a whole
  conversation. A conversation that straddles local midnight has no
  single answer, and `Message.day_chat_id` is the canonical day
  membership (see the comment at `api/day_chats.py`'s channels query).
* Dependents move BEFORE the parent is touched. `context_budget_logs`
  references `day_chats(id)` with no ON DELETE, and it is the one
  dependent the old heal never re-pointed — its delete raised a
  ForeignKeyViolation on the REQUEST's session, which then 500'd the
  whole day index.
* A day is deleted only when three dependent counts read zero IN THE
  SAME TRANSACTION, and only while its `local_date` is still in the
  future. A day that has become today is a legitimate calendar day, and
  deleting it strands the process-wide `_day_chat_cache` entry that
  already points at it — every later INSERT in that process then
  violates `messages.day_chat_id`.
* Both the source and the target day's summaries become lies the moment
  messages move across them, so both are marked stale.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date as Date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sqlalchemy import delete as sa_delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.conversation_resolver import INDEXED_SYSTEM_CHANNELS
from app.db.models import Conversation, Message
from app.db.models.day_chat import ContextBudgetLog, DayChat

logger = logging.getLogger(__name__)

_CHUNK = 500


# ── Pure inputs ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class MsgRow:
    id: str
    created_at: Optional[datetime]
    day_chat_id: Optional[str]
    conversation_id: Optional[str]


@dataclass(frozen=True)
class ConvRow:
    id: str
    day_chat_id: Optional[str]
    channel: Optional[str]
    is_active: bool


@dataclass(frozen=True)
class CblRow:
    id: str
    day_chat_id: Optional[str]
    created_at: Optional[datetime]
    turn_id: Optional[str]


@dataclass(frozen=True)
class DayRow:
    id: str
    local_date: Date


@dataclass(frozen=True)
class RebucketPlan:
    message_moves: Dict[str, Date] = field(default_factory=dict)
    conversation_moves: Dict[str, Date] = field(default_factory=dict)
    # Moves the partial unique index would reject. Landing them with
    # is_active=False takes the row out of the index predicate without
    # deleting anything.
    conversation_deactivate: Dict[str, Date] = field(default_factory=dict)
    cbl_moves: Dict[str, Date] = field(default_factory=dict)
    days_needed: Set[Date] = field(default_factory=set)
    days_emptied: Set[str] = field(default_factory=set)
    days_deletable: Set[str] = field(default_factory=set)
    days_retained: Dict[str, str] = field(default_factory=dict)
    noop: bool = False


@dataclass
class RebucketResult:
    changed: bool = False
    moved_messages: int = 0
    moved_conversations: int = 0
    deactivated: int = 0
    moved_cbls: int = 0
    days_created: List[str] = field(default_factory=list)
    days_deleted: List[str] = field(default_factory=list)
    days_retained: Dict[str, str] = field(default_factory=dict)
    per_date_errors: Dict[str, str] = field(default_factory=dict)
    before_after: List[Tuple[str, str, str]] = field(default_factory=list)


def _load_tz(tz_name: Optional[str]):
    """The zone, or None. Never guesses UTC — a UTC guess is what created
    the future-dated day this module exists to repair."""
    if not tz_name:
        return None
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tz_name)
    except Exception:
        return None


def target_day(created_at: Optional[datetime], tz, local_today: Date) -> Optional[Date]:
    """The local date a stored row belongs to.

    Clamped at ``local_today``: a stored row can never belong to a future
    day, and agent containers have been observed running minutes fast.
    """
    if created_at is None:
        return None
    dt = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    return min(dt.astimezone(tz).date(), local_today)


def plan_rebucket(
    *,
    days: Sequence[DayRow],
    messages: Sequence[MsgRow],
    conversations: Sequence[ConvRow],
    cbls: Sequence[CblRow],
    tz_name: Optional[str],
    now_utc: datetime,
    scope_day_ids: Iterable[str],
) -> RebucketPlan:
    """Decide every move without touching a database."""
    tz = _load_tz(tz_name)
    if tz is None:
        return RebucketPlan(noop=True)

    now = now_utc if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
    local_today = now.astimezone(tz).date()

    scope: Set[str] = {d for d in scope_day_ids if d}
    day_date: Dict[str, Date] = {d.id: d.local_date for d in days}
    date_day: Dict[Date, str] = {d.local_date: d.id for d in days}

    message_moves: Dict[str, Date] = {}
    msg_target: Dict[str, Date] = {}          # every in-scope message, moved or not
    msg_by_conv: Dict[str, List[MsgRow]] = {}
    unresolved_days: Dict[str, str] = {}      # day_id -> reason it cannot empty

    for m in messages:
        if m.day_chat_id not in scope:
            continue
        msg_by_conv.setdefault(m.conversation_id or "", []).append(m)
        tgt = target_day(m.created_at, tz, local_today)
        if tgt is None:
            unresolved_days.setdefault(
                m.day_chat_id, "message with no created_at cannot be dated"
            )
            continue
        msg_target[m.id] = tgt
        if day_date.get(m.day_chat_id) != tgt:
            message_moves[m.id] = tgt

    conversation_moves: Dict[str, Date] = {}
    conversation_deactivate: Dict[str, Date] = {}

    # Occupancy of the partial unique index (user_id, day_chat_id, channel)
    # WHERE channel IN INDEXED_SYSTEM_CHANNELS AND is_active — seeded from
    # the rows that are NOT moving, then extended as moves are granted.
    occupied: Set[Tuple[Date, str]] = set()
    for c in conversations:
        if c.day_chat_id in scope:
            continue
        d = day_date.get(c.day_chat_id)
        if d is not None and c.is_active and (c.channel or "") in INDEXED_SYSTEM_CHANNELS:
            occupied.add((d, c.channel or ""))

    for c in sorted(conversations, key=lambda r: r.id):
        if c.day_chat_id not in scope:
            continue
        own = msg_by_conv.get(c.id, [])
        targets = {msg_target[m.id] for m in own if m.id in msg_target}
        if len(targets) != 1:
            # Zero: nothing datable on it. More than one: the conversation
            # straddles local midnight and Message.day_chat_id is canonical,
            # so it needs no conversation-level answer.
            if own:
                unresolved_days.setdefault(
                    c.day_chat_id,
                    "conversation %s spans %d local dates" % (c.id[:8], len(targets)),
                )
            else:
                unresolved_days.setdefault(
                    c.day_chat_id,
                    "conversation %s has no datable messages" % c.id[:8],
                )
            continue
        tgt = next(iter(targets))
        if day_date.get(c.day_chat_id) == tgt:
            continue
        if (
            c.is_active
            and (c.channel or "") in INDEXED_SYSTEM_CHANNELS
            and (tgt, c.channel or "") in occupied
        ):
            conversation_deactivate[c.id] = tgt
        else:
            conversation_moves[c.id] = tgt
            if c.is_active and (c.channel or "") in INDEXED_SYSTEM_CHANNELS:
                occupied.add((tgt, c.channel or ""))

    cbl_moves: Dict[str, Date] = {}
    for b in cbls:
        if b.day_chat_id not in scope:
            continue
        tgt: Optional[Date] = None
        if b.turn_id and b.turn_id in message_moves:
            tgt = message_moves[b.turn_id]
        elif b.turn_id and b.turn_id in msg_target:
            tgt = msg_target[b.turn_id]
        else:
            tgt = target_day(b.created_at, tz, local_today)
        if tgt is None:
            unresolved_days.setdefault(
                b.day_chat_id, "context_budget_log %s cannot be dated" % b.id[:8]
            )
            continue
        if day_date.get(b.day_chat_id) != tgt:
            cbl_moves[b.id] = tgt

    days_needed = {
        d
        for d in set(message_moves.values())
        | set(conversation_moves.values())
        | set(conversation_deactivate.values())
        | set(cbl_moves.values())
        if d not in date_day
    }

    # A scope day empties only when EVERY dependent leaves it.
    days_emptied: Set[str] = set()
    days_retained: Dict[str, str] = dict(unresolved_days)
    for day_id in scope:
        if day_id in days_retained:
            continue
        stayed_msgs = [
            m for m in messages if m.day_chat_id == day_id and m.id not in message_moves
        ]
        stayed_convs = [
            c
            for c in conversations
            if c.day_chat_id == day_id
            and c.id not in conversation_moves
            and c.id not in conversation_deactivate
        ]
        stayed_cbls = [
            b for b in cbls if b.day_chat_id == day_id and b.id not in cbl_moves
        ]
        if stayed_msgs or stayed_convs or stayed_cbls:
            days_retained[day_id] = (
                "still holds %d messages, %d conversations, %d budget logs"
                % (len(stayed_msgs), len(stayed_convs), len(stayed_cbls))
            )
            continue
        days_emptied.add(day_id)

    days_deletable = {
        d for d in days_emptied if (day_date.get(d) or local_today) > local_today
    }
    for d in days_emptied - days_deletable:
        days_retained[d] = "emptied but local_date is no longer in the future"

    noop = not (
        message_moves
        or conversation_moves
        or conversation_deactivate
        or cbl_moves
        or days_needed
        or days_deletable
    )

    return RebucketPlan(
        message_moves=message_moves,
        conversation_moves=conversation_moves,
        conversation_deactivate=conversation_deactivate,
        cbl_moves=cbl_moves,
        days_needed=days_needed,
        days_emptied=days_emptied,
        days_deletable=days_deletable,
        days_retained=days_retained,
        noop=noop,
    )


# ── Transactional applier ────────────────────────────────────────────


async def _upsert_day(
    db: AsyncSession, user_id: str, local_date: Date, tz_name: Optional[str]
) -> Tuple[str, bool]:
    """(day_chat_id, created). Mirrors `day_chat_resolver.get_or_create_day_chat`
    exactly, including the SQLite fallback branch — two writers of the same
    row must not disagree about how it is written."""
    existing = (
        await db.execute(
            select(DayChat).where(
                DayChat.user_id == user_id, DayChat.local_date == local_date
            )
        )
    ).scalar_one_or_none()
    if existing:
        return existing.id, False

    new_id = str(uuid.uuid4())
    now_naive = datetime.utcnow()
    try:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = (
            pg_insert(DayChat.__table__)
            .values(
                id=new_id,
                user_id=user_id,
                local_date=local_date,
                timezone=tz_name or "UTC",
                started_at=now_naive,
                last_message_at=now_naive,
                message_count=0,
                total_tokens=0,
                summary_status="stale",
            )
            .on_conflict_do_nothing(index_elements=["user_id", "local_date"])
        )
        await db.execute(stmt)
        await db.flush()
    except Exception:
        # SQLite (and any dialect without ON CONFLICT) — fall through to the
        # SELECT + direct insert below.
        pass

    row = (
        await db.execute(
            select(DayChat).where(
                DayChat.user_id == user_id, DayChat.local_date == local_date
            )
        )
    ).scalar_one_or_none()
    if row:
        return row.id, True

    dc = DayChat(
        id=new_id,
        user_id=user_id,
        local_date=local_date,
        timezone=tz_name or "UTC",
        started_at=now_naive,
        last_message_at=now_naive,
        message_count=0,
        total_tokens=0,
        summary_status="stale",
    )
    db.add(dc)
    await db.flush()
    return dc.id, True


async def _count_dependents(db: AsyncSession, day_id: str) -> Tuple[int, int, int]:
    msgs = (
        await db.execute(
            select(func.count()).select_from(Message).where(Message.day_chat_id == day_id)
        )
    ).scalar() or 0
    convs = (
        await db.execute(
            select(func.count())
            .select_from(Conversation)
            .where(Conversation.day_chat_id == day_id)
        )
    ).scalar() or 0
    try:
        # SAVEPOINT: on Postgres a failed statement aborts the whole
        # transaction, so an absent context_budget_logs table would take
        # every other date's committed work with it.
        async with db.begin_nested():
            cbls = (
                await db.execute(
                    select(func.count())
                    .select_from(ContextBudgetLog)
                    .where(ContextBudgetLog.day_chat_id == day_id)
                )
            ).scalar() or 0
    except Exception:
        # No context_budget_logs table (older tenant, or a test schema). An
        # unknown dependent count is NOT zero — refuse the delete.
        cbls = 1
    return int(msgs), int(convs), int(cbls)


async def apply_rebucket(
    db: AsyncSession,
    user_id: str,
    plan: RebucketPlan,
    *,
    logger: Optional[logging.Logger] = None,
    tz_name: Optional[str] = None,
) -> RebucketResult:
    """Execute ``plan``. One SAVEPOINT per TARGET DATE, so one bad date
    cannot take the others with it."""
    log = logger or globals()["logger"]
    result = RebucketResult(days_retained=dict(plan.days_retained))
    if plan.noop:
        return result

    # Where each row sits today — needed for the before/after trail and for
    # the per-(source, target) log line.
    day_of_message: Dict[str, Optional[str]] = {}
    if plan.message_moves:
        for mid, did in (
            await db.execute(
                select(Message.id, Message.day_chat_id).where(
                    Message.id.in_(list(plan.message_moves.keys()))
                )
            )
        ).all():
            day_of_message[mid] = did

    by_date: Dict[Date, dict] = {}
    for mid, d in plan.message_moves.items():
        by_date.setdefault(d, {"msgs": [], "convs": [], "deact": [], "cbls": []})["msgs"].append(mid)
    for cid, d in plan.conversation_moves.items():
        by_date.setdefault(d, {"msgs": [], "convs": [], "deact": [], "cbls": []})["convs"].append(cid)
    for cid, d in plan.conversation_deactivate.items():
        by_date.setdefault(d, {"msgs": [], "convs": [], "deact": [], "cbls": []})["deact"].append(cid)
    for bid, d in plan.cbl_moves.items():
        by_date.setdefault(d, {"msgs": [], "convs": [], "deact": [], "cbls": []})["cbls"].append(bid)
    for d in plan.days_needed:
        by_date.setdefault(d, {"msgs": [], "convs": [], "deact": [], "cbls": []})

    touched_days: Set[str] = set()
    # Per (source_day_id, target_day_id) tallies for the structured trail.
    pair_counts: Dict[Tuple[str, str], List[int]] = {}

    for target_date in sorted(by_date.keys()):
        work = by_date[target_date]
        try:
            async with db.begin_nested():
                target_id, created = await _upsert_day(db, user_id, target_date, tz_name)
                if created:
                    result.days_created.append(target_id)
                touched_days.add(target_id)

                # ORDER IS LOAD-BEARING: dependents first, parent last.
                for i in range(0, len(work["msgs"]), _CHUNK):
                    chunk = work["msgs"][i : i + _CHUNK]
                    await db.execute(
                        update(Message)
                        .where(Message.id.in_(chunk))
                        .values(day_chat_id=target_id)
                    )
                for i in range(0, len(work["convs"]), _CHUNK):
                    chunk = work["convs"][i : i + _CHUNK]
                    await db.execute(
                        update(Conversation)
                        .where(Conversation.id.in_(chunk))
                        .values(day_chat_id=target_id)
                    )
                for i in range(0, len(work["deact"]), _CHUNK):
                    chunk = work["deact"][i : i + _CHUNK]
                    await db.execute(
                        update(Conversation)
                        .where(Conversation.id.in_(chunk))
                        .values(day_chat_id=target_id, is_active=False)
                    )
                if work["cbls"]:
                    for i in range(0, len(work["cbls"]), _CHUNK):
                        chunk = work["cbls"][i : i + _CHUNK]
                        await db.execute(
                            update(ContextBudgetLog)
                            .where(ContextBudgetLog.id.in_(chunk))
                            .values(day_chat_id=target_id)
                        )
        except Exception as e:
            result.per_date_errors[target_date.isoformat()] = repr(e)
            log.warning(
                "[rebucket] user=%s target=%s FAILED (%r) — other dates unaffected",
                user_id[:8], target_date, e,
            )
            continue

        result.moved_messages += len(work["msgs"])
        result.moved_conversations += len(work["convs"])
        result.deactivated += len(work["deact"])
        result.moved_cbls += len(work["cbls"])
        for mid in work["msgs"]:
            src = day_of_message.get(mid) or "-"
            result.before_after.append((mid, src, target_id))
            touched_days.add(src)
            pair_counts.setdefault((src, target_id), [0, 0, 0])[0] += 1
        for cid in work["convs"] + work["deact"]:
            pair_counts.setdefault(("-", target_id), [0, 0, 0])[1] += 1
        for _bid in work["cbls"]:
            pair_counts.setdefault(("-", target_id), [0, 0, 0])[2] += 1

    # Deletes — only for days the plan marked deletable, and only once the
    # three dependent counts read zero in THIS transaction. `sa_delete`, not
    # `db.delete(orm_obj)`: the ORM path de-associates the `conversations`
    # relationship and leaves messages/context_budget_logs untouched, which
    # is exactly the asymmetry that made the old heal look uniform.
    #
    # Each delete sits in its OWN savepoint. Until the commit below, every
    # move above is still uncommitted on this same transaction; on Postgres
    # one failed statement here (a live turn inserting into the source day
    # between the count and the delete → ForeignKeyViolation) would abort
    # the transaction and roll back every date's moves with
    # `per_date_errors` empty. A delete that fails now costs that one day
    # its cleanup and nothing else.
    for day_id in sorted(plan.days_deletable):
        deleted = False
        try:
            async with db.begin_nested():
                msgs, convs, cbls = await _count_dependents(db, day_id)
                if msgs or convs or cbls:
                    result.days_retained[day_id] = (
                        "dependents still present at delete time (m=%d c=%d b=%d)"
                        % (msgs, convs, cbls)
                    )
                else:
                    # `user_id` predicate is defence in depth: the plan only
                    # ever names this user's days, and the row must still be
                    # unreachable from any other user's repair.
                    await db.execute(
                        sa_delete(DayChat).where(
                            DayChat.id == day_id, DayChat.user_id == user_id,
                        )
                    )
                    deleted = True
        except Exception as e:  # noqa: BLE001
            result.days_retained[day_id] = f"delete failed ({e!r})"
            log.warning(
                "[rebucket] user=%s day=%s delete FAILED (%r) — moves unaffected",
                user_id[:8], day_id[:8], e,
            )
            continue
        if deleted:
            result.days_deleted.append(day_id)
            touched_days.discard(day_id)

    # Counters + summaries for every day the plan touched. Both sides are
    # wrong after a move: older messages arrived BEHIND the summary
    # watermark on the target, and the source's summary describes rows that
    # are no longer there. Same per-day savepoint, same reason as above; a
    # counter that could not be refreshed is advisory and is logged.
    for day_id in sorted(touched_days):
        if not day_id or day_id == "-":
            continue
        try:
            async with db.begin_nested():
                count = (
                    await db.execute(
                        select(func.count()).select_from(Message).where(Message.day_chat_id == day_id)
                    )
                ).scalar() or 0
                last_at = (
                    await db.execute(
                        select(func.max(Message.created_at)).where(Message.day_chat_id == day_id)
                    )
                ).scalar()
                values = {
                    "message_count": int(count),
                    "summary_status": "stale",
                    "rolling_summary": None,
                    "summary_up_to_message_id": None,
                    "archival_summary_status": "not_needed",
                }
                if last_at is not None:
                    values["last_message_at"] = last_at
                await db.execute(update(DayChat).where(DayChat.id == day_id).values(**values))
        except Exception as e:  # noqa: BLE001
            result.per_date_errors[f"counters:{day_id[:8]}"] = repr(e)
            log.warning(
                "[rebucket] user=%s day=%s counter refresh FAILED (%r) — moves unaffected",
                user_id[:8], day_id[:8], e,
            )

    await db.commit()

    result.changed = bool(
        result.moved_messages
        or result.moved_conversations
        or result.deactivated
        or result.moved_cbls
        or result.days_created
        or result.days_deleted
    )

    for (src, dst), (m, c, b) in sorted(pair_counts.items()):
        log.info(
            "[rebucket] user=%s src=%s -> dst=%s msgs=%d convs=%d cbls=%d deleted=%s",
            user_id[:8], (src or "-")[:8], (dst or "-")[:8], m, c, b,
            src in result.days_deleted,
        )
    for day_id, reason in sorted(result.days_retained.items()):
        log.info("[rebucket] user=%s retained=%s reason=%s", user_id[:8], day_id[:8], reason)

    if result.changed:
        # A deleted or newly created day id must not survive in any worker's
        # (user_id, local_date) LRU — a cache hit on a deleted row FK-fails
        # every subsequent message INSERT in that process.
        try:
            from app.agent import _day_chat_cache

            _day_chat_cache.invalidate_user(user_id)
        except Exception:
            log.warning("[rebucket] day-chat cache invalidation failed", exc_info=True)

    return result


def rebucket_enabled() -> bool:
    """The kill switch (`settings.day_chat_rebucket_enabled`). Read through
    `getattr` so an older Settings simply behaves as ON."""
    try:
        from app.config import settings as _settings
        return bool(getattr(_settings, "day_chat_rebucket_enabled", True))
    except Exception:  # pragma: no cover
        return True


async def rebucket_user_days(
    db: AsyncSession,
    user_id: str,
    tz_name: Optional[str],
    *,
    scope: str = "future",
    now_utc: Optional[datetime] = None,
    limit_days: int = 90,
    force: bool = False,
) -> RebucketResult:
    """Load, plan, apply. ``scope='future'`` repairs only days whose
    ``local_date`` is still ahead of the user's local today; ``'all'``
    re-derives every loaded day (the operator script's mode).

    Honours the kill switch unless ``force`` (the operator's explicit
    override): every automatic caller — the endpoint heal, the ws_chat
    tz-learn trigger — reaches the database only through here, so one env
    flip stops them all without a rollout."""
    from app.services import health_signals

    if not force and not rebucket_enabled():
        logger.warning("[rebucket] disabled by day_chat_rebucket_enabled — no-op user=%s",
                       (user_id or "")[:8])
        return RebucketResult()

    health_signals.incr("rebucket_runs")
    try:
        tz = _load_tz(tz_name)
        if tz is None:
            return RebucketResult()

        now = now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        local_today = now.astimezone(tz).date()

        day_rows = (
            await db.execute(
                select(DayChat.id, DayChat.local_date)
                .where(DayChat.user_id == user_id)
                .order_by(DayChat.local_date.desc())
                .limit(limit_days)
            )
        ).all()
        days = [DayRow(id=r[0], local_date=r[1]) for r in day_rows]
        if not days:
            return RebucketResult()

        if scope == "all":
            scope_ids = {d.id for d in days}
        else:
            scope_ids = {d.id for d in days if d.local_date > local_today}
        if not scope_ids:
            return RebucketResult()

        msg_rows = (
            await db.execute(
                select(
                    Message.id,
                    Message.created_at,
                    Message.day_chat_id,
                    Message.conversation_id,
                ).where(Message.day_chat_id.in_(sorted(scope_ids)))
            )
        ).all()
        messages = [
            MsgRow(id=r[0], created_at=r[1], day_chat_id=r[2], conversation_id=r[3])
            for r in msg_rows
        ]

        conv_rows = (
            await db.execute(
                select(
                    Conversation.id,
                    Conversation.day_chat_id,
                    Conversation.channel,
                    Conversation.is_active,
                ).where(Conversation.user_id == user_id)
            )
        ).all()
        conversations = [
            ConvRow(id=r[0], day_chat_id=r[1], channel=r[2], is_active=bool(r[3]))
            for r in conv_rows
        ]

        cbls: List[CblRow] = []
        try:
            cbl_rows = (
                await db.execute(
                    select(
                        ContextBudgetLog.id,
                        ContextBudgetLog.day_chat_id,
                        ContextBudgetLog.created_at,
                        ContextBudgetLog.turn_id,
                    ).where(ContextBudgetLog.day_chat_id.in_(sorted(scope_ids)))
                )
            ).all()
            cbls = [
                CblRow(id=r[0], day_chat_id=r[1], created_at=r[2], turn_id=r[3])
                for r in cbl_rows
            ]
        except Exception as e:
            # Table absent (old tenant / test schema). Roll the failed read
            # back before planning — a poisoned session is the defect this
            # whole module exists to remove.
            await db.rollback()
            logger.info("[rebucket] context_budget_logs unreadable (%r) — treating as empty", e)

        plan = plan_rebucket(
            days=days,
            messages=messages,
            conversations=conversations,
            cbls=cbls,
            tz_name=tz_name,
            now_utc=now,
            scope_day_ids=scope_ids,
        )
        if plan.noop:
            return RebucketResult(days_retained=dict(plan.days_retained))

        result = await apply_rebucket(db, user_id, plan, tz_name=tz_name)
        if result.per_date_errors:
            health_signals.incr("rebucket_failures", len(result.per_date_errors))
        return result
    except Exception:
        health_signals.incr("rebucket_failures")
        raise
