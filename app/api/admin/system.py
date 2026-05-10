"""
Admin — System Management (memory, bot dashboard, gateway control)

Moved from app/api/admin.py → app/api/admin/system.py for clean organisation.
Prefixed under /admin — same endpoints, just a cleaner home.
"""

import time
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, case

from app.db import get_db, Memory
from app.api.auth import get_current_user
from app.api.memories import memory_to_response
from app.schemas import MemoryResponse
from app.services.decay_service import get_decay_service
from app.services.consolidation_service import get_consolidation_service

router = APIRouter(prefix="/admin", tags=["Admin — System"])


# ============ Response Models ============

class DecayResult(BaseModel):
    memories_processed: int
    memories_updated: int
    message: str


class ConsolidationResult(BaseModel):
    memories_considered: int
    groups_found: int
    memories_consolidated: int
    message: str


class WeakMemoriesResponse(BaseModel):
    memories: List[MemoryResponse]
    total_count: int
    threshold: float


class ReviewSuggestionsResponse(BaseModel):
    memories: List[MemoryResponse]
    total_count: int


class MemoryHealthStats(BaseModel):
    total_memories: int
    avg_strength: float
    weak_memories_count: int
    strong_memories_count: int
    episodic_count: int
    semantic_count: int
    procedural_count: int
    meta_count: int


# F6 — per-user memory health snapshot for ops debugging.
# Distinct from /admin/health (which scopes to current_user). This one
# takes a user_id and returns the same per-turn signals the runtime
# emits via the [memory_health] log line, plus a 7-day summarizer
# rollup so operators can answer "is memory working for this user?"
# in one HTTP call.

class DaySummarizerStateRow(BaseModel):
    local_date: str  # ISO date
    summary_status: str
    summary_failure_count: int
    summary_last_failure_at: Optional[str]  # ISO datetime
    summary_last_failure_reason: Optional[str]
    archival_summary_status: str
    has_rolling_summary: bool
    has_archival_summary: bool
    message_count: int


class ActiveTaskRow(BaseModel):
    id: str
    content: str
    strength: float
    created_at: Optional[str]
    last_reinforced_at: Optional[str]


class MemoryHealthForUser(BaseModel):
    """Snapshot of memory subsystem state for one user.

    Mirrors the per-turn [memory_health] log line plus aggregates an
    operator would otherwise have to grep across days.
    """
    user_id: str
    user_email: Optional[str]
    # Memory bank
    total_memories: int
    active_memory_count: int
    avg_strength: float
    weak_count: int
    strong_count: int
    by_category: dict
    # Active tasks (open threads — F4)
    active_tasks_live: int
    active_tasks_archived: int
    active_task_samples: List[ActiveTaskRow]
    # Day summarizer state — last 7 days
    summarizer_recent_days: List[DaySummarizerStateRow]
    summarizer_failure_streak: int
    summarizer_last_failure_reason: Optional[str]
    # Day-boundary recap (F8) — would the next "fresh" turn surface anything?
    recent_days_with_archival: int
    recent_days_with_rolling_only: int


# ============ Memory Endpoints ============

@router.post("/decay", response_model=DecayResult)
async def trigger_decay(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    decay_service = get_decay_service(db)
    processed, updated = await decay_service.apply_decay_to_user(current_user.id)
    return DecayResult(
        memories_processed=processed,
        memories_updated=updated,
        message=f"Decay applied: {updated} of {processed} memories had strength reduced",
    )


@router.post("/consolidate", response_model=ConsolidationResult)
async def trigger_consolidation(
    force: bool = False,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    consolidation_service = get_consolidation_service(db)
    considered, groups, consolidated = await consolidation_service.run_consolidation(
        current_user.id, force=force
    )
    return ConsolidationResult(
        memories_considered=considered,
        groups_found=groups,
        memories_consolidated=consolidated,
        message=f"Consolidation complete: {consolidated} memories consolidated into {groups} groups",
    )


@router.get("/weak-memories", response_model=WeakMemoriesResponse)
async def get_weak_memories(
    threshold: float = 0.3,
    limit: int = 50,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    decay_service = get_decay_service(db)
    memories = await decay_service.get_weak_memories(
        current_user.id, threshold=threshold, limit=limit
    )
    return WeakMemoriesResponse(
        memories=[memory_to_response(m) for m in memories],
        total_count=len(memories),
        threshold=threshold,
    )


@router.get("/review-suggestions", response_model=ReviewSuggestionsResponse)
async def get_review_suggestions(
    limit: int = 10,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    decay_service = get_decay_service(db)
    memories = await decay_service.get_memories_to_review(current_user.id, limit=limit)
    return ReviewSuggestionsResponse(
        memories=[memory_to_response(m) for m in memories],
        total_count=len(memories),
    )


@router.post("/memories/{memory_id}/reinforce", response_model=MemoryResponse)
async def reinforce_memory(
    memory_id: str,
    context: str = "manual",
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    decay_service = get_decay_service(db)
    memory = await decay_service.reinforce_memory(
        memory_id, current_user.id, access_context=context
    )
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    return memory_to_response(memory)


@router.post("/memories/{memory_id}/promote", response_model=MemoryResponse)
async def promote_to_semantic(
    memory_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    consolidation_service = get_consolidation_service(db)
    memory = await consolidation_service.promote_to_semantic(memory_id, current_user.id)
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    return memory_to_response(memory)


@router.get("/health", response_model=MemoryHealthStats)
async def get_memory_health(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(
            func.count(Memory.id).label("total"),
            func.avg(Memory.strength).label("avg_strength"),
            func.sum(case((Memory.strength < 0.3, 1), else_=0)).label("weak_count"),
            func.sum(case((Memory.strength >= 0.8, 1), else_=0)).label("strong_count"),
            func.sum(case((Memory.memory_level == "episodic", 1), else_=0)).label("episodic_count"),
            func.sum(case((Memory.memory_level == "semantic", 1), else_=0)).label("semantic_count"),
            func.sum(case((Memory.memory_level == "procedural", 1), else_=0)).label("procedural_count"),
            func.sum(case((Memory.memory_level == "meta", 1), else_=0)).label("meta_count"),
        ).where(and_(Memory.user_id == current_user.id, Memory.is_deleted == False))
    )
    row = result.first()
    return MemoryHealthStats(
        total_memories=row.total or 0,
        avg_strength=round(row.avg_strength or 0.0, 3),
        weak_memories_count=row.weak_count or 0,
        strong_memories_count=row.strong_count or 0,
        episodic_count=row.episodic_count or 0,
        semantic_count=row.semantic_count or 0,
        procedural_count=row.procedural_count or 0,
        meta_count=row.meta_count or 0,
    )


# ============ Bot Dashboard Endpoints ============

_telegram_bot_ref = None
_cron_service_ref = None
_start_time_ref = None


def set_bot_refs(telegram_bot, cron_service=None, start_time=None):
    """Called from main.py to inject references for the admin dashboard."""
    global _telegram_bot_ref, _cron_service_ref, _start_time_ref
    _telegram_bot_ref = telegram_bot
    _cron_service_ref = cron_service
    _start_time_ref = start_time or time.time()


@router.get("/bot/status")
async def get_bot_status(current_user=Depends(get_current_user)):
    if not _telegram_bot_ref:
        return {"status": "offline", "message": "Bot not running"}
    from app.config import settings

    uptime_sec = time.time() - (_start_time_ref or 0)
    hours, rem = divmod(int(uptime_sec), 3600)
    mins, secs = divmod(rem, 60)
    return {
        "status": "online",
        "uptime": f"{hours}h {mins}m {secs}s",
        "uptime_seconds": int(uptime_sec),
        "model": settings.agent_model,
        "fallback_model": settings.agent_fallback_model,
        "active_sessions": len(_telegram_bot_ref._session_map),
        "known_users": len(_telegram_bot_ref._user_map),
    }


@router.get("/bot/sessions")
async def get_bot_sessions(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not _telegram_bot_ref:
        return {"sessions": []}
    from app.db.models import TelegramUserMapping

    sessions = []
    for chat_id, session_id in _telegram_bot_ref._session_map.items():
        tg_name = None
        for tg_id, uid in _telegram_bot_ref._user_map.items():
            if tg_id == chat_id or True:
                result = await db.execute(
                    select(TelegramUserMapping).where(TelegramUserMapping.telegram_id == tg_id)
                )
                mapping = result.scalar_one_or_none()
                if mapping:
                    tg_name = mapping.telegram_name or mapping.telegram_username
                break
        sessions.append({
            "chat_id": chat_id,
            "session_id": session_id[:8] + "..." if session_id else None,
            "telegram_name": tg_name,
        })
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/bot/usage")
async def get_bot_usage(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.db.models import Conversation, TelegramUserMapping

    result = await db.execute(
        select(
            Conversation.user_id,
            func.sum(Conversation.total_tokens).label("total_tokens"),
            func.count(Conversation.id).label("session_count"),
        )
        .group_by(Conversation.user_id)
        .order_by(func.sum(Conversation.total_tokens).desc())
    )
    rows = result.all()
    usage = []
    for row in rows:
        mapping_result = await db.execute(
            select(TelegramUserMapping).where(TelegramUserMapping.user_id == row.user_id)
        )
        mapping = mapping_result.scalar_one_or_none()
        usage.append({
            "user_id": row.user_id[:8] + "...",
            "telegram_name": mapping.telegram_name if mapping else None,
            "telegram_username": mapping.telegram_username if mapping else None,
            "total_tokens": row.total_tokens or 0,
            "session_count": row.session_count,
        })
    return {"usage": usage, "total_users": len(usage)}


@router.get("/bot/cron")
async def get_bot_cron_jobs(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.db.models import CronJob

    result = await db.execute(select(CronJob).order_by(CronJob.created_at.desc()))
    jobs = result.scalars().all()
    return {
        "jobs": [
            {
                "id": str(j.id)[:8],
                "user_id": j.user_id[:8] + "...",
                "name": j.name,
                "schedule": j.schedule_expression,
                "schedule_type": j.schedule_type,
                "enabled": j.enabled,
                "run_count": j.run_count,
                "last_run_at": j.last_run_at.isoformat() if j.last_run_at else None,
                "created_at": j.created_at.isoformat(),
            }
            for j in jobs
        ],
        "total": len(jobs),
    }


@router.get("/bot/errors")
async def get_bot_errors(
    limit: int = 50,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.db.models import AgentError

    result = await db.execute(
        select(AgentError).order_by(AgentError.created_at.desc()).limit(limit)
    )
    errors = result.scalars().all()
    return {
        "errors": [
            {
                "id": str(e.id)[:8],
                "error_type": e.error_type,
                "error_message": e.error_message[:300] if e.error_message else None,
                "user_id": e.user_id[:8] + "..." if e.user_id else None,
                "session_id": e.session_id[:8] + "..." if e.session_id else None,
                "created_at": e.created_at.isoformat(),
            }
            for e in errors
        ],
        "total": len(errors),
    }


@router.post("/bot/broadcast")
async def broadcast_message(
    message: str = Body(..., embed=True),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not _telegram_bot_ref or not _telegram_bot_ref.app:
        raise HTTPException(status_code=503, detail="Bot not running")
    from app.db.models import TelegramUserMapping

    result = await db.execute(select(TelegramUserMapping))
    mappings = result.scalars().all()
    if not mappings:
        return {"sent": 0, "failed": 0, "message": "No users to broadcast to"}

    bot = _telegram_bot_ref.app.bot
    sent = 0
    failed = 0
    for m in mappings:
        try:
            await bot.send_message(chat_id=m.telegram_id, text=message, parse_mode="HTML")
            sent += 1
        except Exception:
            failed += 1
    return {"sent": sent, "failed": failed, "total": len(mappings)}


# ============ Gateway Control Endpoints ============

@router.get("/gateway/config")
async def get_config(current_user=Depends(get_current_user)):
    from app.config import settings

    return {
        "agent_model": settings.agent_model,
        "agent_fallback_model": settings.agent_fallback_model,
        "agent_max_tokens": settings.agent_max_tokens,
        "agent_temperature": settings.agent_temperature,
        "agent_max_tool_rounds": settings.agent_max_tool_rounds,
        "agent_workspace_dir": settings.agent_workspace_dir,
        "workspace_per_user": settings.workspace_per_user,
        "telegram_debounce_ms": settings.telegram_debounce_ms,
        "telegram_max_queue": settings.telegram_max_queue,
        "telegram_require_pairing": settings.telegram_require_pairing,
        "heartbeat_enabled": settings.heartbeat_enabled,
        "heartbeat_interval_hours": settings.heartbeat_interval_hours,
        "decay_enabled": getattr(settings, "decay_enabled", True),
        "consolidation_enabled": getattr(settings, "consolidation_enabled", True),
    }


@router.get("/gateway/processes")
async def list_processes(current_user=Depends(get_current_user)):
    if not _telegram_bot_ref:
        return {"processes": [], "total": 0}
    tool_executor = None
    if hasattr(_telegram_bot_ref, "agent_runner") and _telegram_bot_ref.agent_runner:
        tool_executor = getattr(_telegram_bot_ref.agent_runner, "tool_executor", None)
    if not tool_executor or not hasattr(tool_executor, "_processes"):
        return {"processes": [], "total": 0}
    procs = []
    for entry in tool_executor._processes.values():
        proc = entry["proc"]
        running = proc.returncode is None
        procs.append({
            "id": entry["id"],
            "label": entry["label"],
            "pid": entry["pid"],
            "command": entry["command"][:120],
            "status": "running" if running else f"exited ({proc.returncode})",
            "started_at": entry["started_at"],
            "output_lines": len(entry["output_buffer"]),
            "user_id": entry.get("user_id", "")[:8] + "...",
        })
    return {"processes": procs, "total": len(procs)}


@router.get("/gateway/subagents")
async def list_subagents(current_user=Depends(get_current_user)):
    if not _telegram_bot_ref:
        return {"tasks": [], "total": 0}
    subagent_mgr = None
    if hasattr(_telegram_bot_ref, "agent_runner") and _telegram_bot_ref.agent_runner:
        subagent_mgr = getattr(_telegram_bot_ref.agent_runner, "tool_executor", None)
        if subagent_mgr:
            subagent_mgr = getattr(subagent_mgr, "subagent_manager", None)
    if not subagent_mgr:
        return {"tasks": [], "total": 0}
    tasks = subagent_mgr.list_runs()
    return {"tasks": tasks, "total": len(tasks)}


@router.post("/gateway/reload")
async def reload_config(current_user=Depends(get_current_user)):
    from app.config import Settings

    try:
        new_settings = Settings()
        from app.config import settings as current_settings
        for field_name in new_settings.model_fields:
            setattr(current_settings, field_name, getattr(new_settings, field_name))
        return {
            "status": "reloaded",
            "message": "Configuration reloaded from environment. Some changes may require restart.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {e}")


# ============ F6 — Per-user memory health snapshot ============

@router.get("/memory-health/{user_id}", response_model=MemoryHealthForUser)
async def get_user_memory_health(
    user_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a memory-subsystem snapshot for one user.

    Mirrors the per-turn [memory_health] log line plus a 7-day
    summarizer rollup. Use this to answer:
      - is memory being created? (total_memories, by_category)
      - is decay working? (avg_strength, weak_count)
      - are open threads being tracked? (active_tasks_live)
      - is the summarizer healthy? (summarizer_recent_days,
        summarizer_failure_streak, summarizer_last_failure_reason)
      - would the next warm-start turn have continuity? (recent_days_*)

    Admin-only. Doesn't expose memory CONTENT — just shape and health.
    """
    from datetime import date, timedelta
    from app.db.models.user import User
    from app.db.models.day_chat import DayChat

    # Admin guard — non-admins get their own snapshot only
    if getattr(current_user, "role", None) != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Admin access required for other users")

    # Resolve user
    user_row = (await db.execute(
        select(User).where(User.id == user_id)
    )).scalar_one_or_none()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")

    # ── Memory bank aggregates ──
    bank = (await db.execute(
        select(
            func.count(Memory.id).label("total"),
            func.sum(case((Memory.is_active == True, 1), else_=0)).label("active"),
            func.avg(Memory.strength).label("avg_strength"),
            func.sum(case((Memory.strength < 0.3, 1), else_=0)).label("weak"),
            func.sum(case((Memory.strength >= 0.8, 1), else_=0)).label("strong"),
        ).where(and_(Memory.user_id == user_id, Memory.is_deleted == False))
    )).first()

    cat_rows = (await db.execute(
        select(Memory.category, func.count(Memory.id))
        .where(and_(
            Memory.user_id == user_id,
            Memory.is_deleted == False,
            Memory.is_active == True,
        ))
        .group_by(Memory.category)
    )).all()
    by_category = {(c or "uncategorized"): int(n) for (c, n) in cat_rows}

    # ── Active tasks ──
    at_live = (await db.execute(
        select(func.count(Memory.id)).where(and_(
            Memory.user_id == user_id,
            Memory.category == "active_task",
            Memory.is_active == True,
            Memory.is_deleted == False,
        ))
    )).scalar() or 0

    at_archived = (await db.execute(
        select(func.count(Memory.id)).where(and_(
            Memory.user_id == user_id,
            Memory.category == "active_task",
            Memory.is_active == False,
            Memory.is_deleted == False,
        ))
    )).scalar() or 0

    at_sample_rows = (await db.execute(
        select(Memory).where(and_(
            Memory.user_id == user_id,
            Memory.category == "active_task",
            Memory.is_active == True,
            Memory.is_deleted == False,
        )).order_by(Memory.updated_at.desc()).limit(5)
    )).scalars().all()
    at_samples = [
        ActiveTaskRow(
            id=t.id,
            content=t.content[:200],
            strength=float(t.strength or 0.0),
            created_at=t.created_at.isoformat() if t.created_at else None,
            last_reinforced_at=t.last_reinforced_at.isoformat() if t.last_reinforced_at else None,
        )
        for t in at_sample_rows
    ]

    # ── Summarizer state — last 7 calendar days ──
    today = date.today()
    cutoff = today - timedelta(days=7)
    day_rows = (await db.execute(
        select(DayChat).where(and_(
            DayChat.user_id == user_id,
            DayChat.local_date >= cutoff,
        )).order_by(DayChat.local_date.desc())
    )).scalars().all()

    summarizer_rows = [
        DaySummarizerStateRow(
            local_date=dc.local_date.isoformat(),
            summary_status=dc.summary_status or "unknown",
            summary_failure_count=int(dc.summary_failure_count or 0),
            summary_last_failure_at=(
                dc.summary_last_failure_at.isoformat()
                if dc.summary_last_failure_at else None
            ),
            summary_last_failure_reason=dc.summary_last_failure_reason,
            archival_summary_status=dc.archival_summary_status or "not_needed",
            has_rolling_summary=bool(dc.rolling_summary),
            has_archival_summary=bool(dc.archival_summary),
            message_count=int(dc.message_count or 0),
        )
        for dc in day_rows
    ]

    # Failure streak: count consecutive most-recent days where status='failed'.
    streak = 0
    for r in summarizer_rows:
        if r.summary_status == "failed":
            streak += 1
        else:
            break

    last_failure_reason = next(
        (r.summary_last_failure_reason for r in summarizer_rows if r.summary_last_failure_reason),
        None,
    )

    # ── F8 readiness — what would a fresh-day turn see? ──
    cutoff_2d = today - timedelta(days=2)
    recent_arch = sum(
        1 for dc in day_rows
        if dc.local_date >= cutoff_2d
        and dc.local_date < today
        and dc.archival_summary
        and dc.archival_summary_status == "up_to_date"
    )
    recent_rolling_only = sum(
        1 for dc in day_rows
        if dc.local_date >= cutoff_2d
        and dc.local_date < today
        and not (dc.archival_summary and dc.archival_summary_status == "up_to_date")
        and dc.rolling_summary
    )

    return MemoryHealthForUser(
        user_id=user_id,
        user_email=user_row.email,
        total_memories=int(bank.total or 0),
        active_memory_count=int(bank.active or 0),
        avg_strength=round(float(bank.avg_strength or 0.0), 3),
        weak_count=int(bank.weak or 0),
        strong_count=int(bank.strong or 0),
        by_category=by_category,
        active_tasks_live=int(at_live),
        active_tasks_archived=int(at_archived),
        active_task_samples=at_samples,
        summarizer_recent_days=summarizer_rows,
        summarizer_failure_streak=streak,
        summarizer_last_failure_reason=last_failure_reason,
        recent_days_with_archival=recent_arch,
        recent_days_with_rolling_only=recent_rolling_only,
    )


# ============ Data audit — sidebar gap diagnostic ============

class DataAuditDateBucket(BaseModel):
    date: str                  # YYYY-MM-DD
    conversation_count: int
    has_day_chat: bool
    day_chat_message_count: int


class DataAuditReport(BaseModel):
    user_id: str
    user_email: str
    agent_url: Optional[str]
    hosting_mode: Optional[str]
    deploy_status: Optional[str]
    total_conversations: int
    total_day_chats: int
    conversation_oldest: Optional[str]
    conversation_newest: Optional[str]
    day_chat_oldest: Optional[str]
    day_chat_newest: Optional[str]
    orphan_dates: List[str]    # dates with conversations but no day-chat
    buckets: List[DataAuditDateBucket]


@router.get("/data-audit/{user_id}", response_model=DataAuditReport)
async def data_audit_for_user(
    user_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Per-user audit: how many conversations vs day-chats, and which
    dates are missing day-chat rows.

    Built specifically to answer "why is the sidebar showing only
    N entries?" — proxies to the user's agent, paginates through
    sessions, fetches day-chats, and returns a date-bucketed gap
    report. Orphan dates are the conversations that exist but the
    backfill hasn't covered, so the sidebar's day-chat-mode
    rendering would silently drop them.

    Admin-only by default. A non-admin caller can audit only their
    own user_id (same scoping as memory-health).
    """
    import httpx
    from datetime import date as _date
    from app.db.models.user import User
    from app.db.models.agent import AgentConfig

    if getattr(current_user, "role", None) != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Admin access required for other users")

    user_row = (await db.execute(
        select(User).where(User.id == user_id)
    )).scalar_one_or_none()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")

    cfg = (await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )).scalar_one_or_none()

    if not cfg or not cfg.agent_url or not cfg.agent_api_key:
        return DataAuditReport(
            user_id=user_id,
            user_email=user_row.email,
            agent_url=cfg.agent_url if cfg else None,
            hosting_mode=cfg.hosting_mode if cfg else None,
            deploy_status=cfg.deploy_status if cfg else None,
            total_conversations=0,
            total_day_chats=0,
            conversation_oldest=None,
            conversation_newest=None,
            day_chat_oldest=None,
            day_chat_newest=None,
            orphan_dates=[],
            buckets=[],
        )

    headers = {"X-Agent-Key": cfg.agent_api_key}
    sessions: list = []
    day_chats: list = []
    async with httpx.AsyncClient(timeout=30) as client:
        # Walk sessions across pages — agent caps at limit=100, so loop
        # offsets until we get a short page or hit a safety cap of 1000.
        for offset in range(0, 1000, 100):
            try:
                resp = await client.get(
                    f"{cfg.agent_url}/api/sessions",
                    params={"limit": 100, "offset": offset},
                    headers=headers,
                )
                if resp.status_code != 200:
                    break
                payload = resp.json()
                page = payload.get("sessions", payload) if isinstance(payload, dict) else payload
                if not isinstance(page, list) or not page:
                    break
                sessions.extend(page)
                if len(page) < 100:
                    break
            except Exception:
                break

        # Day-chats — agent caps at limit=90, single page is enough.
        try:
            resp = await client.get(
                f"{cfg.agent_url}/api/day-chats",
                params={"limit": 90},
                headers=headers,
            )
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, list):
                    day_chats = payload
        except Exception:
            pass

    # Bucket conversations by their started_at date (UTC). The agent
    # stores started_at in UTC; comparing UTC-derived bucket vs
    # day_chat.local_date gives a reasonable signal even when the
    # user's tz differs by a few hours — for the gap-detection use
    # case here, off-by-one near midnight is acceptable.
    def _to_iso_date(s: Optional[str]) -> Optional[str]:
        if not s or not isinstance(s, str):
            return None
        # Accept "2026-05-09T..." → "2026-05-09".
        return s[:10] if len(s) >= 10 else None

    conv_dates: dict = {}
    for s in sessions:
        iso = _to_iso_date(s.get("started_at") or s.get("created_at"))
        if iso:
            conv_dates[iso] = conv_dates.get(iso, 0) + 1

    dc_by_date: dict = {}
    for dc in day_chats:
        iso = dc.get("local_date")
        if iso:
            dc_by_date[iso] = int(dc.get("message_count") or 0)

    all_dates = sorted(set(conv_dates.keys()) | set(dc_by_date.keys()), reverse=True)
    orphan_dates = [d for d in all_dates if d in conv_dates and d not in dc_by_date]

    buckets = [
        DataAuditDateBucket(
            date=d,
            conversation_count=conv_dates.get(d, 0),
            has_day_chat=d in dc_by_date,
            day_chat_message_count=dc_by_date.get(d, 0),
        )
        for d in all_dates
    ]

    return DataAuditReport(
        user_id=user_id,
        user_email=user_row.email,
        agent_url=cfg.agent_url,
        hosting_mode=cfg.hosting_mode,
        deploy_status=cfg.deploy_status,
        total_conversations=len(sessions),
        total_day_chats=len(day_chats),
        conversation_oldest=min(conv_dates.keys()) if conv_dates else None,
        conversation_newest=max(conv_dates.keys()) if conv_dates else None,
        day_chat_oldest=min(dc_by_date.keys()) if dc_by_date else None,
        day_chat_newest=max(dc_by_date.keys()) if dc_by_date else None,
        orphan_dates=orphan_dates,
        buckets=buckets,
    )
