"""Agent API endpoints - for Toup agent to store and retrieve memories"""

import json
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, select, and_

from app.db import get_db, Memory, Entity, EntityLink, AgentConfig
from app.db.models import AgentSecurityEvent, EVENT_AGENT_KEY_ROTATED
from app.schemas import (
    AgentStoreRequest, AgentRecallRequest, AgentRecallResponse,
    AgentGraphRequest, AgentGraphResponse,
    MemoryCreate, MemoryResponse, MemoryWithScore, MemoryWithRelations,
    EntityResponse, MemorySearchRequest
)
from app.api.auth import get_current_user
from app.api.memories import memory_to_response
from app.config import settings
from app.services.memory_service import MemoryService
from app.services.memory_gate import MemoryRejected

router = APIRouter(prefix="/agent", tags=["Agent API"])


@router.post("/store", response_model=List[MemoryResponse])
async def agent_store_memories(
    request: AgentStoreRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Store multiple memories from the agent.
    Used when the agent wants to explicitly save information.
    """
    service = MemoryService(db)
    created_memories = []
    
    rejected = []
    for memory_data in request.memories:
        try:
            memory = await service.create_memory(
                current_user.id,
                memory_data,
                source_message_id=None  # Agent-created memories
            )
        except MemoryRejected as exc:
            # One refused memory must not fail the whole batch.
            rejected.append(exc.reason)
            continue
        created_memories.append(memory)
    
    return [memory_to_response(m) for m in created_memories]


@router.post("/recall", response_model=AgentRecallResponse)
async def agent_recall_memories(
    request: AgentRecallRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Recall relevant memories for the agent during a conversation.
    Returns top-k memories ranked by semantic similarity.
    """
    service = MemoryService(db)
    
    # Build search query
    search_request = MemorySearchRequest(
        query=request.query + (" " + request.context if request.context else ""),
        categories=request.categories,
        limit=request.limit,
        include_explanation=True
    )
    
    results, _, _ = await service.search_memories(current_user.id, search_request)
    
    # Filter by minimum similarity
    filtered_results = [
        r for r in results
        if r.similarity_score >= request.min_similarity
    ]
    
    # Generate context summary
    context_summary = None
    if filtered_results:
        # Create a brief summary of retrieved memories
        summaries = [r.summary or r.content[:100] for r in filtered_results[:3]]
        context_summary = " | ".join(summaries)
    
    return AgentRecallResponse(
        memories=filtered_results,
        context_summary=context_summary
    )


@router.post("/graph", response_model=AgentGraphResponse)
async def agent_graph_traversal(
    request: AgentGraphRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Traverse the memory graph starting from a specific memory.
    Returns connected memories and entities up to specified depth.
    """
    service = MemoryService(db)
    
    # Get root memory
    root_memory = await service.get_memory(request.memory_id, current_user.id)
    if not root_memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    # Get related memories
    related_memories = await service.get_related_memories(
        request.memory_id,
        current_user.id,
        depth=request.depth
    )
    
    # Get entities from root memory
    all_entities = []
    if request.include_entities:
        result = await db.execute(
            select(EntityLink).where(EntityLink.memory_id == request.memory_id)
        )
        links = result.scalars().all()
        
        for link in links:
            result = await db.execute(
                select(Entity).where(Entity.id == link.entity_id)
            )
            entity = result.scalar_one_or_none()
            if entity:
                all_entities.append(EntityResponse(
                    id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    mention_count=entity.mention_count,
                    first_seen_at=entity.first_seen_at,
                    last_seen_at=entity.last_seen_at,
                    attributes=json.loads(entity.attributes_json) if entity.attributes_json else None
                ))
    
    # Build response with relations
    related_with_relations = []
    for memory in related_memories:
        # Get entities for each related memory
        entities = []
        if request.include_entities:
            result = await db.execute(
                select(EntityLink).where(EntityLink.memory_id == memory.id)
            )
            links = result.scalars().all()
            for link in links:
                result = await db.execute(
                    select(Entity).where(Entity.id == link.entity_id)
                )
                entity = result.scalar_one_or_none()
                if entity:
                    entities.append(EntityResponse(
                        id=entity.id,
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        mention_count=entity.mention_count,
                        first_seen_at=entity.first_seen_at,
                        last_seen_at=entity.last_seen_at,
                        attributes=json.loads(entity.attributes_json) if entity.attributes_json else None
                    ))
        
        related_with_relations.append(MemoryWithRelations(
            **memory_to_response(memory).model_dump(),
            related_memories=[],  # Don't recurse further
            entities=entities
        ))
    
    return AgentGraphResponse(
        root=memory_to_response(root_memory),
        related=related_with_relations,
        entities=all_entities
    )


@router.get("/context", response_model=AgentRecallResponse)
async def agent_get_context(
    query: str,
    limit: int = 5,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Quick context retrieval for the agent.
    Simpler endpoint for getting relevant memories during conversation.
    """
    service = MemoryService(db)
    
    search_request = MemorySearchRequest(
        query=query,
        limit=limit,
        include_explanation=True
    )
    
    results, _, _ = await service.search_memories(current_user.id, search_request)
    
    context_summary = None
    if results:
        summaries = [r.summary or r.content[:100] for r in results[:3]]
        context_summary = " | ".join(summaries)
    
    return AgentRecallResponse(
        memories=results,
        context_summary=context_summary
    )


@router.get("/entity/{entity_name}/memories", response_model=List[MemoryResponse])
async def get_memories_by_entity(
    entity_name: str,
    limit: int = 20,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all memories related to a specific entity.
    Useful for retrieving everything known about a person, project, etc.
    """
    # Find entity
    result = await db.execute(
        select(Entity).where(
            and_(
                Entity.user_id == current_user.id,
                Entity.name.ilike(f"%{entity_name}%")
            )
        )
    )
    entities = result.scalars().all()
    
    if not entities:
        return []
    
    # Get all memory IDs linked to these entities
    entity_ids = [e.id for e in entities]
    result = await db.execute(
        select(EntityLink.memory_id).where(
            EntityLink.entity_id.in_(entity_ids)
        )
    )
    memory_ids = [row[0] for row in result.fetchall()]
    
    if not memory_ids:
        return []
    
    # Fetch memories
    result = await db.execute(
        select(Memory)
        .where(
            and_(
                Memory.id.in_(memory_ids),
                Memory.user_id == current_user.id,
                Memory.is_deleted == False
            )
        )
        .order_by(Memory.created_at.desc())
        .limit(limit)
    )
    memories = result.scalars().all()

    return [memory_to_response(m) for m in memories]


# ─── T0b: agent_api_key rotation ──────────────────────────────────────


class RotateAgentKeyResponse(BaseModel):
    new_agent_api_key: str
    rotated_at: str  # ISO 8601 UTC
    key_prefix: str  # First 8 chars for display


class LastRotationResponse(BaseModel):
    last_rotated_at: Optional[str] = None  # ISO 8601 UTC, or None if never


@router.post("/rotate-key", response_model=RotateAgentKeyResponse)
async def rotate_agent_key(
    x_sensitive_action_token: Optional[str] = Header(default=None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rotate this user's `agent_api_key`.

    Authentication chain:
      1. The standard JWT/session auth identifies the user (`current_user`).
      2. `X-Sensitive-Action-Token` carries a fresh `ROTATE_AGENT_KEY`-purpose
         token issued by `POST /auth/reauth` with body
         `{"purpose": "rotate_agent_key"}`. Single-use; replay-protected.

    Effect:
      * New `agent_api_key` is generated, persisted to `agent_configs`,
        and pushed into the tenant container's env (via bridge recreate).
      * The new container is verified by polling `/agent/health` with
        the new key before this endpoint returns success.
      * Active WS chat sessions disconnect when the old container is
        killed by recreate. This is the desired behavior for a security
        action — the user is told this on the confirm modal.
      * On any failure (bridge recreate fails, verify times out), the
        DB is rolled back to the old key and the bridge is restored to
        the old env. The old key remains valid.

    Response:
      * The new key is returned EXACTLY ONCE in this response. The
        client must display it for the user to copy and never request
        it again. The DB does not persist any plaintext copy beyond
        this row.
    """
    # Routing guard: agent_router is mounted on both platform and agent
    # containers. Rotation only makes sense from the platform side
    # (it pushes new env to the bridge, which the platform owns the
    # mTLS cert for). On the agent process, surface a 404 so this
    # endpoint is invisible to anyone probing the tenant URL.
    if settings.run_mode == "agent":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )
    if not settings.enable_agent_key_rotation:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Agent key rotation is not enabled in this environment. "
                "Contact the operator to flip ENABLE_AGENT_KEY_ROTATION."
            ),
        )

    # Verify the sensitive-action token (replay-protected, purpose-bound).
    from app.services.sensitive_action_token import (
        SensitiveActionPurpose,
        verify_sensitive_action_token,
    )
    await verify_sensitive_action_token(
        db,
        x_sensitive_action_token or "",
        expected_user_id=str(current_user.id),
        expected_purpose=SensitiveActionPurpose.ROTATE_AGENT_KEY,
    )

    # Atomic rotation. Raises AgentKeyRotationError on any failure;
    # rollback (DB + bridge restore) is handled inside.
    from app.services.agent_key_rotation import (
        AgentKeyRotationError,
        rotate_agent_api_key,
    )
    try:
        result = await rotate_agent_api_key(db, str(current_user.id))
    except AgentKeyRotationError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        )

    return RotateAgentKeyResponse(
        new_agent_api_key=result.new_agent_api_key,
        rotated_at=result.rotated_at.isoformat() + "Z",
        key_prefix=result.key_prefix,
    )


@router.get("/key-last-rotated", response_model=LastRotationResponse)
async def get_last_key_rotation(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return when this user last rotated their agent_api_key, or None
    if never. Drives the "Last rotated: X ago" line on /agent/settings."""
    if settings.run_mode == "agent":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )
    row = (
        await db.execute(
            select(AgentSecurityEvent.occurred_at)
            .where(AgentSecurityEvent.user_id == str(current_user.id))
            .where(AgentSecurityEvent.event_type == EVENT_AGENT_KEY_ROTATED)
            .order_by(desc(AgentSecurityEvent.occurred_at))
            .limit(1)
        )
    ).scalar_one_or_none()
    return LastRotationResponse(
        last_rotated_at=(row.isoformat() + "Z") if row else None,
    )


# ─── T1h: connector tools cache invalidation ───────────────────────────


class RefreshToolsResponse(BaseModel):
    status: str
    tool_count: int


@router.put("/refresh-tools", response_model=RefreshToolsResponse)
async def refresh_connector_tools(request: Request):
    """T1h — Drop the agent's MCP tools cache and refetch immediately.

    Called by the platform after an OAuth connect / disconnect so the
    agent's tool list reflects the change without waiting for the 60s
    TTL. Best-effort from the platform's side: if this 500s or times
    out, the TTL is the safety net.

    Auth model:
      * Tenant agents have a unique `agent_api_key`. The platform
        knows it (it minted it at provision time and pushes it into
        the tenant container's env). The endpoint accepts the same
        `X-Agent-Key` header the soul-sync path uses
        (`app/api/soul.py:353`). Any other caller — JWT user, CSRF —
        is rejected.
      * Run-mode guard: only meaningful on agent containers
        (`run_mode == "agent"`). On the platform process itself,
        return 404 so the endpoint is invisible to anyone probing
        `<platform>/api/agent/refresh-tools`.

    Idempotent: multiple PUTs in quick succession are coalesced by
    the cache's per-instance asyncio.Lock — only one MCP round-trip
    fires regardless of how many platform callers race.
    """
    if settings.run_mode != "agent":
        # On the platform, this endpoint is meaningless — there's no
        # MCP tools cache here. Surface 404 so probers don't learn
        # the endpoint exists.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )

    # X-Agent-Key auth — same primitive as soul.py:353. Reject if
    # missing, malformed, or doesn't match the tenant's key.
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent key",
        )

    # Reach the cache via app.state. The lifespan wires it at boot for
    # dedicated containers; /admin/bind wires it for pool containers.
    # A missing cache here means the container was bound before the
    # bind-time bootstrap existed (or that bootstrap errored) — and
    # since this endpoint already authenticated against
    # settings.agent_api_key, the key the lifespan gate was missing is
    # NOW present. Self-heal on the spot instead of masking the gap
    # with a no_cache 200: this turns every post-OAuth notify into an
    # automatic repair path. Only when construction is genuinely
    # impossible do we still answer no_cache (200, so the platform's
    # call site doesn't treat a degraded agent as a transport failure).
    cache = getattr(request.app.state, "mcp_tools_cache", None)
    if cache is None:
        from app.agent.mcp_bootstrap import ensure_mcp_initialized

        result = await ensure_mcp_initialized(request.app)
        cache = getattr(request.app.state, "mcp_tools_cache", None)
        if cache is None:
            return RefreshToolsResponse(
                status=f"no_cache_{result}", tool_count=0
            )
        # ensure_mcp_initialized already awaited the first refresh;
        # don't refetch again below (the platform's 3s first attempt
        # is tight enough as it is). cached_at == 0 means that fetch
        # failed — say so in the status instead of reporting a clean
        # "initialized" with zero tools (the platform logs the body,
        # and a silent-success here is the exact masking pattern that
        # hid the original no-tools incident).
        if cache.cached_at == 0:
            return RefreshToolsResponse(
                status="initialized_refresh_failed", tool_count=0
            )
        return RefreshToolsResponse(
            status="initialized", tool_count=len(cache.tools)
        )

    # Force an immediate refetch (bypasses TTL). Lock inside the cache
    # coalesces concurrent platform notifications.
    try:
        await cache.refresh()
    except Exception as e:
        # Refetch failed — DB / network blip on the platform side.
        # Cache.invalidate() ensures the next list_tools() retries.
        cache.invalidate()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Refresh failed: {type(e).__name__}: {str(e)[:200]}",
        )
    return RefreshToolsResponse(
        status="invalidated",
        tool_count=len(cache.tools),
    )


class MCPCacheStatusResponse(BaseModel):
    """Snapshot of the agent's MCP tools cache for ops diagnostics.

    Returned by /api/agent/mcp-cache-status. The whole point is that
    when a user reports "the agent says no Gmail tool but the platform
    shows it Connected", an operator can hit this endpoint and see at
    a glance whether the cache is healthy, never-fetched, or
    persistently failing — without having to docker-exec into the
    tenant container to grep stdout.
    """

    has_cache: bool
    tool_count: int
    tools: list[str]
    cached_at_monotonic: float
    last_attempt_at_monotonic: float
    consecutive_failures: int
    last_error: Optional[str]
    is_fresh: bool


@router.get("/mcp-cache-status", response_model=MCPCacheStatusResponse)
async def mcp_cache_status(request: Request):
    """Read-only view of the MCP tools cache state.

    Same X-Agent-Key gate as /refresh-tools. The platform's admin
    dashboard polls this when an operator clicks "Diagnose connectors"
    on a tenant; if `consecutive_failures > 0` we surface the
    `last_error` next to the tile so the failure mode is visible
    instead of hidden behind a generic "no tools available".
    """
    if settings.run_mode != "agent":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )

    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent key",
        )

    cache = getattr(request.app.state, "mcp_tools_cache", None)
    if cache is None:
        return MCPCacheStatusResponse(
            has_cache=False,
            tool_count=0,
            tools=[],
            cached_at_monotonic=0.0,
            last_attempt_at_monotonic=0.0,
            consecutive_failures=0,
            last_error=None,
            is_fresh=False,
        )

    return MCPCacheStatusResponse(
        has_cache=True,
        tool_count=len(cache.tools),
        tools=list(cache.tools),
        cached_at_monotonic=cache.cached_at,
        last_attempt_at_monotonic=getattr(cache, "last_attempt_at", 0.0),
        consecutive_failures=getattr(cache, "consecutive_failures", 0),
        last_error=getattr(cache, "last_error", None),
        is_fresh=cache.is_fresh(),
    )


# ──────────────────────────────────────────────────────────────────────
# Runtime feature flags — read by the tenant agent process.
# ──────────────────────────────────────────────────────────────────────
#
# The tenant agent's DB is partitioned (PLATFORM_ONLY_TABLES excludes
# agent_configs), so the agent process CAN'T `SELECT
# subagent_spawning_enabled FROM agent_configs` against its own DB —
# the table doesn't exist there. The flag lives on the platform DB,
# which only the platform-api can reach.
#
# This endpoint gives the agent process a way to read per-tenant
# runtime flags without growing a second DB connection pool. Auth is
# the standard multi-tenant X-Agent-Key contract: the agent sends its
# own ``agent_api_key`` + ``user_id``, the platform validates them
# against the same row on agent_configs and returns the row's flag
# values. A wrong key for a known user returns 403, not 404, so a
# leaked-key probe can't enumerate user_ids.


class AgentRuntimeFlagsResponse(BaseModel):
    """Per-tenant runtime flag payload. Add new flags as fields here
    — keep the keys snake_case so they pair 1:1 with the
    ``agent_configs`` column names."""
    subagent_spawning_enabled: bool


@router.get("/runtime-flags", response_model=AgentRuntimeFlagsResponse)
async def get_runtime_flags(
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
    x_agent_user_id: Optional[str] = Header(default=None, alias="X-Agent-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """Return the tenant's runtime flag values from the platform DB.

    Called from the agent process (``_read_subagent_flag_for_user`` in
    ``tool_executor.py``) at spawn-time. Single indexed lookup —
    ``agent_configs(user_id)`` is UNIQUE.

    Auth: same X-Agent-Key + X-Agent-User-Id contract used by
    ``streaming.py``, ``credits.py``, etc. Validates the (user, key)
    pair against the agent_configs row so a tenant can only read its
    own flags. 401 on missing headers, 403 on key mismatch."""
    if not x_agent_key or not x_agent_user_id:
        raise HTTPException(
            status_code=401,
            detail="X-Agent-Key + X-Agent-User-Id required",
        )

    cfg = (await db.execute(
        select(AgentConfig).where(
            and_(
                AgentConfig.user_id == x_agent_user_id,
                AgentConfig.agent_api_key == x_agent_key,
            )
        )
    )).scalar_one_or_none()
    if cfg is None:
        raise HTTPException(status_code=403, detail="agent key mismatch")

    return AgentRuntimeFlagsResponse(
        subagent_spawning_enabled=bool(
            getattr(cfg, "subagent_spawning_enabled", False)
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Resume a job parked on a confirmation card
# ─────────────────────────────────────────────────────────────────────


class ResolvePendingActionRequest(BaseModel):
    #: The `connector_pending_actions.id` the user just decided on.
    action_id: str
    #: Terminal state of that card: executed | failed | rejected | expired.
    outcome: str
    #: Optional one-line reason, shown to the user on the non-happy paths.
    detail: Optional[str] = None


class ResolvePendingActionResponse(BaseModel):
    resolved: int
    status: Optional[str] = None


#: Card outcome → the job's terminal status. `executed` is the one that
#: turns the job green, which is the whole point: the work the job
#: narrates is not finished until the staged call actually runs.
_ACTION_OUTCOME_TO_JOB_STATUS = {
    "executed": "completed",
    "failed": "failed",
    "rejected": "cancelled",
    "expired": "cancelled",
}


@router.post("/jobs/resolve-pending-action", response_model=ResolvePendingActionResponse)
async def resolve_job_for_pending_action(
    body: ResolvePendingActionRequest,
    request: Request,
) -> ResolvePendingActionResponse:
    """Close the job that was parked waiting on this confirmation card.

    Why this exists at all: `build_jobs` is an AGENT_ONLY table, but the
    card is approved against the PLATFORM (`connector_pending_actions`
    is PLATFORM_ONLY, and the platform is what actually executes the
    staged call). So the one process that learns the user approved
    cannot reach the one row that has to change. This endpoint is that
    hop, and it is the reason parking a job is safe: without a resume
    path, `waiting_on_user` is an immortal state — the reaper only
    sweeps queued/running, and the turn-end finalizer only touches
    running — and a job stuck at "Waiting on you" forever is just a
    quieter version of the lie this whole change removes.

    Matched on the agent side rather than by a `job_id` carried on the
    platform row, because the agent already knows both ids at park time
    and the platform does not. That keeps the linkage to one nullable
    key inside an existing JSON column instead of a migration on a
    shared table plus a new MCP header.

    Idempotent: only rows still in `waiting_on_user` are touched, so a
    retried delivery is a no-op that reports `resolved=0`.
    """
    if settings.run_mode != "agent":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key",
        )

    new_status = _ACTION_OUTCOME_TO_JOB_STATUS.get(body.outcome)
    if new_status is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown outcome {body.outcome!r}",
        )

    from datetime import datetime as _dt

    from app.agent.job_status import ERR_TOOL_FAILURE, STATUS_WAITING_ON_USER
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    resolved: list[tuple[str, str, str]] = []
    automation_run_ids: set[str] = set()
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(BuildJob).where(BuildJob.status == STATUS_WAITING_ON_USER)
        )).scalars().all()
        for job in rows:
            cfg = job.config_json or {}
            if not isinstance(cfg, dict):
                continue
            if cfg.get("pending_action_id") != body.action_id:
                continue
            if job.job_type == "automation_run":
                # R29: an automation run's terminal rides the engine's
                # exactly-once finalize gate (outcome vocabulary,
                # last-outcome stamp, noteworthy push, session card
                # flip, health) — the generic flip below would bypass
                # all of it. Rejected/expired close as outcome
                # "skipped": the user's decision, not a failure.
                from app.agent.automations.confirm import resolve_parked_run

                if await resolve_parked_run(
                    db, job, outcome=body.outcome, detail=body.detail,
                ):
                    resolved.append((job.id, job.title or "", job.user_id))
                    automation_run_ids.add(job.id)
                continue
            job.status = new_status
            job.completed_at = _dt.utcnow()
            # The park stamped `awaiting_confirmation` copy on these two.
            # Leaving it behind would render "Waiting for you to approve
            # this" underneath a job the client now draws as Done.
            #
            # `tool_failure` ONLY for the outcome that actually failed. A
            # rejected or expired card cancelled the job; nothing broke, and
            # stamping a failure class on a decision the user made is the
            # same category error this whole change is about.
            job.error_class = (
                ERR_TOOL_FAILURE if new_status == "failed" else None
            )
            job.user_message = (
                None if new_status == "completed"
                else (body.detail or "This wasn't approved, so nothing was sent.")
            )
            resolved.append((job.id, job.title or "", job.user_id))
        if resolved:
            await db.commit()

    for job_id, title, user_id in resolved:
        try:
            from app.api.ws_chat import broadcast_to_user

            await broadcast_to_user(user_id, {
                "type": "job_update", "job_id": job_id,
                "name": title, "status": new_status,
            })
        except Exception:  # noqa: BLE001
            pass
        if job_id in automation_run_ids:
            # The engine's own composer already pushed what was
            # noteworthy (notify_run_outcome via the finalize gate);
            # the generic mission events below would double-speak.
            continue
        # The Live Activity is still ALIVE — `notify_job_needs_user`
        # deliberately kept it that way — so a terminal notification is the
        # only thing that ends it. A DB write never touches the card.
        try:
            from app.agent.subagent_orchestrator import _notify_job_event

            if new_status == "completed":
                await _notify_job_event(
                    job_id=job_id, label=title, kind="mission_completed",
                    title=f"✅ Done: {(title or 'background task')[:150]}",
                    body="Approved and sent.", progress=100,
                    dismiss_after_s=900, dedup_suffix="completed",
                )
            else:
                await _notify_job_event(
                    job_id=job_id, label=title, kind="mission_failed",
                    title=f"Stopped: {(title or 'background task')[:150]}",
                    body=(body.detail or "This wasn't approved, so nothing was sent.")[:300],
                    dismiss_after_s=600, dedup_suffix="resolved", urgent=False,
                )
        except Exception:  # noqa: BLE001
            pass

    return ResolvePendingActionResponse(
        resolved=len(resolved),
        status=new_status if resolved else None,
    )
