"""Memory files — the whole /api/memories surface (v3 §4).

Eight product routes and three admin ones, one unit: the FILE. Round 8's 23 routes served `memories`
ROWS — 29 engine fields per row, strength meters, leases, recall counts,
events — and both Memory pages were database-row inspectors built on them.
v3 retires the rows from the product (§1.1): the file's `body_md` is the
memory, and NO payload here carries an engine field. `tests/
test_memories_v3_api.py::test_no_engine_metadata_reaches_a_client` walks
every response and fails on one.

Three invariants survive from round 8, each of which cost a cycle:

1. **Every handler consults `_get_agent_proxy_info`.** `memory_files` is
   AGENT_ONLY: the rows live in the tenant container. A route that forgets
   this serves the platform's empty copy. The structural test greps this
   module's source for the literal.
2. **Writes go through `_proxy_memories_write`, which 502s and NEVER falls
   back.** A write that silently "succeeds" against the platform DB is how
   a user comes to believe they deleted something that is still there.
3. **Literal routes are declared before path captures.** Starlette matches
   in declaration order and `{slug:path}` is greedy.

What CHANGED from round 8: reads no longer fall back to a platform-DB
view. With rows retired there is nothing to synthesize a file from, so an
unreachable agent is `503 {"detail": "Your agent isn't reachable right
now."}` — the clients already own an honest error state, and an empty
"No memories yet" for a full library reads as data loss.
"""

import json
import logging
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_db, Memory, AgentConfig
# Re-exported: `app.mcp_server` imports these three from this module (they
# are the memory write schemas the MCP tools proxy with), and ingest.py /
# agent.py / stats.py import `memory_to_response` below. Both are the
# document/media leg (§3.4), not the memory product.
from app.schemas import (  # noqa: F401
    MemoryCreate, MemoryUpdate, MemoryResponse, MemorySearchRequest,
    MemoryFileInstructRequest,
)
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)


async def _get_user_api_key(db: AsyncSession, user_id: str) -> Optional[str]:
    """Fetch the user's OpenAI API key from agent_configs."""
    try:
        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
            )
            return result.scalar_one_or_none()
    except Exception:
        return None


# ── Agent proxy helpers ────────────────────────────────────────────

async def _get_agent_proxy_info(
    user_id: str, db: AsyncSession
) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) if the user has a remote agent."""
    try:
        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.deploy_status == "active",
                )
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass  # agent_configs table may not exist on agent DBs
    return None


async def _proxy_memories(
    agent_url: str, agent_api_key: str, path: str,
    params: Optional[dict] = None, method: str = "GET", body: Optional[dict] = None,
):
    """Proxy a memories READ to the tenant agent.

    Returns None on any failure. In round 8 that meant "fall back to the
    platform DB"; in v3 it means 503 — see `_agent_unreachable`. The
    fallback existed because rows were mirrored platform-side; files are
    not, and a synthesized-from-nothing file view is a lie, not a
    degradation.

    TKT-LAT-007 (wave 3): shared agent_http client.
    """
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/memories/{path}" if path else f"{agent_url}/api/memories"
    try:
        client = get_agent_http_client()
        if method == "GET":
            resp = await client.get(
                url, headers={"X-Agent-Key": agent_api_key},
                params=params or {}, timeout=10.0,
            )
        else:
            resp = await client.post(
                url, headers={"X-Agent-Key": agent_api_key},
                params=params or {}, json=body or {}, timeout=10.0,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return {"__status__": 404}
        logger.warning("Agent memories proxy %s returned %s", url, resp.status_code)
    except Exception as e:
        logger.warning("Agent memories proxy %s failed: %s", url, e)
    return None


async def _proxy_memories_write(
    agent_url: str, agent_api_key: str, path: str,
    method: str, body: Optional[dict] = None, params: Optional[dict] = None,
):
    """Proxy a memories WRITE (POST/DELETE) to the tenant agent.

    Backlog BE-1. `memory_files` is an AGENT_ONLY table: it exists in the
    tenant container's DB, not in the platform's. Unlike the read helper
    this NEVER falls back to the platform DB. A write that cannot reach the
    tenant must surface as an error: silently "succeeding" against the
    wrong database is how a user comes to believe they deleted something
    that is still there. Raises HTTPException on failure; propagates the
    agent's own status code when it returns one.
    """
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/memories/{path}" if path else f"{agent_url}/api/memories"
    try:
        client = get_agent_http_client()
        resp = await client.request(
            method,
            url,
            headers={"X-Agent-Key": agent_api_key},
            params=params or {},
            json=body,
            # Curation is an LLM round trip on the tenant. 15s was tuned for
            # a row UPDATE; an instruct that has to think needs longer than
            # the client's own patience, not less.
            timeout=45.0 if path.endswith("instruct") or path == "instruct" else 15.0,
        )
    except Exception as e:
        logger.warning("Agent memories write proxy %s %s failed: %s", method, url, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach your agent to save this change. Please try again.",
        )

    if resp.status_code in (200, 201, 204):
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    if resp.status_code == 404:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory file not found"
        )

    logger.warning(
        "Agent memories write proxy %s %s returned %s", method, url, resp.status_code
    )
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Your agent rejected this change. Please try again.",
    )


router = APIRouter(prefix="/memories", tags=["Memories"])


def _agent_unreachable() -> HTTPException:
    """The one degraded answer a file read may give.

    Round 8 synthesized a virtual file view from platform rows when the
    tenant did not answer. v3 has no rows to synthesize from, and "No
    memories yet" for a full library reads as data loss — so the honest
    answer is that the agent is not reachable, which both clients render.
    """
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Your agent isn't reachable right now.",
    )


def _platform_mode() -> bool:
    """True on platform-api, where `memory_files` does not exist at all."""
    from app.config import settings
    return settings.run_mode == "platform"


def _slug_path(slug: str) -> str:
    """VALIDATE the `{slug:path}` capture, or 404. Never sanitize-and-continue.

    This is the only check standing between a user-controlled path segment
    and an X-Agent-Key-AUTHENTICATED proxy URL. It used to be
    `slug.lstrip("/")`, which strips a leading slash and nothing else — so
    `/api/memories/files/..%2f..%2fadmin/whatever` arrived at the tenant
    agent as an authenticated request to a path the caller chose. Starlette
    decodes `%2f` before the `:path` converter sees it, so the traversal is
    already a real `../` by the time this function runs.

    `is_valid_slug` is the right test and it is the one the WRITER uses: the
    same slug regex with a 120-char cap, admitting exactly one namespace
    segment and excluding dots, query strings, fragments, percent-escapes,
    backslashes and every traversal form by construction.

    Rejecting rather than stripping is the point — a sanitizer that "fixes"
    a hostile path still forwards a request the caller shaped, and there is
    no legitimate slug it can repair.

    404 rather than 400: a malformed slug and an unknown slug are the same
    answer to the client ("that file does not exist"), and a distinguishable
    400 tells a prober which shapes reach the parser.
    """
    from app.memory_files import is_valid_slug, section_of_slug

    cleaned = (slug or "").strip().lstrip("/")
    # Shape AND namespace. `is_valid_slug` alone would pass
    # "absolute/path" — well-formed, in no v3 section, and still forwarded.
    # `section_of_slug` is the same predicate `_all_files` uses to decide
    # what is a v3 file at all, so the route and the store agree on what can
    # exist rather than the route being one shape looser.
    if not is_valid_slug(cleaned) or section_of_slug(cleaned) is None:
        logger.warning(
            "[memories] rejected a malformed file slug (%d chars)", len(slug or "")
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory file not found"
        )
    return cleaned


def memory_to_response(memory: Memory) -> MemoryResponse:
    """Row serializer for the DOCUMENT/MEDIA leg only (v3 §3.4).

    No route in this module returns it. It stays here because
    `app/api/ingest.py`, `app/api/agent.py` and `app/api/stats.py` import
    it by this name for their own ingestion payloads, which describe
    documents and media — not the user's memory files.
    """
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        canonical_content=getattr(memory, 'canonical_content', None),
        summary=memory.summary,
        brain_type=getattr(memory, 'brain_type', 'user'),
        category=memory.category,
        memory_type=memory.memory_type,
        importance=memory.importance,
        confidence=memory.confidence,
        strength=memory.strength,
        memory_level=memory.memory_level,
        emotional_salience=memory.emotional_salience,
        last_reinforced_at=memory.last_reinforced_at,
        consolidation_count=memory.consolidation_count,
        decay_rate=memory.decay_rate,
        history=json.loads(memory.history_json) if getattr(memory, 'history_json', None) else None,
        merged_from=json.loads(memory.merged_from_json) if getattr(memory, 'merged_from_json', None) else None,
        superseded_by=getattr(memory, 'superseded_by', None),
        is_active=getattr(memory, 'is_active', True),
        expires_at=getattr(memory, 'expires_at', None),
        file_slug=getattr(memory, 'file_slug', None),
        file_position=getattr(memory, 'file_position', None),
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        last_accessed_at=memory.last_accessed_at,
        access_count=memory.access_count,
        source_type=memory.source_type,
        tags=json.loads(memory.tags_json) if memory.tags_json else None,
        metadata=json.loads(memory.metadata_json) if memory.metadata_json else None,
    )


# ── Files ───────────────────────────────────────────────────────────
# NOTE: literal paths first. `{slug:path}` is greedy and every route below
# it that is not prefixed by a literal would be swallowed.


@router.get("/files", response_model=dict)
async def list_memory_files(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The sectioned file listing: You · People · Topics · Areas · Learned."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories(proxy[0], proxy[1], "files")
        if data is not None and "__status__" not in data:
            return JSONResponse(content=data)
        raise _agent_unreachable()

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_file_ops
    return await memory_file_ops.list_files(db, current_user.id)


@router.get("/files/{slug:path}", response_model=dict)
async def get_memory_file(
    slug: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """One file: title, description, `body_md`, and its links with titles."""
    slug = _slug_path(slug)
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories(proxy[0], proxy[1], f"files/{slug}")
        if data is not None and "__status__" not in data:
            return JSONResponse(content=data)
        if data is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Memory file not found"
            )
        raise _agent_unreachable()

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_file_ops
    file = await memory_file_ops.get_file(db, current_user.id, slug)
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory file not found"
        )
    return file


@router.post("/files/{slug:path}/instruct", response_model=dict)
async def instruct_memory_file(
    slug: str,
    payload: MemoryFileInstructRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """"Tell your agent what to change or remove", scoped to ONE file.

    The curator proposes ops, the deterministic validator judges them, and
    what survives is applied atomically with a change line per mutation.
    Ops naming another file are dropped: the user is looking at one file.
    """
    slug = _slug_path(slug)
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories_write(
            proxy[0], proxy[1], f"files/{slug}/instruct", "POST",
            body=payload.model_dump(),
        )
        return JSONResponse(content=data if data is not None else {
            "applied": 0, "rejected": [], "note": "Nothing to change.",
        })

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_curator
    key = await _get_user_api_key(db, current_user.id)
    try:
        result = await memory_curator.instruct_file(
            db, current_user.id, slug, payload.instruction, api_key=key,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory file not found"
        )
    except Exception as e:
        logger.warning("File instruction failed for %s: %s", slug, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Your agent couldn't apply that change. Please try again.",
        )
    return result


@router.delete("/files/{slug:path}", response_model=dict)
async def delete_memory_file(
    slug: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a file. A SYSTEM file (Profile, Current context, Learned) is
    emptied instead of dropped — the injection depends on it existing."""
    slug = _slug_path(slug)
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        await _proxy_memories_write(proxy[0], proxy[1], f"files/{slug}", "DELETE")
        return JSONResponse(content={"deleted": True})

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_file_ops
    if not await memory_file_ops.delete_file(db, current_user.id, slug):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory file not found"
        )
    return {"deleted": True}


# ── The global instruct box ─────────────────────────────────────────


@router.post("/instruct", response_model=dict)
async def instruct_memory(
    payload: MemoryFileInstructRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """"Tell your agent what to remember" — no file named.

    Same ops engine; the curator routes the fact to the right file and
    creates one when nothing fits.
    """
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories_write(
            proxy[0], proxy[1], "instruct", "POST", body=payload.model_dump(),
        )
        return JSONResponse(content=data if data is not None else {
            "applied": 0, "rejected": [], "note": "Nothing to change.",
        })

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_curator
    key = await _get_user_api_key(db, current_user.id)
    try:
        return await memory_curator.instruct_global(
            db, current_user.id, payload.instruction, api_key=key,
        )
    except Exception as e:
        logger.warning("Global instruction failed for %s: %s", current_user.id[:8], e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Your agent couldn't apply that change. Please try again.",
        )


# ── Search and log ──────────────────────────────────────────────────


@router.get("/search", response_model=dict)
async def search_memory_files(
    q: str = Query(..., min_length=1, description="What to look for"),
    limit: int = Query(10, ge=1, le=50),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search file bodies and titles. Results are FILES, with a snippet."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories(
            proxy[0], proxy[1], "search", {"q": q, "limit": limit}
        )
        if data is not None and "__status__" not in data:
            return JSONResponse(content=data)
        raise _agent_unreachable()

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_file_ops
    return {"results": await memory_file_ops.search_files(db, current_user.id, q, limit)}


@router.get("/log", response_model=dict)
async def memory_log(
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$", description="YYYY-MM"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The Memory log for one month: what the writer changed, by user-local
    day. `day_key` is stored at write time — bucketing UTC timestamps
    client-side files a 23:40-local change under tomorrow."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories(proxy[0], proxy[1], "log", {"month": month})
        if data is not None and "__status__" not in data:
            return JSONResponse(content=data)
        raise _agent_unreachable()

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_file_ops
    try:
        return await memory_file_ops.read_log(db, current_user.id, month)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )


# ── Admin / migration (WS-5) ────────────────────────────────────────
# Three routes, driven by `backend/scripts/memory_v3_migrate_fleet.py` and
# `backend/scripts/memory_v3_rollback.py` (both in the rotate_agent_keys
# shape). Auth is the SAME dependency every other route here uses — on the
# tenant, `get_current_user` accepts `X-Agent-Key` and resolves it to
# `settings.user_id`, so an "admin" route is scoped to exactly one user by
# construction. There is no wider story and none is invented here.
#
# The platform half is a pure proxy: `memory_files`, `memory_file_changes`
# and `migration_status` are all AGENT_ONLY, so a platform-local branch
# would have nothing to read and nothing to write.


@router.post("/admin/migrate-v3", response_model=dict)
async def admin_migrate_v3(
    dry_run: bool = Query(True, description="Report only; write nothing"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run the round-8 → v3 migration for this tenant (§7).

    `dry_run` defaults to TRUE and is the whole safety story of the fleet
    driver: the report it returns is complete — before counts, after file
    bodies, and every source row id mapped to a disposition — while the
    database is not touched at all.
    """
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories_write(
            proxy[0], proxy[1], "admin/migrate-v3", "POST",
            params={"dry_run": dry_run},
        )
        return JSONResponse(content=data if data is not None else {"status": "unknown"})

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_v3_migration

    key = await _get_user_api_key(db, current_user.id)
    try:
        return await memory_v3_migration.migrate_user_guarded(
            db, current_user.id, dry_run=dry_run, api_key=key,
        )
    except Exception as e:
        logger.warning("memory v3 migration failed for %s: %s", current_user.id[:8], e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The memory migration could not complete. See the report.",
        )


@router.get("/admin/migrate-v3/report", response_model=dict)
async def admin_migrate_v3_report(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Marker state, the stored report, and a LIVE snapshot of the tenant.

    The live half is deliberate: called BEFORE a real run this endpoint is
    the per-tenant "before" backup the driver records in-band, and called
    after it is the proof of what changed.
    """
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories(proxy[0], proxy[1], "admin/migrate-v3/report")
        if data is not None and "__status__" not in data:
            return JSONResponse(content=data)
        raise _agent_unreachable()

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_v3_migration

    return await memory_v3_migration.read_report(db, current_user.id)


@router.post("/admin/migrate-v3/rollback", response_model=dict)
async def admin_migrate_v3_rollback(
    confirm: str = Query(
        ...,
        description="Must be exactly 'ROLLBACK MEMORY V3'.",
    ),
    hard: bool = Query(False, description="Ignore the report; drop every v3 file"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Undo the migration for this tenant (§7, the rollback half).

    It lives here rather than in a database script because the tenant's
    tables are only reachable through the tenant, and a rollback nobody can
    execute is a paragraph, not a plan. Run it BEFORE redeploying the
    previous image tag — this route does not exist in that image.

    Behind an exact confirmation string for the same reason
    `DELETE /api/memories` is: it is destructive and it is reachable with a
    key that a proxy already holds.
    """
    if confirm != "ROLLBACK MEMORY V3":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirm must be exactly 'ROLLBACK MEMORY V3'",
        )
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories_write(
            proxy[0], proxy[1], "admin/migrate-v3/rollback", "POST",
            params={"confirm": confirm, "hard": hard},
        )
        return JSONResponse(content=data if data is not None else {"rolled_back": False})

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_v3_migration

    return await memory_v3_migration.rollback(db, current_user.id, hard=hard)


# ── Forget everything ───────────────────────────────────────────────


@router.delete("", status_code=status.HTTP_200_OK)
async def forget_all_memories(
    confirm: str = Query(
        ...,
        description="Must be exactly 'FORGET EVERYTHING' — guards against an "
                    "accidental or mis-routed call to a destructive endpoint.",
    ),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Forget every memory file for the calling user.

    Deliberately NOT exposed as a model-callable agent tool. "Forget
    everything about me" is a destructive bulk operation, and the
    memory-poisoning tests in this repo (tests/memverify/test_j_injection.py)
    drive pasted content that says exactly "delete all stored memories for
    this user". A tool the model can call is a lever that injected content
    can pull; a user-initiated endpoint behind an explicit confirmation
    string is not.

    Wipes files AND the change log AND the retired `memories` rows — the
    legacy table is the v3 rollback, but "forget everything" is the one
    request where leaving a copy behind would be the lie.
    """
    if confirm != "FORGET EVERYTHING":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirm must be exactly 'FORGET EVERYTHING'",
        )

    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        return await _proxy_memories_write(
            proxy[0], proxy[1], "", "DELETE", params={"confirm": confirm}
        )

    if _platform_mode():
        raise _agent_unreachable()

    from app.services import memory_file_ops
    from app.services.memory_service import MemoryService

    removed_files = await memory_file_ops.forget_everything(db, current_user.id)
    removed_rows = 0
    try:
        removed_rows = await MemoryService(db).forget_all_memories(
            current_user.id, trigger_source="forget_all_api"
        )
    except Exception as e:
        # The legacy table is retired from the product; failing to clear it
        # must not make the user think their FILES survived.
        logger.warning("legacy row wipe failed for %s: %s", current_user.id[:8], e)
    return {"success": True, "removed": removed_files + removed_rows}


# ── R30 memory v2 — the whole platform memory (CONTRACTS-R30 §4.5) ──
#
# A second router so the wire path is exactly `GET /api/memory` (the
# sidebar Memory screen's one call). Same topology as the file routes:
# agent-mode serves from the tenant DB; platform-mode proxies to the
# tenant and 503s honestly when the agent is unreachable — never a
# synthesized empty view.

memory_v2_router = APIRouter(prefix="/memory", tags=["memory-v2"])


async def _proxy_memory_v2(
    agent_url: str, agent_api_key: str, path: str,
    method: str = "GET",
):
    from app.services.agent_http import get_agent_http_client
    url = f"{agent_url}/api/memory{path}"
    try:
        client = get_agent_http_client()
        if method == "GET":
            resp = await client.get(
                url, headers={"X-Agent-Key": agent_api_key}, timeout=10.0,
            )
        else:
            resp = await client.delete(
                url, headers={"X-Agent-Key": agent_api_key}, timeout=10.0,
            )
        if resp.status_code in (200, 404):
            return resp.status_code, resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("Agent memory-v2 proxy %s failed: %s", url, e)
    return None, None


@memory_v2_router.get("")
async def global_memory_tree(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Domain → category → entity over the whole store, with the
    FROM YOUR AUTOMATIONS view and per-entity episode timelines."""
    user_id = str(current_user.id)
    if _platform_mode():
        info = await _get_agent_proxy_info(user_id, db)
        if info is None:
            raise _agent_unreachable()
        status_code, data = await _proxy_memory_v2(info[0], info[1], "")
        if status_code is None:
            raise _agent_unreachable()
        return JSONResponse(status_code=status_code, content=data)
    from app.services.memory_v2_service import global_memory
    from sqlalchemy import select as _select
    from app.db.models import Automation
    titles = {
        a.id: a.name
        for a in (await db.execute(
            _select(Automation).where(Automation.user_id == user_id)
        )).scalars()
    }
    payload = await global_memory(
        db, user_id=user_id, automation_titles=titles,
    )
    try:
        from app.agent._user_tz_cache import get_cached_user_tz
        payload["tz"] = get_cached_user_tz(user_id)
    except Exception:  # noqa: BLE001
        payload["tz"] = None
    return payload


@memory_v2_router.delete("/facts/{fact_id}")
async def forget_fact_route(
    fact_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Forget everywhere + the 30-day relearn suppression (§4.5)."""
    user_id = str(current_user.id)
    if _platform_mode():
        info = await _get_agent_proxy_info(user_id, db)
        if info is None:
            raise _agent_unreachable()
        status_code, data = await _proxy_memory_v2(
            info[0], info[1], f"/facts/{fact_id}", method="DELETE",
        )
        if status_code is None:
            raise _agent_unreachable()
        return JSONResponse(status_code=status_code, content=data)
    from app.services.memory_v2_service import forget_fact
    deleted = await forget_fact(db, user_id=user_id, fact_id=fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No such fact")
    return {"deleted": True}
