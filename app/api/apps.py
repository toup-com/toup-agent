"""
Apps & Build Jobs API — REST endpoints for managing user-built apps.

Endpoints:
  GET    /apps/                    - List all apps (with lazy reconciliation)
  GET    /apps/{app_id}            - Get app details
  POST   /apps/{app_id}/start      - Start app (Metro + Web)
  POST   /apps/{app_id}/stop       - Stop app
  POST   /apps/{app_id}/publish-web - Publish web version
  POST   /apps/{app_id}/push-github - Push to GitHub
  DELETE /apps/{app_id}            - Delete app

  GET    /apps/{app_id}/preview/{path} - Reverse-proxy Expo web server (internal)

  GET    /apps/jobs/               - List all build jobs
  GET    /apps/jobs/{job_id}       - Get build job details
  GET    /apps/jobs/{job_id}/logs  - Get structured build logs
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_maker
from app.db.models import App, BuildJob

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/apps", tags=["Apps"])

# ── Module-level refs (set from agent_main.py) ──────────────────────
_app_manager = None
_app_gateway = None
_app_builder_skill = None
_agent_runner = None
_ws_broadcast = None


def set_app_manager(app_manager):
    """Wire the AppManager instance (called from agent_main.py lifespan)."""
    global _app_manager
    _app_manager = app_manager


def set_app_gateway(gateway):
    """Wire the AppGatewaySkill instance so delete can unregister apps."""
    global _app_gateway
    _app_gateway = gateway


def set_app_builder_skill(skill):
    """Wire the AppBuilderSkill instance for resume builds."""
    global _app_builder_skill
    _app_builder_skill = skill


def set_agent_runner(runner, ws_broadcast=None):
    """Wire the AgentRunner instance for executing dashboard tasks."""
    global _agent_runner, _ws_broadcast
    _agent_runner = runner
    _ws_broadcast = ws_broadcast


# ── Response schemas ────────────────────────────────────────────────

class AppResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    slug: str
    status: str
    source: str = "app_builder"  # app_builder, vibecoding, filesystem_discovered
    port: Optional[int] = None
    web_port: Optional[int] = None
    qr_url: Optional[str] = None
    web_url: Optional[str] = None
    publish_url: Optional[str] = None
    github_url: Optional[str] = None
    db_type: str = "none"
    platforms: List[str] = ["web", "ios"]
    created_at: str
    updated_at: str
    # Checkpoint 4a: parsed agentSkill.json manifest (null if missing/invalid)
    skill_json: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    id: str
    app_id: Optional[str] = None
    job_type: str = "auto_builder"  # auto_builder, vibe_code, agent_task
    title: str
    prompt: str
    status: str
    steps: List[Dict[str, Any]]
    model: str = ""
    total_tokens: int = 0
    # DEPRECATED as a display field. Retained so legacy clients that
    # already render it keep working during the mobile rollout; new
    # clients MUST render `user_message` instead. Scheduled for removal
    # once the App Store build carrying the taxonomy is the floor.
    error_message: Optional[str] = None
    # ── Error taxonomy (Mission Control overhaul) ────────────────────
    # `technical_detail` is deliberately ABSENT from this model and must
    # never be added: it is internal-only telemetry. If you find yourself
    # wanting it client-side, the correct move is to widen `user_message`
    # copy or add a taxonomy class — not to ship the exception text.
    error_class: Optional[str] = None
    user_message: Optional[str] = None
    #: Set when status == 'waiting_on_user'. Drives the detail screen's
    #: primary CTA. See docs/audits/mission-control-design.md §2.
    required_action: Optional[Dict[str, Any]] = None
    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    archived_at: Optional[str] = None
    paused_at: Optional[str] = None
    resume_after: Optional[str] = None
    layer: int = 1
    layer2_changes: Optional[List[Dict[str, Any]]] = None
    # Back-link from a job to the Message its handler produced. Set by
    # JobRunner on routine/trigger/agent-task fires (mig 046 unified-
    # jobs arc) — null for app-builder jobs and for any job whose
    # handler hadn't written its result yet. Frontend uses this to
    # deep-link "Open in chat" to the exact agent reply.
    summary_message_id: Optional[str] = None
    # Source linkage (mig 046 unified-jobs arc). When this job was
    # created by a routine fire, trigger fire, or agent task, these
    # point at the parent. The Jobs detail UI uses them to fetch the
    # parent routine/trigger's config + sibling fire history so the
    # detail page can render as a proper scheduled-automation surface
    # rather than just an isolated job row.
    source_kind: Optional[str] = None
    source_id: Optional[str] = None
    fire_instant: Optional[str] = None
    attempt: Optional[int] = None
    # Sub-agent arc (Phase 6 cost-attribution exposure). Null for any
    # non-sub-agent job; populated by the orchestrator on sub-agent
    # rows. ``credit_spent`` is the running total updated by the
    # LLM-proxy credit hook; ``credit_budget_allocated`` is the slice
    # the parent allocated (null when budget enforcement is off).
    # ``parent_job_id`` lets the Jobs detail UI render the parent
    # chain breadcrumb. ``outcome`` is the per-type sub-state already
    # written by other handlers; sub-agents use {success, failed,
    # timeout, cancelled, budget_exhausted}.
    parent_job_id: Optional[str] = None
    credit_budget_allocated: Optional[float] = None
    credit_spent: Optional[float] = None
    outcome: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


# ── Helpers ─────────────────────────────────────────────────────────

def _get_user_id() -> str:
    """Get user_id from settings (agent mode — single user per VPS)."""
    from app.config import settings
    return settings.user_id


async def _app_to_response(app: App) -> AppResponse:
    """Convert App model to response, enriching with live status and manifest."""
    qr_url = None
    web_url = None
    if _app_manager and app.status == "running":
        qr_url = await _app_manager.get_qr_url(app.id)
        web_url = await _app_manager.get_web_url(app.id)

    platforms = app.platforms.split(",") if app.platforms else ["web", "ios"]

    # Load agentSkill.json manifest (Checkpoint 4a)
    skill_json = None
    try:
        from app.services.app_manifest_loader import load_app_manifest
        if _app_manager:
            app_dir = await _app_manager._resolve_app_dir(app.id)
            manifest = load_app_manifest(app_dir, app_id=app.id)
            if manifest:
                skill_json = manifest.model_dump()
    except Exception as e:
        logger.debug("Failed to load manifest for %s: %s", app.id[:8], e)

    return AppResponse(
        id=app.id,
        name=app.name,
        description=app.description,
        slug=app.slug,
        status=app.status,
        source=getattr(app, 'source', None) or "app_builder",
        port=app.port,
        web_port=app.web_port,
        qr_url=qr_url,
        web_url=web_url,
        publish_url=app.publish_url,
        github_url=app.github_url,
        db_type=app.db_type or "none",
        platforms=platforms,
        created_at=app.created_at.isoformat() if app.created_at else "",
        updated_at=app.updated_at.isoformat() if app.updated_at else "",
        skill_json=skill_json,
    )


def _taxonomy_fields(job: BuildJob) -> Dict[str, Any]:
    """Resolve `error_class` / `user_message` for one job row.

    Prefers the stored values; falls back to classifying `error_message`
    on read. The fallback is what lets pre-taxonomy rows — including the
    79-day-old corpses on the founder's board — render humanized copy
    with no data migration and no risk of a backfill going wrong.

    `technical_detail` is NEVER returned. See JobResponse.
    """
    from app.agent.job_status import classify

    stored_class = getattr(job, "error_class", None)
    stored_msg = getattr(job, "user_message", None)
    if stored_class:
        return {"error_class": stored_class, "user_message": stored_msg}

    raw = getattr(job, "error_message", None)
    if not raw:
        return {"error_class": None, "user_message": None}

    verdict = classify(raw)
    return {"error_class": verdict.error_class, "user_message": verdict.user_message}


def _job_to_response(job: BuildJob) -> JobResponse:
    """Convert BuildJob model to response."""
    steps = []
    try:
        steps = json.loads(job.steps_json) if job.steps_json else []
    except (json.JSONDecodeError, TypeError):
        pass

    layer2_changes = None
    try:
        if getattr(job, 'layer2_changes_json', None):
            layer2_changes = json.loads(job.layer2_changes_json)
    except (json.JSONDecodeError, TypeError):
        pass

    # Alias the real model id to a neutral tier label on this user-facing job
    # card (docs/security/audit-2026.md MI-2). Flag-gated (default off).
    _job_model = job.model or ""
    from app.config import settings as _settings
    if _settings.security_leak_filter and _job_model:
        from app.services.model_alias import public_model_label
        _job_model = public_model_label(_job_model)

    return JobResponse(
        id=job.id,
        app_id=job.app_id,
        job_type=getattr(job, 'job_type', None) or "auto_builder",
        title=job.title,
        prompt=job.prompt,
        status=job.status,
        steps=steps,
        model=_job_model,
        total_tokens=job.total_tokens or 0,
        error_message=job.error_message,
        # Classify on read for rows written before the taxonomy landed
        # (and for any writer not yet migrated), so the client NEVER has
        # to fall back to `error_message`. `classify` is a handful of
        # pre-compiled regexes over a short string — cheap enough for a
        # list serializer, and it means the 79-day-old legacy corpses
        # render with humanized copy without a data migration.
        **_taxonomy_fields(job),
        progress_step=getattr(job, 'progress_step', None),
        progress_total=getattr(job, 'progress_total', None),
        archived_at=(
            job.archived_at.isoformat() if getattr(job, 'archived_at', None) else None
        ),
        paused_at=job.paused_at.isoformat() if getattr(job, 'paused_at', None) else None,
        resume_after=job.resume_after.isoformat() if getattr(job, 'resume_after', None) else None,
        layer=getattr(job, 'layer', 1) or 1,
        layer2_changes=layer2_changes,
        summary_message_id=getattr(job, 'summary_message_id', None),
        source_kind=getattr(job, 'source_kind', None),
        source_id=getattr(job, 'source_id', None),
        fire_instant=(
            job.fire_instant.isoformat() if getattr(job, 'fire_instant', None) else None
        ),
        attempt=getattr(job, 'attempt', None),
        # Sub-agent arc — exposed on every job row (None for
        # non-sub-agent rows, which keeps the field optional in the
        # client schema).
        parent_job_id=getattr(job, 'parent_job_id', None),
        credit_budget_allocated=getattr(job, 'credit_budget_allocated', None),
        credit_spent=getattr(job, 'credit_spent', None),
        outcome=getattr(job, 'outcome', None),
        created_at=job.created_at.isoformat() if job.created_at else "",
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


# ── Local reconciliation (agent VPS — direct filesystem access) ────

async def _reconcile_local(user_id: str, db: AsyncSession):
    """Run lazy reconciliation using the local filesystem (no SSH needed).

    The agent runs on the same VPS as the app files, so we can use os.listdir.
    """
    from app.services.reconciliation_service import reconcile_apps, _last_reconcile, _LAZY_COOLDOWN_SECONDS
    import time

    # Quick cooldown check before importing/calling anything heavy
    last = _last_reconcile.get(user_id, 0)
    if time.time() - last < _LAZY_COOLDOWN_SECONDS:
        return

    from app.config import settings
    from app.agent.app_manager import APPS_DIR

    workspace = getattr(settings, 'agent_workspace_dir', None) or './workspace'

    async def local_find(cmd: str) -> str:
        """Simulate SSH find by listing local dirs."""
        import asyncio
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()

    try:
        await reconcile_apps(
            user_id=user_id,
            db=db,
            ssh_cmd_fn=local_find,
            trigger="lazy",
            apps_dir=APPS_DIR,
            workspace_dir=os.path.abspath(workspace),
        )
    except Exception as e:
        logger.warning("[RECONCILE] Lazy reconciliation failed: %s", e)


# ── App Endpoints ───────────────────────────────────────────────────

@router.get("/")
async def list_apps() -> List[AppResponse]:
    """List all apps for the current user, with lazy reconciliation."""
    user_id = _get_user_id()
    async with async_session_maker() as db:
        # Lazy reconciliation: scan filesystem for orphaned dirs / missing DB rows
        await _reconcile_local(user_id, db)

        result = await db.execute(
            select(App).where(App.user_id == user_id).order_by(App.created_at.desc())
        )
        apps = result.scalars().all()
        return [await _app_to_response(app) for app in apps]


@router.get("/jobs/")
async def list_jobs(include_ticks: bool = False) -> List[JobResponse]:
    """List all build jobs for the current user.

    Autopilot ticks (job_type='autopilot_tick') are engine heartbeats,
    not user tasks — the mission itself is the visible unit (Mission
    Control missions). Excluded by default; ``?include_ticks=true``
    keeps them reachable for debugging."""
    user_id = _get_user_id()
    async with async_session_maker() as db:
        where = [BuildJob.user_id == user_id]
        if not include_ticks:
            where.append(BuildJob.job_type != "autopilot_tick")
        result = await db.execute(
            select(BuildJob).where(*where).order_by(BuildJob.created_at.desc())
        )
        jobs = result.scalars().all()
        return [_job_to_response(job) for job in jobs]


class ImportFromToupCodeFile(BaseModel):
    """One file in an import-from-toup-code payload."""
    path: str                 # repo-relative; no leading slash
    content: str              # utf-8 text; binaries rejected up-stream


class ImportFromToupCodeRequest(BaseModel):
    """Materialize a Toup Code session as a real App in this user's
    container.

    The platform-side Toup Code surface (`POST /api/code/save-to-
    workspace` on toup.ai) reads the project's files from its own
    workspace at `/app/workspace/toup-code/<user_id>/<project>/` and
    forwards them here with X-Agent-Key auth. We then:

      1. Slugify the name + dedupe against existing apps.
      2. mkdir the target at `/opt/toup-agent/apps/<slug>/`.
      3. Write every file via `app_manager.write_app_files` so the
         path normalisation + sanity assert lives in one place.
      4. INSERT App row with `source='toup-code'`, `status='ready'`,
         a populated `files_json` backup, and the resolved `app_dir`.

    Returns `{app_id, slug, app_dir, file_count}` so the platform can
    forward to the frontend and the UI can deep-link to the new app.
    """
    name: str                                    # human-readable; → App.name
    description: Optional[str] = None
    files: List[ImportFromToupCodeFile]
    source_metadata: Optional[Dict[str, Any]] = None  # opaque (conv_id, project, etc.)


class ImportFromToupCodeResponse(BaseModel):
    app_id: str
    slug: str
    app_dir: str
    file_count: int
    web_url: Optional[str] = None


# Caps mirror what the platform side enforces — the agent fails closed
# on anything bigger so a misbehaving platform can't fill the disk.
_IMPORT_MAX_FILES = 500
_IMPORT_MAX_BYTES_PER_FILE = 5 * 1024 * 1024     # 5 MB per file
_IMPORT_MAX_TOTAL_BYTES = 50 * 1024 * 1024       # 50 MB total
_IMPORT_FORBIDDEN_PATH_SEGMENTS = {"..", ".git", "node_modules"}


def _slugify_app_name(name: str) -> str:
    """Match the slug shape `app_builder` uses so Toup-Code-imported
    apps live alongside hand-built ones with the same naming pattern."""
    import re
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-")[:40]
    return slug or "toup-code-app"


def _validate_import_path(rel_path: str) -> str:
    """Reject path traversal + dangerous segments. Returns the
    sanitised relative path (no leading slash, no empty segments)."""
    if not rel_path or not isinstance(rel_path, str):
        raise HTTPException(status_code=400, detail="empty file path")
    s = rel_path.strip().lstrip("/").lstrip("\\")
    if not s:
        raise HTTPException(status_code=400, detail="empty file path after strip")
    parts = s.replace("\\", "/").split("/")
    for seg in parts:
        if not seg:
            raise HTTPException(status_code=400, detail=f"empty path segment in {rel_path!r}")
        if seg in _IMPORT_FORBIDDEN_PATH_SEGMENTS:
            raise HTTPException(
                status_code=400,
                detail=f"forbidden path segment {seg!r} in {rel_path!r}",
            )
    return "/".join(parts)


@router.post("/import-from-toup-code", response_model=ImportFromToupCodeResponse)
async def import_from_toup_code(req: ImportFromToupCodeRequest) -> ImportFromToupCodeResponse:
    """Materialise a Toup Code session as a tenant-owned App.

    Auth: relies on the upstream X-Agent-Key middleware (same gate
    every authed endpoint on the agent uses). The endpoint itself
    pulls user_id from `settings.user_id` — single-user-per-container.

    Idempotency: the slug is unique. A second call with the same name
    gets a `-2`, `-3`, … suffix so re-saving the same Toup Code
    session never overwrites a previous import. Re-saving is a user
    choice — they may want to compare versions.
    """
    if _app_manager is None:
        raise HTTPException(
            status_code=503,
            detail="app manager not initialised; agent boot incomplete",
        )

    # ── Validate payload ──
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="`name` is required")
    if len(name) > 200:
        raise HTTPException(status_code=400, detail="`name` too long (200 max)")

    if not req.files:
        raise HTTPException(status_code=400, detail="`files` is empty — nothing to import")
    if len(req.files) > _IMPORT_MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"too many files ({len(req.files)} > {_IMPORT_MAX_FILES})",
        )

    sanitised_files: Dict[str, str] = {}
    total_bytes = 0
    for f in req.files:
        rel = _validate_import_path(f.path)
        if rel in sanitised_files:
            raise HTTPException(status_code=400, detail=f"duplicate path: {rel}")
        content = f.content or ""
        size = len(content.encode("utf-8"))
        if size > _IMPORT_MAX_BYTES_PER_FILE:
            raise HTTPException(
                status_code=413,
                detail=f"file too large: {rel} ({size} > {_IMPORT_MAX_BYTES_PER_FILE})",
            )
        total_bytes += size
        if total_bytes > _IMPORT_MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"import payload exceeds {_IMPORT_MAX_TOTAL_BYTES} bytes",
            )
        sanitised_files[rel] = content

    # ── Slug + dir reservation ──
    import uuid
    from app.agent.app_manager import APPS_DIR
    base_slug = _slugify_app_name(name)
    app_id = str(uuid.uuid4())
    user_id = _get_user_id()

    async with async_session_maker() as db:
        # Dedupe loop — DB UNIQUE on App.slug is the source of truth.
        slug = base_slug
        for attempt in range(1, 200):
            existing = await db.execute(
                select(App.id).where(App.slug == slug).limit(1)
            )
            if not existing.scalar():
                break
            slug = f"{base_slug}-{attempt}"
        else:
            raise HTTPException(
                status_code=409,
                detail=f"could not find a free slug under {base_slug!r}",
            )

        app_dir = os.path.join(APPS_DIR, slug)
        try:
            os.makedirs(app_dir, exist_ok=True)
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"could not create {app_dir}: {e}",
            )

        # ── Write files ──
        try:
            for rel_path, content in sanitised_files.items():
                full_path = os.path.join(app_dir, rel_path)
                # Defense-in-depth: assert resolved path stays under app_dir.
                real_full = os.path.realpath(full_path)
                real_root = os.path.realpath(app_dir)
                if os.path.commonpath([real_full, real_root]) != real_root:
                    raise HTTPException(
                        status_code=400,
                        detail=f"resolved path escapes app_dir: {rel_path}",
                    )
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as fh:
                    fh.write(content)
        except HTTPException:
            raise
        except Exception as e:
            # Best-effort cleanup so a half-imported app doesn't poison
            # the filesystem on partial failure.
            try:
                import shutil
                shutil.rmtree(app_dir, ignore_errors=True)
            except Exception:
                pass
            logger.exception(
                "[apps] import-from-toup-code write failed slug=%s: %s", slug, e,
            )
            raise HTTPException(status_code=500, detail=f"write failed: {e}")

        # Pre-create a `storage/` dir to mirror `scaffold_app`'s contract
        # — apps that later add SQLite persistence rely on it existing.
        try:
            os.makedirs(os.path.join(app_dir, "storage"), exist_ok=True)
        except OSError:
            pass

        # ── INSERT App row ──
        files_backup = json.dumps(sanitised_files)
        plan_json = None
        if req.source_metadata is not None:
            plan_json = json.dumps({"source": "toup-code", **req.source_metadata})

        app = App(
            id=app_id,
            user_id=user_id,
            name=name,
            description=(req.description or "").strip() or None,
            slug=slug,
            status="ready",            # Files are on disk; no build step needed
            source="toup-code",        # Distinct from app_builder / vibecoding
            app_dir=app_dir,
            files_json=files_backup,
            deps_json="{}",
            db_type="none",
            plan_json=plan_json,
            platforms="web",           # Static HTML drops are web-only
        )
        db.add(app)
        await db.commit()
        await db.refresh(app)

    logger.info(
        "[apps] imported from toup-code app_id=%s slug=%s files=%d bytes=%d",
        app_id, slug, len(sanitised_files), total_bytes,
    )

    # Best-effort public URL — same path the `start_app` flow returns.
    web_url: Optional[str] = None
    try:
        if _app_manager:
            web_url = await _app_manager.get_web_url(app_id)  # type: ignore[arg-type]
    except Exception:
        web_url = None

    return ImportFromToupCodeResponse(
        app_id=app_id,
        slug=slug,
        app_dir=app_dir,
        file_count=len(sanitised_files),
        web_url=web_url,
    )


class CreateJobRequest(BaseModel):
    title: str
    description: str = ""


@router.post("/jobs/")
async def create_job(req: CreateJobRequest) -> JobResponse:
    """Create a new agent task job from the dashboard and execute it."""
    user_id = _get_user_id()
    # PR 4c (unified-jobs arc): repoint through ``JobRunner.create_job``
    # so the unified-arc columns (``source_kind``, ``source_id``,
    # ``conversation_id``, ``idempotency_key``) are populated. Behavior
    # preservation: status='running' (this path executes inline,
    # never sits in 'queued'); layer=0 (agent task vs auto_builder=1);
    # no idempotency_key (dashboard input has no natural dedupe key —
    # the user clicking "submit" twice creates two distinct jobs).
    from app.agent.job_runner import JobRunner, TaskSpec
    spec = TaskSpec(
        # Unattended (audit-2026 re-audit round 9): a dashboard task runs
        # fire-and-forget with nobody watching each tool call, so it must use a
        # channel in _MUTATES_UNATTENDED_DENY_CHANNELS — NOT "web" — otherwise
        # injected content in an ingested doc/web/email could drive a mutating
        # connector (gmail send / calendar-drive write) without a confirmation
        # gate. A user's explicit per-tool allow still overrides.
        user_id=user_id,
        channel="agent_task",
        source_kind="manual",
    )
    job = await JobRunner().create_job(
        job_type="agent_task",
        spec=spec,
        title=req.title,
        prompt=req.description or req.title,
        status="running",
        layer=0,
    )
    job_id = job.id

    # Execute the task via the agent runner in background
    if _agent_runner:
        async def _run_dashboard_task():
            from app.agent.job_logger import JobLogger
            blog = JobLogger(job_id, user_id, ws_broadcast=_ws_broadcast)
            try:
                await blog.info(f"Starting task: {req.title}")
                result = await _agent_runner.run(
                    user_message=req.description or req.title,
                    user_id=user_id,
                    session_id=f"dashboard-task-{job_id[:8]}",
                    channel="agent_task",  # unattended — see TaskSpec above (round 9)
                    # Phase 8: enable parent_job_id linkage so any
                    # sub-agent spawned during this dashboard task
                    # lands as a child row in Mission Control.
                    current_job_id=job_id,
                    on_tool_start=lambda name, args: asyncio.ensure_future(
                        blog.tool(f"Using {name}", meta={"tool": name})
                    ),
                )
                blog._total_tokens = getattr(result, 'tokens_total', 0)
                await blog.info(f"Task completed ({getattr(result, 'tokens_total', 0):,} tokens)")
                await blog.persist()
                # Mark completed
                async with async_session_maker() as db:
                    j = await db.get(BuildJob, job_id)
                    if j:
                        j.status = "completed"
                        j.completed_at = datetime.utcnow()
                        j.total_tokens = getattr(result, 'tokens_total', 0)
                        j.model = getattr(result, 'model', '')
                        await db.commit()
                # Broadcast completion
                if _ws_broadcast:
                    await _ws_broadcast(user_id, {
                        "type": "job_update",
                        "job_id": job_id,
                        "status": "completed",
                        "name": req.title,
                    })
            except Exception as e:
                logger.error(f"[DASHBOARD] Task {job_id[:8]} failed: {e}")
                await blog.error(f"Task failed: {e}")
                await blog.persist()
                from app.agent.job_status import (
                    DISPOSITION_NEEDS_USER, STATUS_WAITING_ON_USER, classify,
                    technical_detail,
                )

                _v = classify(e)
                _st = (
                    STATUS_WAITING_ON_USER
                    if _v.disposition == DISPOSITION_NEEDS_USER
                    else "failed"
                )
                async with async_session_maker() as db:
                    j = await db.get(BuildJob, job_id)
                    if j:
                        j.status = _st
                        j.error_message = str(e)[:500]
                        j.error_class = _v.error_class
                        j.user_message = _v.user_message
                        j.technical_detail = technical_detail(e)
                        j.completed_at = (
                            None if _st == STATUS_WAITING_ON_USER else datetime.utcnow()
                        )
                        await db.commit()
                if _ws_broadcast:
                    # The frame carried `str(e)[:200]` — a raw exception on a
                    # live socket straight into the chat UI. Ship taxonomy only.
                    await _ws_broadcast(user_id, {
                        "type": "job_update",
                        "job_id": job_id,
                        "status": _st,
                        "error_class": _v.error_class,
                        "user_message": _v.user_message,
                    })
                # Keep the lock screen / Dynamic Island honest. Without
                # this the card spins on stale progress forever: only a
                # notification closes or re-states a Live Activity, and a
                # DB write alone never touches it.
                if _st == STATUS_WAITING_ON_USER and _v.required_action:
                    try:
                        from app.agent.subagent_orchestrator import notify_job_needs_user

                        await notify_job_needs_user(
                            job_id=job_id, label=title,
                            summary=_v.user_message or "Your agent needs your input.",
                            action_type=_v.required_action,
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("[DASHBOARD] waiting notify skipped")

        asyncio.create_task(_run_dashboard_task())

    return _job_to_response(job)


class UpdateJobStatusRequest(BaseModel):
    #: Any value in `job_status.ALL_STATUSES`. Do NOT re-inline a tuple here —
    #: the hard-coded 4-value tuple this replaced 400'd on `waiting_on_user`
    #: and on the `cancelled`/`timeout`/`budget_exhausted` the sub-agent arc
    #: already writes. Optional so a request can archive without a status change.
    status: Optional[str] = None
    #: Soft-retire from the Activity feed. Archiving NEVER deletes: it stamps
    #: `archived_at`, and the list endpoints filter on it. The audit found no
    #: retention of any kind, so 79-day-old corpses of an already-fixed bug
    #: were still rendering on the founder's board.
    archived: Optional[bool] = None


@router.patch("/jobs/{job_id}")
async def update_job_status(job_id: str, req: UpdateJobStatusRequest) -> JobResponse:
    """Update a job's status and/or archive it."""
    from app.agent.job_status import ALL_STATUSES, STATUS_COMPLETED

    if req.status is not None and req.status not in ALL_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    if req.status is None and req.archived is None:
        raise HTTPException(status_code=400, detail="Nothing to update")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if req.status is not None:
            job.status = req.status
            if req.status == STATUS_COMPLETED and not job.completed_at:
                job.completed_at = datetime.utcnow()
        if req.archived is not None:
            job.archived_at = datetime.utcnow() if req.archived else None
        await db.commit()
        await db.refresh(job)
        return _job_to_response(job)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobResponse:
    """Get a specific build job."""
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_response(job)


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str) -> Dict[str, Any]:
    """Get structured build logs for a job."""
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        logs = []
        try:
            logs = json.loads(job.build_logs_json) if job.build_logs_json else []
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "job_id": job_id,
            "status": job.status,
            "total_tokens": job.total_tokens or 0,
            "logs": logs,
        }


@router.post("/jobs/{job_id}/fix")
async def fix_failed_job(job_id: str) -> Dict[str, Any]:
    """Auto-fix a failed build by patching known infra files and restarting.

    Replaces broken agentBridge.ts/agentActions.ts with reliable stubs,
    restarts the web server, and updates job + app status.
    """
    if not _app_manager:
        raise HTTPException(status_code=503, detail="App manager not available")

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "failed":
            return {"status": job.status, "message": "Job is not in failed state"}

        # Only fix builds that failed at "starting" (preview servers) step.
        # Builds that failed at code gen or earlier need a full rebuild.
        try:
            steps = json.loads(job.steps_json) if job.steps_json else []
            failed_step = next((s for s in steps if s.get("status") == "failed"), None)
            if failed_step and failed_step.get("type") not in ("starting", "ready"):
                return {
                    "status": "cannot_fix",
                    "message": f"Build failed at '{failed_step.get('label', 'unknown')}' — needs a full rebuild, not a quick fix.",
                }
        except (json.JSONDecodeError, TypeError):
            pass

        app = await db.get(App, job.app_id)
        if not app or not app.app_dir:
            raise HTTPException(status_code=404, detail="App not found")

        import os

        # ── Patch known infra files ──
        patches_applied = []

        bridge_path = os.path.join(app.app_dir, "lib", "agentBridge.ts")
        if os.path.exists(bridge_path):
            with open(bridge_path, "r") as f:
                content = f.read()
            if len(content) > 500 or "AgentScreen" not in content:
                bridge_stub = (
                    "// AgentBridge — delegates to platform-injected bridge.\n"
                    "const injected = typeof window !== 'undefined' && (window as any).__TOUP_AGENT_BRIDGE;\n"
                    "const agentBridge = injected || {\n"
                    "  isConnected: false, currentScreen: 'Home',\n"
                    "  sendMessage: async () => {}, onAgentMessage: () => () => {},\n"
                    "  onToolActivity: () => () => {}, setNavigationRef: () => {},\n"
                    "  navigate: () => {}, getScreens: () => [], getActions: () => [],\n"
                    "  setScreens: () => {}, setActions: () => {}, destroy: () => {},\n"
                    "};\n"
                    "export type AgentScreen = { name: string; description: string };\n"
                    "export type AgentActionMeta = { id: string; label: string; handler: string };\n"
                    "export { agentBridge };\n"
                    "export const AgentBridge = agentBridge;\n"
                    "export default agentBridge;\n"
                )
                with open(bridge_path, "w") as f:
                    f.write(bridge_stub)
                patches_applied.append("agentBridge.ts")

        actions_path = os.path.join(app.app_dir, "lib", "agentActions.ts")
        if os.path.exists(actions_path):
            with open(actions_path, "r") as f:
                content = f.read()
            if len(content) > 500:
                actions_stub = (
                    "type AgentAction = { id: string; label: string; description: string; handler: (...args: any[]) => Promise<any> };\n"
                    "const registry: Record<string, AgentAction[]> = {};\n"
                    "export function registerAction(screen: string, action: AgentAction) {\n"
                    "  if (!registry[screen]) registry[screen] = [];\n"
                    "  registry[screen].push(action);\n"
                    "}\n"
                    "export function getActions(screen?: string): AgentAction[] {\n"
                    "  if (!screen) return Object.values(registry).flat();\n"
                    "  return registry[screen] || [];\n"
                    "}\n"
                    "export default { registerAction, getActions };\n"
                )
                with open(actions_path, "w") as f:
                    f.write(actions_stub)
                patches_applied.append("agentActions.ts")

        # ── Restart web server ──
        try:
            await _app_manager.stop_app(app.id)
        except Exception:
            pass

        try:
            metro_port = await _app_manager.start_metro(app.id)
            web_port = await _app_manager.start_web(app.id)

            app.status = "running"
            app.port = metro_port
            app.web_port = web_port
            managed = _app_manager._running.get(app.id)
            if managed:
                app.metro_pid = managed.metro_process.pid if managed.metro_process else None
                app.web_pid = managed.web_process.pid if managed.web_process else None

            job.status = "completed"
            job.error_message = None
            from datetime import datetime
            job.completed_at = datetime.utcnow()

            # Mark the failed "starting" step as done
            try:
                steps = json.loads(job.steps_json) if job.steps_json else []
                for s in steps:
                    if s.get("status") == "failed":
                        s["status"] = "done"
                    elif s.get("status") == "pending":
                        s["status"] = "done"
                job.steps_json = json.dumps(steps)
            except (json.JSONDecodeError, TypeError):
                pass

            await db.commit()

            web_url = await _app_manager.get_web_url(app.id)
            return {
                "status": "fixed",
                "patches": patches_applied,
                "web_url": web_url,
                "app_id": app.id,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Server restart failed: {e}",
                "patches": patches_applied,
            }


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> Dict[str, Any]:
    """Resume a paused build job. Triggers background resume via app_builder skill."""
    from datetime import datetime

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status != "paused":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not paused (status: {job.status}). Only paused jobs can be resumed."
            )

        # Check if enough time has passed
        if job.resume_after and datetime.utcnow() < job.resume_after:
            remaining = (job.resume_after - datetime.utcnow()).total_seconds()
            return {
                "ok": False,
                "message": f"Token limit hasn't reset yet. Try again in {int(remaining)}s",
                "resume_after": job.resume_after.isoformat(),
                "retry_after_seconds": int(remaining),
            }

        # Load checkpoint
        checkpoint = {}
        try:
            checkpoint = json.loads(job.checkpoint_json) if job.checkpoint_json else {}
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(status_code=500, detail="Could not load checkpoint data")

        if not checkpoint:
            raise HTTPException(status_code=500, detail="No checkpoint data found")

        # Mark as running
        job.status = "running"
        job.paused_at = None
        job.resume_after = None
        await db.commit()

        # Update app status
        if job.app_id:
            app = await db.get(App, job.app_id)
            if app:
                app.status = "building"
                await db.commit()

    # Trigger resume via the app_builder skill
    import asyncio
    if _app_builder_skill and hasattr(_app_builder_skill, '_resume_build_app'):
        user_id = _get_user_id()
        asyncio.create_task(
            _app_builder_skill._resume_build_app(job_id, checkpoint, user_id)
        )

    return {
        "ok": True,
        "message": "Build resuming. Check the Jobs tab for progress.",
        "job_id": job_id,
        "checkpoint_step": checkpoint.get("current_step", "unknown"),
        "completed_steps": checkpoint.get("completed_steps", []),
    }


@router.get("/{app_id}")
async def get_app(app_id: str) -> AppResponse:
    """Get a specific app by ID or slug."""
    async with async_session_maker() as db:
        # Try by ID first
        app = await db.get(App, app_id)
        # Fall back to slug lookup
        if not app:
            result = await db.execute(select(App).where(App.slug == app_id).limit(1))
            app = result.scalar_one_or_none()
        if not app:
            raise HTTPException(status_code=404, detail="App not found")
        return await _app_to_response(app)


@router.post("/{app_id}/start")
async def start_app(app_id: str) -> Dict[str, Any]:
    """Start Metro + Web servers for an app."""
    if not _app_manager:
        raise HTTPException(status_code=503, detail="App manager not available")

    async with async_session_maker() as db:
        app = await db.get(App, app_id)
        if not app:
            raise HTTPException(status_code=404, detail="App not found")

        metro_port = await _app_manager.start_metro(app_id)
        web_port = await _app_manager.start_web(app_id)

        app.status = "running"
        app.port = metro_port
        app.web_port = web_port
        managed = _app_manager._running.get(app_id)
        if managed:
            app.metro_pid = managed.metro_process.pid if managed.metro_process else None
            app.web_pid = managed.web_process.pid if managed.web_process else None
        await db.commit()

        qr_url = await _app_manager.get_qr_url(app_id)
        web_url = await _app_manager.get_web_url(app_id)

        return {
            "status": "running",
            "port": metro_port,
            "web_port": web_port,
            "qr_url": qr_url,
            "web_url": web_url,
        }


@router.post("/{app_id}/stop")
async def stop_app(app_id: str) -> Dict[str, str]:
    """Stop app servers."""
    if not _app_manager:
        raise HTTPException(status_code=503, detail="App manager not available")

    async with async_session_maker() as db:
        app = await db.get(App, app_id)
        if not app:
            raise HTTPException(status_code=404, detail="App not found")

        await _app_manager.stop_app(app_id)
        app.status = "stopped"
        app.port = None
        app.web_port = None
        app.metro_pid = None
        app.web_pid = None
        await db.commit()

    return {"status": "stopped"}


@router.post("/{app_id}/publish-web")
async def publish_web(app_id: str, domain: Optional[str] = None) -> Dict[str, str]:
    """Export and publish web version."""
    if not _app_manager:
        raise HTTPException(status_code=503, detail="App manager not available")

    async with async_session_maker() as db:
        app = await db.get(App, app_id)
        if not app:
            raise HTTPException(status_code=404, detail="App not found")

        url = await _app_manager.publish_web(app_id, domain=domain)
        app.publish_url = url
        await db.commit()

    return {"url": url}


@router.post("/{app_id}/push-github")
async def push_github(app_id: str) -> Dict[str, str]:
    """Push current code to GitHub."""
    if not _app_manager:
        raise HTTPException(status_code=503, detail="App manager not available")

    async with async_session_maker() as db:
        app = await db.get(App, app_id)
        if not app:
            raise HTTPException(status_code=404, detail="App not found")

        result = await _app_manager.push_to_github(app_id)

    return {"result": result, "repo_url": app.github_url or ""}


@router.post("/jobs/{job_id}/fix-steps")
async def fix_job_steps(job_id: str) -> Dict[str, Any]:
    """Fix stuck build steps — mark running/pending as done if job is completed."""
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        steps = []
        try:
            steps = json.loads(job.steps_json) if job.steps_json else []
        except (json.JSONDecodeError, TypeError):
            pass
        fixed = 0
        for s in steps:
            if s.get("status") in ("running", "pending"):
                s["status"] = "done"
                fixed += 1
        if fixed:
            job.steps_json = json.dumps(steps)
            if job.status in ("running", "queued"):
                job.status = "completed"
            await db.commit()
        return {"ok": True, "fixed_steps": fixed}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> Dict[str, bool]:
    """Delete a build job record."""
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        await db.delete(job)
        await db.commit()
    return {"ok": True}


@router.get("/{app_id}/preview/{path:path}")
@router.get("/{app_id}/preview")
async def agent_preview_proxy(app_id: str, request: Request, path: str = ""):
    """Reverse-proxy the Expo web dev server running inside this container.

    The platform calls this endpoint instead of hitting the web_port directly,
    because the web_port is only accessible inside the Docker container.

    Includes a server-side retry loop (up to 15s) to absorb the brief dead
    window after app__restart kills and respawns the dev server processes.
    The browser makes ONE request; the proxy absorbs the race transparently.
    """
    import asyncio as _asyncio

    if not _app_manager:
        raise HTTPException(503, "App manager not initialized")

    managed = _app_manager._running.get(app_id)
    if not managed or not managed.web_port:
        raise HTTPException(503, "App web server not running")

    # Lazy-revive: TCP-probe the upstream before entering the retry loop.
    # If the Expo dev server died (process crashed, SIGKILL'd by oom-killer,
    # whatever), this triggers an immediate respawn rather than waiting up
    # to 30s for the background watchdog. Cheaper than 15× failed httpx
    # connects, and the user's own request becomes the recovery event.
    try:
        await _app_manager.ensure_web_alive(app_id)
    except Exception as e:
        logger.warning("[PREVIEW] ensure_web_alive errored for %s: %s", app_id[:8], e)

    target = f"http://127.0.0.1:{managed.web_port}/{path}"
    params = dict(request.query_params)
    if params:
        from urllib.parse import urlencode
        target += f"?{urlencode(params)}"

    # ── Server-side retry loop — absorbs restart dead window ──
    max_attempts = 15
    attempt_delay = 1.0

    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(max_attempts):
            try:
                resp = await client.get(target)

                if resp.status_code == 200:
                    # Dev server is ready — return response
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        headers={
                            k: v for k, v in resp.headers.items()
                            if k.lower() not in ("transfer-encoding", "connection", "content-encoding")
                        },
                    )
                elif resp.status_code in (500, 502, 503):
                    # Server is starting up / recompiling — retry
                    if attempt < max_attempts - 1:
                        logger.info(
                            "[PREVIEW] retry %d/%d app=%s status=%d path=%s",
                            attempt + 1, max_attempts, app_id[:8], resp.status_code, path[:60],
                        )
                        await _asyncio.sleep(attempt_delay)
                        continue
                    else:
                        # Exhausted retries — return error with no-cache headers
                        return Response(
                            content=b"<html><body><h3>App is taking longer than usual to start. Try refreshing in a few seconds.</h3></body></html>",
                            status_code=503,
                            media_type="text/html",
                            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
                        )
                else:
                    # 4xx etc — don't retry, return as-is
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        headers={
                            k: v for k, v in resp.headers.items()
                            if k.lower() not in ("transfer-encoding", "connection", "content-encoding")
                        },
                    )

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, OSError) as e:
                if attempt < max_attempts - 1:
                    logger.info(
                        "[PREVIEW] retry %d/%d app=%s error=%s path=%s",
                        attempt + 1, max_attempts, app_id[:8], type(e).__name__, path[:60],
                    )
                    await _asyncio.sleep(attempt_delay)
                    continue
                else:
                    return Response(
                        content=b"<html><body><h3>App is taking longer than usual to start. Try refreshing in a few seconds.</h3></body></html>",
                        status_code=502,
                        media_type="text/html",
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
                    )

    # Should never reach here, but just in case
    raise HTTPException(502, "Preview server unreachable")


@router.delete("/{app_id}")
async def delete_app(app_id: str) -> Dict[str, bool]:
    """Delete an app (stop servers, remove files, unregister skills, remove DB records + related jobs)."""
    # Get slug + app_dir before deleting so we can pass it to the manager
    app_slug = None
    app_dir = None
    async with async_session_maker() as db:
        app = await db.get(App, app_id)
        if app:
            app_slug = app.slug
            app_dir = app.app_dir

    if _app_manager:
        await _app_manager.delete_app(app_id, app_dir=app_dir)

    # Unregister from AppGatewaySkill (removes tools + domain actions)
    if _app_gateway and app_slug:
        try:
            _app_gateway.unregister_app(app_slug)
            logger.info(f"[DELETE] Unregistered app '{app_slug}' from gateway")
        except Exception as e:
            logger.warning(f"[DELETE] Failed to unregister app '{app_slug}': {e}")

    async with async_session_maker() as db:
        # Delete related build jobs
        from sqlalchemy import delete as sa_delete
        await db.execute(sa_delete(BuildJob).where(BuildJob.app_id == app_id))

        app = await db.get(App, app_id)
        if app:
            await db.delete(app)
        await db.commit()

    return {"ok": True}
