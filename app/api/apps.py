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

from fastapi import APIRouter, Depends, HTTPException
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
    error_message: Optional[str] = None
    paused_at: Optional[str] = None
    resume_after: Optional[str] = None
    layer: int = 1
    layer2_changes: Optional[List[Dict[str, Any]]] = None
    created_at: str
    completed_at: Optional[str] = None


# ── Helpers ─────────────────────────────────────────────────────────

def _get_user_id() -> str:
    """Get user_id from settings (agent mode — single user per VPS)."""
    from app.config import settings
    return settings.user_id


async def _app_to_response(app: App) -> AppResponse:
    """Convert App model to response, enriching with live status."""
    qr_url = None
    web_url = None
    if _app_manager and app.status == "running":
        qr_url = await _app_manager.get_qr_url(app.id)
        web_url = await _app_manager.get_web_url(app.id)

    platforms = app.platforms.split(",") if app.platforms else ["web", "ios"]

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
    )


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

    return JobResponse(
        id=job.id,
        app_id=job.app_id,
        job_type=getattr(job, 'job_type', None) or "auto_builder",
        title=job.title,
        prompt=job.prompt,
        status=job.status,
        steps=steps,
        model=job.model or "",
        total_tokens=job.total_tokens or 0,
        error_message=job.error_message,
        paused_at=job.paused_at.isoformat() if getattr(job, 'paused_at', None) else None,
        resume_after=job.resume_after.isoformat() if getattr(job, 'resume_after', None) else None,
        layer=getattr(job, 'layer', 1) or 1,
        layer2_changes=layer2_changes,
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
async def list_jobs() -> List[JobResponse]:
    """List all build jobs for the current user."""
    user_id = _get_user_id()
    async with async_session_maker() as db:
        result = await db.execute(
            select(BuildJob).where(BuildJob.user_id == user_id).order_by(BuildJob.created_at.desc())
        )
        jobs = result.scalars().all()
        return [_job_to_response(job) for job in jobs]


class CreateJobRequest(BaseModel):
    title: str
    description: str = ""


@router.post("/jobs/")
async def create_job(req: CreateJobRequest) -> JobResponse:
    """Create a new agent task job from the dashboard and execute it."""
    import uuid
    user_id = _get_user_id()
    job_id = str(uuid.uuid4())
    job = BuildJob(
        id=job_id,
        user_id=user_id,
        job_type="agent_task",
        title=req.title,
        prompt=req.description or req.title,
        status="running",
        steps_json="[]",
        model="",
        layer=0,
        created_at=datetime.utcnow(),
    )
    async with async_session_maker() as db:
        db.add(job)
        await db.commit()
        await db.refresh(job)

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
                    channel="web",
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
                async with async_session_maker() as db:
                    j = await db.get(BuildJob, job_id)
                    if j:
                        j.status = "failed"
                        j.error_message = str(e)[:500]
                        j.completed_at = datetime.utcnow()
                        await db.commit()
                if _ws_broadcast:
                    await _ws_broadcast(user_id, {
                        "type": "job_update",
                        "job_id": job_id,
                        "status": "failed",
                        "error_message": str(e)[:200],
                    })

        asyncio.create_task(_run_dashboard_task())

    return _job_to_response(job)


class UpdateJobStatusRequest(BaseModel):
    status: str  # queued, running, completed, failed


@router.patch("/jobs/{job_id}")
async def update_job_status(job_id: str, req: UpdateJobStatusRequest) -> JobResponse:
    """Update a job's status (e.g. drag between kanban columns)."""
    if req.status not in ("queued", "running", "completed", "failed"):
        raise HTTPException(status_code=400, detail="Invalid status")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.status = req.status
        if req.status == "completed" and not job.completed_at:
            job.completed_at = datetime.utcnow()
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


@router.delete("/{app_id}")
async def delete_app(app_id: str) -> Dict[str, bool]:
    """Delete an app (stop servers, remove files, unregister skills, remove DB records + related jobs)."""
    # Get slug before deleting so we can unregister from gateway
    app_slug = None
    async with async_session_maker() as db:
        app = await db.get(App, app_id)
        if app:
            app_slug = app.slug

    if _app_manager:
        await _app_manager.delete_app(app_id)

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
