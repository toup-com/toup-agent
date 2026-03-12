"""
Apps & Build Jobs API — REST endpoints for managing user-built apps.

Endpoints:
  GET    /apps/                    - List all apps
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

import json
import logging
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


def set_app_manager(app_manager):
    """Wire the AppManager instance (called from agent_main.py lifespan)."""
    global _app_manager
    _app_manager = app_manager


# ── Response schemas ────────────────────────────────────────────────

class AppResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    slug: str
    status: str
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
    title: str
    prompt: str
    status: str
    steps: List[Dict[str, Any]]
    model: str = ""
    total_tokens: int = 0
    error_message: Optional[str] = None
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

    return JobResponse(
        id=job.id,
        app_id=job.app_id,
        title=job.title,
        prompt=job.prompt,
        status=job.status,
        steps=steps,
        model=job.model or "",
        total_tokens=job.total_tokens or 0,
        error_message=job.error_message,
        created_at=job.created_at.isoformat() if job.created_at else "",
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


# ── App Endpoints ───────────────────────────────────────────────────

@router.get("/")
async def list_apps() -> List[AppResponse]:
    """List all apps for the current user."""
    user_id = _get_user_id()
    async with async_session_maker() as db:
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
    """Delete an app (stop servers, remove files, remove DB records + related jobs)."""
    if _app_manager:
        await _app_manager.delete_app(app_id)

    async with async_session_maker() as db:
        # Delete related build jobs
        from sqlalchemy import delete as sa_delete
        await db.execute(sa_delete(BuildJob).where(BuildJob.app_id == app_id))

        app = await db.get(App, app_id)
        if app:
            await db.delete(app)
        await db.commit()

    return {"ok": True}
