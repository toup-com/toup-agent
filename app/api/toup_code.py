"""Toup Code — experimental Claude Code IDE integration (2026-05-12).

Deliberately isolated from the existing app_builder / Toup build
pipeline so the feature can be tested end-to-end and either promoted
or ripped out without disturbing the rest of the product.

Endpoints under /code/* (registered with the platform's api_prefix):
  - GET    /code/status   — has a Claude Code OAuth token configured?
  - POST   /code/token    — paste a `claude setup-token` value
  - DELETE /code/token    — clear it
  - POST   /code/spawn    — start a coding session

The OAuth token authenticates against the user's Claude Pro/Max
subscription quota; Toup never bills for these calls. Stored on
agent_configs.claude_code_oauth_token alongside the other channel
tokens.

`spawn` returns a STUB event sequence today so the IDE UI can render
the activity feed and exercise the whole flow before the real
subprocess wiring lands. Real wiring (spawn `claude -p "<prompt>"
--output-format stream-json` in the tenant container with
CLAUDE_CODE_OAUTH_TOKEN in env, parse JSONL, stream over WS/SSE)
comes in a follow-up commit.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db.database import get_db
from app.db.models.agent import AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/code", tags=["toup-code"])


# ── Schemas ────────────────────────────────────────────────────


class TokenStatus(BaseModel):
    configured: bool
    # Last 6 chars only — enough for the user to recognize "yes that's
    # the token I pasted" without ever surfacing the secret.
    masked: str | None = None


class TokenIn(BaseModel):
    token: str = Field(..., min_length=20)


class SpawnRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    # Subdirectory under /workspace where Claude Code will operate.
    # Defaults to "default" so a brand-new user gets a working session
    # without picking a project name first.
    project: str | None = None


class SpawnEvent(BaseModel):
    """One JSONL event from Claude Code's stream-json output.

    Real shapes once wired: session_started, thinking, file_created,
    file_edited (with diff), tool_use (Bash, etc.), message, error,
    session_completed. The stub returns a canned sequence with the
    same shape so the UI doesn't need to change when we swap in real.
    """
    type: str
    payload: dict[str, Any] = {}
    at: str


class SpawnResponse(BaseModel):
    session_id: str
    events: list[SpawnEvent]
    summary: str


# ── Helpers ────────────────────────────────────────────────────


async def _get_or_create_config(user_id: str, db: AsyncSession) -> AgentConfig:
    """Mirror agent_setup._get_or_create_config — first-time users
    don't have an AgentConfig row yet, so create one on demand instead
    of 404'ing on every /code/status call."""
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )
    config = result.scalar_one_or_none()
    if config is None:
        config = AgentConfig(user_id=user_id)
        db.add(config)
        await db.commit()
        await db.refresh(config)
    return config


# ── Endpoints ──────────────────────────────────────────────────


@router.get("/status", response_model=TokenStatus)
async def get_token_status(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    config = await _get_or_create_config(str(current_user.id), db)
    token = config.claude_code_oauth_token
    if not token:
        return TokenStatus(configured=False)
    return TokenStatus(configured=True, masked=f"…{token[-6:]}")


@router.post("/token", response_model=TokenStatus)
async def save_token(
    body: TokenIn,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Store an OAuth token from `claude setup-token`.

    We don't validate against Anthropic here — there's no public
    introspection endpoint for these tokens, and the only honest
    failure mode is at session-spawn time (subprocess will exit
    non-zero with an auth error). Pydantic enforces a 20-char floor
    so accidental empty pastes fail fast.
    """
    token = body.token.strip()
    config = await _get_or_create_config(str(current_user.id), db)
    config.claude_code_oauth_token = token
    config.updated_at = datetime.utcnow()
    await db.commit()
    return TokenStatus(configured=True, masked=f"…{token[-6:]}")


@router.delete("/token", response_model=TokenStatus)
async def clear_token(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    config = await _get_or_create_config(str(current_user.id), db)
    config.claude_code_oauth_token = None
    config.updated_at = datetime.utcnow()
    await db.commit()
    return TokenStatus(configured=False)


@router.post("/spawn", response_model=SpawnResponse)
async def spawn_session(
    body: SpawnRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a Claude Code coding session.

    STUB for now: returns a fixed event sequence so the IDE can render
    the activity feed without the subprocess being wired. The real
    implementation will:
      1. Resolve the user's tenant container + workspace mount
      2. Spawn `claude -p "<prompt>" --output-format stream-json` with
         cwd=/workspace/<project>, CLAUDE_CODE_OAUTH_TOKEN in env
      3. Parse each JSONL line from stdout
      4. Stream events back over WebSocket (or SSE) as they arrive
    """
    config = await _get_or_create_config(str(current_user.id), db)
    if not config.claude_code_oauth_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Connect your Claude Code subscription first.",
        )
    session_id = uuid.uuid4().hex[:12]
    project = (body.project or "default").strip() or "default"
    now = datetime.utcnow().isoformat() + "Z"
    events = [
        SpawnEvent(type="session_started", payload={"project": project, "prompt": body.prompt}, at=now),
        SpawnEvent(type="thinking", payload={"text": "Planning a minimal implementation…"}, at=now),
        SpawnEvent(type="file_created", payload={"path": f"{project}/package.json", "size": 312}, at=now),
        SpawnEvent(type="file_created", payload={"path": f"{project}/index.html"}, at=now),
        SpawnEvent(type="tool_use", payload={"name": "Bash", "command": "npm install"}, at=now),
        SpawnEvent(type="file_edited", payload={"path": f"{project}/index.html", "diff": "+8 -0"}, at=now),
        SpawnEvent(
            type="session_completed",
            payload={"summary": "Stub session — real `claude` subprocess wiring is next."},
            at=now,
        ),
    ]
    return SpawnResponse(
        session_id=session_id,
        events=events,
        summary=(
            "Stub session returned. The UI flow is testable end-to-end; "
            "real Claude Code execution lands in the next push."
        ),
    )
