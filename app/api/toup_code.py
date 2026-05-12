"""Toup Code — experimental Claude Code IDE integration (2026-05-12).

Deliberately isolated from the existing app_builder / Toup build
pipeline so the feature can be tested end-to-end and either promoted
or ripped out without disturbing the rest of the product.

Endpoints under /code/* (registered with the platform's api_prefix):
  - GET    /code/status   — has a Claude Code OAuth token configured?
  - POST   /code/token    — paste a `claude setup-token` value
  - DELETE /code/token    — clear it
  - POST   /code/spawn    — start a coding session (SSE stream)

The OAuth token authenticates against the user's Claude Pro/Max
subscription quota; Toup never bills for these calls. Stored on
agent_configs.claude_code_oauth_token alongside the other channel
tokens.

`spawn` shells out to the official `claude` CLI inside a per-user,
per-project workspace under WORKSPACE_ROOT/toup-code/<user>/<project>,
streams its stream-json output line-by-line, translates each message
into our compact event shape, and emits the result as Server-Sent
Events. The frontend consumes the stream via fetch() + ReadableStream
(EventSource can't POST). When `claude` isn't installed in the image
yet (local dev / pre-deploy state), we emit one `error` event and
close the stream cleanly so the UI doesn't hang.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
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
    # Subdirectory under <workspace_root>/toup-code/<user_id> where
    # Claude Code will operate. Defaults to "default" so a brand-new
    # user gets a working session without picking a project name first.
    project: str | None = None


# ── Helpers ────────────────────────────────────────────────────


async def _get_or_create_config(user_id: str, db: AsyncSession) -> AgentConfig:
    """Mirror agent_setup._get_or_create_config — first-time users don't
    have an AgentConfig row yet, so create one on demand instead of
    404'ing on every /code/status call."""
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


def _safe_segment(value: str, fallback: str = "default") -> str:
    """Strip a path segment to characters that can't escape a workspace
    dir. Conservative — alphanumeric, dash, underscore only."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", value).strip("_")
    return cleaned or fallback


def _workspace_root() -> Path:
    """Resolve where per-user Toup Code workspaces live.

    Honours WORKSPACE_DIR/AGENT_WORKSPACE_DIR (set in prod Docker /
    local dev) and falls back to /tmp/toup-code on a totally bare
    machine. Materializing the root here means a fresh deploy can
    write into it immediately without a separate init step."""
    candidates = [
        os.environ.get("TOUP_CODE_WORKSPACE_ROOT"),
        os.environ.get("WORKSPACE_DIR"),
        os.environ.get("AGENT_WORKSPACE_DIR"),
        "/app/workspace",
        "/tmp",
    ]
    base = next((c for c in candidates if c), "/tmp")
    root = Path(base) / "toup-code"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sse(event_type: str, payload: dict[str, Any]) -> str:
    """Format a single Server-Sent Event frame.

    Our wire shape: every frame's data is a JSON object with a `type`
    discriminator + `payload` + ISO8601 `at` timestamp. The frontend's
    streamSpawn() parses this directly into the same shape it stored
    in stubbed `events` arrays so existing renderers don't change."""
    data = {
        "type": event_type,
        "payload": payload,
        "at": datetime.utcnow().isoformat() + "Z",
    }
    return f"data: {json.dumps(data, default=str)}\n\n"


def _translate_message(msg: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Map one `claude --output-format stream-json` message to zero or
    more (event_type, payload) tuples in our wire shape.

    Reference for the shapes:
      - system init: {"type":"system","subtype":"init","session_id":...,"model":...,"tools":[...]}
      - assistant:   {"type":"assistant","message":{"content":[{type:"text",text:...} | {type:"tool_use",name,input,id}, ...]}}
      - user/tool result: {"type":"user","message":{"content":[{type:"tool_result","tool_use_id":...,"content":...,"is_error":...}]}}
      - result:      {"type":"result","subtype":"success"|"error_*","is_error":bool,"result":str,"duration_ms":int,"num_turns":int,"total_cost_usd":float}

    We're intentionally lossy: only the fields the UI renders survive.
    """
    mtype = msg.get("type")
    if mtype == "system" and msg.get("subtype") == "init":
        yield "system_init", {
            "session_id": msg.get("session_id"),
            "model": msg.get("model"),
            "tools": msg.get("tools", []),
        }
        return

    if mtype == "assistant":
        message = msg.get("message") or {}
        for block in message.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                text = (block.get("text") or "").strip()
                if text:
                    yield "thinking", {"text": text}
            elif btype == "tool_use":
                name = block.get("name") or ""
                inp = block.get("input") or {}
                tool_id = block.get("id")
                if name == "Write":
                    yield "file_created", {"path": inp.get("file_path", ""), "tool_use_id": tool_id}
                elif name in ("Edit", "MultiEdit"):
                    yield "file_edited", {"path": inp.get("file_path", ""), "tool_use_id": tool_id}
                elif name == "Bash":
                    cmd = str(inp.get("command", ""))[:500]
                    yield "tool_use", {"name": "Bash", "command": cmd, "tool_use_id": tool_id}
                elif name == "Read":
                    yield "tool_use", {"name": "Read", "path": inp.get("file_path", ""), "tool_use_id": tool_id}
                else:
                    yield "tool_use", {"name": name, "tool_use_id": tool_id}
        return

    if mtype == "user":
        message = msg.get("message") or {}
        for block in message.get("content") or []:
            if block.get("type") == "tool_result" and block.get("is_error"):
                # Tool results are otherwise too verbose to mirror live;
                # we only surface errors so the UI can flag them.
                raw = block.get("content")
                if isinstance(raw, list):
                    raw = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                yield "tool_error", {
                    "tool_use_id": block.get("tool_use_id"),
                    "content": str(raw or "")[:800],
                }
        return

    if mtype == "result":
        yield "result", {
            "subtype": msg.get("subtype"),
            "is_error": bool(msg.get("is_error")),
            "result": msg.get("result", ""),
            "duration_ms": msg.get("duration_ms"),
            "num_turns": msg.get("num_turns"),
            "total_cost_usd": msg.get("total_cost_usd"),
            "session_id": msg.get("session_id"),
        }
        return


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
    so accidental empty pastes fail fast."""
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


@router.post("/spawn")
async def spawn_session(
    body: SpawnRequest,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a Claude Code coding session, streaming JSONL events as SSE.

    Flow:
      1. Resolve OAuth token from agent_configs.
      2. Materialize per-user, per-project workspace dir.
      3. Spawn `claude -p "<prompt>" --output-format stream-json
         --verbose --permission-mode acceptEdits` with the token in env.
      4. Read stdout line-by-line, translate each JSONL message into
         our event shape, yield as SSE frames.
      5. On stream end, yield a final session_completed frame + [DONE]
         sentinel so the client can close the reader.

    Client disconnection cancels the AsyncIterator which triggers our
    finally block to kill the subprocess — we don't want an orphaned
    `claude` chewing through subscription quota after the tab closes.
    """
    config = await _get_or_create_config(str(current_user.id), db)
    token = config.claude_code_oauth_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Connect your Claude Code subscription first.",
        )

    user_id = _safe_segment(str(current_user.id), "user")
    project = _safe_segment(body.project or "default", "default")
    workspace = _workspace_root() / user_id / project
    workspace.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    prompt_text = body.prompt
    claude_bin = shutil.which("claude")

    async def stream() -> AsyncIterator[str]:
        # Open with a synthetic session_started so the UI has something
        # to render immediately while the subprocess boots.
        yield _sse("session_started", {
            "session_id": session_id,
            "project": project,
            "prompt": prompt_text,
        })

        if claude_bin is None:
            yield _sse("error", {
                "message": (
                    "The `claude` CLI is not installed in this environment yet. "
                    "Deploy with the updated Dockerfile (Node.js + "
                    "@anthropic-ai/claude-code) to enable real execution."
                ),
            })
            yield "data: [DONE]\n\n"
            return

        # Build env — token IN, ANTHROPIC_API_KEY OUT (an API key would
        # otherwise take precedence over the OAuth token and bill the
        # *platform* account instead of the user's subscription).
        env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": token}
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)

        proc: asyncio.subprocess.Process | None = None
        stderr_buf: list[str] = []

        async def drain_stderr() -> None:
            assert proc and proc.stderr
            async for raw in proc.stderr:
                stderr_buf.append(raw.decode("utf-8", "replace"))

        try:
            proc = await asyncio.create_subprocess_exec(
                claude_bin,
                "-p", prompt_text,
                "--output-format", "stream-json",
                "--verbose",
                "--permission-mode", "acceptEdits",
                cwd=str(workspace),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("claude spawn failed: user=%s", user_id)
            yield _sse("error", {"message": f"Failed to start claude: {e!r}"})
            yield "data: [DONE]\n\n"
            return

        stderr_task = asyncio.create_task(drain_stderr())
        last_result_seen = False

        try:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                # Client-gone short-circuit — checked each line so we
                # don't keep paying claude tokens after the tab closes.
                if await request.is_disconnected():
                    logger.info("toup_code: client disconnected, killing claude pid=%s", proc.pid)
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("toup_code: non-JSON line: %r", line[:200])
                    continue
                for evt_type, payload in _translate_message(msg):
                    if evt_type == "result":
                        last_result_seen = True
                    yield _sse(evt_type, payload)

            rc = await proc.wait()
            await stderr_task

            if rc != 0 and not last_result_seen:
                err = "".join(stderr_buf).strip() or f"claude exited with code {rc}"
                yield _sse("error", {"message": err[:1500]})
        finally:
            if proc and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
            if not stderr_task.done():
                stderr_task.cancel()
                with contextlib.suppress(Exception):
                    await stderr_task

        yield _sse("session_completed", {
            "session_id": session_id,
            "exit_code": proc.returncode if proc else None,
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Tells nginx/Caddy not to buffer the response — without this
            # the user only sees events arrive once claude exits.
            "X-Accel-Buffering": "no",
        },
    )
