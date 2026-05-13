"""Toup Code — experimental dual-provider IDE integration.

2026-05-12: Claude Code (Anthropic) shipped.
2026-05-13: GPT Codex (OpenAI) added — first-time screen now shows two
cards with real provider logos and accepts either token. /code/spawn
routes to whichever CLI the user picked.

Endpoints under /code/*:
  - GET    /code/status                 — both providers' connection state
  - POST   /code/token                  — save a token { token, provider }
  - DELETE /code/token?provider=...     — clear a provider's token
  - POST   /code/spawn                  — start a session (SSE), provider-routed
  - GET    /code/file                   — read a file from the workspace

Tokens authenticate against the user's own subscription / API key;
Toup never bills for these calls. Stored on agent_configs.
{claude_code_oauth_token, openai_codex_token} (plaintext, same trust
model as the other channel tokens).

`spawn` shells out to either the official `claude` CLI or the official
`codex` CLI inside a per-user, per-project workspace under
WORKSPACE_ROOT/toup-code/<user>/<project>, streams stream-json output
line-by-line, translates each message into our compact event shape,
and emits SSE frames. When the requested CLI isn't installed in the
image (pre-deploy state) we emit one `error` event and close cleanly.
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
from typing import Any, AsyncIterator, Iterator, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db.database import get_db
from app.db.models.agent import AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/code", tags=["toup-code"])


Provider = Literal["claude", "codex"]
_PROVIDERS: tuple[Provider, ...] = ("claude", "codex")


# ── Schemas ────────────────────────────────────────────────────


class ProviderStatus(BaseModel):
    configured: bool
    # Last 6 chars only — enough for the user to recognize "yes that's
    # the token I pasted" without ever surfacing the secret.
    masked: str | None = None


class ProvidersStatus(BaseModel):
    claude: ProviderStatus
    codex: ProviderStatus


class TokenIn(BaseModel):
    token: str = Field(..., min_length=20)
    provider: Provider = "claude"


class SpawnRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    # Subdirectory under <workspace_root>/toup-code/<user_id> where the
    # session runs. Defaults to "default" so a brand-new user gets a
    # working session without picking a project name first.
    project: str | None = None
    provider: Provider = "claude"


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


# Tokens from `claude setup-token` look like "sk-ant-oat01-..." per
# Anthropic's docs. We DO NOT hard-reject other prefixes (Anthropic
# may rotate the format) — instead, we hint at save time and let the
# subprocess validation be the source of truth.
_OAUTH_PREFIX = "sk-ant-oat"
_API_KEY_PREFIX = "sk-ant-api"

# OAuth tokens are URL-safe base64 (alphanumeric + dash + underscore)
# plus the `sk-ant-oat01-` prefix. Used to verify the cleaned token
# didn't pick up odd characters from clipboard / smart quotes / etc.
_TOKEN_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# OpenAI API keys (the modern format used by the `codex` CLI's
# OPENAI_API_KEY env var) are URL-safe base64 plus `sk-` or `sk-proj-`
# prefixes. Looser than the Anthropic regex above because the leading
# segment can contain dots in service-account keys.
_OPENAI_TOKEN_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _provider_column(provider: Provider) -> str:
    return {
        "claude": "claude_code_oauth_token",
        "codex": "openai_codex_token",
    }[provider]


def _provider_binary_name(provider: Provider) -> str:
    return {"claude": "claude", "codex": "codex"}[provider]


def _provider_env_var(provider: Provider) -> str:
    """Env var the CLI uses to pick up the user's credential."""
    return {
        "claude": "CLAUDE_CODE_OAUTH_TOKEN",
        "codex": "OPENAI_API_KEY",
    }[provider]


def _read_provider_token(config: AgentConfig, provider: Provider) -> str | None:
    return getattr(config, _provider_column(provider), None)


def _write_provider_token(config: AgentConfig, provider: Provider, token: str | None) -> None:
    setattr(config, _provider_column(provider), token)


def _clean_pasted_token(raw: str) -> str:
    """Strip ALL whitespace from a pasted token, not just the ends.

    The /code/spawn 401 cascade traced to this: `claude setup-token`
    prints a ~120-char OAuth token, terminals wrap it across two
    visual lines, and copy-from-terminal embeds a literal `\\n` in
    the middle of the string. `.strip()` only touches the ends so the
    newline survived all the way to Anthropic's bearer auth → 401.

    `\\s+` in Python's re module catches \\n, \\r, \\t, \\f, \\v, and
    Unicode whitespace (non-breaking space, etc.) — anything a paste
    might smuggle in. Tokens themselves are URL-safe base64 + the
    `sk-ant-oat01-` prefix; no legitimate whitespace can ever appear
    inside one. Also strips wrapping quotes for the common shell-paste
    case where someone copied `"sk-ant-oat01-..."` from a docs block."""
    s = re.sub(r"\s+", "", raw or "")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("\"", "'"):
        s = s[1:-1]
    return s


async def _validate_claude_token(token: str, claude_bin: str) -> tuple[bool, str]:
    """Run a no-op `claude -p` invocation to confirm Anthropic accepts
    the OAuth token. Returns (valid, reason_or_message).

    Cost: a few cents of subscription quota per call. Latency: 2-6 s.
    Acceptable for a one-time Connect action; we don't validate on
    every /code/status check.
    """
    env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": token}
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)

    # Use a throwaway tmpdir as cwd so this validation can never touch
    # the user's workspace (even though `claude -p "ok"` shouldn't write
    # anything, belt + braces).
    import tempfile
    with tempfile.TemporaryDirectory(prefix="toup-code-validate-") as tmpdir:
        try:
            proc = await asyncio.create_subprocess_exec(
                claude_bin,
                "-p", "Reply with the single word: ok",
                "--output-format", "stream-json",
                "--verbose",
                "--permission-mode", "acceptEdits",
                cwd=tmpdir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                return False, "Validation timed out after 30s — Anthropic may be slow, please retry."
        except Exception as e:  # pragma: no cover
            return False, f"Failed to invoke claude: {e!r}"

    text = stdout.decode("utf-8", "replace")
    err_text = stderr.decode("utf-8", "replace")

    # Walk the JSONL output looking for the final `result` event. The
    # is_error flag + result message tell us definitively whether
    # Anthropic accepted the token.
    last_result: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "result":
            last_result = msg

    if last_result is None:
        snippet = (err_text or text)[:400].strip()
        return False, f"claude produced no result. stderr: {snippet}" if snippet else "claude produced no result."

    if last_result.get("is_error"):
        result_text = str(last_result.get("result") or "")
        lower = result_text.lower()
        if "401" in result_text or "invalid bearer token" in lower or "authentication" in lower:
            return False, (
                "Anthropic rejected this token (401). Make sure you're pasting "
                "the output of `claude setup-token` (starts with `sk-ant-oat01-`), "
                "not an API key (`sk-ant-api…`). Re-run `claude setup-token` if "
                "the token may have expired."
            )
        return False, f"claude returned an error: {result_text[:400]}"

    return True, "ok"


async def _validate_codex_token(token: str) -> tuple[bool, str]:
    """Ping OpenAI's /v1/models with the pasted API key. Cheaper + faster
    than spawning the codex CLI to validate, and avoids burning a real
    model call on each save. A 200 means OpenAI accepts the key for at
    least one model; the codex CLI uses the same auth path.

    We deliberately don't shell out to `codex exec` for validation —
    the codex CLI on first run interactively offers ChatGPT-login and
    can hang waiting for stdin. /v1/models with the user's key is a
    direct test of the credential and nothing else.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.TimeoutException:
        return False, "Validation timed out after 12s — OpenAI may be slow, please retry."
    except Exception as e:  # pragma: no cover — network jitter
        return False, f"Failed to reach OpenAI: {e!r}"

    if resp.status_code == 200:
        return True, "ok"
    if resp.status_code == 401:
        return False, (
            "OpenAI rejected this key (401). Paste the secret key from "
            "platform.openai.com/api-keys (starts with `sk-` or `sk-proj-…`). "
            "Project keys must have access to at least one model."
        )
    if resp.status_code == 403:
        return False, (
            "OpenAI rejected this key (403). Your project may not have "
            "model access enabled — visit platform.openai.com/api-keys and "
            "confirm the key is active for at least one model."
        )
    if resp.status_code == 429:
        return False, "OpenAI rate-limited the validation request. Retry in a moment."
    detail = resp.text[:300].strip() or f"HTTP {resp.status_code}"
    return False, f"OpenAI returned an unexpected error: {detail}"


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


def _translate_codex_message(msg: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Map one `codex exec --json` event to our wire shape.

    The OpenAI Codex CLI emits Submission/Event envelopes shaped like
        {"id": "...", "msg": {"type": "<event>", ...}}

    Known event types we handle (others are silently dropped):
      - task_started:           {"model","provider"}   → system_init
      - agent_message:          {"message": "..."}      → thinking
      - agent_reasoning:        {"text": "..."}         → thinking
      - exec_command_begin:     {"command":[...]}       → tool_use Bash
      - exec_command_end:       {"exit_code": int,...}  → tool_error (if !=0)
      - patch_apply_begin:      {"changes": {path: {add|update|delete}}}
                                                        → file_created / file_edited
      - error:                  {"message": "..."}      → error / auth_failed
      - task_complete:          {"last_agent_message"}  → result

    Lossy by design — we keep what the IDE renders and drop the rest.
    """
    payload_root = msg.get("msg") if isinstance(msg.get("msg"), dict) else msg
    mtype = payload_root.get("type") if isinstance(payload_root, dict) else None
    if not mtype or not isinstance(payload_root, dict):
        return

    if mtype == "task_started":
        yield "system_init", {
            "session_id": msg.get("id") or payload_root.get("session_id"),
            "model": payload_root.get("model"),
            "tools": [],
        }
        return

    if mtype in ("agent_message", "agent_reasoning"):
        text = str(payload_root.get("message") or payload_root.get("text") or "").strip()
        if text:
            yield "thinking", {"text": text}
        return

    if mtype == "exec_command_begin":
        cmd = payload_root.get("command")
        if isinstance(cmd, list):
            cmd_str = " ".join(str(c) for c in cmd)
        else:
            cmd_str = str(cmd or "")
        yield "tool_use", {
            "name": "Bash",
            "command": cmd_str[:500],
            "tool_use_id": payload_root.get("call_id") or msg.get("id"),
        }
        return

    if mtype == "exec_command_end":
        # Only surface failures — successes already showed as tool_use.
        exit_code = payload_root.get("exit_code")
        if exit_code not in (0, None):
            err = (payload_root.get("stderr") or payload_root.get("stdout") or "")
            yield "tool_error", {
                "tool_use_id": payload_root.get("call_id") or msg.get("id"),
                "content": f"exit {exit_code}: {str(err)[:600]}",
            }
        return

    if mtype == "patch_apply_begin":
        changes = payload_root.get("changes") or {}
        if isinstance(changes, dict):
            for path, kind_obj in changes.items():
                kind: str | None = None
                if isinstance(kind_obj, dict):
                    if "add" in kind_obj:
                        kind = "add"
                    elif "delete" in kind_obj:
                        kind = "delete"
                    elif "update" in kind_obj or "rename" in kind_obj:
                        kind = "update"
                elif isinstance(kind_obj, str):
                    kind = kind_obj
                if kind == "add":
                    yield "file_created", {"path": str(path), "tool_use_id": msg.get("id")}
                elif kind in ("update", "rename"):
                    yield "file_edited", {"path": str(path), "tool_use_id": msg.get("id")}
                # `delete` has no UI event today — skip silently
        return

    if mtype == "error":
        emsg = str(payload_root.get("message") or "")
        lower = emsg.lower()
        if "401" in emsg or "invalid api key" in lower or ("auth" in lower and "fail" in lower):
            yield "auth_failed", {
                "message": (
                    "OpenAI rejected your Codex API key. Click Disconnect, "
                    "then paste a fresh key from platform.openai.com/api-keys."
                ),
                "raw": emsg[:500],
            }
        else:
            yield "error", {"message": emsg[:1500] or "codex reported an error"}
        return

    if mtype == "task_complete":
        result_text = str(payload_root.get("last_agent_message") or "")
        yield "result", {
            "subtype": "success",
            "is_error": False,
            "result": result_text,
            "duration_ms": payload_root.get("duration_ms"),
            "num_turns": payload_root.get("num_turns"),
            "total_cost_usd": payload_root.get("total_cost_usd"),
            "session_id": msg.get("id"),
        }
        return


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
        result_text = str(msg.get("result") or "")
        is_error = bool(msg.get("is_error"))
        # Detect Anthropic auth failure surfacing as a result. Emit a
        # dedicated event so the UI can show a "reconnect your token"
        # banner instead of treating the 401 string as a normal reply.
        if is_error:
            lower = result_text.lower()
            if "401" in result_text or "invalid bearer token" in lower or (
                "authentication" in lower and "fail" in lower
            ):
                yield "auth_failed", {
                    "message": (
                        "Anthropic rejected your Claude Code token. Click "
                        "Disconnect, then re-run `claude setup-token` on your "
                        "machine and paste the fresh output (starts with "
                        "`sk-ant-oat01-…`)."
                    ),
                    "raw": result_text[:500],
                }
                return
        yield "result", {
            "subtype": msg.get("subtype"),
            "is_error": is_error,
            "result": result_text,
            "duration_ms": msg.get("duration_ms"),
            "num_turns": msg.get("num_turns"),
            "total_cost_usd": msg.get("total_cost_usd"),
            "session_id": msg.get("session_id"),
        }
        return


# ── Endpoints ──────────────────────────────────────────────────


def _provider_status_from_token(token: str | None) -> ProviderStatus:
    if not token:
        return ProviderStatus(configured=False)
    return ProviderStatus(configured=True, masked=f"…{token[-6:]}")


@router.get("/status", response_model=ProvidersStatus)
async def get_status(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return connection state for every supported provider in one call.

    The frontend's first-time screen renders both cards regardless of
    state, so collapsing this to a single round-trip keeps the page
    snappy and avoids racing two status fetches against each other."""
    config = await _get_or_create_config(str(current_user.id), db)
    return ProvidersStatus(
        claude=_provider_status_from_token(config.claude_code_oauth_token),
        codex=_provider_status_from_token(config.openai_codex_token),
    )


def _validate_claude_token_shape(token: str) -> None:
    """Raise HTTPException(400) on obvious shape mismatches for Claude
    Code tokens — runs BEFORE the slow subprocess validation so the user
    gets immediate feedback when they pasted the wrong kind of secret."""
    if token.startswith(_API_KEY_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "That looks like an Anthropic API key (starts with "
                "`sk-ant-api…`). This feature uses your Pro/Max subscription "
                "via an OAuth token. Run `claude setup-token` on your machine "
                "and paste its output (starts with `sk-ant-oat01-…`)."
            ),
        )
    if not _TOKEN_ALLOWED_RE.match(token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Token contains characters that don't belong to an OAuth "
                "token (allowed: letters, digits, `-`, `_`). Re-copy directly "
                "from your terminal output — avoid editors that auto-correct "
                "quotes or dashes."
            ),
        )


def _validate_codex_token_shape(token: str) -> None:
    """OpenAI keys are `sk-…` (legacy) or `sk-proj-…` (project keys).
    Reject anything that doesn't match the cipherset (URL-safe base64
    + dot for service-account keys) early."""
    if token.startswith("sk-ant-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "That looks like an Anthropic secret. The Codex card "
                "needs an OpenAI API key — generate one at "
                "platform.openai.com/api-keys (starts with `sk-` or `sk-proj-`)."
            ),
        )
    if not _OPENAI_TOKEN_ALLOWED_RE.match(token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Token contains characters that don't belong to an OpenAI "
                "API key (allowed: letters, digits, `-`, `_`, `.`). Re-copy "
                "directly from the OpenAI dashboard."
            ),
        )
    if not (token.startswith("sk-") or token.startswith("rk_")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "OpenAI API keys start with `sk-` or `sk-proj-`. Generate one "
                "at platform.openai.com/api-keys and paste it here."
            ),
        )


@router.post("/token", response_model=ProviderStatus)
async def save_token(
    body: TokenIn,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Store a credential for the named provider.

    For `claude` we run a no-op `claude -p` subprocess against
    Anthropic's API to confirm the token is accepted. For `codex` we
    ping `GET https://api.openai.com/v1/models` — cheaper than spawning
    the CLI and avoids the interactive-login dance the codex CLI does
    on its first run.

    Both validations are best-effort: when the CLI is missing (pre-deploy
    window) we still save and let /code/spawn surface the auth error
    with the same actionable copy."""
    token = _clean_pasted_token(body.token)
    if len(token) < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "That doesn't look like a token after whitespace was removed. "
                "Make sure you pasted the entire value."
            ),
        )

    if body.provider == "claude":
        _validate_claude_token_shape(token)
        claude_bin = shutil.which("claude")
        if claude_bin:
            valid, reason = await _validate_claude_token(token, claude_bin)
            if not valid:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)
        else:
            logger.warning(
                "toup_code: claude CLI not installed; skipping save-time validation"
            )
    elif body.provider == "codex":
        _validate_codex_token_shape(token)
        valid, reason = await _validate_codex_token(token)
        if not valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)
    else:  # pragma: no cover — Literal type guards this
        raise HTTPException(status_code=400, detail=f"Unknown provider: {body.provider!r}")

    config = await _get_or_create_config(str(current_user.id), db)
    _write_provider_token(config, body.provider, token)
    config.updated_at = datetime.utcnow()
    await db.commit()
    return ProviderStatus(configured=True, masked=f"…{token[-6:]}")


@router.delete("/token", response_model=ProviderStatus)
async def clear_token(
    provider: Provider = Query("claude"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    config = await _get_or_create_config(str(current_user.id), db)
    _write_provider_token(config, provider, None)
    config.updated_at = datetime.utcnow()
    await db.commit()
    return ProviderStatus(configured=False)


@router.get("/file")
async def read_file(
    path: str,
    current_user=Depends(get_current_user),
):
    """Read a file from the calling user's Toup Code workspace.

    `path` is interpreted relative to <workspace_root>/<user_id>, the
    same shape that file_created/file_edited events stream over SSE.
    Path traversal is defeated by resolving + asserting the target
    remains under the user's workspace root (handles `..`, symlinks,
    and absolute paths uniformly).

    Returns the content as UTF-8 text when decodable; otherwise base64
    so the UI can show "binary file (N bytes)" without crashing the
    JSON pipeline. Hard size cap at 5 MB — the IDE preview pane isn't
    a place for serving large blobs."""
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    if path.startswith("/") or any(seg == ".." for seg in path.split("/")):
        # Cheap pre-check; the resolve+relative_to below is the real
        # guard, but failing fast here returns a friendlier message.
        raise HTTPException(status_code=400, detail="Invalid path")

    user_id = _safe_segment(str(current_user.id), "user")
    user_root = (_workspace_root() / user_id).resolve()
    try:
        target = (user_root / path).resolve()
        target.relative_to(user_root)  # raises ValueError if it escapes
    except (ValueError, OSError):
        raise HTTPException(status_code=403, detail="Path escapes workspace")

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Not a regular file")

    size = target.stat().st_size
    if size > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (5 MB limit)")

    try:
        return {
            "path": path,
            "content": target.read_text(encoding="utf-8"),
            "size": size,
            "binary": False,
        }
    except UnicodeDecodeError:
        import base64
        return {
            "path": path,
            "content": base64.b64encode(target.read_bytes()).decode("ascii"),
            "size": size,
            "binary": True,
        }


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
    provider = body.provider
    config = await _get_or_create_config(str(current_user.id), db)
    raw_token = _read_provider_token(config, provider)
    if not raw_token:
        pretty = {"claude": "Claude Code", "codex": "GPT Codex"}[provider]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Connect your {pretty} credential first.",
        )

    # Defense-in-depth: tokens saved before the whitespace fix landed
    # may still have embedded newlines from terminal copy/paste. Clean
    # on read so the existing connector starts working without forcing
    # a re-paste, AND write the cleaned value back so /status etc.
    # don't keep showing a stale masked tail.
    token = _clean_pasted_token(raw_token)
    if token != raw_token:
        _write_provider_token(config, provider, token)
        config.updated_at = datetime.utcnow()
        await db.commit()
        logger.info(
            "toup_code: sanitized stored %s token for user=%s (stripped %d whitespace chars)",
            provider, current_user.id, len(raw_token) - len(token),
        )

    user_id = _safe_segment(str(current_user.id), "user")
    project = _safe_segment(body.project or "default", "default")
    user_workspace_root = (_workspace_root() / user_id).resolve()
    workspace = (user_workspace_root / project).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    prompt_text = body.prompt

    bin_name = _provider_binary_name(provider)
    bin_path = shutil.which(bin_name)

    async def stream() -> AsyncIterator[str]:
        # Open with a synthetic session_started so the UI has something
        # to render immediately while the subprocess boots.
        yield _sse("session_started", {
            "session_id": session_id,
            "project": project,
            "prompt": prompt_text,
            "provider": provider,
        })

        if bin_path is None:
            install_hint = {
                "claude": "npm install -g @anthropic-ai/claude-code",
                "codex": "npm install -g @openai/codex",
            }[provider]
            yield _sse("error", {
                "message": (
                    f"The `{bin_name}` CLI is not installed in this environment yet. "
                    f"Deploy with the updated Dockerfile or run `{install_hint}`."
                ),
            })
            yield "data: [DONE]\n\n"
            return

        # Build env — provider token IN, the *other* provider's vars OUT
        # so a stray platform-side credential can't take precedence and
        # bill the wrong account.
        env = {**os.environ, _provider_env_var(provider): token}
        if provider == "claude":
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
        else:  # codex
            env.pop("OPENAI_API_KEY", None)  # we just set it; ensure no shadowing from os.environ leftovers
            env[_provider_env_var(provider)] = token

        # Per-provider argv.
        if provider == "claude":
            argv = [
                bin_path,
                "-p", prompt_text,
                "--model", "claude-opus-4-7",
                # Sonnet 4.6 covers turns when Opus 4.7 is rate-limited or
                # overloaded so the session degrades instead of failing
                # outright. --fallback-model only works with --print.
                "--fallback-model", "claude-sonnet-4-6",
                "--output-format", "stream-json",
                "--verbose",
                "--permission-mode", "acceptEdits",
            ]
        else:  # codex
            argv = [
                bin_path,
                "exec",
                # `--json` makes the codex CLI emit one JSON event per
                # line on stdout, mirroring the shape we already parse
                # for Claude Code.
                "--json",
                # `--full-auto` skips the per-tool approval prompts that
                # would otherwise block a non-interactive session.
                "--full-auto",
                prompt_text,
            ]

        proc: asyncio.subprocess.Process | None = None
        stderr_buf: list[str] = []

        async def drain_stderr() -> None:
            assert proc and proc.stderr
            async for raw in proc.stderr:
                stderr_buf.append(raw.decode("utf-8", "replace"))

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workspace),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("toup_code spawn failed: provider=%s user=%s", provider, user_id)
            yield _sse("error", {"message": f"Failed to start {bin_name}: {e!r}"})
            yield "data: [DONE]\n\n"
            return

        stderr_task = asyncio.create_task(drain_stderr())
        last_result_seen = False
        translator = _translate_message if provider == "claude" else _translate_codex_message

        try:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                # Client-gone short-circuit — checked each line so we
                # don't keep paying for tokens after the tab closes.
                if await request.is_disconnected():
                    logger.info(
                        "toup_code: client disconnected, killing %s pid=%s",
                        bin_name, proc.pid,
                    )
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("toup_code: non-JSON line: %r", line[:200])
                    continue
                for evt_type, payload in translator(msg):
                    if evt_type == "result":
                        last_result_seen = True
                    # Rewrite absolute file paths to <project>/<rel>
                    # form so the UI can render readable names AND so
                    # the same string round-trips through GET /code/file
                    # without exposing the platform-internal workspace
                    # root. Paths outside the user's workspace are left
                    # alone — they may surface in tool errors etc. and
                    # the UI handles them as opaque.
                    if evt_type in ("file_created", "file_edited"):
                        raw_path = payload.get("path")
                        if isinstance(raw_path, str) and raw_path:
                            try:
                                resolved = (workspace / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
                                payload["path"] = str(resolved.relative_to(user_workspace_root))
                            except (ValueError, OSError):
                                pass
                    yield _sse(evt_type, payload)

            rc = await proc.wait()
            await stderr_task

            if rc != 0 and not last_result_seen:
                err = "".join(stderr_buf).strip() or f"{bin_name} exited with code {rc}"
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
            "provider": provider,
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Tells nginx/Caddy not to buffer the response — without this
            # the user only sees events arrive once the CLI exits.
            "X-Accel-Buffering": "no",
        },
    )
