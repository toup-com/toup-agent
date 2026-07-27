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
import dataclasses
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
from app.config import settings
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


class SuperviseRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=8000)
    conv_id: str | None = None
    provider: Provider = "claude"
    project: str | None = None


class InterjectRequest(BaseModel):
    conv_id: str
    message: str = Field(..., min_length=1, max_length=4000)


class ConvIdRequest(BaseModel):
    conv_id: str


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


def _sse(event_type: str, payload: dict[str, Any], role: str | None = None) -> str:
    """Format a single Server-Sent Event frame.

    Wire shape: every frame's data is a JSON object with a `type`
    discriminator + `payload` + ISO8601 `at` timestamp. The supervisor
    flow adds a `role` field ("user" | "agent" | "cli") so the
    frontend can render the same event stream as three distinct chat
    entities. Direct /code/spawn (no supervision) omits role, which
    the existing renderers handle as a default "cli" surface."""
    data: dict[str, Any] = {
        "type": event_type,
        "payload": payload,
        "at": datetime.utcnow().isoformat() + "Z",
    }
    if role is not None:
        data["role"] = role
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


# ── Supervisor flow ─────────────────────────────────────────────
#
# The supervisor wraps the executor CLI (Claude Code / Codex) in a
# chat-shaped orchestration loop driven by the USER'S own model
# (settings.agent_model). Wire shape: every event carries a `role`
# ("user" | "agent" | "cli") so the frontend can render three
# distinct entities in the same conversation.
#
# Architecture:
#   - SupervisorSession state is in-memory, keyed by conv_id.
#   - /code/supervise opens an SSE stream. The first call creates the
#     session; subsequent calls with the same conv_id continue it.
#   - /code/supervise/interject lets a connected client push a new
#     user message into the running loop's queue. The loop drains
#     this queue between turns.
#   - /code/supervise/stop cancels the session cleanly.
#
# Termination: the supervisor decides "done" on its own (no hard cap).
# Runaway protection comes from (a) the system prompt encouraging
# concise turns, (b) the user's Stop button, (c) the underlying
# CLI's own subscription quota.

SUPERVISOR_SYSTEM_PROMPT = (
    "You are this user's coding supervisor. The user has connected a coding "
    "executor CLI (Claude Code or GPT Codex) and you orchestrate it on their "
    "behalf. You DO NOT write code yourself — you craft precise prompts for "
    "the executor, observe what it built, and decide what to do next.\n"
    "\n"
    "On every turn you MUST respond with EXACTLY ONE JSON object on a single "
    "line and NOTHING else. No markdown fences, no prose before or after.\n"
    "\n"
    "Actions:\n"
    '  {"action": "send_to_cli", "prompt": "<≤200 word prompt for the executor>", "thinking": "<≤2 sentence rationale you want the user to see>"}\n'
    '  {"action": "done", "summary": "<1-2 sentence wrap-up for the user>"}\n'
    '  {"action": "ask_user", "question": "<a focused clarifying question>"}\n'
    "\n"
    "Rules:\n"
    " - Prefer `send_to_cli` until the user's request is satisfied. Keep "
    "prompts focused: one concrete goal per turn (create file X with hero, "
    "or fix the broken Y) — not multi-step plans.\n"
    " - After each executor turn you'll see a synthetic 'CLI completed' "
    "message with the files touched and the executor's reply. Use that to "
    "decide the next action.\n"
    " - `done` only when the user's request is genuinely complete. Don't "
    "ask the executor to 'verify' or 'check' as a final step — declare done.\n"
    " - `ask_user` only when you truly cannot proceed without clarification. "
    "Prefer making a sensible choice and noting it in `thinking`.\n"
    " - Output is parsed as JSON. ANY deviation from one JSON object on one "
    "line breaks the orchestration."
)


@dataclasses.dataclass
class SupervisorSession:
    conv_id: str
    user_id: str
    provider: Provider
    workspace: Path
    workspace_root: Path
    token: str
    messages: list[dict[str, Any]]
    pending_interjects: list[str] = dataclasses.field(default_factory=list)
    canceled: bool = False
    last_seen_at: datetime = dataclasses.field(default_factory=datetime.utcnow)


_SESSIONS: dict[str, SupervisorSession] = {}
_SESSION_MAX_IDLE_S = 60 * 60  # 1 hour — GC handle below


def _gc_supervisor_sessions() -> None:
    """Drop sessions idle longer than _SESSION_MAX_IDLE_S. Called
    opportunistically on each /supervise entry; not a scheduled task."""
    now = datetime.utcnow()
    stale = [
        cid for cid, s in _SESSIONS.items()
        if (now - s.last_seen_at).total_seconds() > _SESSION_MAX_IDLE_S
    ]
    for cid in stale:
        _SESSIONS.pop(cid, None)


def _parse_supervisor_action(raw: str) -> dict[str, Any] | None:
    """Extract the single JSON action from a supervisor reply.

    Tolerant of: leading/trailing whitespace, ```json fences, a stray
    sentence before the JSON. Returns None on irrecoverable malformed
    output so the caller can surface the raw text as agent_message."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Try to extract the first JSON-shaped substring.
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except json.JSONDecodeError:
            return None
    return None


async def _run_cli_for_supervisor(
    session: SupervisorSession,
    cli_prompt: str,
    queue: asyncio.Queue,
) -> tuple[list[str], list[str], str | None, bool]:
    """Spawn the provider's CLI with `cli_prompt`, stream its events
    through `queue` (tagged role="cli"), and return a summary tuple of
    (files_created, files_edited, result_text, auth_failed).

    Reuses the same per-provider CLI invocation as POST /code/spawn
    but pipes events into the supervisor's queue instead of writing
    SSE frames itself."""
    provider = session.provider
    bin_name = _provider_binary_name(provider)
    bin_path = shutil.which(bin_name)
    files_created: list[str] = []
    files_edited: list[str] = []
    result_text: str | None = None
    auth_failed = False

    if bin_path is None:
        install_hint = {
            "claude": "npm install -g @anthropic-ai/claude-code",
            "codex": "npm install -g @openai/codex",
        }[provider]
        await queue.put(_sse("error", {
            "message": (
                f"The `{bin_name}` CLI isn't installed in this environment. "
                f"Deploy with the updated Dockerfile or `{install_hint}`."
            ),
        }, role="cli"))
        return files_created, files_edited, None, False

    env = {**os.environ, _provider_env_var(provider): session.token}
    if provider == "claude":
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
    else:
        env[_provider_env_var(provider)] = session.token

    if provider == "claude":
        argv = [
            bin_path,
            "-p", cli_prompt,
            "--model", "claude-opus-4-7",
            "--fallback-model", "claude-sonnet-4-6",
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", "acceptEdits",
        ]
    else:
        argv = [bin_path, "exec", "--json", "--full-auto", cli_prompt]

    proc: asyncio.subprocess.Process | None = None
    stderr_buf: list[str] = []

    async def drain_stderr() -> None:
        assert proc and proc.stderr
        async for raw in proc.stderr:
            stderr_buf.append(raw.decode("utf-8", "replace"))

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(session.workspace),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:  # pragma: no cover
        await queue.put(_sse("error", {"message": f"Failed to start {bin_name}: {e!r}"}, role="cli"))
        return files_created, files_edited, None, False

    stderr_task = asyncio.create_task(drain_stderr())
    translator = _translate_message if provider == "claude" else _translate_codex_message

    try:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            if session.canceled:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            for evt_type, payload in translator(msg):
                if evt_type in ("file_created", "file_edited"):
                    raw_path = payload.get("path")
                    if isinstance(raw_path, str) and raw_path:
                        try:
                            resolved = (session.workspace / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
                            payload["path"] = str(resolved.relative_to(session.workspace_root))
                        except (ValueError, OSError):
                            pass
                    (files_created if evt_type == "file_created" else files_edited).append(str(payload.get("path") or ""))
                if evt_type == "result":
                    result_text = str(payload.get("result") or "")
                if evt_type == "auth_failed":
                    auth_failed = True
                await queue.put(_sse(evt_type, payload, role="cli"))
        await proc.wait()
        await stderr_task
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

    if proc.returncode not in (0, None) and result_text is None:
        err = "".join(stderr_buf).strip() or f"{bin_name} exited with code {proc.returncode}"
        await queue.put(_sse("error", {"message": err[:1500]}, role="cli"))

    return files_created, files_edited, result_text, auth_failed


async def _run_supervisor_loop(
    session: SupervisorSession,
    queue: asyncio.Queue,
) -> None:
    """Drive the supervisor LLM ↔ CLI dance. Posts events into `queue`.

    The loop continues until: (a) the supervisor declares `done`,
    (b) the supervisor asks the user a question, (c) the session is
    canceled (Stop), or (d) a hard error makes recovery impossible.

    Mid-loop user messages arrive via /supervise/interject which
    appends to `session.pending_interjects`. We drain that queue at
    the top of each iteration before consulting the supervisor."""
    from app.services.internal_llm import call_system_llm

    try:
        while True:
            if session.canceled:
                await queue.put(_sse("agent_message", {"text": "Stopped by user."}, role="agent"))
                break

            # Drain any interjects accumulated since the last turn.
            if session.pending_interjects:
                extra = "\n\n".join(session.pending_interjects)
                session.pending_interjects.clear()
                session.messages.append({
                    "role": "user",
                    "content": f"[NEW USER MESSAGE — incorporate into the next decision]\n{extra}",
                })

            await queue.put(_sse("agent_status", {"text": "thinking…"}, role="agent"))

            try:
                # TKT-LAT-015: optionally pin Haiku for the supervisor loop
                # behind a flag. Default OFF preserves original
                # "user OWNS this orchestration" semantics until product
                # signs off on the model-quality trade.
                _sup_model = (
                    "claude-haiku-4-5-20251001"
                    if settings.toup_code_supervisor_use_haiku
                    else None
                )
                raw = await call_system_llm(
                    user_id=session.user_id,
                    operation_type="user.toup_code.supervisor",
                    model=_sup_model,
                    max_tokens=800,
                    system=SUPERVISOR_SYSTEM_PROMPT,
                    messages=session.messages,
                    timeout=120,
                    # W1.0: session.messages only grows within a session —
                    # a session-stable cache key lets the conversation
                    # prefix self-cache on OpenAI (no-op on Anthropic).
                    prompt_cache_key=f"toupcode:{session.conv_id}",
                    safety_identifier=session.user_id,
                )
            except Exception as e:  # pragma: no cover
                await queue.put(_sse("error", {"message": f"Supervisor LLM error: {e!r}"}, role="agent"))
                break

            if not raw:
                await queue.put(_sse("error", {"message": "Supervisor returned empty response."}, role="agent"))
                break

            action = _parse_supervisor_action(raw)
            session.messages.append({"role": "assistant", "content": raw})

            if action is None:
                # The supervisor returned prose. Surface as a plain
                # message so the user can see what it said, then exit
                # the loop — repeated malformed output would burn
                # tokens with no progress.
                await queue.put(_sse("agent_message", {"text": raw[:2000]}, role="agent"))
                break

            kind = str(action.get("action") or "")
            thinking = str(action.get("thinking") or "").strip()
            if thinking:
                await queue.put(_sse("agent_thinking", {"text": thinking}, role="agent"))

            if kind == "done":
                summary = str(action.get("summary") or "Done.").strip() or "Done."
                await queue.put(_sse("agent_done", {"summary": summary}, role="agent"))
                break

            if kind == "ask_user":
                question = str(action.get("question") or "").strip() or "Could you clarify?"
                await queue.put(_sse("agent_question", {"question": question}, role="agent"))
                # The loop ends here — the next /supervise call with the
                # same conv_id (carrying the user's reply) resumes it.
                break

            if kind == "send_to_cli":
                cli_prompt = str(action.get("prompt") or "").strip()
                if not cli_prompt:
                    await queue.put(_sse("error", {"message": "Supervisor produced an empty CLI prompt."}, role="agent"))
                    break
                await queue.put(_sse("agent_dispatch", {"prompt": cli_prompt}, role="agent"))
                created, edited, cli_result, auth_failed = await _run_cli_for_supervisor(
                    session, cli_prompt, queue,
                )
                if auth_failed:
                    # The CLI rejected the user's token — no point in
                    # asking the supervisor to keep going.
                    break
                # Synthesize a feedback turn so the supervisor knows
                # what the CLI did. Keep it terse — full transcripts
                # blow up context fast on multi-iteration runs.
                feedback_parts: list[str] = ["CLI completed."]
                if created:
                    feedback_parts.append(f"Files created: {', '.join(sorted(set(created)))}")
                if edited:
                    feedback_parts.append(f"Files edited: {', '.join(sorted(set(edited)))}")
                if cli_result:
                    feedback_parts.append(f"CLI reply: {cli_result[:1200]}")
                session.messages.append({
                    "role": "user",
                    "content": "\n".join(feedback_parts),
                })
                continue

            # Unknown action — surface and stop.
            await queue.put(_sse("agent_message", {"text": f"Unknown action `{kind}`. Stopping."}, role="agent"))
            break
    finally:
        # Final marker — frontend uses this to flip session-state idle.
        await queue.put(_sse("session_completed", {"conv_id": session.conv_id}, role=None))
        await queue.put(None)  # sentinel — closes the SSE stream


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


# ── Save to user's container workspace ───────────────────────────────


class SaveToWorkspaceRequest(BaseModel):
    """User clicked "Save to Workspace" in Toup Code — bridge the
    platform-side project files into the user's per-tenant agent
    container as a real App.

    `project` defaults to the most recent supervisor session's conv_id
    when omitted (matching the SSE stream's `project_seg` fallback).
    `name` becomes `App.name`; the agent slugifies + dedupes.
    """
    project: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)


class SaveToWorkspaceResponse(BaseModel):
    app_id: str
    slug: str
    file_count: int
    web_url: Optional[str] = None


# Mirror agent-side caps so platform side fails fast.
_SAVE_MAX_FILES = 500
_SAVE_MAX_BYTES_PER_FILE = 5 * 1024 * 1024
_SAVE_MAX_TOTAL_BYTES = 50 * 1024 * 1024
_SAVE_SKIP_DIRS = {".git", "node_modules", ".next", "dist", "build", "__pycache__", ".venv"}
_SAVE_SKIP_FILE_EXTS = {".pyc", ".lock", ".log", ".DS_Store"}


def _collect_project_files(project_dir: Path) -> tuple[list[dict[str, str]], int]:
    """Walk `project_dir` and return (files, total_bytes).

    Files structure: `[{"path": rel, "content": utf-8 str}, …]`.

    Binaries (non-UTF-8 decodable) and obvious noise (`.git`,
    `node_modules`, lock files, build outputs) are skipped — Toup Code
    sessions are typically a handful of source files; we don't want to
    pollute the user's container with `node_modules` if a stray
    install happened mid-session.

    Raises HTTPException on:
      - >`_SAVE_MAX_FILES`: 413
      - any file >`_SAVE_MAX_BYTES_PER_FILE`: 413 with the offending path
      - cumulative >`_SAVE_MAX_TOTAL_BYTES`: 413
    """
    out: list[dict[str, str]] = []
    total = 0
    for root, dirs, files in os.walk(project_dir):
        # Prune skip dirs in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in _SAVE_SKIP_DIRS]
        for fname in files:
            if any(fname.endswith(ext) for ext in _SAVE_SKIP_FILE_EXTS):
                continue
            full = Path(root) / fname
            rel = str(full.relative_to(project_dir)).replace(os.sep, "/")
            try:
                size = full.stat().st_size
            except OSError:
                continue
            if size > _SAVE_MAX_BYTES_PER_FILE:
                raise HTTPException(
                    status_code=413,
                    detail=f"file too large: {rel} ({size} bytes); cap is {_SAVE_MAX_BYTES_PER_FILE}",
                )
            total += size
            if total > _SAVE_MAX_TOTAL_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"project exceeds {_SAVE_MAX_TOTAL_BYTES} bytes — split or trim before saving",
                )
            try:
                text = full.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Binary — silently skip. The IDE preview pane similarly
                # treats non-text as opaque blobs; we don't ship them.
                continue
            out.append({"path": rel, "content": text})
            if len(out) > _SAVE_MAX_FILES:
                raise HTTPException(
                    status_code=413,
                    detail=f"too many files (>{_SAVE_MAX_FILES})",
                )
    return out, total


async def _resolve_agent_target(user_id: str, db: AsyncSession) -> tuple[str, str]:
    """Look up `(agent_url, agent_api_key)` for this user. 404 when no
    active agent exists — same shape `routines_proxy` uses so the
    frontend's empty-state handling stays uniform across surfaces."""
    from app.db.models import AgentConfig
    from sqlalchemy import select as sa_select
    row = (await db.execute(
        sa_select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
            AgentConfig.user_id == user_id,
            AgentConfig.deploy_status == "active",
        )
    )).first()
    if row is None or not row.agent_url or not row.agent_api_key:
        raise HTTPException(
            status_code=404,
            detail="No active agent container — deploy one before saving to your workspace.",
        )
    return row.agent_url, row.agent_api_key


@router.post("/save-to-workspace", response_model=SaveToWorkspaceResponse)
async def save_to_workspace(
    body: SaveToWorkspaceRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SaveToWorkspaceResponse:
    """Materialise the user's current Toup Code project as an App in
    their per-tenant agent container.

    Why this exists: Toup Code's subprocess (Claude Code / Codex) runs
    on the platform host with its workspace at
    `<workspace_root>/toup-code/<user_id>/<project>/`. Those files are
    a *scratchpad* — they survive the SSE session but never make it
    to the user's actual deployable workspace. This endpoint copies
    them across the bridge so the user can edit, deploy, and ship
    them through the normal Toup Build flow.

    Flow:
      1. Resolve the project dir on the platform (defaults to the
         user's last conv_id when omitted).
      2. Walk + read files (skipping `.git`, `node_modules`, build
         output, binaries). Enforce per-file + total caps.
      3. Look up the user's `(agent_url, agent_api_key)`.
      4. POST the files to the agent's
         `/api/apps/import-from-toup-code` with `X-Agent-Key` auth.
      5. Forward the agent's response (`app_id`, `slug`, `app_dir`)
         to the frontend.

    Failure model:
      - 404 when no active agent: user needs to deploy first.
      - 404 when project dir empty / missing: typo or stale conv_id.
      - 413 on size caps.
      - 502 when the agent rejects or times out: surface the agent's
        status code + body so the operator can see the root cause.
    """
    user_id = _safe_segment(str(current_user.id), "user")
    project_seg = _safe_segment(
        body.project or _last_project_for_user(user_id), "default",
    )
    project_dir = (_workspace_root() / user_id / project_seg).resolve()

    try:
        project_dir.relative_to(_workspace_root().resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="project escapes workspace root")

    if not project_dir.exists() or not project_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"no Toup Code project at {project_seg!r} — start a session before saving",
        )

    files_payload, total_bytes = _collect_project_files(project_dir)
    if not files_payload:
        raise HTTPException(
            status_code=400,
            detail="project directory has no text files to save",
        )

    agent_url, agent_key = await _resolve_agent_target(str(current_user.id), db)

    import httpx  # local — matches the pattern other endpoints in this
                  # module use; avoids paying the import cost at boot.

    # Forward to agent. 30 s ceiling — should comfortably cover even
    # the worst-case 50 MB transfer over the bridge.
    body_payload = {
        "name": body.name.strip(),
        "description": (body.description or "").strip() or None,
        "files": files_payload,
        "source_metadata": {
            "project": project_seg,
            "conv_id": body.project or project_seg,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "file_count": len(files_payload),
            "total_bytes": total_bytes,
        },
    }
    url = f"{agent_url.rstrip('/')}/api/apps/import-from-toup-code"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json=body_payload,
                headers={
                    "X-Agent-Key": agent_key,
                    "Content-Type": "application/json",
                },
            )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"agent unreachable: {type(e).__name__}: {e}",
        ) from e

    if resp.status_code >= 300:
        # Surface the agent's actual error so the operator can fix it —
        # not a generic 5xx that buries the real failure.
        raise HTTPException(
            status_code=502,
            detail=f"agent rejected import (status={resp.status_code}): {resp.text[:400]}",
        )

    data = resp.json()
    return SaveToWorkspaceResponse(
        app_id=data["app_id"],
        slug=data["slug"],
        file_count=int(data.get("file_count") or len(files_payload)),
        web_url=data.get("web_url"),
    )


def _last_project_for_user(user_id_seg: str) -> str:
    """Find the most recently modified project dir under the user's
    workspace root. Used as the fallback when the request omits
    `project` — covers the common "I want to save what I just made"
    case without forcing the frontend to track conv_ids manually.

    Returns an empty string if no project dir exists; the caller then
    404s with a clean error."""
    root = (_workspace_root() / user_id_seg)
    if not root.exists() or not root.is_dir():
        return ""
    try:
        candidates = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return ""
    if not candidates:
        return ""
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


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


@router.post("/supervise")
async def supervise_session(
    body: SuperviseRequest,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Open an SSE stream that drives the supervisor loop.

    First call (no conv_id): creates a new session and starts the
    loop. Subsequent calls with the same conv_id: appends the new
    user message to the existing session's messages and resumes.

    The stream stays open until the supervisor says `done`, asks the
    user a question, or the client disconnects. /supervise/stop
    cancels in-flight.
    """
    _gc_supervisor_sessions()

    config = await _get_or_create_config(str(current_user.id), db)
    raw_token = _read_provider_token(config, body.provider)
    if not raw_token:
        pretty = {"claude": "Claude Code", "codex": "GPT Codex"}[body.provider]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Connect your {pretty} credential first.",
        )
    token = _clean_pasted_token(raw_token)
    if token != raw_token:
        _write_provider_token(config, body.provider, token)
        config.updated_at = datetime.utcnow()
        await db.commit()

    # Look up or create the session.
    session = _SESSIONS.get(body.conv_id) if body.conv_id else None
    user_id_str = str(current_user.id)

    if session is None or session.user_id != user_id_str:
        # New session (also branch taken if a stale conv_id from a
        # different user is supplied — never resume cross-tenant).
        conv_id = body.conv_id or uuid.uuid4().hex[:16]
        user_seg = _safe_segment(user_id_str, "user")
        project_seg = _safe_segment(body.project or conv_id, conv_id)
        user_workspace_root = (_workspace_root() / user_seg).resolve()
        workspace = (user_workspace_root / project_seg).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        session = SupervisorSession(
            conv_id=conv_id,
            user_id=user_id_str,
            provider=body.provider,
            workspace=workspace,
            workspace_root=user_workspace_root,
            token=token,
            messages=[{"role": "user", "content": body.user_message}],
        )
        _SESSIONS[conv_id] = session
    else:
        # Continuation — refresh state.
        session.canceled = False
        session.token = token
        session.provider = body.provider
        session.last_seen_at = datetime.utcnow()
        session.messages.append({"role": "user", "content": body.user_message})

    queue: asyncio.Queue = asyncio.Queue()

    async def stream() -> AsyncIterator[str]:
        # Kick off the session_started immediately so the frontend can
        # show "running" before the supervisor LLM responds.
        yield _sse("session_started", {
            "conv_id": session.conv_id,
            "provider": session.provider,
            "user_message": body.user_message,
        })
        # Loop runs in the background; we drain `queue` here.
        loop_task = asyncio.create_task(_run_supervisor_loop(session, queue))
        try:
            while True:
                # Cancel the loop if the client tab closes — otherwise
                # the supervisor would keep burning the user's tokens
                # with no one to read the reply.
                if await request.is_disconnected():
                    session.canceled = True
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if item is None:  # sentinel
                    break
                yield item
        finally:
            session.canceled = True
            if not loop_task.done():
                # Give the loop a chance to emit a final session_completed
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(loop_task, timeout=2.0)
                if not loop_task.done():
                    loop_task.cancel()
                    with contextlib.suppress(Exception):
                        await loop_task
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/supervise/interject")
async def supervise_interject(
    body: InterjectRequest,
    current_user=Depends(get_current_user),
):
    """Drop a mid-loop user message into the supervisor's queue.

    The next supervisor turn will see it as a synthetic
    `[NEW USER MESSAGE]` user-role turn and decide how to incorporate
    it. Avoids the user having to Stop+restart to course-correct."""
    session = _SESSIONS.get(body.conv_id)
    if session is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if session.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="not your conversation")
    session.pending_interjects.append(body.message)
    session.last_seen_at = datetime.utcnow()
    return {"ok": True, "queued": len(session.pending_interjects)}


@router.post("/supervise/stop")
async def supervise_stop(
    body: ConvIdRequest,
    current_user=Depends(get_current_user),
):
    """Cancel an in-flight supervisor session. Idempotent — calling
    on a session that already ended is a no-op."""
    session = _SESSIONS.get(body.conv_id)
    if session is None:
        return {"ok": True, "missing": True}
    if session.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="not your conversation")
    session.canceled = True
    return {"ok": True}
