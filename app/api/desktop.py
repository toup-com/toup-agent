"""Toup for Mac — device pairing, the relay socket, and local-action consent.

Implements `desktop/docs/RELAY_PROTOCOL.md`. Section references below (§1.1,
§5.3, …) are to that document; the defect each divergence from the Chrome
extension avoids is catalogued in its §9 and in
`desktop/docs/discovery/agent-tool-relay.md`.

WHICH PROCESS OWNS WHICH ROUTE
══════════════════════════════════════════════════════════════════════════
This router is included from BOTH entrypoints (`platform_main.py`,
`agent_main.py`) exactly as `extension.py` is — but unlike `extension.py`,
the split is explicit and load-bearing rather than "mounting the whole
router on the agent is a no-op for those".

PLATFORM (toup.ai, `numReplicas: 2`) owns everything that touches a table:

    POST /desktop/pair/init                  the device, unauthenticated
    POST /desktop/pair/poll                  the device, by device_code
    GET  /desktop/pair/lookup                the approval page
    POST /desktop/pair/approve   /deny       the approval page
    GET  /desktop/devices                    Settings
    POST /desktop/devices/revoke             Settings, or the Mac itself
    GET  /desktop/pending-actions            the confirm card list
    POST /desktop/pending-actions/{id}/approve  |  /reject
    POST /desktop/heartbeat                  ← agent, X-Agent-Key
    POST /desktop/internal/verify-token      ← agent, X-Agent-Key
    POST /desktop/internal/stage-action      ← agent, X-Agent-Key

    POST /desktop/pair/qr  /claim  /cancel   phone connections (CONNECTIONS.md §4.1)
    POST /desktop/token/reissue              the Mac, device token + account bearer
    GET  /desktop/pending-actions/{id}       one card, owner only
    POST /desktop/devices/{id}/tasks         a phone's task for one Mac
    GET  /desktop/tasks  /tasks/{id}         task list and detail
    POST /desktop/tasks/{id}/cancel
    POST /desktop/internal/action-result     ← agent, X-Agent-Key
    POST /desktop/internal/task-status       ← agent, X-Agent-Key

TENANT AGENT (`https://agent-<prefix>.agents.toup.ai`) owns:

    WS   /ws/desktop                         the relay itself — on `ws_router`,
                                             which ONLY agent_main mounts
    POST /desktop/internal/revoked           ← platform, X-Agent-Key
    POST /desktop/internal/dispatch-approved ← platform, X-Agent-Key
    POST /desktop/internal/run-task          ← platform, X-Agent-Key
    POST /desktop/internal/cancel-task       ← platform, X-Agent-Key

The socket is on its own router so a platform replica serves no
`/api/ws/desktop` at all. When it shared `router`, a Mac that dialled the
site's own host landed on a platform replica, which verified the token,
wrote `last_seen_at` and so showed the Mac Online, while the tenant agent's
registry stayed empty and every tool answered "not connected"
(CONNECTIONS.md F-A).

Why the socket terminates on the AGENT and not on the platform, which is
the single most consequential decision in this file:

  * `desktop_bridge.dispatch()` is called by the `desktop` skill, which runs
    inside `tool_executor`, which runs inside the tenant agent container. If
    the socket lived on the platform, every local tool call would have to
    cross a process boundary to reach the registry holding the future it is
    awaiting — which is `agent-tool-relay.md` §1.9 exactly, the defect that
    makes the extension's session routes return `[]` in production. On the
    hot path a hung future is a hung turn, not an empty list.
  * A tenant agent is ONE process, so "close the live socket" (§1.4) is
    unambiguous. The platform's two replicas cannot close each other's
    sockets, so a platform-terminated relay would reintroduce the same
    split in the one place revocation has to work.
  * `pool_service.py:642` mints `agent_url` as
    `https://agent-<prefix>.agents.toup.ai`, and the client's
    `RelayHostPolicy` allows a host that equals or ends with `".toup.ai"`
    (`RelayHostPolicy.swift:129`). So a tenant-agent relay URL satisfies §4
    with no query string, no port and no fragment. This was checked, not
    assumed: had it not held, the design would have had to change.

Consequence, and it is a feature: the agent cannot read `desktop_devices`
(PLATFORM_ONLY — see `app/db/models/desktop.py`), so it CANNOT authenticate
a device without asking the platform. The revocation check is therefore
impossible to skip. `extension.py:841-848` describes the jti denylist as a
"follow-up" and `_decode_extension_token` consults neither `revoked_at` nor
any denylist; here there is no code path that could.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
)
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, select, update as sa_update
from sqlalchemy.exc import IntegrityError, ProgrammingError

from app.agent import desktop_bridge
from app.api import _infra_errors as _infra
from app.api.auth import get_current_user, security
from app.config import settings
from app.db import get_db
from app.db.database import async_session_maker
from app.db.models import (
    AgentConfig,
    DesktopDevice,
    DesktopPairing,
    DesktopPendingAction,
    DesktopTask,
)
from app.services import desktop_task_state as task_state

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Desktop"])
#: The relay socket, and nothing else. Mounted by `agent_main.py` only.
ws_router = APIRouter(tags=["Desktop"])


# ═════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════
PAIR_TTL_S = 600                 # §1.1 `expires_in`
PAIR_POLL_INTERVAL_S = 5         # §1.1 `interval`
#: Online window. The agent heartbeats on connect, every 45 s, and on close;
#: 90 s tolerates two misses before a device reads as offline. Same value and
#: same reasoning as `extension.DEVICE_ONLINE_WINDOW_S`.
DEVICE_ONLINE_WINDOW_S = 90
HEARTBEAT_INTERVAL_S = 45
#: How long a staged local action stays actionable. SHORTER than the
#: connector flow's 24 h (`_PENDING_ACTION_TTL_HOURS`) on purpose: that TTL
#: is sized for "I'll deal with this after lunch" on a draft email, and the
#: reasoning behind it ("a card found by scrolling through last week cannot
#: fire") argues for less time, not the same, when the thing being confirmed
#: is a command on the user's own machine. Approval additionally requires the
#: Mac to be online at tap time, so a long TTL buys nothing here.
PENDING_ACTION_TTL_S = 15 * 60

#: JWT audience and scope. NOT `toup-agent-session` / `extension` — §2 and
#: §9.2: the extension's token deliberately works on `/ws/extension` AND
#: `/ws/chat`, which makes its `scope` a label rather than a restriction.
DESKTOP_TOKEN_AUD = "toup-desktop"
DESKTOP_TOKEN_SCOPE = "desktop"
DESKTOP_TOKEN_ISS = "toup-platform"

#: Domain-separation label for the signing subkey. See `_signing_key()`.
_KEY_LABEL = b"toup-desktop-relay-v1"

_USER_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L

# ── Phone connections (CONNECTIONS.md §3.3, §4) ─────────────────────────
#: A QR challenge lives at most this long, and never past its pairing.
QR_TTL_S = 120
#: The Mac may not mint more often than this.
QR_MIN_REMINT_S = 20
#: After a phone claims a QR, only it may decide, and only for this long.
CLAIM_WINDOW_S = 120
#: An approved bundle nobody collected is withdrawn after this (F-L).
APPROVED_UNCOLLECTED_S = 120
#: A token may be re-issued up to this long after it expired, so a Mac that
#: slept for days recovers without pairing again. Never on its own: the
#: Mac's account bearer is required too (F-F, T8).
REISSUE_GRACE_S = 7 * 24 * 3600
#: The replaced jti still authenticates for this long, so a socket opened
#: with it survives the reconnect onto the new token.
PREV_JTI_GRACE_S = 120
#: A socket the Mac has said nothing on for this long is half-open (F-I).
INBOUND_SILENCE_S = 60
#: The Mac's own confirmation panel waits 2 minutes, and an approved action
#: has to outlive it or someone who IS at the Mac never gets to answer (F-G).
MAC_ASK_WINDOW_S = 120
#: How long an approve or a submit waits for the agent to accept the hand-off.
HANDOFF_BUDGET_S = 5.0
_CHALLENGE_RE = re.compile(r"^[A-Za-z0-9_-]{22,64}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


def _platform_db_local() -> bool:
    """True when THIS process holds the `desktop_*` tables.

    They are PLATFORM_ONLY, so `init_db` creates them under
    `RUN_MODE=platform` and `monolith` and not under `agent`. Every
    cross-process helper below short-circuits to a local query when this is
    True — which is also what makes the whole flow exercisable in a
    single-process test run.
    """
    return (settings.run_mode or "").strip().lower() in ("platform", "monolith")


def _uid_of(current_user: Any) -> str:
    return current_user.id if hasattr(current_user, "id") else current_user["id"]


def _flat(code: str, status: int, detail: str = "") -> JSONResponse:
    """`{error, detail}` at the top level (CONNECTIONS.md §4 conventions).

    The Mac reads `error`, the phone reads `detail`. A route the phone calls
    answers 401 ONLY for a bad account bearer, because the phone signs out
    on any 401.
    """
    body: Dict[str, Any] = {"error": code}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status)


def _session_jti(request: Request) -> Optional[str]:
    """The caller's live session, as `get_current_user` exposed it."""
    return getattr(request.state, "user_session_jti", None)


def _client_via(request: Request) -> str:
    """Audit label only — never a control."""
    client = (request.headers.get("x-toup-client") or "").strip().lower()
    return "app" if client in ("ios", "android") else "web"


def _flatten(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def _challenge_hash(challenge: str) -> str:
    return hashlib.sha256(challenge.encode("ascii")).hexdigest()


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() + "Z" if dt else None


async def _optional_user(
    request: Request,
    credentials=Depends(security),
    db=Depends(get_db),
):
    """The account behind the bearer, or None instead of a bare 401.

    For the routes that must answer a flat `{"error": …}` body the Mac can
    read, rather than FastAPI's `{"detail": …}`.
    """
    try:
        return await get_current_user(request, credentials, db)
    except HTTPException as exc:
        if exc.status_code == 401:
            return None
        raise


async def _connections_on(user_id: Optional[str]) -> bool:
    """The `desktop_connections` registry flag for one account.

    Own session and dark on any error, like `_automations_env_flag`: a flag
    read must never be the thing that turns a feature on.
    """
    if not user_id:
        return False
    try:
        from app.services.feature_flags import is_enabled

        async with async_session_maker() as db:
            return await is_enabled(db, "desktop_connections", user_id)
    except Exception as exc:
        logger.debug("[desktop] connections flag read failed: %s", exc)
        return False


async def _rate_user(key: str, limit: int, window_s: int) -> bool:
    return await _rate_check(key, limit, window_s)


def _public_base_url() -> str:
    candidate = (
        getattr(settings, "public_base_url", None)
        or getattr(settings, "frontend_url", None)
        or getattr(settings, "platform_api_url", "https://toup.ai/api")
    )
    if candidate.endswith("/api"):
        candidate = candidate[:-4]
    return candidate.rstrip("/") or "https://toup.ai"


# ═════════════════════════════════════════════════════════════════════════
# The device token (§2)
# ═════════════════════════════════════════════════════════════════════════
def _signing_key() -> str:
    """A dedicated HMAC key for desktop tokens, derived from `jwt_secret`.

    §2's fourth requirement: "Consider not signing it with
    `AgentConfig.agent_api_key`. That one secret already signs extension
    tokens, authenticates the agent to the platform's MCP server and
    authenticates heartbeats. Reusing it for a credential with filesystem
    reach widens the blast radius of a single key."

    So it is NOT reused, and neither is `jwt_secret` itself:

      * Not `agent_api_key` — audit risk #9, one secret with three roles.
        It is also the wrong shape: it is per-tenant, and this token is
        minted by the platform before any tenant is involved.
      * Not raw `jwt_secret` — that signs account JWTs. Domain separation
        means that even if `aud`/`scope` were mishandled by some future
        reader, a desktop token still cannot verify as an account token or
        vice versa: the signature fails first, which is a stronger boundary
        than a claim check.
      * A derived subkey rather than a new env var, because a new secret
        would have to be distributed to every replica before this code could
        run at all, and the task is explicit that nothing here touches
        production configuration. HKDF-style domain separation over a secret
        the platform already has gives the isolation without the rollout.
    """
    base = (settings.jwt_secret or "").strip()
    if not base:
        raise HTTPException(503, "server is not configured to issue device tokens")
    return hmac.new(base.encode(), _KEY_LABEL, hashlib.sha256).hexdigest()


def _mint_device_token(user_id: str, device_id: str) -> Dict[str, Any]:
    ttl = int(getattr(settings, "desktop_token_ttl_s", 12 * 3600) or 12 * 3600)
    now = int(time.time())
    jti = uuid.uuid4().hex
    payload = {
        "sub": user_id,
        "iss": DESKTOP_TOKEN_ISS,
        "aud": DESKTOP_TOKEN_AUD,
        "scope": DESKTOP_TOKEN_SCOPE,
        "jti": jti,
        "did": device_id,
        "iat": now,
        "exp": now + ttl,
    }
    token = jwt.encode(payload, _signing_key(), algorithm="HS256")
    # A JWT must be a valid HTTP token to ride a subprotocol header (§3.1):
    # no `=` padding, no `,`, no space. base64url JWS satisfies that, but the
    # client raises rather than opening a socket with a mangled header, so a
    # token that could never connect is caught here instead of presenting on
    # the Mac as a mysterious 4001.
    if any(c in token for c in (",", " ", "=")) or not token.isascii():
        raise HTTPException(500, "minted token is not header-safe")
    return {"token": token, "jti": jti, "expires_in": ttl}


def _decode_device_token(token: str) -> Optional[Dict[str, Any]]:
    """Signature/claims only. Says NOTHING about revocation.

    Deliberately not called by the WS handler. Revocation is a DB fact and
    the agent has no copy of that table, so the socket authenticates through
    `_verify_device_token` — which consults the row — and this function
    exists only as the platform-side half of it. Keeping it private and
    un-exported is what stops a future connect path from being written
    against claims alone, which is the extension's exact defect (§9.1).
    """
    if not token:
        return None
    try:
        payload = jwt.decode(
            token,
            _signing_key(),
            algorithms=["HS256"],
            audience=DESKTOP_TOKEN_AUD,
            issuer=DESKTOP_TOKEN_ISS,
        )
    except JWTError as exc:
        logger.info("[desktop] token decode failed: %s", exc)
        return None
    except HTTPException:
        return None
    if payload.get("scope") != DESKTOP_TOKEN_SCOPE:
        logger.warning("[desktop] token wrong scope: %r", payload.get("scope"))
        return None
    if not payload.get("sub"):
        return None
    return payload


# ═════════════════════════════════════════════════════════════════════════
# Schemas
# ═════════════════════════════════════════════════════════════════════════
class PairInitReq(BaseModel):
    device_name: str = Field("Mac", max_length=200)
    app_version: str = Field("0.0.0", max_length=40)
    os_version: str = Field("", max_length=40)
    flavor: str = Field("direct", max_length=16)
    model: Optional[str] = Field(None, max_length=64)


class PairDeviceCodeReq(BaseModel):
    device_code: str = Field(..., min_length=20, max_length=128)


class PairClaimReq(BaseModel):
    challenge: str = Field(..., max_length=128)
    phone_label: Optional[str] = Field(None, max_length=200)


class PairDecisionReq(BaseModel):
    """Exactly one of the two: the typed code, or the claimed QR challenge."""

    user_code: Optional[str] = Field(None, min_length=4, max_length=16)
    challenge: Optional[str] = Field(None, max_length=128)


class ReissueReq(BaseModel):
    device_token: str = Field(..., max_length=4096)


class ActionResultReq(BaseModel):
    user_id: str
    action_id: str = Field(..., max_length=36)
    status: str = Field(..., max_length=16)
    summary: str = Field("", max_length=2000)
    mac_prompt: bool = False


class TaskSubmitReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    client_request_id: str = Field(..., max_length=64)


class RunTaskReq(BaseModel):
    user_id: str
    task_id: str = Field(..., max_length=36)
    device_id: str = Field(..., max_length=36)
    device_name: str = Field("your Mac", max_length=200)
    text: str = Field(..., min_length=1, max_length=2000)


class CancelTaskReq(BaseModel):
    user_id: str
    task_id: str = Field(..., max_length=36)


class TaskStatusReq(BaseModel):
    user_id: str
    task_id: str = Field(..., max_length=36)
    turn_state: str = Field(..., max_length=16)


class PairPollReq(BaseModel):
    device_code: str = Field(..., min_length=20, max_length=128)


class PairCodeReq(BaseModel):
    user_code: str = Field(..., min_length=4, max_length=16)


class RevokeReq(BaseModel):
    #: The account path names one Mac, or says `all: true`. Omitting both is
    #: a 400: a missing field used to mean "every Mac on the account".
    device_id: Optional[str] = Field(None, max_length=36)
    all: bool = False


class HeartbeatReq(BaseModel):
    user_id: str
    device_id: Optional[str] = None
    token_jti: Optional[str] = None
    online: bool = True


class VerifyTokenReq(BaseModel):
    token: str = Field(..., max_length=4096)


class StageActionReq(BaseModel):
    user_id: str
    device_id: str = Field(..., max_length=36)
    tool_name: str = Field(..., max_length=128)
    payload: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = Field(None, max_length=1000)
    channel: str = Field("desktop", max_length=32)
    conversation_id: Optional[str] = Field(None, max_length=36)
    remote_task_id: Optional[str] = Field(None, max_length=36)


class RevokedPushReq(BaseModel):
    user_id: str
    device_id: Optional[str] = None
    token_jti: Optional[str] = None


class DispatchApprovedReq(BaseModel):
    user_id: str
    action_id: str = Field(..., max_length=36)
    device_id: str = Field(..., max_length=36)
    tool_name: str = Field(..., max_length=128)
    task_id: str = Field(..., max_length=64)
    payload: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = Field(None, max_length=240)
    remote_task_id: Optional[str] = Field(None, max_length=36)


# ═════════════════════════════════════════════════════════════════════════
# Rate limiting (per-IP, per-replica) — copied from extension.py:87-119
# ═════════════════════════════════════════════════════════════════════════
_RATE_BUCKETS: Dict[str, list] = {}
_rate_lock = asyncio.Lock()


async def _rate_check(key: str, limit: int, window_s: int) -> bool:
    now = time.time()
    async with _rate_lock:
        bucket = _RATE_BUCKETS.setdefault(key, [])
        cutoff = now - window_s
        i = 0
        while i < len(bucket) and bucket[i] < cutoff:
            i += 1
        if i:
            del bucket[:i]
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def _client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",", 1)[0].strip()
    return req.client.host if req.client else "unknown"


async def _enforce_rate(req: Request, name: str, *, limit: int, window_s: int) -> None:
    if not await _rate_check(f"{name}:{_client_ip(req)}", limit, window_s):
        raise HTTPException(429, f"rate limited ({name}); retry shortly")


# ═════════════════════════════════════════════════════════════════════════
# Agent-key auth for the internal (process-to-process) routes
# ═════════════════════════════════════════════════════════════════════════
async def _user_for_agent_key(agent_key: Optional[str]) -> str:
    """Resolve `X-Agent-Key` → the tenant's user id, or raise.

    The agent never *claims* a user id that is then trusted: the key itself
    identifies the tenant, and any user id in the body must match what this
    returns. That is what stops one tenant's container from verifying,
    staging for, or revoking another tenant's device.
    """
    if not agent_key:
        raise HTTPException(401, "X-Agent-Key required")
    if not _platform_db_local():
        # The receiver's AgentConfig row is tenant-local metadata; it need
        # not contain the platform's credential. Trust only the bound runtime
        # identity, as the agent middleware does, never a body-supplied uid.
        from app.services.runtime_identity import get_agent_api_key, get_user_id
        expected = get_agent_api_key() or ""
        uid = get_user_id()
        if not uid or not expected or not hmac.compare_digest(agent_key, expected):
            raise HTTPException(403, "agent key not recognised")
        return uid
    try:
        async with async_session_maker() as db:
            row = await db.execute(
                select(AgentConfig.user_id).where(
                    AgentConfig.agent_api_key == agent_key,
                )
            )
            uid = row.scalar_one_or_none()
    except Exception as exc:
        logger.warning("[desktop] agent-key resolve failed: %s", exc)
        raise HTTPException(503, "agent key check unavailable")
    if not uid:
        raise HTTPException(403, "agent key not recognised")
    return uid


def _require_agent_bound_user(resolved: str, claimed: str) -> str:
    if claimed and claimed != resolved:
        raise HTTPException(403, "agent key does not own that user")
    return resolved


# ═════════════════════════════════════════════════════════════════════════
# Pairing — device-code, Flow A only (§1.1–§1.3, §9.4)
# ═════════════════════════════════════════════════════════════════════════
#
# There is deliberately NO `pair/issue`. The extension has one
# (`extension.py:562-597`): a page on toup.ai mints a credential and hands
# it to the extension with no user-visible confirmation. §1.3: "A Mac app
# that can read the user's files and run commands is approved on a page that
# names it, or not at all. Do not add Flow B."


async def _gc_pairings(db) -> None:
    """Expire stale rows lazily, on the paths that read them.

    Same reasoning as `connector_pending_actions.py:407-412`: a background
    sweep would be tidier, but the moment someone is looking at a record is
    the moment we know it is stale.

    `expired`, never `denied`. This runs on ANOTHER caller's request —
    `pair/init`, `pair/lookup`, `pair/approve` — so a code that timed out is
    routinely rewritten before its own device polls again. Writing `denied`
    made that poll answer `{"error": "denied"}`, which `PairingClient` maps
    to `.denied`, and the Mac told its user they had declined a pairing on a
    page they never opened. §1.2 gives the clock and the person different
    answers on purpose.

    An APPROVED row whose bundle nobody collected within
    `APPROVED_UNCOLLECTED_S` is withdrawn too, and the device row it created
    is revoked (CONNECTIONS.md F-L): otherwise a pairing approved after the
    Mac quit keeps a live device token at rest in this table for as long as
    the JWT lives.
    """
    now = datetime.utcnow()
    await db.execute(
        sa_update(DesktopPairing)
        .where(DesktopPairing.status == "pending")
        .where(DesktopPairing.expires_at <= now)
        .values(status="expired", decided_at=now)
    )
    stale = (await db.execute(
        select(DesktopPairing.id, DesktopPairing.approved_device_id).where(
            DesktopPairing.status == "approved",
            DesktopPairing.decided_at <= now - timedelta(
                seconds=APPROVED_UNCOLLECTED_S,
            ),
        )
    )).all()
    if stale:
        await _withdraw_approved(db, [r[0] for r in stale],
                                 [r[1] for r in stale if r[1]], now)


async def _withdraw_approved(
    db, pairing_ids: List[str], device_ids: List[str], now: datetime,
) -> None:
    await db.execute(
        sa_update(DesktopPairing)
        .where(DesktopPairing.id.in_(pairing_ids))
        .where(DesktopPairing.status == "approved")
        .values(status="expired", approved_token=None)
    )
    if device_ids:
        await db.execute(
            sa_update(DesktopDevice)
            .where(DesktopDevice.id.in_(device_ids))
            .where(DesktopDevice.revoked_at.is_(None))
            .values(revoked_at=now, token_jti=None, prev_token_jti=None)
        )


@router.post("/desktop/pair/init")
async def pair_init(
    req: PairInitReq, request: Request, current_user=Depends(_optional_user),
) -> Any:
    """§1.1, now bound to the account signed in on the Mac (CONNECTIONS.md A1).

    It used to need no login at all, and `approve` accepted any signed-in
    account: an attacker could start a pairing from a script, talk a victim
    into approving the code, and collect a device token for the VICTIM's
    account (F-K). Binding the pairing to the initiating account, and every
    decision path to that binding, closes it.
    """
    await _enforce_rate(request, "desktop_pair_init", limit=10, window_s=60)
    if current_user is None:
        return _flat("sign_in", 401, "Sign in to Toup on this Mac again.")
    user_id = _uid_of(current_user)
    if not await _connections_on(user_id):
        return _flat("unavailable", 404, "Mac Connections isn't available for this account.")
    jti = _session_jti(request)
    if not jti:
        return _flat(
            "session_unknown", 403,
            "This sign-in can't be used to connect a Mac. Sign in again.",
        )
    if not await _rate_user(f"desktop_pair_init_user:{user_id}", 5, 60):
        raise HTTPException(429, "rate limited (desktop_pair_init); retry shortly")
    now = datetime.utcnow()
    device_code = secrets.token_urlsafe(32)          # >= 128 bits (§1.1)

    try:
        async with async_session_maker() as db:
            await _gc_pairings(db)
            # Retry on collision rather than trusting one draw: `user_code`
            # is unique-indexed, so a clash is an IntegrityError on commit
            # and the user sees a failure they cannot act on.
            for _ in range(20):
                raw = "".join(secrets.choice(_USER_CODE_ALPHABET) for _ in range(8))
                user_code = f"{raw[:4]}-{raw[4:]}"
                exists = await db.execute(
                    select(DesktopPairing.id).where(
                        DesktopPairing.user_code == user_code,
                    )
                )
                if exists.scalar_one_or_none() is None:
                    break
            else:
                raise HTTPException(503, "pairing slot pressure; retry shortly")

            db.add(DesktopPairing(
                device_code=device_code,
                user_code=user_code,
                device_name=(req.device_name or "Mac")[:200],
                app_version=(req.app_version or None) and req.app_version[:40],
                os_version=(req.os_version or None) and req.os_version[:40],
                flavor=(req.flavor or "direct")[:16],
                model=_flatten(req.model, 64) or None,
                user_id=user_id,
                init_session_jti=jti,
                status="pending",
                created_at=now,
                expires_at=now + timedelta(seconds=PAIR_TTL_S),
            ))
            await db.commit()
    except HTTPException:
        raise
    except Exception as exc:
        _infra.raise_if_infrastructure(exc)
        logger.warning("[desktop] pair_init failed: %s", exc)
        raise HTTPException(503, "pairing is unavailable right now")

    # §1.1: `verification_uri` is validated by the device before it is
    # opened (§4) — https, allowed host, no userinfo, no fragment, default
    # port. `_public_base_url()` yields the site origin, so the only way to
    # violate that is to misconfigure the platform's own base URL.
    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": f"{_public_base_url()}/settings/desktop/pair",
        "expires_in": PAIR_TTL_S,
        "interval": PAIR_POLL_INTERVAL_S,
    }


def _pair_error(code: str, status: int = 400) -> Any:
    """§1.2's answer vocabulary, at the TOP LEVEL of the body.

    Not `HTTPException(400, {"error": …})`: FastAPI wraps that as
    `{"detail": {"error": …}}`, and `PairingClient.pollOnce`
    (`ToupRelay/PairingClient.swift:156-176`) reads `json["error"]` off the
    root. Nested, the two answers the user acted on — "you pressed Deny" and
    "the code ran out" — both reach the Mac as `PairingError.server(400)`,
    i.e. a number. `desktop/mock/relay/relay.mjs:79-93` emits the flat shape
    and the app is tested against it.
    """
    return JSONResponse({"error": code}, status_code=status)


def _claim_open(row: DesktopPairing, now: datetime) -> bool:
    return bool(
        row.qr_claimed_at is not None
        and row.claim_expires_at is not None
        and row.claim_expires_at > now
    )


@router.post("/desktop/pair/poll")
async def pair_poll(req: PairPollReq) -> Any:
    """§1.2. 202 pending · 429 slow_down · 400 denied/expired · 200 bundle."""
    now = datetime.utcnow()
    try:
        async with async_session_maker() as db:
            row = (await db.execute(
                select(DesktopPairing).where(
                    DesktopPairing.device_code == req.device_code,
                )
            )).scalar_one_or_none()

            if row is None:
                # Unknown and expired are one answer on purpose: a device
                # code is a bearer, and telling a caller "that code exists
                # but is not yours" is an oracle.
                return _pair_error("expired")

            if row.status == "pending" and row.expires_at <= now:
                await db.execute(
                    sa_update(DesktopPairing)
                    .where(DesktopPairing.id == row.id)
                    .where(DesktopPairing.status == "pending")
                    .values(status="expired", decided_at=now)
                )
                await db.commit()
                return _pair_error("expired")

            if row.status == "pending":
                # §1.2: polling faster than the advertised interval gets
                # `slow_down`, and the device adds 5 s permanently for the
                # rest of the attempt (the OAuth device-flow convention).
                if (
                    row.last_polled_at
                    and (now - row.last_polled_at).total_seconds()
                    < PAIR_POLL_INTERVAL_S - 1
                ):
                    return _pair_error("slow_down", 429)
                claimed = _claim_open(row, now)
                claimed_on = row.qr_claimed_label if claimed else None
                await db.execute(
                    sa_update(DesktopPairing)
                    .where(DesktopPairing.id == row.id)
                    .values(last_polled_at=now)
                )
                await db.commit()
                # `claimed` is what lets the Mac hide its QR and name the
                # phone that opened it (CONNECTIONS.md T2).
                return JSONResponse(
                    {"error": "authorization_pending", "claimed": claimed,
                     "claimed_on": claimed_on},
                    status_code=202,
                )

            if row.status == "denied":
                return _pair_error("denied")
            if row.status != "approved":
                # `expired` (the sweep got here first) or `consumed` (the
                # bundle was collected once and the row was cleared). Both
                # answer `expired`: a replay is not a different outcome, and
                # neither is a clock that ran out on another request's
                # sweep. Only a PERSON's refusal answers `denied`.
                return _pair_error("expired")

            if not await _connections_on(row.user_id):
                return _flat("unavailable", 404, "Mac Connections is disabled.")

            # READ THE BUNDLE BEFORE THE CLAIM, and never off `row`
            # afterwards. The claim's `.values(approved_token=None)` is an
            # ORM-enabled UPDATE, so SQLAlchemy's default
            # `synchronize_session="auto"` evaluates the criteria against
            # the identity map and writes None onto THIS loaded instance —
            # `row.approved_token` read after it is None, every time, and
            # the route answered `expired` to every successful pairing. The
            # in-memory sync is correct behaviour; reading through it was
            # the defect.
            token = row.approved_token
            device_id = row.approved_device_id
            expires_in = row.approved_expires_in
            user_id = row.user_id

            # Approved. Claim it with the same guarded single UPDATE the
            # connector flow uses, so two concurrent polls cannot both
            # collect one token.
            claimed = await db.execute(
                sa_update(DesktopPairing)
                .where(DesktopPairing.id == row.id)
                .where(DesktopPairing.status == "approved")
                .values(
                    status="consumed",
                    decided_at=row.decided_at or now,
                    # Cleared WITH the claim: the row is retained as a
                    # pairing record, and a retained row must not also be a
                    # place a live credential sits at rest.
                    approved_token=None,
                )
            )
            await db.commit()
            if (claimed.rowcount or 0) == 0:
                return _pair_error("expired")
    except HTTPException:
        raise
    except Exception as exc:
        _infra.raise_if_infrastructure(exc)
        logger.warning("[desktop] pair_poll failed: %s", exc)
        raise HTTPException(503, "pairing is unavailable right now")

    if not token or not user_id or not device_id:
        return _pair_error("expired")

    return {
        "access_token": token,
        # §1.2/§7.1: the app checks this against the account signed in
        # inside the app and discards the token BEFORE it reaches the
        # Keychain if they differ. So it must be the account that approved —
        # not an alias, not a tenant id — and stable against what
        # `/api/auth/me` returns. It is `users.id` on both, which
        # `SHARED_COLUMN_AUTHORITY["users.id"]` declares
        # immutable-after-create and "the cross-DB join key".
        "user_id": user_id,
        "device_id": device_id,
        "relay_ws_url": await _relay_ws_url_for(user_id),
        "expires_in": expires_in or int(
            getattr(settings, "desktop_token_ttl_s", 12 * 3600)
        ),
    }


@router.post("/desktop/pair/qr")
async def pair_qr(req: PairDeviceCodeReq, request: Request) -> Any:
    """Mint the QR challenge for a pending, bound pairing (CONNECTIONS.md §3.3).

    Only the holder of the `device_code` — the Mac that started it — can
    mint. Minting replaces the previous challenge, so an older QR stops
    working the moment a new one exists.
    """
    await _enforce_rate(request, "desktop_pair_qr", limit=30, window_s=60)
    now = datetime.utcnow()
    async with async_session_maker() as db:
        row = (await db.execute(
            select(DesktopPairing).where(
                DesktopPairing.device_code == req.device_code,
            )
        )).scalar_one_or_none()
        if (
            row is None or row.status != "pending" or row.expires_at <= now
            or not row.user_id
        ):
            return _flat("expired", 400, "This pairing has ended. Start again.")
        if not await _connections_on(row.user_id):
            return _flat("unavailable", 404)
        if _claim_open(row, now):
            return _flat(
                "claimed", 409,
                "A phone opened this code. Decide there, or cancel here.",
            )
        if (
            row.qr_minted_at is not None
            and (now - row.qr_minted_at).total_seconds() < QR_MIN_REMINT_S
        ):
            return _flat("slow_down", 429)

        challenge = secrets.token_urlsafe(16)
        qr_expires = min(now + timedelta(seconds=QR_TTL_S), row.expires_at)
        r = await db.execute(
            sa_update(DesktopPairing)
            .where(DesktopPairing.id == row.id)
            .where(DesktopPairing.status == "pending")
            .where(or_(
                DesktopPairing.qr_claimed_at.is_(None),
                DesktopPairing.claim_expires_at <= now,
            ))
            .values(
                qr_challenge_hash=_challenge_hash(challenge),
                qr_minted_at=now,
                qr_expires_at=qr_expires,
                qr_claimed_at=None,
                qr_claimed_session_jti=None,
                qr_claimed_label=None,
                claim_expires_at=None,
            )
        )
        await db.commit()
        if (r.rowcount or 0) == 0:
            return _flat("claimed", 409)
        user_code, pairing_expires = row.user_code, row.expires_at
    return {
        "challenge": challenge,
        "expires_in": max(1, int((qr_expires - now).total_seconds())),
        "user_code": user_code,
        "pairing_expires_in": max(1, int((pairing_expires - now).total_seconds())),
    }


@router.post("/desktop/pair/cancel")
async def pair_cancel(req: PairDeviceCodeReq) -> Dict[str, Any]:
    """The Mac stopped waiting. Idempotent, and never an error.

    An approved row whose bundle has not been collected is withdrawn and its
    device revoked: the Mac asked to stop, so nothing it did not collect may
    stay valid.
    """
    now = datetime.utcnow()
    try:
        async with async_session_maker() as db:
            row = (await db.execute(
                select(DesktopPairing).where(
                    DesktopPairing.device_code == req.device_code,
                )
            )).scalar_one_or_none()
            if row is not None and row.status == "pending":
                await db.execute(
                    sa_update(DesktopPairing)
                    .where(DesktopPairing.id == row.id)
                    .where(DesktopPairing.status == "pending")
                    .values(status="expired", decided_at=now,
                            qr_challenge_hash=None)
                )
            elif row is not None and row.status == "approved":
                await _withdraw_approved(
                    db, [row.id],
                    [row.approved_device_id] if row.approved_device_id else [],
                    now,
                )
            await db.commit()
    except Exception as exc:
        logger.info("[desktop] pair_cancel ignored: %s", exc)
    return {"ok": True}


@router.post("/desktop/pair/claim")
async def pair_claim(
    req: PairClaimReq, request: Request, current_user=Depends(get_current_user),
) -> Any:
    """A phone opened the Mac's QR (CONNECTIONS.md §3.3).

    A claim is a read-and-hold, never a decision. It uses the challenge up
    for every other session, and it answers with the Mac's identity so the
    phone can show it before anything is approved.
    """
    user_id = _uid_of(current_user)
    jti = _session_jti(request)
    if not jti:
        return _flat(
            "session_unknown", 403,
            "This sign-in can't be used to connect a Mac. Sign in again.",
        )
    if not await _connections_on(user_id):
        return _flat("unavailable", 404,
                     "Connecting a Mac from your phone isn't available yet.")
    await _enforce_rate(request, "desktop_pair_claim", limit=30, window_s=60)
    if not await _rate_user(f"desktop_pair_claim_user:{user_id}", 10, 60):
        raise HTTPException(429, "rate limited (desktop_pair_claim); retry shortly")

    gone = _flat(
        "expired", 404,
        "This code has run out or was already used. Scan the new one on "
        "your Mac.",
    )
    if not _CHALLENGE_RE.match(req.challenge or ""):
        return gone
    h = _challenge_hash(req.challenge)
    now = datetime.utcnow()
    label = _flatten(req.phone_label, 60) or "your phone"

    async with async_session_maker() as db:
        row = (await db.execute(
            select(DesktopPairing).where(DesktopPairing.qr_challenge_hash == h)
        )).scalar_one_or_none()
        if row is None or row.status != "pending" or row.expires_at <= now:
            return gone
        if row.user_id != user_id:
            # Deliberately NOT used up: a shoulder-surfer signed in to
            # another account must not be able to spoil the owner's QR (T4).
            return _flat(
                "other_account", 403,
                "This Mac is signed in to a different Toup account. Sign in "
                "to that account here, or connect the Mac from that account.",
            )
        if row.init_session_jti == jti:
            return _flat("same_session", 409, "Scan this with your phone.")

        claim_until = min(now + timedelta(seconds=CLAIM_WINDOW_S), row.expires_at)
        # The first claim. Single use lives in `qr_claimed_at IS NULL`: from
        # this UPDATE on, no other session can claim, approve or deny with
        # this challenge.
        first = await db.execute(
            sa_update(DesktopPairing)
            .where(DesktopPairing.id == row.id)
            .where(DesktopPairing.qr_challenge_hash == h)
            .where(DesktopPairing.status == "pending")
            .where(DesktopPairing.user_id == user_id)
            .where(DesktopPairing.expires_at > now)
            .where(DesktopPairing.qr_expires_at > now)
            .where(DesktopPairing.qr_claimed_at.is_(None))
            .where(or_(
                DesktopPairing.init_session_jti.is_(None),
                DesktopPairing.init_session_jti != jti,
            ))
            .values(
                qr_claimed_at=now,
                qr_claimed_session_jti=jti,
                qr_claimed_label=label,
                claim_expires_at=claim_until,
            )
        )
        await db.commit()
        if (first.rowcount or 0) == 0:
            # Re-opening the link on the phone that already holds it
            # continues where it was; the window is not extended.
            again = (await db.execute(
                select(DesktopPairing).where(
                    DesktopPairing.id == row.id,
                    DesktopPairing.qr_challenge_hash == h,
                    DesktopPairing.status == "pending",
                    DesktopPairing.user_id == user_id,
                    DesktopPairing.qr_claimed_session_jti == jti,
                    DesktopPairing.claim_expires_at > now,
                )
            )).scalar_one_or_none()
            if again is None:
                return gone
            claim_until = again.claim_expires_at
    return {
        "device_name": row.device_name,
        "model": row.model,
        "os_version": row.os_version,
        "app_version": row.app_version,
        "flavor": row.flavor,
        "user_code": row.user_code,
        "decide_by": _iso(claim_until),
    }


async def _relay_ws_url_for(user_id: str) -> str:
    """The tenant agent's `/api/ws/desktop`, as a `wss://` URL.

    §4 constrains what may be emitted: no query, no fragment, no port, and
    a host on `toup.ai` or a subdomain. `pool_service` mints
    `https://agent-<prefix>.agents.toup.ai`, so the swap to `wss://` plus
    the fixed path satisfies all of it. An agent whose `agent_url` somehow
    carries a query or a port would be refused BY THE DEVICE, which is the
    right place for that check to live (the app has no CSP to fall back on
    — §9.6) — but emitting one is our bug, so it is refused here too.
    """
    try:
        async with async_session_maker() as db:
            row = await db.execute(
                select(AgentConfig.agent_url).where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.deploy_status == "active",
                )
            )
            agent_url = row.scalar_one_or_none()
    except Exception:
        agent_url = None
    if not agent_url:
        # A monolith / dev run serves `/api/ws/desktop` from THIS process, so
        # there is no tenant agent to name and the site's own origin is the
        # correct answer. Only for `monolith`: under `platform` the socket
        # deliberately terminates on the tenant agent (module header), and
        # handing back the platform's origin would point the Mac at a process
        # whose `desktop_bridge` registry no tool call can reach — §1.9
        # arriving by a different door.
        if (settings.run_mode or "").strip().lower() == "monolith":
            agent_url = _public_base_url()
        else:
            return ""
    ws = (
        agent_url.replace("https://", "wss://").replace("http://", "ws://")
        .rstrip("/") + "/api/ws/desktop"
    )
    if "?" in ws or "#" in ws or "@" in ws:
        logger.error(
            "[desktop] refusing to emit a relay URL with query/fragment/userinfo "
            "for user=%s — check agent_configs.agent_url", user_id[:8],
        )
        return ""
    return ws


@router.get("/desktop/pair/lookup")
async def pair_lookup(
    user_code: str = Query(..., max_length=16),
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """What the approval page shows BEFORE the button (§1.3).

    `extension.py:610-626` is the precedent: name the device, its version
    and its flavour, then ask. The flavour matters here in a way it does not
    for a browser — a `direct` build can run commands and drive other apps,
    a `mas` build cannot — so the page can say what is being granted.

    Another account's code is the same 404 as an unknown one (A2).
    """
    user_id = _uid_of(current_user)
    if not await _rate_user(f"desktop_pair_lookup_user:{user_id}", 20, 60):
        raise HTTPException(429, "rate limited (desktop_pair_lookup); retry shortly")
    async with async_session_maker() as db:
        await _gc_pairings(db)
        await db.commit()
        row = (await db.execute(
            select(DesktopPairing).where(
                DesktopPairing.user_code == user_code,
                DesktopPairing.user_id == user_id,
            )
        )).scalar_one_or_none()
    if not row or row.status != "pending":
        raise HTTPException(404, "code not found or expired")
    return {
        "device_name": row.device_name,
        "model": row.model,
        "app_version": row.app_version,
        "os_version": row.os_version,
        "flavor": row.flavor,
        "status": row.status,
        "expires_at": row.expires_at.isoformat() + "Z",
    }


def _decision_form(req: PairDecisionReq) -> Tuple[Optional[str], Optional[str]]:
    code = (req.user_code or "").strip() or None
    challenge = (req.challenge or "").strip() or None
    if bool(code) == bool(challenge):
        raise HTTPException(400, "send exactly one of user_code or challenge")
    return code, challenge


@router.post("/desktop/pair/approve")
async def pair_approve(
    req: PairDecisionReq, request: Request,
    current_user=Depends(get_current_user),
) -> Any:
    """§1.3. Mint the token, write the device row, stash the bundle.

    Two forms, one claim. The typed path names the `user_code`; the QR path
    names the challenge, and only the session that claimed it may use it,
    only inside its window (CONNECTIONS.md S1). Both are guarded on
    `user_id = caller` INSIDE the statement, and neither overwrites the
    pairing's account any more (F-K).
    """
    user_id = _uid_of(current_user)
    jti = _session_jti(request)
    now = datetime.utcnow()
    code, challenge = _decision_form(req)
    if not await _connections_on(user_id):
        return _flat("unavailable", 404,
                     "Connecting a Mac from your phone isn't available yet.")
    if not await _rate_user(f"desktop_pair_decide_user:{user_id}", 10, 60):
        raise HTTPException(429, "rate limited (desktop_pair_approve); retry shortly")

    gone = _flat("expired", 404, "This code has run out or was already used.")
    if challenge and (not jti or not _CHALLENGE_RE.match(challenge)):
        return gone

    # Resolved BEFORE anything is minted. A bundle with no `relay_ws_url` is
    # a dead pairing that has already spent the one-shot device code and
    # written a device row: the Mac's `RelayHostPolicy` refuses the empty
    # string (§4), so the user sees "refused an address from the server"
    # after an approval that said it worked, and has to start over. The
    # honest answer is to refuse the approval and say why — the usual cause
    # is an account whose agent has not finished provisioning.
    relay_ws_url = await _relay_ws_url_for(user_id)
    if not relay_ws_url:
        return _flat(
            "agent_not_running", 409,
            "Your agent isn't running yet, so there is nothing for this Mac "
            "to connect to. Try again once it has finished starting up.",
        )

    async with async_session_maker() as db:
        await _gc_pairings(db)
        await db.commit()
        if challenge:
            h = _challenge_hash(challenge)
            guard = [
                DesktopPairing.qr_challenge_hash == h,
                DesktopPairing.user_id == user_id,
                DesktopPairing.qr_claimed_session_jti == jti,
                DesktopPairing.claim_expires_at > now,
            ]
        else:
            guard = [
                DesktopPairing.user_code == code,
                DesktopPairing.user_id == user_id,
            ]
        row = (await db.execute(
            select(DesktopPairing).where(*guard)
        )).scalar_one_or_none()
        if not row:
            return gone
        if row.status != "pending" or row.expires_at <= now:
            return _flat("decided", 409, f"This code was already {row.status}.")
        if code and _claim_open(row, now):
            return _flat(
                "claimed", 409,
                "A phone opened this Mac's code. Decide on that phone.",
            )
        if code:
            # The typed path may not decide while a phone's claim is open —
            # otherwise the Mac's "your phone opened this" would be a lie.
            guard.append(or_(
                DesktopPairing.claim_expires_at.is_(None),
                DesktopPairing.claim_expires_at <= now,
            ))

        device_id = str(uuid.uuid4())
        bundle = _mint_device_token(user_id, device_id)

        # Guarded UPDATE: two approvals of one code must not mint two
        # devices. Whoever lands first owns it. The device row is written
        # only once the claim has landed.
        claimed = await db.execute(
            sa_update(DesktopPairing)
            .where(DesktopPairing.id == row.id)
            .where(DesktopPairing.status == "pending")
            .where(DesktopPairing.expires_at > now)
            .where(*guard)
            .values(
                status="approved",
                approved_device_id=device_id,
                approved_token=bundle["token"],
                approved_expires_in=bundle["expires_in"],
                decided_at=now,
            )
        )
        if (claimed.rowcount or 0) == 0:
            await db.rollback()
            return _flat("decided", 409, "This code was already decided.")
        db.add(DesktopDevice(
            id=device_id,
            user_id=user_id,
            device_name=row.device_name[:200],
            app_version=row.app_version,
            os_version=row.os_version,
            flavor=row.flavor,
            model=row.model,
            paired_via="app" if challenge else _client_via(request),
            paired_session_jti=jti,
            token_jti=bundle["jti"],
            token_expires_at=now + timedelta(seconds=bundle["expires_in"]),
            paired_at=now,
        ))
        await db.commit()

    logger.info(
        "[desktop] paired user=%s device=%s flavor=%s via=%s",
        user_id[:8], device_id[:8], row.flavor, "qr" if challenge else "code",
    )
    return {"ok": True, "device_id": device_id, "device_name": row.device_name}


@router.post("/desktop/pair/deny")
async def pair_deny(
    req: PairDecisionReq, request: Request,
    current_user=Depends(get_current_user),
) -> Any:
    user_id = _uid_of(current_user)
    jti = _session_jti(request)
    now = datetime.utcnow()
    code, challenge = _decision_form(req)
    if challenge and not await _connections_on(user_id):
        return _flat("unavailable", 404)
    if not await _rate_user(f"desktop_pair_decide_user:{user_id}", 10, 60):
        raise HTTPException(429, "rate limited (desktop_pair_deny); retry shortly")
    q = (
        sa_update(DesktopPairing)
        .where(DesktopPairing.status == "pending")
        .where(DesktopPairing.user_id == user_id)
        .values(status="denied", decided_at=now)
    )
    if challenge:
        if not jti or not _CHALLENGE_RE.match(challenge):
            return {"ok": True, "denied": False}
        q = (
            q.where(DesktopPairing.qr_challenge_hash == _challenge_hash(challenge))
            .where(DesktopPairing.qr_claimed_session_jti == jti)
            .where(DesktopPairing.claim_expires_at > now)
        )
    else:
        q = q.where(DesktopPairing.user_code == code)
    async with async_session_maker() as db:
        r = await db.execute(q)
        await db.commit()
    return {"ok": True, "denied": (r.rowcount or 0) > 0}


# ═════════════════════════════════════════════════════════════════════════
# Devices + revocation (§1.4, §9.1)
# ═════════════════════════════════════════════════════════════════════════
async def _list_devices(user_id: str) -> List[Dict[str, Any]]:
    try:
        async with async_session_maker() as db:
            rows = (await db.execute(
                select(DesktopDevice)
                .where(
                    DesktopDevice.user_id == user_id,
                    DesktopDevice.revoked_at.is_(None),
                )
                .order_by(DesktopDevice.paired_at.desc())
            )).scalars().all()
    except ProgrammingError:
        return []
    except Exception as exc:
        # A dead database must not read as "you have paired no devices" —
        # the same swallow made `GET /api/routines` answer 200 [] through an
        # eleven-minute pgbouncer outage (2026-09-15). Schema drift still
        # degrades to empty.
        _infra.raise_if_infrastructure(exc)
        logger.debug("[desktop] _list_devices failed: %s", exc)
        return []

    cutoff = datetime.utcnow() - timedelta(seconds=DEVICE_ONLINE_WINDOW_S)
    return [{
        "id": r.id,
        "device_name": r.device_name,
        "app_version": r.app_version,
        "os_version": r.os_version,
        "flavor": r.flavor,
        "paired_at": r.paired_at.isoformat() + "Z" if r.paired_at else None,
        "last_seen_at": (
            r.last_seen_at.isoformat() + "Z" if r.last_seen_at else None
        ),
        "online": bool(r.last_seen_at and r.last_seen_at >= cutoff),
        "token_expires_at": (
            r.token_expires_at.isoformat() + "Z" if r.token_expires_at else None
        ),
        "model": r.model,
        "paired_via": r.paired_via,
    } for r in rows]


@router.get("/desktop/devices")
async def list_devices(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """The list in Settings.

    Reads the platform DB and NOT `desktop_bridge` — §1.9 / recommendation
    11. The bridge's registry lives on the tenant agent; this endpoint is
    served by the platform, which runs two replicas with their own empty
    copies of that module. Presence comes from `last_seen_at`, which the
    agent's heartbeat writes.
    """
    return {"devices": await _list_devices(_uid_of(current_user))}


@router.post("/desktop/devices/revoke")
async def revoke_device(
    req: RevokeReq,
    authorization: Optional[str] = Header(None),
) -> Any:
    """§1.4 and CONNECTIONS.md §6. Never gated: this is the kill switch.

    Accepts EITHER the device token as `Bearer` (what the Mac sends when the
    user presses *Disconnect* locally) OR a user session plus `device_id` or
    `all: true` (what the phone and the web send). Hand-rolled rather than
    `Depends(get_current_user)` because the two credentials are different
    kinds and a dependency that accepts both would have to accept a device
    token wherever it is used.

    1. mark the row revoked and clear BOTH jti values;
    2. the `jti` stops matching, so `_verify_device_token` refuses every
       later connect — there is no separate denylist table because the row
       IS the denylist;
    3. cancel that Mac's unfinished tasks and reject its waiting cards;
    4. close the live socket with `4003`.

    The app treats this call as a courtesy and deletes its own token
    whatever we answer (§1.4) — the user's intent is local-authoritative.
    """
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()

    user_id: Optional[str] = None
    device_id: Optional[str] = req.device_id
    token_jti: Optional[str] = None
    revoke_all = False

    claims = _decode_device_token(bearer) if bearer else None
    if claims:
        # A device token revoking itself. It may revoke NOTHING ELSE: the
        # device id comes from the token, never from the body, so a stolen
        # device token cannot disconnect the user's other Macs.
        user_id = claims["sub"]
        device_id = claims.get("did")
        token_jti = claims.get("jti")
    elif bearer:
        # Not a device token — try it as an ordinary account session.
        # `decode_access_token` returns the user id, not a claims dict.
        try:
            from app.services.auth_service import (
                decode_access_token,
                get_user_by_id,
            )
            user_id = decode_access_token(bearer)
            if user_id:
                async with async_session_maker() as db:
                    u = await get_user_by_id(db, user_id)
                    if not u or not u.is_active:
                        user_id = None
        except Exception:
            user_id = None
        if user_id:
            if not device_id and not req.all:
                # Omitting the device used to revoke every Mac on the
                # account. A missing field must never be the widest action.
                return _flat(
                    "device_required", 400,
                    "Say which Mac to disconnect, or send all: true.",
                )
            revoke_all = bool(req.all) and not device_id

    if not user_id or not (device_id or revoke_all):
        raise HTTPException(401, "a device token or a signed-in session is required")

    try:
        counts = await _revoke_devices(
            user_id, None if revoke_all else [device_id], reason="revoked",
        )
    except Exception as exc:
        logger.warning("[desktop] revoke failed user=%s: %s", user_id[:8], exc)
        raise HTTPException(500, "revoke failed")

    # Close the live socket. In-process when the relay is here (monolith);
    # otherwise a push to the tenant agent, which holds it.
    closed = 0
    if revoke_all:
        closed = await _close_live_socket(user_id, None, None)
    else:
        closed = await _close_live_socket(user_id, device_id, token_jti)
    return {"ok": True, "sockets_closed": closed, **counts}


REVOKED_CARD_SUMMARY = "This Mac was disconnected, so this will not run."


async def _revoke_devices(
    user_id: str, device_ids: Optional[List[str]], *, reason: str,
) -> Dict[str, int]:
    """The row teardown behind every Disconnect, in one transaction.

    `device_ids=None` means every unrevoked Mac on the account (only ever
    from an explicit `all: true`). Everything is filtered on BOTH the user
    and the device ids, so revoking one Mac leaves another Mac's tasks and
    cards exactly as they were (CONNECTIONS.md M1).
    """
    now = datetime.utcnow()
    async with async_session_maker() as db:
        q = select(DesktopDevice.id).where(
            DesktopDevice.user_id == user_id,
            DesktopDevice.revoked_at.is_(None),
        )
        if device_ids is not None:
            q = q.where(DesktopDevice.id.in_([d for d in device_ids if d]))
        ids = [r[0] for r in (await db.execute(q)).all()]
        if not ids:
            return {"revoked": 0, "tasks_cancelled": 0, "actions_rejected": 0}
        r = await db.execute(
            sa_update(DesktopDevice)
            .where(DesktopDevice.user_id == user_id)
            .where(DesktopDevice.id.in_(ids))
            .where(DesktopDevice.revoked_at.is_(None))
            .values(revoked_at=now, token_jti=None, prev_token_jti=None,
                    prev_jti_expires_at=None)
        )
        revoked = r.rowcount or 0
        t = await db.execute(
            sa_update(DesktopTask)
            .where(DesktopTask.user_id == user_id)
            .where(DesktopTask.device_id.in_(ids))
            .where(DesktopTask.status.notin_(tuple(task_state.TERMINAL)))
            .values(
                status="cancelled", cancelled_at=now, cancel_reason=reason,
                outcome=task_state.OUTCOME_DISCONNECTED, finished_at=now,
                updated_at=now,
            )
        )
        a = await db.execute(
            sa_update(DesktopPendingAction)
            .where(DesktopPendingAction.user_id == user_id)
            .where(DesktopPendingAction.device_id.in_(ids))
            .where(DesktopPendingAction.status == "pending")
            .values(
                status="rejected", decided_at=now, decided_via="revoke",
                result_json=json.dumps({
                    "status": "rejected", "summary": REVOKED_CARD_SUMMARY,
                }),
            )
        )
        await db.commit()
    return {
        "revoked": revoked,
        "tasks_cancelled": t.rowcount or 0,
        "actions_rejected": a.rowcount or 0,
    }


@router.post("/desktop/token/reissue")
async def token_reissue(
    req: ReissueReq, request: Request, current_user=Depends(_optional_user),
) -> Any:
    """A fresh device token, for a Mac that holds BOTH credentials (F-F).

    The device token alone is never enough: a token copied off the Mac must
    not be able to extend itself (T8). The account bearer alone is never
    enough either — it names an account, not a Mac. An expired device token
    is accepted within `REISSUE_GRACE_S`, so a Mac that slept for days
    recovers without pairing again. The replaced jti keeps authenticating
    for `PREV_JTI_GRACE_S` so the socket it opened survives the reconnect.
    """
    await _enforce_rate(request, "desktop_token_reissue", limit=10, window_s=60)
    invalid = _flat("invalid", 401, "Connect this Mac again.")
    if current_user is None:
        return invalid
    account_uid = _uid_of(current_user)
    if not await _connections_on(account_uid):
        return _flat("unavailable", 404, "Mac Connections is disabled.")
    try:
        claims = jwt.decode(
            req.device_token, _signing_key(), algorithms=["HS256"],
            audience=DESKTOP_TOKEN_AUD, issuer=DESKTOP_TOKEN_ISS,
            options={"verify_exp": False},
        )
    except (JWTError, HTTPException):
        return invalid
    if claims.get("scope") != DESKTOP_TOKEN_SCOPE or claims.get("sub") != account_uid:
        return invalid
    try:
        exp = int(claims.get("exp") or 0)
    except (TypeError, ValueError):
        return invalid
    if exp + REISSUE_GRACE_S < time.time():
        return invalid

    device_id, old_jti = claims.get("did"), claims.get("jti")
    now = datetime.utcnow()
    async with async_session_maker() as db:
        row = (await db.execute(
            select(DesktopDevice).where(
                DesktopDevice.user_id == account_uid,
                DesktopDevice.id == device_id,
            )
        )).scalar_one_or_none()
        if row is None:
            return invalid
        if row.revoked_at is not None or not row.token_jti or row.token_jti != old_jti:
            return _flat("revoked", 403, "This Mac was disconnected.")
        if not await _rate_user(f"desktop_reissue_device:{device_id}", 1, 600):
            return _flat("slow_down", 429)
        bundle = _mint_device_token(account_uid, device_id)
        r = await db.execute(
            sa_update(DesktopDevice)
            .where(DesktopDevice.id == device_id)
            .where(DesktopDevice.user_id == account_uid)
            .where(DesktopDevice.revoked_at.is_(None))
            .where(DesktopDevice.token_jti == old_jti)
            .values(
                token_jti=bundle["jti"],
                token_expires_at=now + timedelta(seconds=bundle["expires_in"]),
                prev_token_jti=old_jti,
                prev_jti_expires_at=now + timedelta(seconds=PREV_JTI_GRACE_S),
            )
        )
        await db.commit()
        if (r.rowcount or 0) == 0:
            return _flat("revoked", 403, "This Mac was disconnected.")
    return {
        "access_token": bundle["token"],
        "expires_in": bundle["expires_in"],
        "relay_ws_url": await _relay_ws_url_for(account_uid),
    }


def _jti_accepted(row: DesktopDevice, jti: Optional[str], now: datetime) -> bool:
    """The current jti, or the one a re-issue replaced, inside its grace."""
    if not jti:
        return False
    if row.token_jti and row.token_jti == jti:
        return True
    return bool(
        row.prev_token_jti and row.prev_token_jti == jti
        and row.prev_jti_expires_at is not None
        and row.prev_jti_expires_at > now
    )


async def _close_live_socket(
    user_id: str, device_id: Optional[str], token_jti: Optional[str],
) -> int:
    """Close the relay socket for a revoked device, wherever it is held.

    Best-effort by necessity, which is why it is not the ONLY mechanism:

      * connect-time verification refuses the next connect regardless;
      * the agent's heartbeat response carries a revocation verdict, so a
        socket whose close-push was lost dies within `HEARTBEAT_INTERVAL_S`.

    Three layers because this is the control whose failure the audit calls
    "the single most important thing not to reproduce" — and a single
    best-effort HTTP call is exactly the kind of mechanism that works in
    testing and not during the incident.
    """
    if _platform_db_local() and desktop_bridge.is_connected(user_id):
        return await desktop_bridge.close_for_device(
            user_id, device_id=device_id, token_jti=token_jti,
        )
    try:
        from app.api.tenant_proxy import agent_proxy_info, proxy_to_agent
        async with async_session_maker() as db:
            info = await agent_proxy_info(user_id, db)
        if not info:
            return 0
        agent_url, agent_key = info
        res = await proxy_to_agent(
            agent_url, agent_key, "desktop/internal/revoked", "POST",
            json_body={
                "user_id": user_id,
                "device_id": device_id,
                "token_jti": token_jti,
            },
            timeout=5.0,
        )
        return int((res or {}).get("closed") or 0)
    except Exception as exc:
        # Logged at INFO and not swallowed silently: a revoke whose push
        # failed still revoked the row, and the heartbeat backstop will
        # close the socket — but which of the three layers did the work is
        # exactly what an incident needs to know.
        logger.info(
            "[desktop] revoke push to agent failed user=%s (%s) — row is "
            "revoked, heartbeat backstop will close within %ds",
            user_id[:8], exc, HEARTBEAT_INTERVAL_S,
        )
        return 0


# ═════════════════════════════════════════════════════════════════════════
# Internal: token verification (agent → platform)
# ═════════════════════════════════════════════════════════════════════════
@router.post("/desktop/internal/verify-token")
async def verify_token_route(
    req: VerifyTokenReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Dict[str, Any]:
    """Validate a device token AND its revocation state, in one answer.

    The agent cannot do this locally: `desktop_devices` is PLATFORM_ONLY.
    That is deliberate (see the module header) — it makes the revocation
    check a precondition of authentication rather than a follow-up someone
    can forget, which is the extension's §9.1 defect.

    `reason` is a closed vocabulary so the caller can map it to a close code
    without parsing prose: `invalid` / `expired` / `revoked` / `unknown`
    are all permanent (`4001`/`4003`); anything else is transient and must
    NOT delete the app's token.
    """
    tenant_uid = await _user_for_agent_key(x_agent_key)
    claims = _decode_device_token(req.token)
    if not claims:
        return {"ok": False, "reason": "invalid"}
    if claims.get("sub") != tenant_uid:
        # A token for another tenant presented to this one. Refuse, and log:
        # nothing legitimate produces it.
        logger.warning(
            "[desktop] token/tenant mismatch: token_sub=%s agent_user=%s",
            (claims.get("sub") or "?")[:8], tenant_uid[:8],
        )
        return {"ok": False, "reason": "invalid"}

    if not await _connections_on(tenant_uid):
        return {"ok": False, "reason": "disabled"}
    jti = claims.get("jti")
    device_id = claims.get("did")
    async with async_session_maker() as db:
        row = (await db.execute(
            select(DesktopDevice).where(
                DesktopDevice.user_id == tenant_uid,
                DesktopDevice.id == device_id,
            )
        )).scalar_one_or_none()
    if row is None:
        return {"ok": False, "reason": "unknown"}
    if row.revoked_at is not None:
        return {"ok": False, "reason": "revoked"}
    if not _jti_accepted(row, jti, datetime.utcnow()):
        # A superseded token: the device re-paired or re-issued, and this is
        # the older credential past its grace. `revoked` rather than
        # `invalid` because the user-visible truth is the same — this Mac's
        # access was replaced — and both map to a permanent close.
        return {"ok": False, "reason": "revoked"}
    return {
        "ok": True,
        "user_id": tenant_uid,
        "device_id": row.id,
        "jti": jti,
        "device_name": row.device_name,
        "flavor": row.flavor,
    }


async def _verify_device_token(token: str) -> Dict[str, Any]:
    """Agent-side entry point for authenticating a relay connect.

    Returns `{"ok": True, …}`, or `{"ok": False, "reason": …}` where
    `reason` distinguishes permanent from transient. `unavailable` is the
    transient one and matters a lot: §3.2 says `4001`/`4003` are NEVER
    retried and cause the app to DELETE its token, so a platform blip must
    never produce one. It closes `1001` instead — "routine", which the app
    reconnects through quietly.
    """
    if _platform_db_local():
        # Monolith / platform-served relay: the table is right here.
        claims = _decode_device_token(token)
        if not claims:
            return {"ok": False, "reason": "invalid"}
        uid = claims["sub"]
        if not await _connections_on(uid):
            return {"ok": False, "reason": "disabled"}
        if settings.user_id and uid != settings.user_id:
            return {"ok": False, "reason": "invalid"}
        async with async_session_maker() as db:
            row = (await db.execute(
                select(DesktopDevice).where(
                    DesktopDevice.user_id == uid,
                    DesktopDevice.id == claims.get("did"),
                )
            )).scalar_one_or_none()
        if row is None:
            return {"ok": False, "reason": "unknown"}
        if row.revoked_at is not None:
            return {"ok": False, "reason": "revoked"}
        if not _jti_accepted(row, claims.get("jti"), datetime.utcnow()):
            return {"ok": False, "reason": "revoked"}
        return {
            "ok": True, "user_id": uid, "device_id": row.id,
            "jti": claims.get("jti"), "device_name": row.device_name,
            "flavor": row.flavor,
        }

    from app.services.runtime_identity import get_agent_api_key
    agent_key = (get_agent_api_key() or "").strip()
    platform_url = (getattr(settings, "platform_api_url", "") or "").strip()
    if not agent_key or not platform_url.startswith("http"):
        logger.error(
            "[desktop] cannot verify a device token: agent_api_key or "
            "platform_api_url is not configured",
        )
        return {"ok": False, "reason": "unavailable"}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(
                f"{platform_url.rstrip('/')}/desktop/internal/verify-token",
                json={"token": token},
                headers={"X-Agent-Key": agent_key},
            )
        if resp.status_code >= 500:
            return {"ok": False, "reason": "unavailable"}
        body = resp.json()
        if not isinstance(body, dict):
            return {"ok": False, "reason": "unavailable"}
        if resp.status_code >= 400:
            # 401/403 here means OUR agent key is stale, not that the
            # device is bad. Transient, so the app keeps its token.
            return {"ok": False, "reason": "unavailable"}
        return body
    except Exception as exc:
        logger.warning("[desktop] token verification unreachable: %s", exc)
        return {"ok": False, "reason": "unavailable"}


# ═════════════════════════════════════════════════════════════════════════
# The relay socket (§3) — served by the TENANT AGENT
# ═════════════════════════════════════════════════════════════════════════
@ws_router.websocket("/ws/desktop")
async def ws_desktop(websocket: WebSocket) -> None:
    """§3. Subprotocol auth, close codes for refusals, ping/pong liveness.

    There is deliberately NO `?token=` and NO first-frame `{type:"auth"}`
    (§3.1, §9.5): the extension accepts both (`extension.py:657-668`), and a
    token in a URL reaches every proxy log between here and the tenant. The
    app refuses any relay URL carrying a query string at all, so those paths
    are unreachable from it anyway — accepting them would only serve
    something that is not this client.
    """
    # Read the subprotocol list BEFORE accept so we can echo it back (§3.1).
    sub_token: Optional[str] = None
    selected: Optional[str] = None
    subprotocols = list(websocket.scope.get("subprotocols") or [])
    # TRIM each token. `URLSessionWebSocketTask` joins its subprotocol array
    # with ", " and several parsers split on "," alone, so the second token
    # arrives as " bearer.eyJ…" with a LEADING SPACE — measured against
    # `NWProtocolWebSocket`. A server that compares untrimmed rejects every
    # real client (§3.1).
    cleaned = [sp.strip() for sp in subprotocols if isinstance(sp, str)]
    if "toup.auth.v1" in cleaned:
        for sp in cleaned:
            if sp.startswith("bearer."):
                sub_token = sp[len("bearer."):].strip()
                selected = "toup.auth.v1"
                break

    # §3.2: ACCEPT the handshake, then close with a code. A WebSocket
    # upgrade refused with an HTTP status carries no close code, and a
    # native client cannot tell that apart from a dropped network — so it
    # would retry a dead token for the life of the laptop.
    if selected:
        await websocket.accept(subprotocol=selected)
    else:
        # Not echoing `toup.auth.v1` is how the app learns the server did
        # not understand the auth channel: it closes, backs off, retries.
        await websocket.accept()
        await websocket.close(code=4001, reason="auth subprotocol required")
        return

    verdict = await _verify_device_token(sub_token or "")
    if not verdict.get("ok"):
        reason = str(verdict.get("reason") or "invalid")
        if reason == "revoked":
            code = 4003                      # deletes the token, stops
        elif reason in ("invalid", "expired", "unknown"):
            code = 4001                      # deletes the token, stops
        else:
            # `unavailable` — the platform could not answer. NEVER 4001:
            # the app would delete a perfectly good token over a blip.
            code = 1001
        await websocket.close(code=code, reason=reason)
        return

    user_id = verdict["user_id"]
    device_id = verdict.get("device_id")
    token_jti = verdict.get("jti")

    if not settings.desktop_relay_enabled:
        await websocket.close(code=1001, reason="disabled")
        return

    ep = await desktop_bridge.register(
        user_id, websocket, device_id=device_id, token_jti=token_jti,
    )
    await _post_heartbeat(user_id, device_id, token_jti, online=True)

    async def _heartbeat_loop() -> None:
        """Keeps `last_seen_at` fresh AND carries the revocation backstop.

        The response to a heartbeat says whether this device is still
        allowed. So a revoke whose close-push was lost still shuts the
        socket within one interval, without the agent needing an inbound
        route to be reachable.
        """
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
                if desktop_bridge.seconds_since_inbound(ep) > INBOUND_SILENCE_S:
                    # Half-open: the Mac pings every 25 s and has said
                    # nothing for a minute. Posting `online` here is what
                    # kept a sleeping Mac reading Online indefinitely (F-I).
                    # 1001 so the Mac reconnects quietly if it is alive.
                    logger.info(
                        "[desktop] device %s silent for %ds — closing",
                        (device_id or "?")[:8], INBOUND_SILENCE_S,
                    )
                    await desktop_bridge.unregister(user_id, ep)
                    await _post_heartbeat(
                        user_id, device_id, token_jti, online=False,
                    )
                    try:
                        await websocket.close(code=1001, reason="silent")
                    except Exception:
                        pass
                    return
                res = await _post_heartbeat(
                    user_id, device_id, token_jti, online=True,
                )
                if isinstance(res, dict) and res.get("disabled") is True:
                    await desktop_bridge.unregister(user_id, ep)
                    await websocket.close(code=1001, reason="disabled")
                    return
                if isinstance(res, dict) and res.get("revoked") is True:
                    logger.info(
                        "[desktop] heartbeat reports device %s revoked — "
                        "closing socket", (device_id or "?")[:8],
                    )
                    await desktop_bridge.close_for_device(
                        user_id, device_id=device_id, token_jti=token_jti,
                    )
                    return
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("[desktop] heartbeat loop exited: %s", exc)

    hb = asyncio.create_task(_heartbeat_loop())

    try:
        while True:
            raw = await websocket.receive_text()
            desktop_bridge.mark_inbound(ep)
            # §5: frames over 1 MiB are dropped and counted WITHOUT being
            # parsed. Checked before `json.loads` so a hostile frame cannot
            # cost a megabyte of parsing.
            if len(raw) > desktop_bridge.MAX_INBOUND_FRAME_BYTES:
                ep.counters["oversize_frames"] += 1
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                ep.counters["bad_json"] += 1
                continue
            if not isinstance(msg, dict):
                ep.counters["bad_json"] += 1
                continue

            mtype = msg.get("type")
            if mtype == "ping":
                # §3.3: answer EVERY ping. This is the only detector for a
                # half-open socket — after a Mac sleeps the socket commonly
                # reports OPEN and `send` keeps succeeding into a void.
                await websocket.send_json({"type": "pong", "ts": msg.get("ts")})
            elif mtype == "pong":
                pass
            elif mtype == "hello":
                desktop_bridge.handle_hello(ep, msg)
            elif mtype == "tools":
                desktop_bridge.handle_tools(ep, msg)
            else:
                # result / event / anything newer.
                desktop_bridge.deliver_inbound(user_id, msg)
    except Exception as exc:
        logger.info("[desktop] ws closed user=%s: %s", user_id[:8], exc)
    finally:
        hb.cancel()
        await desktop_bridge.unregister(user_id, ep)
        await _post_heartbeat(user_id, device_id, token_jti, online=False)


async def _post_heartbeat(
    user_id: str,
    device_id: Optional[str],
    token_jti: Optional[str],
    *,
    online: bool,
) -> Optional[Dict[str, Any]]:
    """Report presence to the platform. Returns the platform's verdict.

    Local short-circuit when the table is in this process, so a monolith or
    a platform-served relay does not make an HTTP call to itself.
    """
    if _platform_db_local():
        return await _touch_last_seen(user_id, device_id, token_jti, online=online)
    from app.services.runtime_identity import get_agent_api_key
    agent_key = (get_agent_api_key() or "").strip()
    platform_url = (getattr(settings, "platform_api_url", "") or "").strip()
    if not agent_key or not platform_url.startswith("http"):
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{platform_url.rstrip('/')}/desktop/heartbeat",
                json={
                    "user_id": user_id,
                    "device_id": device_id,
                    "token_jti": token_jti,
                    "online": online,
                },
                headers={"X-Agent-Key": agent_key},
            )
        body = resp.json()
        return body if isinstance(body, dict) else None
    except Exception as exc:
        logger.debug("[desktop] heartbeat post failed: %s", exc)
        return None


async def _touch_last_seen(
    user_id: str,
    device_id: Optional[str],
    token_jti: Optional[str],
    *,
    online: bool,
) -> Dict[str, Any]:
    """Write presence and answer whether the device is still allowed.

    Deliberately does NOT materialise a missing row. `extension.py:341-356`
    does, to heal users who paired before its table existed — but there is
    no such population here, and a heartbeat that can CREATE a device row is
    a heartbeat that can un-revoke a Mac. The revoked case must stay
    reportable, so a missing or revoked row answers `revoked: True` and the
    caller closes the socket.
    """
    if online and not await _connections_on(user_id):
        return {"ok": False, "revoked": False, "disabled": True}
    now = datetime.utcnow() if online else (
        datetime.utcnow() - timedelta(seconds=DEVICE_ONLINE_WINDOW_S + 5)
    )
    try:
        async with async_session_maker() as db:
            q = (
                sa_update(DesktopDevice)
                .where(DesktopDevice.user_id == user_id)
                .where(DesktopDevice.revoked_at.is_(None))
                .values(last_seen_at=now)
            )
            if device_id:
                q = q.where(DesktopDevice.id == device_id)
            if token_jti:
                q = q.where(or_(
                    DesktopDevice.token_jti == token_jti,
                    and_(
                        DesktopDevice.prev_token_jti == token_jti,
                        DesktopDevice.prev_jti_expires_at > datetime.utcnow(),
                    ),
                ))
            r = await db.execute(q)
            await db.commit()
            touched = r.rowcount or 0
    except ProgrammingError:
        return {"ok": True, "revoked": False}
    except Exception as exc:
        logger.debug("[desktop] _touch_last_seen failed: %s", exc)
        # Unknown, not revoked. Failing closed here would drop a live
        # socket on a transient DB error.
        return {"ok": False, "revoked": False}
    return {"ok": True, "revoked": touched == 0}


@router.post("/desktop/heartbeat")
async def desktop_heartbeat(
    req: HeartbeatReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Dict[str, Any]:
    tenant_uid = await _user_for_agent_key(x_agent_key)
    _require_agent_bound_user(tenant_uid, req.user_id)
    return await _touch_last_seen(
        tenant_uid, req.device_id, req.token_jti, online=req.online,
    )


@router.post("/desktop/internal/revoked")
async def revoked_push(
    req: RevokedPushReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Dict[str, Any]:
    """Platform → agent: close the live socket for a revoked device.

    Served by the AGENT. Authenticated with the same `X-Agent-Key` the
    platform already uses for every other tenant call, so no new secret and
    no inbound credential.
    """
    tenant_uid = await _user_for_agent_key(x_agent_key)
    _require_agent_bound_user(tenant_uid, req.user_id)
    closed = await desktop_bridge.close_for_device(
        tenant_uid, device_id=req.device_id, token_jti=req.token_jti,
    )
    return {"ok": True, "closed": closed}


# ═════════════════════════════════════════════════════════════════════════
# Consent (§6) — the connector pending-action invariants, on a local action
# ═════════════════════════════════════════════════════════════════════════
def _parse_result(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return {
        "status": str(parsed.get("status") or ""),
        "summary": str(parsed.get("summary") or ""),
    }


def _action_card(row: DesktopPendingAction) -> Dict[str, Any]:
    """The `pending_action` frame body (§6).

    Shaped so every client that already renders a connector confirm card
    renders this one: `tool_executor.py:1438-1470` broadcasts
    `{"type": "pending_action", **card}` with exactly these keys.
    `connector_id` is `"desktop"` — the clients use it to label the card, so
    it has to be a string they can show, and "desktop" is what the user
    calls the thing. The fields after `status` are additive
    (CONNECTIONS.md §4.3) and let the phone draw every state of the card.
    """
    try:
        payload = json.loads(row.payload_json) if row.payload_json else {}
    except (ValueError, TypeError):
        payload = {}
    return {
        "action_id": row.id,
        "connector_id": "desktop",
        "tool_name": row.tool_name,
        "summary": row.reason or "",
        "payload": payload,
        "expires_at": row.expires_at.isoformat() + "Z",
        "status": row.status,
        "device_id": row.device_id,
        "remote_task_id": row.remote_task_id,
        "result": _parse_result(row.result_json),
        "mac_prompt": row.mac_prompt_at is not None,
        "decided_via": row.decided_via,
        "decided_at": _iso(row.decided_at),
    }


# ── Background hand-offs ─────────────────────────────────────────────────
_INFLIGHT: set = set()


def _spawn_tracked(coro, *, name: str) -> "asyncio.Task":
    from app.services.background_tasks import spawn

    task = spawn(coro, name=name)
    _INFLIGHT.add(task)
    task.add_done_callback(_INFLIGHT.discard)
    return task


async def wait_for_handoffs(timeout: float = 10.0) -> None:
    """Wait for every hand-off this module started in this process.

    Approve and dispatch answer before the Mac does, so an outcome arrives
    after the response. Tests use this to observe it; nothing in the request
    path waits on it.
    """
    deadline = time.monotonic() + timeout
    while _INFLIGHT and time.monotonic() < deadline:
        await asyncio.wait(
            list(_INFLIGHT), timeout=max(0.01, deadline - time.monotonic()),
        )


async def _call_agent(
    user_id: str, path: str, body: Dict[str, Any], *, timeout: float,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """POST to the tenant agent and hand back (status, json).

    Not `proxy_to_agent`, which turns every 4xx into an HTTPException: the
    callers here act on the agent's own `error` code (`mac_offline` vs
    `disabled`), so they need the status and the body as they came.
    `(None, {})` means the agent could not be reached at all.
    """
    try:
        from app.api.tenant_proxy import agent_proxy_info
        from app.services.agent_http import get_agent_http_client

        async with async_session_maker() as db:
            info = await agent_proxy_info(user_id, db)
        if not info:
            return None, {}
        agent_url, agent_key = info
        resp = await get_agent_http_client().post(
            f"{agent_url.rstrip('/')}/api/{path.lstrip('/')}",
            json=body, headers={"X-Agent-Key": agent_key}, timeout=timeout,
        )
        try:
            data = resp.json()
        except Exception:
            data = {}
        return resp.status_code, data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.info("[desktop] agent call %s failed user=%s: %r",
                    path, user_id[:8], exc)
        return None, {}


async def _post_to_platform(path: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Agent → platform, on the `_post_heartbeat` pattern."""
    from app.services.runtime_identity import get_agent_api_key
    agent_key = (get_agent_api_key() or "").strip()
    platform_url = (getattr(settings, "platform_api_url", "") or "").strip()
    if not agent_key or not platform_url.startswith("http"):
        logger.warning("[desktop] cannot post %s: platform not configured", path)
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{platform_url.rstrip('/')}/{path.lstrip('/')}",
                json=body, headers={"X-Agent-Key": agent_key},
            )
        out = resp.json()
        return out if isinstance(out, dict) else None
    except Exception as exc:
        logger.info("[desktop] post %s failed: %r", path, exc)
        return None


# ── Remote-task state (CONNECTIONS.md §8.3) ──────────────────────────────
def _online(dev: Optional[DesktopDevice], now: datetime) -> bool:
    return bool(
        dev is not None and dev.revoked_at is None and dev.last_seen_at
        and dev.last_seen_at >= now - timedelta(seconds=DEVICE_ONLINE_WINDOW_S)
    )


async def _rederive(
    db, task: DesktopTask, now: datetime,
    dev: Optional[DesktopDevice] = None,
) -> Tuple[DesktopTask, List[DesktopPendingAction], Optional[DesktopDevice]]:
    """Recompute one task's status and persist a change with a guarded write.

    Every writer ends here. The write is guarded on the task still being
    unfinished, so a terminal status never changes (§8.2): a late result
    updates its own action row and nothing else.
    """
    actions = list((await db.execute(
        select(DesktopPendingAction).where(
            DesktopPendingAction.user_id == task.user_id,
            DesktopPendingAction.remote_task_id == task.id,
        ).order_by(DesktopPendingAction.created_at)
    )).scalars().all())
    if dev is None:
        dev = (await db.execute(
            select(DesktopDevice).where(
                DesktopDevice.id == task.device_id,
                DesktopDevice.user_id == task.user_id,
            )
        )).scalar_one_or_none()
    facts = task_state.DeviceFacts(
        name=dev.device_name if dev else "Your Mac",
        online=_online(dev, now),
        revoked=dev is None or dev.revoked_at is not None,
    )
    status, outcome = task_state.derive(
        task_state.TaskFacts(
            status=task.status, turn_state=task.turn_state,
            created_at=task.created_at, outcome=task.outcome,
            cancelled_at=task.cancelled_at, cancel_reason=task.cancel_reason,
        ),
        [task_state.action_facts_from_row(a) for a in actions],
        facts, now,
    )
    if task.status not in task_state.TERMINAL and (
        status != task.status or outcome != task.outcome
    ):
        values: Dict[str, Any] = {
            "status": status, "outcome": outcome, "updated_at": now,
        }
        if status in task_state.TERMINAL:
            values["finished_at"] = now
        await db.execute(
            sa_update(DesktopTask)
            .where(DesktopTask.id == task.id)
            .where(DesktopTask.status.notin_(tuple(task_state.TERMINAL)))
            .values(**values)
        )
        await db.commit()
        await db.refresh(task)
    return task, actions, dev


async def _rederive_by_id(task_id: Optional[str]) -> None:
    if not task_id:
        return
    try:
        async with async_session_maker() as db:
            task = await db.get(DesktopTask, task_id)
            if task is not None:
                await _rederive(db, task, datetime.utcnow())
    except Exception as exc:
        logger.warning("[desktop] re-derive of task %s failed: %s", task_id, exc)


def _task_json(
    task: DesktopTask, actions: List[DesktopPendingAction],
    dev: Optional[DesktopDevice], now: datetime,
) -> Dict[str, Any]:
    return {
        "id": task.id,
        "device_id": task.device_id,
        "device_name": dev.device_name if dev else None,
        "title": task.title,
        "status": task.status,
        "outcome": task.outcome,
        "conversation_id": task.conversation_id,
        "created_at": _iso(task.created_at),
        "updated_at": _iso(task.updated_at),
        "finished_at": _iso(task.finished_at),
        "actions": [{
            "action_id": a.id,
            "tool_name": a.tool_name,
            "summary": a.reason or "",
            "status": a.status,
            "result": _parse_result(a.result_json),
            "expires_at": _iso(a.expires_at),
            "mac_prompt": a.mac_prompt_at is not None,
        } for a in actions],
        "mac_online": _online(dev, now),
        "mac_last_seen_at": _iso(dev.last_seen_at) if dev else None,
    }


async def _task_view(db, task: DesktopTask, now: datetime) -> Dict[str, Any]:
    task, actions, dev = await _rederive(db, task, now)
    return _task_json(task, actions, dev, now)


@router.post("/desktop/internal/stage-action")
async def stage_action(
    req: StageActionReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Dict[str, Any]:
    """Agent → platform: stage a local action for the user's approval.

    Called by the `desktop` skill instead of dispatching, for every tool
    that mutates, executes or drives the Mac. The row lands on the PLATFORM
    DB because that is the only database both clients talk to — the same
    reason `ConnectorPendingAction` lives there
    (`connectors.py:443-449`) — and because an agent that has since been
    recycled must not be able to strand an approval the user already gave.

    A card staged under a remote task must belong to this tenant's task and
    to that task's Mac (CONNECTIONS.md M4): a card for Mac A under a task
    pinned to Mac B would be approved on the phone as something it is not.
    """
    tenant_uid = await _user_for_agent_key(x_agent_key)
    _require_agent_bound_user(tenant_uid, req.user_id)
    if not await _connections_on(tenant_uid):
        raise HTTPException(404, "Mac Connections is disabled.")

    if req.remote_task_id:
        async with async_session_maker() as db:
            task = await db.get(DesktopTask, req.remote_task_id)
        if task is None or task.user_id != tenant_uid:
            raise HTTPException(409, "that task is not this account's")
        if task.device_id != req.device_id:
            raise HTTPException(409, "that task is pinned to a different Mac")
        if task.status in task_state.TERMINAL:
            raise HTTPException(409, "that task has already finished")

    now = datetime.utcnow()
    action_id = str(uuid.uuid4())
    # The relay idempotency key, minted ONCE here and reused verbatim on
    # every dispatch of this row (§5.3). See the column's comment: a fresh
    # id on retry would defeat the device's result cache and could run a
    # destructive command twice.
    task_id = uuid.uuid4().hex
    reason = " ".join((req.reason or "").split())[:240]

    async with async_session_maker() as db:
        db.add(DesktopPendingAction(
            id=action_id,
            user_id=tenant_uid,
            device_id=req.device_id[:36],
            tool_name=req.tool_name[:128],
            payload_json=json.dumps(req.payload, sort_keys=True, default=str),
            reason=reason or None,
            status="pending",
            channel=(req.channel or "desktop")[:32],
            conversation_id=req.conversation_id,
            task_id=task_id,
            remote_task_id=req.remote_task_id,
            created_at=now,
            expires_at=now + timedelta(seconds=PENDING_ACTION_TTL_S),
        ))
        await db.commit()
        row = (await db.execute(
            select(DesktopPendingAction).where(DesktopPendingAction.id == action_id)
        )).scalar_one()
        card = _action_card(row)
    await _rederive_by_id(req.remote_task_id)

    # Live render, best-effort and never able to fail the staging — the
    # durable path is the row, which the clients list on reload. Same
    # split, and the same rule, as `tool_executor.py:1461-1470`.
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(tenant_uid, {"type": "pending_action", **card})
    except Exception as exc:
        logger.warning(
            "[desktop] pending_action broadcast failed for %s: %s "
            "(row is persisted)", action_id, exc,
        )
    return {"ok": True, **card}


async def _expire_lazily(db, rows: List[DesktopPendingAction], now: datetime) -> None:
    """Lazy expiry on read (`connector_pending_actions.py:405-416`): a card
    the user is looking at RIGHT NOW must never render as actionable past
    its deadline."""
    stale = [r for r in rows if r.status == "pending" and r.expires_at <= now]
    if not stale:
        return
    await db.execute(
        sa_update(DesktopPendingAction)
        .where(DesktopPendingAction.id.in_([r.id for r in stale]))
        .where(DesktopPendingAction.status == "pending")
        .values(status="expired", decided_at=now)
    )
    await db.commit()
    for r in stale:
        r.status = "expired"


@router.get("/desktop/pending-actions")
async def list_pending_actions(
    current_user=Depends(get_current_user),
    action_status: str = Query("pending", max_length=16),
    device_id: Optional[str] = Query(None, max_length=36),
) -> Dict[str, Any]:
    user_id = _uid_of(current_user)
    now = datetime.utcnow()
    async with async_session_maker() as db:
        q = select(DesktopPendingAction).where(
            DesktopPendingAction.user_id == user_id,
        ).order_by(DesktopPendingAction.created_at.desc()).limit(100)
        if action_status != "all":
            q = q.where(DesktopPendingAction.status == action_status)
        if device_id:
            q = q.where(DesktopPendingAction.device_id == device_id)
        rows = list((await db.execute(q)).scalars().all())
        await _expire_lazily(db, rows, now)
        return {"actions": [_action_card(r) for r in rows]}


@router.get("/desktop/pending-actions/{action_id}")
async def get_pending_action(
    action_id: str, current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """One card, for its owner only. Another account's id is a plain 404."""
    user_id = _uid_of(current_user)
    now = datetime.utcnow()
    async with async_session_maker() as db:
        row = (await db.execute(
            select(DesktopPendingAction).where(
                DesktopPendingAction.id == action_id,
                DesktopPendingAction.user_id == user_id,
            )
        )).scalar_one_or_none()
        if row is None:
            raise HTTPException(404, "That request is no longer available.")
        await _expire_lazily(db, [row], now)
        dev = (await db.execute(
            select(DesktopDevice).where(
                DesktopDevice.id == row.device_id,
                DesktopDevice.user_id == user_id,
            )
        )).scalar_one_or_none()
        return {
            **_action_card(row),
            "device_name": dev.device_name if dev else None,
            "device_online": _online(dev, now),
        }


@router.post("/desktop/pending-actions/{action_id}/reject")
async def reject_action(
    action_id: str, request: Request, current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    user_id = _uid_of(current_user)
    now = datetime.utcnow()
    async with async_session_maker() as db:
        r = await db.execute(
            sa_update(DesktopPendingAction)
            .where(DesktopPendingAction.id == action_id)
            .where(DesktopPendingAction.user_id == user_id)
            .where(DesktopPendingAction.status == "pending")
            .values(
                status="rejected", decided_at=now,
                decided_via=_client_via(request),
                decided_session_jti=_session_jti(request),
            )
        )
        await db.commit()
        if (r.rowcount or 0) == 0:
            raise HTTPException(409, "This was already decided or has expired.")
        row = await db.get(DesktopPendingAction, action_id)
        card = _action_card(row)
    await _rederive_by_id(card.get("remote_task_id"))
    # §5.4: `denied` means the user refused, and the agent should stop
    # asking. Nothing is relayed to the device — there is nothing to say.
    return {"ok": True, **card}


@router.post("/desktop/pending-actions/{action_id}/approve")
async def approve_action(
    action_id: str, request: Request, current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Claim one staged local action and hand it to the Mac.

    NOTE the request body: there isn't one. The connector flow accepts
    edited arguments through an `_editable_fields` allowlist
    (`connector_pending_actions.py:206-315`); this one accepts nothing at
    all. `tool_name` and `payload_json` come off the row, so the endpoint
    cannot be talked into running a different tool, or the same tool with
    different arguments, than the card showed — which for a tool whose
    argument is a shell command is the difference between a confirmation
    and a formality.

    ASYNCHRONOUS (CONNECTIONS.md §4.3): the claim, a hand-off inside
    `HANDOFF_BUDGET_S`, and the answer — without waiting for the Mac, whose
    own confirmation panel may take two minutes. The outcome lands on the
    row through `internal/action-result`, and clients re-read the card.
    """
    user_id = _uid_of(current_user)
    if not await _connections_on(user_id):
        return _flat("unavailable", 404, "Mac Connections isn't available for this account.")
    now = datetime.utcnow()

    async with async_session_maker() as db:
        row = (await db.execute(
            select(DesktopPendingAction).where(
                DesktopPendingAction.id == action_id,
                DesktopPendingAction.user_id == user_id,
            )
        )).scalar_one_or_none()
        if row is None:
            raise HTTPException(404, "That request is no longer available.")
        if row.status != "pending":
            raise HTTPException(
                409,
                "This is already running on your Mac."
                if row.status in ("approved", "dispatched")
                else f"This was already {row.status}.",
            )
        if row.expires_at <= now:
            await db.execute(
                sa_update(DesktopPendingAction)
                .where(DesktopPendingAction.id == row.id)
                .where(DesktopPendingAction.status == "pending")
                .values(status="expired", decided_at=now)
            )
            await db.commit()
            await _rederive_by_id(row.remote_task_id)
            raise HTTPException(
                410,
                "This request expired before it was confirmed. Ask your "
                "agent to prepare it again.",
            )

        # ── Device must be ONLINE at tap time (§6, and §4.2's safe
        # default). Checked BEFORE the claim, so refusing leaves the row
        # `pending` and the user can tap again once the Mac wakes. Claiming
        # first would burn the row: `approved` is terminal-for-claiming by
        # design, so an offline tap would produce an action that can never
        # run and can never be re-approved.
        #
        # This is what stops "an approved local command fires hours later
        # when the laptop wakes" — audit risk #5, a failure mode with no
        # precedent to copy.
        dev = (await db.execute(
            select(DesktopDevice).where(
                DesktopDevice.id == row.device_id,
                DesktopDevice.user_id == user_id,
            )
        )).scalar_one_or_none()
        if dev is None or dev.revoked_at is not None:
            raise HTTPException(
                409, "That Mac is no longer connected to your account.",
            )
        if not _online(dev, now):
            raise HTTPException(
                409,
                f"{dev.device_name} is offline, so this can't run right now. "
                "Open Toup on that Mac and confirm again.",
            )

        # ── The claim. One statement, guarded on `status = 'pending'`.
        # Whoever's UPDATE lands first owns it; everyone else sees rowcount
        # 0 and gets a 409. This is the double-tap defence
        # (`connectors.py:420-423`) — do not refactor it into a
        # read-then-write.
        claimed = await db.execute(
            sa_update(DesktopPendingAction)
            .where(DesktopPendingAction.id == row.id)
            .where(DesktopPendingAction.status == "pending")
            .values(
                status="approved", decided_at=now,
                decided_via=_client_via(request),
                decided_session_jti=_session_jti(request),
            )
        )
        await db.commit()
        if (claimed.rowcount or 0) == 0:
            raise HTTPException(409, "This was already decided.")

        try:
            payload = json.loads(row.payload_json) if row.payload_json else {}
        except (ValueError, TypeError):
            payload = {}
        tool_name, device_id, task_id = row.tool_name, row.device_id, row.task_id
        reason, remote_task_id = row.reason, row.remote_task_id

    # ── Relay. The device re-validates EVERYTHING against its own grants
    # regardless of this approval and may answer `denied` or `unavailable`:
    # the server's approval is a PRECONDITION, never an authorisation
    # (§6, recommendation 10). A relay that trusted "the server says the
    # user approved this" would make the server's compromise equal to the
    # Mac's compromise.
    await _hand_off_approved(
        user_id=user_id, action_id=action_id, device_id=device_id,
        tool_name=tool_name, task_id=task_id, payload=payload, reason=reason,
        remote_task_id=remote_task_id,
    )
    await _rederive_by_id(remote_task_id)
    async with async_session_maker() as db:
        fresh = await db.get(DesktopPendingAction, action_id)
        card = _action_card(fresh)
    return {"ok": True, **card}


async def _hand_off_approved(
    *, user_id: str, action_id: str, device_id: str, tool_name: str,
    task_id: str, payload: Dict[str, Any], reason: Optional[str],
    remote_task_id: Optional[str],
) -> None:
    """Give the approved action to whichever process holds THAT Mac's socket.

    In-process when this process holds it (monolith, tests); otherwise the
    tenant agent, which answers 202 at once and dispatches in the
    background. Either way nothing here waits for the Mac.
    """
    if _platform_db_local() and desktop_bridge.is_device_connected(user_id, device_id):
        _spawn_tracked(
            _run_approved_on_device(
                user_id=user_id, action_id=action_id, device_id=device_id,
                tool_name=tool_name, task_id=task_id, payload=payload,
                reason=reason, remote_task_id=remote_task_id,
            ),
            name=f"desktop-approved-{action_id[:8]}",
        )
        return
    status, _ = await _call_agent(
        user_id, "desktop/internal/dispatch-approved",
        {
            "user_id": user_id, "action_id": action_id,
            "device_id": device_id, "tool_name": tool_name,
            "task_id": task_id, "payload": payload, "reason": reason,
            "remote_task_id": remote_task_id,
        },
        timeout=HANDOFF_BUDGET_S,
    )
    if status is None or not (200 <= status < 300):
        await _record_action_result(
            action_id, "failed", "Your Mac could not be reached.",
            user_id=user_id,
        )


#: action ids this process has accepted for dispatch. A retried hand-off of
#: the same approval is answered without sending the Mac a second task.
_DISPATCHED: "Dict[str, float]" = {}
_DISPATCHED_CAP = 1024


@router.post("/desktop/internal/dispatch-approved")
async def dispatch_approved_route(
    req: DispatchApprovedReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Any:
    """Platform → agent: run an approved action on exactly the Mac it names.

    Served by the AGENT, which is the process holding the socket. Answers
    202 at once: the Mac's own ask can take two minutes, and the platform
    must not hold a request open for it (F-B, F-G).
    """
    tenant_uid = await _user_for_agent_key(x_agent_key)
    _require_agent_bound_user(tenant_uid, req.user_id)
    if not settings.desktop_relay_enabled:
        return _flat("disabled", 409, "Mac Connections is disabled.")
    if desktop_bridge.is_task_cancelled(req.remote_task_id):
        return _flat("cancelled", 409, "The user cancelled this task.")
    if req.action_id in _DISPATCHED:
        return JSONResponse({"accepted": True, "duplicate": True}, status_code=202)
    _DISPATCHED[req.action_id] = time.time()
    while len(_DISPATCHED) > _DISPATCHED_CAP:
        _DISPATCHED.pop(next(iter(_DISPATCHED)))
    _spawn_tracked(
        _run_approved_on_device(
            user_id=tenant_uid, action_id=req.action_id,
            device_id=req.device_id, tool_name=req.tool_name,
            task_id=req.task_id, payload=req.payload, reason=req.reason,
            remote_task_id=req.remote_task_id,
        ),
        name=f"desktop-approved-{req.action_id[:8]}",
    )
    return JSONResponse({"accepted": True}, status_code=202)


def _approved_timeout_s(tool_name: str) -> float:
    """The tool's own deadline plus the Mac's two-minute ask (F-G).

    It used to be a flat 55 s, which cut off someone who WAS at the Mac
    half way through the panel that promises them two minutes.
    """
    from app.agent.skills.builtins.desktop.skill import _TIMEOUTS

    return float(_TIMEOUTS.get(tool_name, 30.0)) + MAC_ASK_WINDOW_S


async def _run_approved_on_device(
    *, user_id: str, action_id: str, device_id: str, tool_name: str,
    task_id: str, payload: Dict[str, Any], reason: Optional[str],
    remote_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not settings.desktop_relay_enabled or desktop_bridge.is_task_cancelled(remote_task_id):
        summary = "Mac access was disabled or this task was cancelled; nothing ran."
        await _record_action_result(action_id, "failed", summary, user_id=user_id)
        return {"status": "failed", "summary": summary}
    desktop_bridge.track_relay_id(user_id, task_id, action_id, remote_task_id)
    await _record_action_dispatched(action_id, user_id=user_id)
    try:
        res = await desktop_bridge.dispatch(
            user_id, tool_name, payload,
            timeout_s=_approved_timeout_s(tool_name),
            task_id=task_id, reason=reason, device_id=device_id,
        )
    except desktop_bridge.DesktopDenied as exc:
        # The device refused work the server approved. Expected, and the
        # whole point of double enforcement.
        summary = exc.summary or "Your Mac refused this."
        await _record_action_result(
            action_id, "denied", summary, user_id=user_id,
        )
        return {"status": "denied", "summary": summary}
    except desktop_bridge.DesktopUnavailable:
        summary = "Your Mac disconnected before this could run."
        await _record_action_result(action_id, "failed", summary, user_id=user_id)
        return {"status": "failed", "summary": summary}
    except desktop_bridge.DesktopError as exc:
        summary = exc.summary or exc.status
        await _record_action_result(action_id, "failed", summary, user_id=user_id)
        return {"status": "failed", "summary": summary}
    except asyncio.TimeoutError:
        summary = "Your Mac did not answer."
        await _record_action_result(action_id, "failed", summary, user_id=user_id)
        return {"status": "failed", "summary": summary}

    summary = res.get("summary", "")
    await _record_action_result(action_id, "executed", summary, user_id=user_id)
    return {"status": "executed", "summary": summary}


async def _record_action_dispatched(
    action_id: str, *, user_id: Optional[str] = None,
) -> None:
    if _platform_db_local():
        await _apply_action_result(action_id, "dispatched", "", user_id=user_id)
        return
    await _post_to_platform("desktop/internal/action-result", {
        "user_id": user_id, "action_id": action_id,
        "status": "dispatched", "summary": "",
    })


async def _record_action_result(
    action_id: str, status: str, summary: str, *,
    user_id: Optional[str] = None,
) -> None:
    """Record the outcome. The device's SUMMARY only — never its `data`.

    §8 and ARCHITECTURE.md §5.8: the audit records THAT a tool ran — tool,
    target, decision, exit status, byte counts — and never contents. The
    payload the device returns is for the model; this column is the trail a
    person reads.

    Where the platform DB is in another process — the ordinary production
    case, since this runs on the tenant agent — the result is POSTed to
    `internal/action-result`. It used to return early there, so a row
    approved in production stayed `approved` forever (F-B).
    """
    if _platform_db_local():
        await _apply_action_result(action_id, status, summary, user_id=user_id)
        return
    await _post_to_platform("desktop/internal/action-result", {
        "user_id": user_id, "action_id": action_id,
        "status": status, "summary": " ".join((summary or "").split())[:400],
    })


async def _apply_action_result(
    action_id: str, status: str, summary: str, *,
    user_id: Optional[str] = None, mac_prompt: bool = False,
) -> Optional[str]:
    """The guarded transitions (CONNECTIONS.md §4.5), then the task re-derive.

    approved → dispatched, and approved|dispatched → executed|failed. A
    finished row never moves again, whatever arrives late. `denied` is
    stored as `failed` with `result.status = "denied"`.
    """
    now = datetime.utcnow()
    try:
        async with async_session_maker() as db:
            base = sa_update(DesktopPendingAction).where(
                DesktopPendingAction.id == action_id,
            )
            if user_id:
                base = base.where(DesktopPendingAction.user_id == user_id)
            if status == "dispatched":
                await db.execute(
                    base.where(DesktopPendingAction.status == "approved")
                    .values(status="dispatched", dispatched_at=now)
                )
                if mac_prompt:
                    await db.execute(
                        base.where(DesktopPendingAction.status == "dispatched")
                        .where(DesktopPendingAction.mac_prompt_at.is_(None))
                        .values(mac_prompt_at=now)
                    )
            elif status in ("executed", "failed", "denied"):
                stored = "failed" if status == "denied" else status
                await db.execute(
                    base.where(DesktopPendingAction.status.in_(
                        ("approved", "dispatched"),
                    ))
                    .values(
                        status=stored,
                        result_json=json.dumps({
                            "status": status,
                            "summary": " ".join((summary or "").split())[:400],
                        }),
                    )
                )
            else:
                return None
            await db.commit()
            row = await db.get(DesktopPendingAction, action_id)
            current, remote_task_id = (
                (row.status, row.remote_task_id) if row else (None, None)
            )
    except Exception as exc:
        logger.warning("[desktop] result stamp failed %s: %s", action_id, exc)
        return None
    await _rederive_by_id(remote_task_id)
    return current


@router.post("/desktop/internal/action-result")
async def action_result_route(
    req: ActionResultReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Dict[str, Any]:
    """Agent → platform: what happened to an approved action on the Mac."""
    tenant_uid = await _user_for_agent_key(x_agent_key)
    _require_agent_bound_user(tenant_uid, req.user_id)
    if req.status not in ("dispatched", "executed", "failed", "denied"):
        raise HTTPException(400, "unknown status")
    status = await _apply_action_result(
        req.action_id, req.status, req.summary,
        user_id=tenant_uid, mac_prompt=req.mac_prompt,
    )
    return {"ok": True, "status": status}


async def _on_approval_event(
    user_id: str, action_id: str, remote_task_id: Optional[str], kind: str,
) -> None:
    """The Mac is asking someone at the Mac about an approved action."""
    if kind != "approval_pending" or not action_id:
        return
    if _platform_db_local():
        await _apply_action_result(
            action_id, "dispatched", "", user_id=user_id, mac_prompt=True,
        )
        return
    await _post_to_platform("desktop/internal/action-result", {
        "user_id": user_id, "action_id": action_id,
        "status": "dispatched", "summary": "", "mac_prompt": True,
    })


desktop_bridge.approval_event_hook = _on_approval_event


# ═════════════════════════════════════════════════════════════════════════
# Remote tasks (CONNECTIONS.md §4.4, §8)
# ═════════════════════════════════════════════════════════════════════════
@router.post("/desktop/devices/{device_id}/tasks")
async def submit_task(
    device_id: str, req: TaskSubmitReq, request: Request,
    current_user=Depends(get_current_user),
) -> Any:
    """One sentence from the phone for one Mac.

    Checked in this order: the flag, the Mac is yours and not revoked, a
    replay returns what the first request made, the Mac is online, it has
    fewer than `MAX_OPEN_TASKS_PER_MAC` unfinished, the rate limit. An
    offline Mac writes NO row: nothing is ever queued to run when it wakes
    (T17).
    """
    user_id = _uid_of(current_user)
    if not await _connections_on(user_id):
        return _flat("unavailable", 404,
                     "Sending tasks to your Mac isn't available yet.")
    if not _UUID_RE.match(req.client_request_id or ""):
        return _flat("invalid", 400, "client_request_id must be a UUID.")
    text = (req.text or "").strip()
    if not text:
        return _flat("invalid", 400, "Say what you want your Mac to do.")
    now = datetime.utcnow()
    not_yours = _flat("not_found", 404, "That Mac isn't connected to your account.")

    async with async_session_maker() as db:
        dev = (await db.execute(
            select(DesktopDevice).where(
                DesktopDevice.id == device_id,
                DesktopDevice.user_id == user_id,
            )
        )).scalar_one_or_none()
        if dev is None or dev.revoked_at is not None:
            return not_yours
        existing = (await db.execute(
            select(DesktopTask).where(
                DesktopTask.user_id == user_id,
                DesktopTask.client_request_id == req.client_request_id,
            )
        )).scalar_one_or_none()
        if existing is not None:
            if existing.device_id != device_id:
                return _flat("conflict", 409,
                             "That request was already sent to another Mac.")
            return JSONResponse(await _task_view(db, existing, now), status_code=200)
        if not _online(dev, now):
            return _flat(
                "mac_offline", 409,
                f"{dev.device_name} is offline, so nothing was sent. Open "
                "Toup on that Mac and try again.",
            )
        open_tasks = list((await db.execute(
            select(DesktopTask).where(
                DesktopTask.user_id == user_id,
                DesktopTask.device_id == device_id,
                DesktopTask.status.notin_(tuple(task_state.TERMINAL)),
            )
        )).scalars().all())
        still_open = 0
        for t in open_tasks:
            t, _, _ = await _rederive(db, t, now, dev)
            if t.status not in task_state.TERMINAL:
                still_open += 1
        if still_open >= task_state.MAX_OPEN_TASKS_PER_MAC:
            return _flat(
                "busy", 409,
                f"{dev.device_name} already has {still_open} tasks going. "
                "Wait for one to finish, or cancel one.",
            )
        if not await _rate_user(f"desktop_task_submit_user:{user_id}", 10, 60):
            raise HTTPException(429, "rate limited (desktop_task_submit); retry shortly")

        task = DesktopTask(
            id=str(uuid.uuid4()), user_id=user_id, device_id=device_id,
            title=_flatten(text, 120), status="queued", turn_state="queued",
            client_request_id=req.client_request_id,
            submitted_session_jti=_session_jti(request),
            submitted_via=_client_via(request),
            created_at=now, updated_at=now,
        )
        db.add(task)
        try:
            await db.commit()
        except IntegrityError:
            # A concurrent replay won the unique key; answer with its task.
            await db.rollback()
            existing = (await db.execute(
                select(DesktopTask).where(
                    DesktopTask.user_id == user_id,
                    DesktopTask.client_request_id == req.client_request_id,
                )
            )).scalar_one()
            if existing.device_id != device_id:
                return _flat("conflict", 409,
                             "That request was already sent to another Mac.")
            return JSONResponse(await _task_view(db, existing, now), status_code=200)
        task_id, device_name = task.id, dev.device_name

    code, body = await _hand_off_task(user_id, task_id, device_id, device_name, text)
    async with async_session_maker() as db:
        now = datetime.utcnow()
        if code == 202:
            # The agent can post running/ended before its 202 arrives here.
            # Preserve that newer state but always save the conversation link.
            await db.execute(
                sa_update(DesktopTask)
                .where(DesktopTask.id == task_id, DesktopTask.user_id == user_id)
                .values(conversation_id=str(body.get("conversation_id") or "")[:36] or None)
            )
            await db.execute(
                sa_update(DesktopTask)
                .where(DesktopTask.id == task_id)
                .where(DesktopTask.turn_state == "queued")
                .values(turn_state="accepted", updated_at=now,
                        conversation_id=str(body.get("conversation_id") or "")[:36] or None)
            )
        elif code == 409 and body.get("error") == "mac_offline":
            await db.execute(
                sa_update(DesktopTask)
                .where(DesktopTask.id == task_id)
                .values(turn_state="mac_offline", updated_at=now)
            )
        else:
            # The one terminal status not produced by `derive`: the task
            # never reached an agent, so there is no turn or card to derive
            # from. A failure is a record, never a request that vanished.
            await db.execute(
                sa_update(DesktopTask)
                .where(DesktopTask.id == task_id)
                .where(DesktopTask.status.notin_(tuple(task_state.TERMINAL)))
                .values(status="failed", outcome=task_state.OUTCOME_UNREACHABLE,
                        finished_at=now, updated_at=now)
            )
        await db.commit()
        task = await db.get(DesktopTask, task_id)
        view = await _task_view(db, task, now)
    return JSONResponse(view, status_code=202)


async def _hand_off_task(
    user_id: str, task_id: str, device_id: str, device_name: str, text: str,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Start the task's turn on whichever process holds that Mac's socket."""
    body = {
        "user_id": user_id, "task_id": task_id, "device_id": device_id,
        "device_name": device_name, "text": text,
    }
    if desktop_bridge.is_device_connected(user_id, device_id):
        return await _start_task_run(user_id, RunTaskReq(**body))
    return await _call_agent(
        user_id, "desktop/internal/run-task", body, timeout=HANDOFF_BUDGET_S,
    )


@router.get("/desktop/tasks")
async def list_tasks(
    current_user=Depends(get_current_user),
    device_id: Optional[str] = Query(None, max_length=36),
    limit: int = Query(20, ge=1, le=50),
) -> Dict[str, Any]:
    user_id = _uid_of(current_user)
    now = datetime.utcnow()
    async with async_session_maker() as db:
        q = select(DesktopTask).where(DesktopTask.user_id == user_id)
        if device_id:
            q = q.where(DesktopTask.device_id == device_id)
        rows = list((await db.execute(
            q.order_by(DesktopTask.created_at.desc()).limit(limit)
        )).scalars().all())
        return {"tasks": [await _task_view(db, t, now) for t in rows]}


async def _owned_task(db, task_id: str, user_id: str) -> DesktopTask:
    task = (await db.execute(
        select(DesktopTask).where(
            DesktopTask.id == task_id, DesktopTask.user_id == user_id,
        )
    )).scalar_one_or_none()
    if task is None:
        raise HTTPException(404, "That task isn't available.")
    return task


@router.get("/desktop/tasks/{task_id}")
async def get_task(
    task_id: str, current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    user_id = _uid_of(current_user)
    async with async_session_maker() as db:
        task = await _owned_task(db, task_id, user_id)
        return await _task_view(db, task, datetime.utcnow())


CANCELLED_CARD_SUMMARY = "You cancelled the task, so this will not run."


@router.post("/desktop/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str, current_user=Depends(get_current_user),
) -> Any:
    """Never gated. Idempotent once cancelled; 409 once otherwise finished.

    An action already running on the Mac reports what actually happened on
    its own row; the task shows cancelled.
    """
    user_id = _uid_of(current_user)
    now = datetime.utcnow()
    async with async_session_maker() as db:
        task = await _owned_task(db, task_id, user_id)
        if task.status in task_state.TERMINAL and task.status != "cancelled":
            return _flat("finished", 409, "This task has already finished.")
        r = await db.execute(
            sa_update(DesktopTask)
            .where(DesktopTask.id == task_id)
            .where(DesktopTask.user_id == user_id)
            .where(DesktopTask.status.notin_(tuple(task_state.TERMINAL)))
            .values(status="cancelled", cancelled_at=now, cancel_reason="user",
                    outcome=task_state.OUTCOME_CANCELLED_BY_YOU,
                    finished_at=now, updated_at=now)
        )
        await db.execute(
            sa_update(DesktopPendingAction)
            .where(DesktopPendingAction.user_id == user_id)
            .where(DesktopPendingAction.remote_task_id == task_id)
            .where(DesktopPendingAction.status == "pending")
            .values(status="rejected", decided_at=now, decided_via="cancel",
                    result_json=json.dumps({
                        "status": "rejected", "summary": CANCELLED_CARD_SUMMARY,
                    }))
        )
        await db.commit()
        await db.refresh(task)
        if (r.rowcount or 0) == 0 and task.status != "cancelled":
            return _flat("finished", 409, "This task has already finished.")
        view = await _task_view(db, task, now)

    if (r.rowcount or 0) > 0:
        # Best effort: the row is the truth whether or not the agent hears.
        if desktop_bridge.is_connected(user_id):
            await _cancel_task_locally(user_id, task_id)
        else:
            await _call_agent(
                user_id, "desktop/internal/cancel-task",
                {"user_id": user_id, "task_id": task_id},
                timeout=HANDOFF_BUDGET_S,
            )
    return view


# ── The agent's half of a remote task ───────────────────────────────────
#: task id → conversation id, so a retried run-task starts nothing twice.
_RUN_TASKS: "Dict[str, str]" = {}
_RUN_TASKS_CAP = 512
_RUN_TASK_LOCK = asyncio.Lock()


def _task_frame(text: str, device_name: str) -> str:
    """The one line of context a phone task carries into its turn."""
    return (
        f"{text}\n\n[Sent from the user's phone for their Mac "
        f"\"{_flatten(device_name, 80)}\". Anything that needs a yes appears "
        "on their phone as a confirmation card, and some things can only be "
        "confirmed at the Mac, where nobody may be.]"
    )


async def _start_task_run(
    user_id: str, req: RunTaskReq,
) -> Tuple[int, Dict[str, Any]]:
    async with _RUN_TASK_LOCK:
        return await _start_task_run_once(user_id, req)


async def _start_task_run_once(
    user_id: str, req: RunTaskReq,
) -> Tuple[int, Dict[str, Any]]:
    if desktop_bridge.is_task_cancelled(req.task_id):
        return 409, {"error": "cancelled"}
    if req.task_id in _RUN_TASKS:
        return 202, {"conversation_id": _RUN_TASKS[req.task_id]}
    if not bool(getattr(settings, "desktop_relay_enabled", False)):
        return 409, {"error": "disabled"}
    if not desktop_bridge.is_device_connected(user_id, req.device_id):
        return 409, {"error": "mac_offline"}
    from app.api import apps as apps_api

    runner = getattr(apps_api, "_agent_runner", None)
    if runner is None:
        return 409, {"error": "disabled"}

    from app.db.models import Conversation

    conversation_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        # Its own conversation with no day chat, so a Mac task never merges
        # into today's thread and the phone can open exactly this one.
        db.add(Conversation(
            id=conversation_id, user_id=user_id,
            title=_flatten(req.text, 120), channel="app",
            started_at=datetime.utcnow(),
        ))
        await db.commit()
    _RUN_TASKS[req.task_id] = conversation_id
    while len(_RUN_TASKS) > _RUN_TASKS_CAP:
        _RUN_TASKS.pop(next(iter(_RUN_TASKS)))
    target = desktop_bridge.DesktopTarget(
        task_id=req.task_id, device_id=req.device_id,
        device_name=_flatten(req.device_name, 80) or "your Mac",
    )
    _spawn_tracked(
        _run_task_turn(runner, user_id, req, conversation_id, target),
        name=f"desktop-task-{req.task_id[:8]}",
    )
    return 202, {"conversation_id": conversation_id}


async def _run_task_turn(
    runner: Any, user_id: str, req: RunTaskReq, conversation_id: str,
    target: "desktop_bridge.DesktopTarget",
) -> None:
    token = desktop_bridge.DESKTOP_TARGET.set(target)
    try:
        await _post_task_status(user_id, req.task_id, "running")
        state = "ended"
        try:
            response = await runner.run(
                user_message=_task_frame(req.text, target.device_name),
                display_user_message=req.text,
                user_id=user_id,
                session_id=conversation_id,
                channel="app",
                # No third-party connector tools in a Mac task (Q7): a
                # narrower injection surface, and 13 desktop tools do not
                # push the array toward the provider's cap.
                connector_scope=[],
                cancel_check=lambda: desktop_bridge.is_task_cancelled(req.task_id),
            )
            if getattr(response, "stopped_reason", ""):
                state = "errored"
        except Exception as exc:
            logger.warning("[desktop] task %s turn failed: %s", req.task_id[:8], exc)
            state = "errored"
        if target.flags.get("mac_unavailable"):
            state = "mac_offline"
        await _post_task_status(user_id, req.task_id, state)
    finally:
        desktop_bridge.DESKTOP_TARGET.reset(token)


async def _post_task_status(user_id: str, task_id: str, turn_state: str) -> None:
    if _platform_db_local():
        await _apply_task_status(user_id, task_id, turn_state)
        return
    await _post_to_platform("desktop/internal/task-status", {
        "user_id": user_id, "task_id": task_id, "turn_state": turn_state,
    })


async def _apply_task_status(
    user_id: str, task_id: str, turn_state: str,
) -> Optional[str]:
    now = datetime.utcnow()
    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopTask)
            .where(DesktopTask.id == task_id)
            .where(DesktopTask.user_id == user_id)
            .where(DesktopTask.status.notin_(tuple(task_state.TERMINAL)))
            .values(turn_state=turn_state, updated_at=now)
        )
        await db.commit()
        task = (await db.execute(
            select(DesktopTask).where(
                DesktopTask.id == task_id, DesktopTask.user_id == user_id,
            )
        )).scalar_one_or_none()
        if task is None:
            return None
        task, _, _ = await _rederive(db, task, now)
        return task.status


@router.post("/desktop/internal/run-task")
async def run_task_route(
    req: RunTaskReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Any:
    """Platform → agent: start a phone task's turn, pinned to one Mac."""
    tenant_uid = await _user_for_agent_key(x_agent_key)
    _require_agent_bound_user(tenant_uid, req.user_id)
    code, body = await _start_task_run(tenant_uid, req)
    return JSONResponse(body, status_code=code)


async def _cancel_task_locally(user_id: str, task_id: str) -> int:
    desktop_bridge.mark_task_cancelled(task_id)
    return await desktop_bridge.cancel_task_relays(user_id, task_id)


@router.post("/desktop/internal/cancel-task")
async def cancel_task_route(
    req: CancelTaskReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Dict[str, Any]:
    tenant_uid = await _user_for_agent_key(x_agent_key)
    _require_agent_bound_user(tenant_uid, req.user_id)
    n = await _cancel_task_locally(tenant_uid, req.task_id)
    return {"ok": True, "relay_cancelled": n}


@router.post("/desktop/internal/task-status")
async def task_status_route(
    req: TaskStatusReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Dict[str, Any]:
    tenant_uid = await _user_for_agent_key(x_agent_key)
    _require_agent_bound_user(tenant_uid, req.user_id)
    if req.turn_state not in ("running", "ended", "errored", "mac_offline"):
        raise HTTPException(400, "unknown turn_state")
    status = await _apply_task_status(tenant_uid, req.task_id, req.turn_state)
    if status is None:
        raise HTTPException(404, "no such task for this tenant")
    return {"ok": True, "status": status}


# ═════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═════════════════════════════════════════════════════════════════════════
@router.get("/desktop/status")
async def desktop_status(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """Pairing + presence for the Settings page.

    `paired` and `connected` both come from the platform DB. There is no
    `in_process_live` fallback of the kind `extension_status` carries
    (`extension.py:813-818`): on the platform that term is always False and
    reading it invites the belief that the bridge registry is visible from
    here, which is §1.9's whole story.

    `relay_enabled` reads the per-account registry flag, which is what the
    platform renders into that tenant's env. It used to read the PLATFORM's
    own setting, which is never the tenant's (F-M).
    """
    user_id = _uid_of(current_user)
    devices = await _list_devices(user_id)
    enabled = await _connections_on(user_id)
    return {
        "user_id": user_id,
        "paired": len(devices) > 0,
        "connected": any(d["online"] for d in devices),
        "devices": devices,
        # Stated so the Settings page can explain a paired Mac whose tools
        # do nothing, rather than showing a healthy device and silence.
        "relay_enabled": enabled,
        "connections_enabled": enabled,
    }
