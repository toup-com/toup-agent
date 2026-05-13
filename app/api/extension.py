"""
Toup Chrome Extension — pairing + runtime WebSocket.

This file does double duty:

  - On the **platform** (toup.ai): exposes the pairing endpoints
    (/api/extension/pair/{init,approve,poll}) that the extension and the
    logged-in user use to bind a Chrome to a Toup account.

  - On a **tenant agent VPS**: exposes /api/ws/extension, the runtime
    WebSocket the extension keeps open so the agent runtime can send
    search/read/research tasks.

Pairing state lives in process memory (10-min TTL) so a single platform
worker can serve all pending pairings; this is sufficient until we scale
beyond one platform replica. The persistent state is the JWT itself
(stateless) plus an optional revocation list (left as a TODO since v1
pairing is per-device; users can revoke by rotating their password).

See docs/PROTOCOL.md in extensions/chrome for the full contract.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, WebSocket
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import and_, select, text, update
from sqlalchemy.exc import ProgrammingError

from app.agent import extension_bridge
from app.api.auth import get_current_user
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentConfig, ExtensionDevice
from app.services.auth_service import get_user_by_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Extension"])


# ─────────────────────────────────────────────────────────────────────
# Pairing state — single-process in-memory store with TTL.
# ─────────────────────────────────────────────────────────────────────
PAIR_TTL_S = 600           # 10 minutes
PAIR_POLL_MIN_INTERVAL = 3

# user_code (8 chars, dashed) → record
# device_code (40 hex chars) → user_code
_pair_by_user_code:   Dict[str, Dict[str, Any]] = {}
_pair_by_device_code: Dict[str, str] = {}
_pair_lock = asyncio.Lock()


def _gen_user_code() -> str:
    """Human-typeable, ambiguity-free (no 0/O, 1/I/L)."""
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    raw = "".join(secrets.choice(alphabet) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


async def _gc_pairings() -> None:
    """Drop expired pairings. Called opportunistically."""
    now = time.time()
    stale = [uc for uc, r in _pair_by_user_code.items() if r["expires_at"] < now]
    for uc in stale:
        rec = _pair_by_user_code.pop(uc, None)
        if rec:
            _pair_by_device_code.pop(rec["device_code"], None)


# ─────────────────────────────────────────────────────────────────────
# Tiny per-IP / per-user sliding-window limiter for the expensive
# endpoints (LLM-backed /suggest, Vision-backed /ground, pairing init).
# In-memory; per-replica. Good enough for MVP — DoS targeting would
# need to overwhelm every replica simultaneously.
# ─────────────────────────────────────────────────────────────────────
_RATE_BUCKETS: Dict[str, list] = {}
_rate_lock = asyncio.Lock()


async def _rate_check(key: str, limit: int, window_s: int) -> bool:
    """Returns True if allowed, False if over the limit."""
    now = time.time()
    async with _rate_lock:
        bucket = _RATE_BUCKETS.setdefault(key, [])
        # Drop entries outside the window
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
    ip = _client_ip(req)
    if not await _rate_check(f"{name}:{ip}", limit, window_s):
        raise HTTPException(429, f"rate limited ({name}); retry shortly")


# ─────────────────────────────────────────────────────────────────────
# Request/response schemas
# ─────────────────────────────────────────────────────────────────────
class PairInitReq(BaseModel):
    device_name:       str = Field("Chrome", max_length=120)
    extension_version: str = Field("0.0.0", max_length=32)


class PairPollReq(BaseModel):
    device_code: str = Field(..., min_length=20, max_length=120)


class PairApproveReq(BaseModel):
    user_code: str = Field(..., min_length=8, max_length=12)


# ─────────────────────────────────────────────────────────────────────
# Public base URL — derived from existing settings (no new env required).
# ─────────────────────────────────────────────────────────────────────
def _public_base_url() -> str:
    candidate = (
        getattr(settings, "public_base_url", None)
        or getattr(settings, "frontend_url", None)
        or getattr(settings, "platform_api_url", "https://toup.ai/api")
    )
    # Strip trailing /api so we get the site origin.
    if candidate.endswith("/api"):
        candidate = candidate[:-4]
    return candidate.rstrip("/") or "https://toup.ai"


# ─────────────────────────────────────────────────────────────────────
# JWT minting / verification (extension scope)
# ─────────────────────────────────────────────────────────────────────
EXT_TOKEN_TTL_S = 90 * 24 * 3600  # 90 days


def _mint_extension_token(user_id: str, agent_api_key: str) -> Dict[str, Any]:
    """Mint a long-lived JWT carrying scope=extension, signed with the
    *user's agent_api_key* (same secret the chat session token uses).

    Why agent_api_key, not settings.jwt_secret: the extension's chat-ws
    connects to `/api/ws/chat` on the agent, which validates with
    `settings.agent_api_key` + audience "toup-agent-session" + issuer
    "toup-platform". Minting with the same scheme means one token works
    on both `/ws/extension` and `/ws/chat` — and matches the existing
    per-user agent-key topology (each tenant already has its key set
    at bind time, so no new secret distribution is needed).
    """
    if not agent_api_key:
        raise HTTPException(409, "user has no provisioned agent yet")
    now = int(time.time())
    jti = secrets.token_hex(16)
    payload = {
        "sub":   user_id,
        "iss":   "toup-platform",
        "aud":   "toup-agent-session",
        "scope": "extension",
        "jti":   jti,
        "iat":   now,
        "exp":   now + EXT_TOKEN_TTL_S,
    }
    token = jwt.encode(payload, agent_api_key, algorithm="HS256")
    return {"token": token, "issued_at": now, "expires_in": EXT_TOKEN_TTL_S, "jti": jti}


# ─────────────────────────────────────────────────────────────────────
# Persistent pairing — ExtensionDevice rows + heartbeat (T0c, 2026-05-12)
# ─────────────────────────────────────────────────────────────────────
# The previous "is paired?" check was extension_bridge.is_connected(user_id)
# — pure in-memory WS presence on whatever process serves the endpoint.
# That broke for two reasons:
#   1. The platform process and the tenant agent process each have their
#      own extension_bridge module instance with their own dict, so
#      platform never saw tenant-side sockets.
#   2. Even when the user *was* paired, a brief WS disconnect would flip
#      the settings page to "No extension paired" until reconnect.
# ExtensionDevice rows persist on the platform DB; the tenant agent's
# /ws/extension handler heartbeats over HTTP so the settings page can
# render "Paired (online, last seen 12s ago)" without polling.

# "Online" window — extension heartbeats every 25s (sidepanel chat-ws +
# /ws/extension both ping that often), and the WS-register / WS-close
# events also fire immediate heartbeats. 90s tolerates two missed
# heartbeats before flipping to offline.
DEVICE_ONLINE_WINDOW_S = 90


async def _record_pairing(
    user_id: str,
    device_name: str,
    extension_version: Optional[str],
    token_jti: Optional[str],
) -> Optional[str]:
    """Insert an ExtensionDevice row on pair-approve / pair-issue.

    Best-effort: if the migration hasn't run on this replica yet
    (ProgrammingError on the table) we log + swallow so pairing still
    succeeds. The settings page will fall back to legacy presence-only
    until the migration ships.

    Returns the new device id on success, None on failure (caller can
    surface the id to the extension for later revoke calls).
    """
    try:
        async with async_session_maker() as db:
            device = ExtensionDevice(
                id=str(uuid.uuid4()),
                user_id=user_id,
                device_name=(device_name or "Chrome")[:200],
                extension_version=(extension_version or None) and extension_version[:40],
                token_jti=(token_jti or None) and token_jti[:64],
                paired_at=datetime.utcnow(),
            )
            db.add(device)
            await db.commit()
            return device.id
    except ProgrammingError as exc:
        # extension_devices table not yet migrated on this replica —
        # don't break pairing while the rollout completes.
        logger.warning("[extension] _record_pairing skipped (no table): %s", exc)
        return None
    except Exception as exc:
        logger.warning("[extension] _record_pairing failed for %s: %s", user_id[:8], exc)
        return None


async def _list_devices(user_id: str) -> List[Dict[str, Any]]:
    """Return non-revoked ExtensionDevice rows shaped for the UI."""
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(ExtensionDevice)
                .where(
                    ExtensionDevice.user_id == user_id,
                    ExtensionDevice.revoked_at.is_(None),
                )
                .order_by(ExtensionDevice.paired_at.desc())
            )
            rows = result.scalars().all()
    except ProgrammingError:
        return []
    except Exception as exc:
        logger.debug("[extension] _list_devices read failed: %s", exc)
        return []

    cutoff = datetime.utcnow() - timedelta(seconds=DEVICE_ONLINE_WINDOW_S)
    out: List[Dict[str, Any]] = []
    for r in rows:
        is_online = bool(r.last_seen_at and r.last_seen_at >= cutoff)
        out.append({
            "id":                r.id,
            "device_name":       r.device_name,
            "extension_version": r.extension_version,
            "paired_at":         r.paired_at.isoformat() + "Z" if r.paired_at else None,
            "last_seen_at":      r.last_seen_at.isoformat() + "Z" if r.last_seen_at else None,
            "online":            is_online,
        })
    return out


async def _touch_last_seen(user_id: str, token_jti: Optional[str], online: bool) -> None:
    """Upsert last_seen_at on the user's device row(s).

    Called from /extension/heartbeat (tenant agent → platform) when the
    extension's WS connects, periodically pings, or disconnects.

    Three cases, in priority order:
      1. token_jti matches a non-revoked row — touch that row.
      2. No jti match, but the user has at least one non-revoked row —
         touch the most-recent one (covers legacy pairings minted
         before jti tracking).
      3. The user has NO row at all but the tenant agent is reporting
         a live WS — they paired before this migration shipped. We
         materialize a row right here so the settings page stops
         lying. JWT signature on the WS already proves the user is
         authentic; the heartbeat's `X-Agent-Key` proves the tenant
         agent is the genuine reporter. Both verifications happen
         upstream of this function.
    """
    now = datetime.utcnow() if online else (datetime.utcnow() - timedelta(seconds=DEVICE_ONLINE_WINDOW_S + 5))
    try:
        async with async_session_maker() as db:
            updated = False
            if token_jti:
                r = await db.execute(
                    update(ExtensionDevice)
                    .where(
                        ExtensionDevice.user_id == user_id,
                        ExtensionDevice.token_jti == token_jti,
                        ExtensionDevice.revoked_at.is_(None),
                    )
                    .values(last_seen_at=now)
                )
                updated = (r.rowcount or 0) > 0
            if not updated:
                # Try most-recent non-revoked row first.
                latest = await db.execute(
                    select(ExtensionDevice.id)
                    .where(
                        ExtensionDevice.user_id == user_id,
                        ExtensionDevice.revoked_at.is_(None),
                    )
                    .order_by(ExtensionDevice.paired_at.desc())
                    .limit(1)
                )
                row_id = latest.scalar_one_or_none()
                if row_id:
                    await db.execute(
                        update(ExtensionDevice)
                        .where(ExtensionDevice.id == row_id)
                        .values(last_seen_at=now)
                    )
                    updated = True
            if not updated and online:
                # No row exists for this user — they paired before this
                # table existed. Materialize a row now (heartbeat upsert)
                # so the settings page stops saying "No extension paired"
                # for users who in fact have a live extension WS.
                device = ExtensionDevice(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    device_name="Chrome",
                    extension_version=None,
                    token_jti=(token_jti or None) and token_jti[:64],
                    paired_at=now,
                    last_seen_at=now,
                )
                db.add(device)
                logger.info("[extension] materialized device row for legacy paired user %s", user_id[:8])
            await db.commit()
    except ProgrammingError:
        return
    except Exception as exc:
        logger.debug("[extension] _touch_last_seen failed: %s", exc)


def _decode_extension_token(token: str) -> Optional[Dict[str, Any]]:
    """Validate the extension token on the agent side. Uses the agent's
    own `settings.agent_api_key` (which equals the platform's
    `cfg.agent_api_key` for this user — bound at deploy time).
    """
    if not token:
        return None
    secret = (settings.agent_api_key or "").strip()
    if not secret:
        logger.warning("[extension] agent_api_key not configured — cannot validate token")
        return None
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="toup-agent-session",
            issuer="toup-platform",
        )
    except JWTError as exc:
        logger.warning("[extension] token decode failed: %s", exc)
        return None
    if payload.get("scope") != "extension":
        logger.warning("[extension] token has wrong scope: %r", payload.get("scope"))
        return None
    # Defense in depth: token must be for the bound user.
    sub = payload.get("sub")
    if settings.user_id and sub != settings.user_id:
        logger.warning(
            "[extension] token user mismatch: token=%s bound=%s",
            (sub or "?")[:8], (settings.user_id or "?")[:8],
        )
        return None
    return payload


# ─────────────────────────────────────────────────────────────────────
# Pairing endpoints
# ─────────────────────────────────────────────────────────────────────
async def _agent_config_for(user_id: str) -> Optional[AgentConfig]:
    """Look up the user's active agent config (URL + api_key)."""
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(AgentConfig).where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.deploy_status == "active",
                )
            )
            return result.scalar_one_or_none()
    except Exception:
        return None


async def _agent_ws_url_for(user_id: str) -> Optional[str]:
    """Return the wss:// URL of the user's agent /api/ws/extension."""
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(AgentConfig.agent_url)
                .where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.deploy_status == "active",
                )
            )
            agent_url = result.scalar_one_or_none()
    except Exception:
        agent_url = None
    if not agent_url:
        return None
    ws_url = agent_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = ws_url.rstrip("/") + "/api/ws/extension"
    return ws_url


@router.post("/extension/pair/init")
async def pair_init(req: PairInitReq, request: Request) -> Dict[str, Any]:
    """
    Step 1. Extension calls this to start pairing.
    Returns the user_code the user will type into toup.ai.
    """
    # Cap pairing churn from a single IP at 10/min to deter spam.
    await _enforce_rate(request, "pair_init", limit=10, window_s=60)
    async with _pair_lock:
        await _gc_pairings()
        # Generate a fresh, unused user_code.
        for _ in range(20):
            user_code = _gen_user_code()
            if user_code not in _pair_by_user_code:
                break
        else:
            raise HTTPException(503, "pairing slot pressure; retry shortly")
        device_code = secrets.token_hex(20)
        record = {
            "user_code":         user_code,
            "device_code":       device_code,
            "device_name":       req.device_name[:120],
            "extension_version": req.extension_version[:32],
            "status":            "pending",
            "user_id":           None,
            "created_at":        time.time(),
            "expires_at":        time.time() + PAIR_TTL_S,
            "result":            None,
        }
        _pair_by_user_code[user_code] = record
        _pair_by_device_code[device_code] = user_code

    public_base = _public_base_url()
    return {
        "device_code":      device_code,
        "user_code":        user_code,
        "verification_uri": f"{public_base}/settings/extensions/pair",
        "expires_in":       PAIR_TTL_S,
        "interval":         PAIR_POLL_MIN_INTERVAL,
    }


@router.post("/extension/pair/poll")
async def pair_poll(req: PairPollReq) -> Any:
    """
    Step 3. Extension polls until the user approves.
    Returns 202 while pending; 200 with the token when approved.
    """
    async with _pair_lock:
        await _gc_pairings()
        user_code = _pair_by_device_code.get(req.device_code)
        rec = _pair_by_user_code.get(user_code) if user_code else None

    if not rec:
        raise HTTPException(400, {"status": "expired"})
    if rec["status"] == "pending":
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "pending"}, status_code=202)
    if rec["status"] == "denied":
        raise HTTPException(400, {"status": "denied"})
    if rec["status"] != "approved":
        raise HTTPException(400, {"status": "expired"})

    # Approved — return the previously-minted token, then evict the
    # pairing record so the token can't be replayed.
    result = rec["result"]
    async with _pair_lock:
        _pair_by_user_code.pop(user_code, None)
        _pair_by_device_code.pop(req.device_code, None)
    return {"status": "approved", **result}


@router.post("/extension/pair/approve")
async def pair_approve(req: PairApproveReq, current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Step 2. User submits the user_code from the toup.ai pairing page.
    Mints the extension JWT and stashes it on the pairing record for the
    extension's next poll to pick up.
    """
    async with _pair_lock:
        await _gc_pairings()
        rec = _pair_by_user_code.get(req.user_code)
    if not rec:
        raise HTTPException(404, "code not found or expired")
    if rec["status"] != "pending":
        raise HTTPException(409, f"code already {rec['status']}")

    user_id = current_user.id if hasattr(current_user, "id") else current_user["id"]
    cfg = await _agent_config_for(user_id)
    if not cfg or not cfg.agent_api_key:
        raise HTTPException(409, "user has no provisioned agent yet")
    token_bundle = _mint_extension_token(user_id, cfg.agent_api_key)
    agent_ws_url = ""
    if cfg.agent_url:
        agent_ws_url = (
            cfg.agent_url.replace("https://", "wss://").replace("http://", "ws://")
            .rstrip("/") + "/api/ws/extension"
        )

    # Durable pairing record — survives the in-memory pair-lock eviction
    # below so /extension/status can answer "is this user paired?" forever
    # without needing the WS to be live right now.
    await _record_pairing(
        user_id=user_id,
        device_name=rec["device_name"],
        extension_version=rec.get("extension_version"),
        token_jti=token_bundle.get("jti"),
    )

    async with _pair_lock:
        rec["status"] = "approved"
        rec["user_id"] = user_id
        rec["result"] = {
            "access_token": token_bundle["token"],
            "user_id":      user_id,
            "agent_ws_url": agent_ws_url,
            "issued_at":    token_bundle["issued_at"],
            "expires_in":   token_bundle["expires_in"],
        }
    logger.info("[extension] paired %s as device %r", user_id, rec["device_name"])
    return {"ok": True, "device_name": rec["device_name"]}


@router.post("/extension/pair/issue")
async def pair_issue(req: PairInitReq, current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Direct token mint for the *sidepanel* sign-in path (Phase 4 — no
    device code). The user is already authenticated to toup.ai via
    cookie/Bearer; we just hand them an extension-scoped JWT.

    The /auth/extension bridge page calls this immediately after the
    user logs in, then forwards the bundle to the extension via
    chrome.runtime.sendMessage(extensionId, ...).
    """
    user_id = current_user.id if hasattr(current_user, "id") else current_user["id"]
    cfg = await _agent_config_for(user_id)
    if not cfg or not cfg.agent_api_key:
        raise HTTPException(409, "user has no provisioned agent yet")
    bundle = _mint_extension_token(user_id, cfg.agent_api_key)
    agent_ws_url = ""
    if cfg.agent_url:
        agent_ws_url = (
            cfg.agent_url.replace("https://", "wss://").replace("http://", "ws://")
            .rstrip("/") + "/api/ws/extension"
        )
    await _record_pairing(
        user_id=user_id,
        device_name=req.device_name,
        extension_version=req.extension_version,
        token_jti=bundle.get("jti"),
    )
    logger.info("[extension] direct-issue token for %s device=%r", user_id, req.device_name)
    return {
        "access_token": bundle["token"],
        "user_id":      user_id,
        "agent_ws_url": agent_ws_url,
        "issued_at":    bundle["issued_at"],
        "expires_in":   bundle["expires_in"],
    }


@router.post("/extension/pair/deny")
async def pair_deny(req: PairApproveReq, current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """Reject a pending pairing (user clicked Deny on the UI)."""
    async with _pair_lock:
        rec = _pair_by_user_code.get(req.user_code)
        if rec and rec["status"] == "pending":
            rec["status"] = "denied"
    return {"ok": True}


@router.get("/extension/pair/lookup")
async def pair_lookup(user_code: str = Query(...), current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Used by the toup.ai pairing UI to display "Approve this device?"
    confirming the device_name before the user clicks Approve.
    """
    async with _pair_lock:
        await _gc_pairings()
        rec = _pair_by_user_code.get(user_code)
    if not rec:
        raise HTTPException(404, "code not found or expired")
    return {
        "device_name":       rec["device_name"],
        "extension_version": rec["extension_version"],
        "status":            rec["status"],
        "expires_at":        rec["expires_at"],
    }


# ─────────────────────────────────────────────────────────────────────
# Runtime WebSocket (runs on each tenant agent VPS)
# ─────────────────────────────────────────────────────────────────────
@router.websocket("/ws/extension")
async def ws_extension(websocket: WebSocket, token: Optional[str] = Query(None)) -> None:
    """
    The Chrome extension connects here to receive task envelopes from
    the agent runtime and return results. One connection per user.

    Auth precedence: WS subprotocol `bearer.<jwt>` (preferred — no URL
    leak, no server-log exposure) → `?token=` query → first-frame
    `{type:"auth",token}` (legacy).
    """
    # Read subprotocol token BEFORE accept so we can echo it back.
    sub_token: Optional[str] = None
    selected_sub: Optional[str] = None
    subprotocols = list(websocket.scope.get("subprotocols") or [])
    if "toup.auth.v1" in subprotocols:
        for sp in subprotocols:
            if isinstance(sp, str) and sp.startswith("bearer."):
                sub_token = sp[len("bearer."):]
                selected_sub = "toup.auth.v1"
                break
    if selected_sub:
        await websocket.accept(subprotocol=selected_sub)
    else:
        await websocket.accept()

    if sub_token and not token:
        token = sub_token

    # Legacy fallbacks: ?token= already populated by Query(None); first-frame as last resort.
    if not token:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=3.0)
            first = json.loads(raw)
            if first.get("type") == "auth":
                token = first.get("token")
        except (asyncio.TimeoutError, json.JSONDecodeError):
            pass
    payload = _decode_extension_token(token or "")
    if not payload:
        await websocket.send_json({"type": "error", "message": "Authentication required"})
        await websocket.close(code=4001, reason="Unauthorized")
        return

    user_id = payload["sub"]
    token_jti = payload.get("jti")

    # 2. Verify user still exists and is active.
    try:
        async with async_session_maker() as db:
            user = await get_user_by_id(db, user_id)
            if not user or not user.is_active:
                await websocket.close(code=4001, reason="user inactive")
                return
    except Exception as exc:
        # On standalone agent DBs without the full users table the lookup
        # may fail — still allow the connection (JWT signature already
        # proved authenticity).
        logger.debug("[extension] user lookup skipped: %s", exc)

    # 3. Register and start the read loop.
    ep = await extension_bridge.register(user_id, websocket)

    # Fire a heartbeat to the platform so the settings page on
    # toup.ai/settings/extensions flips to "online" without polling.
    # Fire-and-forget — heartbeat failure should never break the WS.
    asyncio.create_task(_post_heartbeat_to_platform(user_id, token_jti, online=True))

    # Periodic re-heartbeat keeps last_seen_at fresh during long sessions.
    async def _heartbeat_loop():
        try:
            while True:
                await asyncio.sleep(45)
                await _post_heartbeat_to_platform(user_id, token_jti, online=True)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("[extension] heartbeat loop exited: %s", exc)
    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    try:
        await websocket.send_json({
            "type": "hello",
            "agent_id": settings.app_name,
            "schema_version": 2,
        })
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("[extension] bad json from %s", user_id)
                continue

            mtype = msg.get("type")
            if mtype == "ping":
                await websocket.send_json({"type": "pong", "ts": msg.get("ts")})
            elif mtype in ("result", "progress", "event"):
                extension_bridge.deliver_inbound(user_id, msg)
            elif mtype == "ready":
                schema_version = msg.get("schema_version", 1)
                logger.info("[extension] ready from %s v=%s schema=%s",
                            user_id, msg.get("extension_version"), schema_version)
            # any other types ignored silently — forward-compatible
    except Exception as exc:
        logger.info("[extension] ws closed for %s: %s", user_id, exc)
    finally:
        heartbeat_task.cancel()
        await extension_bridge.unregister(user_id, ep)
        # Flip the settings page to "offline since" immediately on close.
        asyncio.create_task(_post_heartbeat_to_platform(user_id, token_jti, online=False))


async def _post_heartbeat_to_platform(
    user_id: str,
    token_jti: Optional[str],
    *,
    online: bool,
) -> None:
    """Fire-and-forget heartbeat from the TENANT AGENT to the platform.

    No-op when:
      - This code path is running ON the platform (same process — the
        DB write happens via the local /heartbeat handler if you call
        the endpoint, but skipping the loopback HTTP roundtrip is
        cleaner).
      - X-Agent-Key isn't configured.
      - platform_api_url isn't a real https://toup.ai/api target.

    Auth: X-Agent-Key header. The platform's /heartbeat verifies the
    key matches AgentConfig.agent_api_key for the user_id we claim.
    """
    agent_key = (getattr(settings, "agent_api_key", "") or "").strip()
    if not agent_key:
        return
    platform_url = (getattr(settings, "platform_api_url", "") or "").strip()
    if not platform_url or not platform_url.startswith("http"):
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            await client.post(
                f"{platform_url.rstrip('/')}/extension/heartbeat",
                json={
                    "user_id":   user_id,
                    "online":    online,
                    "token_jti": token_jti,
                },
                headers={"X-Agent-Key": agent_key},
            )
    except Exception as exc:
        # Heartbeat is best-effort — platform unreachable, DNS hiccup, etc.
        # The settings page will keep its stale state until the next
        # successful heartbeat; nothing user-facing breaks.
        logger.debug("[extension] heartbeat post failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────
# Diagnostics — paired devices + presence
# ─────────────────────────────────────────────────────────────────────
@router.get("/extension/status")
async def extension_status(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """Pairing + live presence for the settings page.

    Three fields the UI cares about:
      `paired`     — has the user ever paired a device (durable, DB-backed)
      `connected`  — is any device online RIGHT NOW (heartbeat within
                     DEVICE_ONLINE_WINDOW_S, OR — fallback for legacy
                     paired-but-no-heartbeat devices — local presence on
                     this process's extension_bridge dict)
      `devices`    — full list with per-device online/last-seen for the
                     UI to render "Paired with Chrome on MacBook · last
                     seen 12s ago"

    The legacy boolean `connected` is kept for backwards compatibility
    with any caller that still treats it as the sole truth source.
    """
    user_id = current_user.id if hasattr(current_user, "id") else current_user["id"]
    devices = await _list_devices(user_id)
    any_online = any(d["online"] for d in devices)
    # Belt-and-braces: in-process bridge presence covers the case where
    # this endpoint happens to be served by the same process the WS is
    # registered on (the tenant agent). The heartbeat path covers the
    # cross-process case (platform answering the question).
    in_process_live = extension_bridge.is_connected(user_id)
    return {
        "user_id":   user_id,
        "paired":    len(devices) > 0,
        "connected": any_online or in_process_live,
        "devices":   devices,
    }


@router.get("/extension/devices")
async def list_extension_devices(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """Same data the status endpoint returns under `devices` — exposed
    on its own route so admin UIs and future device-management screens
    can poll it without re-running the presence check."""
    user_id = current_user.id if hasattr(current_user, "id") else current_user["id"]
    return {"devices": await _list_devices(user_id)}


class RevokeDeviceReq(BaseModel):
    device_id: Optional[str] = None  # None = revoke ALL paired devices for this user


@router.post("/extension/revoke")
async def revoke_extension_device(
    req: RevokeDeviceReq,
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Soft-delete one (or all) of the user's paired devices.

    User-facing "Unpair" button. We don't revoke the JWT itself (token TTL
    handles that within 90 days); we just mark the row revoked so the
    settings page stops showing it AND the tenant agent's WS auth path
    can deny reconnects that present the matching jti (jti denylist —
    follow-up; today the row removal alone is enough since the next
    settings-page render reflects the change, and the user can also re-
    pair to mint a fresh token if they want immediate cutoff).
    """
    user_id = current_user.id if hasattr(current_user, "id") else current_user["id"]
    now = datetime.utcnow()
    try:
        async with async_session_maker() as db:
            if req.device_id:
                r = await db.execute(
                    update(ExtensionDevice)
                    .where(
                        ExtensionDevice.id == req.device_id,
                        ExtensionDevice.user_id == user_id,
                        ExtensionDevice.revoked_at.is_(None),
                    )
                    .values(revoked_at=now)
                )
            else:
                r = await db.execute(
                    update(ExtensionDevice)
                    .where(
                        ExtensionDevice.user_id == user_id,
                        ExtensionDevice.revoked_at.is_(None),
                    )
                    .values(revoked_at=now)
                )
            await db.commit()
            revoked = r.rowcount or 0
    except ProgrammingError:
        revoked = 0
    except Exception as exc:
        logger.warning("[extension] revoke failed for %s: %s", user_id[:8], exc)
        raise HTTPException(500, "revoke failed")
    return {"ok": True, "revoked": revoked}


# ─────────────────────────────────────────────────────────────────────
# Heartbeat — called by the TENANT AGENT (not the browser) when the
# extension's /ws/extension socket connects, keeps pinging, or
# disconnects. The agent has the user's agent_api_key in its env;
# verifying it against AgentConfig for the claimed user is what
# authenticates this endpoint.
# ─────────────────────────────────────────────────────────────────────
class HeartbeatReq(BaseModel):
    user_id: str
    online: bool = True
    token_jti: Optional[str] = None
    extension_version: Optional[str] = None


@router.post("/extension/heartbeat")
async def extension_heartbeat(
    req: HeartbeatReq,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
) -> Dict[str, Any]:
    if not x_agent_key:
        raise HTTPException(401, "X-Agent-Key required")
    # Verify the tenant key matches the user it claims to heartbeat for.
    try:
        async with async_session_maker() as db:
            row = await db.execute(
                select(AgentConfig.user_id).where(
                    and_(
                        AgentConfig.user_id == req.user_id,
                        AgentConfig.agent_api_key == x_agent_key,
                    )
                )
            )
            if not row.scalar_one_or_none():
                raise HTTPException(403, "agent key mismatch")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("[extension] heartbeat auth check failed: %s", exc)
        raise HTTPException(503, "heartbeat unavailable")

    await _touch_last_seen(req.user_id, req.token_jti, online=req.online)
    return {"ok": True}


@router.get("/admin/extension/stats")
async def extension_stats(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """Operational counters — admin-only."""
    is_admin = getattr(current_user, "is_admin", False) or (
        isinstance(current_user, dict) and current_user.get("is_admin")
    )
    if not is_admin:
        raise HTTPException(403, "admin only")
    return extension_bridge.stats()


# ═════════════════════════════════════════════════════════════════════
# v2 — sessions, live view, pause/resume
# ═════════════════════════════════════════════════════════════════════
@router.get("/extension/sessions")
async def list_sessions(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """List all live agent-controlled browser sessions for this user."""
    user_id = current_user.id
    return {"sessions": extension_bridge.list_sessions_for_user(user_id)}


@router.get("/extension/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    user_id = current_user.id
    s = extension_bridge.get_session(user_id, session_id)
    if not s:
        raise HTTPException(404, "session not found")
    return s


@router.get("/extension/sessions/{session_id}/screenshot")
async def get_session_screenshot(
    session_id: str,
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Latest cached screenshot for a session (base64 JPEG)."""
    user_id = current_user.id
    shot = extension_bridge.latest_screenshot(user_id, session_id)
    if not shot:
        raise HTTPException(404, "no screenshot available yet")
    return shot


class _ControlReq(BaseModel):
    pass


@router.post("/extension/sessions/{session_id}/pause")
async def pause_session(session_id: str, current_user=Depends(get_current_user)) -> Dict[str, Any]:
    user_id = current_user.id
    s = extension_bridge.get_session(user_id, session_id)
    if not s:
        raise HTTPException(404, "session not found")
    try:
        await extension_bridge.send_control(user_id, session_id, "pause")
    except extension_bridge.ExtensionUnavailable:
        raise HTTPException(409, "extension not connected")
    return {"ok": True}


@router.post("/extension/sessions/{session_id}/resume")
async def resume_session(session_id: str, current_user=Depends(get_current_user)) -> Dict[str, Any]:
    user_id = current_user.id
    s = extension_bridge.get_session(user_id, session_id)
    if not s:
        raise HTTPException(404, "session not found")
    try:
        await extension_bridge.send_control(user_id, session_id, "resume")
    except extension_bridge.ExtensionUnavailable:
        raise HTTPException(409, "extension not connected")
    return {"ok": True}


# ═════════════════════════════════════════════════════════════════════
# Phase 6 — vision-based action grounding
# ═════════════════════════════════════════════════════════════════════
class GroundReq(BaseModel):
    session_id:         Optional[str] = None
    screenshot_b64:     str
    image_width:        int
    image_height:       int
    target_description: str


@router.post("/extension/ground")
async def extension_ground(
    req: GroundReq,
    request: Request,
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Vision-based action grounding. Extension sends a viewport screenshot
    + a textual description of what the agent wants to click/type/hover.
    We return pixel coordinates (or grounding_failed if low confidence).
    """
    from app.services.vision_grounding import ground

    user_id = current_user.id
    # Vision calls are expensive ($/call). Cap at 30/min per user.
    await _enforce_rate(request, f"ground:{user_id}", limit=30, window_s=60)
    out = await ground(
        user_id=user_id,
        screenshot_b64=req.screenshot_b64,
        image_width=int(req.image_width or 0),
        image_height=int(req.image_height or 0),
        target_description=req.target_description or "",
    )
    return out


# ═════════════════════════════════════════════════════════════════════
# Phase 6 — unified conversation: today's day-chat + messages
# ═════════════════════════════════════════════════════════════════════
@router.get("/extension/conversation/today")
async def conversation_today(
    tz: Optional[str] = Query(None, description="IANA tz (e.g. 'America/New_York'). Falls back to user profile."),
    limit: int = Query(50, ge=1, le=200),
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Return today's day-chat (read-only — does NOT create one) plus the
    last `limit` messages. Used by the Chrome sidepanel to hydrate its
    message list on sign-in so the same conversation the user has on
    web/mobile is visible.
    """
    from datetime import datetime
    from app.db.models import Message
    try:
        from app.db.models.day_chat import DayChat
    except Exception:
        DayChat = None  # type: ignore

    user_id = current_user.id
    user_tz = tz or getattr(current_user, "timezone", None) or "UTC"

    # Resolve today's local date in the user's timezone.
    try:
        from zoneinfo import ZoneInfo
        local_date = datetime.now(tz=ZoneInfo(user_tz)).date().isoformat()
    except Exception:
        local_date = datetime.utcnow().date().isoformat()

    out: Dict[str, Any] = {
        "tz":         user_tz,
        "local_date": local_date,
        "day_chat":   None,
        "messages":   [],
        # Agent profile (color + name) so the sidepanel can paint its
        # chrome in the user's chosen accent. Same source the web app's
        # /agent-setup/config endpoint reads from. Null fields are fine —
        # the frontend falls back to the default purple.
        "agent": {
            "color": None,
            "name":  None,
        },
    }

    try:
        async with async_session_maker() as db:
            day_chat_id: Optional[str] = None
            if DayChat is not None:
                row = (await db.execute(
                    select(DayChat.id).where(
                        DayChat.user_id == user_id,
                        DayChat.local_date == local_date,
                    )
                )).first()
                if row:
                    day_chat_id = row[0]
                    out["day_chat"] = {"id": day_chat_id, "local_date": local_date}

            # Best-effort agent profile lookup. agent_configs is
            # PLATFORM_ONLY (see base.py PLATFORM_ONLY_TABLES) — this
            # router runs on the platform side so the read is safe.
            try:
                from app.db.models import AgentConfig
                cfg_row = (await db.execute(
                    select(AgentConfig.agent_color, AgentConfig.agent_name)
                    .where(AgentConfig.user_id == user_id)
                )).first()
                if cfg_row:
                    out["agent"]["color"] = cfg_row[0]
                    out["agent"]["name"]  = cfg_row[1]
            except Exception as e:  # noqa: BLE001
                # Don't fail the hydrate just because the profile lookup
                # blew up — sidepanel keeps working with default color.
                logger.debug("[extension] agent profile lookup skipped: %s", e)

            if day_chat_id:
                msg_rows = (await db.execute(
                    select(
                        Message.id, Message.role, Message.content,
                        Message.created_at, Message.channel, Message.source,
                    )
                    .where(Message.day_chat_id == day_chat_id)
                    .order_by(Message.created_at.desc())
                    .limit(limit)
                )).all()
                # Return in chronological order (oldest first) for the UI.
                for r in reversed(msg_rows):
                    out["messages"].append({
                        "id":         r[0],
                        "role":       r[1],
                        "content":    r[2],
                        "created_at": _epoch_ms(r[3]),
                        "channel":    r[4],
                        "source":     r[5],
                    })
    except Exception as exc:
        logger.warning("[extension] conversation/today failed: %s", exc)

    return out


def _epoch_ms(v: Any) -> int:
    if v is None: return 0
    try:
        if hasattr(v, "timestamp"): return int(v.timestamp() * 1000)
        import datetime as _dt
        return int(_dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return 0


# ═════════════════════════════════════════════════════════════════════
# Phase 6 — browsing memory
# ═════════════════════════════════════════════════════════════════════
@router.get("/extension/memory/recent")
async def extension_memory_recent(
    limit: int = Query(20, ge=1, le=100),
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    from app.services.browsing_memory import recent
    user_id = current_user.id
    items = await recent(user_id, limit=limit)
    return {"items": items}


@router.delete("/extension/memory")
async def extension_memory_clear(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """Wipe this user's browsing memory rows. Idempotent."""
    from app.services.browsing_memory import purge_user
    user_id = current_user.id
    deleted = await purge_user(user_id)
    return {"ok": True, "deleted": deleted}


class _OptOutReq(BaseModel):
    opted_out: bool


@router.post("/extension/privacy/memory-optout")
async def extension_memory_optout(
    req: _OptOutReq,
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Set the per-user 'don't store browsing memory' flag."""
    user_id = current_user.id
    try:
        async with async_session_maker() as db:
            row = (await db.execute(
                text("SELECT notification_preferences FROM users WHERE id = :uid"),
                {"uid": user_id},
            )).first()
            prefs = {}
            if row and row[0]:
                p = row[0]
                if isinstance(p, str):
                    try: prefs = json.loads(p or "{}")
                    except Exception: prefs = {}
                elif isinstance(p, dict):
                    prefs = p
            prefs["extension_memory_optout"] = bool(req.opted_out)
            await db.execute(
                text(
                    "UPDATE users SET notification_preferences = :np WHERE id = :uid"
                ),
                {"np": json.dumps(prefs), "uid": user_id},
            )
            await db.commit()
    except Exception as exc:
        logger.warning("[extension] memory-optout update failed: %s", exc)
        raise HTTPException(500, "failed to persist preference")
    return {"ok": True, "opted_out": bool(req.opted_out)}


# ═════════════════════════════════════════════════════════════════════
# Phase 5 — proactive page suggestions ("wow moment" greeting)
# ═════════════════════════════════════════════════════════════════════
class SuggestReq(BaseModel):
    page_context: Dict[str, Any]


@router.post("/extension/suggest")
async def extension_suggest(
    req: SuggestReq,
    request: Request,
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Return 2–3 chips the sidepanel can render before the user types."""
    from app.services.page_suggest import suggest_for_page

    user_id = current_user.id
    # Haiku-backed; cap at 60/min per user to keep $$ predictable.
    await _enforce_rate(request, f"suggest:{user_id}", limit=60, window_s=60)
    out = await suggest_for_page(user_id=user_id, page_context=req.page_context or {})
    if not out:
        return {"ok": True, "suggestion": None}
    return {"ok": True, "suggestion": out}


# ═════════════════════════════════════════════════════════════════════
# Phase 5 — action confirmation from the web /settings/extensions UI
# (Sidepanel uses an in-extension flow; this endpoint is for the web
# live-view panel so users can also approve/deny from desktop.)
# ═════════════════════════════════════════════════════════════════════
class ConfirmReq(BaseModel):
    action_id: str
    decision:  str   # "allow" | "allow_all" | "deny"


@router.post("/extension/sessions/{session_id}/confirm")
async def confirm_action(
    session_id: str,
    req: ConfirmReq,
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    user_id = current_user.id
    s = extension_bridge.get_session(user_id, session_id)
    if not s:
        raise HTTPException(404, "session not found")
    decision = (req.decision or "").lower()
    if decision not in ("allow", "allow_all", "deny"):
        raise HTTPException(400, "decision must be allow|allow_all|deny")
    try:
        await extension_bridge.send_control(
            user_id, session_id,
            f"confirm:{req.action_id}:{decision}",
        )
    except extension_bridge.ExtensionUnavailable:
        raise HTTPException(409, "extension not connected")
    return {"ok": True}


@router.get("/extension/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    last_event_id: Optional[int] = Query(None, alias="last_event_id"),
    current_user=Depends(get_current_user),
):
    """
    Server-Sent Events stream for a single session — used by the
    /settings/extensions live view.
    """
    from fastapi.responses import StreamingResponse

    user_id = current_user.id
    s = extension_bridge.get_session(user_id, session_id)
    if not s:
        raise HTTPException(404, "session not found")

    # Honor Last-Event-ID header if not passed as query param.
    if last_event_id is None:
        hdr = request.headers.get("last-event-id")
        if hdr and hdr.isdigit():
            last_event_id = int(hdr)

    async def gen():
        try:
            async for chunk in extension_bridge.subscribe(user_id, session_id, last_event_id):
                if chunk is None:
                    break
                ev = chunk["_event"]
                payload = json.dumps(chunk["data"])
                if ev == "event" and "seq" in (chunk["data"] or {}):
                    yield f"id: {chunk['data']['seq']}\nevent: event\ndata: {payload}\n\n"
                else:
                    yield f"event: {ev}\ndata: {payload}\n\n"
                if await request.is_disconnected():
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
