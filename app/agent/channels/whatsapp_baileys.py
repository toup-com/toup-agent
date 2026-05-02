"""WhatsApp QR-link channel adapter (Path C) — Baileys sidecar bridge.

We tried `neonize` (Python wrapper around the Go `whatsmeow` binary)
first. Pairing kept failing with "Can't link new devices right now"
on the scanning phone — even though the same number successfully
paired the real WhatsApp Desktop app a minute later. Root cause:
Meta's anti-abuse rejects WhatsApp Web clients advertising stale
`version` triples, and `neonize` 0.3.x ships an outdated
`whatsmeow` build. Baileys exposes `fetchLatestBaileysVersion()`
which polls Meta's own check-update endpoint for the freshest
accepted triple, so it stays current automatically.

Architecture
------------
* **Sidecar** — Node.js process at `backend/whatsapp_sidecar/`
  speaks Baileys to Meta. Listens on `127.0.0.1:8002`
  (container-internal only). Auth lives at `/data/whatsapp/auth/`
  (multi-file Baileys layout, survives container restart).
* **This adapter** — spawns the sidecar at `start()`, polls health
  until ready, then opens a long-lived SSE stream to receive
  inbound messages. Outbound goes via `POST /messages/send`. The
  Settings UI's QR pairing endpoints proxy through `force_logout()`
  → sidecar `/pair/logout` and `kick_pair()` → sidecar `/pair/start`.

Production behaviours preserved from the previous implementation:

* **ACL** — `whatsapp_baileys_allowlist` (E.164 list). Anything
  from outside the list is silently dropped at the gate. Empty
  allowlist = block all (secure default).
* **Dedupe** — DB-backed `whatsapp_inbound_dedupe` table the Cloud
  API path uses; Meta retransmits to linked devices on flaky
  networks; without dedupe a transport blip would re-run the LLM.
* **Day-as-Chat** — every inbound flows through the existing
  `make_channel_handler` so messages share the same
  `Message.day_chat_id` as web / Telegram / voice on the same
  local date.
* **Outbound polish** — reuses `markdown_to_whatsapp` / chunking /
  `redact_phone` from `whatsapp_helpers`.
* **Reconnect** — Baileys handles reconnection internally; the
  sidecar exposes connection state via `/health`. We don't run a
  Python-side supervisor anymore.
* **Health surface** — same shape as the Cloud API `health()` so
  `/agent/health` rendering doesn't branch on mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from app.agent.channels.base import (
    BaseChannel,
    ChannelType,
    InboundMessage,
)
from app.agent.channels.whatsapp_helpers import (
    chunk_for_whatsapp,
    markdown_to_whatsapp,
    redact_phone,
)

logger = logging.getLogger(__name__)


def _normalize_e164(raw: str) -> str:
    """Best-effort E.164 normalization for ACL comparison.

    Users paste numbers in many forms — ``+1 (415) 555-2671``,
    ``415.555.2671``, ``00 1 415 555 2671`` — and we want all of them
    to match the JID-derived ``+14155552671`` from inbound. Strategy:
    keep digits only, strip a leading "00" (international-call
    prefix), prepend ``+``. Empty input returns ``""``.
    """
    if not raw:
        return ""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("00"):
        digits = digits[2:]
    return f"+{digits}" if digits else ""


# ── Sidecar config ─────────────────────────────────────────────────

_SIDECAR_HOST = "127.0.0.1"
_SIDECAR_PORT = int(os.environ.get("WHATSAPP_SIDECAR_PORT", "8002"))
_SIDECAR_BASE = f"http://{_SIDECAR_HOST}:{_SIDECAR_PORT}"
_SIDECAR_BOOT_TIMEOUT_S = 20.0   # how long start() waits for /health 200
_SIDECAR_HTTP_TIMEOUT_S = 10.0
_SSE_RECONNECT_BACKOFF_S = 2.0


def _resolve_sidecar_dir() -> Path:
    """Find the directory containing `sidecar.mjs` + `node_modules/`.

    Production layout (`Dockerfile.agent`) puts it at
    `/app/whatsapp_sidecar/`. Local dev / tests can override via
    `WHATSAPP_SIDECAR_DIR`. Falls back to a relative path resolved
    from this file's location so test environments without an explicit
    env var still work.
    """
    override = os.environ.get("WHATSAPP_SIDECAR_DIR")
    if override:
        return Path(override)
    # /app/whatsapp_sidecar — production container layout.
    prod = Path("/app/whatsapp_sidecar")
    if (prod / "sidecar.mjs").is_file():
        return prod
    # Repo-relative fallback: backend/whatsapp_sidecar
    repo_path = Path(__file__).resolve().parents[3] / "whatsapp_sidecar"
    return repo_path


# ── Module-level reference (consumed by `/agent/health` + QR API) ──

_active_channel: Optional["BaileysWhatsAppChannel"] = None


def get_active_baileys_channel() -> Optional["BaileysWhatsAppChannel"]:
    return _active_channel


class BaileysWhatsAppChannel(BaseChannel):
    """Baileys-sidecar-backed WhatsApp adapter."""

    def __init__(self, allowed_numbers: Optional[list[str]] = None):
        super().__init__(ChannelType.WHATSAPP)
        self.allowed_numbers: set[str] = {
            normalised
            for raw in (allowed_numbers or [])
            for normalised in [_normalize_e164(raw)]
            if normalised
        }
        # Sidecar lifecycle
        self._sidecar_proc: Optional[asyncio.subprocess.Process] = None
        self._event_task: Optional[asyncio.Task] = None
        self._sweep_task: Optional[asyncio.Task] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._stopping: bool = False

        # Cached state — refreshed on each /pair/status or sidecar event.
        self._session_status: str = "not_linked"
        self._connected: bool = False
        self._self_e164: Optional[str] = None
        self._latest_qr_data_url: Optional[str] = None
        self._latest_qr_at: Optional[datetime] = None
        self._inbound_count: int = 0
        self._last_inbound_at: Optional[datetime] = None
        self._last_send_at: Optional[datetime] = None
        self._last_send_error: Optional[dict] = None
        self._sidecar_booted_at: Optional[datetime] = None

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn the sidecar + open the inbound event stream.

        Degrades gracefully: if Node isn't installed, the sidecar dir
        is missing, or the sidecar fails to boot within the timeout,
        we log + bail without crashing the rest of the agent.
        """
        global _active_channel

        sidecar_dir = _resolve_sidecar_dir()
        if not (sidecar_dir / "sidecar.mjs").is_file():
            logger.error(
                "[WHATSAPP-BAILEYS] sidecar.mjs not found at %s — "
                "QR-link mode disabled. Rebuild the agent image.",
                sidecar_dir,
            )
            return
        if not (sidecar_dir / "node_modules").is_dir():
            logger.error(
                "[WHATSAPP-BAILEYS] node_modules missing at %s — "
                "QR-link mode disabled. The Dockerfile.agent should "
                "`npm install` during build.",
                sidecar_dir,
            )
            return

        node_bin = shutil.which("node")
        if not node_bin:
            logger.error(
                "[WHATSAPP-BAILEYS] `node` binary not on PATH — "
                "QR-link mode disabled. Install Node 20+."
            )
            return

        # Spawn the sidecar. Its stdout/stderr are inherited so log
        # lines flow into `docker logs <container>` alongside Python
        # output.
        try:
            env = os.environ.copy()
            env.setdefault("WHATSAPP_SIDECAR_PORT", str(_SIDECAR_PORT))
            self._sidecar_proc = await asyncio.create_subprocess_exec(
                node_bin,
                "sidecar.mjs",
                cwd=str(sidecar_dir),
                env=env,
                stdout=sys.stdout.fileno() if hasattr(sys.stdout, "fileno") else None,
                stderr=sys.stderr.fileno() if hasattr(sys.stderr, "fileno") else None,
            )
            logger.info(
                "[WHATSAPP-BAILEYS] sidecar spawned pid=%s port=%s dir=%s",
                self._sidecar_proc.pid, _SIDECAR_PORT, sidecar_dir,
            )
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] sidecar spawn failed")
            return

        # Wait for /health to come up — Node startup takes ~1-2s with
        # warm node_modules; allow up to _SIDECAR_BOOT_TIMEOUT_S.
        self._http = httpx.AsyncClient(
            base_url=_SIDECAR_BASE, timeout=_SIDECAR_HTTP_TIMEOUT_S,
        )
        deadline = asyncio.get_event_loop().time() + _SIDECAR_BOOT_TIMEOUT_S
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await self._http.get("/health")
                if resp.status_code == 200:
                    body = resp.json()
                    self._sidecar_booted_at = datetime.utcnow()
                    self._session_status = body.get("session_status") or "not_linked"
                    self._connected = bool(body.get("connected"))
                    self._self_e164 = body.get("self_e164")
                    logger.info(
                        "[WHATSAPP-BAILEYS] sidecar healthy session=%s connected=%s",
                        self._session_status, self._connected,
                    )
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            logger.error(
                "[WHATSAPP-BAILEYS] sidecar didn't reach /health within %ds — bailing",
                int(_SIDECAR_BOOT_TIMEOUT_S),
            )
            await self._teardown()
            return

        # Periodic sweep of the inbound dedupe table — same module the
        # Cloud API path uses.
        try:
            from app.agent.channels.whatsapp_dedupe import run_sweep_loop
            self._sweep_task = asyncio.create_task(run_sweep_loop())
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] dedupe sweep init failed")

        # Long-lived SSE consumer.
        self._event_task = asyncio.create_task(self._consume_events_forever())

        _active_channel = self
        logger.info(
            "[WHATSAPP-BAILEYS] Channel started — sidecar=%s allowlist_size=%d",
            _SIDECAR_BASE, len(self.allowed_numbers),
        )

        # Self-healing: if the sidecar booted without on-disk Baileys
        # creds (fresh container, post-rollout volume wipe, or first-
        # time setup) it sits idle in `not_linked` until something
        # calls `/pair/start`. Without this proactive kick, the user
        # sees "linked" in the UI from stale DB state, sends a
        # message, and gets no response — exactly the symptom we just
        # debugged. Auto-fire `/pair/start` so a fresh QR is always
        # ready the moment the user opens Settings, no extra click
        # required.
        if self._session_status == "not_linked":
            logger.info(
                "[WHATSAPP-BAILEYS] no creds on boot — auto-kicking pair flow"
            )
            try:
                await self._http.post("/pair/start")
                self._session_status = "linking"
            except Exception:
                logger.exception(
                    "[WHATSAPP-BAILEYS] auto-pair kick failed — user can "
                    "still trigger from Settings"
                )

    async def stop(self) -> None:
        """Graceful shutdown — cancels the SSE consumer + sweep, then
        SIGTERMs the sidecar and waits for it to exit."""
        self._stopping = True
        global _active_channel
        if _active_channel is self:
            _active_channel = None
        await self._teardown()

    async def _teardown(self) -> None:
        # Cancel background tasks first so they don't observe a
        # dying sidecar mid-iteration.
        for task in (self._event_task, self._sweep_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._event_task = None
        self._sweep_task = None

        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None

        if self._sidecar_proc is not None:
            try:
                self._sidecar_proc.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._sidecar_proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._sidecar_proc.kill()
                    await self._sidecar_proc.wait()
            except ProcessLookupError:
                pass
            except Exception:
                logger.exception("[WHATSAPP-BAILEYS] sidecar shutdown failed")
            self._sidecar_proc = None

        logger.info("[WHATSAPP-BAILEYS] Channel stopped")

    # ── SSE consumer ───────────────────────────────────────────

    async def _consume_events_forever(self) -> None:
        """Stream inbound events from the sidecar's SSE endpoint.

        SSE drops cleanly when the sidecar restarts; we reconnect
        with a small backoff so a sidecar bounce doesn't strand us.
        """
        backoff = _SSE_RECONNECT_BACKOFF_S
        while not self._stopping:
            try:
                async with httpx.AsyncClient(
                    base_url=_SIDECAR_BASE, timeout=None,
                ) as client:
                    async with client.stream("GET", "/events") as resp:
                        if resp.status_code != 200:
                            logger.warning(
                                "[WHATSAPP-BAILEYS] sse.bad_status %d — backing off %.1fs",
                                resp.status_code, backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                        backoff = _SSE_RECONNECT_BACKOFF_S  # reset on success
                        async for line in resp.aiter_lines():
                            if not line or line.startswith(":"):
                                continue  # heartbeats / blank lines
                            if not line.startswith("data:"):
                                continue
                            payload_str = line[len("data:"):].strip()
                            if not payload_str:
                                continue
                            try:
                                payload = json.loads(payload_str)
                            except json.JSONDecodeError:
                                logger.debug(
                                    "[WHATSAPP-BAILEYS] sse.malformed_json %s",
                                    payload_str[:200],
                                )
                                continue
                            try:
                                await self._on_sidecar_event(payload)
                            except Exception:
                                logger.exception(
                                    "[WHATSAPP-BAILEYS] event.handler_failed"
                                )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "[WHATSAPP-BAILEYS] sse.disconnected err=%s — backing off %.1fs",
                    exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.6, 30.0)

    async def _on_sidecar_event(self, payload: dict) -> None:
        """Route a single SSE event into the channel's state machine."""
        kind = payload.get("type")

        if kind == "qr":
            # The PNG data URL itself is fetched lazily via
            # /pair/status to avoid bloating SSE frames.
            self._session_status = "linking"
            return

        if kind == "connection_open":
            self._connected = True
            self._session_status = "linked"
            self._self_e164 = payload.get("self_e164") or self._self_e164
            logger.info(
                "[WHATSAPP-BAILEYS] connection.open self=%s",
                redact_phone(self._self_e164 or ""),
            )
            return

        if kind == "logged_out":
            self._connected = False
            self._session_status = "logged_out"
            self._self_e164 = None
            logger.warning("[WHATSAPP-BAILEYS] logged_out — relink required")
            return

        if kind == "message":
            await self._handle_inbound_message(payload)
            return

        # Unknown event types — log at debug, don't fail.
        logger.debug("[WHATSAPP-BAILEYS] event.unknown_type type=%s", kind)

    async def _handle_inbound_message(self, payload: dict) -> None:
        sender_e164 = (payload.get("from") or "").strip()
        text = (payload.get("text") or "").strip()
        message_id = (payload.get("message_id") or "").strip()
        push_name = payload.get("push_name") or sender_e164

        # ACL gate. Empty allowlist = block all.
        sender_norm = _normalize_e164(sender_e164)
        if not sender_norm or sender_norm not in self.allowed_numbers:
            logger.info(
                "[WHATSAPP-BAILEYS] inbound.blocked_by_acl chat=%s",
                redact_phone(sender_norm) or "unknown",
            )
            return

        # Dedupe — same DB table the Cloud API path uses.
        try:
            from app.agent.channels.whatsapp_dedupe import claim as _dedupe_claim
            from app.config import settings as _settings
            owner_user_id = getattr(_settings, "user_id", "") or ""
            if message_id and owner_user_id:
                first_seen = await _dedupe_claim(owner_user_id, message_id)
                if not first_seen:
                    logger.info(
                        "[WHATSAPP-BAILEYS] dedupe.skipped_retry chat=%s msg_id=%s",
                        redact_phone(sender_norm), message_id[:16],
                    )
                    return
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] dedupe.claim_failed")

        if not text:
            logger.info(
                "[WHATSAPP-BAILEYS] inbound.unsupported_kind chat=%s",
                redact_phone(sender_norm),
            )
            return

        self._inbound_count += 1
        self._last_inbound_at = datetime.utcnow()

        inbound = InboundMessage(
            channel=ChannelType.WHATSAPP,
            channel_user_id=sender_norm,
            channel_chat_id=sender_norm,
            text=text,
            media_paths=[],
            username=push_name,
            display_name=push_name,
            raw=payload,
        )
        await self.dispatch(inbound)

    # ── Outbound ───────────────────────────────────────────────

    async def send_text(
        self, chat_id: str, text: str, parse_mode: Optional[str] = None
    ) -> None:
        """Send a text message via the sidecar.

        ``chat_id`` is the recipient's E.164 (matches
        ``InboundMessage.channel_chat_id``). The sidecar accepts E.164
        directly and converts to a JID internally.
        """
        if self._http is None:
            logger.warning(
                "[WHATSAPP-BAILEYS] send.no_session chat=%s — sidecar down, dropping",
                redact_phone(chat_id),
            )
            self._last_send_error = {
                "status": 0,
                "at": datetime.utcnow().isoformat() + "Z",
                "message_excerpt": "sidecar not started",
            }
            return

        converted = markdown_to_whatsapp(text or "")
        chunks = chunk_for_whatsapp(converted)
        if not chunks:
            return

        display_name = await self._resolve_agent_display_name()

        redacted = redact_phone(chat_id)
        for idx, chunk in enumerate(chunks, start=1):
            try:
                resp = await self._http.post(
                    "/messages/send",
                    json={"to": chat_id, "text": chunk, "display_name": display_name},
                )
                if resp.status_code != 200:
                    body = resp.text[:300]
                    logger.warning(
                        "[WHATSAPP-BAILEYS] send.http_%d chat=%s chunk=%d/%d body=%s",
                        resp.status_code, redacted, idx, len(chunks), body,
                    )
                    self._last_send_error = {
                        "status": resp.status_code,
                        "at": datetime.utcnow().isoformat() + "Z",
                        "message_excerpt": body,
                    }
                    return
                self._last_send_at = datetime.utcnow()
                self._last_send_error = None
            except Exception as exc:
                logger.exception(
                    "[WHATSAPP-BAILEYS] send.failed chat=%s chunk=%d/%d",
                    redacted, idx, len(chunks),
                )
                self._last_send_error = {
                    "status": 0,
                    "at": datetime.utcnow().isoformat() + "Z",
                    "message_excerpt": str(exc)[:300],
                }
                return

    async def _resolve_agent_display_name(self) -> Optional[str]:
        """Look up the user's chosen agent name for self-chat headers.

        Order of preference (mirrors `telegram_bot._cmd_start`):
          1. ``agent_configs.agent_name`` (set by Soul page's VPS sync)
          2. ``identities.name`` for the active ``soul`` record, with
             the trailing " Soul" stripped
          3. None — sidecar's "Agent" default kicks in

        No caching: a single indexed SELECT per outbound message is
        cheap, and skipping the cache means a Soul-page rename takes
        effect on the very next reply rather than requiring a restart.
        """
        try:
            from app.config import settings as _s
            from app.db.database import async_session_maker
            from app.db.models import AgentConfig, Identity
            from sqlalchemy import select, and_
            user_id = getattr(_s, "user_id", None)
            if not user_id:
                logger.warning("[WHATSAPP-BAILEYS] agent_name.no_user_id — settings.user_id empty")
                return None
            async with async_session_maker() as db:
                row = await db.execute(
                    select(AgentConfig.agent_name).where(AgentConfig.user_id == user_id)
                )
                name = row.scalar_one_or_none()
                if name and name.strip():
                    return name.strip()
                logger.info(
                    "[WHATSAPP-BAILEYS] agent_name.no_agent_config_row user=%s — falling back to Identity",
                    user_id[:8],
                )
                # Fallback to Identity(type='soul').name (e.g. "doodool Soul")
                row = await db.execute(
                    select(Identity.name).where(and_(
                        Identity.user_id == user_id,
                        Identity.identity_type == "soul",
                        Identity.is_active == True,
                    ))
                )
                ident = row.scalar_one_or_none()
                if ident:
                    cleaned = ident[:-5].strip() if ident.endswith(" Soul") else ident.strip()
                    if cleaned:
                        return cleaned
                logger.warning(
                    "[WHATSAPP-BAILEYS] agent_name.no_identity_row user=%s — sidecar default 'Agent' will be used",
                    user_id[:8],
                )
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] agent_name.lookup_failed")
        return None

    async def send_typing(self, chat_id: str) -> None:
        """Best-effort typing presence. The sidecar doesn't expose a
        typing endpoint yet (Baileys' presence API is per-conversation
        and rarely visible on linked-device sessions). No-op for now.
        """
        return

    # ── QR pairing API (consumed by /qr-* endpoints) ───────────

    def get_pairing_status(self) -> dict:
        """Snapshot for the ``/qr-status`` polling endpoint.

        Reaches into the sidecar synchronously via httpx — this method
        is called from inside FastAPI request handlers which are
        already async-friendly, but the API surface must stay sync to
        avoid breaking older callers. We use a short blocking client.
        """
        # Default snapshot uses cached state if sidecar is down.
        snapshot = {
            "session_status": self._session_status,
            "connected": self._connected,
            "self_e164": self._self_e164,
            "qr_data_url": self._latest_qr_data_url,
            "qr_emitted_at": (
                self._latest_qr_at.isoformat() + "Z"
                if self._latest_qr_at else None
            ),
        }
        try:
            with httpx.Client(base_url=_SIDECAR_BASE, timeout=2.0) as client:
                resp = client.get("/pair/status")
                if resp.status_code == 200:
                    body = resp.json()
                    # Cache for next call + for SSE-driven status display.
                    self._session_status = body.get("session_status") or self._session_status
                    self._connected = bool(body.get("connected"))
                    self._self_e164 = body.get("self_e164")
                    self._latest_qr_data_url = body.get("qr_data_url")
                    qr_at = body.get("qr_emitted_at")
                    if qr_at:
                        try:
                            self._latest_qr_at = datetime.fromisoformat(
                                qr_at.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass
                    return body
        except Exception as exc:
            logger.debug("[WHATSAPP-BAILEYS] pair_status.fetch_failed %s", exc)
        return snapshot

    async def kick_pair(self) -> None:
        """Tell the sidecar to wipe auth + start a fresh QR flow."""
        if self._http is None:
            return
        try:
            await self._http.post("/pair/start")
            self._session_status = "linking"
            self._connected = False
            self._self_e164 = None
            self._latest_qr_data_url = None
            self._latest_qr_at = None
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] pair_start.failed")

    async def force_logout(self) -> None:
        """User clicked "Disconnect" in Settings."""
        if self._http is not None:
            try:
                await self._http.post("/pair/logout")
            except Exception:
                logger.exception("[WHATSAPP-BAILEYS] logout.sidecar_call_failed")
        self._session_status = "not_linked"
        self._connected = False
        self._self_e164 = None
        self._latest_qr_data_url = None
        self._latest_qr_at = None
        logger.info("[WHATSAPP-BAILEYS] force_logout — session cleared")

    # ── Health surface ─────────────────────────────────────────

    def health(self) -> dict:
        """Mirrors the shape of ``WhatsAppChannel.health()`` so
        ``/agent/health`` doesn't branch on transport mode.
        """
        return {
            "configured": True,
            "started": self._sidecar_proc is not None,
            "mode": "qr_link",
            "session_status": self._session_status,
            "connected": self._connected,
            "self_e164": self._self_e164,
            "allowed_numbers_count": len(self.allowed_numbers),
            "inbound_count": self._inbound_count,
            "last_inbound_at": (
                self._last_inbound_at.isoformat() + "Z"
                if self._last_inbound_at else None
            ),
            "last_send_at": (
                self._last_send_at.isoformat() + "Z"
                if self._last_send_at else None
            ),
            "last_send_error": self._last_send_error,
            "sidecar_pid": (
                self._sidecar_proc.pid if self._sidecar_proc else None
            ),
            "sidecar_booted_at": (
                self._sidecar_booted_at.isoformat() + "Z"
                if self._sidecar_booted_at else None
            ),
        }
