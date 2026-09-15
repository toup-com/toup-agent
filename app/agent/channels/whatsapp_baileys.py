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
import time
import uuid
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
# `/pair/status` answers a USER-FACING poll and must never hold one. The
# mobile Channels panel reads it on every visit and every foreground for
# every not-yet-linked user, so a sick sidecar has to cost ~2 s and a cached
# snapshot, not the shared client's full budget. (The blocking client this
# read used to open carried exactly this 2 s.)
_PAIR_STATUS_TIMEOUT_S = 2.0
_SSE_RECONNECT_BACKOFF_S = 2.0
# The sidecar's SSE stream has NO REPLAY (sidecar.mjs `emitEvent` writes to
# currently-attached clients and drops the rest), so a `connection_open` that
# fires while nobody is attached is gone. The cached status is therefore a
# CACHE, not a fact, and it is re-derived from the sidecar on this cadence.
_RECONCILE_INTERVAL_S = 10.0
# Unreachable sidecar: back off geometrically rather than hammer a dead port
# for the life of the container.
_RECONCILE_MAX_BACKOFF_S = 60.0
# …and a FLOOR under the wake. `_wake_reconciler` fires on every SSE attach,
# and a stream that EOFs cleanly re-attaches immediately, so a sick sidecar
# that accepts and closes could otherwise drive back-to-back /health reads
# for as long as it stays sick. No two reconcile reads closer than this.
_RECONCILE_MIN_INTERVAL_S = 1.0


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

# Monotonic token bumped once per restart. An instance that is not on the
# CURRENT generation must stop dispatching: `_stopping` is only ever set by
# `stop()`, and an instance evicted from the registry never receives one —
# which is how a superseded adapter kept a second SSE stream open and ran
# every inbound message a second time.
_generation: int = 0


def get_active_baileys_channel() -> Optional["BaileysWhatsAppChannel"]:
    return _active_channel


def current_generation() -> int:
    return _generation


def bump_generation() -> int:
    global _generation
    _generation += 1
    return _generation


def _signal(name: str) -> None:
    """Best-effort process-lifetime counter. The module is absent on older
    images, and a health counter may never break a WhatsApp turn."""
    try:
        from app.services.health_signals import incr as _incr
        _incr(name)
    except Exception:
        pass


class BaileysWhatsAppChannel(BaseChannel):
    """Baileys-sidecar-backed WhatsApp adapter."""

    def __init__(
        self,
        allowed_numbers: Optional[list[str]] = None,
        generation: int = 0,
    ):
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
        self._generation: int = generation
        self._spawn_token: Optional[str] = None
        self._adopted_sidecar: bool = False

        self._reconcile_task: Optional[asyncio.Task] = None
        self._reconcile_wake: Optional[asyncio.Event] = None

        # Cached state — re-derived from the sidecar by `_reconcile_forever`
        # and advanced early by SSE. Every write goes through
        # `_apply_sidecar_state`; nothing else may assign these three.
        self._session_status: str = "not_linked"
        self._session_status_source: str = "init"
        # Two DIFFERENT clocks, and conflating them is how a cache pretends
        # to be fresh. `_status_since` moves only when the status VALUE
        # changes — it is what tells a stale `linking` (creds wiped, nothing
        # will ever move it) from a reconnect in flight. `_last_sidecar_read_at`
        # moves only on a real READ of the sidecar, so a local optimistic
        # write cannot make the cache look freshly confirmed.
        self._status_since: float = time.monotonic()
        self._last_sidecar_read_at: Optional[float] = None
        # F2: the newest SSE/local write. A `/health` or `/pair/status` body
        # serialised BEFORE it must not be applied after it.
        self._last_push_write_at: float = float("-inf")
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

    def is_current(self) -> bool:
        """True while this instance is the one the process should route on.

        Generation 0 is an UNMANAGED instance — constructed directly rather
        than through `restart_whatsapp_channel`, which is the only caller
        that bumps the counter. It has no rival to be superseded by, so it is
        always current; otherwise a test double or a legacy code path would
        silently drop every inbound the moment any restart had ever run in
        the process."""
        if self._stopping:
            return False
        return self._generation == 0 or self._generation == current_generation()

    async def start(self) -> bool:
        """Spawn the sidecar + open the inbound event stream.

        Degrades gracefully: if Node isn't installed, the sidecar dir
        is missing, or the sidecar fails to boot within the timeout,
        we log + bail without crashing the rest of the agent.

        Returns True only when the SSE stream is attached. The caller must
        not register a False return: `send_text` would then hit its own
        `self._http is None` guard and drop every outbound reply silently.
        """
        global _active_channel

        sidecar_dir = _resolve_sidecar_dir()
        if not (sidecar_dir / "sidecar.mjs").is_file():
            logger.error(
                "[WHATSAPP-BAILEYS] sidecar.mjs not found at %s — "
                "QR-link mode disabled. Rebuild the agent image.",
                sidecar_dir,
            )
            return False
        if not (sidecar_dir / "node_modules").is_dir():
            logger.error(
                "[WHATSAPP-BAILEYS] node_modules missing at %s — "
                "QR-link mode disabled. The Dockerfile.agent should "
                "`npm install` during build.",
                sidecar_dir,
            )
            return False

        node_bin = shutil.which("node")
        if not node_bin:
            logger.error(
                "[WHATSAPP-BAILEYS] `node` binary not on PATH — "
                "QR-link mode disabled. Install Node 20+."
            )
            return False

        # Spawn the sidecar. Its stdout/stderr are inherited so log
        # lines flow into `docker logs <container>` alongside Python
        # output.
        try:
            env = os.environ.copy()
            env.setdefault("WHATSAPP_SIDECAR_PORT", str(_SIDECAR_PORT))
            # Identity for the /health check below: the only way to tell
            # "my child answered" from "a stranger already owns the port".
            self._spawn_token = uuid.uuid4().hex
            env["WHATSAPP_SIDECAR_TOKEN"] = self._spawn_token
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
            _signal("wa_sidecar_spawns")
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] sidecar spawn failed")
            return False

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
                    self._apply_sidecar_state(body, "boot")
                    self._check_sidecar_identity(body)
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
            return False

        # Periodic sweep of the inbound dedupe table — same module the
        # Cloud API path uses.
        try:
            from app.agent.channels.whatsapp_dedupe import run_sweep_loop
            self._sweep_task = asyncio.create_task(run_sweep_loop())
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] dedupe sweep init failed")

        # A restart that started while we were waiting out the boot poll
        # has already minted a newer generation; attaching a second SSE
        # stream here is exactly the duplication this token exists to stop.
        if not self.is_current():
            logger.warning(
                "[WHATSAPP-BAILEYS] start.superseded gen=%d current=%d",
                self._generation, current_generation(),
            )
            await self._teardown()
            return False

        # The wake handle exists BEFORE the stream that fires it: a
        # `_wake_reconciler()` with no Event behind it is a silent no-op,
        # and the attach it reports is the one most worth reacting to.
        self._reconcile_wake = asyncio.Event()

        # Long-lived SSE consumer.
        self._event_task = asyncio.create_task(self._consume_events_forever())

        # …and the thing that makes the SSE stream an OPTIMISATION rather
        # than the source of truth. A `connection_open` emitted between the
        # boot /health read above and the stream actually attaching reaches
        # nobody (no replay), and the cache then reports `linking` while the
        # sidecar is linked — which is what clients render as "Not
        # connected". `_consume_events_forever` wakes this loop the instant a
        # stream attaches, so that window closes at once; failing that, one
        # interval bounds it.
        self._reconcile_task = asyncio.create_task(self._reconcile_forever())

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
                self._apply_sidecar_state(
                    {"session_status": "linking", "connected": False,
                     "self_e164": None},
                    "local",
                )
            except Exception:
                logger.exception(
                    "[WHATSAPP-BAILEYS] auto-pair kick failed — user can "
                    "still trigger from Settings"
                )

        return True

    def _check_sidecar_identity(self, body: dict) -> None:
        """Compare the /health responder against the child we just spawned.

        A MISSING `spawn_token` is UNKNOWN, not a stranger: a sidecar bundle
        that predates the token answers without one, and treating that as a
        foreign process would take WhatsApp dark on every such image.
        """
        theirs = body.get("spawn_token")
        if theirs is None:
            logger.info(
                "[WHATSAPP-BAILEYS] sidecar.spawn_token_absent pid=%s — "
                "sidecar bundle predates the token; identity unverified",
                body.get("pid"),
            )
            return
        if theirs == self._spawn_token:
            return
        logger.error(
            "[WHATSAPP-BAILEYS] sidecar.port_owned_by_stranger ours=%s theirs=%s pid=%s",
            (self._spawn_token or "")[:8], str(theirs)[:8], body.get("pid"),
        )
        self._adopted_sidecar = True
        # Visible to the monitor, not only to /agent/health readers: an
        # adopted stream is a session we can neither restart nor kill, and
        # `restart_whatsapp_channel` excludes it from the config no-op so the
        # next push respawns.
        _signal("wa_sidecar_adopted")
        # Our own child lost the bind and died. Forget it so `_teardown`
        # cannot SIGTERM the stranger's pid — the pid is not ours to kill,
        # and killing it takes down the session that is actually working.
        if (
            self._sidecar_proc is not None
            and getattr(self._sidecar_proc, "returncode", None) is not None
        ):
            self._sidecar_proc = None

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
        for task in (self._event_task, self._sweep_task, self._reconcile_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._event_task = None
        self._sweep_task = None
        self._reconcile_task = None
        self._reconcile_wake = None

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
        while self.is_current():
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
                        # Attached NOW. Everything the sidecar emitted before
                        # this instant was delivered to nobody, so the cache
                        # may already be a lie; re-derive rather than wait out
                        # a whole interval.
                        self._wake_reconciler()
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
        # The loop can be parked mid-event when a newer generation lands;
        # this is the second of the three exits (loop / event / inbound).
        if not self.is_current():
            return
        kind = payload.get("type")

        if kind == "qr":
            # The PNG data URL itself is fetched lazily via
            # /pair/status to avoid bloating SSE frames.
            self._apply_sidecar_state(
                {"session_status": "linking", "connected": self._connected,
                 "self_e164": self._self_e164},
                "sse",
            )
            return

        if kind == "connection_open":
            self._apply_sidecar_state(
                {
                    "session_status": "linked",
                    "connected": True,
                    "self_e164": payload.get("self_e164") or self._self_e164,
                },
                "sse",
            )
            logger.info(
                "[WHATSAPP-BAILEYS] connection.open self=%s",
                redact_phone(self._self_e164 or ""),
            )
            return

        if kind == "logged_out":
            self._apply_sidecar_state(
                {"session_status": "logged_out", "connected": False,
                 "self_e164": None},
                "sse",
            )
            logger.warning("[WHATSAPP-BAILEYS] logged_out — relink required")
            return

        if kind == "message":
            await self._handle_inbound_message(payload)
            return

        # Unknown event types — log at debug, don't fail.
        logger.debug("[WHATSAPP-BAILEYS] event.unknown_type type=%s", kind)

    # ── Sidecar state reconciliation ───────────────────────────

    def _wake_reconciler(self) -> None:
        """Ask for a sidecar re-read on the next loop turn. Never raises."""
        ev = self._reconcile_wake
        if ev is not None:
            ev.set()

    def _apply_sidecar_state(
        self, body: Any, source: str, read_at: Optional[float] = None,
    ) -> None:
        """Adopt the sidecar's view of the session. THE one write path.

        Every producer of session truth — the boot ``/health`` read, an SSE
        frame, the reconciler, the ``/pair/status`` poll, a local pairing
        action — lands here, so the transition log, the freshness stamp and
        the ``wa_status_reconciled`` signal cannot be true of one producer
        and false of another. ``source`` is one of ``boot`` | ``sse`` |
        ``reconcile`` | ``pair_status`` | ``local``.

        ``read_at`` is the monotonic instant the POLLED body was requested.
        A poll is a snapshot of the past: a `/health` answer serialised
        before a `connection_open` frame can still land after it and walk a
        linked session back to `linking`. A polled body older than the
        newest push write is therefore dropped, not merged — it has nothing
        to say that the push has not already said better.
        """
        if not isinstance(body, dict):
            return
        polled = source in ("reconcile", "pair_status", "boot")
        if polled and read_at is not None and read_at < self._last_push_write_at:
            logger.debug(
                "[WHATSAPP-BAILEYS] state.superseded source=%s — a push write "
                "landed while this read was in flight", source,
            )
            return

        prev_status = self._session_status
        prev_connected = self._connected
        prev_self = self._self_e164

        status = body.get("session_status")
        if status:
            self._session_status = str(status)
        self._connected = bool(body.get("connected"))
        self._self_e164 = body.get("self_e164")

        now = time.monotonic()
        self._session_status_source = source
        if polled:
            # Only a real READ refreshes the "we asked the sidecar" clock.
            self._last_sidecar_read_at = now
        else:
            self._last_push_write_at = now
        if self._session_status != prev_status:
            self._status_since = now

        if (
            self._session_status != prev_status
            or self._connected != prev_connected
            or self._self_e164 != prev_self
        ):
            logger.info(
                "[WHATSAPP-BAILEYS] session.transition %s->%s connected=%s "
                "self=%s source=%s",
                prev_status, self._session_status, self._connected,
                redact_phone(self._self_e164 or "") or "none", source,
            )
            if source == "reconcile":
                # The SSE frame that should have carried this never arrived
                # — the stream was not attached when the sidecar emitted it.
                # Diagnostic only; nothing may gate on a health signal.
                _signal("wa_status_reconciled")

    async def _reconcile_once(self, source: str) -> bool:
        """One ``/health`` read applied to the cache. Never raises.

        Returns False when the sidecar could not be read, which is what
        drives the caller's backoff.
        """
        client = self._http
        if client is None:
            return False
        read_at = time.monotonic()
        try:
            resp = await client.get("/health")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "[WHATSAPP-BAILEYS] reconcile.sidecar_unreachable err=%s", exc,
            )
            return False
        try:
            if getattr(resp, "status_code", 0) != 200:
                logger.debug(
                    "[WHATSAPP-BAILEYS] reconcile.bad_status %s",
                    getattr(resp, "status_code", None),
                )
                return False
            body = resp.json()
        except Exception as exc:
            logger.debug("[WHATSAPP-BAILEYS] reconcile.bad_body err=%s", exc)
            return False
        self._apply_sidecar_state(body, source, read_at=read_at)
        return True

    async def _reconcile_forever(self) -> None:
        """Keep the cached session state ≤ one interval behind the sidecar.

        The sidecar is the only process that knows whether WhatsApp is
        linked; SSE is how we hear about it QUICKLY, never how we learn it.
        Sleeps ``_RECONCILE_INTERVAL_S`` between reads, wakes early when the
        SSE stream (re)attaches, and backs off to ``_RECONCILE_MAX_BACKOFF_S``
        while the sidecar is unreachable so a dead port never becomes a spin.
        """
        delay = _RECONCILE_INTERVAL_S
        last_read = float("-inf")
        while self.is_current():
            ev = self._reconcile_wake
            try:
                if ev is None:
                    await asyncio.sleep(delay)
                else:
                    try:
                        await asyncio.wait_for(ev.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
                    ev.clear()
                # The floor. A wake is a HINT, and a sidecar that accepts an
                # SSE connection and immediately EOFs produces one per
                # round-trip; without this, "re-derive on attach" becomes a
                # hot loop against the very process that is already sick.
                gap = time.monotonic() - last_read
                if gap < _RECONCILE_MIN_INTERVAL_S:
                    await asyncio.sleep(_RECONCILE_MIN_INTERVAL_S - gap)
            except asyncio.CancelledError:
                return
            if not self.is_current():
                return
            last_read = time.monotonic()
            try:
                ok = await self._reconcile_once("reconcile")
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("[WHATSAPP-BAILEYS] reconcile.unexpected")
                ok = False
            delay = (
                _RECONCILE_INTERVAL_S
                if ok
                else min(delay * 2, _RECONCILE_MAX_BACKOFF_S)
            )

    async def _handle_inbound_message(self, payload: dict) -> None:
        if not self.is_current():
            return
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
        _claimed = False
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
                    _signal("channel_events_deduped")
                    return
                _claimed = True
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] dedupe.claim_failed")

        if not text:
            logger.info(
                "[WHATSAPP-BAILEYS] inbound.unsupported_kind chat=%s",
                redact_phone(sender_norm),
            )
            return

        # Counted AFTER the kind gate: `channel_events_claimed` is the
        # numerator of the channel-orphan alert (claimed − persisted), so it
        # must count only events the pipeline promised to persist. A
        # sticker or voice note claims the dedupe row and legitimately
        # never reaches the handler — counted above the gate, one photo
        # paged "message vanished" every 30 minutes for the container's
        # life.
        if _claimed:
            _signal("channel_events_claimed")

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

        On agents, ``agent_configs`` is in PLATFORM_ONLY_TABLES so the
        table doesn't exist on the tenant DB. We still try AgentConfig
        first (so the same code path works in monolith / platform-mode
        runs), but the exception is caught locally rather than aborting
        the whole lookup. The real source of truth on agents is the
        ``Identity`` table, which the Soul page's `/api/soul/sync` keeps
        up to date with rows like ``Identity(name='doodool Soul')``.

        No caching: one indexed SELECT per outbound message is cheap,
        and skipping the cache means a Soul-page rename takes effect on
        the very next reply rather than requiring a restart.
        """
        from app.config import settings as _s
        from app.db.database import async_session_maker
        from app.db.models import AgentConfig, Identity
        from sqlalchemy import select, and_

        user_id = getattr(_s, "user_id", None)
        if not user_id:
            logger.warning("[WHATSAPP-BAILEYS] agent_name.no_user_id — settings.user_id empty")
            return None

        # Each query gets its own session so a failed query (e.g.
        # UndefinedTableError on agent containers, where agent_configs
        # doesn't exist) doesn't poison the transaction for the next.
        try:
            async with async_session_maker() as db:
                row = await db.execute(
                    select(AgentConfig.agent_name).where(AgentConfig.user_id == user_id)
                )
                name = row.scalar_one_or_none()
                if name and name.strip():
                    return name.strip()
        except Exception as e:
            # Expected on agent tenants — agent_configs lives on the
            # platform DB only. Log at debug, not exception, so we
            # don't spam logs once per outbound message.
            logger.debug(
                "[WHATSAPP-BAILEYS] agent_name.agent_config_unavailable user=%s: %s",
                user_id[:8], type(e).__name__,
            )

        # Fallback to Identity(type='soul').name (e.g. "doodool Soul").
        # This is the canonical source on agents because soul.py:sync_soul
        # upserts it from the platform's PUT /api/soul.
        try:
            async with async_session_maker() as db:
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
            logger.exception("[WHATSAPP-BAILEYS] agent_name.identity_lookup_failed")
        return None

    async def send_typing(self, chat_id: str) -> None:
        """Best-effort typing presence. The sidecar doesn't expose a
        typing endpoint yet (Baileys' presence API is per-conversation
        and rarely visible on linked-device sessions). No-op for now.
        """
        return

    # ── QR pairing API (consumed by /qr-* endpoints) ───────────

    async def get_pairing_status(self) -> dict:
        """Snapshot for the ``/qr-status`` polling endpoint.

        Reads ``/pair/status`` over the SHARED async client. It used to open
        a blocking ``httpx.Client`` here — and this runs inside a FastAPI
        route on the agent's event loop, which the Settings modal polls
        every ~1.5 s, so every poll stalled the whole loop for as long as
        the sidecar took to answer, up to the client timeout.

        Returns the sidecar's body with the four cache-authoritative keys
        forced onto it: a body that omits one must not delete it from the
        response, which is how a sparse answer used to hand ``/qr/status``
        a snapshot with no ``session_status`` at all.
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
        client = self._http
        if client is None:
            return snapshot
        read_at = time.monotonic()
        try:
            resp = await client.get(
                "/pair/status", timeout=_PAIR_STATUS_TIMEOUT_S,
            )
            if getattr(resp, "status_code", 0) == 200:
                body = resp.json()
                # Cache for next call + for SSE-driven status display.
                self._apply_sidecar_state(body, "pair_status", read_at=read_at)
                self._latest_qr_data_url = body.get("qr_data_url")
                qr_at = body.get("qr_emitted_at")
                if qr_at:
                    try:
                        self._latest_qr_at = datetime.fromisoformat(
                            qr_at.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
                merged = dict(snapshot)
                if isinstance(body, dict):
                    merged.update(body)
                merged["session_status"] = self._session_status
                merged["connected"] = self._connected
                merged["self_e164"] = self._self_e164
                merged["qr_data_url"] = self._latest_qr_data_url
                merged.update(self._status_provenance())
                return merged
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[WHATSAPP-BAILEYS] pair_status.fetch_failed %s", exc)
        snapshot.update(self._status_provenance())
        return snapshot

    def _status_provenance(self) -> dict:
        """How old the cached status is and what last wrote it.

        Rides BOTH the ``/qr/status`` body and ``health()`` because the
        platform's no-downgrade rule needs the same three facts whichever
        route it heard from — and a snapshot served from cache during a
        sidecar outage must still say so rather than look fresh.
        """
        return {
            "session_status_source": self._session_status_source,
            "session_status_stable_s": self._status_stable_s(),
            "since_last_sidecar_read_s": self._sidecar_read_age_s(),
        }

    async def kick_pair(self) -> None:
        """Tell the sidecar to wipe auth + start a fresh QR flow."""
        if self._http is None:
            return
        try:
            await self._http.post("/pair/start")
            self._apply_sidecar_state(
                {"session_status": "linking", "connected": False,
                 "self_e164": None},
                "local",
            )
            self._latest_qr_data_url = None
            self._latest_qr_at = None
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] pair_start.failed")

    async def request_pairing_code(self, phone: str) -> str:
        """Mint an 8-character pairing code for single-device WhatsApp
        linking (no QR). The user enters the returned code in their own
        WhatsApp app under Linked Devices → "Link with phone number
        instead." Used by the mobile onboarding flow where the user
        can't scan their own screen.

        Same teardown semantics as ``kick_pair()`` — wipes any existing
        session before requesting the code. After this returns,
        ``get_pairing_status()`` reports "linking" until the user
        confirms in WhatsApp; clients poll it for completion.
        """
        if self._http is None:
            raise RuntimeError("Baileys sidecar HTTP client not initialised")
        resp = await self._http.post("/pair/code", json={"phone": phone})
        resp.raise_for_status()
        body = resp.json()
        if not body.get("ok"):
            raise RuntimeError(body.get("error") or "Pairing-code request failed")
        self._apply_sidecar_state(
            {"session_status": "linking", "connected": False, "self_e164": None},
            "local",
        )
        self._latest_qr_data_url = None
        self._latest_qr_at = None
        return str(body.get("pairing_code") or "")

    async def force_logout(self) -> None:
        """User clicked "Disconnect" in Settings."""
        if self._http is not None:
            try:
                await self._http.post("/pair/logout")
            except Exception:
                logger.exception("[WHATSAPP-BAILEYS] logout.sidecar_call_failed")
        self._apply_sidecar_state(
            {"session_status": "not_linked", "connected": False,
             "self_e164": None},
            "local",
        )
        self._latest_qr_data_url = None
        self._latest_qr_at = None
        logger.info("[WHATSAPP-BAILEYS] force_logout — session cleared")

    # ── Health surface ─────────────────────────────────────────

    def _sidecar_read_age_s(self) -> Optional[int]:
        """Seconds since we last READ the sidecar. None if we never have."""
        if self._last_sidecar_read_at is None:
            return None
        return int(max(0.0, time.monotonic() - self._last_sidecar_read_at))

    def _status_stable_s(self) -> int:
        """Seconds the session_status VALUE has held, whatever wrote it."""
        return int(max(0.0, time.monotonic() - self._status_since))

    def health(self) -> dict:
        """Mirrors the shape of ``WhatsAppChannel.health()`` so
        ``/agent/health`` doesn't branch on transport mode.
        """
        return {
            "configured": True,
            # "started" used to mean "we once called create_subprocess_exec",
            # which stayed True for an instance whose child died milliseconds
            # later on EADDRINUSE. The SSE task's liveness is the fact an
            # operator actually needs: no stream, no inbound.
            "started": self._event_task is not None and not self._event_task.done(),
            "generation": self._generation,
            "is_current": self.is_current(),
            "adopted_sidecar": self._adopted_sidecar,
            "sidecar_alive": (
                self._sidecar_proc is not None
                and getattr(self._sidecar_proc, "returncode", None) is None
            ),
            "mode": "qr_link",
            "session_status": self._session_status,
            # `session_status` is a CACHE, and these three say how much to
            # trust it. `session_status_stable_s` is the one the platform's
            # downgrade rule reads: a `linking` that has not moved in a
            # minute is a session whose credentials are gone, while a
            # reconnect in flight is seconds old. `since_last_sidecar_read_s`
            # is a different question — when did we last ASK — and a local
            # optimistic write deliberately does not refresh it. Strings and
            # ints are fine here; only `health_signals` is ints-only.
            "session_status_source": self._session_status_source,
            "session_status_stable_s": self._status_stable_s(),
            "since_last_sidecar_read_s": self._sidecar_read_age_s(),
            "reconciler_running": (
                self._reconcile_task is not None
                and not self._reconcile_task.done()
            ),
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
