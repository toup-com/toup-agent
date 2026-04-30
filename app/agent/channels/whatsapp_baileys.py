"""WhatsApp QR-link channel adapter (Path C).

Built on `neonize` — a Python wrapper around the Go `whatsmeow`
library. The user pairs a dedicated phone number ("B") with the agent
by scanning a QR code on first run; thereafter the agent's neonize
session is a "linked device" on B's WhatsApp account, and Meta routes
inbound messages to all linked devices simultaneously.

Mutually exclusive with ``WhatsAppChannel`` (Cloud API). Both extend
``BaseChannel(ChannelType.WHATSAPP)`` and only one is registered per
container, gated on ``settings.whatsapp_mode`` at boot.

Production behaviours (mirrored from the Cloud API adapter so ops
parity stays consistent):

* **ACL** — ``whatsapp_baileys_allowlist`` (E.164 list). Anything from
  outside the list is silently dropped at the gate. Empty allowlist
  = block all (secure default; the user must explicitly opt-in their
  own number when they pair).
* **Dedupe** — same DB-backed ``whatsapp_inbound_dedupe`` table the
  Cloud API path uses. Meta retransmits to linked devices on flaky
  networks; without dedupe a transport blip would re-run the LLM.
* **Day-as-Chat** — every inbound flows through the existing shared
  ``make_channel_handler`` so messages share the same
  ``Message.day_chat_id`` as web / Telegram / voice on the same local
  date. Zero WhatsApp-specific code in the agent runtime.
* **Outbound polish** — reuses ``markdown_to_whatsapp`` / chunking /
  ``redact_phone`` from ``whatsapp_helpers``.
* **Reconnect** — bounded-backoff supervisor loop wraps neonize's
  ``connect()``. Logged-out (401-equivalent) and conflict states are
  terminal: we mark ``whatsapp_session_status='logged_out'`` and stop
  retrying so the user sees "needs relink" in Settings.
* **Health surface** — same shape as the Cloud API ``health()`` so
  ``/agent/health`` rendering doesn't branch on mode.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

from app.agent.channels.base import (
    BaseChannel,
    ChannelType,
    InboundMessage,
)


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
from app.agent.channels.whatsapp_baileys_session import (
    e164_to_jid,
    ensure_store_dir,
    jid_to_e164,
    render_qr_png_data_url,
    session_file_exists,
    store_path,
)
from app.agent.channels.whatsapp_helpers import (
    chunk_for_whatsapp,
    markdown_to_whatsapp,
    redact_phone,
)

logger = logging.getLogger(__name__)


# Module-level reference. /agent/health and the QR-pairing endpoints
# fetch the live channel via get_active_baileys_channel(). Mirrors the
# pattern in whatsapp_channel.py (Cloud API adapter).
_active_channel: Optional["BaileysWhatsAppChannel"] = None


def get_active_baileys_channel() -> Optional["BaileysWhatsAppChannel"]:
    return _active_channel


# Reconnect policy — match OpenClaw's defaults so behaviour is
# predictable for anyone who's debugged Baileys before.
_RECONNECT_INITIAL_S = 2.0
_RECONNECT_MAX_S = 30.0
_RECONNECT_FACTOR = 1.8
_RECONNECT_MAX_ATTEMPTS = 12

# QR strings rotate quickly. Cache the latest emitted string + a
# pre-rendered data URL keyed by emit time so a /qr-status poll can
# return the freshest QR even if the client missed the WS event.
_QR_TTL_SECONDS = 180  # 3 min — neonize regenerates every ~20s while pending


class BaileysWhatsAppChannel(BaseChannel):
    """neonize-backed WhatsApp adapter."""

    def __init__(self, allowed_numbers: Optional[list[str]] = None):
        super().__init__(ChannelType.WHATSAPP)
        # Normalise allowlist to canonical E.164 ("+" + digits only).
        # Users paste numbers in mixed formats; the inbound side
        # always derives a clean E.164 from the JID, so both sides
        # must converge on the same form. _normalize_e164 strips
        # whitespace, parens, dashes, dots; handles "00" prefix.
        # Empty set = block everyone (secure default before pairing).
        self.allowed_numbers: set[str] = {
            normalised
            for raw in (allowed_numbers or [])
            for normalised in [_normalize_e164(raw)]
            if normalised
        }
        self._client = None  # neonize.aioze.client.NewAClient instance
        self._supervisor_task: Optional[asyncio.Task] = None
        self._sweep_task: Optional[asyncio.Task] = None
        self._stopping: bool = False

        # QR pairing state
        self._latest_qr: Optional[str] = None
        self._latest_qr_data_url: Optional[str] = None
        self._latest_qr_at: Optional[datetime] = None

        # Lifecycle / health surface
        self._self_e164: Optional[str] = None
        self._session_status: str = (
            "linked" if session_file_exists() else "not_linked"
        )
        self._connected: bool = False
        self._inbound_count: int = 0
        self._last_inbound_at: Optional[datetime] = None
        self._last_send_at: Optional[datetime] = None
        self._last_send_error: Optional[dict] = None
        self._reconnect_attempts: int = 0

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Boot the supervisor loop and the inbound-dedupe sweep.

        ``connect()`` is called inside the supervisor so we control
        the reconnect schedule. We do NOT block ``start()`` on a
        successful connect — if the session is fresh and the user
        hasn't scanned the QR yet, ``start()`` returns immediately and
        the QR gets emitted asynchronously via the QR event handler.

        Degrades gracefully when prerequisites are missing — neonize
        not installed, ``/data`` not mounted, etc. — so a startup
        failure here doesn't take down the whole agent runtime.
        """
        global _active_channel

        # Acquire the neonize client lazily so a missing dep surfaces
        # at start-time with a clear error rather than at module
        # import.
        try:
            from neonize.aioze.client import NewAClient  # type: ignore
        except ImportError:
            logger.error(
                "[WHATSAPP-BAILEYS] neonize is not installed — "
                "QR-link mode is disabled. Add `neonize>=0.4` to "
                "requirements.txt and rebuild."
            )
            return

        # Provision the on-disk store. In Contabo containers this is a
        # bind-mounted /data volume; in dev environments it may not
        # exist — log + bail rather than crash the whole agent boot.
        try:
            ensure_store_dir()
        except OSError as exc:
            logger.error(
                "[WHATSAPP-BAILEYS] cannot create store dir — %s. "
                "QR-link mode disabled. Ensure /data is writable in "
                "the container's bind mount.",
                exc,
            )
            return

        self._client = NewAClient(str(store_path()))
        self._register_event_handlers()

        # Periodic sweep of the inbound dedupe table — same module the
        # Cloud API path uses. Cancel-on-stop so we don't leak across
        # container reloads.
        from app.agent.channels.whatsapp_dedupe import run_sweep_loop
        self._sweep_task = asyncio.create_task(run_sweep_loop())
        self._supervisor_task = asyncio.create_task(self._run_supervisor())
        _active_channel = self
        logger.info(
            "[WHATSAPP-BAILEYS] Channel started — store=%s allowlist_size=%d",
            store_path(), len(self.allowed_numbers),
        )

    async def stop(self) -> None:
        """Graceful shutdown — cancels the supervisor + sweep, closes
        the neonize client. Identity-aware: only clears the global
        active reference if it still points at *this* instance.
        """
        self._stopping = True
        global _active_channel
        if _active_channel is self:
            _active_channel = None

        if self._supervisor_task:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._supervisor_task = None

        if self._sweep_task:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except (asyncio.CancelledError, Exception):
                pass
            self._sweep_task = None

        if self._client is not None:
            try:
                # neonize's async client exposes disconnect(); guard
                # for API drift.
                disconnect = getattr(self._client, "disconnect", None)
                if disconnect:
                    res = disconnect()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                logger.exception("[WHATSAPP-BAILEYS] disconnect failed")
            self._client = None

        logger.info("[WHATSAPP-BAILEYS] Channel stopped")

    # ── Supervisor loop ────────────────────────────────────────

    async def _run_supervisor(self) -> None:
        """Wraps neonize's ``connect()`` in a bounded-backoff loop.

        ``connect()`` blocks while the session is alive and returns /
        raises on disconnect. On unexpected exit we sleep for a
        backoff window and try again. Logged-out / conflict states
        flip ``_session_status`` and we stop retrying so the user sees
        "needs relink" in Settings.
        """
        backoff = _RECONNECT_INITIAL_S
        while not self._stopping:
            try:
                logger.info(
                    "[WHATSAPP-BAILEYS] supervisor.connecting attempt=%d",
                    self._reconnect_attempts + 1,
                )
                # neonize's async connect() returns a Task — it does NOT
                # block until disconnect. Awaiting the function only
                # awaits its setup; we have to await the returned Task
                # itself to block on the actual session lifetime.
                # Without this, the supervisor spins in a tight loop
                # logging "attempt=1" forever and never lets neonize
                # actually finish connecting.
                connect_task = await self._client.connect()  # type: ignore[union-attr]
                if connect_task is not None:
                    await connect_task
                # Clean exit: connect() returned without error — only
                # happens on graceful stop. Bail.
                if self._stopping:
                    return
            except asyncio.CancelledError:
                return
            except Exception as exc:
                # Distinguish terminal states from transient ones. The
                # exact classification depends on neonize's error
                # surface — we use string matching defensively.
                msg = str(exc).lower()
                if "logged out" in msg or "loggedout" in msg or "401" in msg:
                    logger.warning(
                        "[WHATSAPP-BAILEYS] supervisor.logged_out — "
                        "session is terminal, user must relink"
                    )
                    self._session_status = "logged_out"
                    self._connected = False
                    return
                if "conflict" in msg or "440" in msg:
                    logger.warning(
                        "[WHATSAPP-BAILEYS] supervisor.conflict — "
                        "another linked device displaced us"
                    )
                    self._session_status = "logged_out"
                    self._connected = False
                    return
                self._reconnect_attempts += 1
                if self._reconnect_attempts >= _RECONNECT_MAX_ATTEMPTS:
                    logger.error(
                        "[WHATSAPP-BAILEYS] supervisor.exhausted attempts=%d "
                        "last_error=%s",
                        self._reconnect_attempts, exc,
                    )
                    return
                logger.warning(
                    "[WHATSAPP-BAILEYS] supervisor.transient err=%s "
                    "backing off %.1fs (attempt %d/%d)",
                    exc, backoff, self._reconnect_attempts, _RECONNECT_MAX_ATTEMPTS,
                )
                self._connected = False
                await asyncio.sleep(backoff)
                backoff = min(backoff * _RECONNECT_FACTOR, _RECONNECT_MAX_S)
                continue
            # Successful clean-exit path: reset backoff for next loop.
            backoff = _RECONNECT_INITIAL_S

    # ── Event handlers (registered with neonize) ───────────────

    def _register_event_handlers(self) -> None:
        """Wire neonize event callbacks. Defensive about API drift —
        we look up event classes by attribute access so a neonize
        version that renames a symbol surfaces as a clear log line
        instead of an import-time crash.
        """
        if self._client is None:
            return
        try:
            import neonize.events as ev  # type: ignore
        except ImportError:
            logger.error("[WHATSAPP-BAILEYS] neonize.events unavailable")
            return

        message_ev = getattr(ev, "MessageEv", None)
        qr_ev = getattr(ev, "QREv", None) or getattr(ev, "PairingQREv", None)
        pair_ev = getattr(ev, "PairStatusEv", None) or getattr(ev, "PairSuccessEv", None)
        connected_ev = getattr(ev, "ConnectedEv", None)
        disconnected_ev = getattr(ev, "DisconnectedEv", None)
        logged_out_ev = getattr(ev, "LoggedOutEv", None) or getattr(ev, "LogoutEv", None)

        if message_ev:
            self._client.event(message_ev)(self._on_message_ev)  # type: ignore
        # QR is special in neonize — it goes through ``event.qr(callback)``,
        # NOT ``event(QREv)(callback)``. The default handler just prints
        # the QR to stdout via segno.make_qr().terminal(); to capture
        # the bytes, we override via the .qr setter. Callback signature
        # is ``(client, bytes)``.
        try:
            qr_setter = getattr(self._client.event, "qr", None)
            if callable(qr_setter):
                qr_setter(self._on_qr_bytes)  # type: ignore
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] qr setter registration failed")
        if pair_ev:
            self._client.event(pair_ev)(self._on_pair_ev)  # type: ignore
        if connected_ev:
            self._client.event(connected_ev)(self._on_connected_ev)  # type: ignore
        if disconnected_ev:
            self._client.event(disconnected_ev)(self._on_disconnected_ev)  # type: ignore
        if logged_out_ev:
            self._client.event(logged_out_ev)(self._on_logged_out_ev)  # type: ignore

    async def _on_qr_bytes(self, _client, qr_data: bytes) -> None:
        """Direct QR-bytes callback — registered via ``event.qr(self._on_qr_bytes)``.
        neonize delivers the raw pairing string here as bytes.
        """
        try:
            qr_str = qr_data.decode("utf-8") if isinstance(qr_data, (bytes, bytearray)) else str(qr_data)
        except Exception:
            qr_str = ""
        if not qr_str:
            return
        self._latest_qr = qr_str
        self._latest_qr_data_url = render_qr_png_data_url(qr_str)
        self._latest_qr_at = datetime.utcnow()
        self._session_status = "linking"
        logger.info("[WHATSAPP-BAILEYS] qr.emitted len=%d", len(qr_str))

    async def _on_qr_ev(self, _client, qr_payload) -> None:
        """neonize emits the pairing string repeatedly while waiting
        for the user to scan. We render it once per emit and cache so
        ``/qr-status`` polls return the freshest value without
        re-rendering on every poll.

        ``qr_payload`` is a ``QREv`` protobuf with a ``Codes`` field —
        a list of pairing strings (whatsmeow rotates them every ~20 s
        until the user scans). We always pick the first.
        """
        qr_str = ""
        try:
            codes = getattr(qr_payload, "Codes", None)
            if codes:
                first = codes[0]
                qr_str = first.decode("utf-8") if isinstance(first, (bytes, bytearray)) else str(first)
            elif isinstance(qr_payload, (bytes, bytearray)):
                qr_str = qr_payload.decode("utf-8")
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] qr.parse_failed")
            qr_str = ""
        if not qr_str:
            logger.warning("[WHATSAPP-BAILEYS] qr.empty payload_type=%s", type(qr_payload).__name__)
            return
        self._latest_qr = qr_str
        self._latest_qr_data_url = render_qr_png_data_url(qr_str)
        self._latest_qr_at = datetime.utcnow()
        self._session_status = "linking"
        logger.info("[WHATSAPP-BAILEYS] qr.emitted len=%d", len(qr_str))

    async def _on_pair_ev(self, _client, payload) -> None:
        """Fired on first successful pair. ``payload.ID`` carries the
        scanned device's JID; we extract its E.164 for display.
        """
        self._session_status = "linked"
        self._latest_qr = None
        self._latest_qr_data_url = None
        try:
            jid = getattr(payload, "ID", None) or getattr(payload, "id", None)
            jid_str = str(jid) if jid is not None else ""
            self._self_e164 = jid_to_e164(jid_str) or self._self_e164
        except Exception:
            pass
        logger.info(
            "[WHATSAPP-BAILEYS] pair.success self=%s",
            redact_phone(self._self_e164 or ""),
        )

    async def _on_connected_ev(self, _client, _payload) -> None:
        self._connected = True
        self._reconnect_attempts = 0
        self._session_status = "linked"
        logger.info("[WHATSAPP-BAILEYS] connected")

    async def _on_disconnected_ev(self, _client, _payload) -> None:
        self._connected = False
        logger.info("[WHATSAPP-BAILEYS] disconnected")

    async def _on_logged_out_ev(self, _client, _payload) -> None:
        self._connected = False
        self._session_status = "logged_out"
        logger.warning("[WHATSAPP-BAILEYS] logged_out — relink required")

    async def _on_message_ev(self, _client, message) -> None:
        """The hot path: convert neonize's message struct into Toup's
        ``InboundMessage`` and call ``BaseChannel.dispatch()``.
        """
        try:
            await self._handle_message(message)
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] message_handler.unhandled")

    async def _handle_message(self, message) -> None:
        # Extract sender + body. neonize message layout is whatsmeow's
        # protobuf shape; we access defensively so an unexpected
        # message kind (status, reaction, system) just gets dropped.
        sender_jid = ""
        message_id = ""
        text = ""
        is_from_me = False
        try:
            info = getattr(message, "Info", None)
            if info is not None:
                src = getattr(info, "MessageSource", None)
                if src is not None:
                    sender = getattr(src, "Sender", None)
                    if sender is not None:
                        sender_jid = str(sender) if sender else ""
                    is_from_me = bool(getattr(src, "IsFromMe", False))
                message_id = str(getattr(info, "ID", "") or "")
            content = getattr(message, "Message", None)
            if content is not None:
                text = (
                    getattr(content, "conversation", "")
                    or getattr(getattr(content, "extendedTextMessage", None), "text", "")
                    or ""
                )
        except Exception:
            pass

        if is_from_me:
            return  # echo of our own outbound

        sender_e164 = jid_to_e164(sender_jid)
        chat_label = redact_phone(sender_e164)

        # ACL gate. Empty allowlist = block all.
        if not sender_e164 or sender_e164 not in self.allowed_numbers:
            logger.info(
                "[WHATSAPP-BAILEYS] inbound.blocked_by_acl chat=%s",
                chat_label or "unknown",
            )
            return

        # Dedupe — same DB table the Cloud API path uses. Surviving
        # container restarts is exactly the case Meta retries hit.
        from app.agent.channels.whatsapp_dedupe import claim as _dedupe_claim
        from app.config import settings as _settings
        owner_user_id = getattr(_settings, "user_id", "") or ""
        if message_id and owner_user_id:
            first_seen = await _dedupe_claim(owner_user_id, message_id)
            if not first_seen:
                logger.info(
                    "[WHATSAPP-BAILEYS] dedupe.skipped_retry chat=%s msg_id=%s",
                    chat_label, message_id[:16],
                )
                return

        if not text:
            # Drop non-text messages for now (media support lands in a
            # later commit). The dedupe claim above means a retry with
            # text won't be deduped because no claim was made.
            logger.info(
                "[WHATSAPP-BAILEYS] inbound.unsupported_kind chat=%s",
                chat_label,
            )
            return

        self._inbound_count += 1
        self._last_inbound_at = datetime.utcnow()

        inbound = InboundMessage(
            channel=ChannelType.WHATSAPP,
            channel_user_id=sender_e164,
            channel_chat_id=sender_e164,
            text=text,
            media_paths=[],
            username=sender_e164,
            display_name=sender_e164,
            raw=message,
        )
        await self.dispatch(inbound)

    # ── Outbound ───────────────────────────────────────────────

    async def send_text(
        self, chat_id: str, text: str, parse_mode: Optional[str] = None
    ) -> None:
        """Send a text message via the linked-device session.

        ``chat_id`` is the recipient's E.164 (matches ``InboundMessage.
        channel_chat_id`` from the inbound handler). We convert to a
        whatsmeow JID at send time, apply the markdown→WA transform,
        chunk to fit WhatsApp's per-message limit, and send each chunk
        sequentially.
        """
        if self._client is None or not self._connected:
            logger.warning(
                "[WHATSAPP-BAILEYS] send.no_session chat=%s — dropping",
                redact_phone(chat_id),
            )
            self._last_send_error = {
                "status": 0,
                "at": datetime.utcnow().isoformat() + "Z",
                "message_excerpt": "session not connected",
            }
            return

        jid = e164_to_jid(chat_id)
        if not jid:
            logger.warning(
                "[WHATSAPP-BAILEYS] send.invalid_chat_id chat=%s",
                redact_phone(chat_id),
            )
            return

        converted = markdown_to_whatsapp(text or "")
        chunks = chunk_for_whatsapp(converted)
        if not chunks:
            return

        redacted = redact_phone(chat_id)
        for idx, chunk in enumerate(chunks, start=1):
            try:
                # neonize's send_message accepts a JID + body. The
                # exact API call may be `send_message`, `SendMessage`,
                # or wrapped in a content type — we try the canonical
                # form first.
                send_message = getattr(self._client, "send_message", None)
                if send_message is None:
                    raise RuntimeError("neonize client has no send_message")
                result = send_message(jid, chunk)
                if asyncio.iscoroutine(result):
                    await result
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

    async def send_typing(self, chat_id: str) -> None:
        """Best-effort typing presence. Not all WhatsApp Web clients
        show this for linked devices; we attempt it and swallow
        errors.
        """
        if self._client is None or not self._connected:
            return
        try:
            send_presence = getattr(self._client, "send_chat_presence", None)
            if send_presence is None:
                return
            jid = e164_to_jid(chat_id)
            if not jid:
                return
            result = send_presence(jid, "composing")
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    # ── QR pairing API (consumed by /qr-* endpoints) ───────────

    def get_pairing_status(self) -> dict:
        """Snapshot for the ``/qr-status`` polling endpoint."""
        emitted_at = (
            self._latest_qr_at.isoformat() + "Z"
            if self._latest_qr_at else None
        )
        return {
            "session_status": self._session_status,
            "connected": self._connected,
            "self_e164": self._self_e164,
            "qr_data_url": self._latest_qr_data_url,
            "qr_emitted_at": emitted_at,
        }

    async def force_logout(self) -> None:
        """User clicked "Disconnect" in Settings. Tear down session
        and remove on-disk state so the next pair starts fresh.
        """
        from app.agent.channels.whatsapp_baileys_session import clear_session_files
        try:
            if self._client is not None:
                logout = getattr(self._client, "logout", None)
                if logout:
                    res = logout()
                    if asyncio.iscoroutine(res):
                        await res
        except Exception:
            logger.exception("[WHATSAPP-BAILEYS] logout call failed")
        clear_session_files()
        self._session_status = "not_linked"
        self._connected = False
        self._self_e164 = None
        self._latest_qr = None
        self._latest_qr_data_url = None
        logger.info("[WHATSAPP-BAILEYS] force_logout — session cleared")

    # ── Health surface ─────────────────────────────────────────

    def health(self) -> dict:
        """Mirrors the shape of ``WhatsAppChannel.health()`` so
        ``/agent/health`` doesn't branch on transport mode.
        """
        return {
            "configured": True,
            "started": self._client is not None,
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
            "reconnect_attempts": self._reconnect_attempts,
        }
