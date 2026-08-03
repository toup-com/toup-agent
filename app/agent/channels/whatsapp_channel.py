"""
WhatsApp Channel Adapter — Connects WhatsApp Business API to Toup agent runtime.

Uses the WhatsApp Cloud API (Meta Business) via webhook:
- Incoming text messages
- Image / document / audio attachments
- Read receipts
- Typing indicators via status updates

Requires:
- WHATSAPP_PHONE_NUMBER_ID
- WHATSAPP_ACCESS_TOKEN (permanent system user token)
- WHATSAPP_VERIFY_TOKEN (webhook verification)
"""

import asyncio
import hashlib
import hmac
import logging
from datetime import datetime
from typing import Optional

import httpx

from app.agent.channels.base import (
    BaseChannel,
    ChannelType,
    InboundMessage,
)
from app.services.background_tasks import spawn as _spawn_bg

logger = logging.getLogger(__name__)

GRAPH_API = "https://graph.facebook.com/v19.0"


# Module-level reference to the active channel instance, set by start()
# and cleared by stop(). The /agent/health endpoint reads this so it
# can call health() without needing to thread the channel through
# FastAPI's app state. Pattern mirrors set_agent_runner / set_ws_refs
# elsewhere in agent_main.py.
_active_channel: Optional["WhatsAppChannel"] = None


def get_active_channel() -> Optional["WhatsAppChannel"]:
    """Return the running WhatsAppChannel instance, or None.

    Used by ``/agent/health`` to surface a rich status dict. Returns
    None when WhatsApp is unconfigured or stopped.
    """
    return _active_channel


class WhatsAppChannel(BaseChannel):
    """
    WhatsApp Cloud API channel adapter.

    Unlike Telegram/Discord/Slack this adapter is *webhook-based*:
    the FastAPI app must mount the verification and inbound webhook
    routes.  Call ``register_routes(app)`` during startup.
    """

    def __init__(
        self,
        phone_number_id: str,
        access_token: str,
        verify_token: str,
        app_secret: Optional[str] = None,
        allowed_numbers: Optional[list] = None,
    ):
        super().__init__(ChannelType.WHATSAPP)
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self.verify_token = verify_token
        self.app_secret = app_secret
        self.allowed_numbers = set(allowed_numbers or [])
        self._http: Optional[httpx.AsyncClient] = None
        self._sweep_task: Optional[asyncio.Task] = None
        # One-shot guard: skip the agent_configs write after we've
        # already flipped whatsapp_connected_at this process.
        self._connected_marked: bool = False
        # Health-surface tracking — exposed via health() to /agent/health.
        # Reset per-process: container restart starts a fresh window.
        self._inbound_count: int = 0
        self._last_inbound_at: Optional[datetime] = None
        self._last_send_at: Optional[datetime] = None
        self._last_send_error: Optional[dict] = None
        # One-shot flag for the "no app_secret configured → HMAC bypass"
        # warning so it surfaces on the first inbound but doesn't spam
        # the log on every webhook thereafter.
        self._unsigned_warned: bool = False

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=GRAPH_API,
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=30.0,
        )
        # Periodic sweep of the inbound dedupe table. Cancel-on-stop
        # so we don't leak the task across container reloads.
        from app.agent.channels.whatsapp_dedupe import run_sweep_loop
        self._sweep_task = asyncio.create_task(run_sweep_loop())

        # Security boot warning: if no app_secret is configured the
        # webhook accepts unsigned payloads. Anyone who guesses the
        # webhook URL can post fake inbounds. Surface loudly so an
        # operator sees this in journalctl on every container start.
        if not self.app_secret:
            logger.warning(
                "[WHATSAPP] SECURITY app_secret not configured — inbound "
                "webhooks accepted WITHOUT HMAC signature verification. "
                "Set WHATSAPP_APP_SECRET in agent_configs to enable."
            )

        global _active_channel
        _active_channel = self
        logger.info("[WHATSAPP] Channel started (phone_id=%s)", self.phone_number_id)

    async def stop(self) -> None:
        global _active_channel
        if _active_channel is self:
            _active_channel = None
        if self._sweep_task:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except (asyncio.CancelledError, Exception):
                pass
            self._sweep_task = None
        if self._http:
            await self._http.aclose()
        logger.info("[WHATSAPP] Channel stopped")

    # ── Health surface ─────────────────────────────────────────

    def health(self) -> dict:
        """Return a status dict suitable for ``/agent/health``.

        Pure read — never raises, never blocks. Designed so an
        operator can ``curl https://agent-X.../agent/health`` and tell
        at a glance whether WhatsApp is wired, whether HMAC is on,
        when the last inbound landed, and what (if anything) failed
        on the most recent send.
        """
        return {
            "configured": bool(self.phone_number_id and self.access_token),
            "started": self._http is not None,
            "phone_number_id": self.phone_number_id or None,
            "app_secret_configured": bool(self.app_secret),
            "verify_token_configured": bool(self.verify_token),
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
            "connected_at_marked_in_db": self._connected_marked,
        }

    # ── Route registration ──────────────────────────────────────

    def register_routes(self, app):
        """Mount webhook verification + inbound routes on FastAPI app."""
        from fastapi import Request, Response

        @app.get("/api/whatsapp/webhook")
        async def verify_webhook(request: Request):
            params = request.query_params
            mode = params.get("hub.mode")
            token = params.get("hub.verify_token")
            challenge = params.get("hub.challenge")
            if mode == "subscribe" and token == self.verify_token:
                logger.info("[WHATSAPP] Webhook verified")
                return Response(content=challenge, media_type="text/plain")
            return Response(status_code=403)

        @app.post("/api/whatsapp/webhook")
        async def inbound_webhook(request: Request):
            body = await request.body()
            # HMAC verification path — when app_secret is unset we
            # accept the request to keep the user unblocked, but we
            # surface a one-shot WARNING so an operator sees the
            # security degrade. The boot-time warning in start() was
            # the first signal; this is the second, only emitted on a
            # real inbound (cuts noise if WhatsApp is configured but
            # never used).
            if self.app_secret:
                sig = request.headers.get("X-Hub-Signature-256", "")
                expected = "sha256=" + hmac.new(
                    self.app_secret.encode(), body, hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(sig, expected):
                    logger.warning(
                        "[WHATSAPP] webhook.signature_invalid — rejected. "
                        "Likely a stale app_secret or a request not from Meta."
                    )
                    return Response(status_code=403)
            elif not self._unsigned_warned:
                self._unsigned_warned = True
                logger.warning(
                    "[WHATSAPP] webhook.unsigned_accepted — first inbound "
                    "received without HMAC verification (app_secret unset). "
                    "Subsequent unsigned webhooks will be accepted silently."
                )

            import json
            payload = json.loads(body)
            asyncio.create_task(self._process_payload(payload))
            return Response(status_code=200)

    async def _process_payload(self, payload: dict):
        """Process inbound WhatsApp Cloud API webhook payload."""
        any_user_message = False
        try:
            for entry in payload.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])
                    contacts = value.get("contacts", [])

                    contact_map = {}
                    for c in contacts:
                        wa_id = c.get("wa_id", "")
                        name = c.get("profile", {}).get("name", wa_id)
                        contact_map[wa_id] = name

                    for msg in messages:
                        any_user_message = True
                        await self._handle_message(msg, contact_map)
        except Exception:
            logger.exception("[WHATSAPP] Error processing payload")

        # First real inbound (post-HMAC, non-handshake) flips the green
        # "Connected" badge in the Settings UI. Fire-and-forget so the
        # webhook handler can return 200 immediately. The in-process
        # guard means we hit the DB at most once per container lifetime.
        if any_user_message and not self._connected_marked:
            _spawn_bg(self._mark_connected_if_first())

    async def _mark_connected_if_first(self) -> None:
        """One-shot UPDATE of agent_configs.whatsapp_connected_at.

        Safe to call from concurrent webhook handlers — the in-process
        ``_connected_marked`` flag short-circuits subsequent calls, and
        the SQL guards on ``whatsapp_connected_at IS NULL`` so even
        concurrent first-callers can only succeed once.
        """
        if self._connected_marked:
            return
        try:
            from app.config import settings as _settings
            from app.db.database import async_session_maker
            from app.db.models import AgentConfig
            from sqlalchemy import update

            user_id = getattr(_settings, "user_id", "") or ""
            if not user_id:
                return
            async with async_session_maker() as db:
                stmt = (
                    update(AgentConfig)
                    .where(AgentConfig.user_id == user_id)
                    .where(AgentConfig.whatsapp_connected_at.is_(None))
                    .values(whatsapp_connected_at=datetime.utcnow())
                )
                result = await db.execute(stmt)
                await db.commit()
                # Mark in-process regardless of whether we updated a
                # row — if the column is already non-null another
                # process / earlier run flipped it; we don't need to
                # try again this lifetime.
                self._connected_marked = True
                if getattr(result, "rowcount", 0):
                    logger.info("[WHATSAPP] connected_at set for the first time")
        except Exception:
            # Non-fatal — Settings UI just stays on "Pending webhook"
            # until the next inbound retries this path.
            logger.warning(
                "[WHATSAPP] failed to set connected_at (non-fatal)", exc_info=True
            )

    async def _handle_message(self, msg: dict, contacts: dict):
        """Handle a single inbound message."""
        from_number = msg.get("from", "")

        # ACL check
        if self.allowed_numbers and from_number not in self.allowed_numbers:
            return

        # Dedupe — Meta retries webhook deliveries on any 5xx / timeout
        # / connection-drop within a 7-day window (densely in the first
        # ~30 minutes). claim() is a single ON-CONFLICT-DO-NOTHING
        # write; if it returns False this is a retry of a message we
        # already processed and we silently drop it. See
        # `app/agent/channels/whatsapp_dedupe.py`.
        from app.agent.channels.whatsapp_dedupe import claim as _dedupe_claim
        from app.agent.channels.whatsapp_helpers import redact_phone
        from app.config import settings as _settings

        message_id = msg.get("id", "")
        owner_user_id = getattr(_settings, "user_id", "") or ""
        if message_id and owner_user_id:
            first_seen = await _dedupe_claim(owner_user_id, message_id)
            if not first_seen:
                logger.info(
                    "[WHATSAPP] dedupe.skipped_retry chat=%s msg_id=%s",
                    redact_phone(from_number),
                    message_id[:16],
                )
                return

        # Health-surface counters: bump after dedupe so a re-delivery
        # doesn't double-count, and after ACL so blocked numbers don't
        # inflate the inbound rate.
        self._inbound_count += 1
        self._last_inbound_at = datetime.utcnow()

        msg_type = msg.get("type", "")
        text = ""
        media_paths = []

        if msg_type == "text":
            text = msg.get("text", {}).get("body", "")
        elif msg_type in ("image", "document", "audio", "video"):
            media_obj = msg.get(msg_type, {})
            media_id = media_obj.get("id")
            caption = media_obj.get("caption", "")
            text = caption
            if media_id:
                url = await self._download_media(media_id)
                if url:
                    media_paths.append(url)
        else:
            # Unsupported type (location, sticker, etc.)
            text = f"[{msg_type} message]"

        display_name = contacts.get(from_number, from_number)

        inbound = InboundMessage(
            channel=ChannelType.WHATSAPP,
            channel_user_id=from_number,
            channel_chat_id=from_number,
            text=text,
            media_paths=media_paths,
            username=from_number,
            display_name=display_name,
            raw=msg,
        )

        await self.dispatch(inbound)

    async def _download_media(self, media_id: str) -> Optional[str]:
        """Download media from WhatsApp Cloud API, return local path."""
        if not self._http:
            return None
        try:
            resp = await self._http.get(f"/{media_id}")
            resp.raise_for_status()
            media_url = resp.json().get("url")
            if not media_url:
                return None
            # Download actual file
            dl = await self._http.get(media_url)
            dl.raise_for_status()
            import tempfile
            suffix = ".bin"
            ct = dl.headers.get("content-type", "")
            if "image/jpeg" in ct:
                suffix = ".jpg"
            elif "image/png" in ct:
                suffix = ".png"
            elif "audio" in ct:
                suffix = ".ogg"
            elif "pdf" in ct:
                suffix = ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(dl.content)
            tmp.close()
            return tmp.name
        except Exception:
            logger.exception("[WHATSAPP] Failed to download media %s", media_id)
            return None

    # ── Outbound ──────────────────────────────────────────────

    async def send_text(self, chat_id: str, text: str, parse_mode: Optional[str] = None) -> None:
        """Send a text message via the WhatsApp Cloud API.

        The full reply is markdown-converted (GFM → WhatsApp), then
        split into ≤4096-char chunks, then sent sequentially. Each
        chunk is retried up to 3 times on transient failures (HTTP
        429 / 5xx, network errors). Non-retryable 4xx responses are
        logged once with a redacted recipient and skipped.

        Outbound logs redact the recipient phone number — never log
        full E.164 to journalctl or our logging pipeline.
        """
        if not self._http:
            return
        from app.agent.channels.whatsapp_helpers import (
            chunk_for_whatsapp,
            markdown_to_whatsapp,
            redact_phone,
        )

        converted = markdown_to_whatsapp(text or "")
        chunks = chunk_for_whatsapp(converted)
        if not chunks:
            return
        redacted = redact_phone(chat_id)
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            await self._send_text_chunk(chat_id, chunk, redacted, idx, total)

    async def _send_text_chunk(
        self,
        chat_id: str,
        body: str,
        redacted: str,
        index: int,
        total: int,
    ) -> None:
        """Send one chunk with bounded retry. See :py:meth:`send_text`."""
        from app.agent.channels.whatsapp_helpers import RETRY_STATUS_CODES

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": chat_id,
            "type": "text",
            "text": {"preview_url": False, "body": body},
        }
        max_attempts = 3
        last_transport_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._http.post(
                    f"/{self.phone_number_id}/messages", json=payload,
                )
            except Exception as exc:
                last_transport_error = exc
                if attempt < max_attempts:
                    logger.warning(
                        "[WHATSAPP] send.transport_error chat=%s chunk=%d/%d attempt=%d err=%s",
                        redacted, index, total, attempt, exc,
                    )
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                logger.error(
                    "[WHATSAPP] send.exhausted chat=%s chunk=%d/%d last_error=%s",
                    redacted, index, total, exc,
                )
                return

            status = response.status_code
            if 200 <= status < 300:
                self._last_send_at = datetime.utcnow()
                self._last_send_error = None
                return
            if status in RETRY_STATUS_CODES and attempt < max_attempts:
                logger.warning(
                    "[WHATSAPP] send.retryable status=%d chat=%s chunk=%d/%d attempt=%d",
                    status, redacted, index, total, attempt,
                )
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            # Non-retryable, or retryable but no attempts left.
            body_excerpt = (response.text or "")[:300]
            logger.error(
                "[WHATSAPP] send.failed status=%d chat=%s chunk=%d/%d body=%s",
                status, redacted, index, total, body_excerpt,
            )
            self._last_send_error = {
                "status": status,
                "at": datetime.utcnow().isoformat() + "Z",
                "message_excerpt": body_excerpt,
            }
            return
        if last_transport_error is not None:
            logger.error(
                "[WHATSAPP] send.exhausted chat=%s chunk=%d/%d last_error=%s",
                redacted, index, total, last_transport_error,
            )
            self._last_send_error = {
                "status": 0,  # 0 = transport error (no HTTP status)
                "at": datetime.utcnow().isoformat() + "Z",
                "message_excerpt": str(last_transport_error)[:300],
            }

    async def send_typing(self, chat_id: str) -> None:
        """WhatsApp doesn't support typing indicators for bots."""
        pass

    async def send_file(self, chat_id: str, file_path: str, caption: Optional[str] = None) -> None:
        """Upload and send a document via WhatsApp Cloud API."""
        if not self._http:
            return
        try:
            media_id = await self._upload_media(file_path, "application/octet-stream")
            if media_id:
                payload = {
                    "messaging_product": "whatsapp",
                    "to": chat_id,
                    "type": "document",
                    "document": {"id": media_id},
                }
                if caption:
                    payload["document"]["caption"] = caption
                await self._http.post(f"/{self.phone_number_id}/messages", json=payload)
        except Exception:
            logger.exception("[WHATSAPP] Failed to send file")

    async def send_photo(self, chat_id: str, photo_path: str, caption: Optional[str] = None) -> None:
        """Upload and send an image via WhatsApp Cloud API."""
        if not self._http:
            return
        try:
            media_id = await self._upload_media(photo_path, "image/jpeg")
            if media_id:
                payload = {
                    "messaging_product": "whatsapp",
                    "to": chat_id,
                    "type": "image",
                    "image": {"id": media_id},
                }
                if caption:
                    payload["image"]["caption"] = caption
                await self._http.post(f"/{self.phone_number_id}/messages", json=payload)
        except Exception:
            logger.exception("[WHATSAPP] Failed to send photo")

    async def _upload_media(self, file_path: str, mime_type: str) -> Optional[str]:
        """Upload media to WhatsApp Cloud API, return media_id."""
        if not self._http:
            return None
        try:
            with open(file_path, "rb") as f:
                resp = await self._http.post(
                    f"/{self.phone_number_id}/media",
                    data={"messaging_product": "whatsapp", "type": mime_type},
                    files={"file": (file_path.split("/")[-1], f, mime_type)},
                )
                resp.raise_for_status()
                return resp.json().get("id")
        except Exception:
            logger.exception("[WHATSAPP] Failed to upload media")
            return None
