"""
WhatsApp Channel Adapter — Connects WhatsApp Business API to HexBrain agent runtime.

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
from typing import Optional

import httpx

from app.agent.channels.base import (
    BaseChannel,
    ChannelType,
    InboundMessage,
)

logger = logging.getLogger(__name__)

GRAPH_API = "https://graph.facebook.com/v19.0"


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

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=GRAPH_API,
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=30.0,
        )
        logger.info("[WHATSAPP] Channel started (phone_id=%s)", self.phone_number_id)

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()
        logger.info("[WHATSAPP] Channel stopped")

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
            # Verify signature if app_secret set
            if self.app_secret:
                sig = request.headers.get("X-Hub-Signature-256", "")
                expected = "sha256=" + hmac.new(
                    self.app_secret.encode(), body, hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(sig, expected):
                    return Response(status_code=403)

            import json
            payload = json.loads(body)
            asyncio.create_task(self._process_payload(payload))
            return Response(status_code=200)

    async def _process_payload(self, payload: dict):
        """Process inbound WhatsApp Cloud API webhook payload."""
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
                        await self._handle_message(msg, contact_map)
        except Exception:
            logger.exception("[WHATSAPP] Error processing payload")

    async def _handle_message(self, msg: dict, contacts: dict):
        """Handle a single inbound message."""
        from_number = msg.get("from", "")

        # ACL check
        if self.allowed_numbers and from_number not in self.allowed_numbers:
            return

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
        """Send a text message via WhatsApp Cloud API."""
        if not self._http:
            return
        try:
            await self._http.post(
                f"/{self.phone_number_id}/messages",
                json={
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": chat_id,
                    "type": "text",
                    "text": {"preview_url": False, "body": text[:4096]},
                },
            )
        except Exception:
            logger.exception("[WHATSAPP] Failed to send text to %s", chat_id)

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
