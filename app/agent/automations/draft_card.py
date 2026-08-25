"""Draft-staged card for the proactive draft flow (Round 29).

When an automation's write is `gmail__create_draft` /
`outlook__create_draft`, the executed outbox row leaves a real draft
sitting in the user's drafts folder — and Round 28 surfaced that only
as "Completed — action taken." This module writes the session card
that makes the outcome tangible: who the draft answers, what it says,
where to open it, and the one truth the flow is named for — nothing
was sent.

Shape (CONTRACTS-R29 §4): message-level
`draft_card: {provider, sender, subject, preview, open_url}` plus a
same-named WS frame with NO channel key. `open_url` is best-effort —
Outlook's Graph `webLink` when the provider returned one, Gmail's
drafts folder otherwise; a card with no link still tells the truth.

Composition discipline: `sender`/`subject`/`preview` come from the
STAGED payload (the automation's own rendered write — data the user's
spec composed), never from a provider response body.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DRAFT_TOOLS = frozenset({"gmail__create_draft", "outlook__create_draft"})

_GMAIL_DRAFTS_URL = "https://mail.google.com/mail/u/0/#drafts"


def draft_card_payload(
    *,
    tool_name: str,
    connector_id: str,
    staged_payload: dict,
    result_content: Optional[dict],
) -> dict:
    provider = "outlook" if connector_id == "outlook" else "gmail"
    body = str(staged_payload.get("body") or "")
    preview = " ".join(body.split())[:160]
    open_url: Optional[str] = None
    content = result_content or {}
    if provider == "outlook":
        link = content.get("webLink")
        open_url = str(link) if link else None
    else:
        msg_id = content.get("id")
        open_url = (
            f"{_GMAIL_DRAFTS_URL}/{msg_id}" if msg_id else _GMAIL_DRAFTS_URL
        )
    return {
        "provider": provider,
        "sender": str(staged_payload.get("to") or ""),
        "subject": str(staged_payload.get("subject") or ""),
        "preview": preview,
        "open_url": open_url,
    }


async def write_draft_card(db, row, result: dict) -> Optional[str]:
    """Persist the card in the automation's session and broadcast.
    `row` is the executed AutomationOutbox row; `result` the dispatch
    envelope (kind == "ok"). Best-effort end to end."""
    if row.tool_name not in DRAFT_TOOLS:
        return None
    try:
        from app.db.models import Automation
        from .session import write_session_message

        automation = await db.get(Automation, row.automation_id)
        if automation is None:
            return None
        try:
            staged = json.loads(row.payload_json) if row.payload_json else {}
        except (ValueError, TypeError):
            staged = {}
        content = result.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                content = {}
        card = draft_card_payload(
            tool_name=row.tool_name,
            connector_id=row.connector_id,
            staged_payload=staged if isinstance(staged, dict) else {},
            result_content=content if isinstance(content, dict) else {},
        )
        provider_name = "Outlook" if card["provider"] == "outlook" else "Gmail"
        to = card["sender"] or "the sender"
        text = (
            f"Drafted a reply to {to} — it's waiting in your "
            f"{provider_name} drafts. Nothing was sent."
        )
        msg_id, _day = await write_session_message(
            db,
            user_id=row.user_id,
            automation_id=row.automation_id,
            content=text,
            metadata={"draft_card": card},
            title=automation.name,
        )
        if msg_id:
            try:
                from app.api.ws_chat import broadcast_to_user
                await broadcast_to_user(row.user_id, {
                    "type": "draft_card",
                    "message_id": msg_id,
                    "automation_id": row.automation_id,
                    "run_id": row.job_id,
                    **card,
                })
            except Exception as e:  # noqa: BLE001 — no socket is normal
                logger.debug("[automations] draft card broadcast skipped: %s", e)
        return msg_id
    except Exception as e:  # noqa: BLE001 — a card never fails a send
        logger.warning(
            "[automations] draft card write failed outbox=%s: %s",
            str(getattr(row, "id", "?"))[:8], e,
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None
