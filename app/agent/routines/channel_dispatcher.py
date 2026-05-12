"""Fan a routine's summary out to one or more delivery channels.

The website (Day-as-Chat) is always written via `message_writer` —
that's the canonical record and stays untouched. THIS module adds
parallel pushes to outbound user channels (Telegram, WhatsApp) when
the routine's `config_json.delivery_channels` list asks for them.

Design rules:
  • Best-effort. A single channel failure must NOT mark the routine as
    failed — the user still got their summary on the website, and the
    other channels still got it too. Per-channel exceptions are logged
    and swallowed.
  • Connectivity-gated. If the user didn't set up Telegram, asking for
    Telegram delivery is a logged no-op, not an error. Same for
    WhatsApp.
  • Channel-aware formatting. The body is sent as-is; each adapter's
    own `send_text` already handles its own markdown/length quirks
    (the WhatsApp adapter converts GFM and chunks at 4096 chars,
    Telegram parses MarkdownV2).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from sqlalchemy import select

logger = logging.getLogger(__name__)


# Valid channel slugs. The "website" slug is recognised but is a no-op
# here — Day-as-Chat write is upstream of this module. Listing it
# explicitly keeps the routine config readable ("where does this go?"
# answered by reading config alone, no implicit defaults).
KNOWN_CHANNELS = frozenset({"website", "telegram", "whatsapp"})


def parse_delivery_channels(config_json: Optional[dict]) -> list[str]:
    """Extract + normalise the delivery_channels list from a routine
    config blob. Returns ['website'] when the config is missing or empty
    so legacy routines (created before this field existed) keep working
    exactly as before."""
    if not config_json:
        return ["website"]
    raw = config_json.get("delivery_channels")
    if not raw:
        return ["website"]
    if isinstance(raw, str):
        raw = [raw]
    cleaned = [str(c).strip().lower() for c in raw if c]
    # Drop unknowns rather than fail — a typo shouldn't break delivery.
    valid = [c for c in cleaned if c in KNOWN_CHANNELS]
    # Always include 'website' so the user can find the result on the
    # dashboard even if they only picked Telegram/WhatsApp. The website
    # write also creates the historical record in routine_runs.
    if "website" not in valid:
        valid = ["website", *valid]
    # De-dupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in valid:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


async def deliver_to_extra_channels(
    *,
    user_id: str,
    delivery_channels: Sequence[str],
    routine_name: str,
    content: str,
    db_session_maker,
) -> dict[str, str]:
    """Send `content` to every requested channel except 'website'.

    Returns a `{channel: result_tag}` map for observability — each value
    is one of: 'sent', 'no_recipient', 'no_adapter', 'error'.

    `routine_name` is prepended to the body as a soft header so the user
    can tell WHICH routine just buzzed their phone, especially when
    multiple routines deliver to the same channel.
    """
    targets = [c for c in delivery_channels if c != "website"]
    if not targets:
        return {}

    body = _format_body(routine_name, content)
    results: dict[str, str] = {}

    for ch in targets:
        try:
            if ch == "telegram":
                results["telegram"] = await _send_telegram(user_id, body, db_session_maker)
            elif ch == "whatsapp":
                results["whatsapp"] = await _send_whatsapp(body, db_session_maker)
            else:
                results[ch] = "no_adapter"
        except Exception as e:
            # Per-channel guard — one channel falling over must not block
            # the others. The handler decided this was best-effort.
            logger.warning(
                "[routine_dispatch] channel=%s user=%s err=%s: %s",
                ch, user_id, type(e).__name__, str(e)[:160],
            )
            results[ch] = "error"

    return results


def _format_body(routine_name: str, content: str) -> str:
    """Prefix the body with a short identifying header. On Telegram the
    bot already attaches its display name; on WhatsApp the self-chat
    bubble shows the same body for "me" and "agent" so the header is
    load-bearing for legibility."""
    if not routine_name:
        return content
    return f"🔁 *{routine_name}*\n\n{content}"


# ── Telegram ────────────────────────────────────────────────────────


async def _send_telegram(user_id: str, body: str, db_session_maker) -> str:
    """Send via the agent's Telegram bot to the owner's mapped chat.

    Resolves the chat_id from `TelegramUserMapping` — the same table the
    bot's inbound handler populates on first contact. If the user has
    never DM'd the bot, the mapping doesn't exist and we return
    `no_recipient`."""
    bot = _get_telegram_bot()
    if bot is None or getattr(bot, "app", None) is None:
        return "no_adapter"

    from app.db.models import TelegramUserMapping

    async with db_session_maker() as db:
        chat_id = await db.scalar(
            select(TelegramUserMapping.telegram_id).where(
                TelegramUserMapping.user_id == user_id,
            ).limit(1)
        )

    if not chat_id:
        logger.info(
            "[routine_dispatch] telegram skipped user=%s reason=no_mapping — "
            "owner has not DM'd the bot yet",
            user_id,
        )
        return "no_recipient"

    try:
        await bot.app.bot.send_message(
            chat_id=int(chat_id),
            text=body,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
    except Exception as e:
        # Markdown parsing can blow up on weird LLM output — retry plain.
        logger.info(
            "[routine_dispatch] telegram markdown failed (%s), retrying plain",
            type(e).__name__,
        )
        await bot.app.bot.send_message(
            chat_id=int(chat_id),
            text=body,
            disable_web_page_preview=True,
        )
    return "sent"


def _get_telegram_bot():
    """Late-import to avoid pulling agent_main into test-only paths.
    `agent_main._telegram_bot` is set when the agent's Telegram channel
    boots; None when the user hasn't configured a bot token."""
    try:
        import agent_main  # type: ignore
        return getattr(agent_main, "_telegram_bot", None)
    except Exception:
        return None


# ── WhatsApp ────────────────────────────────────────────────────────


async def _send_whatsapp(body: str, db_session_maker) -> str:
    """Send via whichever WhatsApp adapter is registered (Cloud API or
    Baileys/QR-link). Recipient is the user's own E.164 — same self-chat
    pattern used by the in-conversation flow."""
    from app.agent.channels.registry import ChannelRegistry
    from app.agent.channels.base import ChannelType
    from app.db.models import AgentConfig

    channel = ChannelRegistry.get(ChannelType.WHATSAPP)
    if channel is None:
        return "no_adapter"

    async with db_session_maker() as db:
        cfg = await db.scalar(select(AgentConfig).limit(1))

    recipient = getattr(cfg, "whatsapp_self_e164", None) if cfg else None
    if not recipient:
        logger.info(
            "[routine_dispatch] whatsapp skipped reason=no_self_e164 — "
            "channel registered but no self-chat target configured",
        )
        return "no_recipient"

    await channel.send_text(str(recipient), body)
    return "sent"
