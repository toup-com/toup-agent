"""
Heartbeat Service — Proactive agent that initiates messages.

Periodically runs the agent with a special heartbeat prompt.  If the
agent has something useful to say (pending reminders, follow-ups,
proactive suggestions), it sends the message to the user's Telegram chat.

If the agent responds with ``__HEARTBEAT_SKIP__``, the result is silently
discarded — nothing is sent.

The heartbeat is registered as an APScheduler job in ``main.py``.
"""

import asyncio
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Sentinel the agent returns when it has nothing to say
SKIP_SENTINEL = "__HEARTBEAT_SKIP__"


class HeartbeatService:
    """
    Runs a proactive agent heartbeat on a schedule.

    Dependencies (set via setters after construction):
      - agent_runner: AgentRunner instance
      - telegram_bot: HexBrainTelegramBot instance
    """

    def __init__(self):
        self._agent_runner = None
        self._telegram_bot = None

    def set_agent_runner(self, agent_runner):
        self._agent_runner = agent_runner

    def set_bot(self, telegram_bot):
        self._telegram_bot = telegram_bot

    async def tick(self):
        """
        Run one heartbeat cycle for all mapped Telegram users.

        For each user with a known Telegram chat, runs the agent with the
        heartbeat prompt and sends the result if it's not a skip.
        """
        if not settings.heartbeat_enabled:
            return

        if not self._agent_runner or not self._telegram_bot:
            logger.warning("[HEARTBEAT] Agent runner or bot not available, skipping")
            return

        logger.info("[HEARTBEAT] Running proactive heartbeat cycle")

        # Gather all known user→chat mappings from the bot's cache
        user_chats = self._get_user_chat_pairs()
        if not user_chats:
            logger.info("[HEARTBEAT] No active user/chat pairs found")
            return

        for user_id, chat_id in user_chats:
            try:
                await self._run_for_user(user_id, chat_id)
            except Exception as e:
                logger.warning(f"[HEARTBEAT] Failed for user {user_id}: {e}")

        logger.info(f"[HEARTBEAT] Cycle complete for {len(user_chats)} user(s)")

    async def _run_for_user(self, user_id: str, chat_id: int):
        """Run the heartbeat agent for a single user/chat."""
        logger.debug(f"[HEARTBEAT] Running for user={user_id} chat={chat_id}")

        response = await self._agent_runner.run(
            user_message=f"[Heartbeat] {settings.heartbeat_prompt}",
            user_id=user_id,
            telegram_chat_id=chat_id,
        )

        text = (response.text or "").strip()

        # Agent says nothing notable — skip silently
        if not text or SKIP_SENTINEL in text:
            logger.debug(f"[HEARTBEAT] User {user_id}: skip (nothing to say)")
            return

        # Send the proactive message via Telegram
        await self._send_to_chat(chat_id, text)
        logger.info(
            f"[HEARTBEAT] Sent proactive message to chat {chat_id} "
            f"({len(text)} chars, {response.tokens_total} tokens)"
        )

    async def _send_to_chat(self, chat_id: int, text: str):
        """Send a proactive message to a Telegram chat."""
        from app.agent.streaming import postprocess_for_telegram, split_message

        bot = self._telegram_bot.app.bot
        processed = postprocess_for_telegram(text)
        chunks = split_message(processed)

        for chunk in chunks:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="HTML",
                )
            except Exception:
                # Fallback: plain text
                try:
                    await bot.send_message(chat_id=chat_id, text=chunk)
                except Exception as e:
                    logger.warning(f"[HEARTBEAT] Failed to send to chat {chat_id}: {e}")

    def _get_user_chat_pairs(self) -> list[tuple[str, int]]:
        """
        Get (user_id, chat_id) pairs from the bot's session map.

        Falls back to DB if the in-memory map is empty.
        """
        pairs: list[tuple[str, int]] = []

        # Reverse the bot's _user_map (telegram_id → user_id) and
        # _session_map (chat_id → session_id) to get usable pairs.
        if not self._telegram_bot:
            return pairs

        user_map = getattr(self._telegram_bot, "_user_map", {})
        session_map = getattr(self._telegram_bot, "_session_map", {})

        # chat_id is usually == telegram_user_id for DMs
        for chat_id in session_map:
            tg_user_id = chat_id  # In DMs, chat_id == telegram_user_id
            hex_user_id = user_map.get(tg_user_id)
            if hex_user_id:
                pairs.append((hex_user_id, chat_id))

        return pairs
