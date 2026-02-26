"""HexBrain Telegram Bot — Connects Telegram to the Agent Runtime.

Handles:
- Text messages → AgentRunner
- Photos → download + pass as image_url content blocks (GPT vision)
- Voice/audio → Whisper transcription → AgentRunner
- Documents → read content and include inline
- Commands: /start, /help, /status, /reset, /new, /stop, /whoami, /model, /compact, /usage, /cron, /export, /subagents
- Streaming responses with progressive message edits
- Continuous typing indicator during processing
- Reply-to context support
- ACK reactions (👀) and agent-driven reactions
- Inline button support with callback queries
- Multi-user support with TelegramUserMapping
- Group chat support (@mention detection)
- Error handler for uncaught exceptions
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from typing import Optional

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.agent.agent_runner import AgentRunner
from app.agent.streaming import TelegramStreamHandler, extract_reaction, extract_buttons
from app.agent.message_queue import MessageQueue
from app.config import settings

logger = logging.getLogger(__name__)


class HexBrainTelegramBot:
    """
    Telegram bot that connects to the HexBrain Agent Runtime.

    - Receives messages from Telegram
    - Routes them through AgentRunner (memory + tools + LLM)
    - Streams responses back to Telegram with cursor + throttle
    - Transcribes voice messages via Whisper
    - Passes photos to GPT vision
    """

    def __init__(self, token: str, agent_runner: AgentRunner):
        self.token = token
        self.agent_runner = agent_runner
        self.app: Optional[Application] = None
        self.cron_service = None  # Set after cron service starts
        self.subagent_manager = None  # Set after subagent manager created

        # Map Telegram user ID → HexBrain user ID (in-memory cache)
        self._user_map: dict[int, str] = {}
        # Map Telegram chat ID → session ID
        self._session_map: dict[int, str] = {}

        # Track bot startup time for /status uptime
        self._start_time: float = 0.0

        # Cancellation flags per chat_id for /stop
        self._cancel_flags: dict[int, asyncio.Event] = {}

        # Skill loader (set from main.py after skills are loaded)
        self.skill_loader = None

        # Message debounce queue for rapid messages
        self._message_queue = MessageQueue()

    async def start(self):
        """Initialize and start the bot in polling mode."""
        self._start_time = time.time()

        self.app = (
            Application.builder()
            .token(self.token)
            .build()
        )

        # Register command handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("reset", self._cmd_reset))
        self.app.add_handler(CommandHandler("new", self._cmd_new))
        self.app.add_handler(CommandHandler("stop", self._cmd_stop))
        self.app.add_handler(CommandHandler("whoami", self._cmd_whoami))
        self.app.add_handler(CommandHandler("model", self._cmd_model))
        self.app.add_handler(CommandHandler("compact", self._cmd_compact))
        self.app.add_handler(CommandHandler("usage", self._cmd_usage))
        self.app.add_handler(CommandHandler("cron", self._cmd_cron))
        self.app.add_handler(CommandHandler("export", self._cmd_export))
        self.app.add_handler(CommandHandler("subagents", self._cmd_subagents))
        self.app.add_handler(CommandHandler("skills", self._cmd_skills))
        self.app.add_handler(CommandHandler("pair", self._cmd_pair))
        self.app.add_handler(CommandHandler("think", self._cmd_think))
        self.app.add_handler(CommandHandler("verbose", self._cmd_verbose))
        self.app.add_handler(CommandHandler("activation", self._cmd_activation))
        self.app.add_handler(CommandHandler("config", self._cmd_config))
        self.app.add_handler(CommandHandler("allowlist", self._cmd_allowlist))
        self.app.add_handler(CommandHandler("auto", self._cmd_auto))
        self.app.add_handler(CommandHandler("lanes", self._cmd_lanes))
        self.app.add_handler(CommandHandler("tts", self._cmd_tts_mode))
        self.app.add_handler(CommandHandler("persona", self._cmd_persona))
        self.app.add_handler(CommandHandler("providers", self._cmd_providers))

        # Callback query handler for inline buttons
        self.app.add_handler(CallbackQueryHandler(self._handle_callback_query))

        # Register skill-defined commands (dynamic)
        if hasattr(self, 'skill_loader') and self.skill_loader:
            from app.agent.skills.loader import SkillLoader
            for cmd_def in self.skill_loader.get_all_commands():
                cmd_name = cmd_def.get("command", "")
                handler_fn = cmd_def.get("handler")
                if cmd_name and handler_fn:
                    self.app.add_handler(CommandHandler(cmd_name, handler_fn))
                    logger.info(f"[BOT] Registered skill command: /{cmd_name} (from {cmd_def.get('skill', '?')})")

        # Register message handlers — group + private
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )
        self.app.add_handler(
            MessageHandler(filters.PHOTO, self._handle_photo)
        )
        self.app.add_handler(
            MessageHandler(filters.Document.ALL, self._handle_document)
        )
        self.app.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice)
        )

        # Global error handler
        self.app.add_error_handler(self._error_handler)

        # Start
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

        # Register commands with BotFather so they appear in Telegram's menu
        await self.app.bot.set_my_commands([
            BotCommand("start", "Start the bot"),
            BotCommand("help", "Show available commands"),
            BotCommand("status", "Show bot status and usage"),
            BotCommand("reset", "Reset conversation (clear history)"),
            BotCommand("new", "Start a new session"),
            BotCommand("stop", "Stop current generation"),
            BotCommand("whoami", "Show your user info"),
            BotCommand("model", "Show or switch AI model"),
            BotCommand("compact", "Force context compaction"),
            BotCommand("usage", "Show token usage and cost"),
            BotCommand("cron", "List scheduled jobs"),
            BotCommand("export", "Export conversation history"),
            BotCommand("subagents", "List background tasks"),
            BotCommand("skills", "List loaded skill plugins"),
        ])

        logger.info("🤖 HexBrain Telegram bot started (polling mode)")

    async def stop(self):
        """Stop the bot gracefully."""
        if self.app:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
            except Exception as e:
                logger.warning(f"Bot shutdown error: {e}")
            logger.info("🤖 Telegram bot stopped")

    # ------------------------------------------------------------------
    # Error handler
    # ------------------------------------------------------------------
    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Global error handler for uncaught exceptions."""
        logger.exception(f"Unhandled exception: {context.error}", exc_info=context.error)
        if isinstance(update, Update) and update.effective_chat:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ An unexpected error occurred. Please try again.",
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------
    def _is_allowed(self, user_id: int) -> bool:
        """Check if a Telegram user is allowed to use the bot."""
        if not settings.telegram_allowed_user_ids:
            return True  # No restrictions configured
        return user_id in settings.telegram_allowed_user_ids

    async def _is_paired(self, telegram_user_id: int) -> bool:
        """
        Check if a Telegram user has been paired (when pairing is required).

        Returns True if:
        - ``telegram_require_pairing`` is False (pairing not enforced), or
        - The user has ``is_paired = True`` in the DB mapping.
        """
        if not settings.telegram_require_pairing:
            return True

        from app.db.database import async_session_maker
        from sqlalchemy import select
        from app.db.models import TelegramUserMapping

        async with async_session_maker() as db:
            result = await db.execute(
                select(TelegramUserMapping.is_paired).where(
                    TelegramUserMapping.telegram_id == telegram_user_id
                )
            )
            row = result.scalar_one_or_none()
            return bool(row)  # True only if is_paired == True

    async def _get_hexbrain_user_id(self, telegram_user_id: int, telegram_user=None) -> str:
        """
        Map a Telegram user ID to a HexBrain user ID.
        Uses TelegramUserMapping DB table. Auto-creates user if needed.
        Falls back to config mapping for hardcoded overrides.
        """
        # In-memory cache
        if telegram_user_id in self._user_map:
            return self._user_map[telegram_user_id]

        # Check if there's a configured mapping override (Telegram ID → HexBrain user ID)
        tg_id_str = str(telegram_user_id)
        if tg_id_str in settings.telegram_user_map:
            mapped_id = settings.telegram_user_map[tg_id_str]
            logger.info(f"[AGENT] Telegram user {telegram_user_id} mapped to HexBrain user {mapped_id} via config")
            self._user_map[telegram_user_id] = mapped_id

            # Config-mapped users are implicitly paired — ensure DB reflects that
            if settings.telegram_require_pairing:
                from app.db.database import async_session_maker as _sm
                from sqlalchemy import select as _sel
                from app.db.models import TelegramUserMapping as _TUM

                async with _sm() as _db:
                    _res = await _db.execute(
                        _sel(_TUM).where(_TUM.telegram_id == telegram_user_id)
                    )
                    _m = _res.scalar_one_or_none()
                    if _m and not _m.is_paired:
                        _m.is_paired = True
                        await _db.commit()

            return mapped_id

        from app.db.database import async_session_maker
        from sqlalchemy import select
        from app.db.models import TelegramUserMapping

        async with async_session_maker() as db:
            # Check DB mapping
            result = await db.execute(
                select(TelegramUserMapping).where(
                    TelegramUserMapping.telegram_id == telegram_user_id
                )
            )
            mapping = result.scalar_one_or_none()

            if mapping:
                # If platform owner is set and mapping points elsewhere, update it
                owner_id = settings.user_id
                if owner_id and mapping.user_id != owner_id:
                    logger.info(f"[AGENT] Updating stale mapping: {mapping.user_id} → {owner_id}")
                    mapping.user_id = owner_id

                # Update last_seen and name
                mapping.last_seen_at = datetime.utcnow()
                if telegram_user:
                    mapping.telegram_name = telegram_user.full_name
                    mapping.telegram_username = telegram_user.username
                await db.commit()
                self._user_map[telegram_user_id] = mapping.user_id
                logger.info(f"[AGENT] Telegram user {telegram_user_id} → HexBrain user {mapping.user_id} (from DB)")
                return mapping.user_id

            # No mapping exists — map to platform owner or create new user
            name = "Unknown"
            username = None
            if telegram_user:
                name = telegram_user.full_name or telegram_user.first_name or "Unknown"
                username = telegram_user.username

            # Personal agent: map Telegram user to the platform owner
            # so all sessions (web + Telegram) share the same user_id
            owner_id = settings.user_id
            if owner_id:
                from app.services.auth_service import get_user_by_id
                owner = await get_user_by_id(db, owner_id)
                if owner:
                    new_mapping = TelegramUserMapping(
                        telegram_id=telegram_user_id,
                        user_id=owner.id,
                        telegram_username=username,
                        telegram_name=name,
                    )
                    db.add(new_mapping)
                    await db.commit()
                    logger.info(f"[AGENT] Telegram user {telegram_user_id} → platform owner {owner.id}")
                    self._user_map[telegram_user_id] = owner.id
                    return owner.id

            # Fallback: create a new user (multi-tenant / no owner configured)
            from app.services.auth_service import get_user_by_email, create_user

            email = f"telegram_{telegram_user_id}@hexbrain.bot"

            user = await get_user_by_email(db, email)
            if not user:
                user = await create_user(
                    db,
                    email=email,
                    password=f"tg_{telegram_user_id}_bot_user",
                    name=name,
                )
                logger.info(f"Created HexBrain user for Telegram user {telegram_user_id}: {user.id}")

            # Create mapping
            new_mapping = TelegramUserMapping(
                telegram_id=telegram_user_id,
                user_id=user.id,
                telegram_username=username,
                telegram_name=name,
            )
            db.add(new_mapping)
            await db.commit()

            logger.info(f"[AGENT] Telegram user {telegram_user_id} → HexBrain user {user.id} (new mapping)")
            self._user_map[telegram_user_id] = user.id
            return user.id

    # ------------------------------------------------------------------
    # Cancellation helpers
    # ------------------------------------------------------------------
    def get_cancel_event(self, chat_id: int) -> asyncio.Event:
        """Get or create a cancellation event for a chat."""
        if chat_id not in self._cancel_flags:
            self._cancel_flags[chat_id] = asyncio.Event()
        return self._cancel_flags[chat_id]

    def clear_cancel(self, chat_id: int):
        """Clear the cancellation flag for a chat."""
        if chat_id in self._cancel_flags:
            self._cancel_flags[chat_id].clear()

    def is_cancelled(self, chat_id: int) -> bool:
        """Check if generation is cancelled for a chat."""
        if chat_id in self._cancel_flags:
            return self._cancel_flags[chat_id].is_set()
        return False

    # ------------------------------------------------------------------
    # Uptime helper
    # ------------------------------------------------------------------
    def _format_uptime(self) -> str:
        """Format bot uptime as a human-readable string."""
        elapsed = int(time.time() - self._start_time)
        days, remainder = divmod(elapsed, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Session resolution (cache + DB fallback)
    # ------------------------------------------------------------------
    async def _get_session_id(self, chat_id: int, user_id: str) -> Optional[str]:
        """
        Get the active session ID for a chat.
        Checks in-memory cache first, then falls back to DB query.
        Populates the cache on DB hit so subsequent calls are fast.
        """
        # 1. In-memory cache
        session_id = self._session_map.get(chat_id)
        if session_id:
            return session_id

        # 2. DB fallback — find the most recent active telegram session
        try:
            from app.db.database import async_session_maker
            from app.db.models import Conversation
            from sqlalchemy import select, and_

            async with async_session_maker() as db:
                result = await db.execute(
                    select(Conversation).where(
                        and_(
                            Conversation.user_id == user_id,
                            Conversation.channel == "telegram",
                            Conversation.is_active == True,
                        )
                    ).order_by(Conversation.updated_at.desc()).limit(1)
                )
                conv = result.scalar_one_or_none()
                if conv:
                    self._session_map[chat_id] = conv.id  # populate cache
                    return conv.id
        except Exception as e:
            logger.warning(f"Session DB lookup failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start — welcome message + ensure session."""
        if not self._is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return

        # Ensure user mapping exists (creates session on first message)
        await self._get_hexbrain_user_id(update.effective_user.id, telegram_user=update.effective_user)

        await update.message.reply_text(
            "👋 Hey! I'm <b>Hex</b>, your AI assistant.\n\n"
            "Send me a message, photo, voice note, or file and I'll help.\n\n"
            "Type /help to see all commands.",
            parse_mode="HTML",
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help — show all available commands."""
        if not self._is_allowed(update.effective_user.id):
            return

        await update.message.reply_text(
            "🤖 <b>Available Commands</b>\n\n"
            "/start — Start the bot\n"
            "/help — Show this help message\n"
            "/status — Show bot status and token usage\n"
            "/reset — Reset conversation (clear history)\n"
            "/new — Start a new session\n"
            "/stop — Stop current generation\n"
            "/whoami — Show your user info\n"
            "/model — Show or switch AI model\n"
            "/compact — Force context compaction\n"
            "/usage — Show token usage and cost\n"
            "/cron — List scheduled jobs\n"
            "/export — Export conversation history\n"
            "/subagents — List background tasks\n"
            "/skills — List loaded skill plugins\n"
            "/pair &lt;code&gt; — Pair your account (when required)\n"
            "/think — Extended thinking budget\n"
            "/verbose — Toggle tool narration\n"
            "/activation — Set/show activation prompt\n"
            "/config — View/set runtime config\n"
            "/allowlist — Manage allowed users\n"
            "/auto — Toggle auto-response mode\n"
            "/lanes — View agent execution lanes\n"
            "/tts — Set TTS auto-mode (off/always/inbound/tagged)",
            parse_mode="HTML",
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status — show bot status, model, uptime, session stats, memory count."""
        if not self._is_allowed(update.effective_user.id):
            return

        user_id = await self._get_hexbrain_user_id(update.effective_user.id)

        msg_count = 0
        tokens_used = 0
        mem_count = 0

        try:
            from app.db.database import async_session_maker
            from sqlalchemy import select, func, and_
            from app.db.models import Conversation, Message, Memory

            async with async_session_maker() as db:
                # Find active session for this user on telegram channel
                result = await db.execute(
                    select(Conversation).where(
                        and_(
                            Conversation.user_id == user_id,
                            Conversation.channel == "telegram",
                            Conversation.is_active == True,
                        )
                    ).order_by(Conversation.updated_at.desc()).limit(1)
                )
                conv = result.scalar_one_or_none()

                if conv:
                    # Count messages in this session
                    result = await db.execute(
                        select(func.count(Message.id)).where(
                            Message.conversation_id == conv.id
                        )
                    )
                    msg_count = result.scalar() or 0

                    # Sum tokens used in this session
                    result = await db.execute(
                        select(
                            func.coalesce(func.sum(func.coalesce(Message.tokens_prompt, 0) + func.coalesce(Message.tokens_completion, 0)), 0)
                        ).where(
                            Message.conversation_id == conv.id
                        )
                    )
                    tokens_used = result.scalar() or 0

                # Count user's active memories
                result = await db.execute(
                    select(func.count(Memory.id)).where(
                        and_(
                            Memory.user_id == user_id,
                            Memory.is_active == True,
                            Memory.is_deleted == False,
                        )
                    )
                )
                mem_count = result.scalar() or 0
        except Exception as e:
            logger.warning(f"Status query failed: {e}")

        uptime = self._format_uptime()

        await update.message.reply_text(
            f"📊 <b>Status</b>\n\n"
            f"Model: <code>{settings.agent_model}</code>\n"
            f"Session: {msg_count} messages\n"
            f"Tokens used: {tokens_used:,}\n"
            f"Memories: {mem_count} stored\n"
            f"Uptime: {uptime}",
            parse_mode="HTML",
        )

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset — end current session, start fresh. Memories preserved."""
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = update.effective_chat.id
        user_id = await self._get_hexbrain_user_id(update.effective_user.id)

        # Mark old session as inactive in DB
        old_session_id = await self._get_session_id(chat_id, user_id)
        if old_session_id:
            try:
                from app.db.database import async_session_maker
                from sqlalchemy import select
                from app.db.models import Conversation
                from datetime import datetime
                async with async_session_maker() as db:
                    result = await db.execute(
                        select(Conversation).where(Conversation.id == old_session_id)
                    )
                    conv = result.scalar_one_or_none()
                    if conv:
                        conv.is_active = False
                        conv.ended_at = datetime.utcnow()
                        await db.commit()
            except Exception as e:
                logger.warning(f"Failed to close old session: {e}")

            del self._session_map[chat_id]

        await update.message.reply_text(
            "🔄 Conversation reset. Starting fresh!\n"
            "<i>(Your memories are still intact)</i>",
            parse_mode="HTML",
        )

    async def _cmd_new(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new — same as /reset but different reply text."""
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = update.effective_chat.id
        user_id = await self._get_hexbrain_user_id(update.effective_user.id)

        # Mark old session as inactive
        old_session_id = await self._get_session_id(chat_id, user_id)
        if old_session_id:
            try:
                from app.db.database import async_session_maker
                from sqlalchemy import select
                from app.db.models import Conversation
                from datetime import datetime
                async with async_session_maker() as db:
                    result = await db.execute(
                        select(Conversation).where(Conversation.id == old_session_id)
                    )
                    conv = result.scalar_one_or_none()
                    if conv:
                        conv.is_active = False
                        conv.ended_at = datetime.utcnow()
                        await db.commit()
            except Exception as e:
                logger.warning(f"Failed to close old session: {e}")

            del self._session_map[chat_id]

        await update.message.reply_text("✨ New session started.")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop — cancel in-progress generation."""
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = update.effective_chat.id

        # Cancel any pending debounce queue
        self._message_queue.cancel(chat_id)

        event = self._cancel_flags.get(chat_id)

        if event and not event.is_set():
            # There's an active event that hasn't been cancelled yet
            # Check if there's actually a running generation
            # (the event exists but isn't set = generation is running)
            event.set()
            await update.message.reply_text("⏹ Stopped.")
        else:
            await update.message.reply_text("Nothing to stop.")

    async def _cmd_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pair <code> — pair this Telegram account when pairing is required."""
        if not self._is_allowed(update.effective_user.id):
            return

        if not settings.telegram_require_pairing:
            await update.message.reply_text("ℹ️ Pairing is not required for this bot.")
            return

        # Check if already paired
        if await self._is_paired(update.effective_user.id):
            await update.message.reply_text("✅ You're already paired!")
            return

        # Validate code
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /pair <code>")
            return

        code = args[0].strip()
        if code != settings.telegram_pairing_code:
            logger.warning("Bad pairing attempt from tg_id=%s", update.effective_user.id)
            await update.message.reply_text("❌ Invalid pairing code.")
            return

        # Mark user as paired in DB
        from app.db.database import async_session_maker
        from app.db.models import TelegramUserMapping

        tg_user = update.effective_user

        async with async_session_maker() as db:
            from sqlalchemy import select

            result = await db.execute(
                select(TelegramUserMapping).where(
                    TelegramUserMapping.telegram_id == tg_user.id
                )
            )
            mapping = result.scalar_one_or_none()

            if mapping:
                mapping.is_paired = True
            else:
                # Create mapping + mark paired — _get_hexbrain_user_id will
                # create the full user later on first message
                user_id = await self._get_hexbrain_user_id(tg_user.id)
                result2 = await db.execute(
                    select(TelegramUserMapping).where(
                        TelegramUserMapping.telegram_id == tg_user.id
                    )
                )
                mapping = result2.scalar_one_or_none()
                if mapping:
                    mapping.is_paired = True

            await db.commit()

        logger.info("Paired tg_id=%s (%s)", tg_user.id, tg_user.username)
        await update.message.reply_text("✅ Paired successfully! You can now use the bot.")

    async def _cmd_whoami(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /whoami — show user's Telegram + HexBrain info."""
        if not self._is_allowed(update.effective_user.id):
            return

        tg_user = update.effective_user
        user_id = await self._get_hexbrain_user_id(tg_user.id)

        user_email = "unknown"
        msg_count = 0
        mem_count = 0
        has_active_session = False

        try:
            from app.db.database import async_session_maker
            from sqlalchemy import select, func, and_
            from app.db.models import User, Conversation, Message, Memory

            async with async_session_maker() as db:
                # Get user email
                result = await db.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                if user:
                    user_email = user.email

                # Find active session
                result = await db.execute(
                    select(Conversation).where(
                        and_(
                            Conversation.user_id == user_id,
                            Conversation.channel == "telegram",
                            Conversation.is_active == True,
                        )
                    ).order_by(Conversation.updated_at.desc()).limit(1)
                )
                conv = result.scalar_one_or_none()

                if conv:
                    has_active_session = True
                    # Count messages in session
                    result = await db.execute(
                        select(func.count(Message.id)).where(
                            Message.conversation_id == conv.id
                        )
                    )
                    msg_count = result.scalar() or 0

                # Count user's active memories
                result = await db.execute(
                    select(func.count(Memory.id)).where(
                        and_(
                            Memory.user_id == user_id,
                            Memory.is_active == True,
                            Memory.is_deleted == False,
                        )
                    )
                )
                mem_count = result.scalar() or 0
        except Exception as e:
            logger.warning(f"Whoami query failed: {e}")

        name = tg_user.full_name or tg_user.first_name or "Unknown"
        session_status = f"active ({msg_count} messages)" if has_active_session else "no active session"

        await update.message.reply_text(
            f"👤 <b>Your Info</b>\n\n"
            f"Name: {name}\n"
            f"Telegram ID: <code>{tg_user.id}</code>\n"
            f"HexBrain User: <code>{user_email}</code>\n"
            f"Session: {session_status}\n"
            f"Memories: {mem_count} stored about you",
            parse_mode="HTML",
        )

    async def _cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /model — show or switch the AI model."""
        if not self._is_allowed(update.effective_user.id):
            return

        args = context.args
        if not args:
            if settings.agent_model == "auto":
                mode_info = "⚡ <b>Auto</b> (routes by complexity)\n  • Light → Claude Sonnet 4.5\n  • Medium → GPT-5.2\n  • Heavy → Claude Opus 4.6"
            else:
                mode_info = f"<code>{settings.agent_model}</code>"
            await update.message.reply_text(
                f"🤖 Current model: {mode_info}\n\n"
                f"Switch: <code>/model auto</code> or <code>/model gpt-5.2</code>",
                parse_mode="HTML",
            )
            return

        new_model = args[0].strip()
        allowed_models = [
            "auto",
            "gpt-5.2", "gpt-5", "gpt-4.1", "gpt-4o", "gpt-4o-mini",
            "claude-opus-4-6", "claude-sonnet-4-5", "claude-sonnet-4-5-20250514", "claude-sonnet-4-20250514",
        ]
        if new_model not in allowed_models:
            await update.message.reply_text(
                f"❌ Unknown model: <code>{new_model}</code>\n\n"
                f"Available: {', '.join(f'<code>{m}</code>' for m in allowed_models)}",
                parse_mode="HTML",
            )
            return

        old_model = settings.agent_model
        settings.agent_model = new_model
        # Only update the LLM default_model for real model IDs, not "auto"
        if new_model != "auto":
            self.agent_runner.llm.default_model = new_model

        if new_model == "auto":
            label = "⚡ Auto (Sonnet 4.5 / GPT-5.2 / Opus 4.6)"
        else:
            label = f"<code>{new_model}</code>"
        await update.message.reply_text(
            f"✅ Model switched: <code>{old_model}</code> → {label}",
            parse_mode="HTML",
        )

    async def _cmd_think(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /think — configure extended thinking budget.

        Usage:
            /think         → Show current thinking budget
            /think off     → Disable extended thinking
            /think 4096    → Set thinking budget to 4096 tokens
            /think max     → Set thinking budget to 32768 tokens
        """
        if not self._is_allowed(update.effective_user.id):
            return

        args = context.args
        chat_id = str(update.effective_chat.id)

        # Per-chat thinking budget stored in memory
        if not hasattr(self, '_thinking_budgets'):
            self._thinking_budgets = {}

        if not args:
            current = self._thinking_budgets.get(chat_id, settings.thinking_budget_default)
            status = "🔴 Disabled" if current == 0 else f"�� {current} tokens"
            await update.message.reply_text(
                f"🧠 <b>Extended Thinking</b>\n\n"
                f"Status: {status}\n\n"
                f"<code>/think 4096</code> — Set budget\n"
                f"<code>/think max</code> — Maximum (32768)\n"
                f"<code>/think off</code> — Disable",
                parse_mode="HTML",
            )
            return

        val = args[0].strip().lower()
        if val in ("off", "0", "disable", "none"):
            self._thinking_budgets[chat_id] = 0
            await update.message.reply_text("🧠 Extended thinking: 🔴 <b>Disabled</b>", parse_mode="HTML")
        elif val == "max":
            self._thinking_budgets[chat_id] = 32768
            await update.message.reply_text("🧠 Extended thinking: 🟢 <b>32768 tokens</b> (maximum)", parse_mode="HTML")
        else:
            try:
                budget = int(val)
                budget = max(1024, min(budget, 32768))  # Clamp 1024–32768
                self._thinking_budgets[chat_id] = budget
                await update.message.reply_text(
                    f"🧠 Extended thinking: 🟢 <b>{budget} tokens</b>",
                    parse_mode="HTML",
                )
            except ValueError:
                await update.message.reply_text(
                    f"❌ Invalid value: <code>{val}</code>\nUse a number (1024–32768), <code>max</code>, or <code>off</code>",
                    parse_mode="HTML",
                )

    async def _cmd_verbose(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /verbose — toggle tool narration verbosity."""
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = str(update.effective_chat.id)
        if not hasattr(self, '_verbose_chats'):
            self._verbose_chats = set()

        if chat_id in self._verbose_chats:
            self._verbose_chats.discard(chat_id)
            await update.message.reply_text(
                "🔇 <b>Verbose mode:</b> OFF\n"
                "Tool calls will be narrated briefly.",
                parse_mode="HTML",
            )
        else:
            self._verbose_chats.add(chat_id)
            await update.message.reply_text(
                "🔊 <b>Verbose mode:</b> ON\n"
                "Full tool inputs and outputs will be shown.",
                parse_mode="HTML",
            )

    async def _cmd_activation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /activation — set or show activation (boot) prompt.

        Usage:
            /activation           → Show current activation prompt
            /activation clear     → Remove activation prompt
            /activation <text>    → Set new activation prompt
        """
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = str(update.effective_chat.id)
        if not hasattr(self, '_activation_prompts'):
            self._activation_prompts = {}

        args = context.args
        if not args:
            current = self._activation_prompts.get(chat_id, "")
            if current:
                await update.message.reply_text(
                    f"🚀 <b>Activation prompt:</b>\n\n<code>{current[:500]}</code>",
                    parse_mode="HTML",
                )
            else:
                await update.message.reply_text(
                    "🚀 No activation prompt set.\n\n"
                    "<code>/activation Hello! Please start by checking my calendar.</code>",
                    parse_mode="HTML",
                )
            return

        text = " ".join(args)
        if text.lower() in ("clear", "none", "off", "reset"):
            self._activation_prompts.pop(chat_id, None)
            await update.message.reply_text("🚀 Activation prompt cleared.", parse_mode="HTML")
        else:
            self._activation_prompts[chat_id] = text
            await update.message.reply_text(
                f"🚀 Activation prompt set:\n<code>{text[:300]}</code>",
                parse_mode="HTML",
            )

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config — view or modify runtime config values.

        Usage:
            /config                    → Show key settings
            /config set key value      → Update a setting
            /config unset key          → Reset to default
        """
        if not self._is_allowed(update.effective_user.id):
            return

        args = context.args or []

        if not args:
            from app.config import settings
            lines = [
                "⚙️ <b>Runtime Config</b>\n",
                f"agent_model: <code>{settings.agent_model}</code>",
                f"agent_max_tokens: <code>{settings.agent_max_tokens}</code>",
                f"agent_max_tool_iterations: <code>{settings.agent_max_tool_iterations}</code>",
                f"temperature: <code>{settings.temperature}</code>",
                f"auto_extract_memories: <code>{settings.auto_extract_memories}</code>",
                f"max_history_messages: <code>{settings.max_history_messages}</code>",
                f"memory_recall_limit: <code>{settings.memory_recall_limit}</code>",
                f"sandbox_enabled: <code>{settings.sandbox_enabled}</code>",
                f"heartbeat_enabled: <code>{settings.heartbeat_enabled}</code>",
                f"enable_reranker: <code>{settings.enable_reranker}</code>",
                f"thinking_budget: <code>{settings.thinking_budget_default}</code>",
                f"tool_deny_list: <code>{settings.tool_deny_list}</code>",
                f"tool_elevated_list: <code>{settings.tool_elevated_list}</code>",
            ]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            return

        action = args[0].lower()

        if action == "set" and len(args) >= 3:
            key = args[1]
            value = " ".join(args[2:])
            from app.config import settings

            ALLOWED_KEYS = {
                "temperature", "agent_max_tokens", "agent_max_tool_iterations",
                "max_history_messages", "memory_recall_limit", "auto_extract_memories",
                "heartbeat_enabled", "sandbox_enabled", "enable_reranker",
                "thinking_budget_default", "tool_max_output_chars",
            }
            if key not in ALLOWED_KEYS:
                await update.message.reply_text(
                    f"❌ Key <code>{key}</code> not allowed.\n\nAllowed: {', '.join(sorted(ALLOWED_KEYS))}",
                    parse_mode="HTML",
                )
                return

            # Type coercion
            old_val = getattr(settings, key, None)
            try:
                if isinstance(old_val, bool):
                    new_val = value.lower() in ("true", "1", "yes", "on")
                elif isinstance(old_val, int):
                    new_val = int(value)
                elif isinstance(old_val, float):
                    new_val = float(value)
                else:
                    new_val = value
                setattr(settings, key, new_val)
                await update.message.reply_text(
                    f"✅ <code>{key}</code>: <code>{old_val}</code> → <code>{new_val}</code>",
                    parse_mode="HTML",
                )
            except (ValueError, TypeError) as e:
                await update.message.reply_text(f"❌ Invalid value: {e}", parse_mode="HTML")

        elif action == "unset" and len(args) >= 2:
            key = args[1]
            from app.config import Settings
            default_settings = Settings()
            default_val = getattr(default_settings, key, None)
            if default_val is not None:
                from app.config import settings
                setattr(settings, key, default_val)
                await update.message.reply_text(
                    f"✅ <code>{key}</code> reset to default: <code>{default_val}</code>",
                    parse_mode="HTML",
                )
            else:
                await update.message.reply_text(f"❌ Unknown key: <code>{key}</code>", parse_mode="HTML")
        else:
            await update.message.reply_text(
                "Usage:\n"
                "<code>/config</code> — show settings\n"
                "<code>/config set key value</code>\n"
                "<code>/config unset key</code>",
                parse_mode="HTML",
            )

    async def _cmd_allowlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /allowlist — manage allowed Telegram user IDs.

        Usage:
            /allowlist              → Show current allowlist
            /allowlist add <id>     → Add a user ID
            /allowlist remove <id>  → Remove a user ID
        """
        if not self._is_allowed(update.effective_user.id):
            return

        args = context.args or []
        from app.config import settings

        if not args:
            ids = settings.telegram_allowed_user_ids
            if ids:
                lines = "\n".join(f"  • <code>{uid}</code>" for uid in ids)
                await update.message.reply_text(
                    f"🔒 <b>Allowed Users</b> ({len(ids)}):\n{lines}",
                    parse_mode="HTML",
                )
            else:
                await update.message.reply_text(
                    "🔓 <b>Allowlist is empty</b> — all users can access the bot.",
                    parse_mode="HTML",
                )
            return

        action = args[0].lower()

        if action == "add" and len(args) >= 2:
            try:
                uid = int(args[1])
                if uid not in settings.telegram_allowed_user_ids:
                    settings.telegram_allowed_user_ids.append(uid)
                await update.message.reply_text(
                    f"✅ Added <code>{uid}</code> to allowlist.",
                    parse_mode="HTML",
                )
            except ValueError:
                await update.message.reply_text("❌ User ID must be a number.", parse_mode="HTML")

        elif action == "remove" and len(args) >= 2:
            try:
                uid = int(args[1])
                if uid in settings.telegram_allowed_user_ids:
                    settings.telegram_allowed_user_ids.remove(uid)
                    await update.message.reply_text(
                        f"✅ Removed <code>{uid}</code> from allowlist.",
                        parse_mode="HTML",
                    )
                else:
                    await update.message.reply_text(f"⚠️ <code>{uid}</code> not in allowlist.", parse_mode="HTML")
            except ValueError:
                await update.message.reply_text("❌ User ID must be a number.", parse_mode="HTML")
        else:
            await update.message.reply_text(
                "Usage:\n"
                "<code>/allowlist</code> — show list\n"
                "<code>/allowlist add 123456</code>\n"
                "<code>/allowlist remove 123456</code>",
                parse_mode="HTML",
            )


    async def _cmd_auto(self, update, context):
        """Toggle auto-mode for the current chat."""
        chat_id = update.effective_chat.id
        if not hasattr(self, '_auto_mode_chats'):
            self._auto_mode_chats = set()

        if chat_id in self._auto_mode_chats:
            self._auto_mode_chats.discard(chat_id)
            await update.message.reply_text("🔴 Auto-mode disabled.")
        else:
            self._auto_mode_chats.add(chat_id)
            await update.message.reply_text(
                "🟢 Auto-mode enabled.\n"
                "I will proactively respond to all messages in this chat, "
                "even without being mentioned."
            )

    async def _cmd_lanes(self, update, context):
        """Show agent execution lane statistics."""
        from app.agent.lanes import get_lane_manager
        lm = get_lane_manager()
        stats = lm.get_stats()

        active = stats.get("active", 0)
        total = stats.get("total_runs", 0)
        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)
        by_lane = stats.get("by_lane", {})

        lines = [
            "📊 <b>Agent Lane Status</b>",
            f"Active: {active} / Max: {stats.get('max_concurrent', '?')}",
            f"Total runs: {total} (✅ {completed}, ❌ {failed})",
            "",
            "<b>By Lane:</b>",
        ]
        for lane, count in by_lane.items():
            emoji = {"main": "💬", "subagent": "🤖", "cron": "⏰", "hook": "🔗"}.get(lane, "📦")
            lines.append(f"  {emoji} {lane}: {count} active")

        await update.message.reply_text(
            "\n".join(lines), parse_mode="HTML",
        )

    async def _cmd_tts_mode(self, update, context):
        """Toggle TTS auto-mode (off/always/inbound/tagged)."""
        from app.config import settings
        args = context.args if context.args else []

        if not args:
            current = settings.tts_auto_mode
            await update.message.reply_text(
                f"🔊 TTS auto-mode: <b>{current}</b>\n\n"
                "Usage: /tts off|always|inbound|tagged\n"
                "• off — No automatic TTS\n"
                "• always — All responses as voice\n"
                "• inbound — Reply with voice when user sends voice\n"
                "• tagged — Only when agent uses [[tts:...]] tag",
                parse_mode="HTML",
            )
            return

        mode = args[0].lower()
        if mode not in ("off", "always", "inbound", "tagged"):
            await update.message.reply_text("❌ Invalid mode. Use: off, always, inbound, tagged")
            return

        from app.agent.config_reload import reload_config
        reload_config({"tts_auto_mode": mode})
        await update.message.reply_text(f"🔊 TTS auto-mode set to: <b>{mode}</b>", parse_mode="HTML")


    async def _cmd_persona(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch multi-agent persona for this chat."""
        from app.agent.multi_agent import get_multi_agent_router

        if not settings.multi_agent_enabled:
            await update.message.reply_text("Multi-agent routing is disabled. Set MULTI_AGENT_ENABLED=true.")
            return

        router = get_multi_agent_router()
        args = context.args

        if not args:
            personas = router.list_personas()
            lines = ["🎭 **Available Personas:**\n"]
            for p in personas:
                marker = "→ " if p["name"] == router._default else "  "
                lines.append(f"{marker}**{p['name']}** — {p['description']}")
                if p.get("model"):
                    lines[-1] += f" (model: {p['model']})"
            lines.append(f"\nDefault: {router._default}")
            lines.append("Usage: /persona <name>")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            return

        name = args[0].lower()
        persona = router.get(name)
        if not persona:
            await update.message.reply_text(f"Unknown persona: {name}")
            return

        router.set_default(name)
        await update.message.reply_text(f"🎭 Switched to persona: **{name}** — {persona.description}", parse_mode="Markdown")

    async def _cmd_providers(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show configured model providers and TTS providers."""
        lines = ["🔌 **Providers:**\n"]

        # Model providers
        lines.append("**Models:**")
        lines.append(f"  Agent: {settings.agent_model}")
        lines.append(f"  Fallback: {settings.agent_fallback_model}")
        lines.append(f"  Chat: {settings.default_model}")
        if settings.custom_model_providers:
            lines.append(f"  Custom: {list(settings.custom_model_providers.keys())}")

        # TTS providers
        lines.append("\n**TTS:**")
        lines.append(f"  Provider: {settings.tts_provider}")
        lines.append(f"  Voice: {settings.tts_default_voice}")
        lines.append(f"  Model: {settings.tts_model}")
        lines.append(f"  ElevenLabs: {'configured' if settings.elevenlabs_api_key else 'not set'}")
        lines.append(f"  Edge TTS: available (free)")

        # Thinking
        lines.append("\n**Thinking:**")
        lines.append(f"  Default budget: {settings.thinking_budget_default}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


    async def _cmd_compact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compact — force context compaction on the current session."""
        if not self._is_allowed(update.effective_user.id):
            return

        user_id = await self._get_hexbrain_user_id(update.effective_user.id)
        chat_id = update.effective_chat.id
        session_id = await self._get_session_id(chat_id, user_id)

        if not session_id:
            await update.message.reply_text("No active session to compact.")
            return

        try:
            from app.db.database import async_session_maker
            from sqlalchemy import select, func
            from app.db.models import Message
            from app.agent.context_manager import compact_messages

            async with async_session_maker() as db:
                result = await db.execute(
                    select(Message)
                    .where(Message.conversation_id == session_id)
                    .order_by(Message.created_at.desc())
                    .limit(100)
                )
                rows = list(reversed(result.scalars().all()))

                if len(rows) < 5:
                    await update.message.reply_text("📦 Not enough messages to compact.")
                    return

                messages = [{"role": m.role, "content": m.content} for m in rows if m.role in ("user", "assistant")]
                old_count = len(messages)
                compacted = await compact_messages(messages, settings.agent_model)

                await update.message.reply_text(
                    f"🗜 <b>Compacted:</b> {old_count} messages → {len(compacted)} (summary + recent)\n"
                    f"<i>Context freed. Old messages summarized.</i>",
                    parse_mode="HTML",
                )

        except Exception as e:
            logger.warning(f"Compact failed: {e}")
            await update.message.reply_text(f"❌ Compaction failed: {str(e)[:200]}")

    async def _cmd_usage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /usage — show token usage and estimated cost."""
        if not self._is_allowed(update.effective_user.id):
            return

        user_id = await self._get_hexbrain_user_id(update.effective_user.id)

        session_tokens_in = 0
        session_tokens_out = 0
        today_tokens_in = 0
        today_tokens_out = 0
        all_tokens_in = 0
        all_tokens_out = 0

        try:
            from app.db.database import async_session_maker
            from sqlalchemy import select, func, and_
            from app.db.models import Conversation, Message
            from datetime import datetime, timedelta

            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            async with async_session_maker() as db:
                # Current session tokens
                chat_id = update.effective_chat.id
                session_id = await self._get_session_id(chat_id, user_id)
                if session_id:
                    result = await db.execute(
                        select(
                            func.coalesce(func.sum(func.coalesce(Message.tokens_prompt, 0)), 0),
                            func.coalesce(func.sum(func.coalesce(Message.tokens_completion, 0)), 0),
                        ).where(Message.conversation_id == session_id)
                    )
                    row = result.one()
                    session_tokens_in = row[0]
                    session_tokens_out = row[1]

                # All user conversations
                result = await db.execute(
                    select(Conversation.id).where(Conversation.user_id == user_id)
                )
                conv_ids = [r[0] for r in result.all()]

                if conv_ids:
                    # Today
                    result = await db.execute(
                        select(
                            func.coalesce(func.sum(func.coalesce(Message.tokens_prompt, 0)), 0),
                            func.coalesce(func.sum(func.coalesce(Message.tokens_completion, 0)), 0),
                        ).where(
                            and_(
                                Message.conversation_id.in_(conv_ids),
                                Message.created_at >= today_start,
                            )
                        )
                    )
                    row = result.one()
                    today_tokens_in = row[0]
                    today_tokens_out = row[1]

                    # All time
                    result = await db.execute(
                        select(
                            func.coalesce(func.sum(func.coalesce(Message.tokens_prompt, 0)), 0),
                            func.coalesce(func.sum(func.coalesce(Message.tokens_completion, 0)), 0),
                        ).where(Message.conversation_id.in_(conv_ids))
                    )
                    row = result.one()
                    all_tokens_in = row[0]
                    all_tokens_out = row[1]

        except Exception as e:
            logger.warning(f"Usage query failed: {e}")

        model = settings.agent_model
        pricing = settings.pricing_per_1k.get(model, {"input": 0.003, "output": 0.012})
        rate_in = pricing["input"]
        rate_out = pricing["output"]

        def calc_cost(tin, tout):
            return (tin / 1000.0 * rate_in) + (tout / 1000.0 * rate_out)

        session_cost = calc_cost(session_tokens_in, session_tokens_out)
        today_cost = calc_cost(today_tokens_in, today_tokens_out)
        all_cost = calc_cost(all_tokens_in, all_tokens_out)

        session_total = session_tokens_in + session_tokens_out
        today_total = today_tokens_in + today_tokens_out
        all_total = all_tokens_in + all_tokens_out

        await update.message.reply_text(
            f"📊 <b>Usage Summary</b>\n\n"
            f"Session: {session_total:,} tokens (${session_cost:.2f})\n"
            f"Today: {today_total:,} tokens (${today_cost:.2f})\n"
            f"All time: {all_total:,} tokens (${all_cost:.2f})\n\n"
            f"Model: <code>{model}</code>\n"
            f"Rate: ${rate_in}/1K in, ${rate_out}/1K out",
            parse_mode="HTML",
        )

    async def _cmd_cron(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /cron — list scheduled jobs."""
        if not self._is_allowed(update.effective_user.id):
            return

        user_id = await self._get_hexbrain_user_id(update.effective_user.id)

        if not self.cron_service:
            await update.message.reply_text("⏰ Cron service not available.")
            return

        jobs = await self.cron_service.list_jobs(user_id)
        if not jobs:
            await update.message.reply_text(
                "⏰ No scheduled jobs.\n\n"
                "<i>Ask me to set a reminder or schedule a task.</i>",
                parse_mode="HTML",
            )
            return

        lines = ["⏰ <b>Scheduled Jobs</b>\n"]
        for i, j in enumerate(jobs, 1):
            status = "✅ active" if j["enabled"] else "⏸ disabled"
            lines.append(
                f"{i}. <b>{j['name']}</b>\n"
                f"   Schedule: <code>{j['schedule']}</code> ({j['kind']})\n"
                f"   Status: {status} | Runs: {j['run_count']}\n"
                f"   ID: <code>{j['id'][:8]}</code>"
            )
        lines.append("\n<i>Use the cron tool or ask me to manage jobs.</i>")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _cmd_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /export — export conversation history as a text file."""
        if not self._is_allowed(update.effective_user.id):
            return

        user_id = await self._get_hexbrain_user_id(update.effective_user.id)
        chat_id = update.effective_chat.id
        session_id = await self._get_session_id(chat_id, user_id)

        if not session_id:
            await update.message.reply_text("📭 No active session to export. Start a conversation first.")
            return

        try:
            from app.db.database import async_session_maker
            from app.db.models import Conversation, Message
            from sqlalchemy import select

            async with async_session_maker() as db:
                result = await db.execute(
                    select(Message)
                    .where(Message.conversation_id == session_id)
                    .order_by(Message.created_at.asc())
                )
                messages = result.scalars().all()

            if not messages:
                await update.message.reply_text("📭 No messages in current session.")
                return

            # Build export text
            lines = [
                f"=== HexBrain Conversation Export ===",
                f"Session: {session_id}",
                f"Messages: {len(messages)}",
                f"Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                f"{'=' * 40}\n",
            ]
            for msg in messages:
                role = "You" if msg.role == "user" else "Hex"
                ts = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
                content = msg.content or ""
                lines.append(f"[{ts}] {role}:\n{content}\n")

            export_text = "\n".join(lines)

            # Send as file
            import io
            file_bytes = io.BytesIO(export_text.encode("utf-8"))
            file_bytes.name = f"hexbrain_export_{session_id[:8]}.txt"
            await update.message.reply_document(
                document=file_bytes,
                caption=f"📄 Exported {len(messages)} messages from session <code>{session_id[:8]}</code>",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.exception("Export failed")
            await update.message.reply_text(f"❌ Export failed: {str(e)[:200]}")

    async def _cmd_subagents(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /subagents — list background tasks."""
        if not self._is_allowed(update.effective_user.id):
            return

        if not self.subagent_manager:
            await update.message.reply_text("🔧 Sub-agent system not available.")
            return

        user_id = await self._get_hexbrain_user_id(update.effective_user.id)
        runs = self.subagent_manager.list_runs(user_id=user_id)

        if not runs:
            await update.message.reply_text(
                "🧵 No background tasks.\n\n"
                "<i>Ask me to run something in the background and I'll use the spawn tool.</i>",
                parse_mode="HTML",
            )
            return

        status_emoji = {
            "running": "🔄",
            "done": "✅",
            "error": "❌",
            "timeout": "⏰",
            "cancelled": "🚫",
        }

        lines = ["🧵 <b>Background Tasks</b>\n"]
        for r in runs[:10]:
            emoji = status_emoji.get(r["status"], "📋")
            lines.append(
                f"{emoji} <b>{r['label']}</b>\n"
                f"   Status: {r['status']} | Model: <code>{r['model']}</code>\n"
                f"   Tokens: {r['tokens_used']:,}\n"
                f"   Started: {r['started_at'][:16]}\n"
                f"   ID: <code>{r['id']}</code>"
            )

        if len(runs) > 10:
            lines.append(f"\n<i>... and {len(runs) - 10} more tasks</i>")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _cmd_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /skills — list loaded skill plugins."""
        if not self._is_allowed(update.effective_user.id):
            return

        if not self.skill_loader or not self.skill_loader.loaded_count:
            await update.message.reply_text(
                "🧩 No skills loaded.\n\n"
                "<i>Skills are plugin modules that add tools to the agent.</i>",
                parse_mode="HTML",
            )
            return

        summary = self.skill_loader.get_summary()
        lines = [f"🧩 <b>Loaded Skills ({len(summary)})</b>\n"]

        for s in summary:
            tools_str = ", ".join(f"<code>{t}</code>" for t in s["tools"])
            lines.append(
                f"• <b>{s['name']}</b> v{s['version']}\n"
                f"  {s['description']}\n"
                f"  Tools: {tools_str}"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # Callback query handler (inline buttons)
    # ------------------------------------------------------------------
    async def _handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button clicks — process callback_data as a user message."""
        query = update.callback_query
        await query.answer()  # Acknowledge the button click

        if not self._is_allowed(update.effective_user.id):
            return

        callback_data = query.data
        chat_id = update.effective_chat.id

        # Process the callback as a message to the agent
        # Create a lightweight update-like wrapper for _process_message
        logger.info(f"[TG] Callback query: {callback_data}")

        user_id = await self._get_hexbrain_user_id(update.effective_user.id)
        session_id = await self._get_session_id(chat_id, user_id)

        cancel_event = self.get_cancel_event(chat_id)
        cancel_event.clear()

        await update.effective_chat.send_action(ChatAction.TYPING)

        handler = TelegramStreamHandler(chat_id, bot=update.get_bot())
        await handler.send_initial()
        handler._start_typing()

        try:
            _model = settings.agent_model if settings.agent_model != "auto" else None
            _thinking = getattr(self, '_thinking_budgets', {}).get(chat_id, settings.thinking_budget_default)
            response = await self.agent_runner.run(
                user_message=f"[Button clicked: {callback_data}]",
                user_id=user_id,
                session_id=session_id,
                telegram_chat_id=chat_id,
                on_text_chunk=handler.on_text_chunk,
                on_tool_start=handler.on_tool_start,
                on_tool_end=handler.on_tool_end,
                cancel_check=lambda: cancel_event.is_set(),
                model_override=_model,
                thinking_budget=_thinking,
            )

            self._session_map[chat_id] = response.session_id

            # Extract reaction and buttons from response
            final_text = response.text
            final_text, reaction_emoji = extract_reaction(final_text)
            final_text, buttons = extract_buttons(final_text)

            reply_markup = None
            if buttons:
                keyboard = [[InlineKeyboardButton(label, callback_data=cb)] for label, cb in buttons]
                reply_markup = InlineKeyboardMarkup(keyboard)

            await handler.finalize(final_text, reply_markup=reply_markup)

            if reaction_emoji:
                try:
                    await query.message.set_reaction(reaction_emoji)
                except Exception:
                    pass

        except asyncio.CancelledError:
            handler._stop_typing()
            await handler.finalize("⏹ Generation stopped.")
        except Exception as e:
            logger.exception(f"Callback query error")
            handler._stop_typing()
            await handler.finalize(f"❌ Error: {str(e)[:200]}")
        finally:
            self._cancel_flags.pop(chat_id, None)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------
    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages — supports both private and group chats."""
        if not self._is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ Not authorized.")
            return

        if not await self._is_paired(update.effective_user.id):
            await update.message.reply_text(
                "🔒 This bot requires pairing.\nUse /pair <code> to pair your account."
            )
            return

        text = update.message.text
        if not text or not text.strip():
            return

        # Group chat detection — only respond when @mentioned or replied to
        chat_type = update.effective_chat.type
        if chat_type in ("group", "supergroup"):
            bot_username = (await context.bot.get_me()).username
            is_mentioned = f"@{bot_username}" in text
            is_reply_to_bot = (
                update.message.reply_to_message
                and update.message.reply_to_message.from_user
                and update.message.reply_to_message.from_user.id == context.bot.id
            )

            if not is_mentioned and not is_reply_to_bot:
                return  # Ignore messages not directed at the bot

            # Strip the @mention from text so agent doesn't see it
            if is_mentioned:
                text = text.replace(f"@{bot_username}", "").strip()
                if not text:
                    text = "Hi"

        # Include reply-to context if the user replied to a message
        reply_context = self._extract_reply_context(update)
        if reply_context:
            text = f"{reply_context}\n\n{text}"

        await self._process_message(update, text)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming photos — pass as image_url content blocks for GPT vision."""
        if not self._is_allowed(update.effective_user.id):
            return
        if not await self._is_paired(update.effective_user.id):
            await update.message.reply_text("🔒 Pair first with /pair <code>")
            return

        # Download the largest photo
        photo = update.message.photo[-1]  # Largest resolution

        with tempfile.TemporaryDirectory() as tmpdir:
            file = await context.bot.get_file(photo.file_id)
            path = os.path.join(tmpdir, f"{photo.file_id}.jpg")
            await file.download_to_drive(path)

            caption = update.message.caption or "What's in this image?"
            await self._process_message(update, caption, media_paths=[path])

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming documents with proper file-type detection."""
        if not self._is_allowed(update.effective_user.id):
            return
        if not await self._is_paired(update.effective_user.id):
            await update.message.reply_text("🔒 Pair first with /pair <code>")
            return

        doc = update.message.document
        fname = doc.file_name or f"{doc.file_id}"
        ext = os.path.splitext(fname)[1].lower()
        mime = doc.mime_type or ""

        with tempfile.TemporaryDirectory() as tmpdir:
            file = await context.bot.get_file(doc.file_id)
            path = os.path.join(tmpdir, fname)
            await file.download_to_drive(path)

            caption = update.message.caption or f"I'm sharing this file: {fname}"

            # --- Images → vision model ---
            if mime.startswith("image/") or ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
                await self._process_message(update, caption, media_paths=[path])
                return

            # --- PDF → extract text with PyPDF2 ---
            if ext == ".pdf" or mime == "application/pdf":
                content = self._extract_pdf_text(path)
                if not content:
                    await update.message.reply_text(
                        "📄 I couldn't extract text from this PDF. "
                        "It may be scanned/image-based. Try sending it as a photo instead."
                    )
                    return
                content = content[:50000].replace("\x00", "")
                text = f"{caption}\n\nPDF content ({fname}, {len(content)} chars):\n\n{content}"
                await self._process_message(update, text)
                return

            # --- DOCX → extract text with python-docx ---
            if ext == ".docx" or mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content = self._extract_docx_text(path)
                if not content:
                    await update.message.reply_text("📄 I couldn't extract text from this DOCX file.")
                    return
                content = content[:50000].replace("\x00", "")
                text = f"{caption}\n\nDocument content ({fname}):\n\n{content}"
                await self._process_message(update, text)
                return

            # --- Text-like files → read as UTF-8 ---
            text_exts = (
                ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
                ".json", ".csv", ".xml", ".yaml", ".yml", ".toml",
                ".html", ".css", ".sh", ".bash", ".sql", ".rs",
                ".go", ".java", ".c", ".cpp", ".h", ".rb", ".php",
                ".env", ".ini", ".cfg", ".conf", ".log",
            )
            if ext in text_exts or mime.startswith("text/"):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read(50000)
                    content = content.replace("\x00", "")
                    text = f"{caption}\n\nFile content ({fname}):\n```\n{content}\n```"
                    await self._process_message(update, text)
                except Exception:
                    await self._process_message(update, caption)
                return

            # --- Unknown binary file ---
            await update.message.reply_text(
                f"📎 I received <b>{fname}</b> but I can't read this file type ({ext or mime}).\n"
                f"I support: PDF, DOCX, text files, images, and code files.",
                parse_mode="HTML",
            )

    # ------------------------------------------------------------------
    # Document text extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_pdf_text(path: str) -> str:
        """Extract text from a PDF file using PyPDF2."""
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(path)
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"--- Page {i+1} ---\n{page_text}")
            return "\n\n".join(pages)
        except Exception as e:
            logger.warning(f"PDF extraction failed for {path}: {e}")
            return ""

    @staticmethod
    def _extract_docx_text(path: str) -> str:
        """Extract text from a DOCX file using python-docx."""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.warning(f"DOCX extraction failed for {path}: {e}")
            return ""

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice/audio messages — transcribe with Whisper then process."""
        if not self._is_allowed(update.effective_user.id):
            return
        if not await self._is_paired(update.effective_user.id):
            await update.message.reply_text("🔒 Pair first with /pair <code>")
            return

        voice = update.message.voice or update.message.audio
        if not voice:
            return

        # Send typing while transcribing
        await update.effective_chat.send_action(ChatAction.TYPING)

        with tempfile.TemporaryDirectory() as tmpdir:
            file = await context.bot.get_file(voice.file_id)
            ext = ".ogg" if update.message.voice else ".mp3"
            path = os.path.join(tmpdir, f"{voice.file_id}{ext}")
            await file.download_to_drive(path)

            # Transcribe with Whisper
            from app.agent.voice_handler import transcribe_voice
            transcription = await transcribe_voice(path)

            if transcription.startswith("ERROR:"):
                await update.message.reply_text(
                    f"🎤 Voice transcription failed: {transcription}",
                )
                return

            logger.info(f"[TG] Voice transcribed: {transcription[:200]}")

            # Process the transcribed text through the agent (no echo to user)
            await self._process_message(update, transcription)

    # ------------------------------------------------------------------
    # Reply-to context
    # ------------------------------------------------------------------
    def _extract_reply_context(self, update: Update) -> str:
        """Extract context from a reply-to message, if any."""
        reply = update.message.reply_to_message
        if not reply:
            return ""

        reply_text = reply.text or reply.caption or ""
        if not reply_text:
            return ""

        # Truncate long reply context
        if len(reply_text) > 500:
            reply_text = reply_text[:500] + "..."

        sender = "you" if reply.from_user and reply.from_user.is_bot else "me"
        return f'[Replying to {sender}: "{reply_text}"]'

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    async def _process_message(
        self,
        update: Update,
        text: str,
        media_paths: Optional[list] = None,
    ):
        """
        Process a user message through the agent runtime.
        Streams the response back to Telegram with cursor + typing indicator.
        Uses a cancellation event so /stop can interrupt generation.
        Adds ACK reaction, reply-to, agent reactions, and inline buttons.
        """
        chat_id = update.effective_chat.id
        tg_user_id = update.effective_user.id
        msg_id = update.message.message_id

        # Get HexBrain user ID
        user_id = await self._get_hexbrain_user_id(tg_user_id, telegram_user=update.effective_user)

        # Get or set session (cache + DB fallback)
        session_id = await self._get_session_id(chat_id, user_id)

        # Set up cancellation flag for this chat
        cancel_event = self.get_cancel_event(chat_id)
        cancel_event.clear()  # Reset — generation is starting

        # ACK reaction — immediately show 👀 on the user's message
        try:
            await update.message.set_reaction("👀")
        except Exception:
            pass

        # Send typing indicator
        await update.effective_chat.send_action(ChatAction.TYPING)

        # Create stream handler with reply-to support
        handler = TelegramStreamHandler(
            chat_id,
            bot=update.get_bot(),
            reply_to_message_id=msg_id,
        )
        await handler.send_initial()

        # Start continuous typing indicator (for tool execution pauses)
        handler._start_typing()

        try:
            # Run the agent with cancellation check
            # Pass current model setting — "auto" or None triggers auto-routing
            _model = settings.agent_model if settings.agent_model != "auto" else None

            # Multi-agent routing: if enabled, check persona routing
            if settings.multi_agent_enabled:
                from app.agent.multi_agent import get_multi_agent_router
                _router = get_multi_agent_router()
                _persona = _router.route(text, context={"chat_id": chat_id, "user_id": user_id})
                if _persona.model_override:
                    _model = _persona.model_override
                # Prepend persona's system prompt prefix if any
                if _persona.system_prompt_prefix and _persona.name != "default":
                    text = f"[Agent: {_persona.name}] {text}"

            # Get thinking budget for this chat
            _thinking = getattr(self, '_thinking_budgets', {}).get(chat_id, settings.thinking_budget_default)

            response = await self.agent_runner.run(
                user_message=text,
                user_id=user_id,
                session_id=session_id,
                telegram_chat_id=chat_id,
                on_text_chunk=handler.on_text_chunk,
                on_tool_start=handler.on_tool_start,
                on_tool_end=handler.on_tool_end,
                media_paths=media_paths,
                cancel_check=lambda: cancel_event.is_set(),
                model_override=_model,
                thinking_budget=_thinking,
            )

            # Save session for continuity
            self._session_map[chat_id] = response.session_id

            # Extract reaction and buttons from response
            final_text = response.text
            final_text, reaction_emoji = extract_reaction(final_text)
            final_text, buttons = extract_buttons(final_text)

            # Build inline keyboard if agent included buttons
            reply_markup = None
            if buttons:
                keyboard = [[InlineKeyboardButton(label, callback_data=cb)] for label, cb in buttons]
                reply_markup = InlineKeyboardMarkup(keyboard)

            # Append model label in auto mode
            if response.model and (settings.agent_model == "auto" or not settings.agent_model):
                model_short = response.model.replace("claude-", "").replace("sonnet-4-5-20250514", "sonnet-4.5").replace("opus-4-6", "opus-4.6")
                final_text = f"{final_text}\n\n`⚡ {model_short}`"

            # Send final response (removes cursor)
            await handler.finalize(final_text, reply_markup=reply_markup)

            # Remove ACK reaction
            try:
                await update.message.set_reaction()  # empty = remove
            except Exception:
                pass

            # Apply agent reaction (if any)
            if reaction_emoji:
                try:
                    await update.message.set_reaction(reaction_emoji)
                except Exception:
                    pass

            # Log usage
            logger.info(
                f"[TG {tg_user_id}] "
                f"tokens={response.tokens_total} "
                f"tools={len(response.tool_calls)} "
                f"time={response.processing_time_ms}ms "
                f"memories={response.memories_extracted}"
            )

        except asyncio.CancelledError:
            logger.info(f"[TG {tg_user_id}] Generation cancelled via /stop")
            handler._stop_typing()
            await handler.finalize("⏹ Generation stopped.")
            # Remove ACK
            try:
                await update.message.set_reaction()
            except Exception:
                pass

        except Exception as e:
            logger.exception(f"Agent error for Telegram user {tg_user_id}")
            handler._stop_typing()
            error_text = f"❌ Sorry, something went wrong:\n`{str(e)[:200]}`"
            await handler.finalize(error_text)
            # Remove ACK
            try:
                await update.message.set_reaction()
            except Exception:
                pass

        finally:
            # Remove cancel flag so /stop shows "Nothing to stop" when idle
            self._cancel_flags.pop(chat_id, None)
