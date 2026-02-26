"""
Telegram Stream Handler — Manages streaming LLM output to Telegram messages.

Phase 2 additions:
- reply_to_message_id support (bot replies directly to user message)
- extract_reaction() — parse [[reaction:EMOJI]] from agent output
- extract_buttons() — parse [[button:TEXT|CALLBACK]] from agent output
- Strikethrough (~~text~~) and header (### / ## / #) markdown support
- reply_markup forwarding in _safe_send / _safe_edit
"""

import asyncio
import logging
import re
import time
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# LaTeX → Unicode + Markdown → HTML post-processing
# ------------------------------------------------------------------

_LATEX_REPLACEMENTS = [
    (r"\\times", "×"),
    (r"\\div", "÷"),
    (r"\\pm", "±"),
    (r"\\cdot", "·"),
    (r"\\leq", "≤"),
    (r"\\geq", "≥"),
    (r"\\neq", "≠"),
    (r"\\approx", "≈"),
    (r"\\infty", "∞"),
    (r"\\pi", "π"),
    (r"\\alpha", "α"),
    (r"\\beta", "β"),
    (r"\\theta", "θ"),
    (r"\\sum", "Σ"),
    (r"\\prod", "Π"),
    (r"\\int", "∫"),
    (r"\\Rightarrow", "⇒"),
    (r"\\rightarrow", "→"),
    (r"\\Leftarrow", "⇐"),
    (r"\\leftarrow", "←"),
    (r"\\forall", "∀"),
    (r"\\exists", "∃"),
    (r"\\in", "∈"),
    (r"\\notin", "∉"),
    (r"\\subset", "⊂"),
    (r"\\supset", "⊃"),
    (r"\\cup", "∪"),
    (r"\\cap", "∩"),
]


def clean_latex(text: str) -> str:
    """
    Strip LaTeX math formatting and replace with Unicode symbols.
    GPT sometimes outputs LaTeX despite instructions not to.
    """
    # Strip $$...$$ blocks (display math)
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    # Strip $...$ inline math
    text = re.sub(r"\$(.+?)\$", r"\1", text)
    # Strip \[...\] display math
    text = re.sub(r"\\\[(.+?)\\\]", r"\1", text, flags=re.DOTALL)
    # Strip \(...\) inline math
    text = re.sub(r"\\\((.+?)\\\)", r"\1", text, flags=re.DOTALL)
    # \text{...} → content
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    # \textbf{...} → content
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    # \frac{a}{b} → a/b
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    # \sqrt{x} → √x
    text = re.sub(r"\\sqrt\{([^}]*)\}", r"√\1", text)
    # \sqrt → √
    text = text.replace("\\sqrt", "√")
    # Symbol replacements
    for pattern, replacement in _LATEX_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    # Clean leftover braces from LaTeX: { and }
    text = re.sub(r"(?<!<)\{([^}]*)\}(?!>)", r"\1", text)
    return text


def _escape_html(text: str) -> str:
    """Escape HTML special chars, but preserve our own tags."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def markdown_to_html(text: str) -> str:
    """
    Convert simple Markdown to Telegram-safe HTML.
    Handles: **bold**, *italic*, ~~strikethrough~~, `code`,
             ```code blocks```, [links](url), # headers
    """
    # First, escape HTML entities in the raw text
    text = _escape_html(text)

    # Code blocks (``` ... ```) — must be before inline code
    text = re.sub(
        r"```(\w*)\n?(.*?)```",
        r"<pre>\2</pre>",
        text,
        flags=re.DOTALL,
    )
    # Inline code (`...`)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Strikethrough (~~...~~)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    # Headers — ### / ## / # → bold (Telegram has no <h> tags)
    text = re.sub(r"^#{1,3}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    # Bold (**...**)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic (*...*) — but not inside words like file*name
    text = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", text)
    # Italic (_..._) — underscore style
    text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", text)
    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    return text


def postprocess_for_telegram(text: str) -> str:
    """Full post-processing pipeline: LaTeX cleanup → Markdown to HTML."""
    text = clean_latex(text)
    text = markdown_to_html(text)
    return text


# ------------------------------------------------------------------
# Phase 2: Reaction & Button extraction
# ------------------------------------------------------------------

def extract_reaction(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract [[reaction:EMOJI]] from agent output.
    Returns (cleaned_text, emoji_or_None).
    The agent may place this tag anywhere in its response.
    """
    match = re.search(r"\[\[reaction:([^\]]+)\]\]", text)
    if match:
        emoji = match.group(1).strip()
        cleaned = text[: match.start()] + text[match.end() :]
        cleaned = cleaned.strip()
        return cleaned, emoji
    return text, None


def extract_buttons(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Extract [[button:TEXT|CALLBACK_DATA]] markers from agent output.
    Returns (cleaned_text, [(label, callback_data), ...]).
    """
    buttons: List[Tuple[str, str]] = []
    pattern = re.compile(r"\[\[button:([^|]+)\|([^\]]+)\]\]")

    for m in pattern.finditer(text):
        label = m.group(1).strip()
        callback = m.group(2).strip()
        buttons.append((label, callback))

    cleaned = pattern.sub("", text).strip()
    return cleaned, buttons


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Telegram message length limit
TG_MAX_LEN = 4096

# Minimum interval between message edits (Telegram rate limit)
EDIT_THROTTLE_MS = 300
EDIT_THROTTLE_S = EDIT_THROTTLE_MS / 1000.0

# Cursor shown while streaming
CURSOR = " ▌"


def split_message(text: str, max_len: int = TG_MAX_LEN) -> List[str]:
    """
    Split a long message into chunks that fit Telegram's limit.
    Tries to split at paragraph, sentence, or word boundaries.
    """
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Try to split at double newline (paragraph)
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1 or split_at < max_len // 4:
            # Try single newline
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1 or split_at < max_len // 4:
            # Try period + space (sentence boundary)
            split_at = text.rfind(". ", 0, max_len)
            if split_at != -1:
                split_at += 1  # keep the period
        if split_at == -1 or split_at < max_len // 4:
            # Try space (word boundary)
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1 or split_at < max_len // 4:
            # Hard split
            split_at = max_len

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    return chunks


class TelegramStreamHandler:
    """
    Handles streaming LLM output to a Telegram message by editing it.

    Features:
    - Shows cursor (▌) during streaming
    - Throttles edits to 300ms minimum interval
    - Shows tool execution status
    - Splits long messages
    - Sends continuous typing indicator during tool execution
    - reply_to_message_id support (Phase 2)
    - reply_markup forwarding (Phase 2)
    """

    def __init__(
        self,
        chat_id: int,
        bot,
        reply_to_message_id: Optional[int] = None,
    ):
        self.chat_id = chat_id
        self.bot = bot
        self.reply_to_message_id = reply_to_message_id
        self.message = None  # The Telegram message object being edited
        self.buffer = ""
        self.last_edit_time = 0.0
        self.tool_status = ""
        self._edit_lock = asyncio.Lock()
        self._typing_task: Optional[asyncio.Task] = None
        self._finalized = False

    # Threshold to split during streaming (leave room for cursor + HTML)
    STREAM_SPLIT_THRESHOLD = TG_MAX_LEN - 200

    async def on_text_chunk(self, chunk: str):
        """Called for each text chunk from the LLM stream."""
        self.buffer += chunk

        # If buffer exceeds Telegram limit during streaming, flush the
        # current message and start a new one so the user doesn't see a
        # frozen response while the agent keeps generating.
        if len(self.buffer) > self.STREAM_SPLIT_THRESHOLD:
            await self._flush_and_continue()
            return

        now = time.time()
        if now - self.last_edit_time >= EDIT_THROTTLE_S and len(self.buffer) > 5:
            # Clean LaTeX on-the-fly for streaming preview
            display = clean_latex(self.buffer) + CURSOR
            if len(display) <= TG_MAX_LEN:
                await self._update_message(display)

    async def _flush_and_continue(self):
        """
        Finalize the current streaming message and start a fresh one.

        Called when the buffer grows beyond TG_MAX_LEN during streaming,
        so the user sees progressive output instead of a frozen message.
        """
        # Post-process what we have so far
        flushed_text = postprocess_for_telegram(self.buffer)

        # Split in case post-processing made it longer
        chunks = split_message(flushed_text)

        if self.message:
            # Edit the current message with the first chunk (no cursor)
            await self._safe_edit(self.message, chunks[0])
            # Send any overflow as new messages
            for extra in chunks[1:]:
                await asyncio.sleep(0.3)
                await self._safe_send(extra)
        else:
            for chunk in chunks:
                await self._safe_send(chunk)

        # Reset buffer and start a new message for the continuation
        self.buffer = ""
        self.message = None  # Next _update_message will create a new msg
        self.last_edit_time = 0.0
        logger.debug(f"[STREAM] Flushed {len(flushed_text)} chars, continuing in new message")

    async def on_tool_start(self, tool_name: str):
        """Called when the LLM starts using a tool."""
        self.tool_status = f"\n\n🔧 <i>Using {tool_name}...</i>"
        display = (self.buffer + self.tool_status) if self.buffer else f"🔧 <i>Using {tool_name}...</i>"
        await self._update_message(display[:TG_MAX_LEN])
        # Start continuous typing indicator
        self._start_typing()

    async def on_tool_end(self, tool_name: str, result_summary: str):
        """Called when a tool finishes executing."""
        self._stop_typing()
        self.tool_status = f"\n\n✅ <i>{tool_name} done</i>"
        display = (self.buffer + self.tool_status) if self.buffer else f"✅ <i>{tool_name} done</i>"
        await self._update_message(display[:TG_MAX_LEN])

    async def on_tool_progress(self, tool_name: str, chunk: str):
        """Stream incremental tool output to the user (buffered)."""
        if not chunk or not chunk.strip():
            return
        if not hasattr(self, "_progress_buffers"):
            self._progress_buffers = {}
        key = f"tp_{tool_name}"
        self._progress_buffers[key] = self._progress_buffers.get(key, "") + chunk
        # Flush every 300 chars to avoid Telegram rate limits
        if len(self._progress_buffers[key]) < 300:
            return
        text = self._progress_buffers.pop(key)
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"<pre>{_escape_html(text[:2000])}</pre>",
                parse_mode="HTML",
            )
        except Exception:
            pass  # Don't crash the tool on streaming errors

    async def flush_tool_progress(self):
        """Flush any remaining buffered tool progress."""
        if not hasattr(self, "_progress_buffers"):
            return
        for key, text in list(self._progress_buffers.items()):
            if text.strip():
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=f"<pre>{_escape_html(text[:2000])}</pre>",
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
        self._progress_buffers.clear()

    async def finalize(self, final_text: str, reply_markup=None):
        """Send the complete final response (no cursor)."""
        self._stop_typing()
        self._finalized = True

        if not final_text:
            final_text = "<i>(no response)</i>"
        else:
            final_text = postprocess_for_telegram(final_text)

        chunks = split_message(final_text)

        if self.message:
            # Edit existing message with first chunk
            await self._safe_edit(self.message, chunks[0], reply_markup=reply_markup)
            # Send remaining chunks as new messages
            for extra in chunks[1:]:
                await asyncio.sleep(0.3)
                await self._safe_send(extra)
        else:
            # No message exists yet, send all as new
            for i, chunk in enumerate(chunks):
                if i > 0:
                    await asyncio.sleep(0.3)
                markup = reply_markup if i == len(chunks) - 1 else None
                await self._safe_send(chunk, reply_markup=markup)

    async def send_initial(self, text: str = "🧠 Thinking..."):
        """Send the initial 'thinking' message, replying to the user."""
        try:
            self.message = await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_to_message_id=self.reply_to_message_id,
            )
            self.last_edit_time = time.time()
        except Exception as e:
            logger.warning(f"Failed to send initial message: {e}")

    # ------------------------------------------------------------------
    # Typing indicator
    # ------------------------------------------------------------------
    def _start_typing(self):
        """Start sending typing action every 4 seconds."""
        if self._typing_task and not self._typing_task.done():
            return
        self._typing_task = asyncio.create_task(self._typing_loop())

    def _stop_typing(self):
        """Stop the typing indicator."""
        if self._typing_task and not self._typing_task.done():
            self._typing_task.cancel()
            self._typing_task = None

    async def _typing_loop(self):
        """Send typing action periodically."""
        try:
            while not self._finalized:
                try:
                    from telegram.constants import ChatAction
                    await self.bot.send_chat_action(
                        chat_id=self.chat_id,
                        action=ChatAction.TYPING,
                    )
                except Exception:
                    pass
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _update_message(self, text: str):
        """Update the Telegram message with rate limiting."""
        async with self._edit_lock:
            now = time.time()
            if now - self.last_edit_time < EDIT_THROTTLE_S:
                return  # Skip this update (too fast)

            if self.message is None:
                await self.send_initial(text)
            else:
                await self._safe_edit(self.message, text[:TG_MAX_LEN])

            self.last_edit_time = time.time()

    async def _safe_edit(self, message, text: str, reply_markup=None):
        """Edit a message, trying HTML first then plaintext fallback."""
        try:
            await message.edit_text(
                text=text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
        except Exception:
            try:
                await message.edit_text(text=text, reply_markup=reply_markup)
            except Exception as e:
                if "Message is not modified" not in str(e):
                    logger.debug(f"Message edit failed: {e}")

    async def _safe_send(self, text: str, reply_markup=None):
        """Send a new message with HTML fallback."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
        except Exception:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    reply_markup=reply_markup,
                )
            except Exception as e:
                logger.warning(f"Failed to send message: {e}")
