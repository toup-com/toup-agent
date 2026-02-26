"""
TUI Chat — Terminal-based chat interface.

Provides a text-based interactive chat UI for terminal sessions.
Supports message display, input handling, command parsing,
and formatting for terminal output.

Usage:
    from app.agent.tui_chat import TUIChat

    tui = TUIChat(agent_id="default")
    tui.display_message("assistant", "Hello!")
    tui.display_message("user", "Hi there!")
    output = tui.render()
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class TUIMessage:
    """A message in the TUI chat."""
    role: MessageRole
    content: str
    timestamp: float = 0.0
    tool_name: Optional[str] = None
    tokens: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def format(self, width: int = 80, show_time: bool = False) -> str:
        """Format message for terminal display."""
        prefix_map = {
            MessageRole.USER: "You",
            MessageRole.ASSISTANT: "AI",
            MessageRole.SYSTEM: "System",
            MessageRole.TOOL: f"Tool[{self.tool_name or '?'}]",
        }
        prefix = prefix_map.get(self.role, "?")

        time_str = ""
        if show_time:
            t = time.localtime(self.timestamp)
            time_str = f" [{t.tm_hour:02d}:{t.tm_min:02d}]"

        header = f"┌─ {prefix}{time_str}"
        lines = self.content.split("\n")
        body = "\n".join(f"│ {line}" for line in lines)
        footer = "└" + "─" * min(len(lines[0]) + 2 if lines else 4, width - 1)

        return f"{header}\n{body}\n{footer}"


@dataclass
class TUIState:
    """Current state of the TUI."""
    width: int = 80
    show_timestamps: bool = False
    show_tokens: bool = False
    prompt: str = ">>> "
    status_line: str = ""
    scroll_offset: int = 0
    max_history: int = 100


class TUIChat:
    """
    Terminal-based chat interface.

    Renders chat messages in a terminal-friendly format with
    command support and display options.
    """

    def __init__(
        self,
        agent_id: str = "default",
        width: int = 80,
    ):
        self._agent_id = agent_id
        self._messages: List[TUIMessage] = []
        self._state = TUIState(width=width)
        self._command_history: List[str] = []
        self._input_buffer: str = ""

    def display_message(
        self,
        role: str,
        content: str,
        *,
        tool_name: Optional[str] = None,
        tokens: int = 0,
    ) -> TUIMessage:
        """Add a message to the display."""
        msg = TUIMessage(
            role=MessageRole(role) if role in MessageRole.__members__.values() else MessageRole.SYSTEM,
            content=content,
            tool_name=tool_name,
            tokens=tokens,
        )

        # Handle role string matching
        try:
            msg.role = MessageRole(role)
        except ValueError:
            msg.role = MessageRole.SYSTEM

        self._messages.append(msg)

        # Trim history
        if len(self._messages) > self._state.max_history:
            self._messages = self._messages[-self._state.max_history:]

        return msg

    def render(self, last_n: Optional[int] = None) -> str:
        """Render the chat display."""
        msgs = self._messages[-(last_n or len(self._messages)):]
        parts = []

        # Header
        parts.append(f"╔{'═' * (self._state.width - 2)}╗")
        title = f" Agent: {self._agent_id} "
        pad = self._state.width - 2 - len(title)
        parts.append(f"║{title}{'─' * max(0, pad)}║")
        parts.append(f"╠{'═' * (self._state.width - 2)}╣")

        # Messages
        for msg in msgs:
            formatted = msg.format(
                width=self._state.width,
                show_time=self._state.show_timestamps,
            )
            parts.append(formatted)
            parts.append("")

        # Status
        if self._state.status_line:
            parts.append(f"─ {self._state.status_line}")

        # Prompt
        parts.append(f"\n{self._state.prompt}")

        return "\n".join(parts)

    def parse_input(self, text: str) -> Dict[str, Any]:
        """Parse user input, detecting commands."""
        text = text.strip()
        self._command_history.append(text)

        if text.startswith("/"):
            parts = text[1:].split(maxsplit=1)
            return {
                "type": "command",
                "command": parts[0] if parts else "",
                "args": parts[1] if len(parts) > 1 else "",
                "raw": text,
            }
        else:
            return {
                "type": "message",
                "content": text,
                "raw": text,
            }

    def set_status(self, status: str) -> None:
        """Set the status line."""
        self._state.status_line = status

    def set_option(self, key: str, value: Any) -> bool:
        """Set a display option."""
        if key == "timestamps":
            self._state.show_timestamps = bool(value)
            return True
        elif key == "tokens":
            self._state.show_tokens = bool(value)
            return True
        elif key == "width":
            self._state.width = int(value)
            return True
        elif key == "prompt":
            self._state.prompt = str(value)
            return True
        return False

    def clear(self) -> None:
        """Clear the message history."""
        self._messages.clear()

    def get_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history."""
        return [{"input": cmd} for cmd in self._command_history[-n:]]

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def stats(self) -> Dict[str, Any]:
        by_role: Dict[str, int] = {}
        for m in self._messages:
            by_role[m.role.value] = by_role.get(m.role.value, 0) + 1

        return {
            "agent_id": self._agent_id,
            "total_messages": len(self._messages),
            "by_role": by_role,
            "commands_entered": len(self._command_history),
            "width": self._state.width,
        }
