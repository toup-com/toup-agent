"""
Chat Commands System — /status, /new, /reset, /compact, /model, /usage, /think, /verbose, /activation, /config, /voice
Provides a registry-based command dispatch system for Telegram and other channels.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CommandScope(str, Enum):
    """Where a command can be used."""
    DM = "dm"
    GROUP = "group"
    ALL = "all"


@dataclass
class CommandDef:
    """Definition of a chat command."""
    name: str
    description: str
    handler: Optional[Callable] = None
    scope: CommandScope = CommandScope.ALL
    aliases: List[str] = field(default_factory=list)
    admin_only: bool = False
    hidden: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "scope": self.scope.value,
            "aliases": self.aliases,
            "admin_only": self.admin_only,
            "hidden": self.hidden,
        }


class CommandRegistry:
    """Registry for chat commands with dispatch."""

    def __init__(self):
        self._commands: Dict[str, CommandDef] = {}
        self._aliases: Dict[str, str] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in commands."""
        builtins = [
            CommandDef("status", "Show agent status, model, usage, channel state"),
            CommandDef("new", "Start a fresh session"),
            CommandDef("reset", "Reset current session context"),
            CommandDef("compact", "Force context compaction"),
            CommandDef("model", "Switch model for current session", aliases=["m"]),
            CommandDef("usage", "Show token/cost usage summary"),
            CommandDef("think", "Set thinking level (low/medium/high/xhigh)", aliases=["t"]),
            CommandDef("verbose", "Toggle tool narration verbosity", aliases=["v"]),
            CommandDef("activation", "Set/show activation (boot) prompt"),
            CommandDef("config", "Get/set config values from chat"),
            CommandDef("voice", "Voice/TTS settings"),
            CommandDef("help", "Show available commands", aliases=["h", "?"]),
            CommandDef("skills", "List installed skills"),
            CommandDef("auto", "Toggle auto-mode"),
        ]
        for cmd in builtins:
            self.register(cmd)

    def register(self, cmd: CommandDef) -> None:
        """Register a command."""
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._aliases[alias] = cmd.name

    def unregister(self, name: str) -> bool:
        """Unregister a command."""
        cmd = self._commands.pop(name, None)
        if cmd:
            for alias in cmd.aliases:
                self._aliases.pop(alias, None)
            return True
        return False

    def get(self, name: str) -> Optional[CommandDef]:
        """Get a command by name or alias."""
        resolved = self._aliases.get(name, name)
        return self._commands.get(resolved)

    def list_commands(self, include_hidden: bool = False) -> List[CommandDef]:
        """List all registered commands."""
        cmds = list(self._commands.values())
        if not include_hidden:
            cmds = [c for c in cmds if not c.hidden]
        return sorted(cmds, key=lambda c: c.name)

    def parse(self, text: str) -> Optional[tuple]:
        """Parse a command from message text. Returns (command_name, args) or None."""
        if not text or not text.startswith("/"):
            return None
        parts = text[1:].split(None, 1)
        cmd_name = parts[0].lower().split("@")[0]  # Handle /cmd@botname
        args = parts[1] if len(parts) > 1 else ""
        resolved = self._aliases.get(cmd_name, cmd_name)
        if resolved in self._commands:
            return (resolved, args)
        return None

    async def execute(self, text: str, context: Optional[dict] = None) -> Optional[dict]:
        """Parse and execute a command. Returns result dict or None."""
        parsed = self.parse(text)
        if not parsed:
            return None
        cmd_name, args = parsed
        cmd = self._commands.get(cmd_name)
        if not cmd:
            return {"error": f"Unknown command: {cmd_name}"}
        if not cmd.handler:
            return await self._handle_builtin(cmd_name, args, context or {})
        try:
            import asyncio
            if asyncio.iscoroutinefunction(cmd.handler):
                return await cmd.handler(args, context or {})
            return cmd.handler(args, context or {})
        except Exception as e:
            logger.error(f"Command /{cmd_name} failed: {e}")
            return {"error": str(e)}

    async def _handle_builtin(self, name: str, args: str, ctx: dict) -> dict:
        """Handle built-in commands."""
        if name == "help":
            cmds = self.list_commands()
            lines = [f"/{c.name} — {c.description}" for c in cmds]
            return {"text": "Available commands:\n" + "\n".join(lines)}
        elif name == "status":
            return await self._cmd_status(ctx)
        elif name == "new":
            return {"text": "🆕 New session started.", "action": "new_session"}
        elif name == "reset":
            return {"text": "🔄 Session reset.", "action": "reset_session"}
        elif name == "compact":
            return {"text": "📦 Context compacted.", "action": "compact"}
        elif name == "model":
            return await self._cmd_model(args, ctx)
        elif name == "usage":
            return await self._cmd_usage(ctx)
        elif name == "think":
            return self._cmd_think(args)
        elif name == "verbose":
            return self._cmd_verbose(args, ctx)
        elif name == "activation":
            return self._cmd_activation(args, ctx)
        elif name == "config":
            return self._cmd_config(args, ctx)
        elif name == "voice":
            return {"text": "🎤 Voice settings — use /voice on|off|auto"}
        elif name == "skills":
            return {"text": "📦 Skills — use /skills list|install|remove"}
        elif name == "auto":
            return {"text": "🤖 Auto-mode toggled.", "action": "toggle_auto"}
        return {"text": f"Command /{name} not yet implemented."}

    async def _cmd_status(self, ctx: dict) -> dict:
        model = ctx.get("model", "claude-opus-4-6")
        uptime = ctx.get("uptime", 0)
        channel = ctx.get("channel", "unknown")
        session_id = ctx.get("session_id", "N/A")
        return {
            "text": f"📊 Status\nModel: {model}\nChannel: {channel}\nSession: {session_id}\nUptime: {int(uptime)}s"
        }

    async def _cmd_model(self, args: str, ctx: dict) -> dict:
        if not args.strip():
            current = ctx.get("model", "claude-opus-4-6")
            available = ["claude-opus-4-6", "gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"]
            return {
                "text": f"Current model: {current}\nAvailable: {', '.join(available)}\nUsage: /model <name>"
            }
        model_name = args.strip()
        return {"text": f"✅ Model switched to: {model_name}", "action": "set_model", "model": model_name}

    async def _cmd_usage(self, ctx: dict) -> dict:
        tokens_in = ctx.get("tokens_in", 0)
        tokens_out = ctx.get("tokens_out", 0)
        cost = ctx.get("cost", 0.0)
        return {
            "text": f"📈 Usage\nTokens in: {tokens_in:,}\nTokens out: {tokens_out:,}\nCost: ${cost:.4f}"
        }

    def _cmd_think(self, args: str) -> dict:
        levels = ["low", "medium", "high", "xhigh"]
        if not args.strip():
            return {"text": f"🧠 Thinking levels: {', '.join(levels)}\nUsage: /think <level>"}
        level = args.strip().lower()
        if level not in levels:
            return {"text": f"❌ Invalid level. Choose: {', '.join(levels)}"}
        budgets = {"low": 1024, "medium": 4096, "high": 10000, "xhigh": 32000}
        return {
            "text": f"🧠 Thinking set to: {level} (budget: {budgets[level]} tokens)",
            "action": "set_thinking",
            "level": level,
            "budget": budgets[level],
        }

    def _cmd_verbose(self, args: str, ctx: dict) -> dict:
        current = ctx.get("verbose", False)
        new_val = not current
        return {
            "text": f"🔊 Verbose mode: {'ON' if new_val else 'OFF'}",
            "action": "set_verbose",
            "verbose": new_val,
        }

    def _cmd_activation(self, args: str, ctx: dict) -> dict:
        if not args.strip():
            current = ctx.get("activation_prompt", "(none)")
            return {"text": f"🚀 Activation prompt:\n{current}"}
        return {
            "text": f"✅ Activation prompt set.",
            "action": "set_activation",
            "prompt": args.strip(),
        }

    def _cmd_config(self, args: str, ctx: dict) -> dict:
        if not args.strip():
            return {"text": "⚙️ Config — Usage: /config get <key> | /config set <key> <value>"}
        parts = args.strip().split(None, 2)
        action = parts[0].lower()
        if action == "get" and len(parts) >= 2:
            key = parts[1]
            val = ctx.get(key, "(not set)")
            return {"text": f"⚙️ {key} = {val}"}
        elif action == "set" and len(parts) >= 3:
            key, value = parts[1], parts[2]
            return {"text": f"✅ {key} = {value}", "action": "set_config", "key": key, "value": value}
        return {"text": "❌ Usage: /config get <key> | /config set <key> <value>"}


_registry: Optional[CommandRegistry] = None


def get_command_registry() -> CommandRegistry:
    """Get the global command registry singleton."""
    global _registry
    if _registry is None:
        _registry = CommandRegistry()
    return _registry
