"""
Skill base class — the contract every HexBrain plugin must implement.

A Skill is a self-contained capability module that:
  1. Registers one or more agent tools
  2. Optionally injects system-prompt sections
  3. Can hook into startup/shutdown lifecycle

Example usage:
    class MySkill(Skill):
        meta = SkillMeta(name="my_skill", version="1.0.0", description="...")

        def get_tools(self) -> list[dict]:
            return [...]

        async def execute_tool(self, tool_name, args, ctx):
            ...
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillMeta:
    """Metadata descriptor for a skill plugin."""
    name: str               # Unique slug, e.g. "toup"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    # If True, the skill is only loaded when explicitly enabled via config/env
    requires_opt_in: bool = False


@dataclass
class SkillContext:
    """
    Runtime context passed to every tool execution.

    Gives the skill access to workspace path, current user, DB session factory,
    and a helper to call shell commands safely.
    """
    workspace: str = ""
    user_id: str = ""
    session_id: str = ""
    chat_id: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Skill(abc.ABC):
    """Abstract base class for a HexBrain skill plugin."""

    # Every subclass MUST set this
    meta: SkillMeta

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Return tool definitions in the standard schema:

        [
            {
                "name": "skill_name__tool_name",
                "description": "...",
                "input_schema": { "type": "object", "properties": {...}, "required": [...] },
            },
            ...
        ]

        Tool names MUST be prefixed with `<skill_name>__` (double underscore)
        to avoid collisions.  The loader enforces this.
        """
        ...

    @abc.abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        ctx: SkillContext,
    ) -> str:
        """
        Execute a tool call and return the text result.

        `tool_name` is the full prefixed name, e.g. "toup__create_spec".
        The skill should dispatch to the appropriate handler.
        """
        ...

    # ------------------------------------------------------------------
    # System prompt injection (optional)
    # ------------------------------------------------------------------
    def get_system_prompt_section(self) -> Optional[str]:
        """
        Return an optional section to inject into the agent's system prompt.
        Return None to skip.
        """
        return None

    # ------------------------------------------------------------------
    # Chat command registration (optional)
    # ------------------------------------------------------------------
    def get_commands(self) -> List[Dict[str, Any]]:
        """
        Return chat commands this skill wants to register.

        Each command dict:
          {
              "command": "mycommand",       # without / prefix
              "description": "What it does",
              "handler": self.handle_mycommand,  # async callable(update, context, args)
          }

        Return empty list to skip.
        """
        return []

    # ------------------------------------------------------------------
    # Hook registration (optional)
    # ------------------------------------------------------------------
    def get_hooks(self) -> List[Dict[str, Any]]:
        """
        Return lifecycle hooks this skill wants to subscribe to.

        Each hook dict:
          {
              "event": "BEFORE_TOOL_CALL",  # HookEvent name
              "handler": self.on_tool_call,  # async callable(data)
          }

        Return empty list to skip.
        """
        return []

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional)
    # ------------------------------------------------------------------
    async def on_load(self) -> None:
        """Called once when the skill is loaded at startup."""
        pass

    async def on_unload(self) -> None:
        """Called on shutdown / hot-reload."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prefix(self, name: str) -> str:
        """Convenience: prefix a short tool name with the skill slug."""
        return f"{self.meta.name}__{name}"
