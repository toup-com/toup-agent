"""
Verbose Mode — Toggle tool narration per session.

When verbose is enabled, the agent narrates each tool call
to the user before executing it, including the tool name,
parameters, and results. When off, tools run silently.

Usage:
    from app.agent.verbose_mode import get_verbose_manager

    mgr = get_verbose_manager()
    mgr.set_verbose("session_1", True)
    if mgr.is_verbose("session_1"):
        narration = mgr.format_tool_call("web_search", {"query": "python docs"})
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VerboseLevel(str, Enum):
    OFF = "off"
    BASIC = "basic"       # Tool names only
    DETAILED = "detailed"  # Tool names + params
    FULL = "full"          # Tool names + params + results


@dataclass
class VerboseConfig:
    """Verbose mode configuration for a session."""
    session_id: str
    enabled: bool = False
    level: VerboseLevel = VerboseLevel.BASIC
    show_timing: bool = True
    show_params: bool = True
    show_results: bool = False
    max_result_length: int = 500
    changed_at: float = 0.0
    narration_count: int = 0

    def __post_init__(self):
        if self.changed_at == 0.0:
            self.changed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "enabled": self.enabled,
            "level": self.level.value,
            "show_timing": self.show_timing,
            "show_params": self.show_params,
            "show_results": self.show_results,
            "narration_count": self.narration_count,
        }


class VerboseManager:
    """
    Manages verbose mode (tool narration) per session.

    When enabled, formats human-readable descriptions of
    tool calls for the user to follow along.
    """

    def __init__(self, default_enabled: bool = False):
        self._configs: Dict[str, VerboseConfig] = {}
        self._default_enabled = default_enabled

    def set_verbose(
        self,
        session_id: str,
        enabled: bool,
        *,
        level: VerboseLevel = VerboseLevel.BASIC,
    ) -> VerboseConfig:
        """Set verbose mode for a session."""
        config = self._configs.get(session_id)
        if config:
            config.enabled = enabled
            config.level = level
            config.changed_at = time.time()
        else:
            config = VerboseConfig(
                session_id=session_id,
                enabled=enabled,
                level=level,
            )
            self._configs[session_id] = config

        logger.info(f"[VERBOSE] Session {session_id}: {'ON' if enabled else 'OFF'} ({level.value})")
        return config

    def is_verbose(self, session_id: str) -> bool:
        """Check if verbose mode is enabled for a session."""
        config = self._configs.get(session_id)
        if config:
            return config.enabled
        return self._default_enabled

    def get_level(self, session_id: str) -> VerboseLevel:
        """Get the verbose level for a session."""
        config = self._configs.get(session_id)
        if config and config.enabled:
            return config.level
        return VerboseLevel.OFF

    def get_config(self, session_id: str) -> Optional[VerboseConfig]:
        """Get full verbose config for a session."""
        return self._configs.get(session_id)

    def toggle(self, session_id: str) -> bool:
        """Toggle verbose mode and return new state."""
        config = self._configs.get(session_id)
        if config:
            config.enabled = not config.enabled
            config.changed_at = time.time()
            return config.enabled
        else:
            self.set_verbose(session_id, True)
            return True

    def format_tool_call(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        session_id: str = "",
    ) -> str:
        """
        Format a tool call narration.

        Returns a human-readable description of the tool call.
        """
        config = self._configs.get(session_id) if session_id else None
        level = config.level if config else VerboseLevel.BASIC

        if config:
            config.narration_count += 1

        if level == VerboseLevel.BASIC:
            return f"🔧 Using **{tool_name}**..."

        if level in (VerboseLevel.DETAILED, VerboseLevel.FULL):
            parts = [f"🔧 Using **{tool_name}**"]
            if params:
                param_strs = []
                for k, v in params.items():
                    val_str = str(v)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    param_strs.append(f"  `{k}`: {val_str}")
                parts.append("\n".join(param_strs))
            return "\n".join(parts)

        return f"🔧 {tool_name}"

    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: float = 0.0,
        *,
        session_id: str = "",
    ) -> str:
        """Format a tool result narration."""
        config = self._configs.get(session_id) if session_id else None
        level = config.level if config else VerboseLevel.BASIC
        max_len = config.max_result_length if config else 500

        parts = [f"✅ **{tool_name}** completed"]

        if config and config.show_timing and duration_ms > 0:
            parts[0] += f" ({duration_ms:.0f}ms)"

        if level == VerboseLevel.FULL:
            result_str = str(result)
            if len(result_str) > max_len:
                result_str = result_str[:max_len] + "..."
            parts.append(f"  Result: {result_str}")

        return "\n".join(parts)

    def format_status(self, session_id: str) -> str:
        """Format verbose status for display."""
        config = self._configs.get(session_id)
        if config:
            state = "ON" if config.enabled else "OFF"
            return (
                f"📝 **Verbose: {state}** ({config.level.value})\n"
                f"  Narrations: {config.narration_count}"
            )
        return "📝 **Verbose: OFF** (default)"

    def remove_config(self, session_id: str) -> bool:
        """Remove session config."""
        return self._configs.pop(session_id, None) is not None

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all verbose configs."""
        return [c.to_dict() for c in self._configs.values()]

    def stats(self) -> Dict[str, Any]:
        """Get verbose mode statistics."""
        enabled = sum(1 for c in self._configs.values() if c.enabled)
        total_narrations = sum(c.narration_count for c in self._configs.values())
        return {
            "total_configs": len(self._configs),
            "enabled": enabled,
            "disabled": len(self._configs) - enabled,
            "total_narrations": total_narrations,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[VerboseManager] = None


def get_verbose_manager() -> VerboseManager:
    """Get the global verbose manager."""
    global _manager
    if _manager is None:
        _manager = VerboseManager()
    return _manager
