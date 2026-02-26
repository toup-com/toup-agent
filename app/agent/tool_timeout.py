"""
Per-Tool Timeout — Configurable timeout per tool call.

Each tool can have its own timeout. If a tool exceeds its timeout,
the call is cancelled and a TimeoutError is raised. Falls back to
the global timeout if no per-tool timeout is configured.

Usage:
    from app.agent.tool_timeout import get_timeout_manager

    mgr = get_timeout_manager()
    mgr.set_timeout("web_search", 30.0)
    mgr.set_timeout("exec", 120.0)

    timeout = mgr.get_timeout("web_search")  # 30.0
    timeout = mgr.get_timeout("unknown_tool")  # falls back to default
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolTimeoutConfig:
    """Timeout configuration for a single tool."""
    tool_name: str
    timeout_seconds: float
    warn_at_percent: float = 80.0  # Warn when this % of timeout elapsed
    max_retries: int = 0


@dataclass
class TimeoutEvent:
    """Record of a timeout or near-timeout event."""
    tool_name: str
    elapsed_seconds: float
    timeout_seconds: float
    timed_out: bool
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "timeout_seconds": self.timeout_seconds,
            "timed_out": self.timed_out,
        }


class ToolTimeoutManager:
    """
    Manages per-tool timeout configuration and enforcement.

    Tools can have individual timeouts. If not configured, the global
    default timeout is used. Tracks timeout history for diagnostics.
    """

    def __init__(self, default_timeout: float = 60.0):
        self._default_timeout = default_timeout
        self._timeouts: Dict[str, ToolTimeoutConfig] = {}
        self._history: List[TimeoutEvent] = []
        self._max_history: int = 200

    @property
    def default_timeout(self) -> float:
        return self._default_timeout

    @default_timeout.setter
    def default_timeout(self, value: float):
        if value <= 0:
            raise ValueError("Default timeout must be positive")
        self._default_timeout = value

    def set_timeout(
        self,
        tool_name: str,
        timeout_seconds: float,
        *,
        warn_at_percent: float = 80.0,
        max_retries: int = 0,
    ) -> ToolTimeoutConfig:
        """Set a custom timeout for a specific tool."""
        config = ToolTimeoutConfig(
            tool_name=tool_name,
            timeout_seconds=timeout_seconds,
            warn_at_percent=warn_at_percent,
            max_retries=max_retries,
        )
        self._timeouts[tool_name] = config
        logger.info(f"[TIMEOUT] Set {tool_name} timeout to {timeout_seconds}s")
        return config

    def get_timeout(self, tool_name: str) -> float:
        """Get the effective timeout for a tool (per-tool or default)."""
        config = self._timeouts.get(tool_name)
        if config:
            return config.timeout_seconds
        return self._default_timeout

    def get_config(self, tool_name: str) -> Optional[ToolTimeoutConfig]:
        """Get the full timeout config for a tool."""
        return self._timeouts.get(tool_name)

    def remove_timeout(self, tool_name: str) -> bool:
        """Remove a per-tool timeout (reverts to default)."""
        return self._timeouts.pop(tool_name, None) is not None

    async def execute_with_timeout(
        self,
        tool_name: str,
        coro: Coroutine,
    ) -> Any:
        """
        Execute a coroutine with the tool's configured timeout.

        Args:
            tool_name: Name of the tool being executed.
            coro: The coroutine to execute.

        Returns:
            The result of the coroutine.

        Raises:
            asyncio.TimeoutError: If the tool exceeds its timeout.
        """
        timeout = self.get_timeout(tool_name)
        t0 = time.time()

        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            elapsed = time.time() - t0

            # Check if we're close to timeout
            config = self._timeouts.get(tool_name)
            warn_pct = config.warn_at_percent if config else 80.0
            if elapsed / timeout * 100 >= warn_pct:
                logger.warning(
                    "[TIMEOUT] %s took %.1fs (%.0f%% of %.0fs timeout)",
                    tool_name, elapsed, elapsed / timeout * 100, timeout,
                )
                self._record(tool_name, elapsed, timeout, timed_out=False)

            return result

        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            logger.error(
                "[TIMEOUT] %s timed out after %.1fs (limit: %.0fs)",
                tool_name, elapsed, timeout,
            )
            self._record(tool_name, elapsed, timeout, timed_out=True)
            raise

    def _record(self, tool_name: str, elapsed: float, timeout: float, timed_out: bool):
        """Record a timeout event."""
        event = TimeoutEvent(
            tool_name=tool_name,
            elapsed_seconds=elapsed,
            timeout_seconds=timeout,
            timed_out=timed_out,
        )
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(self, tool_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get timeout event history."""
        events = self._history
        if tool_name:
            events = [e for e in events if e.tool_name == tool_name]
        return [e.to_dict() for e in events[-limit:]]

    def get_timeout_stats(self) -> Dict[str, Any]:
        """Get timeout statistics."""
        total = len(self._history)
        timed_out = sum(1 for e in self._history if e.timed_out)
        by_tool: Dict[str, Dict[str, int]] = {}
        for e in self._history:
            if e.tool_name not in by_tool:
                by_tool[e.tool_name] = {"total": 0, "timed_out": 0}
            by_tool[e.tool_name]["total"] += 1
            if e.timed_out:
                by_tool[e.tool_name]["timed_out"] += 1

        return {
            "total_events": total,
            "total_timeouts": timed_out,
            "timeout_rate": round(timed_out / total * 100, 1) if total else 0.0,
            "by_tool": by_tool,
            "configured_tools": len(self._timeouts),
            "default_timeout": self._default_timeout,
        }

    def list_configured(self) -> List[Dict[str, Any]]:
        """List all tools with custom timeouts."""
        return [
            {
                "tool_name": c.tool_name,
                "timeout_seconds": c.timeout_seconds,
                "warn_at_percent": c.warn_at_percent,
                "max_retries": c.max_retries,
            }
            for c in self._timeouts.values()
        ]

    def clear_history(self) -> int:
        """Clear timeout history. Returns count cleared."""
        count = len(self._history)
        self._history.clear()
        return count


# ── Singleton ────────────────────────────────────────────
_manager: Optional[ToolTimeoutManager] = None


def get_timeout_manager() -> ToolTimeoutManager:
    """Get the global tool timeout manager."""
    global _manager
    if _manager is None:
        _manager = ToolTimeoutManager()
    return _manager
