"""
Thinking Levels — Per-session thinking budget control.

Controls the reasoning effort/budget for the model per session:
  - low:    Minimal reasoning, fastest responses
  - medium: Balanced reasoning (default)
  - high:   Extended reasoning, slower but more thorough
  - xhigh:  Maximum reasoning with extended thinking (Claude)

Usage:
    from app.agent.thinking_levels import get_thinking_manager

    mgr = get_thinking_manager()
    mgr.set_level("session_1", ThinkingLevel.HIGH)
    level = mgr.get_level("session_1")
    params = mgr.get_model_params("session_1")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ThinkingLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


# Budget parameters per level
LEVEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "low": {
        "temperature": 0.3,
        "max_tokens": 1024,
        "thinking_budget": 0,
        "description": "Minimal reasoning — fast, concise responses",
    },
    "medium": {
        "temperature": 0.7,
        "max_tokens": 4096,
        "thinking_budget": 1024,
        "description": "Balanced reasoning — default mode",
    },
    "high": {
        "temperature": 0.8,
        "max_tokens": 8192,
        "thinking_budget": 4096,
        "description": "Extended reasoning — thorough analysis",
    },
    "xhigh": {
        "temperature": 1.0,
        "max_tokens": 16384,
        "thinking_budget": 16384,
        "description": "Maximum reasoning — Claude extended thinking",
    },
}


@dataclass
class ThinkingConfig:
    """Thinking level configuration for a session."""
    session_id: str
    level: ThinkingLevel = ThinkingLevel.MEDIUM
    custom_budget: Optional[int] = None
    changed_at: float = 0.0
    change_count: int = 0

    def __post_init__(self):
        if self.changed_at == 0.0:
            self.changed_at = time.time()

    @property
    def effective_budget(self) -> int:
        if self.custom_budget is not None:
            return self.custom_budget
        return LEVEL_PARAMS[self.level.value]["thinking_budget"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "level": self.level.value,
            "effective_budget": self.effective_budget,
            "change_count": self.change_count,
            "description": LEVEL_PARAMS[self.level.value]["description"],
        }


class ThinkingManager:
    """
    Manages per-session thinking level configuration.

    Controls the reasoning budget and model parameters based
    on the selected thinking level.
    """

    def __init__(self, default_level: ThinkingLevel = ThinkingLevel.MEDIUM):
        self._configs: Dict[str, ThinkingConfig] = {}
        self._default_level = default_level

    @property
    def default_level(self) -> ThinkingLevel:
        return self._default_level

    @default_level.setter
    def default_level(self, value: ThinkingLevel):
        self._default_level = value

    def set_level(
        self,
        session_id: str,
        level: ThinkingLevel,
        *,
        custom_budget: Optional[int] = None,
    ) -> ThinkingConfig:
        """Set thinking level for a session."""
        config = self._configs.get(session_id)
        if config:
            config.level = level
            config.custom_budget = custom_budget
            config.changed_at = time.time()
            config.change_count += 1
        else:
            config = ThinkingConfig(
                session_id=session_id,
                level=level,
                custom_budget=custom_budget,
            )
            self._configs[session_id] = config

        logger.info(f"[THINKING] Session {session_id}: {level.value}")
        return config

    def get_level(self, session_id: str) -> ThinkingLevel:
        """Get the thinking level for a session."""
        config = self._configs.get(session_id)
        if config:
            return config.level
        return self._default_level

    def get_config(self, session_id: str) -> Optional[ThinkingConfig]:
        """Get full thinking config for a session."""
        return self._configs.get(session_id)

    def get_model_params(self, session_id: str) -> Dict[str, Any]:
        """
        Get model parameters based on the session's thinking level.

        Returns temperature, max_tokens, thinking_budget suitable
        for passing to the LLM API.
        """
        level = self.get_level(session_id)
        params = dict(LEVEL_PARAMS[level.value])
        config = self._configs.get(session_id)
        if config and config.custom_budget is not None:
            params["thinking_budget"] = config.custom_budget
        return params

    def cycle_level(self, session_id: str) -> ThinkingLevel:
        """Cycle through thinking levels: low → medium → high → xhigh → low."""
        order = [ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH, ThinkingLevel.XHIGH]
        current = self.get_level(session_id)
        idx = order.index(current) if current in order else 0
        next_level = order[(idx + 1) % len(order)]
        self.set_level(session_id, next_level)
        return next_level

    def format_status(self, session_id: str) -> str:
        """Format thinking level as a human-readable status."""
        level = self.get_level(session_id)
        params = LEVEL_PARAMS[level.value]
        config = self._configs.get(session_id)
        budget = config.effective_budget if config else params["thinking_budget"]

        lines = [
            f"🧠 **Thinking: {level.value}**",
            f"  {params['description']}",
            f"  Temperature: {params['temperature']}",
            f"  Max tokens: {params['max_tokens']:,}",
            f"  Thinking budget: {budget:,}",
        ]
        return "\n".join(lines)

    def remove_config(self, session_id: str) -> bool:
        """Remove session config (revert to default)."""
        return self._configs.pop(session_id, None) is not None

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all session thinking configs."""
        return [c.to_dict() for c in self._configs.values()]

    def list_levels(self) -> List[Dict[str, Any]]:
        """List all available thinking levels with descriptions."""
        return [
            {"level": k, **v}
            for k, v in LEVEL_PARAMS.items()
        ]

    def stats(self) -> Dict[str, Any]:
        """Get thinking manager statistics."""
        by_level: Dict[str, int] = {}
        for c in self._configs.values():
            by_level[c.level.value] = by_level.get(c.level.value, 0) + 1

        return {
            "total_configs": len(self._configs),
            "default_level": self._default_level.value,
            "by_level": by_level,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[ThinkingManager] = None


def get_thinking_manager() -> ThinkingManager:
    """Get the global thinking manager."""
    global _manager
    if _manager is None:
        _manager = ThinkingManager()
    return _manager
