"""
Delivery Modes — Per-channel message delivery configuration.

Controls how the agent delivers responses per channel:
- gateway: Route through central gateway (default)
- direct: Send directly to channel API
- announce: Broadcast-only (no response expected)
- none: Channel disabled for delivery

Usage:
    from app.agent.delivery_modes import get_delivery_manager

    mgr = get_delivery_manager()
    mgr.set_mode("telegram", "main_bot", DeliveryMode.GATEWAY)
    mgr.set_mode("discord", "server_1", DeliveryMode.DIRECT)
    mode = mgr.get_mode("telegram", "main_bot")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeliveryMode(str, Enum):
    GATEWAY = "gateway"      # Route through central gateway
    DIRECT = "direct"        # Send directly to channel API
    ANNOUNCE = "announce"    # Broadcast-only, no responses
    NONE = "none"            # Delivery disabled


@dataclass
class DeliveryConfig:
    """Delivery configuration for a channel context."""
    platform: str
    context_id: str  # channel_id, group_id, or "default"
    mode: DeliveryMode = DeliveryMode.GATEWAY
    priority: int = 0          # Higher = preferred route
    rate_limit: int = 0        # Messages per minute (0 = unlimited)
    batch_delay_ms: int = 0    # Delay for batching messages
    fallback_mode: Optional[DeliveryMode] = None

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.context_id}"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "platform": self.platform,
            "context_id": self.context_id,
            "mode": self.mode.value,
            "priority": self.priority,
        }
        if self.rate_limit:
            d["rate_limit"] = self.rate_limit
        if self.fallback_mode:
            d["fallback_mode"] = self.fallback_mode.value
        return d


class DeliveryManager:
    """
    Manages per-channel delivery modes.

    Controls how agent responses are routed to each channel context
    (group, DM, thread, etc.).
    """

    def __init__(self):
        self._configs: Dict[str, DeliveryConfig] = {}
        self._defaults: Dict[str, DeliveryMode] = {}  # platform -> default mode

    def set_mode(
        self,
        platform: str,
        context_id: str,
        mode: DeliveryMode,
        *,
        priority: int = 0,
        rate_limit: int = 0,
        fallback: Optional[DeliveryMode] = None,
    ) -> DeliveryConfig:
        """Set the delivery mode for a channel context."""
        config = DeliveryConfig(
            platform=platform,
            context_id=context_id,
            mode=mode,
            priority=priority,
            rate_limit=rate_limit,
            fallback_mode=fallback,
        )
        self._configs[config.key] = config
        logger.info(f"[DELIVERY] {config.key} → {mode.value}")
        return config

    def get_mode(self, platform: str, context_id: str) -> DeliveryMode:
        """Get the delivery mode for a context, falling back to platform default."""
        key = f"{platform}:{context_id}"
        config = self._configs.get(key)
        if config:
            return config.mode

        # Check platform default
        return self._defaults.get(platform, DeliveryMode.GATEWAY)

    def get_config(self, platform: str, context_id: str) -> Optional[DeliveryConfig]:
        """Get full delivery config for a context."""
        return self._configs.get(f"{platform}:{context_id}")

    def set_platform_default(self, platform: str, mode: DeliveryMode) -> None:
        """Set the default delivery mode for a platform."""
        self._defaults[platform] = mode

    def remove_config(self, platform: str, context_id: str) -> bool:
        """Remove a delivery config, reverting to default."""
        return self._configs.pop(f"{platform}:{context_id}", None) is not None

    def should_deliver(self, platform: str, context_id: str) -> bool:
        """Check if delivery is enabled for a context."""
        mode = self.get_mode(platform, context_id)
        return mode != DeliveryMode.NONE

    def is_announce_only(self, platform: str, context_id: str) -> bool:
        """Check if context is announce-only (no responses)."""
        return self.get_mode(platform, context_id) == DeliveryMode.ANNOUNCE

    def list_configs(
        self,
        platform: Optional[str] = None,
        mode: Optional[DeliveryMode] = None,
    ) -> List[Dict[str, Any]]:
        """List all delivery configs."""
        configs = list(self._configs.values())
        if platform:
            configs = [c for c in configs if c.platform == platform]
        if mode:
            configs = [c for c in configs if c.mode == mode]
        return [c.to_dict() for c in configs]

    def stats(self) -> Dict[str, Any]:
        by_mode: Dict[str, int] = {}
        for c in self._configs.values():
            by_mode[c.mode.value] = by_mode.get(c.mode.value, 0) + 1

        return {
            "total_configs": len(self._configs),
            "by_mode": by_mode,
            "platform_defaults": {k: v.value for k, v in self._defaults.items()},
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[DeliveryManager] = None


def get_delivery_manager() -> DeliveryManager:
    """Get the global delivery manager."""
    global _manager
    if _manager is None:
        _manager = DeliveryManager()
    return _manager
