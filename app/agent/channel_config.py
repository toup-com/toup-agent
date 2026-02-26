"""
Per-Channel Configuration — nested config for each channel instance.

Each channel can have its own configuration for:
- Model override
- System prompt prefix
- Tool allow/deny lists
- Agent binding
- DM/group policies
- Delivery mode
- Auto-TTS settings

Usage:
    from app.agent.channel_config import get_channel_config_manager

    mgr = get_channel_config_manager()
    mgr.set("telegram", {"model": "gpt-4o", "dm_policy": "pairing"})
    cfg = mgr.get("telegram")
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default per-channel config template
DEFAULT_CHANNEL_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "model": None,             # Override model for this channel
    "agent": None,             # Bind to specific agent persona
    "system_prompt_prefix": "",
    "dm_policy": "open",       # open | pairing | allowlist | disabled
    "group_policy": "open",    # open | allowlist | disabled
    "dm_allowlist": [],
    "group_allowlist": [],
    "tool_allow": None,        # If set, ONLY these tools
    "tool_deny": None,         # If set, block these tools
    "delivery_mode": "gateway",  # gateway | direct | announce | none
    "auto_tts": "off",        # off | always | inbound | tagged
    "tts_voice": None,
    "max_message_length": 4096,
    "rate_limit_per_minute": 30,
    "group_require_mention": True,
    "reactions_enabled": True,
    "thread_support": True,
    "inline_buttons": True,
    "media_enabled": True,
}


class ChannelConfigManager:
    """
    Manages per-channel configuration overrides.

    Each channel gets a copy of DEFAULT_CHANNEL_CONFIG that can be
    customized. Unknown keys are stored but flagged with a warning.
    """

    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._listeners: List[Any] = []

    def get(self, channel: str) -> Dict[str, Any]:
        """Get config for a channel (returns defaults if not configured)."""
        if channel not in self._configs:
            self._configs[channel] = copy.deepcopy(DEFAULT_CHANNEL_CONFIG)
        return self._configs[channel]

    def set(self, channel: str, overrides: Dict[str, Any]) -> Dict[str, str]:
        """
        Set config values for a channel.

        Args:
            channel: Channel name (telegram, discord, etc.)
            overrides: Dict of config key → value

        Returns:
            Dict of key → status ("updated" | "created" | "warning:unknown_key")
        """
        if channel not in self._configs:
            self._configs[channel] = copy.deepcopy(DEFAULT_CHANNEL_CONFIG)

        results = {}
        for key, value in overrides.items():
            if key in DEFAULT_CHANNEL_CONFIG:
                old = self._configs[channel].get(key)
                self._configs[channel][key] = value
                status = "updated" if old != value else "unchanged"
                results[key] = status
                if old != value:
                    logger.info(f"[CHANNEL-CONFIG] {channel}.{key}: {old!r} → {value!r}")
            else:
                self._configs[channel][key] = value
                results[key] = "warning:unknown_key"
                logger.warning(f"[CHANNEL-CONFIG] Unknown key {channel}.{key} = {value!r}")

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(channel, overrides)
            except Exception:
                pass

        return results

    def reset(self, channel: str) -> bool:
        """Reset a channel's config to defaults."""
        if channel in self._configs:
            self._configs[channel] = copy.deepcopy(DEFAULT_CHANNEL_CONFIG)
            return True
        return False

    def remove(self, channel: str) -> bool:
        """Remove a channel's config entirely."""
        return self._configs.pop(channel, None) is not None

    def get_value(self, channel: str, key: str, default: Any = None) -> Any:
        """Get a single config value for a channel."""
        cfg = self.get(channel)
        return cfg.get(key, default)

    def list_channels(self) -> List[str]:
        """List all channels with custom configuration."""
        return sorted(self._configs.keys())

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all channel configs."""
        return {ch: dict(cfg) for ch, cfg in self._configs.items()}

    def get_channels_by_agent(self, agent_name: str) -> List[str]:
        """Find all channels bound to a specific agent."""
        return [
            ch for ch, cfg in self._configs.items()
            if cfg.get("agent") == agent_name
        ]

    def get_channels_by_model(self, model: str) -> List[str]:
        """Find all channels using a specific model override."""
        return [
            ch for ch, cfg in self._configs.items()
            if cfg.get("model") == model
        ]

    def on_change(self, callback) -> None:
        """Register a config change listener."""
        self._listeners.append(callback)

    def diff_from_defaults(self, channel: str) -> Dict[str, Any]:
        """Show only the config values that differ from defaults."""
        cfg = self.get(channel)
        return {
            k: v for k, v in cfg.items()
            if k in DEFAULT_CHANNEL_CONFIG and v != DEFAULT_CHANNEL_CONFIG[k]
        }

    def validate(self, channel: str) -> List[str]:
        """Validate a channel's config and return any issues."""
        issues = []
        cfg = self.get(channel)

        dm_policy = cfg.get("dm_policy", "open")
        if dm_policy not in ("open", "pairing", "allowlist", "disabled"):
            issues.append(f"Invalid dm_policy: {dm_policy}")

        group_policy = cfg.get("group_policy", "open")
        if group_policy not in ("open", "allowlist", "disabled"):
            issues.append(f"Invalid group_policy: {group_policy}")

        delivery = cfg.get("delivery_mode", "gateway")
        if delivery not in ("gateway", "direct", "announce", "none"):
            issues.append(f"Invalid delivery_mode: {delivery}")

        auto_tts = cfg.get("auto_tts", "off")
        if auto_tts not in ("off", "always", "inbound", "tagged"):
            issues.append(f"Invalid auto_tts: {auto_tts}")

        rate = cfg.get("rate_limit_per_minute", 30)
        if not isinstance(rate, (int, float)) or rate < 0:
            issues.append(f"Invalid rate_limit_per_minute: {rate}")

        return issues


# ── Singleton ────────────────────────────────────────────
_manager: Optional[ChannelConfigManager] = None


def get_channel_config_manager() -> ChannelConfigManager:
    """Get the global channel config manager."""
    global _manager
    if _manager is None:
        _manager = ChannelConfigManager()
    return _manager
