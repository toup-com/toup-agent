"""
Channel Bindings — Route channels/groups to specific agents.

Maps channel identifiers (Telegram group IDs, Discord guild IDs,
Slack workspace IDs) to specific agent instances. Allows multiple
agents to serve different groups through the same gateway.

Usage:
    from app.agent.channel_bindings import get_binding_manager

    mgr = get_binding_manager()
    mgr.bind("telegram", group_id="-100123456", agent_id="agent-coder")
    mgr.bind("discord", group_id="guild_789", agent_id="agent-support")

    agent = mgr.resolve("telegram", group_id="-100123456")  # → "agent-coder"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChannelBinding:
    """A binding between a channel/group and an agent."""
    channel_type: str
    group_id: str
    agent_id: str
    priority: int = 0
    enabled: bool = True
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def binding_key(self) -> str:
        return f"{self.channel_type}:{self.group_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_type": self.channel_type,
            "group_id": self.group_id,
            "agent_id": self.agent_id,
            "priority": self.priority,
            "enabled": self.enabled,
        }


class ChannelBindingManager:
    """
    Manages channel-to-agent bindings.

    When a message arrives on a channel, the binding manager
    determines which agent should handle it based on the
    channel type and group/channel ID.
    """

    DEFAULT_AGENT = "default"

    def __init__(self, default_agent: str = "default"):
        self._bindings: Dict[str, ChannelBinding] = {}
        self._default_agent = default_agent

    @property
    def default_agent(self) -> str:
        return self._default_agent

    @default_agent.setter
    def default_agent(self, value: str):
        self._default_agent = value

    def bind(
        self,
        channel_type: str,
        group_id: str,
        agent_id: str,
        *,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChannelBinding:
        """
        Bind a channel/group to a specific agent.

        Args:
            channel_type: Channel type (telegram, discord, slack, etc.)
            group_id: The group/channel/guild identifier.
            agent_id: The agent to route messages to.
            priority: Higher priority overrides lower.
        """
        binding = ChannelBinding(
            channel_type=channel_type,
            group_id=group_id,
            agent_id=agent_id,
            priority=priority,
            metadata=metadata or {},
        )
        self._bindings[binding.binding_key] = binding
        logger.info(f"[BINDINGS] Bound {channel_type}:{group_id} → {agent_id}")
        return binding

    def unbind(self, channel_type: str, group_id: str) -> bool:
        """Remove a binding."""
        key = f"{channel_type}:{group_id}"
        return self._bindings.pop(key, None) is not None

    def resolve(
        self,
        channel_type: str,
        group_id: str,
    ) -> str:
        """
        Resolve which agent should handle a message.

        Returns the bound agent_id or the default agent.
        """
        key = f"{channel_type}:{group_id}"
        binding = self._bindings.get(key)

        if binding and binding.enabled:
            return binding.agent_id

        # Check for channel-wide binding (wildcard group)
        wildcard_key = f"{channel_type}:*"
        wildcard = self._bindings.get(wildcard_key)
        if wildcard and wildcard.enabled:
            return wildcard.agent_id

        return self._default_agent

    def get_binding(self, channel_type: str, group_id: str) -> Optional[ChannelBinding]:
        """Get a specific binding."""
        key = f"{channel_type}:{group_id}"
        return self._bindings.get(key)

    def list_bindings(
        self,
        channel_type: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List bindings with optional filters."""
        bindings = list(self._bindings.values())

        if channel_type:
            bindings = [b for b in bindings if b.channel_type == channel_type]
        if agent_id:
            bindings = [b for b in bindings if b.agent_id == agent_id]

        return [b.to_dict() for b in bindings]

    def list_agents(self) -> List[str]:
        """List all unique agent IDs that have bindings."""
        agents = set()
        for b in self._bindings.values():
            agents.add(b.agent_id)
        agents.add(self._default_agent)
        return sorted(agents)

    def get_groups_for_agent(self, agent_id: str) -> List[Dict[str, str]]:
        """Get all groups bound to a specific agent."""
        return [
            {"channel_type": b.channel_type, "group_id": b.group_id}
            for b in self._bindings.values()
            if b.agent_id == agent_id and b.enabled
        ]

    def disable_binding(self, channel_type: str, group_id: str) -> bool:
        """Disable a binding (keeps it but routes to default)."""
        key = f"{channel_type}:{group_id}"
        binding = self._bindings.get(key)
        if binding:
            binding.enabled = False
            return True
        return False

    def enable_binding(self, channel_type: str, group_id: str) -> bool:
        """Enable a disabled binding."""
        key = f"{channel_type}:{group_id}"
        binding = self._bindings.get(key)
        if binding:
            binding.enabled = True
            return True
        return False

    def bulk_bind(self, bindings: List[Dict[str, Any]]) -> int:
        """Bind multiple channel/groups at once."""
        count = 0
        for b in bindings:
            self.bind(
                channel_type=b["channel_type"],
                group_id=b["group_id"],
                agent_id=b["agent_id"],
                priority=b.get("priority", 0),
            )
            count += 1
        return count

    def stats(self) -> Dict[str, Any]:
        """Get binding statistics."""
        by_channel: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}
        enabled = 0
        for b in self._bindings.values():
            by_channel[b.channel_type] = by_channel.get(b.channel_type, 0) + 1
            by_agent[b.agent_id] = by_agent.get(b.agent_id, 0) + 1
            if b.enabled:
                enabled += 1

        return {
            "total_bindings": len(self._bindings),
            "enabled": enabled,
            "disabled": len(self._bindings) - enabled,
            "by_channel": by_channel,
            "by_agent": by_agent,
            "default_agent": self._default_agent,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[ChannelBindingManager] = None


def get_binding_manager() -> ChannelBindingManager:
    """Get the global channel binding manager."""
    global _manager
    if _manager is None:
        _manager = ChannelBindingManager()
    return _manager
