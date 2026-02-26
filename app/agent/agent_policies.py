"""
Agent-to-Agent Policies — Controls which agents can spawn/communicate.

Defines allowAgents lists, spawn restrictions, and inter-agent
communication policies.

Usage:
    from app.agent.agent_policies import get_policy_manager

    mgr = get_policy_manager()
    mgr.set_policy("agent-1", allowed_agents=["agent-2", "agent-3"])
    mgr.can_spawn("agent-1", "agent-2")  # True
    mgr.can_spawn("agent-1", "agent-99")  # False
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PolicyMode(str, Enum):
    ALLOW_ALL = "allow_all"      # Any agent can spawn/communicate
    ALLOW_LIST = "allow_list"    # Only specified agents
    DENY_LIST = "deny_list"      # All except specified
    DENY_ALL = "deny_all"        # No inter-agent communication


class SpawnPermission(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class AgentPolicy:
    """Policy for an agent's inter-agent communication."""
    agent_id: str
    mode: PolicyMode = PolicyMode.ALLOW_ALL
    allowed_agents: Set[str] = field(default_factory=set)
    denied_agents: Set[str] = field(default_factory=set)
    max_concurrent_spawns: int = 5
    max_spawn_depth: int = 3
    can_be_spawned: bool = True
    require_approval_from: Set[str] = field(default_factory=set)
    spawn_count: int = 0
    last_spawn_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "mode": self.mode.value,
            "allowed_agents": sorted(self.allowed_agents) if self.allowed_agents else [],
            "denied_agents": sorted(self.denied_agents) if self.denied_agents else [],
            "max_concurrent_spawns": self.max_concurrent_spawns,
            "max_spawn_depth": self.max_spawn_depth,
            "can_be_spawned": self.can_be_spawned,
            "spawn_count": self.spawn_count,
        }


class AgentPolicyManager:
    """
    Manages agent-to-agent communication policies.

    Controls spawn permissions, communication restrictions,
    and depth limits for multi-agent systems.
    """

    def __init__(self):
        self._policies: Dict[str, AgentPolicy] = {}
        self._active_spawns: Dict[str, Set[str]] = {}  # spawner -> set of spawned

    def set_policy(
        self,
        agent_id: str,
        *,
        mode: PolicyMode = PolicyMode.ALLOW_ALL,
        allowed_agents: Optional[List[str]] = None,
        denied_agents: Optional[List[str]] = None,
        max_concurrent: int = 5,
        max_depth: int = 3,
        can_be_spawned: bool = True,
    ) -> AgentPolicy:
        """Set the policy for an agent."""
        policy = AgentPolicy(
            agent_id=agent_id,
            mode=mode,
            allowed_agents=set(allowed_agents) if allowed_agents else set(),
            denied_agents=set(denied_agents) if denied_agents else set(),
            max_concurrent_spawns=max_concurrent,
            max_spawn_depth=max_depth,
            can_be_spawned=can_be_spawned,
        )
        self._policies[agent_id] = policy
        logger.info(f"[POLICY] Set policy for {agent_id}: {mode.value}")
        return policy

    def get_policy(self, agent_id: str) -> AgentPolicy:
        """Get policy for an agent, returning default if not set."""
        if agent_id in self._policies:
            return self._policies[agent_id]
        # Default: allow all
        return AgentPolicy(agent_id=agent_id)

    def can_spawn(self, spawner_id: str, target_id: str) -> SpawnPermission:
        """Check if spawner can spawn target agent."""
        policy = self.get_policy(spawner_id)
        target_policy = self.get_policy(target_id)

        # Check if target allows being spawned
        if not target_policy.can_be_spawned:
            return SpawnPermission.DENIED

        # Check concurrent spawn limit
        active = self._active_spawns.get(spawner_id, set())
        if len(active) >= policy.max_concurrent_spawns:
            return SpawnPermission.DENIED

        # Check approval requirement
        if target_id in policy.require_approval_from:
            return SpawnPermission.REQUIRES_APPROVAL

        # Check mode
        if policy.mode == PolicyMode.ALLOW_ALL:
            return SpawnPermission.ALLOWED
        elif policy.mode == PolicyMode.DENY_ALL:
            return SpawnPermission.DENIED
        elif policy.mode == PolicyMode.ALLOW_LIST:
            if target_id in policy.allowed_agents:
                return SpawnPermission.ALLOWED
            return SpawnPermission.DENIED
        elif policy.mode == PolicyMode.DENY_LIST:
            if target_id in policy.denied_agents:
                return SpawnPermission.DENIED
            return SpawnPermission.ALLOWED

        return SpawnPermission.DENIED

    def record_spawn(self, spawner_id: str, target_id: str) -> None:
        """Record a spawn event."""
        if spawner_id not in self._active_spawns:
            self._active_spawns[spawner_id] = set()
        self._active_spawns[spawner_id].add(target_id)

        policy = self.get_policy(spawner_id)
        policy.spawn_count += 1
        policy.last_spawn_at = time.time()

    def record_despawn(self, spawner_id: str, target_id: str) -> None:
        """Record when a spawned agent completes."""
        active = self._active_spawns.get(spawner_id, set())
        active.discard(target_id)

    def add_allowed(self, agent_id: str, target_id: str) -> None:
        """Add an agent to the allow list."""
        policy = self._policies.setdefault(agent_id, AgentPolicy(agent_id=agent_id))
        policy.allowed_agents.add(target_id)
        if policy.mode == PolicyMode.ALLOW_ALL:
            policy.mode = PolicyMode.ALLOW_LIST

    def remove_allowed(self, agent_id: str, target_id: str) -> None:
        """Remove an agent from the allow list."""
        policy = self._policies.get(agent_id)
        if policy:
            policy.allowed_agents.discard(target_id)

    def add_denied(self, agent_id: str, target_id: str) -> None:
        """Add an agent to the deny list."""
        policy = self._policies.setdefault(agent_id, AgentPolicy(agent_id=agent_id))
        policy.denied_agents.add(target_id)

    def get_active_spawns(self, agent_id: str) -> List[str]:
        """Get list of agents currently spawned by this agent."""
        return sorted(self._active_spawns.get(agent_id, set()))

    def list_policies(self) -> List[Dict[str, Any]]:
        """List all policies."""
        return [p.to_dict() for p in self._policies.values()]

    def stats(self) -> Dict[str, Any]:
        total_active = sum(len(s) for s in self._active_spawns.values())
        return {
            "total_policies": len(self._policies),
            "total_active_spawns": total_active,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[AgentPolicyManager] = None


def get_policy_manager() -> AgentPolicyManager:
    """Get the global agent policy manager."""
    global _manager
    if _manager is None:
        _manager = AgentPolicyManager()
    return _manager
