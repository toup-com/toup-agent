"""
Activation Prompt — Boot message system for the agent.

When a session starts, the activation prompt is sent as the first
message to the agent, before any user input. This can be used to
set context, run initialization tasks, or establish the agent's
persona for the session.

Usage:
    from app.agent.activation import get_activation_manager

    mgr = get_activation_manager()
    mgr.set_prompt("agent-1", "You are a helpful coding assistant. Check for updates.")
    prompt = mgr.get_prompt("agent-1")
    mgr.set_global_prompt("System initialized. Ready for commands.")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActivationConfig:
    """Activation prompt configuration for an agent."""
    agent_id: str
    prompt: str
    enabled: bool = True
    run_on_session_start: bool = True
    run_on_reconnect: bool = False
    last_triggered: Optional[float] = None
    trigger_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "prompt": self.prompt[:200] + ("..." if len(self.prompt) > 200 else ""),
            "enabled": self.enabled,
            "run_on_session_start": self.run_on_session_start,
            "run_on_reconnect": self.run_on_reconnect,
            "trigger_count": self.trigger_count,
        }


class ActivationManager:
    """
    Manages activation prompts for agents and sessions.

    Activation prompts are sent to the agent at the start of a session
    (or on reconnect) to initialize context.
    """

    def __init__(self):
        self._configs: Dict[str, ActivationConfig] = {}
        self._global_prompt: Optional[str] = None
        self._history: List[Dict[str, Any]] = []

    def set_prompt(
        self,
        agent_id: str,
        prompt: str,
        *,
        enabled: bool = True,
        run_on_session_start: bool = True,
        run_on_reconnect: bool = False,
    ) -> ActivationConfig:
        """Set the activation prompt for an agent."""
        config = ActivationConfig(
            agent_id=agent_id,
            prompt=prompt,
            enabled=enabled,
            run_on_session_start=run_on_session_start,
            run_on_reconnect=run_on_reconnect,
        )
        self._configs[agent_id] = config
        logger.info(f"[ACTIVATION] Set prompt for {agent_id} ({len(prompt)} chars)")
        return config

    def get_prompt(self, agent_id: str) -> Optional[str]:
        """Get the activation prompt for an agent."""
        config = self._configs.get(agent_id)
        if config and config.enabled:
            return config.prompt
        return self._global_prompt

    def get_config(self, agent_id: str) -> Optional[ActivationConfig]:
        """Get the full activation config."""
        return self._configs.get(agent_id)

    def remove_prompt(self, agent_id: str) -> bool:
        """Remove an agent's activation prompt."""
        return self._configs.pop(agent_id, None) is not None

    def set_global_prompt(self, prompt: str) -> None:
        """Set a global activation prompt (fallback for all agents)."""
        self._global_prompt = prompt
        logger.info(f"[ACTIVATION] Set global prompt ({len(prompt)} chars)")

    def get_global_prompt(self) -> Optional[str]:
        """Get the global activation prompt."""
        return self._global_prompt

    def clear_global_prompt(self) -> None:
        """Clear the global activation prompt."""
        self._global_prompt = None

    def trigger(self, agent_id: str, *, event: str = "session_start") -> Optional[str]:
        """
        Trigger activation and return the prompt if applicable.

        Args:
            agent_id: The agent to trigger for.
            event: The triggering event (session_start, reconnect).

        Returns:
            The prompt string if activation should run, None otherwise.
        """
        config = self._configs.get(agent_id)

        if config:
            if not config.enabled:
                return None
            if event == "session_start" and not config.run_on_session_start:
                return None
            if event == "reconnect" and not config.run_on_reconnect:
                return None

            config.last_triggered = time.time()
            config.trigger_count += 1

            self._history.append({
                "agent_id": agent_id,
                "event": event,
                "timestamp": config.last_triggered,
                "prompt_length": len(config.prompt),
            })

            return config.prompt

        # Fall back to global
        if self._global_prompt:
            self._history.append({
                "agent_id": agent_id,
                "event": event,
                "timestamp": time.time(),
                "prompt_length": len(self._global_prompt),
                "global": True,
            })
            return self._global_prompt

        return None

    def disable(self, agent_id: str) -> bool:
        """Disable activation for an agent."""
        config = self._configs.get(agent_id)
        if config:
            config.enabled = False
            return True
        return False

    def enable(self, agent_id: str) -> bool:
        """Enable activation for an agent."""
        config = self._configs.get(agent_id)
        if config:
            config.enabled = True
            return True
        return False

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all activation configs."""
        return [c.to_dict() for c in self._configs.values()]

    def get_history(self, agent_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trigger history."""
        history = self._history
        if agent_id:
            history = [h for h in history if h["agent_id"] == agent_id]
        return history[-limit:]

    def stats(self) -> Dict[str, Any]:
        """Get activation statistics."""
        total_triggers = sum(c.trigger_count for c in self._configs.values())
        enabled = sum(1 for c in self._configs.values() if c.enabled)
        return {
            "total_configs": len(self._configs),
            "enabled": enabled,
            "disabled": len(self._configs) - enabled,
            "total_triggers": total_triggers,
            "has_global": self._global_prompt is not None,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[ActivationManager] = None


def get_activation_manager() -> ActivationManager:
    """Get the global activation manager."""
    global _manager
    if _manager is None:
        _manager = ActivationManager()
    return _manager
