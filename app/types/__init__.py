"""Typed cross-layer contracts for Toup.

Contracts here are intentionally separate from:
  - SQLAlchemy models (`app/db/models/`) — DB schema layer
  - Inline API pydantic models (`app/api/*`) — request/response shapes

A contract here represents a structured payload that crosses a service
boundary (platform → bridge, platform → agent env file, CI → platform
webhook, etc.) and whose shape needs a canonical definition both sides
can import.
"""

from .agent_env import AgentEnvContract

__all__ = ["AgentEnvContract"]
