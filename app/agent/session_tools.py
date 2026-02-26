"""
Session Management Tools — session_status, sessions_send, agents_list
Extended session tools for agent-to-agent communication and status tracking.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SessionKind(str, Enum):
    """Types of sessions."""
    MAIN = "main"
    SUBAGENT = "subagent"
    CRON = "cron"
    WEBHOOK = "webhook"
    SYSTEM = "system"


@dataclass
class SessionStatus:
    """Status of a session."""
    session_id: str
    kind: SessionKind
    model: str
    thinking_level: str = "off"
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    messages: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    channel: str = "unknown"
    user_id: str = ""
    agent_id: str = "default"
    is_active: bool = True

    def uptime(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "kind": self.kind.value,
            "model": self.model,
            "thinking_level": self.thinking_level,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "messages": self.messages,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "uptime": self.uptime(),
            "channel": self.channel,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "is_active": self.is_active,
        }


@dataclass
class AgentInfo:
    """Information about an available agent."""
    agent_id: str
    name: str
    model: str = "claude-opus-4-6"
    description: str = ""
    tools: List[str] = field(default_factory=list)
    max_sessions: int = 10
    active_sessions: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "model": self.model,
            "description": self.description,
            "tools": self.tools,
            "max_sessions": self.max_sessions,
            "active_sessions": self.active_sessions,
        }


class SessionManager:
    """Extended session manager with cross-session communication."""

    def __init__(self):
        self._sessions: Dict[str, SessionStatus] = {}
        self._agents: Dict[str, AgentInfo] = {}
        self._message_log: List[dict] = []
        self._register_default_agent()

    def _register_default_agent(self):
        self._agents["default"] = AgentInfo(
            agent_id="default",
            name="HexBrain",
            model="claude-opus-4-6",
            description="Primary HexBrain agent",
            tools=["read_file", "write_file", "edit_file", "exec", "web_search", "web_fetch"],
        )

    def create_session(
        self,
        kind: SessionKind = SessionKind.MAIN,
        model: str = "claude-opus-4-6",
        channel: str = "unknown",
        user_id: str = "",
        agent_id: str = "default",
    ) -> SessionStatus:
        """Create a new session."""
        session = SessionStatus(
            session_id=str(uuid.uuid4()),
            kind=kind,
            model=model,
            channel=channel,
            user_id=user_id,
            agent_id=agent_id,
        )
        self._sessions[session.session_id] = session
        agent = self._agents.get(agent_id)
        if agent:
            agent.active_sessions += 1
        return session

    def get_session(self, session_id: str) -> Optional[SessionStatus]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False
            agent = self._agents.get(session.agent_id)
            if agent and agent.active_sessions > 0:
                agent.active_sessions -= 1
            return True
        return False

    def list_sessions(
        self,
        kind: Optional[SessionKind] = None,
        active_only: bool = True,
        agent_id: Optional[str] = None,
    ) -> List[SessionStatus]:
        """List sessions with filters."""
        sessions = list(self._sessions.values())
        if kind:
            sessions = [s for s in sessions if s.kind == kind]
        if active_only:
            sessions = [s for s in sessions if s.is_active]
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        return sorted(sessions, key=lambda s: s.last_activity, reverse=True)

    def session_status(self, session_id: str) -> Optional[dict]:
        """Get detailed session status."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.to_dict()

    async def sessions_send(
        self,
        from_session: str,
        to_session: str,
        message: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Send a message from one session to another."""
        sender = self._sessions.get(from_session)
        receiver = self._sessions.get(to_session)
        if not sender:
            return {"error": f"Sender session {from_session} not found"}
        if not receiver:
            return {"error": f"Receiver session {to_session} not found"}
        if not receiver.is_active:
            return {"error": f"Receiver session {to_session} is not active"}

        msg_record = {
            "id": str(uuid.uuid4()),
            "from_session": from_session,
            "to_session": to_session,
            "message": message,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self._message_log.append(msg_record)
        receiver.messages += 1
        receiver.last_activity = time.time()
        return {"sent": True, "message_id": msg_record["id"]}

    def register_agent(self, agent: AgentInfo) -> None:
        """Register an agent."""
        self._agents[agent.agent_id] = agent

    def agents_list(self) -> List[AgentInfo]:
        """List all available agents."""
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        return self._agents.get(agent_id)

    def track_usage(
        self, session_id: str, tokens_in: int = 0, tokens_out: int = 0, cost: float = 0.0
    ) -> None:
        """Track token usage for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.tokens_in += tokens_in
            session.tokens_out += tokens_out
            session.cost += cost
            session.last_activity = time.time()

    def get_message_log(self, session_id: Optional[str] = None, limit: int = 50) -> List[dict]:
        """Get cross-session message log."""
        msgs = self._message_log
        if session_id:
            msgs = [m for m in msgs if m["from_session"] == session_id or m["to_session"] == session_id]
        return msgs[-limit:]


_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager singleton."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
