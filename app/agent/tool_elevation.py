"""
Tool Elevation — Elevated tools require user confirmation before execution.

Certain tools (exec, write_file, delete, etc.) can be marked as "elevated"
and require explicit user approval before the agent executes them.
Supports allow/deny lists and per-provider policies.

Usage:
    from app.agent.tool_elevation import get_elevation_manager

    mgr = get_elevation_manager()
    mgr.mark_elevated("exec", reason="Runs arbitrary shell commands")
    mgr.mark_elevated("delete_file", reason="Destructive operation")

    if mgr.requires_confirmation("exec"):
        # Prompt user for confirmation
        mgr.approve("exec", session_id="s1")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ElevationLevel(str, Enum):
    NORMAL = "normal"        # No confirmation needed
    ELEVATED = "elevated"    # Requires confirmation
    RESTRICTED = "restricted" # Always denied unless explicitly allowed
    BLOCKED = "blocked"      # Never allowed


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class ElevatedTool:
    """Configuration for an elevated tool."""
    tool_name: str
    level: ElevationLevel = ElevationLevel.ELEVATED
    reason: str = ""
    timeout_seconds: float = 120.0  # Approval expires after this
    auto_approve_count: int = 0     # Auto-approve after N approvals
    approved_count: int = 0
    denied_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "level": self.level.value,
            "reason": self.reason,
            "approved_count": self.approved_count,
            "denied_count": self.denied_count,
        }


@dataclass
class ApprovalRequest:
    """A pending approval request."""
    request_id: str
    tool_name: str
    session_id: str
    arguments_summary: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = 0.0
    resolved_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "session_id": self.session_id,
            "status": self.status.value,
            "arguments_summary": self.arguments_summary,
        }


class ToolElevationManager:
    """
    Manages tool elevation levels and approval workflow.

    Tools can be:
    - normal: execute without confirmation
    - elevated: require user confirmation
    - restricted: denied unless explicitly allowed
    - blocked: never allowed
    """

    # Default elevated tools
    DEFAULT_ELEVATED = {
        "exec": "Runs arbitrary shell commands",
        "apply_patch": "Modifies files via patch",
        "delete_file": "Deletes files permanently",
    }

    def __init__(self):
        self._tools: Dict[str, ElevatedTool] = {}
        self._requests: Dict[str, ApprovalRequest] = {}
        self._session_approvals: Dict[str, Set[str]] = {}  # session → approved tools
        self._counter: int = 0

        # Register defaults
        for tool, reason in self.DEFAULT_ELEVATED.items():
            self.mark_elevated(tool, reason=reason)

    def mark_elevated(
        self,
        tool_name: str,
        *,
        level: ElevationLevel = ElevationLevel.ELEVATED,
        reason: str = "",
        timeout_seconds: float = 120.0,
        auto_approve_count: int = 0,
    ) -> ElevatedTool:
        """Mark a tool as elevated."""
        tool = ElevatedTool(
            tool_name=tool_name,
            level=level,
            reason=reason,
            timeout_seconds=timeout_seconds,
            auto_approve_count=auto_approve_count,
        )
        self._tools[tool_name] = tool
        logger.info(f"[ELEVATION] Marked {tool_name} as {level.value}: {reason}")
        return tool

    def mark_normal(self, tool_name: str) -> bool:
        """Remove elevation from a tool."""
        return self._tools.pop(tool_name, None) is not None

    def requires_confirmation(self, tool_name: str, session_id: str = "") -> bool:
        """Check if a tool requires confirmation."""
        tool = self._tools.get(tool_name)
        if not tool:
            return False

        if tool.level == ElevationLevel.NORMAL:
            return False

        if tool.level == ElevationLevel.BLOCKED:
            return True  # Always blocked

        # Check if session has standing approval
        if session_id and tool_name in self._session_approvals.get(session_id, set()):
            return False

        # Check auto-approve threshold
        if tool.auto_approve_count > 0 and tool.approved_count >= tool.auto_approve_count:
            return False

        return True

    def is_blocked(self, tool_name: str) -> bool:
        """Check if a tool is completely blocked."""
        tool = self._tools.get(tool_name)
        return tool is not None and tool.level == ElevationLevel.BLOCKED

    def request_approval(
        self,
        tool_name: str,
        session_id: str,
        *,
        arguments_summary: str = "",
    ) -> ApprovalRequest:
        """Create an approval request for an elevated tool."""
        self._counter += 1
        req_id = f"approval_{self._counter}"

        request = ApprovalRequest(
            request_id=req_id,
            tool_name=tool_name,
            session_id=session_id,
            arguments_summary=arguments_summary,
        )
        self._requests[req_id] = request
        return request

    def approve(self, request_id: str, *, grant_session: bool = True) -> bool:
        """Approve a pending request."""
        req = self._requests.get(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False

        req.status = ApprovalStatus.APPROVED
        req.resolved_at = time.time()

        tool = self._tools.get(req.tool_name)
        if tool:
            tool.approved_count += 1

        # Grant standing approval for this session
        if grant_session:
            if req.session_id not in self._session_approvals:
                self._session_approvals[req.session_id] = set()
            self._session_approvals[req.session_id].add(req.tool_name)

        return True

    def deny(self, request_id: str) -> bool:
        """Deny a pending request."""
        req = self._requests.get(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False

        req.status = ApprovalStatus.DENIED
        req.resolved_at = time.time()

        tool = self._tools.get(req.tool_name)
        if tool:
            tool.denied_count += 1

        return True

    def revoke_session_approval(self, session_id: str, tool_name: Optional[str] = None) -> bool:
        """Revoke standing approval for a session."""
        if session_id not in self._session_approvals:
            return False
        if tool_name:
            self._session_approvals[session_id].discard(tool_name)
        else:
            del self._session_approvals[session_id]
        return True

    def get_elevation_level(self, tool_name: str) -> ElevationLevel:
        """Get the elevation level of a tool."""
        tool = self._tools.get(tool_name)
        return tool.level if tool else ElevationLevel.NORMAL

    def list_elevated(self) -> List[Dict[str, Any]]:
        """List all elevated tools."""
        return [t.to_dict() for t in self._tools.values()]

    def list_pending(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List pending approval requests."""
        reqs = [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]
        if session_id:
            reqs = [r for r in reqs if r.session_id == session_id]
        return [r.to_dict() for r in reqs]

    def stats(self) -> Dict[str, Any]:
        """Get elevation statistics."""
        total_requests = len(self._requests)
        approved = sum(1 for r in self._requests.values() if r.status == ApprovalStatus.APPROVED)
        denied = sum(1 for r in self._requests.values() if r.status == ApprovalStatus.DENIED)
        pending = sum(1 for r in self._requests.values() if r.status == ApprovalStatus.PENDING)

        return {
            "elevated_tools": len(self._tools),
            "total_requests": total_requests,
            "approved": approved,
            "denied": denied,
            "pending": pending,
            "sessions_with_approvals": len(self._session_approvals),
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[ToolElevationManager] = None


def get_elevation_manager() -> ToolElevationManager:
    """Get the global tool elevation manager."""
    global _manager
    if _manager is None:
        _manager = ToolElevationManager()
    return _manager
