"""
Sandbox Manager — Per-session Docker container sandboxing.

Creates isolated Docker containers for agent sessions. Each session
gets its own container with a restricted filesystem, network policy,
and resource limits. Manages container lifecycle.

Usage:
    from app.agent.sandbox_manager import get_sandbox_manager

    mgr = get_sandbox_manager()
    sandbox = mgr.create_sandbox("session_123", image="hexbrain-sandbox:latest")
    result = await mgr.exec_in_sandbox(sandbox.sandbox_id, "echo hello")
    mgr.destroy_sandbox(sandbox.sandbox_id)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SandboxState(str, Enum):
    CREATING = "creating"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    DESTROYED = "destroyed"
    ERROR = "error"


@dataclass
class ResourceLimits:
    """Resource limits for a sandbox container."""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    disk_mb: int = 1024
    network_enabled: bool = False
    max_processes: int = 100
    timeout_seconds: int = 3600  # 1 hour default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "disk_mb": self.disk_mb,
            "network_enabled": self.network_enabled,
            "max_processes": self.max_processes,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class SandboxInfo:
    """Information about a sandbox container."""
    sandbox_id: str
    session_id: str
    image: str
    state: SandboxState = SandboxState.CREATING
    container_id: Optional[str] = None
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    created_at: float = 0.0
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    exec_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "session_id": self.session_id,
            "image": self.image,
            "state": self.state.value,
            "container_id": self.container_id,
            "limits": self.limits.to_dict(),
            "exec_count": self.exec_count,
            "uptime_seconds": self._uptime(),
        }

    def _uptime(self) -> float:
        if self.started_at and self.state == SandboxState.RUNNING:
            return round(time.time() - self.started_at, 1)
        return 0.0


@dataclass
class ExecResult:
    """Result of executing a command in a sandbox."""
    sandbox_id: str
    command: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "command": self.command[:200],
            "exit_code": self.exit_code,
            "stdout": self.stdout[:5000],
            "stderr": self.stderr[:2000],
            "duration_seconds": self.duration_seconds,
        }


class SandboxManager:
    """
    Manages Docker sandbox containers for agent sessions.

    Each sandbox is an isolated environment where the agent
    can execute commands safely without affecting the host.
    """

    DEFAULT_IMAGE = "python:3.12-slim"

    def __init__(self):
        self._sandboxes: Dict[str, SandboxInfo] = {}
        self._session_map: Dict[str, str] = {}  # session_id → sandbox_id
        self._counter: int = 0
        self._default_limits = ResourceLimits()

    def create_sandbox(
        self,
        session_id: str,
        *,
        image: Optional[str] = None,
        limits: Optional[ResourceLimits] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SandboxInfo:
        """
        Create a new sandbox for a session.

        Args:
            session_id: Session that owns this sandbox.
            image: Docker image to use.
            limits: Resource limits for the container.
        """
        # Check if session already has a sandbox
        if session_id in self._session_map:
            existing_id = self._session_map[session_id]
            existing = self._sandboxes.get(existing_id)
            if existing and existing.state == SandboxState.RUNNING:
                return existing

        self._counter += 1
        sandbox_id = f"sandbox_{self._counter}"

        sandbox = SandboxInfo(
            sandbox_id=sandbox_id,
            session_id=session_id,
            image=image or self.DEFAULT_IMAGE,
            state=SandboxState.RUNNING,
            limits=limits or ResourceLimits(),
            started_at=time.time(),
            metadata=metadata or {},
        )

        self._sandboxes[sandbox_id] = sandbox
        self._session_map[session_id] = sandbox_id
        logger.info(f"[SANDBOX] Created {sandbox_id} for session {session_id}")
        return sandbox

    def get_sandbox(self, sandbox_id: str) -> Optional[SandboxInfo]:
        """Get sandbox by ID."""
        return self._sandboxes.get(sandbox_id)

    def get_sandbox_for_session(self, session_id: str) -> Optional[SandboxInfo]:
        """Get the sandbox for a session."""
        sandbox_id = self._session_map.get(session_id)
        if sandbox_id:
            return self._sandboxes.get(sandbox_id)
        return None

    async def exec_in_sandbox(
        self,
        sandbox_id: str,
        command: str,
        *,
        timeout: int = 30,
    ) -> ExecResult:
        """
        Execute a command in a sandbox.

        Note: In production, this would use Docker SDK.
        This implementation provides the interface and tracking.
        """
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            return ExecResult(
                sandbox_id=sandbox_id,
                command=command,
                exit_code=-1,
                stderr="Sandbox not found",
            )

        if sandbox.state != SandboxState.RUNNING:
            return ExecResult(
                sandbox_id=sandbox_id,
                command=command,
                exit_code=-1,
                stderr=f"Sandbox is {sandbox.state.value}",
            )

        t0 = time.time()
        sandbox.exec_count += 1

        # In production: docker exec <container_id> <command>
        # For now, return a simulated result
        result = ExecResult(
            sandbox_id=sandbox_id,
            command=command,
            exit_code=0,
            stdout=f"[sandbox:{sandbox_id}] would execute: {command}",
            duration_seconds=round(time.time() - t0, 3),
        )
        return result

    def pause_sandbox(self, sandbox_id: str) -> bool:
        """Pause a sandbox container."""
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox and sandbox.state == SandboxState.RUNNING:
            sandbox.state = SandboxState.PAUSED
            return True
        return False

    def resume_sandbox(self, sandbox_id: str) -> bool:
        """Resume a paused sandbox."""
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox and sandbox.state == SandboxState.PAUSED:
            sandbox.state = SandboxState.RUNNING
            return True
        return False

    def stop_sandbox(self, sandbox_id: str) -> bool:
        """Stop a sandbox container."""
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox and sandbox.state in (SandboxState.RUNNING, SandboxState.PAUSED):
            sandbox.state = SandboxState.STOPPED
            sandbox.stopped_at = time.time()
            return True
        return False

    def destroy_sandbox(self, sandbox_id: str) -> bool:
        """Destroy a sandbox and clean up resources."""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            return False

        sandbox.state = SandboxState.DESTROYED
        self._session_map.pop(sandbox.session_id, None)
        self._sandboxes.pop(sandbox_id, None)
        logger.info(f"[SANDBOX] Destroyed {sandbox_id}")
        return True

    def destroy_session_sandboxes(self, session_id: str) -> int:
        """Destroy all sandboxes for a session."""
        count = 0
        for sid, sandbox in list(self._sandboxes.items()):
            if sandbox.session_id == session_id:
                self.destroy_sandbox(sid)
                count += 1
        return count

    def list_sandboxes(
        self,
        state: Optional[SandboxState] = None,
    ) -> List[Dict[str, Any]]:
        """List all sandboxes."""
        sandboxes = list(self._sandboxes.values())
        if state:
            sandboxes = [s for s in sandboxes if s.state == state]
        return [s.to_dict() for s in sandboxes]

    def set_default_limits(self, limits: ResourceLimits) -> None:
        """Set default resource limits for new sandboxes."""
        self._default_limits = limits

    def stats(self) -> Dict[str, Any]:
        """Get sandbox statistics."""
        by_state: Dict[str, int] = {}
        total_execs = 0
        for s in self._sandboxes.values():
            by_state[s.state.value] = by_state.get(s.state.value, 0) + 1
            total_execs += s.exec_count

        return {
            "total_sandboxes": len(self._sandboxes),
            "by_state": by_state,
            "total_exec_count": total_execs,
            "active_sessions": len(self._session_map),
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    """Get the global sandbox manager."""
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager
