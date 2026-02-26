"""
Agent Workspace — Per-agent isolated filesystem.

Each agent gets its own workspace directory with isolated:
- Files (AGENT.md, TOOLS.md, skills, workspace files)
- Configuration
- Session data

Prevents agents from interfering with each other's state.

Usage:
    from app.agent.agent_workspace import get_workspace_manager

    mgr = get_workspace_manager()
    ws = mgr.create_workspace("agent-1")
    path = ws.get_file_path("notes.md")
    ws.write_file("notes.md", "Hello!")
"""

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceInfo:
    """Information about an agent workspace."""
    agent_id: str
    root_path: str
    created_at: float = 0.0
    last_accessed: float = 0.0
    file_count: int = 0
    total_size_bytes: int = 0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "root_path": self.root_path,
            "created_at": self.created_at,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
        }


class AgentWorkspace:
    """
    Represents an individual agent's workspace.

    All file operations are sandboxed to the workspace root.
    """

    def __init__(self, agent_id: str, root_path: str):
        self.agent_id = agent_id
        self.root_path = root_path
        self._created_at = time.time()
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create the workspace directory structure."""
        dirs = [
            self.root_path,
            os.path.join(self.root_path, "files"),
            os.path.join(self.root_path, "skills"),
            os.path.join(self.root_path, "config"),
            os.path.join(self.root_path, "sessions"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _resolve(self, relative_path: str) -> str:
        """Resolve a relative path within the workspace. Prevents escape."""
        resolved = os.path.normpath(os.path.join(self.root_path, "files", relative_path))
        if not resolved.startswith(os.path.normpath(self.root_path)):
            raise ValueError(f"Path escapes workspace: {relative_path}")
        return resolved

    def write_file(self, relative_path: str, content: str) -> str:
        """Write a file in the workspace."""
        path = self._resolve(relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"[WORKSPACE] {self.agent_id}: wrote {relative_path}")
        return path

    def read_file(self, relative_path: str) -> str:
        """Read a file from the workspace."""
        path = self._resolve(relative_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{relative_path} not found in workspace {self.agent_id}")
        with open(path) as f:
            return f.read()

    def delete_file(self, relative_path: str) -> bool:
        """Delete a file from the workspace."""
        path = self._resolve(relative_path)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def file_exists(self, relative_path: str) -> bool:
        """Check if a file exists."""
        path = self._resolve(relative_path)
        return os.path.exists(path)

    def list_files(self, subdirectory: str = "") -> List[str]:
        """List files in the workspace."""
        base = self._resolve(subdirectory) if subdirectory else os.path.join(self.root_path, "files")
        if not os.path.isdir(base):
            return []

        result = []
        for root, _, files in os.walk(base):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, os.path.join(self.root_path, "files"))
                result.append(rel)
        return sorted(result)

    def get_file_path(self, relative_path: str) -> str:
        """Get the absolute path for a workspace file."""
        return self._resolve(relative_path)

    def get_size(self) -> int:
        """Calculate total workspace size in bytes."""
        total = 0
        for root, _, files in os.walk(self.root_path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total

    def get_info(self) -> WorkspaceInfo:
        """Get workspace information."""
        files = self.list_files()
        return WorkspaceInfo(
            agent_id=self.agent_id,
            root_path=self.root_path,
            created_at=self._created_at,
            last_accessed=time.time(),
            file_count=len(files),
            total_size_bytes=self.get_size(),
        )

    def destroy(self) -> bool:
        """Destroy the entire workspace."""
        if os.path.exists(self.root_path):
            shutil.rmtree(self.root_path)
            return True
        return False


class WorkspaceManager:
    """
    Manages workspaces for multiple agents.

    Each agent gets an isolated directory. Supports creation,
    listing, cleanup, and quota enforcement.
    """

    DEFAULT_BASE = "/tmp/hexbrain_workspaces"

    def __init__(self, base_path: Optional[str] = None):
        self._base_path = base_path or self.DEFAULT_BASE
        self._workspaces: Dict[str, AgentWorkspace] = {}
        self._quota_bytes: int = 100 * 1024 * 1024  # 100MB per workspace
        os.makedirs(self._base_path, exist_ok=True)

    @property
    def base_path(self) -> str:
        return self._base_path

    def create_workspace(self, agent_id: str) -> AgentWorkspace:
        """Create or get a workspace for an agent."""
        if agent_id in self._workspaces:
            return self._workspaces[agent_id]

        root = os.path.join(self._base_path, agent_id)
        ws = AgentWorkspace(agent_id=agent_id, root_path=root)
        self._workspaces[agent_id] = ws
        logger.info(f"[WORKSPACE] Created workspace for {agent_id}")
        return ws

    def get_workspace(self, agent_id: str) -> Optional[AgentWorkspace]:
        """Get an existing workspace."""
        return self._workspaces.get(agent_id)

    def destroy_workspace(self, agent_id: str) -> bool:
        """Destroy an agent's workspace."""
        ws = self._workspaces.pop(agent_id, None)
        if ws:
            ws.destroy()
            return True
        return False

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List all workspaces."""
        return [ws.get_info().to_dict() for ws in self._workspaces.values()]

    def check_quota(self, agent_id: str) -> Dict[str, Any]:
        """Check quota usage for an agent."""
        ws = self._workspaces.get(agent_id)
        if not ws:
            return {"agent_id": agent_id, "exists": False}

        size = ws.get_size()
        return {
            "agent_id": agent_id,
            "exists": True,
            "used_bytes": size,
            "quota_bytes": self._quota_bytes,
            "usage_percent": round(size / self._quota_bytes * 100, 1),
            "within_quota": size <= self._quota_bytes,
        }

    def set_quota(self, quota_bytes: int) -> None:
        """Set the per-workspace quota."""
        self._quota_bytes = quota_bytes

    def cleanup_empty(self) -> int:
        """Remove workspaces with no files."""
        removed = 0
        for agent_id in list(self._workspaces):
            ws = self._workspaces[agent_id]
            if not ws.list_files():
                self.destroy_workspace(agent_id)
                removed += 1
        return removed

    def stats(self) -> Dict[str, Any]:
        """Get workspace statistics."""
        total_size = sum(ws.get_size() for ws in self._workspaces.values())
        total_files = sum(len(ws.list_files()) for ws in self._workspaces.values())
        return {
            "total_workspaces": len(self._workspaces),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "quota_bytes": self._quota_bytes,
            "base_path": self._base_path,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[WorkspaceManager] = None


def get_workspace_manager() -> WorkspaceManager:
    """Get the global workspace manager."""
    global _manager
    if _manager is None:
        _manager = WorkspaceManager()
    return _manager
