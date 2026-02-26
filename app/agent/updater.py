"""
Update Mechanism — Self-update support for HexBrain.

Checks for new versions, downloads updates, and applies them.
Supports both git-based and release-based update strategies.

Usage:
    from app.agent.updater import get_updater

    updater = get_updater()
    info = await updater.check_for_updates()
    if info["update_available"]:
        result = await updater.apply_update()
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CURRENT_VERSION = "1.0.0"


class UpdateStrategy(str, Enum):
    GIT = "git"
    RELEASE = "release"
    DOCKER = "docker"
    MANUAL = "manual"


class UpdateState(str, Enum):
    IDLE = "idle"
    CHECKING = "checking"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    APPLYING = "applying"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class VersionInfo:
    """Information about a version."""
    version: str
    release_date: str = ""
    changelog: str = ""
    download_url: str = ""
    commit_hash: str = ""
    breaking_changes: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "release_date": self.release_date,
            "changelog": self.changelog,
            "breaking_changes": self.breaking_changes,
            "commit_hash": self.commit_hash,
        }


@dataclass
class UpdateResult:
    """Result of an update operation."""
    success: bool
    from_version: str
    to_version: str
    strategy: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "strategy": self.strategy,
            "message": self.message,
        }


class Updater:
    """
    Manages HexBrain version checking and updates.

    Supports multiple update strategies:
    - git: Pull latest from the repository
    - release: Download from GitHub releases
    - docker: Pull latest Docker image
    - manual: Provide instructions
    """

    def __init__(self, strategy: UpdateStrategy = UpdateStrategy.GIT):
        self._strategy = strategy
        self._current_version = CURRENT_VERSION
        self._state = UpdateState.IDLE
        self._latest_info: Optional[VersionInfo] = None
        self._update_history: List[UpdateResult] = []
        self._repo_path = os.environ.get("HEXBRAIN_REPO_PATH", "/app")

    @property
    def current_version(self) -> str:
        return self._current_version

    @property
    def state(self) -> UpdateState:
        return self._state

    @property
    def strategy(self) -> UpdateStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: UpdateStrategy):
        self._strategy = value

    async def check_for_updates(self) -> Dict[str, Any]:
        """
        Check if a newer version is available.

        Returns:
            Dict with current_version, latest_version, update_available.
        """
        self._state = UpdateState.CHECKING

        try:
            if self._strategy == UpdateStrategy.GIT:
                info = await self._check_git()
            elif self._strategy == UpdateStrategy.DOCKER:
                info = await self._check_docker()
            else:
                info = VersionInfo(version=self._current_version)

            self._latest_info = info
            update_available = info.version != self._current_version

            if update_available:
                self._state = UpdateState.AVAILABLE
            else:
                self._state = UpdateState.IDLE

            return {
                "current_version": self._current_version,
                "latest_version": info.version,
                "update_available": update_available,
                "strategy": self._strategy.value,
                "info": info.to_dict(),
            }

        except Exception as e:
            self._state = UpdateState.FAILED
            return {
                "current_version": self._current_version,
                "latest_version": None,
                "update_available": False,
                "error": str(e),
            }

    async def apply_update(self, to_version: Optional[str] = None) -> UpdateResult:
        """
        Apply an update to the specified or latest version.
        """
        target = to_version or (self._latest_info.version if self._latest_info else self._current_version)
        self._state = UpdateState.APPLYING

        try:
            if self._strategy == UpdateStrategy.GIT:
                result = await self._apply_git(target)
            elif self._strategy == UpdateStrategy.DOCKER:
                result = UpdateResult(
                    success=False,
                    from_version=self._current_version,
                    to_version=target,
                    strategy=self._strategy.value,
                    message="Docker updates require manual: docker compose pull && docker compose up -d",
                )
            else:
                result = UpdateResult(
                    success=False,
                    from_version=self._current_version,
                    to_version=target,
                    strategy=self._strategy.value,
                    message="Manual update required",
                )

            self._update_history.append(result)
            self._state = UpdateState.COMPLETE if result.success else UpdateState.FAILED
            return result

        except Exception as e:
            result = UpdateResult(
                success=False,
                from_version=self._current_version,
                to_version=target,
                strategy=self._strategy.value,
                message=str(e),
            )
            self._update_history.append(result)
            self._state = UpdateState.FAILED
            return result

    async def _check_git(self) -> VersionInfo:
        """Check for git updates by fetching remote."""
        try:
            proc = await asyncio.create_subprocess_shell(
                f"cd {self._repo_path} && git fetch origin main --dry-run 2>&1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            output = (stdout or b"").decode() + (stderr or b"").decode()

            # Get local commit
            proc2 = await asyncio.create_subprocess_shell(
                f"cd {self._repo_path} && git rev-parse HEAD",
                stdout=asyncio.subprocess.PIPE,
            )
            stdout2, _ = await proc2.communicate()
            local_hash = (stdout2 or b"").decode().strip()[:8]

            has_updates = bool(output.strip())
            return VersionInfo(
                version=f"{self._current_version}-next" if has_updates else self._current_version,
                commit_hash=local_hash,
            )
        except Exception:
            return VersionInfo(version=self._current_version)

    async def _check_docker(self) -> VersionInfo:
        """Check for Docker image updates."""
        return VersionInfo(version=self._current_version)

    async def _apply_git(self, target: str) -> UpdateResult:
        """Apply a git-based update."""
        try:
            proc = await asyncio.create_subprocess_shell(
                f"cd {self._repo_path} && git pull origin main 2>&1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            output = (stdout or b"").decode()

            success = proc.returncode == 0
            return UpdateResult(
                success=success,
                from_version=self._current_version,
                to_version=target,
                strategy="git",
                message=output[:500],
                details={"return_code": proc.returncode},
            )
        except Exception as e:
            return UpdateResult(
                success=False,
                from_version=self._current_version,
                to_version=target,
                strategy="git",
                message=str(e),
            )

    def get_update_history(self) -> List[Dict[str, Any]]:
        """Get history of update attempts."""
        return [r.to_dict() for r in self._update_history]

    def status(self) -> Dict[str, Any]:
        """Get current updater status."""
        return {
            "current_version": self._current_version,
            "state": self._state.value,
            "strategy": self._strategy.value,
            "latest_info": self._latest_info.to_dict() if self._latest_info else None,
            "update_count": len(self._update_history),
        }


# ── Singleton ────────────────────────────────────────────
_updater: Optional[Updater] = None


def get_updater() -> Updater:
    """Get the global updater instance."""
    global _updater
    if _updater is None:
        _updater = Updater()
    return _updater
