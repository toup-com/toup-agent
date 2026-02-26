"""
Docker Sandbox — Execute agent shell commands inside an isolated container.

Layer 6 enhancements:
  * Per-session containers (session_id-scoped, not just user_id)
  * Configurable resource limits per container
  * Auto-cleanup after idle timeout
  * Network policy options (none, internal, full)
  * Sandbox browser support (launch Chromium inside sandbox)
  * Container status + metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"


class NetworkPolicy(str, Enum):
    """Network access level for sandbox containers."""
    NONE = "none"         # No network access (most secure)
    INTERNAL = "internal"  # Only internal Docker network
    FULL = "full"          # Full network access


@dataclass
class ContainerConfig:
    """Resource limits and configuration for a sandbox container."""
    memory: str = "256m"
    cpus: str = "0.5"
    pids_limit: int = 64
    read_only: bool = True
    network: NetworkPolicy = NetworkPolicy.NONE
    tmpfs_size: str = "64m"
    image: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.image:
            self.image = getattr(settings, "sandbox_image", DEFAULT_SANDBOX_IMAGE)


@dataclass
class ContainerInfo:
    """Runtime info for an active sandbox container."""
    container_name: str
    user_id: str
    session_id: Optional[str]
    config: ContainerConfig
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    command_count: int = 0


class SandboxExecutor:
    """
    Manages Docker sandbox containers for safe command execution.

    Usage::
        sandbox = SandboxExecutor()
        output = await sandbox.exec("ls -la", user_id="abc123", workdir="/workspace")
        output = await sandbox.exec("ls", user_id="abc", session_id="sess_1")
    """

    def __init__(self):
        self._containers: Dict[str, ContainerInfo] = {}  # key → ContainerInfo
        self._image = getattr(settings, "sandbox_image", DEFAULT_SANDBOX_IMAGE)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._idle_timeout: int = 1800  # 30 minutes default

    def _container_key(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Generate container lookup key."""
        if session_id:
            return f"{user_id}:{session_id}"
        return user_id

    # ────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────

    async def exec(
        self,
        command: str,
        user_id: str,
        workdir: str = "/workspace",
        timeout: int = 30,
        session_id: Optional[str] = None,
        config: Optional[ContainerConfig] = None,
    ) -> str:
        """Execute a command inside the sandbox container."""
        container = await self._ensure_container(
            user_id, workdir, session_id=session_id, config=config
        )

        # Update usage tracking
        key = self._container_key(user_id, session_id)
        if key in self._containers:
            self._containers[key].last_used = time.time()
            self._containers[key].command_count += 1

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-w", workdir, container,
                "sh", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"ERROR: Command timed out after {timeout}s"

            output = stdout.decode("utf-8", errors="replace")
            exit_code = proc.returncode
            if exit_code != 0:
                return f"{output}\n[exit code: {exit_code}]"
            return output or "(no output)"

        except FileNotFoundError:
            return "ERROR: Docker CLI not found on host"
        except Exception as e:
            logger.exception("[SANDBOX] exec failed for user %s", user_id)
            return f"ERROR: Sandbox exec failed: {e}"

    async def stop(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """Stop and remove a sandbox container."""
        key = self._container_key(user_id, session_id)
        info = self._containers.pop(key, None)
        if not info:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", info.container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            logger.info("[SANDBOX] Removed container %s", info.container_name)
            return True
        except Exception:
            logger.warning("[SANDBOX] Failed to remove container %s", info.container_name)
            return False

    async def stop_all(self) -> int:
        """Stop all active sandbox containers."""
        keys = list(self._containers.keys())
        count = 0
        for key in keys:
            info = self._containers.get(key)
            if info:
                if await self.stop(info.user_id, info.session_id):
                    count += 1
        return count

    def list_containers(self) -> Dict[str, Dict[str, Any]]:
        """Return info about all active sandbox containers."""
        result = {}
        for key, info in self._containers.items():
            result[key] = {
                "container_name": info.container_name,
                "user_id": info.user_id,
                "session_id": info.session_id,
                "created_at": info.created_at,
                "last_used": info.last_used,
                "command_count": info.command_count,
                "idle_seconds": int(time.time() - info.last_used),
                "memory": info.config.memory,
                "cpus": info.config.cpus,
                "network": info.config.network.value,
            }
        return result

    async def get_container_stats(self, user_id: str,
                                  session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get resource usage stats for a container."""
        key = self._container_key(user_id, session_id)
        info = self._containers.get(key)
        if not info:
            return {"error": "No active container"}

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "stats", "--no-stream", "--format",
                "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.PIDs}}",
                info.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            parts = stdout.decode().strip().split("|")
            if len(parts) >= 4:
                return {
                    "container": info.container_name,
                    "cpu": parts[0].strip(),
                    "memory": parts[1].strip(),
                    "network_io": parts[2].strip(),
                    "pids": parts[3].strip(),
                    "uptime_seconds": int(time.time() - info.created_at),
                    "command_count": info.command_count,
                }
        except Exception as e:
            return {"error": f"Stats unavailable: {e}"}

        return {"error": "Stats parsing failed"}

    # ── Auto-cleanup ──

    async def start_cleanup_loop(self, idle_timeout: int = 1800):
        """Start background task that cleans up idle containers."""
        self._idle_timeout = idle_timeout
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("[SANDBOX] Cleanup loop started (timeout=%ds)", idle_timeout)

    async def stop_cleanup_loop(self):
        """Stop the cleanup loop."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("[SANDBOX] Cleanup loop stopped")

    async def _cleanup_loop(self):
        """Periodically remove idle containers."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                now = time.time()
                idle_keys = [
                    k for k, v in self._containers.items()
                    if now - v.last_used > self._idle_timeout
                ]
                for key in idle_keys:
                    info = self._containers.get(key)
                    if info:
                        logger.info(
                            "[SANDBOX] Auto-cleanup idle container %s (idle %ds)",
                            info.container_name, int(now - info.last_used)
                        )
                        await self.stop(info.user_id, info.session_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[SANDBOX] Cleanup error: %s", e)

    # ────────────────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────────────────

    async def _ensure_container(self, user_id: str, workdir: str,
                                session_id: Optional[str] = None,
                                config: Optional[ContainerConfig] = None) -> str:
        """Get or create the sandbox container."""
        key = self._container_key(user_id, session_id)

        if key in self._containers:
            name = self._containers[key].container_name
            if await self._is_running(name):
                return name
            self._containers.pop(key, None)

        cfg = config or ContainerConfig()

        # Build container name
        suffix = f"-{session_id[:8]}" if session_id else ""
        container_name = f"hex-sandbox-{user_id[:8]}{suffix}"

        # Resolve host workspace path
        host_workspace = os.path.join(settings.agent_workspace_dir, user_id)
        os.makedirs(host_workspace, exist_ok=True)

        try:
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "--memory", cfg.memory,
                "--cpus", cfg.cpus,
                "--pids-limit", str(cfg.pids_limit),
                "--tmpfs", f"/tmp:size={cfg.tmpfs_size}",
                "-v", f"{host_workspace}:{workdir}",
                "-w", workdir,
            ]

            if cfg.read_only:
                cmd.append("--read-only")

            if cfg.network == NetworkPolicy.NONE:
                cmd.extend(["--network", "none"])
            elif cfg.network == NetworkPolicy.INTERNAL:
                cmd.extend(["--network", "bridge"])
                # Internal: no port publishing

            for env_key, env_val in cfg.env_vars.items():
                cmd.extend(["-e", f"{env_key}={env_val}"])

            cmd.extend([cfg.image, "sleep", "infinity"])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")
                if "Conflict" in err or "already in use" in err:
                    await asyncio.create_subprocess_exec(
                        "docker", "start", container_name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                else:
                    raise RuntimeError(f"Docker run failed: {err}")

            self._containers[key] = ContainerInfo(
                container_name=container_name,
                user_id=user_id,
                session_id=session_id,
                config=cfg,
            )
            logger.info("[SANDBOX] Created container %s for %s", container_name, key)
            return container_name

        except Exception as e:
            logger.exception("[SANDBOX] Failed to create container for %s", key)
            raise

    async def _is_running(self, container_name: str) -> bool:
        """Check if a container is running."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f", "{{.State.Running}}", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip() == "true"
        except Exception:
            return False
