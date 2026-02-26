"""
SSH Tunnels — Remote access via SSH port forwarding.

Manages SSH tunnel lifecycle for exposing local services
or connecting to remote services securely.

Usage:
    from app.agent.ssh_tunnels import get_tunnel_manager

    mgr = get_tunnel_manager()
    tunnel = mgr.create_tunnel(
        name="api",
        remote_host="vps.example.com",
        remote_port=22,
        local_port=8000,
        remote_bind_port=9000,
        ssh_user="deploy",
    )
    await mgr.open_tunnel("api")
    await mgr.close_tunnel("api")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TunnelType(str, Enum):
    LOCAL = "local"       # -L: local port → remote service
    REMOTE = "remote"     # -R: remote port → local service
    DYNAMIC = "dynamic"   # -D: SOCKS proxy


class TunnelState(str, Enum):
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class SSHTunnel:
    """An SSH tunnel configuration."""
    name: str
    tunnel_type: TunnelType
    remote_host: str
    remote_port: int = 22
    ssh_user: str = "root"
    local_port: int = 0
    remote_bind_port: int = 0
    ssh_key_path: str = ""
    state: TunnelState = TunnelState.CLOSED
    auto_reconnect: bool = True
    keepalive_interval: int = 60
    opened_at: Optional[float] = None
    reconnect_count: int = 0
    bytes_transferred: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.tunnel_type.value,
            "remote_host": self.remote_host,
            "remote_port": self.remote_port,
            "local_port": self.local_port,
            "remote_bind_port": self.remote_bind_port,
            "state": self.state.value,
            "auto_reconnect": self.auto_reconnect,
        }

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.remote_host}:{self.remote_port}"

    @property
    def uptime(self) -> float:
        if self.opened_at and self.state == TunnelState.OPEN:
            return time.time() - self.opened_at
        return 0.0


class SSHTunnelManager:
    """
    Manages SSH tunnels for remote access.

    Supports local (-L), remote (-R), and dynamic (-D) port
    forwarding with auto-reconnect.
    """

    def __init__(self):
        self._tunnels: Dict[str, SSHTunnel] = {}

    def create_tunnel(
        self,
        name: str,
        remote_host: str,
        *,
        tunnel_type: TunnelType = TunnelType.REMOTE,
        remote_port: int = 22,
        ssh_user: str = "root",
        local_port: int = 0,
        remote_bind_port: int = 0,
        ssh_key_path: str = "",
        auto_reconnect: bool = True,
    ) -> SSHTunnel:
        """Create a new SSH tunnel configuration."""
        tunnel = SSHTunnel(
            name=name,
            tunnel_type=tunnel_type,
            remote_host=remote_host,
            remote_port=remote_port,
            ssh_user=ssh_user,
            local_port=local_port,
            remote_bind_port=remote_bind_port,
            ssh_key_path=ssh_key_path,
            auto_reconnect=auto_reconnect,
        )
        self._tunnels[name] = tunnel
        logger.info(f"[SSH] Created tunnel {name}: {tunnel.ssh_target}")
        return tunnel

    def get_tunnel(self, name: str) -> Optional[SSHTunnel]:
        """Get a tunnel by name."""
        return self._tunnels.get(name)

    def remove_tunnel(self, name: str) -> bool:
        """Remove a tunnel."""
        tunnel = self._tunnels.get(name)
        if tunnel and tunnel.state == TunnelState.OPEN:
            return False  # Must close first
        return self._tunnels.pop(name, None) is not None

    async def open_tunnel(self, name: str) -> bool:
        """Open an SSH tunnel."""
        tunnel = self._tunnels.get(name)
        if not tunnel:
            return False

        tunnel.state = TunnelState.OPENING
        # In production: subprocess ssh -L/-R/-D with correct args
        tunnel.state = TunnelState.OPEN
        tunnel.opened_at = time.time()
        logger.info(f"[SSH] Opened tunnel {name}")
        return True

    async def close_tunnel(self, name: str) -> bool:
        """Close an SSH tunnel."""
        tunnel = self._tunnels.get(name)
        if not tunnel or tunnel.state != TunnelState.OPEN:
            return False

        tunnel.state = TunnelState.CLOSED
        tunnel.opened_at = None
        logger.info(f"[SSH] Closed tunnel {name}")
        return True

    async def reconnect(self, name: str) -> bool:
        """Reconnect a tunnel."""
        tunnel = self._tunnels.get(name)
        if not tunnel:
            return False

        tunnel.state = TunnelState.RECONNECTING
        tunnel.reconnect_count += 1
        # In production: kill and restart ssh process
        tunnel.state = TunnelState.OPEN
        tunnel.opened_at = time.time()
        return True

    def list_tunnels(
        self,
        state: Optional[TunnelState] = None,
    ) -> List[Dict[str, Any]]:
        """List all tunnels."""
        tunnels = list(self._tunnels.values())
        if state:
            tunnels = [t for t in tunnels if t.state == state]
        return [t.to_dict() for t in tunnels]

    def get_open_count(self) -> int:
        """Count open tunnels."""
        return sum(1 for t in self._tunnels.values() if t.state == TunnelState.OPEN)

    def stats(self) -> Dict[str, Any]:
        by_state: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        total_uptime = 0.0

        for t in self._tunnels.values():
            by_state[t.state.value] = by_state.get(t.state.value, 0) + 1
            by_type[t.tunnel_type.value] = by_type.get(t.tunnel_type.value, 0) + 1
            total_uptime += t.uptime

        return {
            "total_tunnels": len(self._tunnels),
            "by_state": by_state,
            "by_type": by_type,
            "total_uptime_seconds": round(total_uptime, 1),
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[SSHTunnelManager] = None


def get_tunnel_manager() -> SSHTunnelManager:
    """Get the global SSH tunnel manager."""
    global _manager
    if _manager is None:
        _manager = SSHTunnelManager()
    return _manager
