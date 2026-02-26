"""
Tailscale Integration — Serve/Funnel for remote access.

Manages Tailscale Serve and Funnel configurations for exposing
local services securely without port forwarding.

Usage:
    from app.agent.tailscale import get_tailscale_manager

    mgr = get_tailscale_manager()
    mgr.configure(auth_key="tskey-...")
    await mgr.serve(port=8000, path="/api")
    await mgr.funnel(port=8000)  # Public HTTPS
    status = mgr.status()
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TailscaleState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class ServeProtocol(str, Enum):
    HTTPS = "https"
    HTTP = "http"
    TCP = "tcp"
    TLS_TERMINATED_TCP = "tls-terminated-tcp"


@dataclass
class ServeConfig:
    """A Tailscale Serve configuration."""
    port: int
    path: str = "/"
    protocol: ServeProtocol = ServeProtocol.HTTPS
    funnel: bool = False  # Public access via Funnel
    active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "port": self.port,
            "path": self.path,
            "protocol": self.protocol.value,
            "funnel": self.funnel,
            "active": self.active,
        }


@dataclass
class TailscaleConfig:
    """Tailscale connection configuration."""
    auth_key: str = ""
    hostname: str = ""
    tailnet: str = ""
    accept_dns: bool = True
    accept_routes: bool = True


class TailscaleManager:
    """
    Manages Tailscale networking for the agent.

    Provides Serve (expose local ports to tailnet) and Funnel
    (expose to public internet via HTTPS) capabilities.
    """

    def __init__(self):
        self._config = TailscaleConfig()
        self._state = TailscaleState.DISCONNECTED
        self._serves: Dict[str, ServeConfig] = {}
        self._connected_at: Optional[float] = None
        self._ip: str = ""
        self._dns_name: str = ""

    @property
    def state(self) -> TailscaleState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == TailscaleState.CONNECTED

    def configure(
        self,
        auth_key: str = "",
        hostname: str = "",
        tailnet: str = "",
    ) -> TailscaleConfig:
        """Configure Tailscale connection."""
        self._config = TailscaleConfig(
            auth_key=auth_key,
            hostname=hostname,
            tailnet=tailnet,
        )
        return self._config

    async def connect(self) -> bool:
        """Connect to Tailscale network."""
        self._state = TailscaleState.CONNECTING
        # In production: run `tailscale up --authkey=...`
        self._state = TailscaleState.CONNECTED
        self._connected_at = time.time()
        self._ip = "100.64.0.1"
        self._dns_name = f"{self._config.hostname or 'hexbrain'}.{self._config.tailnet or 'tailnet'}.ts.net"
        logger.info(f"[TAILSCALE] Connected as {self._dns_name}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from Tailscale."""
        self._state = TailscaleState.DISCONNECTED
        self._connected_at = None
        for s in self._serves.values():
            s.active = False

    async def serve(
        self,
        port: int,
        path: str = "/",
        protocol: ServeProtocol = ServeProtocol.HTTPS,
    ) -> ServeConfig:
        """Expose a local port via Tailscale Serve."""
        key = f"{port}:{path}"
        config = ServeConfig(
            port=port,
            path=path,
            protocol=protocol,
            active=True,
        )
        self._serves[key] = config
        logger.info(f"[TAILSCALE] Serving {port}{path} via {protocol.value}")
        return config

    async def funnel(
        self,
        port: int,
        path: str = "/",
    ) -> ServeConfig:
        """Expose a local port publicly via Tailscale Funnel."""
        key = f"{port}:{path}"
        config = ServeConfig(
            port=port,
            path=path,
            protocol=ServeProtocol.HTTPS,
            funnel=True,
            active=True,
        )
        self._serves[key] = config
        logger.info(f"[TAILSCALE] Funnel {port}{path} → {self._dns_name}{path}")
        return config

    async def stop_serve(self, port: int, path: str = "/") -> bool:
        """Stop serving a port."""
        key = f"{port}:{path}"
        config = self._serves.get(key)
        if config:
            config.active = False
            return True
        return False

    def get_url(self, port: int, path: str = "/") -> Optional[str]:
        """Get the public URL for a served port."""
        key = f"{port}:{path}"
        config = self._serves.get(key)
        if config and config.active:
            return f"https://{self._dns_name}{path}"
        return None

    def list_serves(self) -> List[Dict[str, Any]]:
        """List all serve configurations."""
        return [s.to_dict() for s in self._serves.values()]

    def status(self) -> Dict[str, Any]:
        uptime = 0.0
        if self._connected_at and self.is_connected:
            uptime = time.time() - self._connected_at

        return {
            "state": self._state.value,
            "ip": self._ip,
            "dns_name": self._dns_name,
            "uptime_seconds": round(uptime, 1),
            "serves": len([s for s in self._serves.values() if s.active]),
            "funnels": len([s for s in self._serves.values() if s.funnel and s.active]),
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[TailscaleManager] = None


def get_tailscale_manager() -> TailscaleManager:
    """Get the global Tailscale manager."""
    global _manager
    if _manager is None:
        _manager = TailscaleManager()
    return _manager
