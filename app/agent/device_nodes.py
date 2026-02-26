"""
Device Nodes — Camera, Screen, Location, and Notification capabilities.

Manages connections to device nodes (macOS, iOS, Android) that provide
hardware capabilities: camera capture, screen recording, GPS location,
and native OS notifications.

Usage:
    from app.agent.device_nodes import get_node_manager

    mgr = get_node_manager()
    mgr.register_node("iphone_1", capabilities=["camera", "location", "notifications"])
    photo = await mgr.capture_camera("iphone_1")
    loc = await mgr.get_location("iphone_1")
    await mgr.send_notification("iphone_1", title="Alert", body="Task done!")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class NodeCapability(str, Enum):
    CAMERA = "camera"
    SCREEN = "screen"
    LOCATION = "location"
    NOTIFICATIONS = "notifications"
    MICROPHONE = "microphone"
    CLIPBOARD = "clipboard"


class NodeState(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"


class NodePlatform(str, Enum):
    MACOS = "macos"
    IOS = "ios"
    ANDROID = "android"
    LINUX = "linux"
    WINDOWS = "windows"


@dataclass
class DeviceNode:
    """A registered device node."""
    node_id: str
    platform: NodePlatform = NodePlatform.MACOS
    state: NodeState = NodeState.ONLINE
    capabilities: Set[str] = field(default_factory=set)
    last_seen: float = 0.0
    registered_at: float = 0.0
    request_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.registered_at == 0.0:
            self.registered_at = time.time()
        if self.last_seen == 0.0:
            self.last_seen = time.time()

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "platform": self.platform.value,
            "state": self.state.value,
            "capabilities": sorted(self.capabilities),
            "request_count": self.request_count,
        }


@dataclass
class CaptureResult:
    """Result of a device capture operation."""
    node_id: str
    capability: str
    success: bool = True
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "node_id": self.node_id,
            "capability": self.capability,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.error:
            d["error"] = self.error
        return d


class DeviceNodeManager:
    """
    Manages device node connections and capabilities.

    Device nodes are native apps (macOS, iOS, Android) that connect
    to the gateway and provide hardware access.
    """

    def __init__(self):
        self._nodes: Dict[str, DeviceNode] = {}

    def register_node(
        self,
        node_id: str,
        *,
        platform: NodePlatform = NodePlatform.MACOS,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeviceNode:
        """Register a device node."""
        node = DeviceNode(
            node_id=node_id,
            platform=platform,
            capabilities=set(capabilities or []),
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        logger.info(f"[DEVICE] Registered node {node_id} ({platform.value})")
        return node

    def unregister_node(self, node_id: str) -> bool:
        """Unregister a device node."""
        return self._nodes.pop(node_id, None) is not None

    def get_node(self, node_id: str) -> Optional[DeviceNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def set_state(self, node_id: str, state: NodeState) -> bool:
        """Update a node's state."""
        node = self._nodes.get(node_id)
        if node:
            node.state = state
            node.last_seen = time.time()
            return True
        return False

    def heartbeat(self, node_id: str) -> bool:
        """Update last_seen for a node."""
        node = self._nodes.get(node_id)
        if node:
            node.last_seen = time.time()
            node.state = NodeState.ONLINE
            return True
        return False

    def find_nodes_with_capability(self, capability: str) -> List[DeviceNode]:
        """Find all online nodes with a specific capability."""
        return [
            n for n in self._nodes.values()
            if n.has_capability(capability) and n.state == NodeState.ONLINE
        ]

    async def capture_camera(self, node_id: str) -> CaptureResult:
        """Capture a photo from a node's camera."""
        return await self._request(node_id, NodeCapability.CAMERA)

    async def capture_screen(self, node_id: str) -> CaptureResult:
        """Capture the screen from a node."""
        return await self._request(node_id, NodeCapability.SCREEN)

    async def get_location(self, node_id: str) -> CaptureResult:
        """Get GPS location from a node."""
        return await self._request(node_id, NodeCapability.LOCATION)

    async def send_notification(
        self,
        node_id: str,
        *,
        title: str = "",
        body: str = "",
    ) -> CaptureResult:
        """Send a native notification to a node."""
        node = self._nodes.get(node_id)
        if not node:
            return CaptureResult(
                node_id=node_id,
                capability="notifications",
                success=False,
                error="Node not found",
            )
        if not node.has_capability("notifications"):
            return CaptureResult(
                node_id=node_id,
                capability="notifications",
                success=False,
                error="Node lacks notifications capability",
            )

        node.request_count += 1
        # In production: send via WebSocket to the node
        return CaptureResult(
            node_id=node_id,
            capability="notifications",
            data={"title": title, "body": body, "sent": True},
        )

    async def _request(self, node_id: str, capability: NodeCapability) -> CaptureResult:
        """Send a capability request to a node."""
        t0 = time.time()
        node = self._nodes.get(node_id)

        if not node:
            return CaptureResult(
                node_id=node_id,
                capability=capability.value,
                success=False,
                error="Node not found",
            )

        if node.state != NodeState.ONLINE:
            return CaptureResult(
                node_id=node_id,
                capability=capability.value,
                success=False,
                error=f"Node is {node.state.value}",
            )

        if not node.has_capability(capability.value):
            return CaptureResult(
                node_id=node_id,
                capability=capability.value,
                success=False,
                error=f"Node lacks {capability.value} capability",
            )

        node.request_count += 1
        node.last_seen = time.time()

        # In production: send request via WebSocket, await response
        return CaptureResult(
            node_id=node_id,
            capability=capability.value,
            data=f"[{capability.value}_capture_placeholder]",
            duration_ms=(time.time() - t0) * 1000,
        )

    def list_nodes(
        self,
        state: Optional[NodeState] = None,
        capability: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all registered nodes."""
        nodes = list(self._nodes.values())
        if state:
            nodes = [n for n in nodes if n.state == state]
        if capability:
            nodes = [n for n in nodes if n.has_capability(capability)]
        return [n.to_dict() for n in nodes]

    def stats(self) -> Dict[str, Any]:
        by_platform: Dict[str, int] = {}
        by_state: Dict[str, int] = {}
        total_requests = 0

        for n in self._nodes.values():
            by_platform[n.platform.value] = by_platform.get(n.platform.value, 0) + 1
            by_state[n.state.value] = by_state.get(n.state.value, 0) + 1
            total_requests += n.request_count

        return {
            "total_nodes": len(self._nodes),
            "by_platform": by_platform,
            "by_state": by_state,
            "total_requests": total_requests,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[DeviceNodeManager] = None


def get_node_manager() -> DeviceNodeManager:
    """Get the global device node manager."""
    global _manager
    if _manager is None:
        _manager = DeviceNodeManager()
    return _manager
