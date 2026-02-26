"""
Presence Tracking — Per-channel connection state, probes, and health.

Tracks which channels are connected, their last activity, and runs
periodic health probes to detect stale connections.

Usage:
    from app.agent.presence import get_presence_tracker

    tracker = get_presence_tracker()
    tracker.update("telegram", connected=True, details={"bot_id": "123"})
    status = tracker.get_status("telegram")
    all_status = tracker.get_all()
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ChannelPresence:
    """Presence state for a single channel."""
    channel: str
    state: ConnectionState = ConnectionState.UNKNOWN
    connected_at: float = 0.0
    last_activity: float = 0.0
    last_probe: float = 0.0
    probe_latency_ms: float = 0.0
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    reconnect_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        uptime = 0.0
        if self.connected_at > 0 and self.state == ConnectionState.CONNECTED:
            uptime = time.time() - self.connected_at

        return {
            "channel": self.channel,
            "state": self.state.value,
            "uptime_seconds": round(uptime, 1),
            "last_activity_ago": round(time.time() - self.last_activity, 1) if self.last_activity else None,
            "last_probe_ago": round(time.time() - self.last_probe, 1) if self.last_probe else None,
            "probe_latency_ms": round(self.probe_latency_ms, 1),
            "error": self.error_message or None,
            "message_count": self.message_count,
            "reconnect_count": self.reconnect_count,
            "details": self.details,
        }


# Probe callback type
ProbeCallback = Callable[[str], Coroutine[Any, Any, bool]]


class PresenceTracker:
    """
    Tracks connection presence for all channels.

    Each channel has a ChannelPresence that records:
    - Connection state (connected/disconnected/error)
    - Uptime / last activity / last probe
    - Message count and reconnect count
    """

    def __init__(self):
        self._channels: Dict[str, ChannelPresence] = {}
        self._probes: Dict[str, ProbeCallback] = {}

    def update(
        self,
        channel: str,
        *,
        connected: Optional[bool] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        message: bool = False,
    ) -> ChannelPresence:
        """
        Update presence state for a channel.

        Args:
            channel: Channel identifier (telegram, discord, etc.)
            connected: True=connected, False=disconnected, None=no change
            error: Error message (sets state to ERROR)
            details: Additional channel-specific details
            message: If True, increment message count and update last_activity
        """
        if channel not in self._channels:
            self._channels[channel] = ChannelPresence(channel=channel)

        presence = self._channels[channel]

        if error:
            presence.state = ConnectionState.ERROR
            presence.error_message = error
        elif connected is True:
            was_disconnected = presence.state != ConnectionState.CONNECTED
            presence.state = ConnectionState.CONNECTED
            presence.error_message = ""
            if was_disconnected:
                presence.connected_at = time.time()
                if presence.reconnect_count > 0 or presence.last_activity > 0:
                    presence.reconnect_count += 1
        elif connected is False:
            presence.state = ConnectionState.DISCONNECTED

        if details:
            presence.details.update(details)

        if message:
            presence.message_count += 1
            presence.last_activity = time.time()

        return presence

    def get_status(self, channel: str) -> Optional[Dict[str, Any]]:
        """Get presence status for a specific channel."""
        p = self._channels.get(channel)
        return p.to_dict() if p else None

    def get_all(self) -> List[Dict[str, Any]]:
        """Get presence status for all channels."""
        return [p.to_dict() for p in self._channels.values()]

    def is_connected(self, channel: str) -> bool:
        """Check if a channel is currently connected."""
        p = self._channels.get(channel)
        return p is not None and p.state == ConnectionState.CONNECTED

    def connected_channels(self) -> List[str]:
        """List all currently connected channels."""
        return [
            name for name, p in self._channels.items()
            if p.state == ConnectionState.CONNECTED
        ]

    def register_probe(self, channel: str, callback: ProbeCallback) -> None:
        """Register a health probe callback for a channel."""
        self._probes[channel] = callback

    async def run_probe(self, channel: str) -> Dict[str, Any]:
        """Run a health probe for a channel."""
        if channel not in self._probes:
            return {"channel": channel, "ok": False, "error": "No probe registered"}

        presence = self._channels.get(channel)
        if not presence:
            return {"channel": channel, "ok": False, "error": "Channel not tracked"}

        t0 = time.time()
        try:
            ok = await self._probes[channel](channel)
            latency = (time.time() - t0) * 1000
            presence.last_probe = time.time()
            presence.probe_latency_ms = latency
            if ok:
                presence.state = ConnectionState.CONNECTED
                presence.error_message = ""
            else:
                presence.state = ConnectionState.ERROR
                presence.error_message = "Probe returned False"
            return {"channel": channel, "ok": ok, "latency_ms": round(latency, 1)}
        except Exception as e:
            latency = (time.time() - t0) * 1000
            presence.last_probe = time.time()
            presence.probe_latency_ms = latency
            presence.state = ConnectionState.ERROR
            presence.error_message = str(e)
            return {"channel": channel, "ok": False, "error": str(e), "latency_ms": round(latency, 1)}

    async def run_all_probes(self) -> List[Dict[str, Any]]:
        """Run health probes for all registered channels."""
        results = []
        for channel in self._probes:
            result = await self.run_probe(channel)
            results.append(result)
        return results

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all channel presence."""
        total = len(self._channels)
        connected = sum(1 for p in self._channels.values() if p.state == ConnectionState.CONNECTED)
        errored = sum(1 for p in self._channels.values() if p.state == ConnectionState.ERROR)
        total_messages = sum(p.message_count for p in self._channels.values())

        return {
            "total_channels": total,
            "connected": connected,
            "disconnected": total - connected - errored,
            "errored": errored,
            "total_messages": total_messages,
        }

    def clear(self, channel: Optional[str] = None) -> None:
        """Clear presence data for a channel or all channels."""
        if channel:
            self._channels.pop(channel, None)
            self._probes.pop(channel, None)
        else:
            self._channels.clear()
            self._probes.clear()


# ── Singleton ────────────────────────────────────────────
_tracker: Optional[PresenceTracker] = None


def get_presence_tracker() -> PresenceTracker:
    """Get the global presence tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PresenceTracker()
    return _tracker
