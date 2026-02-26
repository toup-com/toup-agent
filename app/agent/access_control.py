"""
DM Pairing & Access Control — pairing codes, per-channel policies, tool allow/deny lists.
"""

import hashlib
import logging
import secrets
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DMPolicy(str, Enum):
    """DM access policy modes."""
    OPEN = "open"           # Anyone can DM
    PAIRING = "pairing"     # Require pairing code
    ALLOWLIST = "allowlist"  # Only allowlisted users
    DISABLED = "disabled"    # No DMs


class GroupPolicy(str, Enum):
    """Group access policy modes."""
    OPEN = "open"           # Any group
    ALLOWLIST = "allowlist"  # Only allowlisted groups
    DISABLED = "disabled"    # No groups


@dataclass
class PairingCode:
    """A DM pairing code."""
    code: str
    user_id: str
    channel: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0
    used: bool = False
    approved: bool = False
    username: str = ""

    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "user_id": self.user_id,
            "channel": self.channel,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "used": self.used,
            "approved": self.approved,
            "username": self.username,
            "expired": self.is_expired(),
        }


@dataclass
class ChannelPolicy:
    """Access policy for a specific channel."""
    channel: str
    dm_policy: DMPolicy = DMPolicy.PAIRING
    group_policy: GroupPolicy = GroupPolicy.OPEN
    dm_allowlist: Set[str] = field(default_factory=set)
    group_allowlist: Set[str] = field(default_factory=set)
    blocked_users: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "dm_policy": self.dm_policy.value,
            "group_policy": self.group_policy.value,
            "dm_allowlist": sorted(self.dm_allowlist),
            "group_allowlist": sorted(self.group_allowlist),
            "blocked_users": sorted(self.blocked_users),
        }


@dataclass
class ToolPolicy:
    """Tool access control policy."""
    allowed_tools: Optional[Set[str]] = None   # None = all allowed
    denied_tools: Set[str] = field(default_factory=set)
    elevated_tools: Set[str] = field(default_factory=set)  # Require confirmation

    def is_allowed(self, tool_name: str) -> bool:
        if tool_name in self.denied_tools:
            return False
        if self.allowed_tools is not None:
            return tool_name in self.allowed_tools
        return True

    def is_elevated(self, tool_name: str) -> bool:
        return tool_name in self.elevated_tools

    def to_dict(self) -> dict:
        return {
            "allowed_tools": sorted(self.allowed_tools) if self.allowed_tools else None,
            "denied_tools": sorted(self.denied_tools),
            "elevated_tools": sorted(self.elevated_tools),
        }


class AccessController:
    """Central access control manager."""

    def __init__(self):
        self._channel_policies: Dict[str, ChannelPolicy] = {}
        self._pairing_codes: Dict[str, PairingCode] = {}
        self._tool_policy = ToolPolicy()
        self._owner_ids: Set[str] = set()
        self._pairing_ttl: float = 3600  # 1 hour default

    def set_owner(self, user_id: str) -> None:
        """Set a user as owner (admin)."""
        self._owner_ids.add(user_id)

    def is_owner(self, user_id: str) -> bool:
        return user_id in self._owner_ids

    def get_channel_policy(self, channel: str) -> ChannelPolicy:
        """Get or create policy for a channel."""
        if channel not in self._channel_policies:
            self._channel_policies[channel] = ChannelPolicy(channel=channel)
        return self._channel_policies[channel]

    def set_dm_policy(self, channel: str, policy: DMPolicy) -> None:
        cp = self.get_channel_policy(channel)
        cp.dm_policy = policy

    def set_group_policy(self, channel: str, policy: GroupPolicy) -> None:
        cp = self.get_channel_policy(channel)
        cp.group_policy = policy

    def add_to_allowlist(self, channel: str, user_id: str, is_group: bool = False) -> None:
        cp = self.get_channel_policy(channel)
        if is_group:
            cp.group_allowlist.add(user_id)
        else:
            cp.dm_allowlist.add(user_id)

    def remove_from_allowlist(self, channel: str, user_id: str, is_group: bool = False) -> None:
        cp = self.get_channel_policy(channel)
        if is_group:
            cp.group_allowlist.discard(user_id)
        else:
            cp.dm_allowlist.discard(user_id)

    def block_user(self, channel: str, user_id: str) -> None:
        cp = self.get_channel_policy(channel)
        cp.blocked_users.add(user_id)

    def unblock_user(self, channel: str, user_id: str) -> None:
        cp = self.get_channel_policy(channel)
        cp.blocked_users.discard(user_id)

    def check_dm_access(self, channel: str, user_id: str) -> dict:
        """Check if a user can DM on a channel."""
        cp = self.get_channel_policy(channel)

        if user_id in cp.blocked_users:
            return {"allowed": False, "reason": "blocked"}
        if self.is_owner(user_id):
            return {"allowed": True, "reason": "owner"}

        if cp.dm_policy == DMPolicy.OPEN:
            return {"allowed": True, "reason": "open_policy"}
        elif cp.dm_policy == DMPolicy.DISABLED:
            return {"allowed": False, "reason": "dm_disabled"}
        elif cp.dm_policy == DMPolicy.ALLOWLIST:
            if user_id in cp.dm_allowlist:
                return {"allowed": True, "reason": "allowlisted"}
            return {"allowed": False, "reason": "not_in_allowlist"}
        elif cp.dm_policy == DMPolicy.PAIRING:
            if user_id in cp.dm_allowlist:
                return {"allowed": True, "reason": "paired"}
            return {"allowed": False, "reason": "pairing_required"}

        return {"allowed": False, "reason": "unknown_policy"}

    def check_group_access(self, channel: str, group_id: str) -> dict:
        """Check if a group is allowed on a channel."""
        cp = self.get_channel_policy(channel)

        if cp.group_policy == GroupPolicy.OPEN:
            return {"allowed": True, "reason": "open_policy"}
        elif cp.group_policy == GroupPolicy.DISABLED:
            return {"allowed": False, "reason": "groups_disabled"}
        elif cp.group_policy == GroupPolicy.ALLOWLIST:
            if group_id in cp.group_allowlist:
                return {"allowed": True, "reason": "allowlisted"}
            return {"allowed": False, "reason": "not_in_allowlist"}

        return {"allowed": False, "reason": "unknown_policy"}

    def generate_pairing_code(self, user_id: str, channel: str, username: str = "") -> PairingCode:
        """Generate a pairing code for a new user."""
        code = secrets.token_hex(3).upper()  # 6-char hex code
        pairing = PairingCode(
            code=code,
            user_id=user_id,
            channel=channel,
            expires_at=time.time() + self._pairing_ttl,
            username=username,
        )
        self._pairing_codes[code] = pairing
        return pairing

    def approve_pairing(self, code: str) -> dict:
        """Approve a pairing code (owner action)."""
        pairing = self._pairing_codes.get(code)
        if not pairing:
            return {"success": False, "error": "Invalid code"}
        if pairing.is_expired():
            return {"success": False, "error": "Code expired"}
        if pairing.used:
            return {"success": False, "error": "Code already used"}

        pairing.approved = True
        pairing.used = True
        self.add_to_allowlist(pairing.channel, pairing.user_id)
        return {"success": True, "user_id": pairing.user_id, "channel": pairing.channel}

    def list_pending_pairings(self) -> List[PairingCode]:
        """List pending (unapproved) pairing codes."""
        return [
            p for p in self._pairing_codes.values()
            if not p.used and not p.is_expired()
        ]

    def get_tool_policy(self) -> ToolPolicy:
        return self._tool_policy

    def set_tool_denied(self, tools: List[str]) -> None:
        self._tool_policy.denied_tools = set(tools)

    def set_tool_elevated(self, tools: List[str]) -> None:
        self._tool_policy.elevated_tools = set(tools)

    def check_tool_access(self, tool_name: str) -> dict:
        """Check if a tool is allowed."""
        allowed = self._tool_policy.is_allowed(tool_name)
        elevated = self._tool_policy.is_elevated(tool_name)
        return {
            "allowed": allowed,
            "elevated": elevated,
            "tool": tool_name,
        }


_controller: Optional[AccessController] = None


def get_access_controller() -> AccessController:
    """Get the global access controller singleton."""
    global _controller
    if _controller is None:
        _controller = AccessController()
    return _controller
