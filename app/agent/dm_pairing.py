"""
DM Pairing — Pairing code system for unknown senders.

When an unknown user sends a DM, they receive a pairing code.
The owner can approve the code to grant access. Supports
multiple policy modes per channel.

Usage:
    from app.agent.dm_pairing import get_pairing_manager

    mgr = get_pairing_manager()
    code = mgr.create_pairing("telegram", "user_456")
    mgr.approve_pairing(code.code)
    assert mgr.is_paired("telegram", "user_456")
"""

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DMPolicy(str, Enum):
    OPEN = "open"           # Anyone can DM
    PAIRING = "pairing"     # Require pairing code
    ALLOWLIST = "allowlist"  # Only pre-approved users
    DISABLED = "disabled"    # No DMs allowed


@dataclass
class PairingCode:
    """A DM pairing code."""
    code: str
    channel_type: str
    user_id: str
    status: str = "pending"  # pending, approved, rejected, expired
    created_at: float = 0.0
    resolved_at: Optional[float] = None
    expires_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + 86400  # 24h default

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def is_pending(self) -> bool:
        return self.status == "pending" and not self.is_expired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "channel_type": self.channel_type,
            "user_id": self.user_id,
            "status": self.status if not self.is_expired else "expired",
            "age_seconds": round(time.time() - self.created_at, 1),
        }


class PairingManager:
    """
    Manages DM pairing codes and access policies.

    Provides a secure handshake for unknown users:
    1. Unknown user sends a DM
    2. System generates a pairing code
    3. Owner approves/rejects the code
    4. User is added to paired list on approval
    """

    def __init__(self):
        self._codes: Dict[str, PairingCode] = {}  # code → PairingCode
        self._paired: Dict[str, set] = {}  # "channel:user_id" → set
        self._policies: Dict[str, DMPolicy] = {}  # channel_type → policy
        self._default_policy = DMPolicy.PAIRING

    def set_policy(self, channel_type: str, policy: DMPolicy) -> None:
        """Set DM policy for a channel type."""
        self._policies[channel_type] = policy
        logger.info(f"[PAIRING] Policy for {channel_type}: {policy.value}")

    def get_policy(self, channel_type: str) -> DMPolicy:
        """Get DM policy for a channel type."""
        return self._policies.get(channel_type, self._default_policy)

    def check_access(self, channel_type: str, user_id: str) -> bool:
        """
        Check if a user has DM access on a channel.

        Returns True if the user can send messages.
        """
        policy = self.get_policy(channel_type)

        if policy == DMPolicy.OPEN:
            return True
        if policy == DMPolicy.DISABLED:
            return False
        if policy in (DMPolicy.PAIRING, DMPolicy.ALLOWLIST):
            return self.is_paired(channel_type, user_id)

        return False

    def create_pairing(
        self,
        channel_type: str,
        user_id: str,
        *,
        ttl_hours: int = 24,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PairingCode:
        """
        Create a pairing code for an unknown user.

        Returns a code that the owner must approve.
        """
        code_str = secrets.token_hex(4).upper()  # 8-char hex code

        code = PairingCode(
            code=code_str,
            channel_type=channel_type,
            user_id=user_id,
            expires_at=time.time() + (ttl_hours * 3600),
            metadata=metadata or {},
        )

        self._codes[code_str] = code
        logger.info(f"[PAIRING] Created code {code_str} for {channel_type}:{user_id}")
        return code

    def approve_pairing(self, code: str) -> bool:
        """Approve a pairing code, granting the user access."""
        entry = self._codes.get(code)
        if not entry or entry.is_expired or entry.status != "pending":
            return False

        entry.status = "approved"
        entry.resolved_at = time.time()

        # Add to paired users
        self._add_paired(entry.channel_type, entry.user_id)
        logger.info(f"[PAIRING] Approved {code} for {entry.channel_type}:{entry.user_id}")
        return True

    def reject_pairing(self, code: str) -> bool:
        """Reject a pairing code."""
        entry = self._codes.get(code)
        if not entry or entry.status != "pending":
            return False

        entry.status = "rejected"
        entry.resolved_at = time.time()
        return True

    def is_paired(self, channel_type: str, user_id: str) -> bool:
        """Check if a user is paired on a channel."""
        key = f"{channel_type}:{user_id}"
        paired_set = self._paired.get(channel_type, set())
        return user_id in paired_set

    def _add_paired(self, channel_type: str, user_id: str) -> None:
        """Add a user to the paired set."""
        if channel_type not in self._paired:
            self._paired[channel_type] = set()
        self._paired[channel_type].add(user_id)

    def add_to_allowlist(self, channel_type: str, user_id: str) -> None:
        """Directly add a user to the allowlist (no pairing needed)."""
        self._add_paired(channel_type, user_id)

    def remove_from_allowlist(self, channel_type: str, user_id: str) -> bool:
        """Remove a user from the allowlist."""
        paired_set = self._paired.get(channel_type, set())
        if user_id in paired_set:
            paired_set.discard(user_id)
            return True
        return False

    def get_pairing(self, code: str) -> Optional[PairingCode]:
        """Get a pairing code entry."""
        return self._codes.get(code)

    def list_pending(self, channel_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all pending pairing codes."""
        pending = [
            c.to_dict() for c in self._codes.values()
            if c.is_pending and (channel_type is None or c.channel_type == channel_type)
        ]
        return pending

    def list_paired_users(self, channel_type: str) -> List[str]:
        """List all paired users for a channel."""
        return sorted(self._paired.get(channel_type, set()))

    def cleanup_expired(self) -> int:
        """Remove expired pairing codes."""
        expired = [k for k, v in self._codes.items() if v.is_expired]
        for k in expired:
            self._codes.pop(k, None)
        return len(expired)

    def stats(self) -> Dict[str, Any]:
        """Get pairing statistics."""
        pending = sum(1 for c in self._codes.values() if c.is_pending)
        approved = sum(1 for c in self._codes.values() if c.status == "approved")
        rejected = sum(1 for c in self._codes.values() if c.status == "rejected")
        paired_counts = {k: len(v) for k, v in self._paired.items()}

        return {
            "total_codes": len(self._codes),
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "policies": {k: v.value for k, v in self._policies.items()},
            "paired_users": paired_counts,
            "default_policy": self._default_policy.value,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[PairingManager] = None


def get_pairing_manager() -> PairingManager:
    """Get the global pairing manager."""
    global _manager
    if _manager is None:
        _manager = PairingManager()
    return _manager
