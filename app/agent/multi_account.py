"""
Multi-Account — Multiple bot accounts per channel type.

Supports running multiple Telegram bots, WhatsApp numbers,
Discord bots, etc. simultaneously. Each account is identified
by a unique account_id and routes through the channel registry.

Usage:
    from app.agent.multi_account import get_account_manager

    mgr = get_account_manager()
    mgr.add_account("telegram", "bot_1", credentials={"token": "..."})
    mgr.add_account("telegram", "bot_2", credentials={"token": "..."})
    accounts = mgr.list_accounts("telegram")
    await mgr.connect_account("telegram", "bot_1")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AccountState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ChannelAccount:
    """A single account on a channel platform."""
    account_id: str
    platform: str
    credentials: Dict[str, str] = field(default_factory=dict)
    state: AccountState = AccountState.DISCONNECTED
    is_primary: bool = False
    label: str = ""
    connected_at: Optional[float] = None
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.account_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "platform": self.platform,
            "state": self.state.value,
            "is_primary": self.is_primary,
            "label": self.label or self.account_id,
            "message_count": self.message_count,
        }


class MultiAccountManager:
    """
    Manages multiple accounts per channel platform.

    Allows running multiple bot tokens, numbers, or credentials
    for the same channel type simultaneously.
    """

    def __init__(self):
        self._accounts: Dict[str, ChannelAccount] = {}  # key -> account
        self._primaries: Dict[str, str] = {}  # platform -> account_id

    def add_account(
        self,
        platform: str,
        account_id: str,
        *,
        credentials: Optional[Dict[str, str]] = None,
        is_primary: bool = False,
        label: str = "",
    ) -> ChannelAccount:
        """Add a new account for a platform."""
        account = ChannelAccount(
            account_id=account_id,
            platform=platform,
            credentials=credentials or {},
            is_primary=is_primary,
            label=label,
        )

        self._accounts[account.key] = account

        # Set as primary if first or explicitly requested
        if is_primary or platform not in self._primaries:
            self._primaries[platform] = account_id
            account.is_primary = True

        logger.info(f"[MULTI-ACCT] Added {account.key} (primary={account.is_primary})")
        return account

    def remove_account(self, platform: str, account_id: str) -> bool:
        """Remove an account."""
        key = f"{platform}:{account_id}"
        acct = self._accounts.pop(key, None)
        if not acct:
            return False

        # If removed primary, assign next available
        if self._primaries.get(platform) == account_id:
            remaining = [a for a in self._accounts.values() if a.platform == platform]
            if remaining:
                self._primaries[platform] = remaining[0].account_id
                remaining[0].is_primary = True
            else:
                del self._primaries[platform]

        return True

    def get_account(self, platform: str, account_id: str) -> Optional[ChannelAccount]:
        """Get a specific account."""
        return self._accounts.get(f"{platform}:{account_id}")

    def get_primary(self, platform: str) -> Optional[ChannelAccount]:
        """Get the primary account for a platform."""
        primary_id = self._primaries.get(platform)
        if primary_id:
            return self._accounts.get(f"{platform}:{primary_id}")
        return None

    def set_primary(self, platform: str, account_id: str) -> bool:
        """Set an account as primary for its platform."""
        key = f"{platform}:{account_id}"
        acct = self._accounts.get(key)
        if not acct:
            return False

        # Unset old primary
        old_id = self._primaries.get(platform)
        if old_id:
            old_key = f"{platform}:{old_id}"
            old = self._accounts.get(old_key)
            if old:
                old.is_primary = False

        self._primaries[platform] = account_id
        acct.is_primary = True
        return True

    async def connect_account(self, platform: str, account_id: str) -> bool:
        """Connect a specific account."""
        acct = self.get_account(platform, account_id)
        if not acct:
            return False

        acct.state = AccountState.CONNECTING
        # In production: instantiate channel and call connect()
        acct.state = AccountState.CONNECTED
        acct.connected_at = time.time()
        return True

    async def disconnect_account(self, platform: str, account_id: str) -> bool:
        """Disconnect a specific account."""
        acct = self.get_account(platform, account_id)
        if not acct:
            return False

        acct.state = AccountState.DISCONNECTED
        acct.connected_at = None
        return True

    def list_accounts(
        self,
        platform: Optional[str] = None,
        state: Optional[AccountState] = None,
    ) -> List[Dict[str, Any]]:
        """List all accounts, optionally filtered."""
        accounts = list(self._accounts.values())
        if platform:
            accounts = [a for a in accounts if a.platform == platform]
        if state:
            accounts = [a for a in accounts if a.state == state]
        return [a.to_dict() for a in accounts]

    def list_platforms(self) -> List[str]:
        """List platforms with registered accounts."""
        return list(set(a.platform for a in self._accounts.values()))

    def record_message(self, platform: str, account_id: str) -> None:
        """Record a message sent/received on an account."""
        acct = self.get_account(platform, account_id)
        if acct:
            acct.message_count += 1

    def stats(self) -> Dict[str, Any]:
        by_platform: Dict[str, int] = {}
        by_state: Dict[str, int] = {}
        for a in self._accounts.values():
            by_platform[a.platform] = by_platform.get(a.platform, 0) + 1
            by_state[a.state.value] = by_state.get(a.state.value, 0) + 1

        return {
            "total_accounts": len(self._accounts),
            "by_platform": by_platform,
            "by_state": by_state,
            "platforms_with_primary": list(self._primaries.keys()),
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[MultiAccountManager] = None


def get_account_manager() -> MultiAccountManager:
    """Get the global multi-account manager."""
    global _manager
    if _manager is None:
        _manager = MultiAccountManager()
    return _manager
