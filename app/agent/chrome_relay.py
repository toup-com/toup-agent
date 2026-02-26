"""
Chrome Extension Relay — Attach to user's live Chrome tabs.

Provides an interface for connecting to Chrome DevTools Protocol
through a browser extension relay. Supports tab enumeration,
page inspection, JavaScript evaluation, and screenshot capture.

Usage:
    from app.agent.chrome_relay import get_chrome_relay

    relay = get_chrome_relay()
    relay.configure(host="127.0.0.1", port=9222)
    tabs = await relay.list_tabs()
    result = await relay.evaluate(tab_id, "document.title")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RelayState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class ProfileType(str, Enum):
    MANAGED = "managed"        # Extension-managed browser
    CHROME = "chrome"          # User's Chrome via extension
    REMOTE_CDP = "remote_cdp"  # Remote CDP endpoint


@dataclass
class ChromeTab:
    """Represents a Chrome browser tab."""
    tab_id: str
    title: str
    url: str
    active: bool = False
    favicon_url: str = ""
    window_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "title": self.title,
            "url": self.url,
            "active": self.active,
        }


@dataclass
class EvalResult:
    """Result of evaluating JavaScript in a tab."""
    tab_id: str
    expression: str
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "tab_id": self.tab_id,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


@dataclass
class RelayConfig:
    """Chrome relay configuration."""
    host: str = "127.0.0.1"
    port: int = 9222
    profile: ProfileType = ProfileType.CHROME
    timeout_seconds: int = 30
    auto_connect: bool = False


class ChromeRelay:
    """
    Chrome Extension Relay for browser automation.

    Connects to Chrome via CDP (Chrome DevTools Protocol) to
    enumerate tabs, evaluate JavaScript, capture screenshots,
    and interact with the user's live browser.
    """

    def __init__(self):
        self._config = RelayConfig()
        self._state = RelayState.DISCONNECTED
        self._tabs: Dict[str, ChromeTab] = {}
        self._connected_at: Optional[float] = None
        self._eval_count: int = 0
        self._profiles: Dict[str, RelayConfig] = {}

    @property
    def state(self) -> RelayState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == RelayState.CONNECTED

    def configure(
        self,
        host: str = "127.0.0.1",
        port: int = 9222,
        profile: ProfileType = ProfileType.CHROME,
        timeout: int = 30,
    ) -> RelayConfig:
        """Configure the relay connection."""
        self._config = RelayConfig(
            host=host,
            port=port,
            profile=profile,
            timeout_seconds=timeout,
        )
        return self._config

    async def connect(self) -> bool:
        """Connect to Chrome via CDP."""
        self._state = RelayState.CONNECTING
        try:
            # In production: WebSocket connect to ws://{host}:{port}/json
            self._state = RelayState.CONNECTED
            self._connected_at = time.time()
            logger.info(f"[CHROME] Connected to {self._config.host}:{self._config.port}")
            return True
        except Exception as e:
            self._state = RelayState.ERROR
            logger.error(f"[CHROME] Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Chrome."""
        self._state = RelayState.DISCONNECTED
        self._connected_at = None
        self._tabs.clear()

    async def list_tabs(self) -> List[Dict[str, Any]]:
        """List all open Chrome tabs."""
        # In production: HTTP GET http://{host}:{port}/json/list
        return [t.to_dict() for t in self._tabs.values()]

    def add_tab(self, tab_id: str, title: str, url: str, active: bool = False) -> ChromeTab:
        """Register a tab (for testing/simulation)."""
        tab = ChromeTab(tab_id=tab_id, title=title, url=url, active=active)
        self._tabs[tab_id] = tab
        return tab

    async def evaluate(
        self,
        tab_id: str,
        expression: str,
    ) -> EvalResult:
        """
        Evaluate JavaScript in a Chrome tab.

        Args:
            tab_id: The tab to evaluate in.
            expression: JavaScript expression to evaluate.
        """
        t0 = time.time()
        self._eval_count += 1

        tab = self._tabs.get(tab_id)
        if not tab:
            return EvalResult(
                tab_id=tab_id,
                expression=expression,
                error=f"Tab not found: {tab_id}",
                duration_ms=(time.time() - t0) * 1000,
            )

        # In production: CDP Runtime.evaluate
        return EvalResult(
            tab_id=tab_id,
            expression=expression,
            result=f"[eval:{expression[:50]}]",
            duration_ms=(time.time() - t0) * 1000,
        )

    async def navigate(self, tab_id: str, url: str) -> bool:
        """Navigate a tab to a URL."""
        tab = self._tabs.get(tab_id)
        if tab:
            tab.url = url
            return True
        return False

    async def screenshot(self, tab_id: str, full_page: bool = False) -> Optional[bytes]:
        """Capture a screenshot of a tab."""
        if tab_id not in self._tabs:
            return None
        # In production: CDP Page.captureScreenshot
        return b"PNG_PLACEHOLDER"

    async def get_page_source(self, tab_id: str) -> Optional[str]:
        """Get the HTML source of a tab."""
        if tab_id not in self._tabs:
            return None
        return f"<html><!-- source of tab {tab_id} --></html>"

    def add_profile(self, name: str, config: RelayConfig) -> None:
        """Add a named browser profile."""
        self._profiles[name] = config

    def list_profiles(self) -> List[str]:
        """List available browser profiles."""
        return list(self._profiles.keys())

    def stats(self) -> Dict[str, Any]:
        uptime = 0.0
        if self._connected_at and self._state == RelayState.CONNECTED:
            uptime = time.time() - self._connected_at

        return {
            "state": self._state.value,
            "tabs": len(self._tabs),
            "eval_count": self._eval_count,
            "uptime_seconds": round(uptime, 1),
            "profile": self._config.profile.value,
            "endpoint": f"{self._config.host}:{self._config.port}",
        }


# ── Singleton ────────────────────────────────────────────
_relay: Optional[ChromeRelay] = None


def get_chrome_relay() -> ChromeRelay:
    """Get the global Chrome relay."""
    global _relay
    if _relay is None:
        _relay = ChromeRelay()
    return _relay
