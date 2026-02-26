"""
Browser Automation Service — Playwright-based headless browser control.

Layer 6 enhancements:
  * AI Snapshot (accessibility tree extraction for stable agent actions)
  * Tab Management (keep pages open, list/switch/close tabs)
  * Browser Profiles (managed, chrome, remote CDP)
  * Sandbox Browser (isolated browser in Docker sandbox)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Browser Profiles
# ──────────────────────────────────────────────────────────────

class BrowserProfile(str, Enum):
    """Browser launch profiles."""
    MANAGED = "managed"       # Playwright-managed headless Chromium
    CHROME = "chrome"         # Attach to user's Chrome via CDP
    REMOTE = "remote"         # Connect to a remote CDP endpoint

# ──────────────────────────────────────────────────────────────
# Tab Manager — Persistent page/tab lifecycle
# ──────────────────────────────────────────────────────────────

class TabManager:
    """
    Manages named browser tabs (pages) with persistence.
    Instead of creating and closing a page per interaction,
    pages can be kept alive and referenced by ID.
    """

    def __init__(self):
        self._tabs: Dict[str, Any] = {}  # tab_id → page
        self._counter: int = 0

    async def open_tab(self, browser, url: str = "about:blank",
                       viewport: Optional[Dict[str, int]] = None) -> str:
        """Open a new tab and return its ID."""
        kwargs = {}
        if viewport:
            kwargs["viewport"] = viewport
        page = await browser.new_page(**kwargs)
        if url and url != "about:blank":
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self._counter += 1
        tab_id = f"tab_{self._counter}"
        self._tabs[tab_id] = page
        logger.info("[BROWSER] Opened tab %s → %s", tab_id, url)
        return tab_id

    async def close_tab(self, tab_id: str) -> bool:
        """Close a tab by ID."""
        page = self._tabs.pop(tab_id, None)
        if page:
            try:
                await page.close()
            except Exception:
                pass
            logger.info("[BROWSER] Closed tab %s", tab_id)
            return True
        return False

    async def close_all(self):
        """Close all tabs."""
        for tid in list(self._tabs.keys()):
            await self.close_tab(tid)

    def get_tab(self, tab_id: str) -> Optional[Any]:
        """Get a page by tab ID."""
        return self._tabs.get(tab_id)

    def list_tabs(self) -> List[Dict[str, str]]:
        """List all open tabs with their URLs and titles."""
        result = []
        for tid, page in self._tabs.items():
            try:
                result.append({
                    "tab_id": tid,
                    "url": page.url,
                    "title": "",  # title requires await
                })
            except Exception:
                result.append({"tab_id": tid, "url": "unknown", "title": ""})
        return result

    async def list_tabs_async(self) -> List[Dict[str, str]]:
        """List all open tabs with titles (async)."""
        result = []
        for tid, page in self._tabs.items():
            try:
                title = await page.title()
                result.append({"tab_id": tid, "url": page.url, "title": title})
            except Exception:
                result.append({"tab_id": tid, "url": "unknown", "title": ""})
        return result

    @property
    def count(self) -> int:
        return len(self._tabs)


# ──────────────────────────────────────────────────────────────
# AI Snapshot — Accessibility tree extraction
# ──────────────────────────────────────────────────────────────

async def ai_snapshot(page, format: str = "aria") -> str:
    """
    Extract a stable accessibility tree from a page for agent actions.

    Args:
        page: Playwright page object
        format: 'aria' (ARIA tree) or 'ai' (simplified AI-friendly)

    Returns:
        Text representation of the page's accessibility tree.
    """
    try:
        snapshot = await page.accessibility.snapshot()
        if not snapshot:
            return "(empty accessibility tree)"

        if format == "ai":
            return _format_ai_tree(snapshot)
        else:
            return _format_aria_tree(snapshot)
    except Exception as e:
        logger.warning("[BROWSER] AI snapshot failed: %s", e)
        return f"ERROR: AI snapshot failed: {e}"


def _format_aria_tree(node: Dict, indent: int = 0) -> str:
    """Format accessibility tree as indented ARIA text."""
    lines = []
    prefix = "  " * indent
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    parts = [role]
    if name:
        parts.append(f'\"{name}\"')
    if value:
        parts.append(f"value={value}")

    # Add key properties
    for prop in ("checked", "selected", "expanded", "level", "disabled"):
        if prop in node:
            parts.append(f"{prop}={node[prop]}")

    lines.append(f"{prefix}{' '.join(parts)}")

    for child in node.get("children", []):
        lines.append(_format_aria_tree(child, indent + 1))

    return "\n".join(lines)


def _format_ai_tree(node: Dict, indent: int = 0) -> str:
    """Format accessibility tree in simplified AI format with action hints."""
    lines = []
    prefix = "  " * indent
    role = node.get("role", "")
    name = node.get("name", "")

    # Skip structural-only nodes
    skip_roles = {"generic", "none", "presentation"}
    if role in skip_roles and not name and not node.get("children"):
        return ""

    # Build actionable description
    actionable = ""
    if role in ("button", "link", "menuitem", "tab"):
        actionable = " [clickable]"
    elif role in ("textbox", "searchbox", "combobox", "spinbutton"):
        actionable = " [editable]"
    elif role in ("checkbox", "radio", "switch"):
        checked = node.get("checked", "")
        actionable = f" [toggleable, checked={checked}]"

    if name or actionable:
        label = f'\"{name}\"' if name else ""
        lines.append(f"{prefix}[{role}] {label}{actionable}")

    for child in node.get("children", []):
        child_text = _format_ai_tree(child, indent + 1)
        if child_text:
            lines.append(child_text)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Core Browser Service (enhanced)
# ──────────────────────────────────────────────────────────────

_browser = None
_playwright = None
_lock = asyncio.Lock()
_tab_manager = TabManager()
_active_profile = BrowserProfile.MANAGED


def get_tab_manager() -> TabManager:
    """Get the global tab manager."""
    return _tab_manager


async def _get_browser(profile: Optional[BrowserProfile] = None,
                       cdp_url: Optional[str] = None):
    """Lazy-init browser based on profile."""
    global _browser, _playwright, _active_profile
    profile = profile or _active_profile

    async with _lock:
        if _browser and _browser.is_connected():
            return _browser
        try:
            from playwright.async_api import async_playwright
            _playwright = await async_playwright().start()

            if profile == BrowserProfile.REMOTE and cdp_url:
                _browser = await _playwright.chromium.connect_over_cdp(cdp_url)
                logger.info("[BROWSER] Connected to remote CDP: %s", cdp_url)
            elif profile == BrowserProfile.CHROME:
                _browser = await _playwright.chromium.connect_over_cdp(
                    "http://localhost:9222"
                )
                logger.info("[BROWSER] Attached to Chrome via CDP")
            else:
                _browser = await _playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                logger.info("[BROWSER] Chromium launched (headless)")

            _active_profile = profile
            return _browser
        except ImportError:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        except Exception as e:
            logger.exception("[BROWSER] Failed to launch browser")
            raise


async def shutdown_browser():
    """Gracefully close the browser, all tabs, and Playwright instance."""
    global _browser, _playwright
    await _tab_manager.close_all()
    if _browser:
        await _browser.close()
        _browser = None
    if _playwright:
        await _playwright.stop()
        _playwright = None
    logger.info("[BROWSER] Shut down")


async def navigate(url: str, wait_until: str = "domcontentloaded",
                   timeout: int = 30, tab_id: Optional[str] = None) -> dict:
    """
    Navigate to a URL and return page info.

    If tab_id is provided, reuse that tab; otherwise create and close a temporary page.
    """
    browser = await _get_browser()

    if tab_id:
        page = _tab_manager.get_tab(tab_id)
        if not page:
            return {"error": f"Tab not found: {tab_id}"}
        resp = await page.goto(url, wait_until=wait_until, timeout=timeout * 1000)
        status = resp.status if resp else 0
        title = await page.title()
        return {"url": page.url, "title": title, "status": status, "tab_id": tab_id}

    page = await browser.new_page()
    try:
        resp = await page.goto(url, wait_until=wait_until, timeout=timeout * 1000)
        status = resp.status if resp else 0
        title = await page.title()
        return {"url": page.url, "title": title, "status": status}
    finally:
        await page.close()


async def screenshot(url: str = None, full_page: bool = False,
                     timeout: int = 30, tab_id: Optional[str] = None) -> str:
    """Take a screenshot. If tab_id provided, use that tab."""
    browser = await _get_browser()

    if tab_id:
        page = _tab_manager.get_tab(tab_id)
        if not page:
            return f"ERROR: Tab not found: {tab_id}"
    else:
        if not url:
            return "ERROR: Either url or tab_id required"
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        await page.screenshot(path=tmp.name, full_page=full_page)
        logger.info("[BROWSER] Screenshot saved: %s", tmp.name)
        return tmp.name
    finally:
        if not tab_id:
            await page.close()


async def extract_text(url: str = None, selector: Optional[str] = None,
                       timeout: int = 30, tab_id: Optional[str] = None) -> str:
    """Extract visible text from a page or a specific CSS selector."""
    browser = await _get_browser()

    if tab_id:
        page = _tab_manager.get_tab(tab_id)
        if not page:
            return f"ERROR: Tab not found: {tab_id}"
    else:
        if not url:
            return "ERROR: Either url or tab_id required"
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)

    try:
        if selector:
            el = await page.query_selector(selector)
            if not el:
                return f"ERROR: Selector not found: {selector}"
            text = await el.inner_text()
        else:
            text = await page.inner_text("body")

        if len(text) > 50_000:
            text = text[:50_000] + "\n\n[truncated]"
        return text
    finally:
        if not tab_id:
            await page.close()


async def run_action(
    url: str = None,
    action: str = "",
    selector: Optional[str] = None,
    value: Optional[str] = None,
    timeout: int = 30,
    tab_id: Optional[str] = None,
) -> str:
    """
    Perform an action on a page element.

    Actions:
    * click, fill, evaluate — original actions
    * snapshot — AI accessibility tree extraction (format: aria or ai)
    * tabs_list — list all open tabs
    * tab_open — open a new persistent tab
    * tab_close — close a tab by ID
    """
    # ── Tab management actions that don't need a page ──
    if action == "tabs_list":
        tabs = await _tab_manager.list_tabs_async()
        return json.dumps({"tabs": tabs, "count": len(tabs)}, indent=2)

    if action == "tab_close":
        target = tab_id or value
        if not target:
            return "ERROR: 'tab_id' or 'value' (tab ID) required for tab_close"
        ok = await _tab_manager.close_tab(target)
        return f"Closed tab {target}" if ok else f"Tab not found: {target}"

    # ── Actions requiring a browser ──
    if not action or action not in ("click", "fill", "evaluate", "snapshot", "tab_open"):
        if not url and not tab_id:
            return f"ERROR: Unknown action '{action}'. Use: click, fill, evaluate, snapshot, tabs_list, tab_open, tab_close"

    browser = await _get_browser()

    if action == "tab_open":
        tab_url = url or "about:blank"
        new_id = await _tab_manager.open_tab(browser, tab_url)
        return json.dumps({"tab_id": new_id, "url": tab_url})

    # ── Actions requiring a page ──
    if tab_id:
        page = _tab_manager.get_tab(tab_id)
        if not page:
            return f"ERROR: Tab not found: {tab_id}"
    else:
        if not url:
            return "ERROR: Either url or tab_id required"
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)

    try:
        if action == "snapshot":
            fmt = value or "aria"
            return await ai_snapshot(page, format=fmt)

        elif action == "click":
            if not selector:
                return "ERROR: 'selector' required for click"
            await page.click(selector, timeout=5000)
            return f"Clicked: {selector}"

        elif action == "fill":
            if not selector or value is None:
                return "ERROR: 'selector' and 'value' required for fill"
            await page.fill(selector, value, timeout=5000)
            return f"Filled {selector} with '{value[:50]}'"

        elif action == "evaluate":
            if not value:
                return "ERROR: 'value' (JavaScript code) required for evaluate"
            result = await page.evaluate(value)
            return str(result)[:10_000]

        else:
            return f"ERROR: Unknown action '{action}'. Use: click, fill, evaluate, snapshot, tabs_list, tab_open, tab_close"
    finally:
        if not tab_id:
            await page.close()
