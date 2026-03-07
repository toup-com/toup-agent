"""
Browser Automation Service — Real Chromium browser with anti-detect patches.

Uses a REAL headed Chromium browser (not headless) for maximum anti-detection.
On headless servers (VPS), automatically starts Xvfb virtual display.
Combined with Patchright (CDP-level anti-detect), this is virtually undetectable.

Features:
  * Real Chromium browser (headed mode via Xvfb on VPS)
  * Patchright CDP-level anti-detection (bypasses Cloudflare, Google captchas)
  * Stealth fingerprinting (realistic plugins, WebGL, canvas, navigator)
  * AI Snapshot (accessibility tree extraction for stable agent actions)
  * Tab Management (keep pages open, list/switch/close tabs)
  * Browser Profiles (managed, chrome, remote CDP)
  * Persistent browser context with cookies
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import shutil
import subprocess
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Xvfb — Virtual display for headed browser on headless servers
# ──────────────────────────────────────────────────────────────

_xvfb_proc: subprocess.Popen | None = None
_xvfb_display: str | None = None


def _has_display() -> bool:
    """Check if a real display is available (desktop or existing Xvfb)."""
    display = os.environ.get("DISPLAY")
    if display:
        return True
    # macOS always has a display (even without DISPLAY env var)
    import sys
    if sys.platform == "darwin":
        return True
    return False


def _start_xvfb() -> str | None:
    """Start Xvfb virtual display if no display available. Returns display string."""
    global _xvfb_proc, _xvfb_display

    if _xvfb_display:
        return _xvfb_display

    if _has_display():
        logger.info("[BROWSER] Real display detected, no Xvfb needed")
        return os.environ.get("DISPLAY")

    # Check if Xvfb is installed
    xvfb_path = shutil.which("Xvfb")
    if not xvfb_path:
        logger.warning("[BROWSER] Xvfb not found — will use headless mode. Install with: apt-get install -y xvfb")
        return None

    # Find a free display number
    for display_num in range(99, 199):
        lock_file = f"/tmp/.X{display_num}-lock"
        if not os.path.exists(lock_file):
            break
    else:
        display_num = 99

    display = f":{display_num}"

    try:
        _xvfb_proc = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1280x720x24", "-ac", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give Xvfb a moment to start
        import time
        time.sleep(0.5)

        if _xvfb_proc.poll() is not None:
            logger.warning("[BROWSER] Xvfb failed to start")
            _xvfb_proc = None
            return None

        os.environ["DISPLAY"] = display
        _xvfb_display = display
        logger.info("[BROWSER] Xvfb started on display %s (real Chromium mode)", display)
        return display
    except Exception as e:
        logger.warning("[BROWSER] Failed to start Xvfb: %s", e)
        return None


def _stop_xvfb():
    """Stop Xvfb if we started it."""
    global _xvfb_proc, _xvfb_display
    if _xvfb_proc:
        try:
            _xvfb_proc.terminate()
            _xvfb_proc.wait(timeout=5)
        except Exception:
            try:
                _xvfb_proc.kill()
            except Exception:
                pass
        _xvfb_proc = None
        _xvfb_display = None
        logger.info("[BROWSER] Xvfb stopped")


# ──────────────────────────────────────────────────────────────
# Local Proxy Forwarder — handles proxy auth so Chromium doesn't have to
# ──────────────────────────────────────────────────────────────

_proxy_server: asyncio.AbstractServer | None = None
_proxy_port: int | None = None


async def _start_proxy_forwarder(upstream_url: str) -> int:
    """
    Start a local TCP proxy that forwards to an authenticated upstream proxy.

    Chromium doesn't support proxy auth in headed mode (ERR_PROXY_AUTH_UNSUPPORTED).
    This spins up a local proxy on 127.0.0.1 (no auth) that injects the
    Proxy-Authorization header before forwarding to the real proxy.

    Returns the local port number.
    """
    global _proxy_server, _proxy_port

    if _proxy_port is not None:
        return _proxy_port

    import base64
    from urllib.parse import urlparse

    parsed = urlparse(upstream_url)
    upstream_host = parsed.hostname
    upstream_port = parsed.port or 80
    auth_header = None
    if parsed.username:
        creds = f"{parsed.username}:{parsed.password or ''}"
        auth_header = f"Proxy-Authorization: Basic {base64.b64encode(creds.encode()).decode()}\r\n"

    async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_client(client_reader: asyncio.StreamReader,
                             client_writer: asyncio.StreamWriter):
        try:
            # Read the request line (e.g. "CONNECT host:443 HTTP/1.1\r\n")
            first_line = await asyncio.wait_for(client_reader.readline(), timeout=30)
            if not first_line:
                client_writer.close()
                return

            # Read remaining headers
            headers = bytearray(first_line)
            while True:
                line = await asyncio.wait_for(client_reader.readline(), timeout=30)
                headers.extend(line)
                if line == b"\r\n" or not line:
                    break

            # Connect to upstream proxy
            up_reader, up_writer = await asyncio.open_connection(
                upstream_host, upstream_port
            )

            # Inject auth header into the request to the upstream proxy
            if auth_header:
                # Insert Proxy-Authorization after the first line
                first_line_end = headers.index(b"\r\n") + 2
                modified = (
                    bytes(headers[:first_line_end])
                    + auth_header.encode()
                    + bytes(headers[first_line_end:])
                )
            else:
                modified = bytes(headers)

            up_writer.write(modified)
            await up_writer.drain()

            # For CONNECT tunnels, check upstream response first
            if first_line.upper().startswith(b"CONNECT"):
                # Read upstream proxy response
                up_response = bytearray()
                while True:
                    resp_line = await asyncio.wait_for(up_reader.readline(), timeout=30)
                    up_response.extend(resp_line)
                    if resp_line == b"\r\n" or not resp_line:
                        break
                # Forward the response to client
                client_writer.write(bytes(up_response))
                await client_writer.drain()

            # Bidirectional pipe
            await asyncio.gather(
                _pipe(client_reader, up_writer),
                _pipe(up_reader, client_writer),
            )
        except Exception as e:
            logger.debug("[PROXY-FWD] Connection error: %s", e)
        finally:
            try:
                client_writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(_handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    _proxy_server = server
    _proxy_port = port
    logger.info(
        "[PROXY-FWD] Local forwarder on :%d → %s:%d (auth injected)",
        port, upstream_host, upstream_port,
    )
    return port


# ──────────────────────────────────────────────────────────────
# Browser Profiles
# ──────────────────────────────────────────────────────────────

class BrowserProfile(str, Enum):
    """Browser launch profiles."""
    MANAGED = "managed"       # Playwright-managed headless Chromium
    CHROME = "chrome"         # Attach to user's Chrome via CDP
    REMOTE = "remote"         # Connect to a remote CDP endpoint


# ──────────────────────────────────────────────────────────────
# Stealth configuration — matches toup-browser app's engine.py
# ──────────────────────────────────────────────────────────────

STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-infobars",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-features=TranslateUI,AutomationControlled,OptimizationHints",
    "--disable-ipc-flooding-protection",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-extensions",
    "--disable-hang-monitor",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-sync",
    "--disable-translate",
    "--metrics-recording-only",
    "--no-first-run",
    "--password-store=basic",
    "--use-mock-keychain",
    "--enable-features=NetworkService,NetworkServiceInProcess",
    "--lang=en-US,en",
    "--window-size=1280,720",
]

STEALTH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/137.0.0.0 Safari/537.36"
)

# JS to inject into every page to evade bot detection — comprehensive stealth
STEALTH_INIT_SCRIPT = """
() => {
    // ─── webdriver ───
    Object.defineProperty(navigator, 'webdriver', { get: () => false });
    // Also delete it from the prototype
    delete Object.getPrototypeOf(navigator).webdriver;

    // ─── navigator overrides ───
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    Object.defineProperty(navigator, 'platform', { get: () => 'MacIntel' });
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
    Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
    Object.defineProperty(navigator, 'maxTouchPoints', { get: () => 0 });
    Object.defineProperty(navigator, 'vendor', { get: () => 'Google Inc.' });

    // ─── plugins (realistic PluginArray) ───
    const makeFakePlugin = (name, desc, filename) => {
        const p = { name, description: desc, filename, length: 1 };
        p[0] = { type: 'application/x-google-chrome-pdf', suffixes: 'pdf', description: 'Portable Document Format' };
        Object.setPrototypeOf(p, Plugin.prototype);
        return p;
    };
    const fakePlugins = [
        makeFakePlugin('Chrome PDF Plugin', 'Portable Document Format', 'internal-pdf-viewer'),
        makeFakePlugin('Chrome PDF Viewer', '', 'mhjfbmdgcfjbbpaeojofohoefgiehjai'),
        makeFakePlugin('Native Client', '', 'internal-nacl-plugin'),
    ];
    Object.defineProperty(navigator, 'plugins', {
        get: () => {
            const arr = fakePlugins;
            arr.item = (i) => arr[i] || null;
            arr.namedItem = (n) => arr.find(p => p.name === n) || null;
            arr.refresh = () => {};
            Object.setPrototypeOf(arr, PluginArray.prototype);
            return arr;
        },
    });
    Object.defineProperty(navigator, 'mimeTypes', {
        get: () => {
            const mt = [{ type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format', enabledPlugin: fakePlugins[0] }];
            mt.item = (i) => mt[i] || null;
            mt.namedItem = (n) => mt.find(m => m.type === n) || null;
            Object.setPrototypeOf(mt, MimeTypeArray.prototype);
            return mt;
        },
    });

    // ─── chrome runtime ───
    if (!window.chrome) window.chrome = {};
    window.chrome.runtime = { OnInstalledReason: {}, OnRestartRequiredReason: {}, PlatformArch: {}, PlatformNaclArch: {}, PlatformOs: {}, RequestUpdateCheckStatus: {}, connect: function(){}, sendMessage: function(){} };
    window.chrome.loadTimes = function() { return { commitLoadTime: Date.now()/1000, connectionInfo: 'h2', finishDocumentLoadTime: Date.now()/1000, finishLoadTime: Date.now()/1000, firstPaintAfterLoadTime: 0, firstPaintTime: Date.now()/1000, navigationType: 'Other', npnNegotiatedProtocol: 'h2', requestTime: Date.now()/1000 - 0.5, startLoadTime: Date.now()/1000 - 0.5, wasAlternateProtocolAvailable: false, wasFetchedViaSpdy: true, wasNpnNegotiated: true }; };
    window.chrome.csi = function() { return { onloadT: Date.now(), pageT: Date.now() - performance.timing.navigationStart, startE: performance.timing.navigationStart, tran: 15 }; };
    window.chrome.app = { isInstalled: false, InstallState: { DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' }, RunningState: { CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running' } };

    // ─── permissions ───
    const origQuery = window.navigator.permissions.query.bind(window.navigator.permissions);
    window.navigator.permissions.query = (params) => {
        if (params.name === 'notifications') return Promise.resolve({ state: Notification.permission });
        if (params.name === 'push') return Promise.resolve({ state: 'prompt' });
        if (params.name === 'midi' || params.name === 'camera' || params.name === 'microphone') return Promise.resolve({ state: 'prompt' });
        return origQuery(params).catch(() => Promise.resolve({ state: 'prompt' }));
    };

    // ─── WebGL vendor/renderer ───
    const getParam = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {
        if (p === 37445) return 'Intel Inc.';
        if (p === 37446) return 'Intel Iris OpenGL Engine';
        return getParam.call(this, p);
    };
    // WebGL2
    if (typeof WebGL2RenderingContext !== 'undefined') {
        const getParam2 = WebGL2RenderingContext.prototype.getParameter;
        WebGL2RenderingContext.prototype.getParameter = function(p) {
            if (p === 37445) return 'Intel Inc.';
            if (p === 37446) return 'Intel Iris OpenGL Engine';
            return getParam2.call(this, p);
        };
    }

    // ─── canvas fingerprint noise ───
    const origToDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function(type) {
        if (type === 'image/png' || !type) {
            const ctx = this.getContext('2d');
            if (ctx) {
                const style = ctx.fillStyle;
                ctx.fillStyle = 'rgba(255,255,255,0.01)';
                ctx.fillRect(0, 0, 1, 1);
                ctx.fillStyle = style;
            }
        }
        return origToDataURL.apply(this, arguments);
    };
    const origToBlob = HTMLCanvasElement.prototype.toBlob;
    HTMLCanvasElement.prototype.toBlob = function(cb, type, quality) {
        if (type === 'image/png' || !type) {
            const ctx = this.getContext('2d');
            if (ctx) {
                const style = ctx.fillStyle;
                ctx.fillStyle = 'rgba(255,255,255,0.01)';
                ctx.fillRect(0, 0, 1, 1);
                ctx.fillStyle = style;
            }
        }
        return origToBlob.apply(this, arguments);
    };

    // ─── Notification mock ───
    if (!window.Notification) {
        window.Notification = { permission: 'default', requestPermission: () => Promise.resolve('default') };
    }

    // ─── iframe contentWindow trap ───
    try {
        const origContentWindow = Object.getOwnPropertyDescriptor(HTMLIFrameElement.prototype, 'contentWindow');
        Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
            get: function() {
                const w = origContentWindow.get.call(this);
                if (w) {
                    try { Object.defineProperty(w.navigator, 'webdriver', { get: () => false }); } catch(e) {}
                }
                return w;
            },
        });
    } catch(e) {}

    // ─── Connection / Network Information ───
    if (!navigator.connection) {
        Object.defineProperty(navigator, 'connection', {
            get: () => ({ effectiveType: '4g', rtt: 50, downlink: 10, saveData: false }),
        });
    }
}
"""


# ──────────────────────────────────────────────────────────────
# Atlas-style Human Interaction — Bezier mouse, realistic typing
# ──────────────────────────────────────────────────────────────

def _bezier_curve(
    start: Tuple[float, float],
    end: Tuple[float, float],
    steps: int = 0,
) -> List[Tuple[float, float]]:
    """Generate a human-like Bezier curve path between two points.

    Uses 2 random control points offset from the straight line to create
    a natural-looking mouse movement path, similar to how ChatGPT Atlas
    moves its virtual cursor.
    """
    sx, sy = start
    ex, ey = end
    dist = math.hypot(ex - sx, ey - sy)

    # More steps for longer distances, fewer for short ones
    if steps <= 0:
        steps = max(12, min(35, int(dist / 15)))

    # Two random control points — offset perpendicular to the line
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    spread = dist * random.uniform(0.15, 0.4)
    angle = math.atan2(ey - sy, ex - sx) + math.pi / 2

    # Control point 1: biased toward start
    c1x = sx + (mx - sx) * random.uniform(0.2, 0.5) + math.cos(angle) * spread * random.choice([-1, 1])
    c1y = sy + (my - sy) * random.uniform(0.2, 0.5) + math.sin(angle) * spread * random.choice([-1, 1])

    # Control point 2: biased toward end
    c2x = ex - (ex - mx) * random.uniform(0.2, 0.5) + math.cos(angle) * spread * random.choice([-1, 1]) * 0.5
    c2y = ey - (ey - my) * random.uniform(0.2, 0.5) + math.sin(angle) * spread * random.choice([-1, 1]) * 0.5

    points = []
    for i in range(steps + 1):
        t = i / steps
        # Add slight timing jitter (ease-in-out feel)
        t = t * t * (3 - 2 * t)  # smoothstep
        u = 1 - t
        # Cubic Bezier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        x = u**3 * sx + 3 * u**2 * t * c1x + 3 * u * t**2 * c2x + t**3 * ex
        y = u**3 * sy + 3 * u**2 * t * c1y + 3 * u * t**2 * c2y + t**3 * ey
        # Add tiny noise (human hand tremor)
        if 0 < i < steps:
            x += random.gauss(0, 0.8)
            y += random.gauss(0, 0.8)
        points.append((round(x, 1), round(y, 1)))

    return points


async def human_move_mouse(page, from_pos: Tuple[float, float], to_pos: Tuple[float, float]):
    """Move mouse along a Bezier curve path — indistinguishable from real human movement."""
    points = _bezier_curve(from_pos, to_pos)
    for x, y in points:
        await page.mouse.move(x, y)
        # Variable speed: slower at start/end, faster in middle (like a real hand)
        await asyncio.sleep(random.uniform(0.004, 0.018))


async def human_click(page, x: float, y: float, from_pos: Tuple[float, float] = (640, 360)):
    """Move mouse naturally to coordinates, then click with realistic timing."""
    await human_move_mouse(page, from_pos, (x, y))
    # Small pre-click pause (human reaction time)
    await asyncio.sleep(random.uniform(0.05, 0.15))
    # Realistic click: mousedown → short hold → mouseup
    await page.mouse.down()
    await asyncio.sleep(random.uniform(0.04, 0.12))
    await page.mouse.up()
    # Small post-click settle
    await asyncio.sleep(random.uniform(0.08, 0.2))


async def human_type(page, text: str, selector: str = None):
    """Type text character by character with human-like variable delays.

    If selector is provided, clicks the element first (using coordinates).
    Mimics real typing: faster for common letters, slower for shifts/specials,
    occasional micro-pauses between words.
    """
    for i, char in enumerate(text):
        await page.keyboard.press(char if len(char) == 1 else char)
        # Variable delay per character
        if char == " ":
            # Slightly longer pause between words
            await asyncio.sleep(random.uniform(0.06, 0.16))
        elif char in ".,!?;:":
            # Longer pause after punctuation
            await asyncio.sleep(random.uniform(0.1, 0.25))
        elif char.isupper() or char in "!@#$%^&*()":
            # Shift key = slower
            await asyncio.sleep(random.uniform(0.05, 0.12))
        else:
            # Normal typing speed: 40-80ms per char (realistic WPM range)
            await asyncio.sleep(random.uniform(0.035, 0.09))

        # Occasional micro-pause (thinking/hesitation) — ~5% chance
        if random.random() < 0.05:
            await asyncio.sleep(random.uniform(0.15, 0.4))


async def human_scroll(page, direction: str = "down", amount: int = 500):
    """Scroll with realistic incremental wheel events instead of one big jump."""
    total = 0
    while total < amount:
        # Each wheel tick: 40-120px (like a real mouse wheel)
        tick = random.randint(40, 120)
        tick = min(tick, amount - total)
        delta = tick if direction == "down" else -tick
        await page.mouse.wheel(0, delta)
        total += tick
        await asyncio.sleep(random.uniform(0.03, 0.08))
    # Settle after scrolling
    await asyncio.sleep(random.uniform(0.1, 0.3))


async def get_element_center(page, selector: str) -> Optional[Tuple[float, float]]:
    """Get the center coordinates of an element by CSS selector."""
    try:
        box = await page.evaluate("""(sel) => {
            const el = document.querySelector(sel);
            if (!el) return null;
            const r = el.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) return null;
            return {
                x: r.left + r.width / 2 + (Math.random() - 0.5) * Math.min(r.width * 0.3, 8),
                y: r.top + r.height / 2 + (Math.random() - 0.5) * Math.min(r.height * 0.3, 4)
            };
        }""", selector)
        if box:
            return (box["x"], box["y"])
    except Exception:
        pass
    return None


async def get_text_element_center(page, text: str) -> Optional[Tuple[float, float]]:
    """Get the center coordinates of an element by its visible text."""
    try:
        box = await page.evaluate("""(text) => {
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) {
                const node = walker.currentNode;
                if (node.textContent && node.textContent.trim().toLowerCase().includes(text.toLowerCase())) {
                    const el = node.parentElement;
                    if (el) {
                        const r = el.getBoundingClientRect();
                        if (r.width > 0 && r.height > 0) {
                            return {
                                x: r.left + r.width / 2 + (Math.random() - 0.5) * Math.min(r.width * 0.3, 8),
                                y: r.top + r.height / 2 + (Math.random() - 0.5) * Math.min(r.height * 0.3, 4)
                            };
                        }
                    }
                }
            }
            return null;
        }""", text)
        if box:
            return (box["x"], box["y"])
    except Exception:
        pass
    return None


async def get_element_center_by_index(page, index: int) -> Optional[Tuple[float, float]]:
    """Get center coordinates of an interactive element by its index."""
    try:
        box = await page.evaluate("""(idx) => {
            const SELECTOR = "a[href], button, input, select, textarea, [role='button'], [role='link'], [role='tab'], [onclick]";
            const els = Array.from(document.querySelectorAll(SELECTOR));
            const visible = els.filter(el => {
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0 && !el.id?.startsWith('__toup_');
            });
            if (idx < visible.length) {
                const r = visible[idx].getBoundingClientRect();
                return {
                    x: r.left + r.width / 2 + (Math.random() - 0.5) * Math.min(r.width * 0.3, 8),
                    y: r.top + r.height / 2 + (Math.random() - 0.5) * Math.min(r.height * 0.3, 4)
                };
            }
            return null;
        }""", index)
        if box:
            return (box["x"], box["y"])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────
# Tab Manager — Persistent page/tab lifecycle
# ──────────────────────────────────────────────────────────────

class TabManager:
    """Manages named browser tabs (pages) with persistence."""

    def __init__(self):
        self._tabs: Dict[str, Any] = {}  # tab_id → page
        self._counter: int = 0

    async def open_tab(self, context_or_browser, url: str = "about:blank",
                       viewport: Optional[Dict[str, int]] = None) -> str:
        """Open a new tab from the stealth context and return its ID."""
        page = await context_or_browser.new_page()
        if viewport:
            await page.set_viewport_size(viewport)
        if url and url != "about:blank":
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        self._counter += 1
        tab_id = f"tab_{self._counter}"
        self._tabs[tab_id] = page
        logger.info("[BROWSER] Opened tab %s → %s", tab_id, url)
        return tab_id

    async def close_tab(self, tab_id: str) -> bool:
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
        for tid in list(self._tabs.keys()):
            await self.close_tab(tid)

    def get_tab(self, tab_id: str) -> Optional[Any]:
        return self._tabs.get(tab_id)

    def list_tabs(self) -> List[Dict[str, str]]:
        result = []
        for tid, page in self._tabs.items():
            try:
                result.append({"tab_id": tid, "url": page.url, "title": ""})
            except Exception:
                result.append({"tab_id": tid, "url": "unknown", "title": ""})
        return result

    async def list_tabs_async(self) -> List[Dict[str, str]]:
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
    lines = []
    prefix = "  " * indent
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")
    parts = [role]
    if name:
        parts.append(f'"{name}"')
    if value:
        parts.append(f"value={value}")
    for prop in ("checked", "selected", "expanded", "level", "disabled"):
        if prop in node:
            parts.append(f"{prop}={node[prop]}")
    lines.append(f"{prefix}{' '.join(parts)}")
    for child in node.get("children", []):
        lines.append(_format_aria_tree(child, indent + 1))
    return "\n".join(lines)


def _format_ai_tree(node: Dict, indent: int = 0) -> str:
    lines = []
    prefix = "  " * indent
    role = node.get("role", "")
    name = node.get("name", "")
    skip_roles = {"generic", "none", "presentation"}
    if role in skip_roles and not name and not node.get("children"):
        return ""
    actionable = ""
    if role in ("button", "link", "menuitem", "tab"):
        actionable = " [clickable]"
    elif role in ("textbox", "searchbox", "combobox", "spinbutton"):
        actionable = " [editable]"
    elif role in ("checkbox", "radio", "switch"):
        checked = node.get("checked", "")
        actionable = f" [toggleable, checked={checked}]"
    if name or actionable:
        label = f'"{name}"' if name else ""
        lines.append(f"{prefix}[{role}] {label}{actionable}")
    for child in node.get("children", []):
        child_text = _format_ai_tree(child, indent + 1)
        if child_text:
            lines.append(child_text)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Core Browser Service — stealth-enhanced
# ──────────────────────────────────────────────────────────────

_browser = None
_context = None  # Stealth browser context (all pages share it)
_playwright = None
_lock = asyncio.Lock()
_tab_manager = TabManager()
_active_profile = BrowserProfile.MANAGED


def get_tab_manager() -> TabManager:
    return _tab_manager


async def get_stealth_context():
    """Get the shared stealth browser context. Creates browser if needed."""
    global _context
    await _get_browser()  # ensure browser is running
    if _context:
        return _context
    return _context


async def _get_browser(profile: Optional[BrowserProfile] = None,
                       cdp_url: Optional[str] = None):
    """Lazy-init browser with stealth settings."""
    global _browser, _context, _playwright, _active_profile
    profile = profile or _active_profile

    async with _lock:
        if _browser and _browser.is_connected():
            return _browser
        try:
            # Patchright = patched Playwright that bypasses CDP detection (Cloudflare, Google captchas)
            # Falls back to regular Playwright if not installed
            try:
                from patchright.async_api import async_playwright
                logger.info("[BROWSER] Using Patchright (anti-detect Playwright)")
            except ImportError:
                from playwright.async_api import async_playwright
                logger.info("[BROWSER] Patchright not found, using standard Playwright")
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
                # Strategy: Real headed Chromium > full Chromium + --headless=new > Xvfb headed
                # CRITICAL: Never use chromium_headless_shell — it's trivially detectable.
                # Instead, use the FULL Chromium binary with Chrome's own --headless=new flag.
                # This runs the complete browser engine headlessly = undetectable.
                real_chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
                display = _start_xvfb()
                use_headed = display is not None

                launch_args = list(STEALTH_ARGS)
                if not use_headed:
                    launch_args.append("--headless=new")

                # Proxy support — residential proxy eliminates datacenter IP detection
                # Chromium doesn't support proxy auth natively (ERR_PROXY_AUTH_UNSUPPORTED)
                # so we run a local forwarder that handles auth transparently
                from app.config import settings
                browser_proxy_arg = None
                if getattr(settings, "browser_proxy", ""):
                    local_port = await _start_proxy_forwarder(settings.browser_proxy)
                    browser_proxy_arg = {"server": f"http://127.0.0.1:{local_port}"}
                    logger.info("[BROWSER] Local proxy forwarder on port %d", local_port)

                if real_chrome:
                    _browser = await _playwright.chromium.launch(
                        executable_path=real_chrome,
                        headless=not use_headed,
                        args=launch_args,
                        proxy=browser_proxy_arg,
                    )
                    mode = "headed + Xvfb" if use_headed else "headless=new"
                    logger.info("[BROWSER] Real Chrome launched (%s): %s", mode, real_chrome)
                else:
                    _browser = await _playwright.chromium.launch(
                        headless=not use_headed,
                        args=launch_args,
                        proxy=browser_proxy_arg,
                    )
                    mode = "headed + Xvfb" if use_headed else "headless=new"
                    logger.info("[BROWSER] Chromium launched (%s + Patchright stealth)", mode)

            # Create a stealth context — all tabs share this context
            _context = await _browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=STEALTH_USER_AGENT,
                java_script_enabled=True,
                ignore_https_errors=True,
                locale="en-US",
                timezone_id="America/New_York",
                color_scheme="light",
            )
            _context.set_default_timeout(30_000)

            # Inject stealth scripts into every new page
            await _context.add_init_script(STEALTH_INIT_SCRIPT)
            logger.info("[BROWSER] Stealth context created")

            _active_profile = profile
            return _browser
        except ImportError:
            raise RuntimeError(
                "Browser not installed. Run: pip install patchright && patchright install chromium"
            )
        except Exception as e:
            logger.exception("[BROWSER] Failed to launch browser")
            raise


async def shutdown_browser():
    """Gracefully close the browser, all tabs, Playwright instance, and Xvfb."""
    global _browser, _context, _playwright
    await _tab_manager.close_all()
    if _context:
        await _context.close()
        _context = None
    if _browser:
        await _browser.close()
        _browser = None
    if _playwright:
        await _playwright.stop()
        _playwright = None
    _stop_xvfb()
    logger.info("[BROWSER] Shut down")


async def navigate(url: str, wait_until: str = "domcontentloaded",
                   timeout: int = 30, tab_id: Optional[str] = None) -> dict:
    browser = await _get_browser()
    ctx = await get_stealth_context()

    if tab_id:
        page = _tab_manager.get_tab(tab_id)
        if not page:
            return {"error": f"Tab not found: {tab_id}"}
        resp = await page.goto(url, wait_until=wait_until, timeout=timeout * 1000)
        status = resp.status if resp else 0
        title = await page.title()
        return {"url": page.url, "title": title, "status": status, "tab_id": tab_id}

    page = await ctx.new_page()
    try:
        resp = await page.goto(url, wait_until=wait_until, timeout=timeout * 1000)
        status = resp.status if resp else 0
        title = await page.title()
        return {"url": page.url, "title": title, "status": status}
    finally:
        await page.close()


async def screenshot(url: str = None, full_page: bool = False,
                     timeout: int = 30, tab_id: Optional[str] = None) -> str:
    browser = await _get_browser()
    ctx = await get_stealth_context()

    if tab_id:
        page = _tab_manager.get_tab(tab_id)
        if not page:
            return f"ERROR: Tab not found: {tab_id}"
    else:
        if not url:
            return "ERROR: Either url or tab_id required"
        page = await ctx.new_page()
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
    browser = await _get_browser()
    ctx = await get_stealth_context()

    if tab_id:
        page = _tab_manager.get_tab(tab_id)
        if not page:
            return f"ERROR: Tab not found: {tab_id}"
    else:
        if not url:
            return "ERROR: Either url or tab_id required"
        page = await ctx.new_page()
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
    if action == "tabs_list":
        tabs = await _tab_manager.list_tabs_async()
        return json.dumps({"tabs": tabs, "count": len(tabs)}, indent=2)

    if action == "tab_close":
        target = tab_id or value
        if not target:
            return "ERROR: 'tab_id' or 'value' (tab ID) required for tab_close"
        ok = await _tab_manager.close_tab(target)
        return f"Closed tab {target}" if ok else f"Tab not found: {target}"

    if not action or action not in ("click", "fill", "evaluate", "snapshot", "tab_open"):
        if not url and not tab_id:
            return f"ERROR: Unknown action '{action}'. Use: click, fill, evaluate, snapshot, tabs_list, tab_open, tab_close"

    browser = await _get_browser()
    ctx = await get_stealth_context()

    if action == "tab_open":
        tab_url = url or "about:blank"
        new_id = await _tab_manager.open_tab(ctx, tab_url)
        return json.dumps({"tab_id": new_id, "url": tab_url})

    if tab_id:
        page = _tab_manager.get_tab(tab_id)
        if not page:
            return f"ERROR: Tab not found: {tab_id}"
    else:
        if not url:
            return "ERROR: Either url or tab_id required"
        page = await ctx.new_page()
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
            return f"ERROR: Unknown action '{action}'"
    finally:
        if not tab_id:
            await page.close()
