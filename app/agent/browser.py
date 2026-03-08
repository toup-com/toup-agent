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

    Chromium doesn't support proxy auth (ERR_PROXY_AUTH_UNSUPPORTED).
    This spins up a local HTTP proxy on 127.0.0.1 (no auth) that adds the
    Proxy-Authorization header before forwarding to the real upstream proxy.
    """
    global _proxy_server, _proxy_port

    if _proxy_port is not None:
        return _proxy_port

    import base64
    from urllib.parse import urlparse

    parsed = urlparse(upstream_url)
    upstream_host = parsed.hostname
    upstream_port = parsed.port or 80
    auth_b64 = None
    if parsed.username:
        creds = f"{parsed.username}:{parsed.password or ''}"
        auth_b64 = base64.b64encode(creds.encode()).decode()
        logger.info("[PROXY-FWD] Upstream: %s:%d user=%s", upstream_host, upstream_port, parsed.username)
    else:
        logger.info("[PROXY-FWD] Upstream: %s:%d (no auth)", upstream_host, upstream_port)

    async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError, OSError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_client(client_reader: asyncio.StreamReader,
                             client_writer: asyncio.StreamWriter):
        up_writer = None
        try:
            # Read the request line (e.g. "CONNECT host:443 HTTP/1.1\r\n")
            first_line = await asyncio.wait_for(client_reader.readline(), timeout=30)
            if not first_line:
                client_writer.close()
                return

            is_connect = first_line.upper().startswith(b"CONNECT")
            target = first_line.split(b" ")[1].decode(errors="replace") if b" " in first_line else "?"
            logger.info("[PROXY-FWD] %s %s", "CONNECT" if is_connect else "HTTP", target)

            # Read remaining headers from browser
            raw_headers = bytearray(first_line)
            while True:
                line = await asyncio.wait_for(client_reader.readline(), timeout=30)
                raw_headers.extend(line)
                if line == b"\r\n" or not line:
                    break

            # Connect to upstream proxy (with retry)
            up_reader, up_writer = None, None
            for _retry in range(2):
                try:
                    up_reader, up_writer = await asyncio.wait_for(
                        asyncio.open_connection(upstream_host, upstream_port),
                        timeout=30,
                    )
                    break
                except Exception as e:
                    if _retry == 0:
                        logger.warning("[PROXY-FWD] Upstream connect failed, retrying: %s", e)
                        await asyncio.sleep(2)
                    else:
                        logger.error("[PROXY-FWD] Cannot connect to upstream %s:%d: %s", upstream_host, upstream_port, e)
            if not up_writer:
                client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await client_writer.drain()
                client_writer.close()
                return

            # Build request with auth header injected
            first_line_end = raw_headers.index(b"\r\n") + 2
            auth_line = f"Proxy-Authorization: Basic {auth_b64}\r\n".encode() if auth_b64 else b""
            modified = bytes(raw_headers[:first_line_end]) + auth_line + bytes(raw_headers[first_line_end:])

            up_writer.write(modified)
            await up_writer.drain()

            if is_connect:
                # Read upstream CONNECT response
                resp_first = await asyncio.wait_for(up_reader.readline(), timeout=30)
                up_response = bytearray(resp_first)
                while True:
                    resp_line = await asyncio.wait_for(up_reader.readline(), timeout=30)
                    up_response.extend(resp_line)
                    if resp_line == b"\r\n" or not resp_line:
                        break

                status_line = resp_first.decode(errors="replace").strip()
                # Check if upstream accepted the CONNECT
                if b"200" in resp_first:
                    logger.info("[PROXY-FWD] CONNECT %s → 200 OK", target)
                    # Send success to browser
                    client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    await client_writer.drain()
                else:
                    # Upstream rejected — DON'T forward 407 to browser
                    logger.error("[PROXY-FWD] CONNECT %s REJECTED: %s", target, status_line)
                    client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                    await client_writer.drain()
                    client_writer.close()
                    return
            # else: for regular HTTP, upstream sends the HTTP response directly

            # Bidirectional pipe
            await asyncio.gather(
                _pipe(client_reader, up_writer),
                _pipe(up_reader, client_writer),
            )
        except Exception as e:
            logger.error("[PROXY-FWD] Error: %s", e)
        finally:
            for w in [client_writer, up_writer]:
                if w:
                    try:
                        w.close()
                    except Exception:
                        pass

    server = await asyncio.start_server(_handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    _proxy_server = server
    _proxy_port = port
    logger.info("[PROXY-FWD] Listening on 127.0.0.1:%d", port)
    return port


# ──────────────────────────────────────────────────────────────
# Captcha Detection & Solving
# ──────────────────────────────────────────────────────────────

class CaptchaType(str, Enum):
    CLOUDFLARE_TURNSTILE = "cloudflare_turnstile"
    RECAPTCHA_V2 = "recaptcha_v2"
    HCAPTCHA = "hcaptcha"
    PERIMETERX = "perimeterx"
    DATADOME = "datadome"
    GENERIC = "generic"


async def detect_captcha(page) -> dict | None:
    """Detect captcha on the current page via DOM inspection. Returns dict with type info or None."""
    try:
        result = await page.evaluate("""() => {
            const body = document.body ? document.body.innerText : '';
            const title = document.title || '';
            const html = document.documentElement.innerHTML || '';

            // Cloudflare Turnstile
            if (document.querySelector('iframe[src*="challenges.cloudflare.com"]') ||
                document.querySelector('.cf-turnstile') ||
                document.querySelector('[data-sitekey]') && html.includes('cloudflare')) {
                return { type: 'cloudflare_turnstile' };
            }

            // Cloudflare interstitial ("Just a moment..." / "Checking your browser")
            if ((title.includes('Just a moment') || title.includes('Attention Required') ||
                 body.includes('Checking your browser') || body.includes('Enable JavaScript and cookies')) &&
                html.includes('cloudflare')) {
                return { type: 'cloudflare_turnstile' };
            }

            // reCAPTCHA v2
            if (document.querySelector('iframe[src*="google.com/recaptcha"]') ||
                document.querySelector('.g-recaptcha') ||
                document.querySelector('#recaptcha')) {
                return { type: 'recaptcha_v2' };
            }

            // hCaptcha
            if (document.querySelector('iframe[src*="hcaptcha.com"]') ||
                document.querySelector('.h-captcha')) {
                return { type: 'hcaptcha' };
            }

            // PerimeterX
            if (document.querySelector('#px-captcha') ||
                document.querySelector('iframe[src*="captcha.px-cdn.net"]') ||
                body.includes('Press & Hold') ||
                html.includes('_pxCaptcha')) {
                return { type: 'perimeterx' };
            }

            // DataDome
            if (document.querySelector('iframe[src*="captcha-delivery.com"]') ||
                document.querySelector('iframe[src*="interstitial-delivery.com"]') ||
                html.includes('datadome')) {
                return { type: 'datadome' };
            }

            // Generic challenge pages
            const challengeWords = ['verify you are human', 'are you a robot',
                'security check', 'access denied', 'please verify',
                'bot protection', 'human verification'];
            const lower = (title + ' ' + body.slice(0, 500)).toLowerCase();
            for (const w of challengeWords) {
                if (lower.includes(w)) {
                    return { type: 'generic' };
                }
            }

            return null;
        }""")
        if result:
            logger.info("[CAPTCHA] Detected: %s on %s", result["type"], page.url)
        return result
    except Exception as e:
        logger.debug("[CAPTCHA] Detection error: %s", e)
        return None


async def solve_captcha(page, captcha_info: dict, max_attempts: int = 2) -> bool:
    """Attempt to solve a detected captcha. Returns True if solved."""
    captcha_type = captcha_info.get("type", "")
    logger.info("[CAPTCHA] Attempting to solve: %s", captcha_type)

    for attempt in range(max_attempts):
        try:
            if captcha_type == CaptchaType.CLOUDFLARE_TURNSTILE:
                solved = await _solve_cloudflare(page)
            elif captcha_type == CaptchaType.PERIMETERX:
                solved = await _solve_perimeterx(page)
            elif captcha_type == CaptchaType.RECAPTCHA_V2:
                solved = await _solve_recaptcha_v2(page)
            elif captcha_type == CaptchaType.DATADOME:
                solved = await _solve_datadome(page)
            else:
                # Generic/hCaptcha — wait and check if it auto-resolves
                await asyncio.sleep(3)
                check = await detect_captcha(page)
                solved = check is None

            if solved:
                logger.info("[CAPTCHA] Solved %s on attempt %d", captcha_type, attempt + 1)
                return True

            logger.info("[CAPTCHA] Attempt %d failed for %s, retrying...", attempt + 1, captcha_type)
            await asyncio.sleep(2)
        except Exception as e:
            logger.error("[CAPTCHA] Error solving %s: %s", captcha_type, e)

    logger.warning("[CAPTCHA] Failed to solve %s after %d attempts", captcha_type, max_attempts)
    return False


async def _solve_cloudflare(page) -> bool:
    """Solve Cloudflare Turnstile / interstitial challenge."""
    # Wait for Turnstile to load
    await asyncio.sleep(3)

    # Try to find and click the Turnstile checkbox in its iframe
    try:
        turnstile_frame = page.frame_locator('iframe[src*="challenges.cloudflare.com"]')
        checkbox = turnstile_frame.locator('input[type="checkbox"], .ctp-checkbox-label, #challenge-stage')
        if await checkbox.count() > 0:
            box = await checkbox.first.bounding_box()
            if box:
                await human_click(page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                logger.info("[CAPTCHA] Clicked Turnstile checkbox")
    except Exception as e:
        logger.debug("[CAPTCHA] Turnstile iframe click: %s", e)

    # Also try clicking any visible verify button on the main page
    try:
        for selector in ['#challenge-stage input', 'button:has-text("Verify")', 'input[type="button"][value*="Verify"]']:
            btn = page.locator(selector)
            if await btn.count() > 0:
                box = await btn.first.bounding_box()
                if box:
                    await human_click(page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    break
    except Exception:
        pass

    # Wait for the page to pass the challenge (URL change or content change)
    await asyncio.sleep(5)
    check = await detect_captcha(page)
    return check is None


async def _solve_perimeterx(page) -> bool:
    """Solve PerimeterX press-and-hold challenge."""
    try:
        # Find the press-and-hold button
        px_btn = page.locator('#px-captcha')
        if await px_btn.count() == 0:
            px_btn = page.locator('[aria-label*="hold"], button:has-text("Hold")')
        if await px_btn.count() == 0:
            return False

        box = await px_btn.first.bounding_box()
        if not box:
            return False

        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

        # Move to button naturally
        await page.mouse.move(cx, cy, steps=random.randint(8, 15))
        await asyncio.sleep(random.uniform(0.2, 0.5))

        # Press and hold for 8-12 seconds with micro-movements
        await page.mouse.down()
        hold_time = random.uniform(8, 12)
        elapsed = 0
        while elapsed < hold_time:
            # Slight hand tremor during hold
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            await page.mouse.move(cx + dx, cy + dy)
            sleep_t = random.uniform(0.1, 0.3)
            await asyncio.sleep(sleep_t)
            elapsed += sleep_t
        await page.mouse.up()
        logger.info("[CAPTCHA] PerimeterX hold released after %.1fs", hold_time)

        # Wait for page to reload
        await asyncio.sleep(5)
        check = await detect_captcha(page)
        return check is None
    except Exception as e:
        logger.error("[CAPTCHA] PerimeterX error: %s", e)
        return False


async def _solve_recaptcha_v2(page) -> bool:
    """Solve reCAPTCHA v2 — click checkbox, if image challenge appears use vision."""
    try:
        # Find and click "I'm not a robot" checkbox
        recaptcha_frame = page.frame_locator('iframe[src*="google.com/recaptcha"][title*="reCAPTCHA"]')
        checkbox = recaptcha_frame.locator('.recaptcha-checkbox-border, #recaptcha-anchor')
        if await checkbox.count() > 0:
            box = await checkbox.first.bounding_box()
            if box:
                await human_click(page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                logger.info("[CAPTCHA] Clicked reCAPTCHA checkbox")
                await asyncio.sleep(3)

        # Check if we passed (sometimes checkbox click alone works)
        check = await detect_captcha(page)
        if check is None:
            return True

        # Image challenge appeared — try vision solving
        # Find the challenge iframe
        challenge_frame = page.frame_locator('iframe[src*="google.com/recaptcha"][title*="challenge"]')
        if await challenge_frame.locator('body').count() == 0:
            return False

        # Take screenshot of the challenge for vision solving
        for round_num in range(3):
            try:
                # Get challenge instruction text
                instruction = await challenge_frame.locator('.rc-imageselect-desc, .rc-imageselect-desc-wrapper').inner_text()
                logger.info("[CAPTCHA] reCAPTCHA challenge: %s", instruction)

                # Screenshot the challenge area
                challenge_el = challenge_frame.locator('.rc-imageselect-challenge, #rc-imageselect')
                screenshot = await challenge_el.first.screenshot()

                # Send to Claude Vision for solving
                solution = await _vision_solve_image_captcha(screenshot, instruction)
                if not solution:
                    return False

                # Click the identified tiles
                tiles = challenge_frame.locator('.rc-imageselect-tile, td[role="button"]')
                tile_count = await tiles.count()
                for idx in solution:
                    if 0 < idx <= tile_count:
                        tile_box = await tiles.nth(idx - 1).bounding_box()
                        if tile_box:
                            await human_click(page,
                                tile_box["x"] + tile_box["width"] / 2,
                                tile_box["y"] + tile_box["height"] / 2)
                            await asyncio.sleep(random.uniform(0.3, 0.6))

                # Click verify
                verify_btn = challenge_frame.locator('#recaptcha-verify-button, button:has-text("Verify")')
                if await verify_btn.count() > 0:
                    vbox = await verify_btn.first.bounding_box()
                    if vbox:
                        await human_click(page, vbox["x"] + vbox["width"] / 2, vbox["y"] + vbox["height"] / 2)

                await asyncio.sleep(3)
                check = await detect_captcha(page)
                if check is None:
                    return True
                # New images might have appeared — continue loop
            except Exception as e:
                logger.debug("[CAPTCHA] reCAPTCHA round %d error: %s", round_num, e)
                break

        return False
    except Exception as e:
        logger.error("[CAPTCHA] reCAPTCHA error: %s", e)
        return False


async def _solve_datadome(page) -> bool:
    """Solve DataDome challenge — wait for auto-resolution or click slider."""
    try:
        await asyncio.sleep(4)

        # DataDome sometimes has a slider
        dd_frame = page.frame_locator('iframe[src*="captcha-delivery.com"], iframe[src*="interstitial"]')
        slider = dd_frame.locator('.captcha-slider, .dd-slider, input[type="range"]')
        if await slider.count() > 0:
            box = await slider.first.bounding_box()
            if box:
                # Drag slider from left to right
                start_x = box["x"] + 5
                end_x = box["x"] + box["width"] - 5
                y = box["y"] + box["height"] / 2
                await page.mouse.move(start_x, y)
                await page.mouse.down()
                steps = random.randint(15, 25)
                for i in range(steps):
                    x = start_x + (end_x - start_x) * (i + 1) / steps
                    await page.mouse.move(x + random.gauss(0, 0.5), y + random.gauss(0, 0.3))
                    await asyncio.sleep(random.uniform(0.02, 0.06))
                await page.mouse.up()
                logger.info("[CAPTCHA] DataDome slider dragged")

        await asyncio.sleep(4)
        check = await detect_captcha(page)
        return check is None
    except Exception as e:
        logger.error("[CAPTCHA] DataDome error: %s", e)
        return False


async def _vision_solve_image_captcha(screenshot_bytes: bytes, instruction: str) -> list[int] | None:
    """Use Claude Vision to solve an image captcha. Returns list of tile indices (1-indexed)."""
    try:
        import base64
        import httpx

        from app.config import settings
        api_key = getattr(settings, "anthropic_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("[CAPTCHA] No ANTHROPIC_API_KEY for vision solving")
            return None

        img_b64 = base64.b64encode(screenshot_bytes).decode()
        prompt = (
            f"You are solving an image CAPTCHA challenge. The instruction says: \"{instruction}\"\n"
            "The image shows a grid of numbered tiles (numbered 1 to N, left-to-right, top-to-bottom).\n"
            "Return ONLY a JSON array of the tile numbers that match the instruction.\n"
            "Example: [1, 4, 7]\n"
            "If you cannot determine the answer, return []."
        )

        # OAuth tokens (sk-ant-oat*) use Bearer auth, regular keys use x-api-key
        _is_oauth = "sk-ant-oat" in api_key
        _headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if _is_oauth:
            _headers["authorization"] = f"Bearer {api_key}"
        else:
            _headers["x-api-key"] = api_key

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=_headers,
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                },
            )
            data = resp.json()
            text = data.get("content", [{}])[0].get("text", "")
            logger.info("[CAPTCHA] Vision response: %s", text)

            # Parse JSON array from response
            import re
            match = re.search(r'\[[\d,\s]*\]', text)
            if match:
                return json.loads(match.group())
            return None
    except Exception as e:
        logger.error("[CAPTCHA] Vision solving error: %s", e)
        return None


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

    // ─── Window dimensions (headless reports 0) ───
    Object.defineProperty(window, 'outerWidth', { get: () => 1280 });
    Object.defineProperty(window, 'outerHeight', { get: () => 720 });
    Object.defineProperty(window, 'innerWidth', { get: () => 1280 });
    Object.defineProperty(window, 'innerHeight', { get: () => 720 });

    // ─── Screen properties (match Xvfb display) ───
    Object.defineProperty(screen, 'width', { get: () => 1280 });
    Object.defineProperty(screen, 'height', { get: () => 720 });
    Object.defineProperty(screen, 'availWidth', { get: () => 1280 });
    Object.defineProperty(screen, 'availHeight', { get: () => 720 });
    Object.defineProperty(screen, 'colorDepth', { get: () => 24 });
    Object.defineProperty(screen, 'pixelDepth', { get: () => 24 });

    // ─── AudioContext fingerprint noise ───
    try {
        const origGetChannelData = AudioBuffer.prototype.getChannelData;
        AudioBuffer.prototype.getChannelData = function(channel) {
            const data = origGetChannelData.call(this, channel);
            for (let i = 0; i < Math.min(data.length, 10); i++) {
                data[i] += (Math.random() - 0.5) * 0.0001;
            }
            return data;
        };
    } catch(e) {}

    // ─── Battery API (headless often lacks this) ───
    if (!navigator.getBattery) {
        navigator.getBattery = () => Promise.resolve({
            charging: true, chargingTime: 0, dischargingTime: Infinity, level: 1,
            addEventListener: () => {}, removeEventListener: () => {},
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
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                # Timeout is OK — page may have partially loaded, continue anyway
                logger.warning("[BROWSER] open_tab goto timeout for %s: %s (continuing)", url, e)
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
                proxy_val = getattr(settings, "browser_proxy", "") or ""
                if proxy_val.strip():
                    logger.info("[BROWSER] BROWSER_PROXY is set (%d chars), starting forwarder...", len(proxy_val))
                    local_port = await _start_proxy_forwarder(proxy_val.strip())
                    # Quick connectivity test — verify upstream proxy accepts our credentials
                    proxy_works = False
                    try:
                        test_r, test_w = await asyncio.wait_for(
                            asyncio.open_connection("127.0.0.1", local_port), timeout=5
                        )
                        test_w.write(b"CONNECT httpbin.org:443 HTTP/1.1\r\nHost: httpbin.org:443\r\n\r\n")
                        await test_w.drain()
                        resp = await asyncio.wait_for(test_r.readline(), timeout=10)
                        resp_str = resp.decode(errors="replace").strip()
                        logger.info("[BROWSER] Proxy test response: %s", resp_str)
                        test_w.close()
                        proxy_works = b"200" in resp
                    except Exception as e:
                        logger.error("[BROWSER] Proxy test FAILED: %s", e)

                    if proxy_works:
                        browser_proxy_arg = {"server": f"http://127.0.0.1:{local_port}"}
                        logger.info("[BROWSER] Proxy OK — browser will use residential proxy")
                    else:
                        logger.warning("[BROWSER] Proxy credentials REJECTED — launching WITHOUT proxy (captchas possible)")
                else:
                    logger.info("[BROWSER] No BROWSER_PROXY set, launching without proxy")

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
