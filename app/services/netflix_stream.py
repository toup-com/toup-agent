"""
Netflix Streaming — Chrome (Widevine) + FFmpeg → HLS.

Handles login, profile selection, and content playback automatically
via Chrome DevTools Protocol (CDP).
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

HLS_SEGMENT_DURATION = 1
HLS_PLAYLIST_SIZE = 5
FFMPEG_FRAMERATE = 24
FFMPEG_VIDEO_BITRATE = "2500k"


class NetflixStream:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.hls_dir = Path(tempfile.mkdtemp(prefix=f"nf-{stream_id}-"))
        self.display: Optional[str] = None
        self.cdp_port: Optional[int] = None
        self.xvfb_proc: Optional[asyncio.subprocess.Process] = None
        self.chrome_proc: Optional[asyncio.subprocess.Process] = None
        self.ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self.running = False

    async def start(self, search_query: str, email: str, password: str) -> str:
        """Start Netflix stream. Returns HLS dir path."""
        try:
            self.running = True

            # 1. Xvfb
            self.display = await self._start_xvfb()
            logger.info("[NF] Xvfb on %s", self.display)

            # 2. Setup PulseAudio
            await self._setup_pulse()

            # 3. Launch Chrome with CDP
            self.cdp_port = 9300 + hash(self.stream_id) % 100
            await self._start_chrome()
            logger.info("[NF] Chrome on CDP port %d", self.cdp_port)

            # 4. Automate Netflix login + navigate to content
            await self._netflix_login_and_play(search_query, email, password)
            logger.info("[NF] Netflix playing")

            # 5. FFmpeg capture → HLS
            await self._start_ffmpeg()
            logger.info("[NF] FFmpeg capturing")

            # Wait for first HLS segment
            playlist = self.hls_dir / "stream.m3u8"
            for _ in range(20):
                if playlist.exists() and playlist.stat().st_size > 0:
                    return str(playlist)
                await asyncio.sleep(1)
            raise RuntimeError("HLS not ready in time")

        except Exception as e:
            logger.exception("[NF] Start failed")
            await self.stop()
            raise

    async def stop(self):
        self.running = False
        # Stop FFmpeg
        if self.ffmpeg_proc and self.ffmpeg_proc.returncode is None:
            try:
                self.ffmpeg_proc.terminate()
                await asyncio.wait_for(self.ffmpeg_proc.wait(), timeout=5)
            except Exception:
                try: self.ffmpeg_proc.kill()
                except Exception: pass
        # Close Playwright browser
        if hasattr(self, '_browser') and self._browser:
            try: await self._browser.close()
            except Exception: pass
        if hasattr(self, '_playwright') and self._playwright:
            try: await self._playwright.stop()
            except Exception: pass
        # Stop Xvfb
        if self.xvfb_proc and self.xvfb_proc.returncode is None:
            try:
                self.xvfb_proc.terminate()
                await asyncio.wait_for(self.xvfb_proc.wait(), timeout=5)
            except Exception:
                try: self.xvfb_proc.kill()
                except Exception: pass
        await asyncio.sleep(1)
        shutil.rmtree(self.hls_dir, ignore_errors=True)

    # ── Infrastructure ─────────────────────────────────────────────

    async def _start_xvfb(self) -> str:
        for num in range(120, 160):
            if os.path.exists(f"/tmp/.X{num}-lock"):
                continue
            self.xvfb_proc = await asyncio.create_subprocess_exec(
                "Xvfb", f":{num}", "-screen", "0", "1280x720x24", "-ac",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.sleep(0.5)
            if self.xvfb_proc.returncode is None:
                return f":{num}"
        raise RuntimeError("No free display")

    async def _setup_pulse(self):
        env = {"HOME": "/root", "DISPLAY": self.display}
        os.makedirs("/root/.config/pulse", exist_ok=True)
        await (await asyncio.create_subprocess_exec(
            "pulseaudio", "--start", "--daemonize=yes",
            env={**os.environ, **env},
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )).wait()

    async def _start_chrome(self):
        """Launch Chrome via Playwright (Patchright) for proper automation."""
        try:
            from patchright.async_api import async_playwright
        except ImportError:
            from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        chrome = shutil.which("google-chrome-stable") or "/opt/google/chrome/google-chrome"
        data_dir = "/tmp/nf-chrome-shared"
        os.makedirs(data_dir, exist_ok=True)

        self._browser = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=data_dir,
            executable_path=chrome,
            headless=False,
            args=[
                "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-features=TranslateUI",
                f"--remote-debugging-port={self.cdp_port}",
            ],
            viewport={"width": 1280, "height": 720},
            no_viewport=False,
            ignore_https_errors=True,
        )
        self._page = self._browser.pages[0] if self._browser.pages else await self._browser.new_page()
        logger.info("[NF] Playwright Chrome launched")

    # ── Netflix Automation via CDP ─────────────────────────────────

    async def _cdp_send(self, ws_url: str, method: str, params: dict = None) -> dict:
        """Send a CDP command via WebSocket."""
        import websockets
        try:
            async with websockets.connect(ws_url, max_size=10_000_000, open_timeout=10) as ws:
                msg = {"id": 1, "method": method, "params": params or {}}
                await ws.send(json.dumps(msg))
                async for raw in ws:
                    resp = json.loads(raw)
                    if resp.get("id") == 1:
                        return resp.get("result", {})
        except Exception as e:
            logger.warning("[NF] CDP send failed (%s): %s", method, e)
            return {}

    async def _cdp_evaluate(self, ws_url: str, expression: str) -> any:
        """Evaluate JS in the browser."""
        result = await self._cdp_send(ws_url, "Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value")

    async def _cdp_navigate(self, ws_url: str, url: str):
        """Navigate to URL and wait for load."""
        await self._cdp_send(ws_url, "Page.navigate", {"url": url})
        await asyncio.sleep(3)

    async def _cdp_type_text(self, ws_url: str, selectors: str, text: str):
        """Focus an input and type text using CDP Input.insertText (works with React)."""
        sel_list = [s.strip() for s in selectors.split(",")]
        # Focus the first matching element
        for sel in sel_list:
            result = await self._cdp_evaluate(ws_url, f"""
                (function() {{
                    var el = document.querySelector('{sel}');
                    if (el) {{ el.focus(); el.value = ''; return true; }}
                    return false;
                }})();
            """)
            if result:
                break
        await asyncio.sleep(0.3)
        # Use insertText — works with React controlled inputs
        await self._cdp_send(ws_url, "Input.insertText", {"text": text})
        await asyncio.sleep(0.3)

    async def _cdp_click(self, ws_url: str, selectors: str):
        """Click first matching element from comma-separated selectors."""
        # Split selectors and try each one
        sel_list = [s.strip() for s in selectors.split(",")]
        for sel in sel_list:
            await self._cdp_evaluate(ws_url, f"""
                (function() {{
                    var el = document.querySelector('{sel}');
                    if (el) {{ el.click(); return true; }}
                    return false;
                }})();
            """)
        await asyncio.sleep(0.5)

    async def _get_page_ws(self) -> str:
        """Get the WebSocket URL for the first browser page (with retries)."""
        for attempt in range(10):
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"http://127.0.0.1:{self.cdp_port}/json", timeout=3)
                    pages = r.json()
                    for p in pages:
                        if p.get("type") == "page":
                            return p["webSocketDebuggerUrl"]
            except Exception:
                pass
            await asyncio.sleep(1)
        raise RuntimeError("No CDP page found after 10 retries")

    async def _netflix_login_and_play(self, query: str, email: str, password: str):
        """Login to Netflix and navigate to content using Playwright."""
        page = self._page

        # Navigate to Netflix
        await page.goto("https://www.netflix.com/browse", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        current_url = page.url
        logger.info("[NF] Current URL: %s", current_url)

        # Login if needed
        if "/login" in current_url:
            logger.info("[NF] Login required, filling credentials via Playwright...")

            # Fill email — try multiple selectors
            email_input = page.locator('input[name="userLoginId"], input[type="email"], input[data-uia="login-field"]').first
            await email_input.fill(email)
            await page.wait_for_timeout(500)

            # Click Continue/Sign In
            submit_btn = page.locator('button:has-text("Continue"), button:has-text("Sign In"), button[type="submit"]').first
            await submit_btn.click()
            await page.wait_for_timeout(4000)

            # Fill password (may be on new page or same page)
            pw_input = page.locator('input[name="password"], input[type="password"]').first
            try:
                await pw_input.fill(password, timeout=5000)
                await page.wait_for_timeout(500)

                # Click Sign In
                sign_in = page.locator('button:has-text("Sign In"), button:has-text("Continue"), button[type="submit"]').first
                await sign_in.click()
                await page.wait_for_timeout(6000)
            except Exception as e:
                logger.info("[NF] Password step skipped (maybe single-page login): %s", e)

            logger.info("[NF] After login: %s", page.url)

        # Profile picker
        if "/profiles" in page.url or "browse" in page.url:
            try:
                profile = page.locator('.profile-link, .profile-icon, [data-profile-guid], .choose-profile .profile, li.profile a').first
                await profile.click(timeout=5000)
                await page.wait_for_timeout(3000)
                logger.info("[NF] Profile selected")
            except Exception:
                logger.info("[NF] No profile picker or already past it")

        # Search for content
        search_url = f"https://www.netflix.com/search?q={quote(query)}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_timeout(3000)
        logger.info("[NF] Search page: %s", page.url)

        # Click first result
        try:
            card = page.locator('.title-card a, .slider-item a, [data-uia="title-card"] a, .title-card, .boxart-container').first
            await card.click(timeout=5000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.warning("[NF] Could not click search result: %s", e)

        # Click play button if visible
        try:
            play_btn = page.locator('[data-uia="play-button"], .playLink, a[href*="/watch/"]').first
            await play_btn.click(timeout=3000)
            await page.wait_for_timeout(2000)
        except Exception:
            pass

        logger.info("[NF] Final URL: %s", page.url)

    # ── FFmpeg ─────────────────────────────────────────────────────

    async def _start_ffmpeg(self):
        playlist = str(self.hls_dir / "stream.m3u8")
        seg_pattern = str(self.hls_dir / "seg_%03d.ts")
        env = {**os.environ, "DISPLAY": self.display, "HOME": "/root"}

        # Try with audio first
        args = [
            "ffmpeg", "-y",
            "-f", "x11grab", "-video_size", "1280x720",
            "-framerate", str(FFMPEG_FRAMERATE), "-i", self.display,
            "-f", "pulse", "-i", "default",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-b:v", FFMPEG_VIDEO_BITRATE, "-g", str(FFMPEG_FRAMERATE),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-f", "hls", "-hls_time", str(HLS_SEGMENT_DURATION),
            "-hls_list_size", str(HLS_PLAYLIST_SIZE),
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", seg_pattern,
            playlist,
        ]

        self.ffmpeg_proc = await asyncio.create_subprocess_exec(
            *args, env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.sleep(2)

        # If audio fails, retry video-only
        if self.ffmpeg_proc.returncode is not None:
            logger.warning("[NF] FFmpeg with audio failed, retrying video-only")
            args_vo = [
                "ffmpeg", "-y",
                "-f", "x11grab", "-video_size", "1280x720",
                "-framerate", str(FFMPEG_FRAMERATE), "-i", self.display,
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-b:v", FFMPEG_VIDEO_BITRATE, "-g", str(FFMPEG_FRAMERATE),
                "-pix_fmt", "yuv420p",
                "-f", "hls", "-hls_time", str(HLS_SEGMENT_DURATION),
                "-hls_list_size", str(HLS_PLAYLIST_SIZE),
                "-hls_flags", "delete_segments+append_list",
                "-hls_segment_filename", seg_pattern,
                playlist,
            ]
            self.ffmpeg_proc = await asyncio.create_subprocess_exec(
                *args_vo, env=env,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )


# ── Manager ────────────────────────────────────────────────────────

_active: dict[str, NetflixStream] = {}


async def start_netflix_stream(stream_id: str, netflix_url: str, email: str, password: str, **kw) -> str:
    if stream_id in _active:
        await stop_netflix_stream(stream_id)
    # Extract search query from URL or use as-is
    query = netflix_url
    if "search?q=" in netflix_url:
        query = netflix_url.split("search?q=")[-1].replace("%20", " ").replace("+", " ")
    elif "/watch/" in netflix_url or "/title/" in netflix_url:
        query = netflix_url  # pass URL directly
    stream = NetflixStream(stream_id)
    _active[stream_id] = stream
    await stream.start(query, email, password)
    return str(stream.hls_dir)


async def stop_netflix_stream(stream_id: str):
    s = _active.pop(stream_id, None)
    if s:
        await s.stop()


def get_stream_hls_dir(stream_id: str) -> Optional[Path]:
    s = _active.get(stream_id)
    return s.hls_dir if s else None
