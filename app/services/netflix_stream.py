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
        for name in ["ffmpeg_proc", "chrome_proc", "xvfb_proc"]:
            proc = getattr(self, name, None)
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
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
        env = {**os.environ, "DISPLAY": self.display, "HOME": "/root"}
        chrome = shutil.which("google-chrome-stable") or "/opt/google/chrome/google-chrome"
        data_dir = "/tmp/nf-chrome-shared"  # shared across streams for cookie persistence
        os.makedirs(data_dir, exist_ok=True)

        self.chrome_proc = await asyncio.create_subprocess_exec(
            chrome,
            "--no-sandbox", "--disable-gpu", "--window-size=1280,720",
            "--disable-dev-shm-usage", "--disable-infobars", "--disable-extensions",
            "--autoplay-policy=no-user-gesture-required",
            "--disable-features=TranslateUI",
            f"--remote-debugging-port={self.cdp_port}",
            f"--user-data-dir={data_dir}",
            "about:blank",
            env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        # Wait for CDP to be ready
        for attempt in range(30):
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"http://127.0.0.1:{self.cdp_port}/json/version", timeout=2)
                    if r.status_code == 200:
                        logger.info("[NF] CDP ready after %d attempts", attempt + 1)
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
        raise RuntimeError("Chrome CDP not ready after 30s")

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
        """Login to Netflix and navigate to content."""
        ws_url = await self._get_page_ws()

        # Navigate to Netflix
        await self._cdp_navigate(ws_url, "https://www.netflix.com/browse")
        await asyncio.sleep(2)

        # Refresh ws_url after navigation (page might change)
        ws_url = await self._get_page_ws()

        # Check current URL — are we logged in?
        current_url = await self._cdp_evaluate(ws_url, "window.location.href")
        logger.info("[NF] Current URL: %s", current_url)

        # If on login page, do login
        if current_url and "/login" in str(current_url):
            logger.info("[NF] Login required, automating...")

            # Fill email
            await self._cdp_evaluate(ws_url, f"""
                document.querySelector('input[name="userLoginId"]').value = '{email}';
                document.querySelector('input[name="userLoginId"]').dispatchEvent(new Event('input', {{bubbles: true}}));
            """)
            await asyncio.sleep(0.5)

            # Fill password
            await self._cdp_evaluate(ws_url, f"""
                document.querySelector('input[name="password"]').value = '{password}';
                document.querySelector('input[name="password"]').dispatchEvent(new Event('input', {{bubbles: true}}));
            """)
            await asyncio.sleep(0.5)

            # Click sign in
            await self._cdp_evaluate(ws_url, """
                document.querySelector('button[data-uia="login-submit-button"]').click();
            """)
            await asyncio.sleep(5)

            # Refresh ws_url after login redirect
            ws_url = await self._get_page_ws()
            current_url = await self._cdp_evaluate(ws_url, "window.location.href")
            logger.info("[NF] After login URL: %s", current_url)

        # If on profile picker, click first profile
        if current_url and ("/profiles" in str(current_url) or "/browse" in str(current_url)):
            await asyncio.sleep(1)
            ws_url = await self._get_page_ws()
            await self._cdp_evaluate(ws_url, """
                const profiles = document.querySelectorAll('.profile-icon, [data-profile-guid], .choose-profile .profile');
                if (profiles.length > 0) profiles[0].click();
            """)
            await asyncio.sleep(3)

        # Navigate to search for the content
        ws_url = await self._get_page_ws()
        search_url = f"https://www.netflix.com/search?q={quote(query)}"
        await self._cdp_navigate(ws_url, search_url)
        await asyncio.sleep(3)

        # Click first result to start playing
        ws_url = await self._get_page_ws()
        await self._cdp_evaluate(ws_url, """
            // Click first title card in search results
            const cards = document.querySelectorAll('.title-card, .slider-item, [data-uia="title-card"]');
            if (cards.length > 0) {
                const link = cards[0].querySelector('a') || cards[0];
                link.click();
            }
        """)
        await asyncio.sleep(2)

        # Try to click play button if visible
        ws_url = await self._get_page_ws()
        await self._cdp_evaluate(ws_url, """
            const playBtn = document.querySelector('[data-uia="play-button"], .playLink, .maturity-rating-overlay button');
            if (playBtn) playBtn.click();
        """)
        await asyncio.sleep(2)

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
