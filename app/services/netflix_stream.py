"""
Streaming Engine — Chrome (Widevine) + FFmpeg → HLS.

Supports Netflix, Prime Video, Disney+, Crave, and other providers.
Uses per-user persistent browser sessions to avoid re-login on every stream.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

HLS_SEGMENT_DURATION = 1
HLS_PLAYLIST_SIZE = 5
FFMPEG_FRAMERATE = 24
FFMPEG_VIDEO_BITRATE = "2500k"

# Per-user session directories: /tmp/toup-sessions/{user_id}/{provider}/
SESSION_BASE = Path("/tmp/toup-sessions")


# ── Provider configs ──────────────────────────────────────────────────

PROVIDER_CONFIG = {
    "netflix": {
        "browse_url": "https://www.netflix.com/browse",
        "login_url": "https://www.netflix.com/login",
        "search_url": "https://www.netflix.com/search?q={query}",
        "login_check": lambda url: "/login" in url,
        "email_selectors": 'input[name="userLoginId"], input[type="email"], input[data-uia="login-field"]',
        "password_selectors": 'input[name="password"], input[type="password"]',
        "submit_selectors": 'button:has-text("Continue"), button:has-text("Sign In"), button[type="submit"]',
        "profile_selectors": '.profile-link, .profile-icon, [data-profile-guid], .choose-profile .profile, li.profile a',
        "profile_check": lambda url: "/profiles" in url,
        "result_selectors": '.title-card a, .slider-item a, [data-uia="title-card"] a, .title-card, .boxart-container',
        "play_selectors": '[data-uia="play-button"], .playLink, a[href*="/watch/"]',
    },
    "prime": {
        "browse_url": "https://www.primevideo.com",
        "login_url": "https://www.primevideo.com/auth-redirect/ref=atv_nb_lcl_en_CA",
        "search_url": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={query}",
        "login_check": lambda url: "ap/signin" in url or "/auth-redirect" in url,
        "email_selectors": 'input[name="email"], input[type="email"], #ap_email',
        "password_selectors": 'input[name="password"], input[type="password"], #ap_password',
        "submit_selectors": '#signInSubmit, input[type="submit"], button:has-text("Sign in"), button:has-text("Sign-In")',
        "profile_selectors": '[data-testid="profile-avatar"], .profile-selector a',
        "profile_check": lambda url: "/profiles" in url,
        "result_selectors": '[data-testid="search-result-item"] a, .av-hover-wrapper a, article a',
        "play_selectors": '[data-testid="play-button"], .playButton, button:has-text("Play"), a:has-text("Watch")',
    },
    "disney": {
        "browse_url": "https://www.disneyplus.com",
        "login_url": "https://www.disneyplus.com/login",
        "search_url": "https://www.disneyplus.com/search?q={query}",
        "login_check": lambda url: "/login" in url,
        "email_selectors": 'input[type="email"], input[name="email"]',
        "password_selectors": 'input[type="password"], input[name="password"]',
        "submit_selectors": 'button:has-text("Continue"), button:has-text("Sign In"), button[type="submit"]',
        "profile_selectors": '[data-testid="profile-avatar"], .profile-avatar-link',
        "profile_check": lambda url: "/select-profile" in url,
        "result_selectors": '[data-testid="search-result"] a, .search-result a',
        "play_selectors": 'button:has-text("Play"), [data-testid="play-button"]',
    },
    "crave": {
        "browse_url": "https://www.crave.ca",
        "login_url": "https://www.crave.ca/login",
        "search_url": "https://www.crave.ca/search?q={query}",
        "login_check": lambda url: "/login" in url or "/signin" in url,
        "email_selectors": 'input[type="email"], input[name="email"], #email',
        "password_selectors": 'input[type="password"], input[name="password"]',
        "submit_selectors": 'button:has-text("Sign In"), button:has-text("Log In"), button[type="submit"]',
        "profile_selectors": '.profile-selector a',
        "profile_check": lambda url: "/profiles" in url,
        "result_selectors": '.media-card a, .search-result a, article a',
        "play_selectors": 'button:has-text("Play"), .play-button, a:has-text("Watch")',
    },
}


def _get_session_dir(user_id: str, provider: str) -> Path:
    """Get per-user per-provider Chrome data dir for session persistence."""
    d = SESSION_BASE / user_id / provider
    d.mkdir(parents=True, exist_ok=True)
    return d


class StreamEngine:
    """Generic streaming engine — works with any supported provider."""

    def __init__(self, stream_id: str, provider: str = "netflix", user_id: str = ""):
        self.stream_id = stream_id
        self.provider = provider
        self.user_id = user_id
        self.config = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG["netflix"])
        self.hls_dir = Path(tempfile.mkdtemp(prefix=f"stream-{stream_id}-"))
        self.display: Optional[str] = None
        self.cdp_port: Optional[int] = None
        self.xvfb_proc: Optional[asyncio.subprocess.Process] = None
        self.ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self.running = False
        self._browser = None
        self._playwright = None
        self._page = None

    async def start(self, search_query: str, email: str, password: str) -> str:
        """Start stream. Returns HLS playlist path."""
        try:
            self.running = True

            # 1. Xvfb
            self.display = await self._start_xvfb()
            logger.info("[Stream:%s] Xvfb on %s", self.provider, self.display)

            # 2. PulseAudio
            await self._setup_pulse()

            # 3. Chrome with per-user session
            self.cdp_port = 9300 + hash(self.stream_id) % 100
            await self._start_chrome()
            logger.info("[Stream:%s] Chrome on CDP port %d", self.provider, self.cdp_port)

            # 4. Session-aware login + navigate
            await self._login_and_play(search_query, email, password)
            logger.info("[Stream:%s] Playing", self.provider)

            # 5. FFmpeg capture → HLS
            await self._start_ffmpeg()
            logger.info("[Stream:%s] FFmpeg capturing", self.provider)

            # Wait for first segment
            playlist = self.hls_dir / "stream.m3u8"
            for _ in range(20):
                if playlist.exists() and playlist.stat().st_size > 0:
                    return str(playlist)
                await asyncio.sleep(1)
            raise RuntimeError("HLS not ready in time")

        except Exception:
            logger.exception("[Stream:%s] Start failed", self.provider)
            await self.stop()
            raise

    async def stop(self):
        self.running = False
        if self.ffmpeg_proc and self.ffmpeg_proc.returncode is None:
            try:
                self.ffmpeg_proc.terminate()
                await asyncio.wait_for(self.ffmpeg_proc.wait(), timeout=5)
            except Exception:
                try: self.ffmpeg_proc.kill()
                except Exception: pass
        if self._browser:
            try: await self._browser.close()
            except Exception: pass
        if self._playwright:
            try: await self._playwright.stop()
            except Exception: pass
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
        """Launch Chrome with per-user persistent session dir."""
        try:
            from patchright.async_api import async_playwright
        except ImportError:
            from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        chrome = shutil.which("google-chrome-stable") or "/opt/google/chrome/google-chrome"

        # Per-user session dir — cookies survive across streams
        data_dir = str(_get_session_dir(self.user_id, self.provider))

        self._browser = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=data_dir,
            executable_path=chrome,
            headless=False,
            args=[
                "--no-sandbox", "--disable-dev-shm-usage",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-features=TranslateUI",
                "--disable-gpu",
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--disable-accelerated-video-decode",
                "--disable-gpu-compositing",
                f"--remote-debugging-port={self.cdp_port}",
            ],
            viewport={"width": 1280, "height": 720},
            no_viewport=False,
            ignore_https_errors=True,
        )
        self._page = self._browser.pages[0] if self._browser.pages else await self._browser.new_page()

    # ── Session-aware login ────────────────────────────────────────

    async def _login_and_play(self, query: str, email: str, password: str):
        """Navigate to content. Only login if session is expired."""
        page = self._page
        cfg = self.config

        # Try navigating directly to content first (session may be warm)
        search_url = cfg["search_url"].format(query=quote(query))
        await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        current_url = page.url
        logger.info("[Stream:%s] Navigated to: %s", self.provider, current_url)

        # Check if we got redirected to login
        if cfg["login_check"](current_url):
            logger.info("[Stream:%s] Session expired — logging in", self.provider)
            await self._do_login(page, email, password, cfg)

            # After login, navigate to content again
            await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(2000)
            logger.info("[Stream:%s] After re-login: %s", self.provider, page.url)
        else:
            logger.info("[Stream:%s] Session warm — skipped login", self.provider)

        # Handle profile picker if present
        if cfg["profile_check"](page.url):
            try:
                profile = page.locator(cfg["profile_selectors"]).first
                await profile.click(timeout=5000)
                await page.wait_for_timeout(3000)
                # Re-navigate to search after profile selection
                await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(2000)
            except Exception:
                logger.info("[Stream:%s] No profile picker or already past it", self.provider)

        # Click first search result
        try:
            card = page.locator(cfg["result_selectors"]).first
            await card.click(timeout=5000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            logger.warning("[Stream:%s] Could not click result: %s", self.provider, e)

        # Click play button
        try:
            play_btn = page.locator(cfg["play_selectors"]).first
            await play_btn.click(timeout=3000)
            await page.wait_for_timeout(2000)
        except Exception:
            pass

        logger.info("[Stream:%s] Final URL: %s", self.provider, page.url)

    async def _do_login(self, page, email: str, password: str, cfg: dict):
        """Perform the full login flow."""
        # Fill email
        email_input = page.locator(cfg["email_selectors"]).first
        await email_input.fill(email)
        await page.wait_for_timeout(500)

        # Submit email (some providers have 2-step login)
        submit_btn = page.locator(cfg["submit_selectors"]).first
        await submit_btn.click()
        await page.wait_for_timeout(4000)

        # Fill password (may be on new page or same page)
        pw_input = page.locator(cfg["password_selectors"]).first
        try:
            await pw_input.fill(password, timeout=5000)
            await page.wait_for_timeout(500)

            sign_in = page.locator(cfg["submit_selectors"]).first
            await sign_in.click()
            await page.wait_for_timeout(6000)
        except Exception as e:
            logger.info("[Stream:%s] Password step skipped: %s", self.provider, e)

        logger.info("[Stream:%s] Login completed: %s", self.provider, page.url)

    # ── FFmpeg ─────────────────────────────────────────────────────

    async def _start_ffmpeg(self):
        playlist = str(self.hls_dir / "stream.m3u8")
        seg_pattern = str(self.hls_dir / "seg_%03d.ts")
        env = {**os.environ, "DISPLAY": self.display, "HOME": "/root"}

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

        # Retry video-only if audio fails
        if self.ffmpeg_proc.returncode is not None:
            logger.warning("[Stream:%s] FFmpeg audio failed, retrying video-only", self.provider)
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


# ── Session warm-up (background login to prime cookies) ───────────────

async def warm_session(user_id: str, provider: str, email: str, password: str) -> str:
    """Login in headless Chrome to establish session cookies. Returns status."""
    cfg = PROVIDER_CONFIG.get(provider)
    if not cfg:
        return "unsupported"

    try:
        from patchright.async_api import async_playwright
    except ImportError:
        from playwright.async_api import async_playwright

    chrome = shutil.which("google-chrome-stable") or "/opt/google/chrome/google-chrome"
    data_dir = str(_get_session_dir(user_id, provider))

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch_persistent_context(
            user_data_dir=data_dir,
            executable_path=chrome,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()

        await page.goto(cfg["browse_url"], wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        if cfg["login_check"](page.url):
            # Do login
            email_input = page.locator(cfg["email_selectors"]).first
            await email_input.fill(email)
            await page.wait_for_timeout(500)

            submit_btn = page.locator(cfg["submit_selectors"]).first
            await submit_btn.click()
            await page.wait_for_timeout(4000)

            pw_input = page.locator(cfg["password_selectors"]).first
            try:
                await pw_input.fill(password, timeout=5000)
                await page.wait_for_timeout(500)
                sign_in = page.locator(cfg["submit_selectors"]).first
                await sign_in.click()
                await page.wait_for_timeout(6000)
            except Exception:
                pass

            # Check if login succeeded
            final_url = page.url
            if cfg["login_check"](final_url):
                logger.warning("[warm_session:%s] Login may have failed: %s", provider, final_url)
                await browser.close()
                return "blocked"

            # Handle profile picker
            if cfg["profile_check"](final_url):
                try:
                    profile = page.locator(cfg["profile_selectors"]).first
                    await profile.click(timeout=5000)
                    await page.wait_for_timeout(2000)
                except Exception:
                    pass

            logger.info("[warm_session:%s] Login successful for user %s", provider, user_id)
            await browser.close()
            return "warm"
        else:
            logger.info("[warm_session:%s] Already logged in for user %s", provider, user_id)
            await browser.close()
            return "warm"
    except Exception as e:
        logger.exception("[warm_session:%s] Failed: %s", provider, e)
        return "cold"
    finally:
        await pw.stop()


async def check_session(user_id: str, provider: str) -> str:
    """Quick check if session cookies are still valid. Returns status."""
    cfg = PROVIDER_CONFIG.get(provider)
    if not cfg:
        return "unsupported"

    data_dir = _get_session_dir(user_id, provider)
    if not (data_dir / "Default").exists():
        return "cold"

    try:
        from patchright.async_api import async_playwright
    except ImportError:
        from playwright.async_api import async_playwright

    chrome = shutil.which("google-chrome-stable") or "/opt/google/chrome/google-chrome"

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch_persistent_context(
            user_data_dir=str(data_dir),
            executable_path=chrome,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()
        await page.goto(cfg["browse_url"], wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(2000)

        status = "expired" if cfg["login_check"](page.url) else "warm"
        await browser.close()
        return status
    except Exception:
        return "cold"
    finally:
        await pw.stop()


# ── Manager (backward-compatible) ─────────────────────────────────────

_active: dict[str, StreamEngine] = {}


async def start_netflix_stream(
    stream_id: str,
    netflix_url: str,
    email: str,
    password: str,
    user_id: str = "",
    provider: str = "netflix",
    **kw,
) -> str:
    if stream_id in _active:
        await stop_netflix_stream(stream_id)

    query = netflix_url
    if "search?q=" in netflix_url:
        query = netflix_url.split("search?q=")[-1].replace("%20", " ").replace("+", " ")
    elif "/watch/" in netflix_url or "/title/" in netflix_url:
        query = netflix_url

    stream = StreamEngine(stream_id, provider=provider, user_id=user_id)
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
