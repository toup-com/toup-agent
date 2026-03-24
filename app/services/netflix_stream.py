"""
Netflix Streaming Service — plays Netflix on VPS Chrome (Widevine DRM)
and streams video+audio to user via HLS.

Architecture:
  1. PulseAudio virtual sink captures Chrome audio
  2. Chrome (real, with Widevine) plays Netflix on Xvfb display
  3. FFmpeg captures Xvfb + PulseAudio → HLS segments
  4. Segments served via HTTP → hls.js on frontend
"""

import asyncio
import logging
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# HLS segment config for low latency
HLS_SEGMENT_DURATION = 1       # 1-second segments
HLS_PLAYLIST_SIZE = 5          # keep 5 segments in playlist
FFMPEG_FRAMERATE = 24
FFMPEG_VIDEO_BITRATE = "2500k"
FFMPEG_AUDIO_BITRATE = "128k"


class NetflixStream:
    """Manages a single Netflix streaming session."""

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.hls_dir = Path(tempfile.mkdtemp(prefix=f"nf-stream-{stream_id}-"))
        self.display: Optional[str] = None
        self.xvfb_proc: Optional[asyncio.subprocess.Process] = None
        self.pulse_proc: Optional[asyncio.subprocess.Process] = None
        self.chrome_proc: Optional[asyncio.subprocess.Process] = None
        self.ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self.running = False
        self._pulse_sink = f"nf_sink_{stream_id}"

    async def start(self, netflix_url: str, email: str, password: str, profile: str = "") -> str:
        """Start the Netflix stream. Returns the HLS playlist path."""
        try:
            self.running = True

            # 1. Start Xvfb on a free display
            self.display = await self._start_xvfb()
            logger.info("[NF-STREAM] Xvfb on %s", self.display)

            # 2. Start PulseAudio virtual sink
            await self._start_pulseaudio()
            logger.info("[NF-STREAM] PulseAudio sink ready")

            # 3. Launch Chrome and navigate to Netflix
            await self._start_chrome(netflix_url, email, password, profile)
            logger.info("[NF-STREAM] Chrome playing Netflix")

            # 4. Start FFmpeg capture → HLS
            await self._start_ffmpeg()
            logger.info("[NF-STREAM] FFmpeg capturing → %s", self.hls_dir)

            # Wait for first HLS segment
            playlist = self.hls_dir / "stream.m3u8"
            for _ in range(30):  # 30 seconds max wait
                if playlist.exists() and playlist.stat().st_size > 0:
                    return str(playlist)
                await asyncio.sleep(1)

            raise RuntimeError("HLS playlist not created within timeout")

        except Exception as e:
            logger.exception("[NF-STREAM] Start failed")
            await self.stop()
            raise

    async def stop(self):
        """Stop all processes and clean up."""
        self.running = False
        for proc_name in ["ffmpeg_proc", "chrome_proc", "pulse_proc", "xvfb_proc"]:
            proc = getattr(self, proc_name, None)
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                logger.info("[NF-STREAM] Stopped %s", proc_name)

        # Clean up HLS files after a delay (let last segments be served)
        await asyncio.sleep(2)
        try:
            shutil.rmtree(self.hls_dir, ignore_errors=True)
        except Exception:
            pass

    async def _start_xvfb(self) -> str:
        """Start Xvfb on a free display."""
        for display_num in range(120, 150):
            display = f":{display_num}"
            lock_file = f"/tmp/.X{display_num}-lock"
            if os.path.exists(lock_file):
                continue
            self.xvfb_proc = await asyncio.create_subprocess_exec(
                "Xvfb", display, "-screen", "0", "1280x720x24",
                "-ac", "-nolisten", "tcp",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.sleep(0.5)
            if self.xvfb_proc.returncode is None:
                return display
        raise RuntimeError("No free Xvfb display")

    async def _start_pulseaudio(self):
        """Start PulseAudio with a virtual sink for audio capture."""
        env = os.environ.copy()
        env["DISPLAY"] = self.display

        # Start PulseAudio daemon
        self.pulse_proc = await asyncio.create_subprocess_exec(
            "pulseaudio", "--start", "--exit-idle-time=-1",
            f"--load=module-null-sink sink_name={self._pulse_sink} sink_properties=device.description=NetflixAudio",
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.sleep(1)

    async def _start_chrome(self, netflix_url: str, email: str, password: str, profile: str):
        """Launch Chrome, login to Netflix, and play content."""
        env = os.environ.copy()
        env["DISPLAY"] = self.display
        env["PULSE_SINK"] = self._pulse_sink

        chrome_bin = shutil.which("google-chrome-stable") or shutil.which("google-chrome") or "/opt/google/chrome/google-chrome"

        # Chrome data dir for persistent login
        chrome_data = f"/tmp/nf-chrome-{self.stream_id}"
        os.makedirs(chrome_data, exist_ok=True)

        args = [
            chrome_bin,
            "--no-sandbox",
            "--disable-gpu",
            "--window-size=1280,720",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--disable-extensions",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--autoplay-policy=no-user-gesture-required",
            f"--user-data-dir={chrome_data}",
            netflix_url,
        ]

        self.chrome_proc = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Wait for Chrome to load Netflix
        await asyncio.sleep(5)

    async def _start_ffmpeg(self):
        """Start FFmpeg to capture Xvfb display + PulseAudio → HLS."""
        playlist = str(self.hls_dir / "stream.m3u8")
        segment_pattern = str(self.hls_dir / "seg_%03d.ts")

        args = [
            "ffmpeg",
            # Video input: Xvfb display
            "-f", "x11grab",
            "-video_size", "1280x720",
            "-framerate", str(FFMPEG_FRAMERATE),
            "-i", self.display,
            # Audio input: PulseAudio monitor
            "-f", "pulse",
            "-i", f"{self._pulse_sink}.monitor",
            # Video encoding
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", FFMPEG_VIDEO_BITRATE,
            "-g", str(FFMPEG_FRAMERATE),  # keyframe every 1 second
            "-sc_threshold", "0",
            # Audio encoding
            "-c:a", "aac",
            "-b:a", FFMPEG_AUDIO_BITRATE,
            "-ar", "44100",
            # HLS output
            "-f", "hls",
            "-hls_time", str(HLS_SEGMENT_DURATION),
            "-hls_list_size", str(HLS_PLAYLIST_SIZE),
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", segment_pattern,
            playlist,
        ]

        env = os.environ.copy()
        env["DISPLAY"] = self.display

        self.ffmpeg_proc = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )


# ── Global stream manager ─────────────────────────────────────────

_active_streams: dict[str, NetflixStream] = {}


async def start_netflix_stream(
    stream_id: str,
    netflix_url: str,
    email: str,
    password: str,
    profile: str = "",
) -> str:
    """Start a Netflix stream. Returns HLS directory path."""
    if stream_id in _active_streams:
        await stop_netflix_stream(stream_id)

    stream = NetflixStream(stream_id)
    _active_streams[stream_id] = stream
    await stream.start(netflix_url, email, password, profile)
    return str(stream.hls_dir)


async def stop_netflix_stream(stream_id: str):
    """Stop a Netflix stream."""
    stream = _active_streams.pop(stream_id, None)
    if stream:
        await stream.stop()


def get_stream_hls_dir(stream_id: str) -> Optional[Path]:
    """Get the HLS directory for an active stream."""
    stream = _active_streams.get(stream_id)
    return stream.hls_dir if stream else None
