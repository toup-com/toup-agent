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
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HLS_SEGMENT_DURATION = 1
HLS_PLAYLIST_SIZE = 5
FFMPEG_FRAMERATE = 24
FFMPEG_VIDEO_BITRATE = "2500k"
FFMPEG_AUDIO_BITRATE = "128k"


class NetflixStream:
    """Manages a single Netflix streaming session."""

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.hls_dir = Path(tempfile.mkdtemp(prefix=f"nf-{stream_id}-"))
        self.display: Optional[str] = None
        self.xvfb_proc: Optional[asyncio.subprocess.Process] = None
        self.chrome_proc: Optional[asyncio.subprocess.Process] = None
        self.ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self.running = False
        self._pulse_sink = f"nf_{stream_id[:8]}"

    async def start(self, netflix_url: str, email: str, password: str, profile: str = "") -> str:
        """Start the Netflix stream. Returns HLS playlist path."""
        try:
            self.running = True

            # 1. Xvfb
            self.display = await self._start_xvfb()
            logger.info("[NF-STREAM] Xvfb on %s", self.display)

            # 2. PulseAudio virtual sink
            await self._setup_pulseaudio()
            logger.info("[NF-STREAM] PulseAudio ready")

            # 3. Chrome → Netflix
            await self._start_chrome(netflix_url)
            logger.info("[NF-STREAM] Chrome launched on %s", netflix_url)

            # 4. FFmpeg capture → HLS
            await self._start_ffmpeg()
            logger.info("[NF-STREAM] FFmpeg capturing → %s", self.hls_dir)

            # Wait for first segment
            playlist = self.hls_dir / "stream.m3u8"
            for _ in range(30):
                if playlist.exists() and playlist.stat().st_size > 0:
                    return str(playlist)
                await asyncio.sleep(1)

            raise RuntimeError("HLS playlist not created within timeout")

        except Exception as e:
            logger.exception("[NF-STREAM] Start failed: %s", e)
            await self.stop()
            raise

    async def stop(self):
        """Stop all processes."""
        self.running = False
        for name in ["ffmpeg_proc", "chrome_proc", "xvfb_proc"]:
            proc = getattr(self, name, None)
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
        # Cleanup HLS files
        await asyncio.sleep(1)
        shutil.rmtree(self.hls_dir, ignore_errors=True)

    async def _start_xvfb(self) -> str:
        for num in range(120, 150):
            display = f":{num}"
            if os.path.exists(f"/tmp/.X{num}-lock"):
                continue
            self.xvfb_proc = await asyncio.create_subprocess_exec(
                "Xvfb", display, "-screen", "0", "1280x720x24", "-ac", "-nolisten", "tcp",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.sleep(0.5)
            if self.xvfb_proc.returncode is None:
                return display
        raise RuntimeError("No free Xvfb display")

    async def _setup_pulseaudio(self):
        """Ensure PulseAudio is running and create a virtual sink."""
        env = os.environ.copy()
        env["DISPLAY"] = self.display
        env["HOME"] = "/root"

        # Create pulse cookie dir
        os.makedirs("/root/.config/pulse", exist_ok=True)

        # Start PulseAudio if not running (--start is idempotent)
        proc = await asyncio.create_subprocess_exec(
            "pulseaudio", "--start", "--daemonize=yes",
            env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        await asyncio.sleep(0.5)

        # Load null sink for this stream
        proc = await asyncio.create_subprocess_exec(
            "pactl", "load-module", "module-null-sink",
            f"sink_name={self._pulse_sink}",
            f"sink_properties=device.description=Netflix_{self.stream_id[:8]}",
            env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

        # Set as default sink so Chrome uses it
        proc = await asyncio.create_subprocess_exec(
            "pactl", "set-default-sink", self._pulse_sink,
            env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    async def _start_chrome(self, netflix_url: str):
        """Launch real Chrome with Widevine on the Xvfb display."""
        env = os.environ.copy()
        env["DISPLAY"] = self.display
        env["HOME"] = "/root"

        chrome_bin = (
            shutil.which("google-chrome-stable")
            or shutil.which("google-chrome")
            or "/opt/google/chrome/google-chrome"
        )

        chrome_data = f"/tmp/nf-chrome-{self.stream_id[:8]}"
        os.makedirs(chrome_data, exist_ok=True)

        args = [
            chrome_bin,
            "--no-sandbox",
            "--disable-gpu",
            "--window-size=1280,720",
            "--start-maximized",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--disable-extensions",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--autoplay-policy=no-user-gesture-required",
            "--disable-features=TranslateUI",
            f"--user-data-dir={chrome_data}",
            netflix_url,
        ]

        self.chrome_proc = await asyncio.create_subprocess_exec(
            *args, env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        # Wait for page load
        await asyncio.sleep(6)

    async def _start_ffmpeg(self):
        """FFmpeg captures Xvfb display + PulseAudio → HLS."""
        playlist = str(self.hls_dir / "stream.m3u8")
        seg_pattern = str(self.hls_dir / "seg_%03d.ts")

        # Try capturing audio from PulseAudio monitor; if it fails, video-only
        args = [
            "ffmpeg", "-y",
            # Video: X11 grab
            "-f", "x11grab",
            "-video_size", "1280x720",
            "-framerate", str(FFMPEG_FRAMERATE),
            "-i", self.display,
            # Audio: PulseAudio monitor (may fail silently)
            "-f", "pulse",
            "-i", f"{self._pulse_sink}.monitor",
            # Video encoding
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", FFMPEG_VIDEO_BITRATE,
            "-g", str(FFMPEG_FRAMERATE),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
            # Audio encoding
            "-c:a", "aac",
            "-b:a", FFMPEG_AUDIO_BITRATE,
            "-ar", "44100",
            # HLS output
            "-f", "hls",
            "-hls_time", str(HLS_SEGMENT_DURATION),
            "-hls_list_size", str(HLS_PLAYLIST_SIZE),
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", seg_pattern,
            playlist,
        ]

        env = os.environ.copy()
        env["DISPLAY"] = self.display
        env["HOME"] = "/root"

        self.ffmpeg_proc = await asyncio.create_subprocess_exec(
            *args, env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )

        # Check if FFmpeg started OK
        await asyncio.sleep(2)
        if self.ffmpeg_proc.returncode is not None:
            stderr = (await self.ffmpeg_proc.stderr.read()).decode(errors="replace")[:500]
            logger.warning("[NF-STREAM] FFmpeg failed (trying video-only): %s", stderr)
            # Retry without audio
            args_no_audio = [
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
                *args_no_audio, env=env,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )


# ── Global stream manager ─────────────────────────────────────────

_active_streams: dict[str, NetflixStream] = {}


async def start_netflix_stream(
    stream_id: str, netflix_url: str,
    email: str, password: str, profile: str = "",
) -> str:
    if stream_id in _active_streams:
        await stop_netflix_stream(stream_id)
    stream = NetflixStream(stream_id)
    _active_streams[stream_id] = stream
    await stream.start(netflix_url, email, password, profile)
    return str(stream.hls_dir)


async def stop_netflix_stream(stream_id: str):
    stream = _active_streams.pop(stream_id, None)
    if stream:
        await stream.stop()


def get_stream_hls_dir(stream_id: str) -> Optional[Path]:
    stream = _active_streams.get(stream_id)
    return stream.hls_dir if stream else None
