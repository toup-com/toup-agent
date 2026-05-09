"""
App Manager — Manages Expo/Metro dev server processes for user-built apps.

Handles the full lifecycle: scaffold, write files, install deps, start/stop
Metro (mobile) + Expo Web, database setup, GitHub, and web publishing.
"""

import asyncio
import json
import logging
import os
import re
import signal
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

APPS_DIR = os.environ.get("TOUP_APPS_DIR", "/opt/toup-agent/apps")
METRO_PORT_RANGE = (3001, 3050)
WEB_PORT_RANGE = (4001, 4050)
MAX_LOG_LINES = 500

# ── npm install constants ─────────────────────────────────────────
NPM_CI_TIMEOUT = 180       # seconds — npm ci is faster, tighter budget
NPM_INSTALL_TIMEOUT = 300  # seconds — npm install for first-time installs
NPM_FLAGS = ["--no-audit", "--no-fund", "--loglevel=error"]
NPM_RETRY_TRANSIENT_PATTERNS = ["ETIMEDOUT", "ECONNRESET", "EAI_AGAIN", "network", "FETCH_ERROR", "503", "timed out"]


@dataclass
class ManagedApp:
    """In-memory state for a running app."""
    app_id: str
    metro_port: Optional[int] = None
    web_port: Optional[int] = None
    metro_process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    web_process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    log_buffer: deque = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    started_at: float = 0.0
    # Watchdog cooldown — avoid thrashing on crash-loop apps. Updated by
    # ensure_web_alive() each time it triggers a revive.
    last_revive_at: float = 0.0


class AppManager:
    """Manages Expo/Metro dev server processes for user-built apps."""

    def __init__(self):
        self._running: Dict[str, ManagedApp] = {}
        self._used_metro_ports: Set[int] = set()
        self._used_web_ports: Set[int] = set()
        self._public_ip: Optional[str] = None
        os.makedirs(APPS_DIR, exist_ok=True)

    async def _resolve_app_dir(self, app_id: str, app_dir: Optional[str] = None) -> str:
        """Resolve the canonical app directory for a given app.

        If app_dir is provided and exists, use it directly.
        Otherwise, look up app.app_dir from the database.
        Never recompute from APPS_DIR + app_id — that was the root cause of
        the slug-vs-UUID directory mismatch bug.
        """
        if app_dir and os.path.exists(app_dir):
            return app_dir
        # DB lookup
        try:
            from app.db.database import async_session_maker
            from app.db.models import App
            async with async_session_maker() as db:
                app = await db.get(App, app_id)
                if app and app.app_dir and os.path.exists(app.app_dir):
                    return app.app_dir
        except Exception:
            pass
        # Last resort: try both slug-based and UUID-based paths
        uuid_dir = os.path.join(APPS_DIR, app_id)
        if os.path.exists(uuid_dir):
            return uuid_dir
        logger.error(f"[APP] Cannot resolve app_dir for {app_id} (tried {app_dir}, DB lookup, {uuid_dir})")
        return uuid_dir  # return it anyway so callers get a clear error

    # ── Public IP ───────────────────────────────────────

    async def _get_public_ip(self) -> str:
        """Get VPS public IP (cached)."""
        if self._public_ip:
            return self._public_ip
        try:
            proc = await asyncio.create_subprocess_exec(
                "curl", "-s", "-4", "--max-time", "5", "https://ifconfig.me",
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            ip = (stdout or b"").decode().strip()
            if ip and re.match(r"^\d+\.\d+\.\d+\.\d+$", ip):
                self._public_ip = ip
                return ip
        except Exception:
            pass
        self._public_ip = "127.0.0.1"
        return self._public_ip

    # ── Core Lifecycle ──────────────────────────────────

    async def scaffold_app(self, app_id: str, app_name: str, slug: str = "") -> str:
        """Create a new Expo app directory. Returns the app directory path."""
        app_dir = os.path.join(APPS_DIR, slug or app_id)
        if os.path.exists(app_dir):
            logger.warning(f"App dir already exists: {app_dir}, removing")
            shutil.rmtree(app_dir)

        logger.info(f"[APP] Scaffolding {app_name} at {app_dir}")
        # Create with npx create-expo-app
        safe_name = re.sub(r"[^a-zA-Z0-9-]", "-", app_name.lower()).strip("-")[:30] or "app"
        proc = await asyncio.create_subprocess_exec(
            "npx", "create-expo-app@latest", safe_name,
            "--template", "blank-typescript",
            "--no-install",  # We'll install deps separately
            cwd=APPS_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "npm_config_yes": "true"},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = (stdout or b"").decode() + (stderr or b"").decode()

        # Expo creates dir with the safe_name, rename to app_id
        created_dir = os.path.join(APPS_DIR, safe_name)
        if os.path.exists(created_dir) and created_dir != app_dir:
            os.rename(created_dir, app_dir)
        elif not os.path.exists(app_dir):
            # Fallback: create manually
            os.makedirs(app_dir, exist_ok=True)
            logger.warning(f"[APP] Scaffold fallback — created empty dir. Output: {output[:500]}")

        # Create storage directory
        os.makedirs(os.path.join(app_dir, "storage"), exist_ok=True)

        logger.info(f"[APP] Scaffolded {app_name} at {app_dir}")
        return app_dir

    async def write_app_files(self, app_id: str, files: Dict[str, str], app_dir: Optional[str] = None) -> None:
        """Write files dict to app directory on disk."""
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        assert os.path.exists(app_dir), f"app_dir {app_dir} does not exist"
        for file_path, content in files.items():
            # Normalize path
            rel_path = file_path.lstrip("/")
            full_path = os.path.join(app_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
        logger.info(f"[APP] Wrote {len(files)} files to {app_dir}")

    async def install_deps(self, app_id: str, deps: Optional[List[str]] = None, app_dir: Optional[str] = None) -> str:
        """Run npm install in app dir with clean state, retries, and error classification.

        Strategy:
          1. Always clean node_modules + lock artifacts before install
          2. Use npm ci if lock file exists, npm install otherwise
          3. On transient network error, retry once with clean state
          4. Capture both stdout/stderr for actionable error messages
        """
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        if not os.path.exists(app_dir):
            raise RuntimeError(f"[INSTALL] app_dir does not exist: {app_dir}")

        output = await self._npm_install_with_retry(app_id, app_dir)

        # Install extra deps using `npx expo install` for version compatibility
        if deps:
            cmd2 = ["npx", "expo", "install"] + deps
            env = {**os.environ, "EXPO_NO_TELEMETRY": "1"}
            proc2 = await asyncio.create_subprocess_exec(
                *cmd2, cwd=app_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout2, stderr2 = await asyncio.wait_for(proc2.communicate(), timeout=NPM_INSTALL_TIMEOUT)
                output += "\n" + (stdout2 or b"").decode()
            except asyncio.TimeoutError:
                proc2.kill()
                logger.warning("[INSTALL] expo install timed out for %s", app_id[:8])

        logger.info("[INSTALL] Deps installed for %s", app_id[:8])
        return output[:2000]

    async def _npm_install_with_retry(self, app_id: str, app_dir: str) -> str:
        """Run npm install with clean state and one retry on transient failure."""
        last_error = ""
        for attempt in range(2):
            try:
                return await self._npm_install_clean(app_id, app_dir, attempt)
            except RuntimeError as e:
                last_error = str(e)
                classification = self._classify_install_error(last_error)
                logger.warning(
                    "[INSTALL] Attempt %d failed for %s — classified=%s: %s",
                    attempt + 1, app_id[:8], classification, last_error[:200],
                )
                if classification == "transient" and attempt == 0:
                    logger.info("[INSTALL] Retrying %s after transient failure...", app_id[:8])
                    continue
                raise
        raise RuntimeError(last_error)

    async def _npm_install_clean(self, app_id: str, app_dir: str, attempt: int = 0) -> str:
        """Single npm install attempt with clean state."""
        nm_path = os.path.join(app_dir, "node_modules")
        lock_path = os.path.join(app_dir, "package-lock.json")
        # Use a shared npm cache across all apps so repeated deps (react-native,
        # expo, etc.) are fetched once and reused. Previous per-app cache meant
        # every build downloaded ~400MB of the same packages.
        npm_cache = os.path.join("/tmp", "npm-cache-shared")

        # Always clean node_modules for a deterministic install
        if os.path.isdir(nm_path):
            logger.info("[INSTALL] Cleaning node_modules in %s (attempt %d)", app_dir, attempt + 1)
            shutil.rmtree(nm_path, ignore_errors=True)

        # Use npm ci if lock file exists (faster, deterministic), npm install otherwise
        if os.path.isfile(lock_path):
            cmd = ["npm", "ci", f"--cache={npm_cache}"] + NPM_FLAGS
            timeout = NPM_CI_TIMEOUT
            logger.info("[INSTALL] Using npm ci for %s (lock file exists)", app_id[:8])
        else:
            cmd = ["npm", "install", f"--cache={npm_cache}"] + NPM_FLAGS
            timeout = NPM_INSTALL_TIMEOUT
            logger.info("[INSTALL] Using npm install for %s (no lock file)", app_id[:8])

        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=app_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            elapsed = time.time() - t0
            raise RuntimeError(
                f"npm install timed out after {elapsed:.0f}s in {app_dir}"
            )

        elapsed = time.time() - t0
        stdout_text = (stdout or b"").decode()
        stderr_text = (stderr or b"").decode()

        if proc.returncode != 0:
            # Combine last 50 lines of stderr + stdout for debugging
            combined = (stderr_text + "\n" + stdout_text).strip()
            tail = "\n".join(combined.split("\n")[-50:])
            logger.error(
                "[INSTALL] npm failed (exit %d) in %.1fs for %s:\n%s",
                proc.returncode, elapsed, app_id[:8], tail[:1500],
            )
            raise RuntimeError(f"npm install failed (exit {proc.returncode}): {tail[:1000]}")

        logger.info("[INSTALL] npm succeeded in %.1fs for %s", elapsed, app_id[:8])

        return stdout_text[:2000]

    @staticmethod
    def _classify_install_error(error_text: str) -> str:
        """Classify an npm install error for auto-repair routing.

        Returns one of:
          transient — network blip, registry timeout (retry in place)
          bad_dep   — 404, version conflict, peer dep mismatch (regenerate package.json)
          disk      — ENOSPC, ENOMEM (fail loudly, don't retry)
          stale     — corrupted node_modules (clean and retry)
          unknown   — unclassified (fail with raw error)
        """
        upper = error_text.upper()

        # Transient network errors
        for pattern in NPM_RETRY_TRANSIENT_PATTERNS:
            if pattern.upper() in upper:
                return "transient"

        # Bad dependency
        if any(s in upper for s in ["404", "NOT FOUND", "ERESOLVE", "PEER DEP", "NO MATCHING VERSION", "ETARGET"]):
            return "bad_dep"

        # Disk/memory
        if any(s in upper for s in ["ENOSPC", "ENOMEM", "NO SPACE LEFT"]):
            return "disk"

        # Stale state
        if any(s in upper for s in ["ENOENT", "EACCES", "EPERM", "LOCK"]):
            return "stale"

        return "unknown"

    async def _ensure_node_modules(self, app_id: str, app_dir: str) -> None:
        """Reinstall node_modules if missing (e.g. after stop cleaned them)."""
        nm_dir = os.path.join(app_dir, "node_modules")
        if not os.path.exists(nm_dir):
            logger.info(f"[APP] node_modules missing for {app_id}, reinstalling...")
            await self._npm_install_with_retry(app_id, app_dir)

    async def start_metro(self, app_id: str, app_dir: Optional[str] = None) -> int:
        """Start Metro bundler for mobile preview. Returns allocated port."""
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        assert os.path.exists(app_dir), f"app_dir {app_dir} does not exist"
        await self._ensure_node_modules(app_id, app_dir)
        port = self._allocate_port(METRO_PORT_RANGE, self._used_metro_ports)

        managed = self._running.get(app_id)
        if not managed:
            managed = ManagedApp(app_id=app_id)
            self._running[app_id] = managed

        # Kill existing metro if running
        if managed.metro_process and managed.metro_process.returncode is None:
            managed.metro_process.terminate()
            await asyncio.sleep(1)

        proc = await asyncio.create_subprocess_exec(
            "npx", "expo", "start", "--port", str(port), "--lan",
            cwd=app_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "EXPO_NO_TELEMETRY": "1", "CI": "1"},
        )

        managed.metro_port = port
        managed.metro_process = proc
        managed.started_at = time.time()
        # Port already claimed in _allocate_port

        # Start log reader
        asyncio.create_task(self._read_output(managed, proc, "metro"))

        # Wait for Metro to be ready (max 30s)
        await self._wait_for_ready(managed, timeout=30)

        logger.info(f"[APP] Metro started for {app_id} on port {port}")
        return port

    async def start_web(self, app_id: str, app_dir: Optional[str] = None) -> int:
        """Start Expo web server. Returns allocated port."""
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        assert os.path.exists(app_dir), f"app_dir {app_dir} does not exist"
        await self._ensure_node_modules(app_id, app_dir)
        port = self._allocate_port(WEB_PORT_RANGE, self._used_web_ports)

        managed = self._running.get(app_id)
        if not managed:
            managed = ManagedApp(app_id=app_id)
            self._running[app_id] = managed

        # Kill existing web if running
        if managed.web_process and managed.web_process.returncode is None:
            managed.web_process.terminate()
            await asyncio.sleep(1)

        proc = await asyncio.create_subprocess_exec(
            "npx", "expo", "start", "--web", "--port", str(port),
            cwd=app_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "EXPO_NO_TELEMETRY": "1", "BROWSER": "none"},
        )

        managed.web_port = port
        managed.web_process = proc
        # Port already claimed in _allocate_port

        # Start log reader
        asyncio.create_task(self._read_output(managed, proc, "web"))

        # Wait for server to actually be ready (up to 30s).
        # If the process dies before the socket binds, fail loudly — silently
        # returning a port for a dead process used to mark the app "running"
        # in DB while the preview proxy got connection-refused for hours.
        import socket
        for _ in range(30):
            await asyncio.sleep(1)
            if proc.returncode is not None:
                # Free the claimed port so the next start can reuse it.
                self._used_web_ports.discard(port)
                managed.web_port = None
                managed.web_process = None
                raise RuntimeError(
                    f"Expo web server for {app_id} exited early (code {proc.returncode}) — "
                    f"check log_buffer for crash details"
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    logger.info(f"[APP] Web server ready for {app_id} on port {port}")
                    return port
            except (OSError, ConnectionRefusedError):
                continue

        # Process is alive but socket never bound — leave it tracked so the
        # watchdog can monitor and respawn if needed, but warn loudly.
        logger.warning(f"[APP] Web server for {app_id} on port {port} did not bind in 30s")
        return port

    async def stop_app(self, app_id: str) -> bool:
        """Stop both Metro and web processes for an app."""
        managed = self._running.get(app_id)
        if not managed:
            return False

        stopped = False
        for proc, port_set, port in [
            (managed.metro_process, self._used_metro_ports, managed.metro_port),
            (managed.web_process, self._used_web_ports, managed.web_port),
        ]:
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                stopped = True
            if port:
                port_set.discard(port)

        managed.metro_process = None
        managed.web_process = None
        managed.metro_port = None
        managed.web_port = None
        del self._running[app_id]

        # Free disk: remove node_modules (~400MB per app).
        # package-lock.json is kept so npm install is deterministic on next start.
        try:
            app_dir = await self._resolve_app_dir(app_id)
            nm_dir = os.path.join(app_dir, "node_modules")
            if os.path.exists(nm_dir):
                shutil.rmtree(nm_dir)
                logger.info(f"[APP] Cleaned node_modules for {app_id}")
        except Exception as e:
            logger.debug(f"[APP] node_modules cleanup skipped for {app_id}: {e}")

        logger.info(f"[APP] Stopped {app_id}")
        return stopped

    async def restart_app(self, app_id: str, app_dir: Optional[str] = None) -> Dict[str, Any]:
        """Full restart: stop (with node_modules cleanup) + start from scratch.
        Use restart_processes() for hot restarts after file edits."""
        await self.stop_app(app_id)
        metro_port = await self.start_metro(app_id, app_dir=app_dir)
        web_port = await self.start_web(app_id, app_dir=app_dir)
        return {"metro_port": metro_port, "web_port": web_port}

    async def restart_processes(self, app_id: str) -> Dict[str, Any]:
        """Hot-restart: kill and respawn Metro + Web processes WITHOUT wiping node_modules.

        Used by app__restart tool after file edits. Typical restart time: 2-3 seconds.
        Falls back to full install if node_modules is missing (e.g. after crash/manual cleanup).
        """
        import socket

        t_start = time.time()
        managed = self._running.get(app_id)
        app_dir = await self._resolve_app_dir(app_id)

        # ── Gracefully kill existing processes ──
        if managed:
            for proc_name, proc in [("metro", managed.metro_process), ("web", managed.web_process)]:
                if proc and proc.returncode is None:
                    try:
                        proc.terminate()
                        await asyncio.wait_for(proc.wait(), timeout=5)
                        logger.debug(f"[APP] {proc_name} terminated gracefully for {app_id}")
                    except (asyncio.TimeoutError, ProcessLookupError):
                        try:
                            proc.kill()
                            logger.debug(f"[APP] {proc_name} killed (force) for {app_id}")
                        except ProcessLookupError:
                            pass
        else:
            managed = ManagedApp(app_id=app_id)
            self._running[app_id] = managed

        # ── Check node_modules — fall back to full install only if missing ──
        nm_dir = os.path.join(app_dir, "node_modules")
        if not os.path.exists(nm_dir):
            logger.warning(f"[APP] node_modules missing during restart for {app_id}, reinstalling...")
            await self._npm_install_with_retry(app_id, app_dir)

        # ── Reuse existing ports if possible, otherwise allocate new ones ──
        metro_port = managed.metro_port or self._allocate_port(METRO_PORT_RANGE, self._used_metro_ports)
        web_port = managed.web_port or self._allocate_port(WEB_PORT_RANGE, self._used_web_ports)

        # ── Spawn Metro ──
        metro_proc = await asyncio.create_subprocess_exec(
            "npx", "expo", "start", "--port", str(metro_port), "--lan",
            cwd=app_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "EXPO_NO_TELEMETRY": "1", "CI": "1"},
        )
        managed.metro_port = metro_port
        managed.metro_process = metro_proc
        managed.started_at = time.time()
        asyncio.create_task(self._read_output(managed, metro_proc, "metro"))

        # ── Spawn Web ──
        web_proc = await asyncio.create_subprocess_exec(
            "npx", "expo", "start", "--web", "--port", str(web_port),
            cwd=app_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "EXPO_NO_TELEMETRY": "1", "BROWSER": "none"},
        )
        managed.web_port = web_port
        managed.web_process = web_proc
        asyncio.create_task(self._read_output(managed, web_proc, "web"))

        # ── Wait for web server to be ready (up to 15s) ──
        for attempt in range(15):
            await asyncio.sleep(1)
            if web_proc.returncode is not None:
                logger.error(f"[APP] Web server exited early during restart for {app_id} (code {web_proc.returncode})")
                break
            try:
                with socket.create_connection(("127.0.0.1", web_port), timeout=1):
                    duration_ms = int((time.time() - t_start) * 1000)
                    logger.info(
                        "app_restart_complete app=%s duration_ms=%d wiped_node_modules=False",
                        app_id[:8], duration_ms,
                    )
                    return {"metro_port": metro_port, "web_port": web_port, "duration_ms": duration_ms}
            except (OSError, ConnectionRefusedError):
                continue

        duration_ms = int((time.time() - t_start) * 1000)
        logger.warning(
            "app_restart_slow app=%s duration_ms=%d wiped_node_modules=%s",
            app_id[:8], duration_ms, not os.path.exists(nm_dir),
        )
        return {"metro_port": metro_port, "web_port": web_port, "duration_ms": duration_ms}

    # ── Liveness + watchdog ────────────────────────────────────────
    #
    # Apps die. Expo crashes, Metro hangs, parent SIGKILL leaves zombies.
    # The original code returned a port from start_web/restart_processes
    # and never checked again — so a dead `web_process` left AppManager
    # confidently advertising a port that nothing was listening on, and
    # the preview proxy 503'd forever.
    #
    # `_probe_web_port` is a 1s TCP probe (cheaper than HTTP). It's the
    # single source of truth for "is this app's web server up." Both the
    # background watchdog and the preview proxy's lazy-revive path call
    # it before deciding whether to respawn.
    #
    # Cooldown: a crashing app could otherwise be respawned every 30s
    # by the watchdog. last_revive_at gates re-entry to once per 60s.

    REVIVE_COOLDOWN_S = 60
    WATCHDOG_INTERVAL_S = 30

    async def _probe_web_port(self, port: int, timeout: float = 1.0) -> bool:
        """TCP-probe 127.0.0.1:port — return True if accept()s a connection."""
        import socket
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: socket.create_connection(("127.0.0.1", port), timeout=timeout).close(),
            )
            return True
        except (OSError, ConnectionRefusedError, socket.timeout):
            return False

    async def ensure_web_alive(self, app_id: str) -> bool:
        """Probe app's web port; if dead, trigger a restart (with cooldown).

        Returns True if web was already alive OR became alive after revive.
        Returns False if app is unknown, on cooldown, or revive failed.
        Idempotent — safe to call from preview proxy on every miss.
        """
        managed = self._running.get(app_id)
        if not managed or not managed.web_port:
            return False

        if await self._probe_web_port(managed.web_port):
            return True

        now = time.time()
        if now - managed.last_revive_at < self.REVIVE_COOLDOWN_S:
            logger.debug(
                "[APP] revive skipped for %s (cooldown %ds remaining)",
                app_id[:8], int(self.REVIVE_COOLDOWN_S - (now - managed.last_revive_at)),
            )
            return False

        managed.last_revive_at = now
        logger.warning(
            "[APP] auto-revive triggered for %s — web port %d not responding",
            app_id[:8], managed.web_port,
        )
        try:
            await self.restart_processes(app_id)
        except Exception as e:
            logger.error("[APP] auto-revive failed for %s: %s", app_id[:8], e)
            return False
        return await self._probe_web_port(managed.web_port)

    async def watchdog_loop(self) -> None:
        """Background task: probe every running app and revive on failure.

        Runs forever once started. Started from agent_main.py lifespan.
        Skips silently when AppManager has no apps (warm-pool members).
        """
        logger.info("[APP] watchdog started (interval=%ds)", self.WATCHDOG_INTERVAL_S)
        while True:
            try:
                await asyncio.sleep(self.WATCHDOG_INTERVAL_S)
                # Snapshot keys — _running can mutate while we iterate.
                for app_id in list(self._running.keys()):
                    try:
                        await self.ensure_web_alive(app_id)
                    except Exception as e:
                        logger.warning("[APP] watchdog probe failed for %s: %s", app_id[:8], e)
            except asyncio.CancelledError:
                logger.info("[APP] watchdog stopped")
                raise
            except Exception as e:
                logger.error("[APP] watchdog loop error: %s", e, exc_info=True)
                # Don't die — sleep and continue.
                await asyncio.sleep(5)

    async def get_logs(self, app_id: str, lines: int = 50) -> List[str]:
        """Get recent logs from app's Metro + web processes."""
        managed = self._running.get(app_id)
        if not managed:
            return []
        return list(managed.log_buffer)[-lines:]

    async def get_qr_url(self, app_id: str) -> Optional[str]:
        """Get Expo Go QR URL for mobile preview."""
        managed = self._running.get(app_id)
        if not managed or not managed.metro_port:
            return None
        ip = await self._get_public_ip()
        return f"exp://{ip}:{managed.metro_port}"

    async def get_web_url(self, app_id: str) -> Optional[str]:
        """Get web preview URL (routed through platform proxy)."""
        managed = self._running.get(app_id)
        if not managed or not managed.web_port:
            return None
        # Return clean platform proxy URL using slug
        from app.config import settings
        slug = await self._get_app_slug(app_id)
        base = settings.platform_api_url.replace("/api", "")  # https://toup.ai
        return f"{base}/workspace/apps/{slug or app_id}"

    async def _get_app_slug(self, app_id: str) -> Optional[str]:
        """Look up app slug from DB."""
        try:
            from app.db.database import async_session_maker
            from app.db.models import App
            async with async_session_maker() as db:
                app = await db.get(App, app_id)
                return app.slug if app else None
        except Exception:
            return None

    # ── Database & Storage ──────────────────────────────

    async def setup_database(self, app_id: str, db_type: str = "sqlite", app_dir: Optional[str] = None) -> str:
        """Set up database for the app. Returns db path/url."""
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        assert os.path.exists(app_dir), f"app_dir {app_dir} does not exist"

        if db_type == "sqlite":
            # Create data directory and db file
            data_dir = os.path.join(app_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "app.db")

            # Generate helper file
            db_helper = '''import * as SQLite from 'expo-sqlite';

const db = SQLite.openDatabaseSync('app.db');

export async function initDatabase(): Promise<void> {
  await db.execAsync(`
    CREATE TABLE IF NOT EXISTS _migrations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      applied_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
  `);
}

export function getDatabase(): SQLite.SQLiteDatabase {
  return db;
}

export async function runQuery(sql: string, params: any[] = []): Promise<any[]> {
  const result = await db.getAllAsync(sql, params);
  return result;
}

export async function runExec(sql: string): Promise<void> {
  await db.execAsync(sql);
}

export default db;
'''
            lib_dir = os.path.join(app_dir, "src", "lib")
            os.makedirs(lib_dir, exist_ok=True)
            with open(os.path.join(lib_dir, "db.ts"), "w") as f:
                f.write(db_helper)

            logger.info(f"[APP] SQLite database set up for {app_id}")
            return db_path

        elif db_type == "supabase":
            # Generate Supabase client helper
            supabase_helper = '''import { createClient } from '@supabase/supabase-js';

// Set these in your environment or replace directly
const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL || '';
const SUPABASE_ANON_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || '';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

export default supabase;
'''
            lib_dir = os.path.join(app_dir, "src", "lib")
            os.makedirs(lib_dir, exist_ok=True)
            with open(os.path.join(lib_dir, "supabase.ts"), "w") as f:
                f.write(supabase_helper)

            logger.info(f"[APP] Supabase client set up for {app_id}")
            return "supabase"

        return ""

    async def setup_storage(self, app_id: str, app_dir: Optional[str] = None) -> str:
        """Create storage directory for the app."""
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        storage_dir = os.path.join(app_dir, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        return storage_dir

    # ── GitHub ──────────────────────────────────────────

    async def create_github_repo(self, app_id: str, app_name: str, app_dir: Optional[str] = None) -> Dict[str, str]:
        """Create a private GitHub repo and init git in app dir."""
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        assert os.path.exists(app_dir), f"app_dir {app_dir} does not exist"
        safe_name = re.sub(r"[^a-zA-Z0-9-]", "-", app_name.lower()).strip("-")[:40] or "app"

        # Init git
        await self._run_cmd(["git", "init"], cwd=app_dir)
        await self._run_cmd(["git", "add", "."], cwd=app_dir)
        await self._run_cmd(
            ["git", "commit", "-m", f"Initial commit: {app_name}"],
            cwd=app_dir,
            env={**os.environ, "GIT_AUTHOR_NAME": "Toup Agent", "GIT_AUTHOR_EMAIL": "agent@toup.ai",
                 "GIT_COMMITTER_NAME": "Toup Agent", "GIT_COMMITTER_EMAIL": "agent@toup.ai"},
        )

        # Try creating repo via gh CLI
        try:
            result = await self._run_cmd(
                ["gh", "repo", "create", safe_name, "--private", "--source", app_dir, "--push"],
                cwd=app_dir,
                timeout=30,
            )
            # Parse repo URL from output
            repo_url = ""
            for line in result.split("\n"):
                if "github.com" in line:
                    repo_url = line.strip()
                    break
            if not repo_url:
                repo_url = f"https://github.com/{safe_name}"

            return {"repo_name": safe_name, "repo_url": repo_url}
        except Exception as e:
            logger.warning(f"[APP] GitHub repo creation failed (gh CLI may not be configured): {e}")
            return {"repo_name": "", "repo_url": "", "error": str(e)}

    async def push_to_github(self, app_id: str, app_dir: Optional[str] = None) -> str:
        """Commit and push changes to GitHub."""
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        await self._run_cmd(["git", "add", "."], cwd=app_dir)
        await self._run_cmd(
            ["git", "commit", "-m", f"Update app ({time.strftime('%Y-%m-%d %H:%M')})"],
            cwd=app_dir,
            env={**os.environ, "GIT_AUTHOR_NAME": "Toup Agent", "GIT_AUTHOR_EMAIL": "agent@toup.ai",
                 "GIT_COMMITTER_NAME": "Toup Agent", "GIT_COMMITTER_EMAIL": "agent@toup.ai"},
        )
        result = await self._run_cmd(["git", "push"], cwd=app_dir, timeout=30)
        return result

    # ── Publishing ──────────────────────────────────────

    async def publish_web(self, app_id: str, domain: Optional[str] = None, app_dir: Optional[str] = None) -> str:
        """Export web build and serve. Returns URL."""
        app_dir = await self._resolve_app_dir(app_id, app_dir)

        # Export static web build
        await self._run_cmd(
            ["npx", "expo", "export", "--platform", "web"],
            cwd=app_dir, timeout=120,
        )

        dist_dir = os.path.join(app_dir, "dist")
        if not os.path.exists(dist_dir):
            raise RuntimeError("Web export failed — no dist/ directory created")

        ip = await self._get_public_ip()

        if domain:
            # TODO: Configure Caddy/nginx reverse proxy for custom domain
            logger.info(f"[APP] Custom domain {domain} requested — manual DNS setup required")
            return f"https://{domain}"

        # Serve via a simple static file server or existing web port
        return f"http://{ip}:{self._running.get(app_id, ManagedApp(app_id=app_id)).web_port or 4001}"

    # ── State Management ────────────────────────────────

    def get_status(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an app."""
        managed = self._running.get(app_id)
        if not managed:
            return None
        return {
            "app_id": app_id,
            "metro_port": managed.metro_port,
            "web_port": managed.web_port,
            "metro_running": managed.metro_process is not None and managed.metro_process.returncode is None,
            "web_running": managed.web_process is not None and managed.web_process.returncode is None,
            "uptime": time.time() - managed.started_at if managed.started_at else 0,
            "log_lines": len(managed.log_buffer),
        }

    def list_all(self) -> List[Dict[str, Any]]:
        """List all running apps."""
        return [self.get_status(aid) for aid in self._running if self.get_status(aid)]

    async def restore_on_startup(self, db_session_maker) -> int:
        """Restore previously-running apps from DB on agent restart."""
        from app.db.models import App as AppModel
        from sqlalchemy import select

        restored = 0
        try:
            async with db_session_maker() as db:
                result = await db.execute(
                    select(AppModel).where(AppModel.status.in_(["running", "ready"]))
                )
                apps = result.scalars().all()

                for app_row in apps:
                    app_dir = app_row.app_dir
                    if not os.path.exists(app_dir):
                        app_row.status = "error"
                        app_row.metro_pid = None
                        app_row.web_pid = None
                        continue

                    if app_row.status == "running":
                        try:
                            metro_port = await self.start_metro(app_row.id, app_dir=app_dir)
                            web_port = await self.start_web(app_row.id, app_dir=app_dir)
                            app_row.port = metro_port
                            app_row.web_port = web_port
                            managed = self._running.get(app_row.id)
                            if managed and managed.metro_process:
                                app_row.metro_pid = managed.metro_process.pid
                            if managed and managed.web_process:
                                app_row.web_pid = managed.web_process.pid
                            # Verify web server is actually responding
                            if web_port:
                                await self._wait_for_ready(managed, timeout=90)
                            restored += 1
                            logger.info(f"[APP] Restored {app_row.name} (metro={metro_port}, web={web_port})")
                        except Exception as e:
                            logger.error(f"[APP] Failed to restore {app_row.id}: {e}")
                            app_row.status = "error"

                await db.commit()
        except Exception as e:
            logger.error(f"[APP] Failed to restore apps: {e}")

        return restored

    async def cleanup(self) -> None:
        """Stop all running apps on shutdown."""
        app_ids = list(self._running.keys())
        for app_id in app_ids:
            try:
                await self.stop_app(app_id)
            except Exception as e:
                logger.error(f"[APP] Error stopping {app_id}: {e}")
        logger.info(f"[APP] Cleaned up {len(app_ids)} app(s)")

    # ── Internal Helpers ────────────────────────────────

    def _allocate_port(self, port_range: Tuple[int, int], used: Set[int]) -> int:
        """Find next free port in range. Claims it atomically (adds to `used` before returning)."""
        for port in range(port_range[0], port_range[1] + 1):
            if port not in used:
                used.add(port)  # Claim before returning — prevents TOCTOU race
                return port
        raise RuntimeError(f"No free ports in range {port_range[0]}-{port_range[1]}")

    async def _read_output(self, managed: ManagedApp, proc: asyncio.subprocess.Process, label: str) -> None:
        """Read stdout from a process and append to log buffer."""
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                managed.log_buffer.append(f"[{label}] {text}")
        except Exception:
            pass

    async def _wait_for_ready(self, managed: ManagedApp, timeout: int = 90) -> None:
        """Wait for Metro/Expo to report ready (log-based + HTTP probe)."""
        start = time.time()
        ready_patterns = ["Metro waiting on port", "Logs for your project", "Starting Metro", "Web is waiting on"]
        while time.time() - start < timeout:
            # Check recent logs for ready signal
            for line in list(managed.log_buffer)[-10:]:
                for pattern in ready_patterns:
                    if pattern.lower() in line.lower():
                        return
            # Also try HTTP probe on web port — catches cases where logs are delayed
            if managed.web_port:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                        f"http://127.0.0.1:{managed.web_port}/",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
                    if stdout.decode().strip() == "200":
                        return
                except Exception:
                    pass
            await asyncio.sleep(2)
        logger.warning(f"[APP] Metro ready timeout for {managed.app_id} after {timeout}s")

    async def _run_cmd(
        self, cmd: List[str], cwd: str = "/tmp",
        timeout: int = 60, env: Optional[dict] = None,
    ) -> str:
        """Run a shell command and return stdout."""
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = (stdout or b"").decode()
        if proc.returncode != 0:
            err = (stderr or b"").decode()
            logger.warning(f"[APP] Command failed: {' '.join(cmd)} → {err[:500]}")
        return output

    async def delete_app(self, app_id: str, app_dir: Optional[str] = None) -> bool:
        """Stop app and delete its directory."""
        stopped = await self.stop_app(app_id)
        if not stopped:
            # App not in _running (e.g. after agent restart) — kill orphan
            # processes by PID from DB record
            await self._kill_orphan_processes(app_id)
        app_dir = await self._resolve_app_dir(app_id, app_dir)
        if os.path.exists(app_dir):
            shutil.rmtree(app_dir)
            logger.info(f"[APP] Deleted {app_dir}")
            return True
        return False

    async def _kill_orphan_processes(self, app_id: str) -> None:
        """Kill Metro/web processes by PID from DB when not tracked in _running."""
        try:
            from app.db.database import async_session_maker
            from app.db.models.app import App
            async with async_session_maker() as db:
                app = await db.get(App, app_id)
                if not app:
                    return
                for pid in (app.metro_pid, app.web_pid):
                    if pid:
                        try:
                            os.kill(pid, signal.SIGTERM)
                            logger.info(f"[APP] Killed orphan process {pid} for {app_id}")
                        except ProcessLookupError:
                            pass
                        except OSError as e:
                            logger.debug(f"[APP] Could not kill PID {pid}: {e}")
        except Exception as e:
            logger.debug(f"[APP] Orphan cleanup failed for {app_id}: {e}")
