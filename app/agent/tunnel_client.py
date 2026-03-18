"""
Agent Tunnel Client — Connects the terminal agent to the platform.

When `toup run` starts, this client establishes a persistent WebSocket
connection to the platform. The platform routes voice tool calls through
this tunnel so they execute on the user's machine with full computer access.

Architecture:
  Terminal Agent ──WS (outbound)──→ wss://toup.ai/api/ws/agent-tunnel
"""

import asyncio
import json
import logging
from typing import Optional, TYPE_CHECKING

import websockets

if TYPE_CHECKING:
    from app.agent.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

RECONNECT_DELAY = 2  # seconds between reconnect attempts
MAX_RECONNECT_DELAY = 5  # cap low — Railway proxy kills WS connections periodically, so reconnect fast


class AgentTunnelClient:
    """Persistent WebSocket tunnel from terminal agent to platform."""

    def __init__(
        self,
        platform_url: str,
        auth_token: str,
        tool_executor: "ToolExecutor",
    ):
        # Convert HTTP URL to WebSocket URL
        ws_url = platform_url.replace("https://", "wss://").replace("http://", "ws://")
        # Remove trailing /api if present, we'll add the full path
        ws_url = ws_url.rstrip("/")
        if ws_url.endswith("/api"):
            ws_url = ws_url[:-4]
        self.ws_url = f"{ws_url}/api/ws/agent-tunnel?token={auth_token}"
        self._auth_token = auth_token

        self.tool_executor = tool_executor
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def start(self):
        """Start the tunnel connection (non-blocking, runs in background)."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[TUNNEL-CLIENT] Starting tunnel connection...")

    async def stop(self):
        """Stop the tunnel connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_loop(self):
        """Reconnecting event loop."""
        delay = RECONNECT_DELAY

        while self._running:
            try:
                await self._connect_and_listen()
                delay = RECONNECT_DELAY  # reset on clean disconnect
            except Exception as e:
                if not self._running:
                    break
                self._connected = False
                # Don't spam the terminal — just a brief note
                print(f"⟳ Tunnel reconnecting in {delay}s...")
                logger.debug("[TUNNEL-CLIENT] Connection lost: %s", e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)

    async def _connect_and_listen(self):
        """Connect to platform and handle messages."""
        # Mask the token in logs
        safe_url = self.ws_url.split("?")[0]
        logger.info("[TUNNEL-CLIENT] Connecting to %s", safe_url)

        async with websockets.connect(
            self.ws_url,
            max_size=10 * 1024 * 1024,
            ping_interval=15,   # Protocol-level pings every 15s (keeps proxies alive)
            ping_timeout=30,
        ) as ws:
            self._ws = ws
            self._connected = True
            logger.info("[TUNNEL-CLIENT] Connected to platform tunnel")

            # Send auth token as first message (fallback if query param is stripped by proxy)
            await ws.send(json.dumps({"type": "auth", "token": self._auth_token}))

            print("🔗 Connected to toup.ai — voice tools will execute locally")

            # Client-side heartbeat: send app-level pong every 15s
            # This ensures data flows through Railway's proxy regularly
            async def client_heartbeat():
                while True:
                    try:
                        await asyncio.sleep(15)
                        await ws.send(json.dumps({"type": "pong"}))
                    except Exception:
                        break

            hb_task = asyncio.create_task(client_heartbeat())

            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong"}))

                    elif msg_type == "connected":
                        logger.info("[TUNNEL-CLIENT] Platform confirmed connection")

                    elif msg_type == "tool_call":
                        # Execute tool in background (don't block the message loop)
                        asyncio.create_task(self._handle_tool_call(ws, msg))

                    elif msg_type == "config_update":
                        # Live settings sync: platform pushes updated .env
                        env_content = msg.get("env_content", "")
                        if env_content:
                            logger.info("[TUNNEL-CLIENT] Config update received — writing .env and restarting...")
                            print("📥 Settings updated from toup.ai — syncing config...")
                            try:
                                self._write_env(env_content)
                                await ws.send(json.dumps({"type": "status", "status": "config_written"}))
                                print("✅ Config synced — restarting agent...")
                            except Exception as e:
                                logger.exception("[TUNNEL-CLIENT] Failed to write .env: %s", e)
                                print(f"❌ Failed to write .env: {e}")
                                try:
                                    await ws.send(json.dumps({"type": "status", "status": "config_error", "error": str(e)}))
                                except Exception:
                                    pass
                                continue
                        else:
                            logger.warning("[TUNNEL-CLIENT] Empty config_update — treating as restart")
                            print("🔄 Restart signal from toup.ai...")
                        import os
                        os._exit(0)  # Service manager auto-restarts with new .env

                    elif msg_type == "http_forward":
                        asyncio.create_task(self._handle_http_forward(ws, msg))

                    elif msg_type == "restart":
                        logger.info("[TUNNEL-CLIENT] Restart command received — restarting agent...")
                        print("🔄 Settings changed on toup.ai — restarting agent...")
                        try:
                            await ws.send(json.dumps({"type": "status", "status": "restarting"}))
                        except Exception:
                            pass
                        import os
                        os._exit(0)  # Service manager auto-restarts

                    elif msg_type == "error":
                        logger.error("[TUNNEL-CLIENT] Platform error: %s", msg.get("message"))
            finally:
                hb_task.cancel()

        self._connected = False
        print("🔌 Disconnected from toup.ai tunnel")

    def _write_env(self, env_content: str):
        """Write updated .env file to the agent's working directory.

        Finds the .env file relative to the agent's working directory
        (same directory as agent_main.py). Creates a backup before overwriting.
        """
        import os
        import shutil

        # Find .env path: check cwd first, then script directory
        candidates = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"),
        ]

        env_path = None
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.exists(c):
                env_path = c
                break

        # Default to cwd if no existing .env found
        if not env_path:
            env_path = os.path.join(os.getcwd(), ".env")

        # Backup existing .env
        if os.path.exists(env_path):
            backup_path = env_path + ".bak"
            shutil.copy2(env_path, backup_path)
            logger.info("[TUNNEL-CLIENT] Backed up .env to %s", backup_path)

        # Write new .env
        with open(env_path, "w") as f:
            f.write(env_content)
        logger.info("[TUNNEL-CLIENT] Wrote updated .env to %s (%d bytes)", env_path, len(env_content))

    async def _handle_http_forward(self, ws, msg: dict):
        """Forward an HTTP request to the local agent server and return the result."""
        call_id = msg.get("id", "")
        method = msg.get("method", "GET")
        path = msg.get("path", "/")
        body = msg.get("body")

        try:
            import httpx
            # Agent runs on localhost — use the same port from settings
            from app.config import settings
            port = getattr(settings, 'port', 8001)
            url = f"http://127.0.0.1:{port}{path}"

            # Authenticate with the agent's own API key
            fwd_headers = {}
            if getattr(settings, 'agent_api_key', None):
                fwd_headers["X-Agent-Key"] = settings.agent_api_key

            async with httpx.AsyncClient(timeout=10) as client:
                if method == "GET":
                    resp = await client.get(url, headers=fwd_headers)
                elif method == "POST":
                    resp = await client.post(url, headers=fwd_headers, json=body or {})
                elif method == "DELETE":
                    resp = await client.delete(url, headers=fwd_headers)
                else:
                    resp = await client.get(url, headers=fwd_headers)

                result = resp.text
        except Exception as e:
            logger.warning("[TUNNEL-CLIENT] HTTP forward failed: %s %s → %s", method, path, e)
            result = json.dumps({"error": str(e)})

        try:
            await ws.send(json.dumps({
                "type": "tool_result",
                "id": call_id,
                "result": result,
            }))
        except Exception:
            pass

    async def _handle_tool_call(self, ws, msg: dict):
        """Execute a tool call from the platform and send the result back."""
        call_id = msg.get("id", "")
        tool_name = msg.get("tool_name", "")
        arguments = msg.get("arguments", {})
        user_id = msg.get("user_id", "")

        print(f"🔧 [TUNNEL] Executing tool: {tool_name}({arguments})")

        try:
            if user_id:
                self.tool_executor._current_user_id = user_id

            result = await self.tool_executor.execute(tool_name, arguments)

            await ws.send(json.dumps({
                "type": "tool_result",
                "id": call_id,
                "result": result,
            }))
            print(f"✅ [TUNNEL] Tool {tool_name} completed (result: {str(result)[:200]})")

        except Exception as e:
            logger.exception("[TUNNEL-CLIENT] Tool %s failed", tool_name)
            try:
                await ws.send(json.dumps({
                    "type": "tool_result",
                    "id": call_id,
                    "result": f"ERROR: {e}",
                }))
            except Exception:
                pass
