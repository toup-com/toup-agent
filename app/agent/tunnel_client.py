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

RECONNECT_DELAY = 5  # seconds between reconnect attempts
MAX_RECONNECT_DELAY = 60


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
                logger.warning("[TUNNEL-CLIENT] Connection lost: %s. Reconnecting in %ds...", e, delay)
                self._connected = False
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
            ping_interval=None,  # Disable protocol-level pings; platform sends app-level heartbeat
            ping_timeout=None,
        ) as ws:
            self._ws = ws
            self._connected = True
            logger.info("[TUNNEL-CLIENT] Connected to platform tunnel")

            # Send auth token as first message (fallback if query param is stripped by proxy)
            await ws.send(json.dumps({"type": "auth", "token": self._auth_token}))

            print("🔗 Connected to toup.ai — voice tools will execute locally")

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

                elif msg_type == "error":
                    logger.error("[TUNNEL-CLIENT] Platform error: %s", msg.get("message"))

        self._connected = False
        print("🔌 Disconnected from toup.ai tunnel")

    async def _handle_tool_call(self, ws, msg: dict):
        """Execute a tool call from the platform and send the result back."""
        call_id = msg.get("id", "")
        tool_name = msg.get("tool_name", "")
        arguments = msg.get("arguments", {})
        user_id = msg.get("user_id", "")

        logger.info("[TUNNEL-CLIENT] Executing tool: %s(%s)", tool_name, arguments)

        try:
            if user_id:
                self.tool_executor._current_user_id = user_id

            result = await self.tool_executor.execute(tool_name, arguments)

            await ws.send(json.dumps({
                "type": "tool_result",
                "id": call_id,
                "result": result,
            }))
            logger.info("[TUNNEL-CLIENT] Tool %s completed", tool_name)

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
