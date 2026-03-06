"""
WebSocket Browser Endpoint — Real-time browser control and screenshot streaming.

Protocol:
  Client sends JSON:
    { "type": "browser_action", "action": "tab_open", "url": "..." }
    { "type": "browser_action", "action": "navigate", "url": "..." }
    { "type": "browser_action", "action": "click", "x": 100, "y": 200 }
    { "type": "browser_action", "action": "type", "text": "..." }
    { "type": "browser_action", "action": "screenshot" }
    { "type": "browser_action", "action": "back" }
    { "type": "browser_action", "action": "forward" }
    { "type": "browser_action", "action": "chat", "message": "..." }
    { "type": "ping" }

  Server sends JSON:
    { "type": "browser_state", "url": "...", "title": "...", "tab_id": "..." }
    { "type": "screenshot", "image": "<base64 png>" }
    { "type": "agent_message", "content": "..." }
    { "type": "agent_thinking" }
    { "type": "agent_stream", "token": "..." }
    { "type": "tool_use", "tool": "..." }
    { "type": "error", "message": "..." }
    { "type": "pong" }

Authentication:
  Connect with token as query param: ws://host/api/ws/browser?token=JWT_TOKEN
"""

import asyncio
import base64
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket Browser"])

# References set at startup
_agent_runner = None
_skill_loader = None


def set_ws_browser_refs(agent_runner, skill_loader=None):
    """Set references to the agent runner (called from agent_main.py lifespan)."""
    global _agent_runner, _skill_loader
    _agent_runner = agent_runner
    _skill_loader = skill_loader


async def _authenticate_ws(token: str) -> Optional[str]:
    """Validate a JWT token and return the user_id, or None."""
    try:
        from app.services import decode_access_token, get_user_by_id
        from app.db.database import async_session_maker

        user_id = decode_access_token(token)
        if not user_id:
            return None

        async with async_session_maker() as db:
            user = await get_user_by_id(db, user_id)
            if user and user.is_active:
                return user.id
        return None
    except Exception as e:
        logger.warning(f"WS browser auth failed: {e}")
        return None


@router.websocket("/ws/browser")
async def ws_browser(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    agent_key: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time browser control.

    Streams screenshots, handles navigation, clicks, typing,
    and agent chat for autonomous browsing.
    """
    await websocket.accept()
    user_id: Optional[str] = None

    # Auth via agent_key (platform proxy mode)
    if agent_key and settings.agent_api_key and agent_key == settings.agent_api_key:
        user_id = settings.user_id

    # Auth via JWT
    if not user_id and token:
        user_id = await _authenticate_ws(token)

    if not user_id:
        await websocket.send_json({"type": "error", "message": "Authentication required"})
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Import browser module
    browser_available = True
    try:
        from app.agent.browser import (
            _get_browser, get_tab_manager, screenshot as browser_screenshot,
            navigate as browser_navigate, run_action, ai_snapshot,
        )
    except ImportError as e:
        logger.warning(f"[WS Browser] Browser module not available: {e}")
        browser_available = False

    tab_manager = get_tab_manager() if browser_available else None
    active_tab_id: Optional[str] = None
    session_id: Optional[str] = None
    chat_task: Optional[asyncio.Task] = None

    async def send_state(tab_id: str):
        """Send current browser state to client."""
        if not tab_manager:
            return
        page = tab_manager.get_tab(tab_id)
        if not page:
            return
        try:
            title = await page.title()
            url = page.url
            await websocket.send_json({
                "type": "browser_state",
                "url": url,
                "title": title,
                "tab_id": tab_id,
            })
        except Exception as e:
            logger.warning(f"[WS Browser] send_state error: {e}")

    async def send_screenshot(tab_id: str):
        """Take and send a screenshot as base64."""
        if not tab_manager:
            return
        page = tab_manager.get_tab(tab_id)
        if not page:
            return
        try:
            png_bytes = await page.screenshot(type="png")
            b64 = base64.b64encode(png_bytes).decode("ascii")
            await websocket.send_json({
                "type": "screenshot",
                "image": b64,
            })
        except Exception as e:
            logger.warning(f"[WS Browser] screenshot error: {e}")

    async def handle_chat(message: str):
        """Send user message to agent with browser context."""
        nonlocal session_id, active_tab_id

        if not _agent_runner:
            await websocket.send_json({"type": "error", "message": "Agent not available"})
            return

        try:
            await websocket.send_json({"type": "agent_thinking"})
        except Exception:
            return

        # Build browser context for the agent
        browser_context = ""
        if browser_available and active_tab_id and tab_manager:
            page = tab_manager.get_tab(active_tab_id)
            if page:
                try:
                    title = await page.title()
                    url = page.url
                    snapshot = await ai_snapshot(page, format="ai")
                    browser_context = (
                        f"\n\n[BROWSER CONTEXT — the user is viewing this in the Toup Browser]\n"
                        f"Current URL: {url}\n"
                        f"Page title: {title}\n"
                        f"Tab ID: {active_tab_id}\n"
                        f"Accessibility snapshot:\n{snapshot[:3000]}\n"
                        f"You can use the 'browser' tool to interact with this page.\n"
                        f"[/BROWSER CONTEXT]"
                    )
                except Exception as e:
                    logger.warning("[WS Browser] Browser context error: %s", e)

        augmented_message = message + browser_context

        # Stream callbacks (same signature as ws_chat.py)
        async def on_text_chunk(chunk: str):
            try:
                await websocket.send_json({"type": "agent_stream", "token": chunk})
            except Exception:
                pass

        async def on_tool_start(tool_name: str):
            try:
                await websocket.send_json({"type": "tool_use", "tool": tool_name})
            except Exception:
                pass

        async def on_tool_end(tool_name: str, summary: str):
            pass

        try:
            result = await _agent_runner.run(
                user_id=user_id,
                user_message=augmented_message,
                session_id=session_id,
                on_text_chunk=on_text_chunk,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )
            session_id = result.session_id
            reply = result.text or ""
            await websocket.send_json({
                "type": "agent_message",
                "content": reply,
                "session_id": session_id,
            })
            # After agent finishes, take a fresh screenshot
            if active_tab_id:
                await asyncio.sleep(0.5)
                await send_screenshot(active_tab_id)
                await send_state(active_tab_id)
        except Exception as e:
            logger.exception("[WS Browser] Agent error")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    # ── Main message loop ──
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type", "")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type != "browser_action":
                continue

            action = data.get("action", "")

            try:
                if action == "tab_open":
                    if not browser_available:
                        await websocket.send_json({"type": "error", "message": "Browser not available on this agent"})
                        continue
                    url = data.get("url", "about:blank")
                    browser = await _get_browser()
                    tab_id = await tab_manager.open_tab(browser, url)
                    active_tab_id = tab_id
                    await send_state(tab_id)
                    await asyncio.sleep(0.5)
                    await send_screenshot(tab_id)

                elif action == "navigate":
                    if not browser_available:
                        await websocket.send_json({"type": "error", "message": "Browser not available on this agent"})
                        continue
                    url = data.get("url", "")
                    if not url:
                        continue
                    if not active_tab_id:
                        browser = await _get_browser()
                        active_tab_id = await tab_manager.open_tab(browser, url)
                    else:
                        page = tab_manager.get_tab(active_tab_id)
                        if page:
                            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    await send_state(active_tab_id)
                    await asyncio.sleep(0.5)
                    await send_screenshot(active_tab_id)

                elif action == "screenshot":
                    if active_tab_id:
                        await send_screenshot(active_tab_id)

                elif action == "click":
                    x = data.get("x", 0)
                    y = data.get("y", 0)
                    if active_tab_id and tab_manager:
                        page = tab_manager.get_tab(active_tab_id)
                        if page:
                            await page.mouse.click(x, y)
                            await asyncio.sleep(0.5)
                            await send_state(active_tab_id)
                            await send_screenshot(active_tab_id)

                elif action == "type":
                    text = data.get("text", "")
                    if active_tab_id and text and tab_manager:
                        page = tab_manager.get_tab(active_tab_id)
                        if page:
                            await page.keyboard.type(text)
                            await asyncio.sleep(0.3)
                            await send_screenshot(active_tab_id)

                elif action == "back":
                    if active_tab_id and tab_manager:
                        page = tab_manager.get_tab(active_tab_id)
                        if page:
                            await page.go_back(timeout=10000)
                            await send_state(active_tab_id)
                            await asyncio.sleep(0.5)
                            await send_screenshot(active_tab_id)

                elif action == "forward":
                    if active_tab_id and tab_manager:
                        page = tab_manager.get_tab(active_tab_id)
                        if page:
                            await page.go_forward(timeout=10000)
                            await send_state(active_tab_id)
                            await asyncio.sleep(0.5)
                            await send_screenshot(active_tab_id)

                elif action == "chat":
                    message = data.get("message", "")
                    if message:
                        # Run chat as background task so pings keep flowing
                        if chat_task and not chat_task.done():
                            await websocket.send_json({"type": "error", "message": "Agent is still processing the previous message"})
                        else:
                            chat_task = asyncio.create_task(handle_chat(message))

                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})

            except Exception as e:
                logger.exception(f"[WS Browser] Action error: {action}")
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("[WS Browser] Client disconnected")
    except Exception as e:
        logger.exception("[WS Browser] Unexpected error")
    finally:
        # Cancel any running chat task
        if chat_task and not chat_task.done():
            chat_task.cancel()
        # Don't close tabs on disconnect — keep them alive for reconnection
