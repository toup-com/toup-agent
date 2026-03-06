"""
WebSocket Browser Endpoint — Agentic browser control.

The user sends natural language commands ("find me a laptop on amazon").
The agent autonomously browses: navigates, clicks, types, scrolls, reads pages.
Screenshots are streamed back after every action so the user sees live progress.

Protocol:
  Client sends JSON:
    { "type": "browser_action", "action": "tab_open", "url": "..." }
    { "type": "browser_action", "action": "navigate", "url": "..." }
    { "type": "browser_action", "action": "click", "x": 100, "y": 200 }
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
"""

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket Browser"])

# References set at startup
_agent_runner = None
_skill_loader = None


def set_ws_browser_refs(agent_runner, skill_loader=None):
    global _agent_runner, _skill_loader
    _agent_runner = agent_runner
    _skill_loader = skill_loader


async def _authenticate_ws(token: str) -> Optional[str]:
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


# ---------------------------------------------------------------------------
# Browser tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

BROWSER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate the browser to a URL. Use this to open websites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to (e.g. https://www.google.com)"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click on an element identified by CSS selector, text content, or coordinates. Prefer selector or text over coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the element to click"},
                    "text": {"type": "string", "description": "Visible text of the element to click (partial match)"},
                    "x": {"type": "number", "description": "X coordinate to click (use as last resort)"},
                    "y": {"type": "number", "description": "Y coordinate to click (use as last resort)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into an input field. First specify the target element, then the text to type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the input element"},
                    "text": {"type": "string", "description": "Text to type into the field"},
                    "press_enter": {"type": "boolean", "description": "Press Enter after typing (default: false)"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the page up or down to see more content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction"},
                    "amount": {"type": "integer", "description": "Pixels to scroll (default: 500)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the current page's text content and interactive elements. Use this to understand what's on the page before interacting.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Go back to the previous page in browser history.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a short time for the page to load or update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "milliseconds": {"type": "integer", "description": "Time to wait in ms (default: 1000)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call this when you have completed the user's request. Include a summary of what you did/found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of what you accomplished for the user"},
                },
                "required": ["summary"],
            },
        },
    },
]

BROWSER_SYSTEM_PROMPT = """You are an AI browser agent. You control a real web browser to help the user accomplish tasks on the web.

You can navigate to any website, click elements, type into forms, scroll pages, and read page content. You work autonomously — the user tells you what they want and you figure out the steps.

## How to browse
1. Navigate to the relevant website
2. Read the page to understand what's on it
3. Interact: click links/buttons, fill forms, scroll to find content
4. Continue until the task is done, then call `done` with a summary

## Rules
- Always call `read_page` after navigating or clicking to see the updated content
- When searching, type the query and press Enter
- If a page is complex, scroll down to find more content
- If something doesn't work, try a different approach
- When you've accomplished the task, call `done` with a clear summary
- Be efficient — don't waste steps on unnecessary actions
- For search engines, prefer Google (https://www.google.com/search?q=...)"""


# ---------------------------------------------------------------------------
# Browser tool executor — runs actions on the real Playwright browser
# ---------------------------------------------------------------------------

async def _exec_browser_tool(
    name: str,
    args: Dict[str, Any],
    page,
    tab_manager,
    browser_mod,
) -> str:
    """Execute a browser tool and return the text result."""
    try:
        if name == "navigate":
            url = args.get("url", "").strip()
            if not url:
                return "ERROR: url is required"
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            title = await page.title()
            return f"Navigated to {page.url} — {title}"

        elif name == "click":
            selector = args.get("selector")
            text = args.get("text")
            x, y = args.get("x"), args.get("y")

            if selector:
                await page.click(selector, timeout=5_000)
            elif text:
                # Click by visible text
                loc = page.get_by_text(text, exact=False).first
                await loc.click(timeout=5_000)
            elif x is not None and y is not None:
                await page.mouse.click(float(x), float(y))
            else:
                return "ERROR: provide selector, text, or x/y coordinates"

            await asyncio.sleep(0.5)
            await _wait_stable(page)
            title = await page.title()
            return f"Clicked. Now on: {page.url} — {title}"

        elif name == "type_text":
            text_to_type = args.get("text", "")
            selector = args.get("selector")
            press_enter = args.get("press_enter", False)

            if selector:
                await page.fill(selector, text_to_type, timeout=5_000)
                if press_enter:
                    await page.press(selector, "Enter")
            else:
                # Type into the focused element
                await page.keyboard.type(text_to_type, delay=30)
                if press_enter:
                    await page.keyboard.press("Enter")

            await asyncio.sleep(0.3)
            if press_enter:
                await _wait_stable(page)
            return f"Typed '{text_to_type[:50]}'" + (" and pressed Enter" if press_enter else "")

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = int(args.get("amount", 500))
            delta = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta)
            await asyncio.sleep(0.5)
            return f"Scrolled {direction} by {amount}px"

        elif name == "read_page":
            title = await page.title()
            url = page.url
            # Get text content (truncated)
            body_text = await page.inner_text("body")
            if len(body_text) > 4000:
                body_text = body_text[:4000] + "\n...(truncated)"
            # Get interactive elements
            elements = await _get_interactive_elements(page)
            return (
                f"URL: {url}\nTitle: {title}\n\n"
                f"--- Page Content ---\n{body_text}\n\n"
                f"--- Interactive Elements ---\n{elements}"
            )

        elif name == "go_back":
            await page.go_back(wait_until="domcontentloaded", timeout=10_000)
            title = await page.title()
            return f"Went back to: {page.url} — {title}"

        elif name == "wait":
            ms = int(args.get("milliseconds", 1000))
            await asyncio.sleep(ms / 1000)
            return f"Waited {ms}ms"

        elif name == "done":
            return args.get("summary", "Task completed.")

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


async def _wait_stable(page, timeout: int = 3000):
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except Exception:
        pass
    await asyncio.sleep(0.3)


async def _get_interactive_elements(page, limit: int = 40) -> str:
    """Extract clickable/interactive elements for the agent."""
    try:
        elements = await page.evaluate("""() => {
            const items = [];
            const selectors = 'a, button, input, select, textarea, [role="button"], [role="link"], [role="tab"], [onclick]';
            document.querySelectorAll(selectors).forEach((el, i) => {
                if (i >= 60) return;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                const tag = el.tagName.toLowerCase();
                const type = el.type || '';
                const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().slice(0, 80);
                const href = el.href || '';
                items.push({ tag, type, text, href: href.slice(0, 120) });
            });
            return items;
        }""")
        lines = []
        for i, el in enumerate(elements[:limit]):
            parts = [f"[{i}] <{el['tag']}>"]
            if el.get("type"):
                parts.append(f"type={el['type']}")
            if el.get("text"):
                parts.append(f'"{el["text"]}"')
            if el.get("href"):
                parts.append(f"-> {el['href']}")
            lines.append(" ".join(parts))
        return "\n".join(lines) if lines else "(no interactive elements found)"
    except Exception as e:
        return f"(could not extract elements: {e})"


# ---------------------------------------------------------------------------
# Agentic browser loop
# ---------------------------------------------------------------------------

async def _run_browser_agent(
    websocket: WebSocket,
    page,
    tab_manager,
    browser_mod,
    user_message: str,
    conversation: List[Dict[str, Any]],
):
    """
    Agentic loop: call LLM with browser tools, execute tool calls,
    send screenshots, repeat until agent calls `done` or gives a text response.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = settings.agent_model

    # Add user message
    conversation.append({"role": "user", "content": user_message})

    max_steps = 15  # Safety limit

    for step in range(max_steps):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": BROWSER_SYSTEM_PROMPT}] + conversation,
                tools=BROWSER_TOOLS,
                max_completion_tokens=2048,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("[WS Browser] LLM call failed: %s", e)
            await websocket.send_json({"type": "error", "message": f"LLM error: {e}"})
            return

        choice = response.choices[0]
        msg = choice.message

        # If the model returned text without tool calls — we're done
        if not msg.tool_calls:
            text = msg.content or ""
            conversation.append({"role": "assistant", "content": text})
            # Stream the text
            if text:
                await websocket.send_json({"type": "agent_message", "content": text})
            return

        # Process tool calls
        # Store assistant message with tool_calls
        conversation.append(msg.model_dump())

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("[WS Browser] Tool call: %s(%s)", fn_name, json.dumps(fn_args)[:200])

            # Notify client of tool use
            await websocket.send_json({"type": "tool_use", "tool": fn_name})

            # Execute the browser tool
            result = await _exec_browser_tool(fn_name, fn_args, page, tab_manager, browser_mod)

            # Add tool result to conversation
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:4000],  # Truncate long results
            })

            # Send screenshot after each action (except read_page and wait)
            if fn_name not in ("read_page", "wait", "done"):
                await _send_screenshot(websocket, page)
                await _send_state(websocket, page, tab_manager)

            # If the agent called "done", send the summary and return
            if fn_name == "done":
                await websocket.send_json({
                    "type": "agent_message",
                    "content": fn_args.get("summary", "Task completed."),
                })
                # Final screenshot
                await _send_screenshot(websocket, page)
                return

    # Hit max steps
    await websocket.send_json({
        "type": "agent_message",
        "content": "I've taken many steps. Let me know if you'd like me to continue.",
    })
    await _send_screenshot(websocket, page)


# ---------------------------------------------------------------------------
# Screenshot / state helpers
# ---------------------------------------------------------------------------

async def _send_screenshot(websocket: WebSocket, page):
    try:
        png_bytes = await page.screenshot(type="png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        await websocket.send_json({"type": "screenshot", "image": b64})
    except Exception as e:
        logger.warning("[WS Browser] Screenshot failed: %s", e)


async def _send_state(websocket: WebSocket, page, tab_manager):
    try:
        await websocket.send_json({
            "type": "browser_state",
            "url": page.url,
            "title": await page.title(),
            "tab_id": "active",
        })
    except Exception as e:
        logger.warning("[WS Browser] send_state error: %s", e)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/browser")
async def ws_browser(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    agent_key: Optional[str] = Query(None),
):
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
    browser_mod = None
    try:
        from app.agent import browser as browser_mod
        from app.agent.browser import _get_browser, get_tab_manager
    except ImportError as e:
        logger.warning(f"[WS Browser] Browser module not available: {e}")
        browser_available = False

    tab_manager = get_tab_manager() if browser_available else None
    active_page = None
    chat_task: Optional[asyncio.Task] = None
    conversation: List[Dict[str, Any]] = []  # Persistent conversation for agentic loop

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
                    active_page = tab_manager.get_tab(tab_id)
                    await _send_state(websocket, active_page, tab_manager)
                    await asyncio.sleep(0.5)
                    await _send_screenshot(websocket, active_page)

                elif action == "navigate":
                    if not browser_available:
                        await websocket.send_json({"type": "error", "message": "Browser not available"})
                        continue
                    url = data.get("url", "")
                    if not url:
                        continue
                    if not active_page:
                        browser = await _get_browser()
                        tab_id = await tab_manager.open_tab(browser, url)
                        active_page = tab_manager.get_tab(tab_id)
                    else:
                        if not url.startswith(("http://", "https://", "about:")):
                            url = "https://" + url
                        await active_page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    await _send_state(websocket, active_page, tab_manager)
                    await asyncio.sleep(0.5)
                    await _send_screenshot(websocket, active_page)

                elif action == "screenshot":
                    if active_page:
                        await _send_screenshot(websocket, active_page)

                elif action == "click":
                    x = data.get("x", 0)
                    y = data.get("y", 0)
                    if active_page:
                        await active_page.mouse.click(x, y)
                        await asyncio.sleep(0.5)
                        await _send_state(websocket, active_page, tab_manager)
                        await _send_screenshot(websocket, active_page)

                elif action == "back":
                    if active_page:
                        await active_page.go_back(timeout=10_000)
                        await _send_state(websocket, active_page, tab_manager)
                        await asyncio.sleep(0.5)
                        await _send_screenshot(websocket, active_page)

                elif action == "forward":
                    if active_page:
                        await active_page.go_forward(timeout=10_000)
                        await _send_state(websocket, active_page, tab_manager)
                        await asyncio.sleep(0.5)
                        await _send_screenshot(websocket, active_page)

                elif action == "chat":
                    message = data.get("message", "").strip()
                    if not message:
                        continue
                    if not browser_available:
                        await websocket.send_json({"type": "error", "message": "Browser not available"})
                        continue

                    # Ensure we have a page
                    if not active_page:
                        browser = await _get_browser()
                        tab_id = await tab_manager.open_tab(browser)
                        active_page = tab_manager.get_tab(tab_id)

                    if chat_task and not chat_task.done():
                        await websocket.send_json({"type": "error", "message": "Agent is still working on the previous request"})
                    else:
                        await websocket.send_json({"type": "agent_thinking"})
                        chat_task = asyncio.create_task(
                            _run_browser_agent(
                                websocket, active_page, tab_manager, browser_mod,
                                message, conversation,
                            )
                        )

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
        if chat_task and not chat_task.done():
            chat_task.cancel()
