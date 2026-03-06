"""
WebSocket Browser Endpoint — Agentic browser control.

The user sends natural language commands ("find me a laptop on amazon").
The agent autonomously browses: navigates, clicks, types, scrolls, reads pages.
Screenshots stream after every action so the user sees live progress.
A visible cursor overlay shows exactly where the agent is clicking/typing.

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
    { "type": "screenshot", "image": "<base64 png>", "cursor": { "x": ..., "y": ... } }
    { "type": "agent_message", "content": "..." }
    { "type": "agent_thinking" }
    { "type": "tool_use", "tool": "...", "args": {...} }
    { "type": "cursor_move", "x": ..., "y": ... }
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
# Cursor overlay — injected into page so it appears in screenshots
# ---------------------------------------------------------------------------

CURSOR_INJECT_JS = """
(function() {
  let cursor = document.getElementById('__agent_cursor__');
  if (!cursor) {
    cursor = document.createElement('div');
    cursor.id = '__agent_cursor__';
    cursor.innerHTML = `
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M5 3L19 12L12 13L9 20L5 3Z" fill="rgba(139,92,246,0.95)" stroke="white" stroke-width="1.5" stroke-linejoin="round"/>
      </svg>
      <div style="position:absolute;top:28px;left:12px;background:rgba(139,92,246,0.9);color:white;font-size:11px;padding:2px 8px;border-radius:4px;white-space:nowrap;font-family:sans-serif;pointer-events:none;">Agent</div>
    `;
    cursor.style.cssText = 'position:fixed;z-index:999999;pointer-events:none;transition:left 0.3s ease,top 0.3s ease;left:-100px;top:-100px;';
    document.body.appendChild(cursor);
  }
  return true;
})()
"""

CURSOR_MOVE_JS = """
(function(x, y) {
  const cursor = document.getElementById('__agent_cursor__');
  if (cursor) {
    cursor.style.left = x + 'px';
    cursor.style.top = y + 'px';
  }
})(%d, %d)
"""

CURSOR_HIDE_JS = """
(function() {
  const cursor = document.getElementById('__agent_cursor__');
  if (cursor) {
    cursor.style.left = '-100px';
    cursor.style.top = '-100px';
  }
})()
"""

# Highlight element before clicking — adds a brief glow effect
HIGHLIGHT_ELEMENT_JS = """
(function(selector) {
  try {
    const el = document.querySelector(selector);
    if (el) {
      const rect = el.getBoundingClientRect();
      let highlight = document.getElementById('__agent_highlight__');
      if (!highlight) {
        highlight = document.createElement('div');
        highlight.id = '__agent_highlight__';
        highlight.style.cssText = 'position:fixed;z-index:999998;pointer-events:none;border:2px solid rgba(139,92,246,0.8);border-radius:4px;background:rgba(139,92,246,0.1);transition:all 0.2s ease;';
        document.body.appendChild(highlight);
      }
      highlight.style.left = rect.left + 'px';
      highlight.style.top = rect.top + 'px';
      highlight.style.width = rect.width + 'px';
      highlight.style.height = rect.height + 'px';
      highlight.style.opacity = '1';
      setTimeout(() => { highlight.style.opacity = '0'; }, 1500);
      return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    }
  } catch(e) {}
  return null;
})(%s)
"""

GET_ELEMENT_CENTER_JS = """
(function(selector) {
  try {
    const el = document.querySelector(selector);
    if (el) {
      const rect = el.getBoundingClientRect();
      return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    }
  } catch(e) {}
  return null;
})(%s)
"""

GET_TEXT_ELEMENT_CENTER_JS = """
(function(text) {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.textContent && node.textContent.trim().toLowerCase().includes(text.toLowerCase())) {
      const el = node.parentElement;
      if (el) {
        const rect = el.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
        }
      }
    }
  }
  return null;
})(%s)
"""


# ---------------------------------------------------------------------------
# Browser tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

BROWSER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate the browser to a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to (e.g. https://www.google.com)"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element. Use selector or text to target it. The cursor will move to the element visually before clicking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of element to click"},
                    "text": {"type": "string", "description": "Visible text of the element to click (partial match)"},
                    "index": {"type": "integer", "description": "Index from the interactive elements list (e.g. 0, 1, 2...)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into an input field. The cursor moves to the field before typing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the input element"},
                    "text": {"type": "string", "description": "Text to type"},
                    "clear": {"type": "boolean", "description": "Clear field before typing (default: true)"},
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
            "description": "Scroll the page to see more content.",
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
            "name": "go_back",
            "description": "Go back to the previous page.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for the page to load/update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "milliseconds": {"type": "integer", "description": "Time to wait in ms (default: 2000)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call when you have completed the user's task. Include a summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of what you accomplished"},
                },
                "required": ["summary"],
            },
        },
    },
]

BROWSER_SYSTEM_PROMPT = """You are an AI browser agent controlling a real headless browser. You help users accomplish tasks on the web.

## Your workflow for EVERY action:
1. After navigating to any page, you will automatically receive a screenshot and page analysis
2. Study the screenshot and interactive elements carefully
3. Decide what to click, type, or scroll based on what you see
4. After each action, you'll get a new screenshot — analyze it before the next step
5. Continue until done, then call `done` with a summary

## Rules:
- You ALWAYS get a screenshot + page content after every action — use this to decide your next step
- Use CSS selectors (preferred) or text content to target elements
- When searching on a site, type the query and set press_enter=true
- If a page has lots of content, scroll down to find what you need
- If something doesn't work (element not found, etc.), try a different selector or approach
- When clicking buttons/links, use the selector from the interactive elements list
- For Google, use: https://www.google.com/search?q=YOUR+QUERY
- Be methodical: navigate → see page → interact → see result → continue
- When done, call `done` with a clear summary of what you found/did"""


# ---------------------------------------------------------------------------
# Page analysis — extracts text + interactive elements for the LLM
# ---------------------------------------------------------------------------

async def _analyze_page(page) -> str:
    """Get page content + interactive elements for LLM context."""
    try:
        title = await page.title()
        url = page.url

        # Get visible text
        body_text = await page.inner_text("body")
        if len(body_text) > 5000:
            body_text = body_text[:5000] + "\n...(truncated)"

        # Get interactive elements with selectors
        elements = await _get_interactive_elements(page)

        return (
            f"Current page: {url}\n"
            f"Title: {title}\n\n"
            f"--- Visible Content ---\n{body_text}\n\n"
            f"--- Interactive Elements ---\n{elements}"
        )
    except Exception as e:
        return f"Page analysis failed: {e}"


async def _get_interactive_elements(page, limit: int = 50) -> str:
    """Extract clickable/interactive elements with their CSS selectors."""
    try:
        elements = await page.evaluate("""() => {
            const items = [];
            const selectors = 'a, button, input, select, textarea, [role="button"], [role="link"], [role="tab"], [role="menuitem"], [onclick], [type="submit"]';
            document.querySelectorAll(selectors).forEach((el, i) => {
                if (i >= 80) return;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                if (rect.top > window.innerHeight + 200) return;

                const tag = el.tagName.toLowerCase();
                const type = el.type || '';
                const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().slice(0, 80);
                const href = el.href || '';
                const id = el.id || '';
                const name = el.name || '';
                const cls = el.className ? (typeof el.className === 'string' ? el.className.split(' ').slice(0,3).join('.') : '') : '';

                // Build a usable CSS selector
                let sel = tag;
                if (id) sel = '#' + CSS.escape(id);
                else if (name) sel = tag + '[name="' + name + '"]';
                else if (type && tag === 'input') sel = 'input[type="' + type + '"]';
                else if (el.getAttribute('aria-label')) sel = tag + '[aria-label="' + el.getAttribute('aria-label') + '"]';

                items.push({
                    tag, type, text, href: href.slice(0, 120), selector: sel,
                    x: Math.round(rect.left + rect.width/2),
                    y: Math.round(rect.top + rect.height/2),
                });
            });
            return items;
        }""")
        lines = []
        for i, el in enumerate(elements[:limit]):
            parts = [f"[{i}]"]
            parts.append(f"<{el['tag']}>")
            if el.get("type"):
                parts.append(f"type={el['type']}")
            if el.get("text"):
                parts.append(f'"{el["text"]}"')
            if el.get("href"):
                parts.append(f"-> {el['href']}")
            parts.append(f"selector=\"{el['selector']}\"")
            parts.append(f"at ({el['x']},{el['y']})")
            lines.append(" ".join(parts))
        return "\n".join(lines) if lines else "(no interactive elements found)"
    except Exception as e:
        return f"(could not extract elements: {e})"


# ---------------------------------------------------------------------------
# Cursor controller — moves cursor + highlights elements
# ---------------------------------------------------------------------------

class CursorController:
    """Tracks and animates the agent cursor on the page."""

    def __init__(self, page, websocket: WebSocket):
        self.page = page
        self.ws = websocket
        self.x = 640  # center of 1280 viewport
        self.y = 360  # center of 720 viewport

    async def inject(self):
        """Inject cursor overlay element into page."""
        try:
            await self.page.evaluate(CURSOR_INJECT_JS)
        except Exception:
            pass

    async def move_to(self, x: int, y: int, send_event: bool = True):
        """Move cursor to position with animation."""
        self.x = x
        self.y = y
        try:
            await self.page.evaluate(CURSOR_MOVE_JS % (x, y))
        except Exception:
            pass
        if send_event:
            try:
                await self.ws.send_json({"type": "cursor_move", "x": x, "y": y})
            except Exception:
                pass

    async def move_to_selector(self, selector: str) -> Optional[Dict]:
        """Move cursor to center of element matched by selector. Returns center coords."""
        try:
            result = await self.page.evaluate(
                HIGHLIGHT_ELEMENT_JS % json.dumps(selector)
            )
            if result:
                await self.move_to(int(result["x"]), int(result["y"]))
                return result
        except Exception:
            pass
        return None

    async def move_to_text(self, text: str) -> Optional[Dict]:
        """Move cursor to element containing text. Returns center coords."""
        try:
            result = await self.page.evaluate(
                GET_TEXT_ELEMENT_CENTER_JS % json.dumps(text)
            )
            if result:
                await self.move_to(int(result["x"]), int(result["y"]))
                return result
        except Exception:
            pass
        return None

    async def hide(self):
        """Hide the cursor (move offscreen)."""
        try:
            await self.page.evaluate(CURSOR_HIDE_JS)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Browser tool executor
# ---------------------------------------------------------------------------

async def _exec_browser_tool(
    name: str,
    args: Dict[str, Any],
    page,
    cursor: CursorController,
) -> str:
    """Execute a browser tool. Cursor moves to targets before actions."""
    try:
        if name == "navigate":
            url = args.get("url", "").strip()
            if not url:
                return "ERROR: url is required"
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await _wait_stable(page)
            # Re-inject cursor after navigation
            await cursor.inject()
            await cursor.move_to(640, 360, send_event=False)
            title = await page.title()
            # Auto-analyze page
            analysis = await _analyze_page(page)
            return f"Navigated to {page.url} — {title}\n\n{analysis}"

        elif name == "click":
            selector = args.get("selector")
            text = args.get("text")
            index = args.get("index")

            if selector:
                # Move cursor to element, highlight it, then click
                center = await cursor.move_to_selector(selector)
                await asyncio.sleep(0.4)  # Let user see cursor move
                try:
                    await page.click(selector, timeout=5_000)
                except Exception:
                    # Fallback: JS click
                    try:
                        await page.evaluate(f"document.querySelector({json.dumps(selector)})?.click()")
                    except Exception as e:
                        return f"ERROR clicking selector '{selector}': {e}"
            elif text:
                center = await cursor.move_to_text(text)
                await asyncio.sleep(0.4)
                loc = page.get_by_text(text, exact=False).first
                try:
                    await loc.click(timeout=5_000)
                except Exception:
                    try:
                        await loc.dispatch_event("click")
                    except Exception as e:
                        return f"ERROR clicking text '{text}': {e}"
            elif index is not None:
                # Click by index from element list — get element by index
                center = await page.evaluate("""(idx) => {
                    const els = document.querySelectorAll('a, button, input, select, textarea, [role="button"], [role="link"], [role="tab"], [onclick]');
                    const visible = [];
                    els.forEach(el => {
                        const r = el.getBoundingClientRect();
                        if (r.width > 0 && r.height > 0) visible.push(el);
                    });
                    if (idx < visible.length) {
                        const r = visible[idx].getBoundingClientRect();
                        visible[idx].click();
                        return { x: r.left + r.width/2, y: r.top + r.height/2 };
                    }
                    return null;
                }""", index)
                if center:
                    await cursor.move_to(int(center["x"]), int(center["y"]))
                    await asyncio.sleep(0.3)
                else:
                    return f"ERROR: element index {index} not found"
            else:
                return "ERROR: provide selector, text, or index"

            await asyncio.sleep(0.5)
            await _wait_stable(page)
            await cursor.inject()  # Re-inject in case page changed
            title = await page.title()
            analysis = await _analyze_page(page)
            return f"Clicked. Now on: {page.url} — {title}\n\n{analysis}"

        elif name == "type_text":
            text_to_type = args.get("text", "")
            selector = args.get("selector")
            clear = args.get("clear", True)
            press_enter = args.get("press_enter", False)

            if selector:
                # Move cursor to input field
                await cursor.move_to_selector(selector)
                await asyncio.sleep(0.3)
                if clear:
                    await page.fill(selector, text_to_type, timeout=5_000)
                else:
                    await page.click(selector, timeout=5_000)
                    await page.keyboard.type(text_to_type, delay=30)
                if press_enter:
                    await page.press(selector, "Enter")
            else:
                # Type into focused element
                await page.keyboard.type(text_to_type, delay=30)
                if press_enter:
                    await page.keyboard.press("Enter")

            await asyncio.sleep(0.3)
            if press_enter:
                await _wait_stable(page)
                await cursor.inject()

            result = f"Typed '{text_to_type[:60]}'" + (" + Enter" if press_enter else "")
            if press_enter:
                analysis = await _analyze_page(page)
                result += f"\n\n{analysis}"
            return result

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = int(args.get("amount", 500))
            delta = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta)
            await asyncio.sleep(0.6)
            # Move cursor to indicate scroll direction
            cy = 200 if direction == "up" else 520
            await cursor.move_to(640, cy)
            analysis = await _analyze_page(page)
            return f"Scrolled {direction} {amount}px\n\n{analysis}"

        elif name == "go_back":
            await page.go_back(wait_until="domcontentloaded", timeout=10_000)
            await _wait_stable(page)
            await cursor.inject()
            title = await page.title()
            analysis = await _analyze_page(page)
            return f"Back to: {page.url} — {title}\n\n{analysis}"

        elif name == "wait":
            ms = int(args.get("milliseconds", 2000))
            await asyncio.sleep(ms / 1000)
            analysis = await _analyze_page(page)
            return f"Waited {ms}ms\n\n{analysis}"

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
    cursor: CursorController,
):
    """
    Agentic loop: LLM calls browser tools, we execute them, screenshot after
    every action, and loop until the agent calls done.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = settings.agent_model

    # First: take a screenshot of current page + analyze it
    await cursor.inject()
    current_analysis = await _analyze_page(page)
    await _send_screenshot(websocket, page, cursor)

    # Build user message with current page context
    context_message = user_message
    if page.url and page.url != "about:blank":
        context_message = (
            f"{user_message}\n\n"
            f"[Current browser state]\n{current_analysis}"
        )

    conversation.append({"role": "user", "content": context_message})

    max_steps = 20

    for step in range(max_steps):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": BROWSER_SYSTEM_PROMPT}] + conversation,
                tools=BROWSER_TOOLS,
                max_completion_tokens=2048,
                temperature=0.2,
            )
        except Exception as e:
            logger.error("[WS Browser] LLM call failed: %s", e)
            await websocket.send_json({"type": "error", "message": f"LLM error: {e}"})
            return

        choice = response.choices[0]
        msg = choice.message

        # If model returned text without tool calls — send as message
        if not msg.tool_calls:
            text = msg.content or ""
            conversation.append({"role": "assistant", "content": text})
            if text:
                await websocket.send_json({"type": "agent_message", "content": text})
            return

        # Process tool calls
        conversation.append(msg.model_dump())

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("[WS Browser] Step %d: %s(%s)", step + 1, fn_name, json.dumps(fn_args)[:200])

            # Notify client of tool use with args
            await websocket.send_json({
                "type": "tool_use",
                "tool": fn_name,
                "args": fn_args,
            })

            # Execute the browser tool (cursor moves + page analysis included)
            result = await _exec_browser_tool(fn_name, fn_args, page, cursor)

            # Add tool result to conversation
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:6000],
            })

            # Screenshot after EVERY action
            await _send_screenshot(websocket, page, cursor)
            await _send_state(websocket, page, tab_manager)

            # If done, send summary and return
            if fn_name == "done":
                await websocket.send_json({
                    "type": "agent_message",
                    "content": fn_args.get("summary", "Task completed."),
                })
                return

    # Hit max steps
    await websocket.send_json({
        "type": "agent_message",
        "content": "I've taken many steps. Let me know if you'd like me to continue.",
    })
    await _send_screenshot(websocket, page, cursor)


# ---------------------------------------------------------------------------
# Screenshot / state helpers
# ---------------------------------------------------------------------------

async def _send_screenshot(websocket: WebSocket, page, cursor: CursorController):
    try:
        png_bytes = await page.screenshot(type="png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        await websocket.send_json({
            "type": "screenshot",
            "image": b64,
            "cursor": {"x": cursor.x, "y": cursor.y},
        })
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
    cursor: Optional[CursorController] = None
    chat_task: Optional[asyncio.Task] = None
    conversation: List[Dict[str, Any]] = []

    # -- Main message loop --
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
                    tab_id = await tab_manager.open_tab(browser, url,
                                                        viewport={"width": 1280, "height": 720})
                    active_page = tab_manager.get_tab(tab_id)
                    cursor = CursorController(active_page, websocket)
                    await cursor.inject()
                    await _send_state(websocket, active_page, tab_manager)
                    await asyncio.sleep(0.5)
                    await _send_screenshot(websocket, active_page, cursor)

                elif action == "navigate":
                    if not browser_available:
                        await websocket.send_json({"type": "error", "message": "Browser not available"})
                        continue
                    url = data.get("url", "")
                    if not url:
                        continue
                    if not active_page:
                        browser = await _get_browser()
                        tab_id = await tab_manager.open_tab(browser, url,
                                                            viewport={"width": 1280, "height": 720})
                        active_page = tab_manager.get_tab(tab_id)
                        cursor = CursorController(active_page, websocket)
                    else:
                        if not url.startswith(("http://", "https://", "about:")):
                            url = "https://" + url
                        await active_page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    await cursor.inject()
                    await _send_state(websocket, active_page, tab_manager)
                    await asyncio.sleep(0.5)
                    await _send_screenshot(websocket, active_page, cursor)

                elif action == "screenshot":
                    if active_page and cursor:
                        await _send_screenshot(websocket, active_page, cursor)

                elif action == "click":
                    x = data.get("x", 0)
                    y = data.get("y", 0)
                    if active_page and cursor:
                        await cursor.move_to(x, y)
                        await asyncio.sleep(0.2)
                        await active_page.mouse.click(x, y)
                        await asyncio.sleep(0.5)
                        await cursor.inject()
                        await _send_state(websocket, active_page, tab_manager)
                        await _send_screenshot(websocket, active_page, cursor)

                elif action == "back":
                    if active_page and cursor:
                        await active_page.go_back(timeout=10_000)
                        await cursor.inject()
                        await _send_state(websocket, active_page, tab_manager)
                        await asyncio.sleep(0.5)
                        await _send_screenshot(websocket, active_page, cursor)

                elif action == "forward":
                    if active_page and cursor:
                        await active_page.go_forward(timeout=10_000)
                        await cursor.inject()
                        await _send_state(websocket, active_page, tab_manager)
                        await asyncio.sleep(0.5)
                        await _send_screenshot(websocket, active_page, cursor)

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
                        tab_id = await tab_manager.open_tab(browser,
                                                            viewport={"width": 1280, "height": 720})
                        active_page = tab_manager.get_tab(tab_id)
                        cursor = CursorController(active_page, websocket)

                    if chat_task and not chat_task.done():
                        await websocket.send_json({"type": "error", "message": "Agent is still working on the previous request"})
                    else:
                        await websocket.send_json({"type": "agent_thinking"})
                        chat_task = asyncio.create_task(
                            _run_browser_agent(
                                websocket, active_page, tab_manager, browser_mod,
                                message, conversation, cursor,
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
