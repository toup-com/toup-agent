"""
WebSocket Browser Endpoint — Agentic browser control with visual overlays.

The user sends natural language commands ("find me a laptop on amazon").
The agent autonomously browses: navigates, clicks, types, scrolls, reads pages.
Screenshots stream after every action so the user sees live progress.
Visual overlays (glowing cursor, click ripple, element highlight) show the agent's actions.

Ported from the toup-browser Electron app's overlay system.
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
# Agent overlay system — ported from toup-browser Electron app
# Glowing cursor, page dim, element highlight, click ripple, scroll indicator
# ---------------------------------------------------------------------------

OVERLAY_INJECT_JS = """
(() => {
  if (document.getElementById('__toup_overlay')) return;

  const style = document.createElement('style');
  style.id = '__toup_styles';
  style.textContent = `
    @keyframes __toup_pulse {
      0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; }
      50% { transform: translate(-50%, -50%) scale(1.4); opacity: 0.3; }
    }
    @keyframes __toup_ripple {
      0% { transform: translate(-50%, -50%) scale(0); opacity: 0.6; }
      100% { transform: translate(-50%, -50%) scale(3); opacity: 0; }
    }
    @keyframes __toup_glow_border {
      0%, 100% { box-shadow: 0 0 8px 2px rgba(99,102,241,0.5), 0 0 20px 4px rgba(99,102,241,0.2); }
      50% { box-shadow: 0 0 16px 4px rgba(99,102,241,0.8), 0 0 40px 8px rgba(99,102,241,0.3); }
    }
    @keyframes __toup_scroll_arrow {
      0% { transform: translateY(0); opacity: 1; }
      100% { transform: translateY(30px); opacity: 0; }
    }
    #__toup_overlay { transition: opacity 0.3s ease; }
    #__toup_cursor_dot {
      transition: left 0.55s cubic-bezier(0.22, 1, 0.36, 1), top 0.55s cubic-bezier(0.22, 1, 0.36, 1);
    }
    #__toup_cursor_arrow {
      transition: left 0.55s cubic-bezier(0.22, 1, 0.36, 1), top 0.55s cubic-bezier(0.22, 1, 0.36, 1);
    }
    #__toup_highlight_box {
      transition: left 0.3s ease, top 0.3s ease, width 0.3s ease, height 0.3s ease;
    }
  `;
  document.head.appendChild(style);

  // Page dim overlay
  const overlay = document.createElement('div');
  overlay.id = '__toup_overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.08);pointer-events:none;z-index:2147483640;opacity:0;';
  document.body.appendChild(overlay);

  // Cursor glow dot
  const dot = document.createElement('div');
  dot.id = '__toup_cursor_dot';
  dot.style.cssText = 'position:fixed;width:20px;height:20px;border-radius:50%;background:radial-gradient(circle,#818cf8 0%,#6366f1 50%,transparent 70%);pointer-events:none;z-index:2147483647;display:none;transform:translate(-50%,-50%);';
  document.body.appendChild(dot);

  // Cursor glow halo (pulsing)
  const glow = document.createElement('div');
  glow.id = '__toup_cursor_glow';
  glow.style.cssText = 'position:fixed;width:48px;height:48px;border-radius:50%;background:radial-gradient(circle,rgba(99,102,241,0.4) 0%,rgba(99,102,241,0.1) 50%,transparent 70%);pointer-events:none;z-index:2147483646;display:none;transform:translate(-50%,-50%);animation:__toup_pulse 1.5s ease-in-out infinite;';
  document.body.appendChild(glow);

  // Cursor pointer SVG arrow
  const arrow = document.createElement('div');
  arrow.id = '__toup_cursor_arrow';
  arrow.innerHTML = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><g filter="url(#__ts)"><path d="M7 4l16 9.5-7.5 1.5-2 7z" fill="#6366f1"/><path d="M7 4l16 9.5-7.5 1.5-2 7z" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></g><defs><filter id="__ts" x="4" y="2" width="24" height="26" filterUnits="userSpaceOnUse"><feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity="0.4"/></filter></defs></svg>';
  arrow.style.cssText = 'position:fixed;pointer-events:none;z-index:2147483647;display:none;transform:translate(-4px,-2px);transition:left 0.55s cubic-bezier(0.22,1,0.36,1),top 0.55s cubic-bezier(0.22,1,0.36,1);';
  document.body.appendChild(arrow);

  // Click ripple
  const ripple = document.createElement('div');
  ripple.id = '__toup_ripple';
  ripple.style.cssText = 'position:fixed;width:24px;height:24px;border-radius:50%;border:2px solid #6366f1;pointer-events:none;z-index:2147483646;display:none;transform:translate(-50%,-50%) scale(0);';
  document.body.appendChild(ripple);

  // Element highlight box (glowing border)
  const hlBox = document.createElement('div');
  hlBox.id = '__toup_highlight_box';
  hlBox.style.cssText = 'position:fixed;pointer-events:none;z-index:2147483644;border:2px solid rgba(99,102,241,0.8);border-radius:8px;background:rgba(99,102,241,0.06);display:none;animation:__toup_glow_border 1.5s ease-in-out infinite;';
  document.body.appendChild(hlBox);

  // Label pill
  const label = document.createElement('div');
  label.id = '__toup_label';
  label.style.cssText = 'position:fixed;pointer-events:none;z-index:2147483647;background:linear-gradient(135deg,#6366f1,#818cf8);color:#fff;font:600 11px/1 -apple-system,system-ui,sans-serif;padding:5px 10px;border-radius:6px;white-space:nowrap;display:none;box-shadow:0 2px 12px rgba(99,102,241,0.4);letter-spacing:0.3px;';
  document.body.appendChild(label);

  // Scroll indicator
  const scrollInd = document.createElement('div');
  scrollInd.id = '__toup_scroll_ind';
  scrollInd.style.cssText = 'position:fixed;left:50%;pointer-events:none;z-index:2147483645;display:none;';
  scrollInd.innerHTML = '<svg width="32" height="32" viewBox="0 0 32 32" fill="none"><path d="M8 12l8 8 8-8" stroke="#6366f1" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  document.body.appendChild(scrollInd);
})()
"""

OVERLAY_SET_ACTIVE_JS = """
(() => {
  const o = document.getElementById('__toup_overlay');
  if (o) o.style.opacity = '%s';
})()
"""

OVERLAY_MOVE_CURSOR_JS = """
(() => {
  const dot = document.getElementById('__toup_cursor_dot');
  const glow = document.getElementById('__toup_cursor_glow');
  const arrow = document.getElementById('__toup_cursor_arrow');
  const lbl = document.getElementById('__toup_label');
  if (!dot) return;
  [dot, glow, arrow].forEach(el => { el.style.display = 'block'; });
  dot.style.left = '%dpx'; dot.style.top = '%dpx';
  glow.style.left = '%dpx'; glow.style.top = '%dpx';
  arrow.style.left = '%dpx'; arrow.style.top = '%dpx';
  if (lbl && '%s') {
    lbl.style.display = 'block';
    lbl.textContent = '%s';
    lbl.style.left = (%d + 24) + 'px';
    lbl.style.top = (%d - 12) + 'px';
  }
})()
"""

OVERLAY_CLICK_RIPPLE_JS = """
(() => {
  const r = document.getElementById('__toup_ripple');
  if (r) {
    r.style.display = 'block';
    r.style.left = '%dpx'; r.style.top = '%dpx';
    r.style.animation = 'none';
    r.offsetHeight;
    r.style.animation = '__toup_ripple 0.5s ease-out forwards';
    setTimeout(() => { r.style.display = 'none'; }, 600);
  }
})()
"""

OVERLAY_HIGHLIGHT_JS = """
(() => {
  const box = document.getElementById('__toup_highlight_box');
  if (!box) return;
  box.style.display = 'block';
  box.style.left = '%dpx'; box.style.top = '%dpx';
  box.style.width = '%dpx'; box.style.height = '%dpx';
})()
"""

OVERLAY_HIDE_CURSOR_JS = """
(() => {
  ['__toup_cursor_dot','__toup_cursor_glow','__toup_cursor_arrow','__toup_label','__toup_highlight_box'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
})()
"""

OVERLAY_SCROLL_JS = """
(() => {
  const s = document.getElementById('__toup_scroll_ind');
  if (!s) return;
  s.style.display = 'block';
  s.style.top = '%s';
  s.style.transform = '%s';
  s.style.animation = 'none';
  s.offsetHeight;
  s.style.animation = '__toup_scroll_arrow 0.8s ease-out forwards';
  setTimeout(() => { s.style.display = 'none'; }, 900);
})()
"""

# Get element bounding box by selector
GET_ELEMENT_RECT_JS = """
((selector) => {
  try {
    const el = document.querySelector(selector);
    if (el) {
      const r = el.getBoundingClientRect();
      return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), cx: Math.round(r.x + r.width/2), cy: Math.round(r.y + r.height/2) };
    }
  } catch(e) {}
  return null;
})(%s)
"""

GET_TEXT_ELEMENT_RECT_JS = """
((text) => {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.textContent && node.textContent.trim().toLowerCase().includes(text.toLowerCase())) {
      const el = node.parentElement;
      if (el) {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
          return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), cx: Math.round(r.x + r.width/2), cy: Math.round(r.y + r.height/2) };
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
                    "url": {"type": "string", "description": "URL to navigate to"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element. The cursor will move to it visually before clicking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of element to click"},
                    "text": {"type": "string", "description": "Visible text of the element to click"},
                    "index": {"type": "integer", "description": "Index from the interactive elements list"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into an input field. The cursor moves to the field first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the input"},
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
                    "milliseconds": {"type": "integer", "description": "Time to wait (default: 2000)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call when the task is complete. Include a summary.",
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

BROWSER_SYSTEM_PROMPT = """You are an AI browser agent controlling a real web browser. You help users accomplish tasks on the web.

## Your workflow for EVERY action:
1. After navigating to any page, you automatically receive a page analysis (text + interactive elements)
2. Study the content and interactive elements carefully
3. Decide what to click, type, or scroll based on what you see
4. After each action, you get a new analysis — use it before the next step
5. Continue until done, then call `done` with a summary

## Rules:
- After every action you get page content + interactive elements — use this to plan next steps
- Use CSS selectors (preferred) or text content to target elements
- When searching, type the query and set press_enter=true
- Scroll down to find more content if needed
- If something doesn't work, try a different selector or approach
- For Google: https://www.google.com/search?q=YOUR+QUERY
- Be methodical: navigate -> see page -> interact -> see result -> continue
- Call `done` with a clear summary when finished"""


# ---------------------------------------------------------------------------
# Page analysis
# ---------------------------------------------------------------------------

async def _analyze_page(page) -> str:
    try:
        title = await page.title()
        url = page.url
        body_text = await page.evaluate("""() => {
            const clone = document.body.cloneNode(true);
            clone.querySelectorAll('script, style, noscript, svg, [id^="__toup_"]').forEach(el => el.remove());
            return (clone.innerText || '').trim();
        }""")
        if len(body_text) > 5000:
            body_text = body_text[:5000] + "\n...(truncated)"
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
    try:
        elements = await page.evaluate("""() => {
            const SELECTOR = "a[href], button, input, select, textarea, [role='button'], [role='link'], [role='tab'], [role='menuitem'], [role='checkbox'], [role='radio'], [onclick], [tabindex], [contenteditable='true']";
            const els = Array.from(document.querySelectorAll(SELECTOR));
            const visible = els.filter(el => {
                const r = el.getBoundingClientRect();
                const s = window.getComputedStyle(el);
                return r.width > 0 && r.height > 0 && s.display !== 'none' && s.visibility !== 'hidden' && !el.id?.startsWith('__toup_');
            });
            return visible.slice(0, 80).map((el, i) => {
                const r = el.getBoundingClientRect();
                const tag = el.tagName.toLowerCase();
                const type = el.type || '';
                const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().slice(0, 80);
                const href = el.href || '';
                const id = el.id || '';
                const name = el.name || '';
                let sel = tag;
                if (id) sel = '#' + CSS.escape(id);
                else if (name) sel = tag + '[name="' + name + '"]';
                else if (type && tag === 'input') sel = 'input[type="' + type + '"]';
                else if (el.getAttribute('aria-label')) sel = tag + '[aria-label="' + el.getAttribute('aria-label').replace(/"/g, '\\\\"') + '"]';
                return { tag, type, text, href: href.slice(0, 120), selector: sel, x: Math.round(r.left + r.width/2), y: Math.round(r.top + r.height/2) };
            });
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
            parts.append(f'selector="{el["selector"]}"')
            parts.append(f"at ({el['x']},{el['y']})")
            lines.append(" ".join(parts))
        return "\n".join(lines) if lines else "(no interactive elements found)"
    except Exception as e:
        return f"(could not extract elements: {e})"


# ---------------------------------------------------------------------------
# Visual cursor controller — mirrors toup-browser's overlay system
# ---------------------------------------------------------------------------

class AgentOverlay:
    """Controls the visual overlay injected into the page."""

    def __init__(self, page, websocket: WebSocket):
        self.page = page
        self.ws = websocket
        self.x = 640
        self.y = 360

    async def inject(self):
        try:
            await self.page.evaluate(OVERLAY_INJECT_JS)
        except Exception:
            pass

    async def set_active(self, active: bool):
        try:
            await self.page.evaluate(OVERLAY_SET_ACTIVE_JS % ("1" if active else "0"))
        except Exception:
            pass

    async def move_cursor(self, x: int, y: int, label: str = ""):
        self.x = x
        self.y = y
        safe_label = label.replace("'", "").replace('"', '')[:30]
        try:
            await self.page.evaluate(
                OVERLAY_MOVE_CURSOR_JS % (x, y, x, y, x, y, safe_label, safe_label, x, y)
            )
        except Exception:
            pass
        try:
            await self.ws.send_json({"type": "cursor_move", "x": x, "y": y})
        except Exception:
            pass

    async def click_ripple(self, x: int, y: int):
        try:
            await self.page.evaluate(OVERLAY_CLICK_RIPPLE_JS % (x, y))
        except Exception:
            pass

    async def highlight_element(self, x: int, y: int, w: int, h: int):
        try:
            await self.page.evaluate(OVERLAY_HIGHLIGHT_JS % (x, y, w, h))
        except Exception:
            pass

    async def scroll_visual(self, direction: str):
        is_down = direction == "down"
        top = "60%" if is_down else "40%"
        transform = "translateX(-50%)" if is_down else "translateX(-50%) rotate(180deg)"
        try:
            await self.page.evaluate(OVERLAY_SCROLL_JS % (top, transform))
        except Exception:
            pass

    async def hide_cursor(self):
        try:
            await self.page.evaluate(OVERLAY_HIDE_CURSOR_JS)
        except Exception:
            pass

    async def move_to_selector(self, selector: str) -> Optional[Dict]:
        try:
            rect = await self.page.evaluate(GET_ELEMENT_RECT_JS % json.dumps(selector))
            if rect:
                await self.highlight_element(rect["x"], rect["y"], rect["w"], rect["h"])
                await self.move_cursor(rect["cx"], rect["cy"])
                return rect
        except Exception:
            pass
        return None

    async def move_to_text(self, text: str) -> Optional[Dict]:
        try:
            rect = await self.page.evaluate(GET_TEXT_ELEMENT_RECT_JS % json.dumps(text))
            if rect:
                await self.highlight_element(rect["x"], rect["y"], rect["w"], rect["h"])
                await self.move_cursor(rect["cx"], rect["cy"])
                return rect
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Browser tool executor
# ---------------------------------------------------------------------------

async def _exec_browser_tool(
    name: str,
    args: Dict[str, Any],
    page,
    overlay: AgentOverlay,
) -> str:
    try:
        if name == "navigate":
            url = args.get("url", "").strip()
            if not url:
                return "ERROR: url is required"
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await _wait_stable(page)
            await overlay.inject()
            await overlay.set_active(True)
            await overlay.move_cursor(640, 360, "")
            title = await page.title()
            analysis = await _analyze_page(page)
            return f"Navigated to {page.url} — {title}\n\n{analysis}"

        elif name == "click":
            selector = args.get("selector")
            text = args.get("text")
            index = args.get("index")

            if selector:
                rect = await overlay.move_to_selector(selector)
                await asyncio.sleep(0.5)
                if rect:
                    await overlay.click_ripple(rect["cx"], rect["cy"])
                try:
                    await page.click(selector, timeout=5_000)
                except Exception:
                    try:
                        await page.evaluate(f"document.querySelector({json.dumps(selector)})?.click()")
                    except Exception as e:
                        return f"ERROR clicking '{selector}': {e}"
            elif text:
                rect = await overlay.move_to_text(text)
                await asyncio.sleep(0.5)
                if rect:
                    await overlay.click_ripple(rect["cx"], rect["cy"])
                loc = page.get_by_text(text, exact=False).first
                try:
                    await loc.click(timeout=5_000)
                except Exception:
                    try:
                        await loc.dispatch_event("click")
                    except Exception as e:
                        return f"ERROR clicking text '{text}': {e}"
            elif index is not None:
                center = await page.evaluate("""(idx) => {
                    const SELECTOR = "a[href], button, input, select, textarea, [role='button'], [role='link'], [role='tab'], [onclick]";
                    const els = Array.from(document.querySelectorAll(SELECTOR));
                    const visible = els.filter(el => {
                        const r = el.getBoundingClientRect();
                        return r.width > 0 && r.height > 0 && !el.id?.startsWith('__toup_');
                    });
                    if (idx < visible.length) {
                        const r = visible[idx].getBoundingClientRect();
                        visible[idx].click();
                        return { x: Math.round(r.left + r.width/2), y: Math.round(r.top + r.height/2) };
                    }
                    return null;
                }""", index)
                if center:
                    await overlay.move_cursor(center["x"], center["y"])
                    await asyncio.sleep(0.3)
                    await overlay.click_ripple(center["x"], center["y"])
                else:
                    return f"ERROR: element index {index} not found"
            else:
                return "ERROR: provide selector, text, or index"

            await asyncio.sleep(0.5)
            await _wait_stable(page)
            await overlay.inject()
            title = await page.title()
            analysis = await _analyze_page(page)
            return f"Clicked. Now on: {page.url} — {title}\n\n{analysis}"

        elif name == "type_text":
            text_to_type = args.get("text", "")
            selector = args.get("selector")
            clear = args.get("clear", True)
            press_enter = args.get("press_enter", False)

            if selector:
                rect = await overlay.move_to_selector(selector)
                await asyncio.sleep(0.3)
                if clear:
                    await page.fill(selector, text_to_type, timeout=5_000)
                else:
                    await page.click(selector, timeout=5_000)
                    await page.keyboard.type(text_to_type, delay=30)
                if press_enter:
                    await page.press(selector, "Enter")
            else:
                await page.keyboard.type(text_to_type, delay=30)
                if press_enter:
                    await page.keyboard.press("Enter")

            await asyncio.sleep(0.3)
            if press_enter:
                await _wait_stable(page)
                await overlay.inject()

            result = f"Typed '{text_to_type[:60]}'" + (" + Enter" if press_enter else "")
            if press_enter:
                analysis = await _analyze_page(page)
                result += f"\n\n{analysis}"
            return result

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = int(args.get("amount", 500))
            delta = amount if direction == "down" else -amount
            await overlay.scroll_visual(direction)
            await page.mouse.wheel(0, delta)
            await asyncio.sleep(0.6)
            cy = 200 if direction == "up" else 520
            await overlay.move_cursor(640, cy, "")
            analysis = await _analyze_page(page)
            return f"Scrolled {direction} {amount}px\n\n{analysis}"

        elif name == "go_back":
            await page.go_back(wait_until="domcontentloaded", timeout=10_000)
            await _wait_stable(page)
            await overlay.inject()
            title = await page.title()
            analysis = await _analyze_page(page)
            return f"Back to: {page.url} — {title}\n\n{analysis}"

        elif name == "wait":
            ms = int(args.get("milliseconds", 2000))
            await asyncio.sleep(ms / 1000)
            analysis = await _analyze_page(page)
            return f"Waited {ms}ms\n\n{analysis}"

        elif name == "done":
            await overlay.hide_cursor()
            await overlay.set_active(False)
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
    overlay: AgentOverlay,
):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = settings.agent_model

    # Activate overlay
    await overlay.inject()
    await overlay.set_active(True)

    # Initial page analysis + screenshot
    current_analysis = await _analyze_page(page)
    await _send_screenshot(websocket, page, overlay)

    context_message = user_message
    if page.url and page.url != "about:blank":
        context_message = f"{user_message}\n\n[Current browser state]\n{current_analysis}"

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

        if not msg.tool_calls:
            text = msg.content or ""
            conversation.append({"role": "assistant", "content": text})
            if text:
                await websocket.send_json({"type": "agent_message", "content": text})
            await overlay.hide_cursor()
            await overlay.set_active(False)
            return

        conversation.append(msg.model_dump())

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("[WS Browser] Step %d: %s(%s)", step + 1, fn_name, json.dumps(fn_args)[:200])

            await websocket.send_json({
                "type": "tool_use",
                "tool": fn_name,
                "args": fn_args,
            })

            result = await _exec_browser_tool(fn_name, fn_args, page, overlay)

            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:6000],
            })

            # Screenshot after every action
            await _send_screenshot(websocket, page, overlay)
            await _send_state(websocket, page, tab_manager)

            if fn_name == "done":
                await websocket.send_json({
                    "type": "agent_message",
                    "content": fn_args.get("summary", "Task completed."),
                })
                return

    await websocket.send_json({
        "type": "agent_message",
        "content": "I've taken many steps. Let me know if you'd like me to continue.",
    })
    await _send_screenshot(websocket, page, overlay)
    await overlay.hide_cursor()
    await overlay.set_active(False)


# ---------------------------------------------------------------------------
# Screenshot / state helpers
# ---------------------------------------------------------------------------

async def _send_screenshot(websocket: WebSocket, page, overlay: AgentOverlay):
    try:
        png_bytes = await page.screenshot(type="png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        await websocket.send_json({
            "type": "screenshot",
            "image": b64,
            "cursor": {"x": overlay.x, "y": overlay.y},
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

    if agent_key and settings.agent_api_key and agent_key == settings.agent_api_key:
        user_id = settings.user_id

    if not user_id and token:
        user_id = await _authenticate_ws(token)

    if not user_id:
        await websocket.send_json({"type": "error", "message": "Authentication required"})
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Import browser module — now uses stealth context
    browser_available = True
    browser_mod = None
    try:
        from app.agent import browser as browser_mod
        from app.agent.browser import _get_browser, get_tab_manager, get_stealth_context
    except ImportError as e:
        logger.warning(f"[WS Browser] Browser module not available: {e}")
        browser_available = False

    tab_manager = get_tab_manager() if browser_available else None
    active_page = None
    overlay: Optional[AgentOverlay] = None
    chat_task: Optional[asyncio.Task] = None
    conversation: List[Dict[str, Any]] = []

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
                        await websocket.send_json({"type": "error", "message": "Browser not available"})
                        continue
                    url = data.get("url", "about:blank")
                    ctx = await get_stealth_context()
                    tab_id = await tab_manager.open_tab(ctx, url,
                                                        viewport={"width": 1280, "height": 720})
                    active_page = tab_manager.get_tab(tab_id)
                    overlay = AgentOverlay(active_page, websocket)
                    await overlay.inject()
                    await _send_state(websocket, active_page, tab_manager)
                    await asyncio.sleep(0.5)
                    await _send_screenshot(websocket, active_page, overlay)

                elif action == "navigate":
                    if not browser_available:
                        await websocket.send_json({"type": "error", "message": "Browser not available"})
                        continue
                    url = data.get("url", "")
                    if not url:
                        continue
                    if not active_page:
                        ctx = await get_stealth_context()
                        tab_id = await tab_manager.open_tab(ctx, url,
                                                            viewport={"width": 1280, "height": 720})
                        active_page = tab_manager.get_tab(tab_id)
                        overlay = AgentOverlay(active_page, websocket)
                    else:
                        if not url.startswith(("http://", "https://", "about:")):
                            url = "https://" + url
                        await active_page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    await overlay.inject()
                    await _send_state(websocket, active_page, tab_manager)
                    await asyncio.sleep(0.5)
                    await _send_screenshot(websocket, active_page, overlay)

                elif action == "screenshot":
                    if active_page and overlay:
                        await _send_screenshot(websocket, active_page, overlay)

                elif action == "click":
                    x = data.get("x", 0)
                    y = data.get("y", 0)
                    if active_page and overlay:
                        await overlay.move_cursor(x, y)
                        await asyncio.sleep(0.2)
                        await overlay.click_ripple(x, y)
                        await active_page.mouse.click(x, y)
                        await asyncio.sleep(0.5)
                        await overlay.inject()
                        await _send_state(websocket, active_page, tab_manager)
                        await _send_screenshot(websocket, active_page, overlay)

                elif action == "back":
                    if active_page and overlay:
                        await active_page.go_back(timeout=10_000)
                        await overlay.inject()
                        await _send_state(websocket, active_page, tab_manager)
                        await asyncio.sleep(0.5)
                        await _send_screenshot(websocket, active_page, overlay)

                elif action == "forward":
                    if active_page and overlay:
                        await active_page.go_forward(timeout=10_000)
                        await overlay.inject()
                        await _send_state(websocket, active_page, tab_manager)
                        await asyncio.sleep(0.5)
                        await _send_screenshot(websocket, active_page, overlay)

                elif action == "chat":
                    message = data.get("message", "").strip()
                    if not message:
                        continue
                    if not browser_available:
                        await websocket.send_json({"type": "error", "message": "Browser not available"})
                        continue

                    if not active_page:
                        ctx = await get_stealth_context()
                        tab_id = await tab_manager.open_tab(ctx,
                                                            viewport={"width": 1280, "height": 720})
                        active_page = tab_manager.get_tab(tab_id)
                        overlay = AgentOverlay(active_page, websocket)

                    if chat_task and not chat_task.done():
                        await websocket.send_json({"type": "error", "message": "Agent is still working"})
                    else:
                        await websocket.send_json({"type": "agent_thinking"})
                        chat_task = asyncio.create_task(
                            _run_browser_agent(
                                websocket, active_page, tab_manager, browser_mod,
                                message, conversation, overlay,
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
        if overlay:
            try:
                await overlay.hide_cursor()
                await overlay.set_active(False)
            except Exception:
                pass
