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
import random
import sys
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket Browser"])

def _is_mac() -> bool:
    return sys.platform == "darwin"

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
    {"type": "function", "function": {"name": "navigate", "description": "Navigate to a URL. ONLY use to go to a NEW website.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "click", "description": "Click an element by selector, text, or index.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "description": "CSS selector"}, "text": {"type": "string", "description": "Visible text"}, "index": {"type": "integer", "description": "Index from elements list"}}}}},
    {"type": "function", "function": {"name": "type_text", "description": "Type text into an input field.", "parameters": {"type": "object", "properties": {"selector": {"type": "string"}, "text": {"type": "string"}, "clear": {"type": "boolean"}, "press_enter": {"type": "boolean"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "select_date", "description": "Select a date on a calendar/date picker. Use this instead of click for date selection.", "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "Date in YYYY-MM-DD format, e.g. 2026-03-19"}}, "required": ["date"]}}},
    {"type": "function", "function": {"name": "scroll", "description": "Scroll the page.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["up", "down"]}, "amount": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "select_dropdown", "description": "Select an option from a dropdown menu. Atomically clicks the trigger to open the dropdown, then clicks the option. Use this for ANY dropdown (trip type, passenger count, cabin class, sort order, etc.).", "parameters": {"type": "object", "properties": {"trigger_text": {"type": "string", "description": "Text of the dropdown trigger button (e.g. 'Round trip', '1 passenger', 'Economy')"}, "option_text": {"type": "string", "description": "Text of the option to select (e.g. 'One way', '2 passengers', 'Business')"}}, "required": ["trigger_text", "option_text"]}}},
    {"type": "function", "function": {"name": "go_back", "description": "Go back to previous page.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "wait", "description": "Wait for page to load.", "parameters": {"type": "object", "properties": {"milliseconds": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "done", "description": "Task complete — include summary.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]

BROWSER_SYSTEM_PROMPT = """You are an AI browser agent controlling a REAL web browser. You can SEE the page via screenshots attached to each message.

## CRITICAL — YOU MUST USE THE ACTUAL WEBSITE:
- NEVER just read Google search results or AI Overviews and call it done.
- You have a REAL browser — navigate to the actual website and USE it.
- Google search snippets and AI Overviews are NOT a substitute for using the website.

## HANDLE POPUPS AND OVERLAYS FIRST:
Before interacting with a page, dismiss ANY visible popups or promotional overlays:
- Cookie/privacy banners → click "Accept" or "Accept All" (NEVER "Cookie preferences" or "Manage")
- Promotional overlays (e.g. "Try AI powered Flight Deals") → click "Got it", "Dismiss", "No thanks", or X
- Modal dialogs → click "Close" or the X button
- If ANY overlay is covering the form, dismiss it FIRST before typing

## RULES:
1. LOOK at the screenshot CAREFULLY before EVERY action. Overlays covering the form? Dismiss FIRST.
2. NEVER click hamburger menus, "Main menu", "Sign In", or account/profile links — only interact with FORM elements.
3. NEVER navigate away from a page you're working on. Stay on the current site. If you accidentally navigate away, go back immediately.
4. DROPDOWNS: ALWAYS use `select_dropdown` tool for dropdowns. NEVER use two separate `click` calls — the dropdown will close between clicks.
5. ONE action at a time. Screenshot → analyze → one action → next screenshot → repeat.
6. Use the interactive elements list AND screenshot together.
7. ALWAYS use a tool — never just talk.
8. READ FIELD LABELS CAREFULLY before clicking.
9. If you can't see the full form, SCROLL UP first.

## Tools:
- `click` — use `selector` (CSS preferred), `text` (visible text), or `index` (from elements list)
- `type_text` — type into input fields. Use `selector` to target. `press_enter=true` to submit.
- `select_dropdown` — MUST use for ANY dropdown selection. Atomically opens dropdown + clicks option in one step. Example: `select_dropdown(trigger_text="Round trip", option_text="One way")`
- `scroll` — scroll down/up to see more content
- `navigate` — ONLY to go to a NEW website
- `select_date` — select a date in calendar pickers (YYYY-MM-DD format)
- `wait` — wait for page to load/update
- `done` — call with summary when task is complete

## FILLING FORMS:
For EACH text field:
1. CLICK the correct field (match the label/placeholder, not position)
2. TYPE the value
3. WAIT for autocomplete dropdown in next screenshot
4. CLICK the correct suggestion from the dropdown
5. Only THEN move to the NEXT field

## DATE PICKERS:
- Click the date field to open the calendar
- Use `select_date` with YYYY-MM-DD format (e.g. select_date(date="2026-03-19"))
- After select_date, click "Done" to confirm

## COMPLETING THE TASK:
- Only call `done` when you have ACTUAL results (prices, listings, etc.)
- Scroll through results to find the best options before summarizing
- Include specific details: prices, names, times, etc.
- If a form submission fails, try again — don't give up immediately
- STAY on the current website until done
- NEVER ask clarifying questions — just do it"""


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
            const SELECTOR = "a[href], button, input, select, textarea, [role='button'], [role='link'], [role='tab'], [role='menuitem'], [role='checkbox'], [role='radio'], [role='gridcell'], [role='option'], [role='listbox'], [role='switch'], [data-iso], [onclick], [tabindex], [contenteditable='true']";
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
                const placeholder = el.placeholder || '';
                const ariaLabel = el.getAttribute('aria-label') || '';
                const text = (el.innerText || el.value || '').trim().slice(0, 80);
                const href = el.href || '';
                const id = el.id || '';
                const name = el.name || '';
                let sel = tag;
                if (id) sel = '#' + CSS.escape(id);
                else if (name) sel = tag + '[name="' + name + '"]';
                else if (ariaLabel) sel = tag + '[aria-label="' + ariaLabel.replace(/"/g, '\\\\"') + '"]';
                else if (type && tag === 'input') sel = 'input[type="' + type + '"]';
                return { tag, type, text, placeholder, aria_label: ariaLabel, href: href.slice(0, 120), selector: sel, x: Math.round(r.left + r.width/2), y: Math.round(r.top + r.height/2) };
            });
        }""")
        lines = []
        for i, el in enumerate(elements[:limit]):
            parts = [f"[{i}] <{el['tag']}>"]
            if el.get("type"):
                parts.append(f"type={el['type']}")
            if el.get("placeholder"):
                parts.append(f'placeholder="{el["placeholder"]}"')
            if el.get("aria_label"):
                parts.append(f'aria-label="{el["aria_label"]}"')
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
# Auto-dismiss common popups (cookie banners, modals, overlays)
# ---------------------------------------------------------------------------

async def _auto_dismiss_popups(page):
    """Click away cookie banners, consent popups, and modal overlays instantly."""
    try:
        dismissed = await page.evaluate("""() => {
            const dismissed = [];

            // Words that indicate WRONG buttons (settings/preferences panels)
            const rejectWords = ['preferences', 'settings', 'customize', 'manage', 'options', 'more info', 'learn more', 'details', 'back'];

            function isRejectButton(text) {
                const t = text.toLowerCase();
                return rejectWords.some(w => t.includes(w));
            }

            // Promotional overlays — ONLY match elements that are clearly
            // full-screen/modal overlays, NOT trip type dropdowns or menus.
            // Strategy: find buttons with dismiss-like text ONLY inside elements
            // that look like promotional overlays (large, covering viewport).
            const dismissTexts = new Set(['got it', 'no thanks', 'dismiss', 'not now', 'skip', 'maybe later']);
            const allBtns = document.querySelectorAll('button, [role="button"]');
            for (const btn of allBtns) {
                const text = (btn.textContent || '').trim().toLowerCase();
                if (!dismissTexts.has(text)) continue;
                if (btn.offsetParent === null) continue;
                // Check if this button is inside a promo/modal overlay
                // (ancestor that covers >50% of viewport OR has promo-like classes)
                let isPromo = false;
                let el = btn.parentElement;
                while (el && el !== document.body) {
                    const cls = (el.className || '').toString().toLowerCase();
                    const role = (el.getAttribute('role') || '').toLowerCase();
                    // Check for overlay-like classes
                    if (cls.match(/promo|overlay|banner|modal|consent|cookie|popup|interstitial/)) {
                        isPromo = true; break;
                    }
                    // Check for full-screen overlay (covers >40% of viewport)
                    if (role === 'dialog' || role === 'alertdialog') {
                        const r = el.getBoundingClientRect();
                        if (r.width > window.innerWidth * 0.4 && r.height > window.innerHeight * 0.3) {
                            isPromo = true; break;
                        }
                    }
                    el = el.parentElement;
                }
                if (isPromo) {
                    btn.click();
                    dismissed.push('promo-dismiss: ' + text);
                    return dismissed;
                }
            }

            // High-priority: known accept button IDs/selectors
            const acceptSelectors = [
                '#onetrust-accept-btn-handler',
                '.onetrust-accept-btn-handler',
                '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
                '#CybotCookiebotDialogBodyButtonAccept',
                'button[id*="accept" i]',
                'a[id*="accept" i]',
                '[data-testid*="accept" i]',
            ];

            for (const sel of acceptSelectors) {
                const btn = document.querySelector(sel);
                if (btn && btn.offsetParent !== null) {
                    const text = (btn.textContent || '').trim();
                    if (!isRejectButton(text)) {
                        btn.click();
                        dismissed.push('accept-selector: ' + sel);
                        return dismissed;
                    }
                }
            }

            // Text-based: find visible buttons with accept-like text
            // SAFE texts — specific enough to not match legitimate UI
            const safeAcceptTexts = [
                'accept all', 'accept cookies', 'accept all cookies',
                'allow all', 'allow all cookies', 'i agree',
                'accept', 'agree', 'allow',
                'save & continue', 'save and continue',
            ];
            // UNSAFE texts — only match these inside cookie/consent containers
            const unsafeAcceptTexts = ['got it', 'ok', 'dismiss', 'continue'];

            const allButtons2 = document.querySelectorAll('button, a[role="button"], [class*="btn"], [role="button"]');

            // First pass: safe texts (match anywhere)
            for (const acceptText of safeAcceptTexts) {
                for (const btn of allButtons2) {
                    const text = (btn.textContent || '').trim().toLowerCase();
                    if (text.length > 40) continue;
                    if (isRejectButton(text)) continue;
                    if (text === acceptText || text.startsWith(acceptText + ' ')) {
                        if (btn.offsetParent !== null) {
                            btn.click();
                            dismissed.push('text: ' + text);
                            return dismissed;
                        }
                    }
                }
            }

            // Second pass: unsafe texts — ONLY inside cookie/consent/overlay containers
            for (const acceptText of unsafeAcceptTexts) {
                for (const btn of allButtons2) {
                    const text = (btn.textContent || '').trim().toLowerCase();
                    if (text.length > 40) continue;
                    if (isRejectButton(text)) continue;
                    if (text === acceptText || text.startsWith(acceptText + ' ')) {
                        if (btn.offsetParent === null) continue;
                        // Check ancestor for consent/cookie context
                        let inConsent = false;
                        let anc = btn.parentElement;
                        while (anc && anc !== document.body) {
                            const cls = (anc.className || '').toString().toLowerCase();
                            const id = (anc.id || '').toLowerCase();
                            if ((cls + id).match(/cookie|consent|gdpr|privacy|banner|overlay|promo|popup|interstitial/)) {
                                inConsent = true; break;
                            }
                            anc = anc.parentElement;
                        }
                        if (inConsent) {
                            btn.click();
                            dismissed.push('text-consent: ' + text);
                            return dismissed;
                        }
                    }
                }
            }

            // Modal close buttons (only for non-cookie modals)
            const closeButtons = document.querySelectorAll(
                'button[aria-label="Close" i], button[aria-label="dismiss" i], ' +
                '[role="dialog"] button[aria-label*="close" i]'
            );
            for (const btn of closeButtons) {
                const text = (btn.textContent || '').trim().toLowerCase();
                if (btn.offsetParent !== null && (text === '' || text === 'close' || text === '×' || text === 'x')) {
                    btn.click();
                    dismissed.push('modal-close: ' + (text || 'aria-close'));
                    return dismissed;
                }
            }

            return dismissed;
        }""")
        if dismissed:
            logger.info("[BROWSER] Auto-dismissed popups: %s", dismissed)
            await asyncio.sleep(0.5)  # Wait for animations
            return True
        return False
    except Exception as e:
        logger.debug("[BROWSER] Auto-dismiss failed (ok): %s", e)
        return False


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
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception:
                pass  # Timeout OK — page may have partially loaded
            await _wait_stable(page)
            # Human-like pause after page load (Atlas behavior)
            await asyncio.sleep(random.uniform(0.3, 0.8))

            # ── Auto-solve captchas ──
            from app.agent.browser import detect_captcha, solve_captcha
            captcha = await detect_captcha(page)
            if captcha:
                solved = await solve_captcha(page, captcha)
                if solved:
                    await _wait_stable(page)
                    await asyncio.sleep(0.5)

            # ── Auto-dismiss popups — they load async, so retry with delays ──
            logger.warning("[NAVIGATE] Auto-dismiss pass 1 (immediate)")
            d1 = await _auto_dismiss_popups(page)
            await asyncio.sleep(2.0)
            logger.warning("[NAVIGATE] Auto-dismiss pass 2 (after 2s)")
            d2 = await _auto_dismiss_popups(page)
            await asyncio.sleep(1.5)
            logger.warning("[NAVIGATE] Auto-dismiss pass 3 (after 3.5s)")
            d3 = await _auto_dismiss_popups(page)
            await asyncio.sleep(2.0)
            logger.warning("[NAVIGATE] Auto-dismiss pass 4 (after 5.5s)")
            d4 = await _auto_dismiss_popups(page)
            if any([d1, d2, d3, d4]):
                logger.warning("[NAVIGATE] Popups dismissed across passes")

            await overlay.inject()
            await overlay.set_active(True)
            await overlay.move_cursor(640, 360, "")
            title = await page.title()
            analysis = await _analyze_page(page)
            # Warn agent if captcha is still present after solving attempt
            captcha_after = await detect_captcha(page)
            if captcha_after:
                analysis = f"WARNING: Page shows a {captcha_after['type']} captcha that could not be auto-solved.\n\n{analysis}"
            return f"Navigated to {page.url} — {title}\n\n{analysis}"

        elif name == "click":
            # Atlas-style: resolve element → Bezier mouse move → real mouse click
            from app.agent.browser import (
                human_click, human_move_mouse,
                get_element_center, get_text_element_center, get_element_center_by_index,
            )

            selector = args.get("selector")
            text = args.get("text")
            index = args.get("index")
            cursor_pos = (overlay.x, overlay.y)

            target = None
            if selector:
                rect = await overlay.move_to_selector(selector)
                target = await get_element_center(page, selector)
            elif text:
                rect = await overlay.move_to_text(text)
                target = await get_text_element_center(page, text)
            elif index is not None:
                target = await get_element_center_by_index(page, index)
                if target:
                    await overlay.move_cursor(int(target[0]), int(target[1]))
            else:
                return "ERROR: provide selector, text, or index"

            if not target:
                # Fallback: try DOM click if coordinates unavailable
                if selector:
                    try:
                        await page.evaluate(f"document.querySelector({json.dumps(selector)})?.click()")
                    except Exception as e:
                        return f"ERROR clicking '{selector}': {e}"
                elif text:
                    try:
                        loc = page.get_by_text(text, exact=True).first
                        try:
                            await loc.click(timeout=3_000)
                        except Exception:
                            # Retry with force=True (Google Material ripple overlays intercept clicks)
                            try:
                                await loc.click(timeout=3_000, force=True)
                            except Exception:
                                loc = page.get_by_text(text, exact=False).first
                                await loc.click(timeout=3_000, force=True)
                    except Exception as e:
                        return f"ERROR clicking text '{text}': {e}"
                else:
                    return f"ERROR: element index {index} not found"
            else:
                # Human-like Bezier mouse move → real mouse click at coordinates
                await human_click(page, target[0], target[1], from_pos=cursor_pos)
                overlay.x, overlay.y = int(target[0]), int(target[1])
                await overlay.click_ripple(int(target[0]), int(target[1]))

            await asyncio.sleep(random.uniform(0.3, 0.6))
            await _wait_stable(page)

            # ── Auto-dismiss overlays that appear after click (Flight Deals, etc.) ──
            dismissed = await _auto_dismiss_popups(page)
            if dismissed:
                logger.warning("[CLICK] Auto-dismissed overlay after click")
                await asyncio.sleep(0.5)
                await _wait_stable(page)

            await overlay.inject()
            title = await page.title()
            analysis = await _analyze_page(page)
            return f"Clicked. Now on: {page.url} — {title}\n\n{analysis}"

        elif name == "type_text":
            # Atlas-style: move mouse to field → click → type char-by-char
            from app.agent.browser import (
                human_click, human_type, get_element_center,
            )

            text_to_type = args.get("text", "")
            selector = args.get("selector")
            clear = args.get("clear", True)
            press_enter = args.get("press_enter", False)
            cursor_pos = (overlay.x, overlay.y)

            if selector:
                rect = await overlay.move_to_selector(selector)
                target = await get_element_center(page, selector)
                if target:
                    # Move mouse to field and click it (human-like)
                    await human_click(page, target[0], target[1], from_pos=cursor_pos)
                    overlay.x, overlay.y = int(target[0]), int(target[1])
                    await asyncio.sleep(random.uniform(0.1, 0.25))
                else:
                    # Fallback: programmatic focus
                    try:
                        await page.click(selector, timeout=3_000)
                    except Exception:
                        pass

                if clear:
                    # Select all then type over (more human than page.fill)
                    await page.keyboard.press("Control+a" if not _is_mac() else "Meta+a")
                    await asyncio.sleep(random.uniform(0.05, 0.12))

                # Type character by character with human timing
                await human_type(page, text_to_type)

                if press_enter:
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    await page.keyboard.press("Enter")
            else:
                await human_type(page, text_to_type)
                if press_enter:
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    await page.keyboard.press("Enter")

            # Wait for autocomplete/dropdown to appear after typing
            await asyncio.sleep(random.uniform(0.8, 1.2))
            if press_enter:
                await _wait_stable(page)
            await overlay.inject()

            result = f"Typed '{text_to_type[:60]}'" + (" + Enter" if press_enter else "")
            # ALWAYS analyze page after typing — LLM needs to see dropdown suggestions
            analysis = await _analyze_page(page)
            result += f"\n\nLOOK at the screenshot — if you see a dropdown/autocomplete list, click the correct suggestion BEFORE moving to the next field.\n\n{analysis}"
            return result

        elif name == "select_date":
            # Smart date selection — finds calendar cells by multiple strategies
            from app.agent.browser import human_click
            date_str = args.get("date", "")
            if not date_str:
                return "ERROR: date is required (YYYY-MM-DD format)"

            # Parse date for multiple matching strategies
            try:
                from datetime import datetime
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                day = dt.day
                # Multiple date format strings to search for
                date_searches = [
                    f"{dt.strftime('%Y-%m-%d')}",          # 2026-03-19 (data-iso)
                    f"{dt.strftime('%A, %B')} {day}, {dt.year}",  # Wednesday, March 19, 2026
                    f"{dt.strftime('%B')} {day}",           # March 19
                ]
            except ValueError:
                return f"ERROR: invalid date format '{date_str}'. Use YYYY-MM-DD."

            # Strategy 1: data-iso attribute (Google Flights, many calendar libs)
            cell = await page.evaluate("""(iso) => {
                const el = document.querySelector(`[data-iso="${iso}"]`);
                if (el) {
                    const r = el.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0) return { x: r.left + r.width/2, y: r.top + r.height/2 };
                }
                return null;
            }""", date_str)

            # Strategy 2: aria-label containing the date
            if not cell:
                for search in date_searches:
                    cell = await page.evaluate("""(search) => {
                        const els = document.querySelectorAll('[aria-label]');
                        for (const el of els) {
                            const label = el.getAttribute('aria-label') || '';
                            if (label.includes(search)) {
                                const r = el.getBoundingClientRect();
                                if (r.width > 0 && r.height > 0 && r.width < 200)
                                    return { x: r.left + r.width/2, y: r.top + r.height/2 };
                            }
                        }
                        return null;
                    }""", search)
                    if cell:
                        break

            # Strategy 3: gridcell role with matching day number
            if not cell:
                cell = await page.evaluate("""(day) => {
                    const cells = document.querySelectorAll('[role="gridcell"], [role="button"]');
                    for (const el of cells) {
                        const text = (el.textContent || '').trim();
                        if (text === String(day) || text.startsWith(day + '\\n') || text.startsWith(day + ' ')) {
                            const r = el.getBoundingClientRect();
                            if (r.width > 0 && r.height > 0 && r.width < 100)
                                return { x: r.left + r.width/2, y: r.top + r.height/2 };
                        }
                    }
                    return null;
                }""", str(day))

            if cell:
                await overlay.move_cursor(int(cell["x"]), int(cell["y"]), f"Selecting {date_str}")
                await human_click(page, cell["x"], cell["y"], from_pos=(overlay.x, overlay.y))
                overlay.x, overlay.y = int(cell["x"]), int(cell["y"])
                await overlay.click_ripple(int(cell["x"]), int(cell["y"]))
                await asyncio.sleep(0.5)
                analysis = await _analyze_page(page)
                return f"Selected date {date_str} on calendar.\n\n{analysis}"
            else:
                # Fallback: try clicking the day number as text
                try:
                    loc = page.get_by_text(str(day), exact=True).first
                    await loc.click(timeout=3_000)
                    await asyncio.sleep(0.5)
                    analysis = await _analyze_page(page)
                    return f"Clicked date {day} (text fallback).\n\n{analysis}"
                except Exception:
                    return f"ERROR: Could not find date {date_str} on the calendar. Make sure the calendar is open and showing the correct month."

        elif name == "scroll":
            # Atlas-style: incremental wheel events instead of one big jump
            from app.agent.browser import human_scroll

            direction = args.get("direction", "down")
            amount = int(args.get("amount", 500))
            await overlay.scroll_visual(direction)
            await human_scroll(page, direction, amount)
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

        elif name == "select_dropdown":
            # Atomic dropdown selection: click trigger → wait → click option
            # Solves the timing problem where dropdowns close during LLM round-trip
            from app.agent.browser import human_click, get_text_element_center

            trigger = args.get("trigger_text", "")
            option = args.get("option_text", "")
            if not trigger or not option:
                return "ERROR: trigger_text and option_text are required"

            cursor_pos = (overlay.x, overlay.y)

            # Step 1: Click the trigger to open the dropdown
            logger.warning("[SELECT_DROPDOWN] Clicking trigger: '%s'", trigger)
            trigger_target = await get_text_element_center(page, trigger)
            if trigger_target:
                logger.warning("[SELECT_DROPDOWN] Found trigger at (%d, %d)", trigger_target[0], trigger_target[1])
                await human_click(page, trigger_target[0], trigger_target[1], from_pos=cursor_pos)
                overlay.x, overlay.y = int(trigger_target[0]), int(trigger_target[1])
            else:
                logger.warning("[SELECT_DROPDOWN] Text match failed for trigger '%s', trying fallbacks", trigger)
                # Fallback 1: Playwright text locator
                _trigger_clicked = False
                try:
                    loc = page.get_by_text(trigger, exact=True).first
                    await loc.click(timeout=3_000)
                    _trigger_clicked = True
                    logger.warning("[SELECT_DROPDOWN] Trigger clicked via exact text locator")
                except Exception:
                    try:
                        loc = page.get_by_text(trigger, exact=False).first
                        await loc.click(timeout=3_000)
                        _trigger_clicked = True
                        logger.warning("[SELECT_DROPDOWN] Trigger clicked via fuzzy text locator")
                    except Exception:
                        pass

                # Fallback 2: aria-label selector (for Google Flights trip type button)
                if not _trigger_clicked:
                    _aria_selectors = [
                        f'button[aria-label*="{trigger}" i]',
                        f'[role="button"][aria-label*="{trigger}" i]',
                        f'button[aria-label*="trip" i]',  # Google Flights specific
                    ]
                    for _sel in _aria_selectors:
                        try:
                            await page.click(_sel, timeout=2_000)
                            _trigger_clicked = True
                            logger.warning("[SELECT_DROPDOWN] Trigger clicked via selector: %s", _sel)
                            break
                        except Exception:
                            continue

                if not _trigger_clicked:
                    return f"ERROR: Could not click dropdown trigger '{trigger}'"

            # Step 2: Wait for dropdown to appear (crucial timing)
            await asyncio.sleep(random.uniform(0.4, 0.7))

            # Step 3: Click the option from the now-open dropdown
            logger.warning("[SELECT_DROPDOWN] Clicking option: '%s'", option)
            option_target = await get_text_element_center(page, option)
            _option_clicked = False
            if option_target:
                logger.warning("[SELECT_DROPDOWN] Found option at (%d, %d)", option_target[0], option_target[1])
                await human_click(page, option_target[0], option_target[1],
                                  from_pos=(overlay.x, overlay.y))
                overlay.x, overlay.y = int(option_target[0]), int(option_target[1])
                await overlay.click_ripple(int(option_target[0]), int(option_target[1]))
                _option_clicked = True
            else:
                logger.warning("[SELECT_DROPDOWN] Text match failed for option '%s', trying fallbacks", option)
                # Fallback 1: Playwright locator
                try:
                    loc = page.get_by_text(option, exact=True).first
                    await loc.click(timeout=2_000)
                    _option_clicked = True
                    logger.warning("[SELECT_DROPDOWN] Option clicked via exact text locator")
                except Exception:
                    try:
                        loc = page.get_by_text(option, exact=False).first
                        await loc.click(timeout=2_000)
                        _option_clicked = True
                        logger.warning("[SELECT_DROPDOWN] Option clicked via fuzzy text locator")
                    except Exception:
                        pass

                # Fallback 2: JavaScript DOM search (case-insensitive, broad)
                if not _option_clicked:
                    try:
                        clicked = await page.evaluate("""(optionText) => {
                            const optLower = optionText.toLowerCase().trim();

                            // Strategy 1: Standard dropdown roles
                            const roleEls = document.querySelectorAll(
                                '[role="option"], [role="menuitem"], [role="listbox"] *, li, [class*="menu"] *, [class*="dropdown"] *'
                            );
                            for (const el of roleEls) {
                                const text = (el.textContent || '').trim().toLowerCase();
                                if (text === optLower && el.offsetParent !== null) {
                                    el.click();
                                    return 'exact-role';
                                }
                            }

                            // Strategy 2: Any visible element with matching text (covers Material Design)
                            const allEls = document.querySelectorAll('*');
                            let bestMatch = null;
                            let bestLen = Infinity;
                            for (const el of allEls) {
                                if (el.offsetParent === null) continue;
                                const text = (el.textContent || '').trim().toLowerCase();
                                // Exact match on inner text (not children's text)
                                const ownText = Array.from(el.childNodes)
                                    .filter(n => n.nodeType === 3)
                                    .map(n => n.textContent.trim().toLowerCase())
                                    .join(' ');
                                if (ownText === optLower) {
                                    el.click();
                                    return 'exact-own-text';
                                }
                                // Full textContent match — pick shortest (most specific)
                                if (text === optLower && text.length < bestLen) {
                                    bestMatch = el;
                                    bestLen = text.length;
                                }
                            }
                            if (bestMatch) {
                                bestMatch.click();
                                return 'exact-textcontent';
                            }

                            // Strategy 3: Partial match (contains)
                            for (const el of roleEls) {
                                const text = (el.textContent || '').trim().toLowerCase();
                                if (text.includes(optLower) && el.offsetParent !== null) {
                                    el.click();
                                    return 'partial';
                                }
                            }

                            return false;
                        }""", option)
                        if clicked:
                            _option_clicked = True
                            logger.warning("[SELECT_DROPDOWN] Option clicked via JS DOM search (strategy: %s)", clicked)
                        else:
                            logger.warning("[SELECT_DROPDOWN] JS DOM search found no match for '%s'", option)
                    except Exception as e:
                        logger.warning("[SELECT_DROPDOWN] JS DOM search failed: %s", e)

                if not _option_clicked:
                    return f"ERROR: Dropdown opened but could not find option '{option}'"

            await asyncio.sleep(random.uniform(0.3, 0.5))
            await _wait_stable(page)
            await overlay.inject()
            title = await page.title()
            analysis = await _analyze_page(page)
            return f"Selected '{option}' from dropdown (trigger: '{trigger}'). Now on: {page.url} — {title}\n\n{analysis}"

        elif name == "done":
            _done_summary = args.get("summary", "Task completed.")
            _done_lower = _done_summary.lower()
            _fail_indicators = ["could not", "couldn't", "unable to", "failed to", "sorry",
                                "not able", "cannot", "can't", "wasn't able", "didn't work",
                                "encountered an error", "blocked", "captcha", "unfortunately",
                                "i apologize", "not possible", "didn't find", "no results"]
            if any(fw in _done_lower for fw in _fail_indicators):
                logger.warning("[DONE] ❌ Failure detected in done() summary: %s", _done_summary[:200])
                return (f"ERROR: You tried to finish with a failure message. Do NOT call done() until you have "
                        f"actual results. Your message was: '{_done_summary[:150]}'. "
                        f"Try a different approach instead.")
            await overlay.hide_cursor()
            await overlay.set_active(False)
            return _done_summary

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
    try:
        await _run_browser_agent_inner(
            websocket, page, tab_manager, browser_mod,
            user_message, conversation, overlay,
        )
    except Exception as e:
        logger.exception("[WS Browser] Agent loop crashed")
        try:
            await websocket.send_json({"type": "error", "message": f"Agent error: {e}"})
        except Exception:
            pass


async def _run_browser_agent_inner(
    websocket: WebSocket,
    page,
    tab_manager,
    browser_mod,
    user_message: str,
    conversation: List[Dict[str, Any]],
    overlay: AgentOverlay,
):
    from openai import AsyncOpenAI
    import base64 as _b64mod

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = "gpt-5.4"

    logger.warning("=" * 80)
    logger.warning("[BROWSER AGENT] NEW SESSION")
    logger.warning("[BROWSER AGENT] User message: %s", user_message)
    logger.warning("[BROWSER AGENT] Model: %s", model)
    logger.warning("=" * 80)

    # Activate overlay
    await overlay.inject()
    await overlay.set_active(True)

    # Initial page analysis + screenshot
    current_analysis = await _analyze_page(page)
    await _send_screenshot(websocket, page, overlay)

    # ── Skill system: classify message and load domain-specific knowledge ──
    from app.agent.browser_skills import SkillRegistry
    registry = SkillRegistry()
    system_prompt, classification = registry.build_augmented_prompt(BROWSER_SYSTEM_PROMPT, user_message)
    logger.warning("[BROWSER AGENT] Skill: category=%s confidence=%.2f params=%s",
                   classification.category, classification.confidence, classification.params)
    logger.warning("[BROWSER AGENT] System prompt length: %d chars", len(system_prompt))

    # Tell frontend which skill is active
    try:
        await websocket.send_json({
            "type": "skill_active",
            "category": classification.category,
            "confidence": classification.confidence,
            "params": classification.params,
        })
    except Exception:
        pass

    context_message = user_message

    # ── Send plan preview to user ──
    if classification.confidence >= 0.5 and classification.params:
        plan_parts = [f"Category: {classification.category.replace('_', ' ').title()}"]
        for k, v in classification.params.items():
            plan_parts.append(f"{k.replace('_', ' ').title()}: {v}")
        plan_text = " | ".join(plan_parts)
        try:
            await websocket.send_json({"type": "narration", "text": f"Plan: {plan_text}"})
        except Exception:
            pass

    if page.url and page.url != "about:blank":
        context_message += f"\n\n[Current browser state]\n{current_analysis}"

    # Start CDP screencast — event-driven live streaming (replaces polling)
    screencast = CDPScreencast(websocket, page, overlay)
    await screencast.start()

    try:
        # Build user message with vision screenshot
        try:
            _init_png = await page.screenshot(type="png")
            _init_b64 = _b64mod.b64encode(_init_png).decode()
            conversation.append({"role": "user", "content": [
                {"type": "text", "text": context_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_init_b64}", "detail": "high"}},
            ]})
        except Exception:
            conversation.append({"role": "user", "content": context_message})

        max_steps = 30
        consecutive_same_actions = 0
        last_action_key = ""
        _force_tool_next = False  # Set after retry injection to prevent LLM from responding with text

        for step in range(max_steps):
            logger.warning("-" * 60)
            logger.warning("[STEP %d/%d] Calling LLM...", step + 1, max_steps)

            # Log current page state
            try:
                _cur_url = page.url
                _cur_title = await page.title()
                logger.warning("[STEP %d] Page: %s — %s", step + 1, _cur_url, _cur_title)
            except Exception:
                pass

            # Narration: tell the user what's happening during LLM thinking
            await websocket.send_json({
                "type": "narration",
                "text": "Looking at the page..." if step == 0 else "Deciding next action...",
            })

            tc_mode = "required" if (step == 0 or _force_tool_next) else "auto"
            _force_tool_next = False  # Reset after using
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}] + conversation,
                    tools=BROWSER_TOOLS,
                    tool_choice=tc_mode,
                    max_completion_tokens=2048,
                    temperature=0.2,
                )
            except Exception as e:
                logger.error("[STEP %d] LLM CALL FAILED: %s", step + 1, e)
                await websocket.send_json({"type": "error", "message": f"LLM error: {e}"})
                return

            msg = response.choices[0].message

            # ── Log the LLM's thinking / reasoning ──
            if msg.content:
                logger.warning("[STEP %d] LLM THINKING: %s", step + 1, msg.content[:500])

            if not msg.tool_calls:
                text = msg.content or ""
                logger.warning("[STEP %d] LLM returned TEXT (no tool call): %s", step + 1, text[:300])

                # ── Check if the text response is a failure/give-up — if so, force retry ──
                _text_lower = text.lower()
                _failure_words_text = ["could not", "couldn't", "unable to", "failed to", "sorry",
                                       "not able", "cannot", "can't", "wasn't able", "didn't work",
                                       "encountered an error", "blocked", "captcha", "i apologize",
                                       "unfortunately", "not possible"]
                _is_text_failure = any(fw in _text_lower for fw in _failure_words_text)
                if _is_text_failure and step < max_steps - 3:
                    logger.warning("[STEP %d] ❌ TEXT FAILURE DETECTED — forcing retry with tool_choice=required", step + 1)
                    conversation.append({"role": "assistant", "content": text})
                    conversation.append({
                        "role": "user",
                        "content": "[SYSTEM] Do NOT give up or explain failure. You MUST continue working on the task. "
                                   "Use a tool NOW to take your next action. Try a different approach if the previous one failed.",
                    })
                    _force_tool_next = True
                    continue

                conversation.append({"role": "assistant", "content": text})
                if text:
                    await websocket.send_json({"type": "agent_message", "content": text})
                return

            # Log all tool calls the LLM wants to make
            for _tc in msg.tool_calls:
                logger.warning("[STEP %d] LLM TOOL CALL: %s(%s)", step + 1, _tc.function.name, _tc.function.arguments[:300])

            conversation.append(msg.model_dump())

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.warning("[STEP %d] EXECUTING: %s(%s)", step + 1, fn_name, json.dumps(fn_args)[:300])

                # ── Stuck detection: if agent repeats the same action 3+ times, inject a hint ──
                action_key = f"{fn_name}:{json.dumps(fn_args, sort_keys=True)[:100]}"
                if action_key == last_action_key:
                    consecutive_same_actions += 1
                else:
                    consecutive_same_actions = 0
                    last_action_key = action_key

                if consecutive_same_actions >= 2:
                    logger.warning("[STEP %d] ⚠️ STUCK DETECTED: same action repeated %d times: %s", step + 1, consecutive_same_actions + 1, action_key[:100])
                    conversation.append({
                        "role": "user",
                        "content": "[SYSTEM] You are repeating the same action. This action is not working. Try a DIFFERENT approach: use a different selector, scroll to find the element, or try an alternative strategy.",
                    })
                    consecutive_same_actions = 0

                # Narration: describe the action about to happen
                _narration_map = {
                    "navigate": lambda a: f"Navigating to {a.get('url', '...')}",
                    "click": lambda a: f"Clicking {a.get('text') or a.get('selector', 'element')}",
                    "type_text": lambda a: f'Typing "{(a.get("text", ""))[:40]}"',
                    "select_dropdown": lambda a: f"Selecting '{a.get('option_text', '...')}' from dropdown",
                    "select_date": lambda a: f"Selecting date {a.get('date', '...')}",
                    "scroll": lambda a: f"Scrolling {a.get('direction', 'down')}",
                    "go_back": lambda _: "Going back",
                    "wait": lambda _: "Waiting for page to load...",
                    "done": lambda a: a.get("summary", "Done!"),
                }
                narr_fn = _narration_map.get(fn_name)
                if narr_fn:
                    await websocket.send_json({"type": "narration", "text": narr_fn(fn_args)})

                await websocket.send_json({
                    "type": "tool_use",
                    "tool": fn_name,
                    "args": fn_args,
                })

                # Send cursor position BEFORE executing (so frontend animates cursor to target)
                if fn_name in ("click", "type_text") and overlay:
                    await websocket.send_json({
                        "type": "cursor_move",
                        "x": overlay.x, "y": overlay.y,
                    })

                result = await _exec_browser_tool(fn_name, fn_args, page, overlay)
                logger.warning("[STEP %d] RESULT: %s", step + 1, result[:500])

                # Log page URL after action (detect if navigation happened unexpectedly)
                try:
                    _post_url = page.url
                    if fn_name != "navigate" and _post_url != _cur_url:
                        logger.warning("[STEP %d] ⚠️ URL CHANGED unexpectedly: %s → %s", step + 1, _cur_url, _post_url)

                    # ── URL-based recovery: auto-navigate back if agent landed on wrong page ──
                    _wrong_page_patterns = [
                        ("/travel/flights/deals", "https://www.google.com/travel/flights"),
                        ("/travel/flights/explore", "https://www.google.com/travel/flights"),
                        ("accounts.google.com/", _cur_url),  # Sign-in redirect → go back to where we were
                    ]
                    for _bad_pattern, _recovery_url in _wrong_page_patterns:
                        if _bad_pattern in (_post_url or "") and fn_name != "navigate":
                            logger.warning("[STEP %d] 🔄 AUTO-RECOVERY: wrong page '%s' detected, navigating back to %s",
                                           step + 1, _bad_pattern, _recovery_url)
                            try:
                                await page.goto(_recovery_url, wait_until="domcontentloaded", timeout=15_000)
                            except Exception:
                                pass
                            await _wait_stable(page)
                            await asyncio.sleep(1.0)
                            await _auto_dismiss_popups(page)
                            await overlay.inject()
                            result = f"[AUTO-RECOVERY] You accidentally navigated to {_post_url}. Navigated back to {_recovery_url}. Continue with your task from here.\n\n" + await _analyze_page(page)
                            break
                except Exception:
                    pass

                # ── Validator: append verification prompt to tool result ──
                if fn_name == "type_text":
                    _typed_text = fn_args.get("text", "").upper()
                    _is_airport_code = len(_typed_text) == 3 and _typed_text.isalpha()
                    if _is_airport_code:
                        result += "\n\n[VERIFY] ⚠️ You typed an airport code. Look at the screenshot — there should be an autocomplete dropdown with airport suggestions. You MUST click the correct suggestion before proceeding. The airport is NOT set until you click the autocomplete option."
                    else:
                        result += "\n\n[VERIFY] Look at the screenshot. Did the text appear in the correct field? Is there an autocomplete dropdown you need to select from?"
                elif fn_name == "click":
                    result += "\n\n[VERIFY] Look at the screenshot. Did the click have the expected effect? Did a dropdown open? Did the page change? Is there an overlay blocking the form?"
                elif fn_name == "select_dropdown":
                    result += "\n\n[VERIFY] Look at the screenshot. Did the dropdown option get selected? Does the trigger button now show the new value?"
                elif fn_name == "select_date":
                    result += "\n\n[VERIFY] Look at the screenshot. Is the date highlighted on the calendar? If so, click 'Done' to confirm the date selection."
                elif fn_name == "navigate":
                    result += "\n\n[VERIFY] Did the page load correctly? Are there any overlays or popups to dismiss before proceeding?"

                # Send cursor position AFTER executing (final position)
                if fn_name in ("click", "type_text", "scroll") and overlay:
                    await websocket.send_json({
                        "type": "cursor_move",
                        "x": overlay.x, "y": overlay.y,
                    })

                # Vision: attach screenshot for visual actions so LLM sees the page
                # type_text included — LLM must see autocomplete dropdowns after typing
                _visual_actions = {"navigate", "click", "scroll", "go_back", "type_text", "select_date", "select_dropdown"}
                if fn_name in _visual_actions:
                    try:
                        _snap = await page.screenshot(type="png")
                        _snap_b64 = _b64mod.b64encode(_snap).decode()
                        tool_content = [
                            {"type": "text", "text": result[:12000]},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_snap_b64}", "detail": "high"}},
                        ]
                    except Exception:
                        tool_content = result[:12000]
                else:
                    tool_content = result[:12000]

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                })

                await _send_state(websocket, page, tab_manager)

                if fn_name == "done":
                    _summary = fn_args.get("summary", "Task completed.")

                    # If the done() tool itself rejected (returned ERROR), force retry
                    if result.startswith("ERROR:") and step < max_steps - 3:
                        logger.warning("[STEP %d] ❌ done() rejected — failure detected, forcing retry", step + 1)
                        _force_tool_next = True
                        continue  # tool result already in conversation, LLM will see the error

                    # Double-check: manual failure word detection as backup
                    _summary_lower = _summary.lower()
                    _failure_words = ["could not", "couldn't", "unable to", "failed to", "sorry",
                                      "not able", "cannot", "can't", "wasn't able", "didn't work",
                                      "encountered an error", "blocked", "captcha", "unfortunately",
                                      "i apologize", "not possible"]
                    _is_failure = any(fw in _summary_lower for fw in _failure_words)
                    if _is_failure and step < max_steps - 3:
                        logger.warning("[STEP %d] ❌ AGENT TRIED TO GIVE UP: %s — forcing retry", step + 1, _summary[:200])
                        # Remove the done tool result from conversation and inject retry message
                        conversation.pop()  # Remove tool result
                        conversation.pop()  # Remove assistant message with tool call
                        conversation.append({
                            "role": "user",
                            "content": f"[SYSTEM] Do NOT give up. Your previous attempt failed: '{_summary[:200]}'. Try a DIFFERENT approach:\n"
                                       f"1. Navigate directly to the target URL if you haven't already\n"
                                       f"2. Try different selectors or scroll to find elements\n"
                                       f"3. If current site is blocked, try an alternative website\n"
                                       f"Continue working — do NOT call done() until you have actual results.",
                        })
                        _force_tool_next = True
                        continue
                    logger.warning("[STEP %d] ✅ TASK COMPLETE: %s", step + 1, _summary[:300])
                    await websocket.send_json({
                        "type": "agent_message",
                        "content": _summary,
                    })
                    return

        logger.warning("[BROWSER AGENT] ❌ MAX STEPS REACHED (%d) — agent did not finish", max_steps)
        await websocket.send_json({
            "type": "agent_message",
            "content": "I've taken many steps. Let me know if you'd like me to continue.",
        })
    finally:
        await screencast.stop()
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


# ---------------------------------------------------------------------------
# CDP Screencast — event-driven frame streaming (replaces screenshot polling)
# Frames are sent only when the page actually renders (much more efficient
# and lower latency than polling page.screenshot()).
# ---------------------------------------------------------------------------

class CDPScreencast:
    """
    Uses Chrome DevTools Protocol Page.startScreencast for live streaming.
    Frames fire only when the page content changes — zero CPU waste on static pages.
    Falls back to screenshot polling if CDP is unavailable.
    """

    def __init__(self, websocket: WebSocket, page, overlay: AgentOverlay):
        self.ws = websocket
        self.page = page
        self.overlay = overlay
        self._cdp = None
        self._running = False
        self._fallback = False
        self._stop_event = asyncio.Event()
        self._fallback_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start screencast — try CDP first, fall back to polling."""
        self._running = True
        self._stop_event.clear()
        try:
            # Playwright / Patchright CDP session
            self._cdp = await self.page.context.new_cdp_session(self.page)
            self._cdp.on("Page.screencastFrame", self._on_frame)
            await self._cdp.send("Page.startScreencast", {
                "format": "jpeg",
                "quality": 60,
                "maxWidth": 1280,
                "maxHeight": 720,
                "everyNthFrame": 1,
            })
            logger.info("[SCREENCAST] CDP screencast started")
        except Exception as e:
            logger.warning("[SCREENCAST] CDP unavailable (%s), falling back to polling", e)
            self._fallback = True
            self._fallback_task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        """Stop screencast."""
        self._running = False
        self._stop_event.set()
        if self._cdp and not self._fallback:
            try:
                await self._cdp.send("Page.stopScreencast")
                await self._cdp.detach()
            except Exception:
                pass
            self._cdp = None
        if self._fallback_task:
            self._fallback_task.cancel()
            try:
                await self._fallback_task
            except (asyncio.CancelledError, Exception):
                pass

    def _on_frame(self, params: dict):
        """CDP screencastFrame event handler — fires when page renders a new frame."""
        if not self._running:
            return
        session_id = params.get("sessionId", 0)
        b64_data = params.get("data", "")
        # ACK immediately to get next frame (backpressure)
        asyncio.create_task(self._ack_and_send(session_id, b64_data))

    async def _ack_and_send(self, session_id: int, b64_data: str):
        try:
            # ACK the frame so CDP sends the next one
            if self._cdp:
                await self._cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
        except Exception:
            pass
        try:
            await self.ws.send_json({
                "type": "screenshot",
                "image": b64_data,
                "format": "jpeg",
            })
        except Exception:
            pass

    async def _poll_loop(self):
        """Fallback: screenshot polling at 6 FPS if CDP is unavailable."""
        interval = 1.0 / 6
        while not self._stop_event.is_set():
            try:
                jpg_bytes = await self.page.screenshot(type="jpeg", quality=55)
                b64 = base64.b64encode(jpg_bytes).decode("ascii")
                await self.ws.send_json({
                    "type": "screenshot",
                    "image": b64,
                    "format": "jpeg",
                })
            except Exception:
                pass
            await asyncio.sleep(interval)


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
