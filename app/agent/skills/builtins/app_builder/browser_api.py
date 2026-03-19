"""
Browser API — Lightweight internal interface for sub-agents to use
the platform's Patchright browser for web research.

Reuses the existing browser infrastructure (stealth context, tab manager)
without interfering with the user-facing browser session.

Usage:
    from .browser_api import search, read_page

    results = await search("fitness tracker apps 2025")
    content = await read_page("https://example.com/article")
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Max text to return from a single page
MAX_PAGE_TEXT = 15_000
# Timeout for page loads
NAV_TIMEOUT_MS = 20_000


async def _get_page():
    """Create a new page from the shared stealth context.
    Returns (page, cleanup_fn) — caller MUST call cleanup_fn when done."""
    from app.agent.browser import get_stealth_context
    ctx = await get_stealth_context()
    page = await ctx.new_page()

    async def cleanup():
        try:
            await page.close()
        except Exception:
            pass

    return page, cleanup


async def read_page(url: str, timeout_ms: int = NAV_TIMEOUT_MS) -> str:
    """Navigate to a URL and extract its text content.

    Uses the platform's stealth browser — bypasses bot detection.
    Opens a temporary page, extracts text, closes it.

    Returns extracted text (max 15K chars) or error string.
    """
    page = None
    cleanup = None
    try:
        page, cleanup = await _get_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # Wait a moment for JS rendering
        await asyncio.sleep(1)

        # Extract main content — prefer article/main, fall back to body
        text = await page.evaluate("""() => {
            // Remove noise elements
            const remove = ['script', 'style', 'nav', 'footer', 'header',
                           'aside', 'noscript', 'iframe', '.ad', '.ads',
                           '[class*="cookie"]', '[class*="banner"]',
                           '[class*="popup"]', '[class*="modal"]'];
            remove.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => el.remove());
            });

            // Try article/main first, then body
            const main = document.querySelector('article')
                      || document.querySelector('main')
                      || document.querySelector('[role="main"]')
                      || document.body;
            if (!main) return '';

            return main.innerText || '';
        }""")

        if not text or not text.strip():
            # Fallback: just get body text
            text = await page.inner_text("body")

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text.strip())

        if len(text) > MAX_PAGE_TEXT:
            text = text[:MAX_PAGE_TEXT] + "\n\n[truncated]"

        return text or "(empty page)"

    except Exception as exc:
        logger.warning("[BROWSER_API] read_page failed for %s: %s", url, exc)
        return f"(failed to read page: {exc})"
    finally:
        if cleanup:
            await cleanup()


async def search(query: str, count: int = 5) -> List[Dict[str, str]]:
    """Search Google and extract results.

    Uses the platform's stealth browser to navigate Google search.
    Returns list of {title, url, snippet} dicts.

    Falls back to DuckDuckGo HTML if Google fails.
    """
    results = await _google_search(query, count)
    if results:
        return results

    # Fallback to DuckDuckGo
    results = await _ddg_search(query, count)
    return results


async def _google_search(query: str, count: int) -> List[Dict[str, str]]:
    """Search Google via the stealth browser."""
    page = None
    cleanup = None
    try:
        page, cleanup = await _get_page()
        search_url = f"https://www.google.com/search?q={_url_encode(query)}&num={count + 5}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        await asyncio.sleep(1.5)  # Let results render

        # Extract search results via JS
        results = await page.evaluate("""(count) => {
            const results = [];
            // Google organic results
            const items = document.querySelectorAll('div.g, div[data-sokoban-container]');
            for (const item of items) {
                if (results.length >= count) break;
                const linkEl = item.querySelector('a[href^="http"]');
                const titleEl = item.querySelector('h3');
                const snippetEl = item.querySelector('[data-sncf], .VwiC3b, [style*="-webkit-line-clamp"]');
                if (linkEl && titleEl) {
                    results.push({
                        title: titleEl.innerText || '',
                        url: linkEl.href || '',
                        snippet: snippetEl ? snippetEl.innerText : ''
                    });
                }
            }
            return results;
        }""", count)

        if results:
            logger.info("[BROWSER_API] Google search: %d results for '%s'", len(results), query[:50])
        return results or []

    except Exception as exc:
        logger.warning("[BROWSER_API] Google search failed: %s", exc)
        return []
    finally:
        if cleanup:
            await cleanup()


async def _ddg_search(query: str, count: int) -> List[Dict[str, str]]:
    """Fallback: search DuckDuckGo via the stealth browser."""
    page = None
    cleanup = None
    try:
        page, cleanup = await _get_page()
        search_url = f"https://html.duckduckgo.com/html/?q={_url_encode(query)}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        await asyncio.sleep(1)

        results = await page.evaluate("""(count) => {
            const results = [];
            const items = document.querySelectorAll('.result');
            for (const item of items) {
                if (results.length >= count) break;
                const linkEl = item.querySelector('.result__title a');
                const snippetEl = item.querySelector('.result__snippet');
                if (linkEl) {
                    results.push({
                        title: linkEl.innerText || '',
                        url: linkEl.href || '',
                        snippet: snippetEl ? snippetEl.innerText : ''
                    });
                }
            }
            return results;
        }""", count)

        if results:
            logger.info("[BROWSER_API] DuckDuckGo search: %d results for '%s'", len(results), query[:50])
        return results or []

    except Exception as exc:
        logger.warning("[BROWSER_API] DuckDuckGo search failed: %s", exc)
        return []
    finally:
        if cleanup:
            await cleanup()


def _url_encode(text: str) -> str:
    """Simple URL encoding for search queries."""
    from urllib.parse import quote_plus
    return quote_plus(text)


async def search_formatted(query: str, count: int = 5) -> str:
    """Search and return results as a formatted string (for LLM consumption)."""
    results = await search(query, count)
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', 'Untitled')}")
        lines.append(f"   {r.get('url', '')}")
        lines.append(f"   {r.get('snippet', '')}")
        lines.append("")
    return "\n".join(lines)
