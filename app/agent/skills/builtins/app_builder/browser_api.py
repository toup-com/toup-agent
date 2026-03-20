"""
Browser API — Lightweight headless browser for sub-agent research.

Uses Patchright (or Playwright fallback) in pure headless mode.
Manages its own browser instance separate from the user-facing browser
so it never conflicts with the user's /browser session.

Usage:
    from .browser_api import search, read_page

    results = await search("fitness tracker apps 2025")
    content = await read_page("https://example.com/article")
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Max text to return from a single page
MAX_PAGE_TEXT = 15_000
# Timeout for page loads
NAV_TIMEOUT_MS = 20_000

# ── Own headless browser instance (separate from user-facing browser) ────

_browser = None
_context = None
_pw = None
_lock = asyncio.Lock()


async def _ensure_browser():
    """Lazy-init a headless-only browser for research. Separate from user browser."""
    global _browser, _context, _pw

    async with _lock:
        if _browser and _browser.is_connected():
            return _context

        try:
            try:
                from patchright.async_api import async_playwright
            except ImportError:
                from playwright.async_api import async_playwright

            _pw = await async_playwright().start()
            _browser = await _pw.chromium.launch(
                headless=True,
                args=[
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--disable-infobars",
                ],
            )
            _context = await _browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/137.0.0.0 Safari/537.36"
                ),
                java_script_enabled=True,
                ignore_https_errors=True,
                locale="en-US",
            )
            _context.set_default_timeout(NAV_TIMEOUT_MS)
            logger.info("[BROWSER_API] Headless research browser launched")
            return _context

        except ImportError:
            raise RuntimeError(
                "Browser not available. Install: pip install patchright && patchright install chromium"
            )
        except Exception:
            logger.exception("[BROWSER_API] Failed to launch headless browser")
            raise


async def shutdown():
    """Shut down the research browser."""
    global _browser, _context, _pw
    if _context:
        try:
            await _context.close()
        except Exception:
            pass
        _context = None
    if _browser:
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
    if _pw:
        try:
            await _pw.stop()
        except Exception:
            pass
        _pw = None


async def _get_page():
    """Create a new page from the research browser context.
    Returns (page, cleanup_fn) — caller MUST call cleanup_fn when done."""
    ctx = await _ensure_browser()
    page = await ctx.new_page()

    async def cleanup():
        try:
            await page.close()
        except Exception:
            pass

    return page, cleanup


# ── Public API ───────────────────────────────────────────────────────────

async def read_page(url: str, timeout_ms: int = NAV_TIMEOUT_MS) -> str:
    """Navigate to a URL and extract its text content.

    Uses a headless stealth browser — bypasses basic bot detection.
    Opens a temporary page, extracts text, closes it.

    Returns extracted text (max 15K chars) or error string.
    """
    cleanup = None
    try:
        page, cleanup = await _get_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await asyncio.sleep(1)

        # Extract main content — prefer article/main, fall back to body
        text = await page.evaluate("""() => {
            const remove = ['script', 'style', 'nav', 'footer', 'header',
                           'aside', 'noscript', 'iframe', '.ad', '.ads',
                           '[class*="cookie"]', '[class*="banner"]',
                           '[class*="popup"]', '[class*="modal"]'];
            remove.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => el.remove());
            });
            const main = document.querySelector('article')
                      || document.querySelector('main')
                      || document.querySelector('[role="main"]')
                      || document.body;
            if (!main) return '';
            return main.innerText || '';
        }""")

        if not text or not text.strip():
            text = await page.inner_text("body")

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

    Uses a headless stealth browser to navigate Google search.
    Returns list of {title, url, snippet} dicts.

    Falls back to DuckDuckGo if Google fails.
    """
    results = await _google_search(query, count)
    if results:
        return results
    results = await _bing_search(query, count)
    if results:
        return results
    return await _ddg_search(query, count)


async def _google_search(query: str, count: int) -> List[Dict[str, str]]:
    """Search Google via the headless browser."""
    cleanup = None
    try:
        page, cleanup = await _get_page()
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={count + 5}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        await asyncio.sleep(1.5)

        results = await page.evaluate("""(count) => {
            const results = [];
            // Strategy 1: Find all h3 elements (Google result titles) and walk up to container
            const h3s = document.querySelectorAll('h3');
            for (const h3 of h3s) {
                if (results.length >= count) break;
                // Walk up to find the link wrapper
                let container = h3.closest('a[href^="http"]') || h3.parentElement?.closest('a[href^="http"]');
                let linkEl = container || h3.parentElement?.querySelector('a[href^="http"]');
                if (!linkEl) {
                    // Try finding link as sibling or ancestor
                    let walk = h3.parentElement;
                    for (let i = 0; i < 5 && walk; i++) {
                        linkEl = walk.querySelector('a[href^="http"]');
                        if (linkEl) break;
                        walk = walk.parentElement;
                    }
                }
                if (linkEl && h3.innerText) {
                    // Find snippet near the h3
                    let snippet = '';
                    let walk = h3.parentElement;
                    for (let i = 0; i < 5 && walk; i++) {
                        const spans = walk.querySelectorAll('span, div');
                        for (const s of spans) {
                            const t = s.innerText || '';
                            if (t.length > 40 && t.length < 500 && t !== h3.innerText) {
                                snippet = t;
                                break;
                            }
                        }
                        if (snippet) break;
                        walk = walk.parentElement;
                    }
                    const url = linkEl.href || '';
                    // Skip Google internal links
                    if (url.includes('google.com/search') || url.includes('accounts.google')) continue;
                    results.push({
                        title: h3.innerText,
                        url: url,
                        snippet: snippet
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


async def _bing_search(query: str, count: int) -> List[Dict[str, str]]:
    """Fallback: search Bing (more permissive with headless browsers)."""
    cleanup = None
    try:
        page, cleanup = await _get_page()
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count={count + 5}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        await asyncio.sleep(1.5)

        results = await page.evaluate("""(count) => {
            const results = [];
            const items = document.querySelectorAll('li.b_algo, .b_algo');
            for (const item of items) {
                if (results.length >= count) break;
                const linkEl = item.querySelector('h2 a, a[href^="http"]');
                const snippetEl = item.querySelector('.b_caption p, p');
                if (linkEl && linkEl.innerText) {
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
            logger.info("[BROWSER_API] Bing search: %d results for '%s'", len(results), query[:50])
        return results or []

    except Exception as exc:
        logger.warning("[BROWSER_API] Bing search failed: %s", exc)
        return []
    finally:
        if cleanup:
            await cleanup()


async def _ddg_search(query: str, count: int) -> List[Dict[str, str]]:
    """Fallback: search DuckDuckGo via the headless browser (JS version)."""
    cleanup = None
    try:
        page, cleanup = await _get_page()
        search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        await asyncio.sleep(3)  # DDG JS needs time to render

        results = await page.evaluate("""(count) => {
            const results = [];
            // DDG result articles
            const articles = document.querySelectorAll('article, [data-testid="result"]');
            for (const article of articles) {
                if (results.length >= count) break;
                const linkEl = article.querySelector('a[href^="http"]');
                const titleEl = article.querySelector('h2, h3, [data-testid="result-title"]');
                const snippetEl = article.querySelector('[data-result="snippet"], .result__snippet, span');
                if (linkEl && (titleEl || linkEl.innerText)) {
                    const url = linkEl.href || '';
                    if (url.includes('duckduckgo.com')) continue;
                    results.push({
                        title: (titleEl || linkEl).innerText || '',
                        url: url,
                        snippet: snippetEl ? snippetEl.innerText : ''
                    });
                }
            }
            // Fallback: try finding any h2 with links (generic)
            if (results.length === 0) {
                const h2s = document.querySelectorAll('h2');
                for (const h2 of h2s) {
                    if (results.length >= count) break;
                    const link = h2.closest('a[href^="http"]') || h2.querySelector('a[href^="http"]');
                    if (link && h2.innerText && !link.href.includes('duckduckgo.com')) {
                        results.push({
                            title: h2.innerText,
                            url: link.href,
                            snippet: ''
                        });
                    }
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
