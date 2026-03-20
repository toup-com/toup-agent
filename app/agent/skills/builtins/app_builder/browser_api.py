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
from urllib.parse import quote_plus, unquote

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
    """Search the web and extract results.

    Uses a headless stealth browser. Tries Google first, then Bing,
    then DuckDuckGo HTML scrape (httpx, no browser) as last resort.

    Returns list of {title, url, snippet} dicts.
    """
    results = await _google_search(query, count)
    if results:
        return results
    results = await _bing_search(query, count)
    if results:
        return results
    return await _ddg_html_search(query, count)


async def _google_search(query: str, count: int) -> List[Dict[str, str]]:
    """Search Google via the headless browser."""
    cleanup = None
    try:
        page, cleanup = await _get_page()
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={count + 5}&hl=en"
        await page.goto(search_url, wait_until="load", timeout=NAV_TIMEOUT_MS)
        await asyncio.sleep(2)

        # Scroll down to trigger lazy-loaded results
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        await asyncio.sleep(0.5)

        results = await page.evaluate("""(count) => {
            const results = [];
            const h3s = document.querySelectorAll('h3');
            for (const h3 of h3s) {
                if (results.length >= count) break;
                let container = h3.closest('a[href^="http"]') || h3.parentElement?.closest('a[href^="http"]');
                let linkEl = container || h3.parentElement?.querySelector('a[href^="http"]');
                if (!linkEl) {
                    let walk = h3.parentElement;
                    for (let i = 0; i < 5 && walk; i++) {
                        linkEl = walk.querySelector('a[href^="http"]');
                        if (linkEl) break;
                        walk = walk.parentElement;
                    }
                }
                if (linkEl && h3.innerText) {
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
    """Fallback: search Bing via headless browser."""
    cleanup = None
    try:
        page, cleanup = await _get_page()
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count={count + 5}&setlang=en&cc=US"
        await page.goto(search_url, wait_until="load", timeout=NAV_TIMEOUT_MS)
        await asyncio.sleep(2)

        results = await page.evaluate("""(count) => {
            const results = [];
            const items = document.querySelectorAll('li.b_algo, .b_algo');
            for (const item of items) {
                if (results.length >= count) break;
                const h2a = item.querySelector('h2 a');
                const snippetEl = item.querySelector('.b_caption p') || item.querySelector('p');
                if (!h2a || !h2a.innerText) continue;

                // Extract URL: prefer h2a.href (Bing redirect), fall back to cite text
                let url = '';
                const rawHref = h2a.getAttribute('href') || '';
                if (rawHref.startsWith('http') && !rawHref.includes('bing.com/ck/')) {
                    url = rawHref;
                } else {
                    // cite shows "example.com › path › page" — reconstruct URL
                    const cite = item.querySelector('cite');
                    if (cite) {
                        url = cite.innerText.trim().replace(/\s*›\s*/g, '/');
                        if (!url.startsWith('http')) url = 'https://' + url;
                    }
                }

                if (url && !url.includes('bing.com')) {
                    results.push({
                        title: h2a.innerText,
                        url: url,
                        snippet: snippetEl ? snippetEl.innerText.substring(0, 300) : ''
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


async def _ddg_html_search(query: str, count: int) -> List[Dict[str, str]]:
    """Last-resort fallback: DuckDuckGo HTML via httpx (no browser needed).

    The JS version of DDG blocks headless browsers, so we use the
    old-school HTML-only version with httpx.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/137.0.0.0 Safari/537.36"
                    ),
                },
            )
            resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select(".result, .result--main")[:count]

        if not items:
            return []

        results = []
        for r in items:
            title_el = r.select_one(".result__title a, .result__a")
            snippet_el = r.select_one(".result__snippet")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            # DDG wraps URLs in a redirect — extract actual URL
            if "uddg=" in href:
                try:
                    href = unquote(href.split("uddg=")[1].split("&")[0])
                except Exception:
                    pass
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if title and href.startswith("http"):
                results.append({"title": title, "url": href, "snippet": snippet})

        if results:
            logger.info("[BROWSER_API] DDG HTML search: %d results for '%s'", len(results), query[:50])
        return results

    except Exception as exc:
        logger.warning("[BROWSER_API] DDG HTML search failed: %s", exc)
        return []


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
