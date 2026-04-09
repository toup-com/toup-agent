"""
Research Sub-Agent — Researches an app category and generates informed questions.

Search backend priority (fast path first, browser as last resort):
  1. httpx → DuckDuckGo HTML (fastest, no browser overhead)
  2. httpx → Google (if API key configured)
  3. Headless browser → Google (slowest, highest quality, last resort)

Each search runs with tight per-backend timeouts. Three concurrent queries
race via asyncio.gather(return_exceptions=True). First usable result wins.
Results cached in an LRU by query hash (1000 entries, 1h TTL).

Called internally by AppBuilderSkill after the user picks a direction (Layer 0).
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Callable, Awaitable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Timeouts (all in seconds) ──────────────────────────────────────
RESEARCH_TIMEOUT = 25          # Hard ceiling for entire research step
HTTPX_CONNECT_TIMEOUT = 5      # httpx connect timeout
HTTPX_TOTAL_TIMEOUT = 8        # httpx total request timeout
BROWSER_PAGE_TIMEOUT = 8000    # Browser page load timeout (ms, passed to playwright)
LLM_TIMEOUT = 15               # LLM question generation timeout

# ── Cache ──────────────────────────────────────────────────────────
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 3600  # 1 hour

_search_cache: OrderedDict[str, Tuple[float, str]] = OrderedDict()


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]


def _cache_get(query: str) -> Optional[str]:
    key = _cache_key(query)
    if key in _search_cache:
        ts, result = _search_cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            _search_cache.move_to_end(key)
            return result
        del _search_cache[key]
    return None


def _cache_set(query: str, result: str) -> None:
    key = _cache_key(query)
    _search_cache[key] = (time.time(), result)
    while len(_search_cache) > CACHE_MAX_SIZE:
        _search_cache.popitem(last=False)


# ── Search backends ────────────────────────────────────────────────

async def _httpx_duckduckgo(query: str, count: int = 5) -> str:
    """Fast path: DuckDuckGo HTML scraping via httpx (no browser needed)."""
    import httpx
    t0 = time.time()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(HTTPX_TOTAL_TIMEOUT, connect=HTTPX_CONNECT_TIMEOUT),
            follow_redirects=True,
        ) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Toup Research Agent)"},
            )
            resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result")[:count]
        if not results:
            logger.info("[RESEARCH] httpx/ddg returned 0 results for '%s' in %.1fs", query, time.time() - t0)
            return ""

        lines: List[str] = []
        for i, r in enumerate(results, 1):
            title_el = r.select_one(".result__title a")
            snippet_el = r.select_one(".result__snippet")
            title = title_el.get_text(strip=True) if title_el else "Untitled"
            url = title_el.get("href", "") if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            lines.append(f"{i}. {title}")
            if url:
                lines.append(f"   {url}")
            lines.append(f"   {snippet}")
            lines.append("")
        result = "\n".join(lines)
        logger.info("[RESEARCH] httpx/ddg: %d results for '%s' in %.1fs", len(results), query[:40], time.time() - t0)
        return result
    except Exception as exc:
        logger.warning("[RESEARCH] httpx/ddg failed for '%s' in %.1fs: %s", query[:40], time.time() - t0, exc)
        return ""


async def _browser_search_backend(query: str, count: int = 5) -> str:
    """Last resort: headless browser Google search. Slow but high quality."""
    t0 = time.time()
    try:
        from .browser_api import search_formatted
        result = await asyncio.wait_for(
            search_formatted(query, count),
            timeout=BROWSER_PAGE_TIMEOUT / 1000 + 2,  # browser timeout + overhead
        )
        logger.info("[RESEARCH] browser/google: results for '%s' in %.1fs", query[:40], time.time() - t0)
        return result
    except asyncio.TimeoutError:
        logger.warning("[RESEARCH] browser/google TIMEOUT for '%s' after %.1fs", query[:40], time.time() - t0)
        return ""
    except Exception as exc:
        logger.warning("[RESEARCH] browser/google failed for '%s' in %.1fs: %s", query[:40], time.time() - t0, exc)
        return ""


async def _search_with_fallback(query: str, count: int = 5) -> str:
    """Try search backends in priority order. Return first usable result."""
    # Check cache first
    cached = _cache_get(query)
    if cached:
        logger.info("[RESEARCH] cache hit for '%s'", query[:40])
        return cached

    # Try httpx/DuckDuckGo first (fast path)
    result = await _httpx_duckduckgo(query, count)
    if result and len(result) > 50:
        _cache_set(query, result)
        return result

    # Fall back to headless browser (slow path)
    result = await _browser_search_backend(query, count)
    if result and len(result) > 50:
        _cache_set(query, result)
        return result

    logger.warning("[RESEARCH] all search backends exhausted for '%s'", query[:40])
    return f"(no results found for: {query})"


async def _read_page_fast(url: str) -> str:
    """Read a page via httpx first, browser as fallback."""
    t0 = time.time()
    # Try httpx first (no browser overhead)
    try:
        import httpx
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(HTTPX_TOTAL_TIMEOUT, connect=HTTPX_CONNECT_TIMEOUT),
            follow_redirects=True,
        ) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (Toup Research Agent)"})
            resp.raise_for_status()
            if len(resp.text) > 200:
                # Strip HTML tags for a rough text extraction
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    logger.info("[RESEARCH] httpx read: %d chars from %s in %.1fs", len(text), url[:60], time.time() - t0)
                    return text[:5000]
    except Exception as exc:
        logger.debug("[RESEARCH] httpx read failed for %s: %s", url[:60], exc)

    # Fall back to browser
    try:
        from .browser_api import read_page
        content = await asyncio.wait_for(read_page(url), timeout=BROWSER_PAGE_TIMEOUT / 1000 + 2)
        if content and not content.startswith("(failed") and len(content) > 100:
            logger.info("[RESEARCH] browser read: %d chars from %s in %.1fs", len(content), url[:60], time.time() - t0)
            return content[:5000]
    except Exception as exc:
        logger.debug("[RESEARCH] browser read failed for %s: %s", url[:60], exc)

    return ""


# ── Prompts ────────────────────────────────────────────────────────

RESEARCH_SYSTEM_PROMPT = """You are a Research Agent for an app builder platform.

Your job: Given an app category/direction the user chose, research what exists in that space
and generate EXACTLY 5-7 research-informed questions with clickable option buttons.

You will receive:
1. The user's original idea
2. The specific direction they chose
3. Web search results about competing apps and common features in this space
4. Full article content from top search results (when available)

Based on the research, generate questions that:
- Reference SPECIFIC findings (app names, features, patterns you found)
- Help the user make informed decisions by showing what similar apps do
- Cover: key differentiators, feature priorities, content approach, engagement model, monetization
- Each question MUST have 2-4 clickable [[option]] buttons on the next line
- Questions should feel insightful — the user should think "wow, they did their homework"

Output format — ONLY the questions, nothing else:
```
1. **[Question text referencing research finding]**
[[Option A]] [[Option B]] [[Option C]]

2. **[Question text referencing research finding]**
[[Option A]] [[Option B]]

...
```

RULES:
- Exactly 5-7 questions, no more, no less
- Every question MUST reference something specific from the research (an app name, a feature, a pattern)
- Every question MUST have [[option]] buttons on the NEXT LINE
- Do NOT include any preamble, explanation, or closing text — ONLY the numbered questions
- Keep questions SHORT (one line). Put detail in the options.
- Options should be mutually exclusive and cover the reasonable choices
"""

RESEARCH_USER_TEMPLATE = """User's original idea: "{original_idea}"
Direction they chose: "{chosen_direction}"

Here is what I found from researching this app category:

--- SEARCH RESULTS ---
{search_results}
--- END SEARCH RESULTS ---

--- ARTICLE CONTENT (from top results) ---
{article_content}
--- END ARTICLE CONTENT ---

Generate 5-7 research-informed questions with [[option]] buttons based on these findings.
Reference specific apps, features, and patterns you found in the research."""


# ── Main entry point ───────────────────────────────────────────────

async def run_research(
    original_idea: str,
    chosen_direction: str,
    call_llm: Callable[..., Awaitable[str]],
    job_id: Optional[str] = None,
) -> str:
    """
    Research the chosen app direction and return formatted questions.

    Search order: httpx/DuckDuckGo (fast) → browser/Google (slow).
    Hard timeout of RESEARCH_TIMEOUT seconds. Falls back to LLM-only
    if all search backends fail.
    """
    t0 = time.time()
    log_prefix = f"[RESEARCH job={job_id[:8]}]" if job_id else "[RESEARCH]"
    logger.info("%s Starting for: %s", log_prefix, chosen_direction[:60])

    try:
        result = await asyncio.wait_for(
            _run_research_inner(original_idea, chosen_direction, call_llm, log_prefix),
            timeout=RESEARCH_TIMEOUT,
        )
        logger.info("%s Completed in %.1fs — %d chars", log_prefix, time.time() - t0, len(result))
        return result
    except asyncio.TimeoutError:
        logger.warning("%s TIMEOUT after %ds — falling back to LLM-only", log_prefix, RESEARCH_TIMEOUT)
        result = await _generate_fallback_questions(original_idea, chosen_direction, call_llm)
        logger.info("%s Fallback completed in %.1fs — %d chars", log_prefix, time.time() - t0, len(result))
        return result
    except Exception as exc:
        logger.error("%s ERROR: %s: %s", log_prefix, type(exc).__name__, exc)
        # Last resort: try fallback questions
        try:
            return await _generate_fallback_questions(original_idea, chosen_direction, call_llm)
        except Exception:
            raise exc


async def _run_research_inner(
    original_idea: str,
    chosen_direction: str,
    call_llm: Callable[..., Awaitable[str]],
    log_prefix: str = "[RESEARCH]",
) -> str:
    """Inner research function — runs search + article reading + LLM."""

    # Step 1: Run 3 targeted searches CONCURRENTLY with fallback chain per query
    t1 = time.time()
    logger.info("%s Step 1: Starting concurrent searches (httpx-first)...", log_prefix)
    search_queries = [
        f"best {chosen_direction} apps 2025 features",
        f"{chosen_direction} app UX design patterns common features",
        f"{chosen_direction} mobile app what users want reviews",
    ]

    async def _safe_search(query: str) -> Tuple[str, List[str]]:
        """Run search with full fallback chain, return (text, urls)."""
        try:
            result = await _search_with_fallback(query, count=5)
            urls = [
                line.strip() for line in result.split("\n")
                if line.strip().startswith("http")
            ][:2]
            return (f"Query: {query}\n{result}", urls)
        except Exception as exc:
            logger.warning("%s Search failed for '%s': %s", log_prefix, query[:40], exc)
            return (f"Query: {query}\n(search failed)", [])

    search_results = await asyncio.gather(
        *[_safe_search(q) for q in search_queries],
        return_exceptions=True,
    )

    all_search_text: List[str] = []
    top_urls: List[str] = []
    for sr in search_results:
        if isinstance(sr, Exception):
            logger.warning("%s Search exception: %s", log_prefix, sr)
            continue
        text, urls = sr
        all_search_text.append(text)
        for u in urls:
            if len(top_urls) < 3:
                top_urls.append(u)

    combined_search = "\n\n".join(all_search_text) if all_search_text else "(no search results)"
    logger.info("%s Step 1 done in %.1fs — %d results, %d URLs", log_prefix, time.time() - t1, len(all_search_text), len(top_urls))

    # Step 2: Read top 2 articles concurrently (httpx-first, browser fallback)
    t2 = time.time()
    article_contents: List[str] = []
    if top_urls:
        async def _safe_read(url: str) -> str:
            try:
                content = await _read_page_fast(url)
                if content and len(content) > 100:
                    return f"URL: {url}\n{content}"
            except Exception as exc:
                logger.debug("%s Failed to read %s: %s", log_prefix, url[:60], exc)
            return ""

        read_results = await asyncio.gather(
            *[_safe_read(u) for u in top_urls[:2]],
            return_exceptions=True,
        )
        for rr in read_results:
            if isinstance(rr, str) and rr:
                article_contents.append(rr)

    combined_articles = "\n\n---\n\n".join(article_contents) if article_contents else "(no articles could be read)"
    logger.info("%s Step 2 done in %.1fs — %d articles", log_prefix, time.time() - t2, len(article_contents))

    # Step 3: LLM question generation
    t3 = time.time()
    logger.info("%s Step 3: Generating questions via LLM...", log_prefix)
    user_message = RESEARCH_USER_TEMPLATE.format(
        original_idea=original_idea,
        chosen_direction=chosen_direction,
        search_results=combined_search,
        article_content=combined_articles,
    )

    questions = await call_llm(
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        user_message=user_message,
        model="",
        purpose="research_agent_questions",
        max_tokens=4000,
    )
    logger.info("%s Step 3 done in %.1fs — %d chars", log_prefix, time.time() - t3, len(questions))

    return questions.strip()


async def _generate_fallback_questions(
    original_idea: str,
    chosen_direction: str,
    call_llm: Callable[..., Awaitable[str]],
) -> str:
    """Generate research-style questions without web search (used on timeout/failure)."""
    fallback_prompt = (
        f'User\'s original idea: "{original_idea}"\n'
        f'Direction they chose: "{chosen_direction}"\n\n'
        "Web research was unavailable. Generate 5-7 insightful questions about this app category "
        "based on your knowledge of similar apps and common patterns in this space. "
        "Reference well-known apps where relevant. "
        "Each question MUST have [[option]] buttons on the next line."
    )
    questions = await call_llm(
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        user_message=fallback_prompt,
        model="",
        purpose="research_fallback_questions",
        max_tokens=4000,
    )
    return questions.strip()
