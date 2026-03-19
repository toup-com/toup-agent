"""
Research Sub-Agent — Researches an app category via the platform's browser,
then generates research-informed questions with clickable options.

Uses the platform's Patchright stealth browser (same engine as /browser page)
to search Google and read full articles — much richer than API snippets.

Called internally by AppBuilderSkill after the user picks a direction (Layer 0).
The user never sees this agent directly — its output is presented by the main agent.
"""

import logging
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


async def _browser_search(query: str, count: int = 5) -> str:
    """Search via the platform's stealth browser. Falls back to httpx if browser unavailable."""
    try:
        from .browser_api import search_formatted
        return await search_formatted(query, count)
    except Exception as exc:
        logger.warning("Browser search failed, falling back to httpx: %s", exc)
        return await _httpx_search_fallback(query, count)


async def _browser_read_page(url: str) -> str:
    """Read page content via the platform's stealth browser."""
    try:
        from .browser_api import read_page
        return await read_page(url)
    except Exception as exc:
        logger.warning("Browser read_page failed for %s: %s", url, exc)
        return f"(failed to read: {exc})"


async def _httpx_search_fallback(query: str, count: int) -> str:
    """Last-resort fallback: DuckDuckGo HTML scraping via httpx (no browser needed)."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Toup Agent)"},
            )
            resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result")[:count]
        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            title_el = r.select_one(".result__title a")
            snippet_el = r.select_one(".result__snippet")
            title = title_el.get_text(strip=True) if title_el else "Untitled"
            url = title_el.get("href", "") if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            lines.append(f"{i}. {title}")
            lines.append(f"   {url}")
            lines.append(f"   {snippet}")
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        return f"Search fallback failed: {exc}"


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------

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


async def run_research(
    original_idea: str,
    chosen_direction: str,
    call_llm: Callable[..., Awaitable[str]],
) -> str:
    """
    Research the chosen app direction and return formatted questions.

    Uses the platform's stealth browser to:
    1. Search Google for competing apps and features
    2. Read full articles from top results for deeper insights
    3. Feed everything to the LLM to generate research-informed questions

    Args:
        original_idea: The user's raw input (e.g., "I want a fitness app")
        chosen_direction: The direction the user picked from Layer 0
        call_llm: Reference to AppBuilderSkill._call_llm for making LLM calls

    Returns:
        Formatted string with 5-7 research-informed questions and [[option]] buttons
    """
    # Step 1: Run 3 targeted searches
    search_queries = [
        f"best {chosen_direction} apps 2025 features",
        f"{chosen_direction} app UX design patterns common features",
        f"{chosen_direction} mobile app what users want reviews",
    ]

    all_search_results = []
    top_urls = []

    for query in search_queries:
        try:
            result = await _browser_search(query, count=5)
            all_search_results.append(f"Query: {query}\n{result}")
            # Extract URLs from results for deep reading
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("http") and len(top_urls) < 3:
                    top_urls.append(line)
        except Exception as exc:
            logger.warning("Search failed for '%s': %s", query, exc)
            all_search_results.append(f"Query: {query}\n(search failed)")

    combined_search = "\n\n".join(all_search_results)

    # Step 2: Read top 2-3 articles for deeper content
    article_contents = []
    for url in top_urls[:2]:  # Read max 2 articles to keep it fast
        try:
            content = await _browser_read_page(url)
            if content and not content.startswith("(failed") and len(content) > 100:
                # Truncate each article to keep LLM context manageable
                truncated = content[:5000] if len(content) > 5000 else content
                article_contents.append(f"URL: {url}\n{truncated}")
        except Exception as exc:
            logger.warning("Failed to read article %s: %s", url, exc)

    combined_articles = "\n\n---\n\n".join(article_contents) if article_contents else "(no articles could be read)"

    # Step 3: Call LLM to analyze research and generate questions
    user_message = RESEARCH_USER_TEMPLATE.format(
        original_idea=original_idea,
        chosen_direction=chosen_direction,
        search_results=combined_search,
        article_content=combined_articles,
    )

    questions = await call_llm(
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        user_message=user_message,
        model="claude-sonnet-4-6",
        purpose="research_agent_questions",
        max_tokens=4000,
    )

    return questions.strip()
