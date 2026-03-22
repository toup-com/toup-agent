"""
Toup Page Reader — Extract clean article text from any URL without a browser.

Strategy:
  1. Fetch page via httpx with Chrome-like headers
  2. Try trafilatura for article extraction (best quality)
  3. Fall back to BeautifulSoup readability-style extraction
  4. Only use the Patchright browser if HTTP fetch gets empty/JS-only content

No API keys needed. No CAPTCHA. Works on 95% of websites.
"""

import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _extract_with_trafilatura(html: str, url: str) -> Optional[str]:
    """Extract article text using trafilatura (high-quality extraction)."""
    try:
        import trafilatura
        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_images=False,
            favor_recall=True,
            deduplicate=True,
        )
        return result
    except ImportError:
        logger.debug("[ToupReader] trafilatura not installed, using BeautifulSoup fallback")
        return None
    except Exception as e:
        logger.warning("[ToupReader] trafilatura extraction failed: %s", e)
        return None


def _extract_with_bs4(html: str) -> str:
    """Fallback article extraction using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "iframe", "form", "button", "svg", "img"]):
        tag.decompose()

    # Remove hidden elements
    for el in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none")}):
        el.decompose()
    for el in soup.find_all(attrs={"hidden": True}):
        el.decompose()

    # Try to find the main content area
    content = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find(attrs={"id": re.compile(r"content|article|post|entry", re.I)})
        or soup.find(attrs={"class": re.compile(r"content|article|post|entry", re.I)})
        or soup.find("body")
    )

    if not content:
        return ""

    # Extract text with structure
    text = content.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def _extract_metadata(soup: BeautifulSoup) -> dict:
    """Extract page metadata (title, description, author, date)."""
    meta = {}

    # Title
    og_title = soup.find("meta", property="og:title")
    title_tag = soup.find("title")
    meta["title"] = (
        og_title["content"] if og_title and og_title.get("content")
        else title_tag.get_text(strip=True) if title_tag
        else ""
    )

    # Description
    og_desc = soup.find("meta", property="og:description")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    meta["description"] = (
        og_desc["content"] if og_desc and og_desc.get("content")
        else meta_desc["content"] if meta_desc and meta_desc.get("content")
        else ""
    )

    # Author
    author = soup.find("meta", attrs={"name": "author"})
    meta["author"] = author["content"] if author and author.get("content") else ""

    # Published date
    for prop in ["article:published_time", "datePublished", "og:article:published_time"]:
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            meta["date"] = tag["content"]
            break

    return meta


async def toup_read_page(url: str, max_chars: int = 15000) -> str:
    """
    Fetch and extract clean text from a URL without a browser.

    Returns formatted text with title and content.
    Returns empty string if the page can't be fetched or is JS-only
    (caller should fall back to browser).
    """
    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        # Handle plain text and JSON directly
        if "text/plain" in content_type or "application/json" in content_type:
            text = resp.text[:max_chars]
            return text

        # Handle non-HTML content
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return f"(Binary content: {content_type})"

        html = resp.text

        # Check if page is mostly JS-rendered (very little text content)
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body")
        if body:
            body_text = body.get_text(strip=True)
            # If body has < 100 chars of text, it's probably JS-rendered
            if len(body_text) < 100:
                logger.info("[ToupReader] Page appears JS-rendered (%d chars), caller should use browser", len(body_text))
                return ""  # Signal to caller: use browser fallback

        # Extract metadata
        meta = _extract_metadata(soup)

        # Try trafilatura first (best quality), then BS4
        text = _extract_with_trafilatura(html, url) or _extract_with_bs4(html)

        if not text:
            return ""

        # Format output with metadata
        parts = []
        if meta.get("title"):
            parts.append(f"# {meta['title']}")
        if meta.get("author"):
            parts.append(f"Author: {meta['author']}")
        if meta.get("date"):
            parts.append(f"Date: {meta['date']}")
        if parts:
            parts.append("")  # blank line
        parts.append(text)

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"
        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.info("[ToupReader] 403 Forbidden for %s — site blocks non-browser access", url)
            return ""  # Signal: use browser fallback
        return f"ERROR: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        logger.warning("[ToupReader] Failed to read %s: %s", url, e)
        return ""  # Signal: use browser fallback
