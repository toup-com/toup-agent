"""
Toup Page Reader — Extract clean article text from any URL without a browser.

Strategy:
  1. Fetch page via httpx with Chrome-like headers
  2. Try trafilatura for article extraction (best quality)
  3. Fall back to BeautifulSoup readability-style extraction
  4. Only use the Patchright browser if HTTP fetch gets empty/JS-only content

No API keys needed. No CAPTCHA. Works on 95% of websites.
"""

import asyncio
import logging
import ipaddress
import re
import socket
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.agent.smart_fetch._cache import TTLCache

logger = logging.getLogger(__name__)


def _assert_public_url(url: str) -> None:
    """SSRF guard (docs/security/audit-2026.md, re-audit): web_fetch only ever
    needs the public internet. Reject any URL whose host resolves to a
    private / loopback / link-local / metadata / CGNAT address so injected
    content can't point the agent at internal services (cloud metadata, the
    docker-bridge pgbouncer, another tenant's container, the bridge admin API).
    Called on the initial URL AND on every redirect hop."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"web_fetch: unsupported URL scheme {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("web_fetch: URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except Exception as e:
        raise ValueError(f"web_fetch: cannot resolve host {host!r}: {e}")
    _cgnat = ipaddress.ip_network("100.64.0.0/10")  # CGNAT / Tailscale range
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                or (ip.version == 4 and ip in _cgnat)):
            raise ValueError(
                f"web_fetch: refusing to fetch internal address {ip} (host {host!r})")


async def _guarded_get(client: "httpx.AsyncClient", url: str, headers: dict,
                       max_redirects: int = 5) -> "httpx.Response":
    """GET with the SSRF guard applied to the initial URL and every redirect
    hop (client must be created with follow_redirects=False).

    Round 4 (item 7b): the guard's ``getaddrinfo`` is a blocking syscall —
    run it in a worker thread so a slow resolver (100s of ms is normal)
    cannot stall the event loop and, with it, every concurrent fetch."""
    current = url
    for _ in range(max_redirects + 1):
        await asyncio.to_thread(_assert_public_url, current)
        resp = await client.get(current, headers=headers)
        if resp.is_redirect and resp.headers.get("location"):
            current = urljoin(current, resp.headers["location"])
            continue
        return resp
    raise ValueError("web_fetch: too many redirects")


# One shared client per process (Round 4, item 7b). A client per fetch paid a
# fresh TCP+TLS handshake on every page — with the model batching 3–4 reads
# per round that is 3–4 handshakes per round for nothing. Redirects stay
# manual + guarded (see _guarded_get). Not tenant-specific: it holds no
# cookies (cookies disabled) and no per-user state, so it survives /admin/bind.
_SHARED_CLIENT: Optional[httpx.AsyncClient] = None
_SHARED_CLIENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _client() -> httpx.AsyncClient:
    global _SHARED_CLIENT, _SHARED_CLIENT_LOOP
    loop = asyncio.get_running_loop()
    if (
        _SHARED_CLIENT is None
        or bool(getattr(_SHARED_CLIENT, "is_closed", False))
        or _SHARED_CLIENT_LOOP is not loop
        or _SHARED_CLIENT_LOOP.is_closed()
    ):
        _SHARED_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.fetch_http_timeout_s, connect=5.0),
            follow_redirects=False,  # redirects are followed manually, guarded per hop
            max_redirects=5,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            cookies=None,
        )
        _SHARED_CLIENT_LOOP = loop
    return _SHARED_CLIENT

# Per-tenant TTL+LRU cache of extracted page text, keyed on (requested url,
# max_chars) and the final post-redirect url. Cleared on /admin/bind
# (smart_fetch.clear_caches) so an in-place re-bind can't leak across tenants.
_PAGE_CACHE = TTLCache(maxsize=settings.fetch_cache_max, ttl_s=settings.fetch_cache_ttl_s)

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


def _bs_parser() -> str:
    """lxml is ~5–10× faster than html.parser on large pages and is already
    in the image (trafilatura depends on it). Fall back only if it is
    genuinely missing."""
    try:
        import lxml  # noqa: F401
        return "lxml"
    except ImportError:
        return "html.parser"


def _extract_with_bs4(html: str) -> str:
    """Fallback article extraction using BeautifulSoup."""
    soup = BeautifulSoup(html, _bs_parser())

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


def page_cache_key(url: str, max_chars: int) -> tuple:
    """Canonical cache key for a fetched page.

    Exported for the same reason ``search.cache_key`` is: a caller that fronts
    this cache with its own earlier tier (``ToolExecutor._tool_web_fetch``)
    must hash the url exactly the way ``toup_read_page`` does below, or the two
    tiers cache-miss each other forever and the probe is pure overhead.
    """
    return (url.strip(), max_chars)


def page_cache_get(url: str, max_chars: int) -> Optional[str]:
    """Read-through for callers ahead of this module in the fetch chain.

    ``ToolExecutor._tool_web_fetch`` consults this BEFORE metering, so a page
    served from cache is never billed — the same rule web_search already
    follows via ``search.cache_get``. Returns None when caching is disabled
    (``settings.fetch_cache_enabled``) or on a miss.

    This function is the one that went missing. The call site landed in
    tool_executor.py on 2026-07-31 and this half never did, so every
    ``web_fetch`` raised AttributeError, was swallowed by ``execute()``'s
    catch-all, and came back to the model as `ERROR: AttributeError: ...`.
    The agent could not read a web page on any tenant for four days, and
    nothing failed loudly enough for anyone to notice.
    """
    if not settings.fetch_cache_enabled:
        return None
    return _PAGE_CACHE.get(page_cache_key(url, max_chars))


def _lxml_meta(tree) -> dict:
    """Page metadata from an lxml tree via a handful of xpaths — cheap.
    (NOT trafilatura.extract_metadata: its date pass took 12 s on one real
    docs page, measured.)"""
    meta: dict = {}
    def _first(xp: str) -> str:
        try:
            v = tree.xpath(xp)
        except Exception:
            return ""
        if not v:
            return ""
        v0 = v[0]
        return (v0 if isinstance(v0, str) else (v0.text or "")).strip()
    meta["title"] = (
        _first("//meta[@property='og:title']/@content")
        or _first("//head/title/text()")
        or _first("//title/text()")
    )
    meta["description"] = (
        _first("//meta[@property='og:description']/@content")
        or _first("//meta[@name='description']/@content")
    )
    meta["author"] = _first("//meta[@name='author']/@content")
    date = (
        _first("//meta[@property='article:published_time']/@content")
        or _first("//meta[@name='datePublished']/@content")
        or _first("//meta[@property='og:article:published_time']/@content")
        or _first("//meta[@name='article:published_time']/@content")
    )
    if date:
        meta["date"] = date
    return meta


def _parse_and_extract(html: str, url: str, max_chars: int) -> Tuple[str, str]:
    """Sync, CPU-bound half of the reader — runs in a worker thread.

    Returns ``(status, payload)``: ``("js", body_text)`` for a JS-rendered
    shell, ``("empty", "")`` when nothing extractable, ``("ok", result)``.

    ONE parse (lxml, ~10–100 ms on a 500 KB page) serves the JS-shell check,
    the metadata, and trafilatura's extraction (it accepts the tree). The
    previous shape parsed every page TWICE — a full BeautifulSoup pass just
    to measure body text and read four <meta> tags, then trafilatura's own
    parse — at 1.5–2 s of CPU per page (measured on the founder's fetches).
    BeautifulSoup remains the fallback for pages trafilatura yields nothing on.
    """
    tree = None
    try:
        import trafilatura
        tree = trafilatura.load_html(html)
    except ImportError:
        trafilatura = None  # type: ignore[assignment]
    except Exception:
        tree = None

    if tree is not None:
        # JS-shell check on VISIBLE text: drop script/style/noscript first so
        # a 200 KB bundle in a <script> can't pass an empty page as content.
        try:
            for el in list(tree.iter("script", "style", "noscript", "template")):
                el.drop_tree()
        except Exception:
            pass
        try:
            body = tree.find(".//body")
            body_text = (body.text_content() if body is not None else tree.text_content()) or ""
        except Exception:
            body_text = ""
        body_text = " ".join(body_text.split())
        if len(body_text) < 100:
            return "js", body_text
        meta = _lxml_meta(tree)
        text = None
        try:
            text = trafilatura.extract(
                tree,
                url=url,
                include_comments=False,
                include_tables=True,
                include_links=False,
                include_images=False,
                favor_recall=True,
                deduplicate=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[ToupReader] trafilatura extraction failed: %s", e)
            text = None
        if not text:
            text = _extract_with_bs4(html)
    else:
        # No trafilatura / unparsable by lxml → the historical BeautifulSoup path.
        soup = BeautifulSoup(html, _bs_parser())
        body = soup.find("body")
        if body:
            body_text = body.get_text(strip=True)
            if len(body_text) < 100:
                return "js", body_text
        meta = _extract_metadata(soup)
        text = _extract_with_trafilatura(html, url) or _extract_with_bs4(html)
    if not text:
        return "empty", ""

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
    return "ok", result


async def toup_read_page(url: str, max_chars: int = 15000) -> str:
    """
    Fetch and extract clean text from a URL without a browser.

    Returns formatted text with title and content.
    Returns empty string if the page can't be fetched or is JS-only
    (caller should fall back to browser).

    Successful extractions are served from a short-lived per-tenant TTL+LRU
    cache (kill-switch ``settings.fetch_cache_enabled``); a repeat fetch of the
    same url within the TTL returns with zero network calls. Empty results (JS
    pages, 403s, errors that signal a browser fallback) are never cached.
    """
    cache_key = None
    if settings.fetch_cache_enabled:
        cache_key = page_cache_key(url, max_chars)
        cached = _PAGE_CACHE.get(cache_key)
        if cached is not None:
            logger.info("[PERF] web_fetch cache=hit url=%s", url[:80])
            return cached
    try:
        _t_net = asyncio.get_running_loop().time()
        resp = await _guarded_get(_client(), url, _HEADERS, max_redirects=5)
        resp.raise_for_status()
        _net_ms = int((asyncio.get_running_loop().time() - _t_net) * 1000)

        content_type = resp.headers.get("content-type", "")

        # Handle plain text and JSON directly
        if "text/plain" in content_type or "application/json" in content_type:
            text = resp.text[:max_chars]
            return text

        # Handle non-HTML content
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return f"(Binary content: {content_type})"

        html = resp.text

        # Round 4 (item 7b): parse + extract are CPU-bound (BeautifulSoup +
        # trafilatura on a few hundred KB of HTML: 1–6 s measured on the
        # founder's tenant). Done inline they BLOCKED THE EVENT LOOP, which
        # serialised the "parallel" fetch batch — the other pages' requests
        # only started after the first page's extraction finished. In a
        # worker thread the batch truly overlaps and the loop keeps serving
        # the stream. Bounded so a pathological page can't pin a worker.
        _t_parse = asyncio.get_running_loop().time()
        try:
            status, payload = await asyncio.wait_for(
                asyncio.to_thread(_parse_and_extract, html, url, max_chars),
                timeout=settings.fetch_extract_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("[ToupReader] extraction timed out (%.0fs) for %s",
                           settings.fetch_extract_timeout_s, url[:80])
            return ""
        _parse_ms = int((asyncio.get_running_loop().time() - _t_parse) * 1000)
        logger.info("[PERF] web_fetch net=%dms extract=%dms bytes=%d url=%s",
                    _net_ms, _parse_ms, len(html), url[:80])
        if status == "js":
            logger.info("[ToupReader] Page appears JS-rendered (%d chars), caller should use browser", len(payload))
            return ""  # Signal to caller: use browser fallback
        if status == "empty":
            return ""
        result = payload
        if cache_key is not None and result:
            _PAGE_CACHE.set(cache_key, result)
            final_key = (str(resp.url).strip(), max_chars)
            if final_key != cache_key:
                _PAGE_CACHE.set(final_key, result)  # dedup redirect chains
            logger.info("[PERF] web_fetch cache=miss url=%s", url[:80])
        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.info("[ToupReader] 403 Forbidden for %s — site blocks non-browser access", url)
            return ""  # Signal: use browser fallback
        return f"ERROR: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        logger.warning("[ToupReader] Failed to read %s: %s", url, e)
        return ""  # Signal: use browser fallback
