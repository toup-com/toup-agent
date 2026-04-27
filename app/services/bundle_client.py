"""
Bundle-aware LLM client factory.

Single place where Anthropic / OpenAI SDK clients are constructed with
the right base_url, api_key, and (critically) user-agent for the active
LLM mode. Every direct `AsyncAnthropic(...)` / `AsyncOpenAI(...)` call
in agent code should go through `make_anthropic_client` / `make_openai_client`
here so that:

  - bundle subscribers never accidentally bypass the proxy (would expose
    a master provider key OR fail with no key)
  - the SDK's default user-agent never reaches Cloudflare in front of
    toup.ai (was 403'd as a bot signature — matin incident 2026-04-27)
  - per-user OpenAI auto-provisioned keys flow through the proxy's
    routing instead of being hardcoded into the agent's env

When llm_mode != "bundle" or toup_token is unset, the helpers fall back
to BYOK construction with the caller's key. OAuth-token detection
(sk-ant-oat) is preserved on the BYOK path.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


# WAF-safe user-agent that Cloudflare's bot management does not flag.
# The default SDK user-agent ("Anthropic/Python X.Y.Z" / "OpenAI/Python ...")
# is a known automation signature; Cloudflare returns 403 "Your request
# was blocked." for those on toup.ai. Identifying as a Toup-internal
# client passes cleanly. If we ever update the WAF rules, we can retire
# this override.
_WAF_SAFE_UA = "toup-agent/1.0 (bundle-proxy)"


def _bundle_active() -> bool:
    return settings.llm_mode == "bundle" and bool(settings.toup_token)


def _proxy_http_client(timeout: float = 120.0):
    """Build an httpx.AsyncClient with the WAF-safe user-agent."""
    import httpx
    return httpx.AsyncClient(
        headers={"user-agent": _WAF_SAFE_UA},
        timeout=timeout,
    )


def make_anthropic_client(byok_key: Optional[str] = None):
    """
    Return an AsyncAnthropic client appropriate for the active LLM mode.

    Bundle mode: routes through the Toup proxy with TOUP_TOKEN as auth.
    BYOK mode: direct to api.anthropic.com using `byok_key` (caller-provided)
    or settings.anthropic_api_key, with sk-ant-oat OAuth detection.

    Returns None if neither bundle mode is active nor a key is available
    (caller should handle this — usually means misconfigured tenant).
    """
    import anthropic

    if _bundle_active():
        base_url = f"{settings.platform_api_url.rstrip('/')}/llm"
        return anthropic.AsyncAnthropic(
            api_key=settings.toup_token,
            base_url=base_url,
            http_client=_proxy_http_client(),
        )

    key = (byok_key or settings.anthropic_api_key or "").strip()
    if not key:
        return None

    # OAuth token (Claude Code style) needs special headers
    if "sk-ant-oat" in key:
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        import httpx
        return anthropic.AsyncAnthropic(
            auth_token=key,
            http_client=httpx.AsyncClient(
                headers={"user-agent": "claude-code/1.0.33"},
            ),
            default_headers={
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                "x-app": "claude-code",
            },
        )
    return anthropic.AsyncAnthropic(api_key=key)


def make_openai_client(byok_key: Optional[str] = None):
    """
    Return an AsyncOpenAI client appropriate for the active LLM mode.

    Bundle mode: routes through the Toup proxy at /llm/openai/v1.
    BYOK mode: direct to api.openai.com using `byok_key` or
    settings.openai_api_key.

    Returns None if neither bundle mode is active nor a key is available.
    """
    from openai import AsyncOpenAI

    if _bundle_active():
        base_url = f"{settings.platform_api_url.rstrip('/')}/llm/openai/v1"
        return AsyncOpenAI(
            api_key=settings.toup_token,
            base_url=base_url,
            http_client=_proxy_http_client(),
        )

    key = (byok_key or settings.openai_api_key or "").strip()
    if not key:
        return None
    return AsyncOpenAI(api_key=key)
