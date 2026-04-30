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


# ── Claude Code CLI identity (OAuth-token traffic) ────────────────────
# The actual Claude Code CLI's outbound request signature. Anthropic
# classifies OAuth-token traffic into entitlement / billing buckets
# via these headers; matching the real CLI is the safest bet for
# consistent behaviour on Claude subscription tokens.
#
# Source: anthropics/claude-code release 2.1.2 (verified 2026-04-27).
# Update both strings in lockstep when bumping to a newer CLI version.
#
# Decision history: previously we sent `claude-code/1.0.33` + `x-app:
# claude-code`. That worked until shared-subscription debugging surfaced
# evidence Anthropic may differentiate billing buckets by these strings.
# Switched to the CLI's exact identity to match the bucket Claude Code
# users would normally consume from. See docs/llm-model-refactor-plan.md
# §2.1 for the full reasoning.
_CLAUDE_CLI_USER_AGENT = "claude-cli/2.1.2 (external, cli)"
_CLAUDE_CLI_X_APP = "cli"
_CLAUDE_CODE_BETA = "claude-code-20250219,oauth-2025-04-20"


def _bundle_active() -> bool:
    return settings.llm_mode == "bundle" and bool(settings.toup_token)


async def _normalize_headers_for_waf(request):
    """httpx event hook that rewrites outbound headers to bypass Cloudflare WAF.

    The Anthropic / OpenAI SDKs add bot-signature headers regardless of what
    we pass at construction time:
      - user-agent: "AsyncAnthropic/Python 0.96.0" — flagged as automation
      - x-stainless-*: telemetry headers (lang, package-version, os, arch,
        runtime, async/stream helper, retry-count) — Stainless is the SDK
        generator Anthropic and OpenAI both use, and Cloudflare matches on
        the x-stainless-package-version pattern.

    Setting these in the SDK's `default_headers` *appends* values rather than
    overriding (we get "AsyncAnthropic/Python 0.96.0, toup-agent/1.0"). The
    only way to fully override is at the httpx layer, just before the request
    goes on the wire. This hook does that.
    """
    request.headers["user-agent"] = _WAF_SAFE_UA
    for h in list(request.headers.keys()):
        if h.lower().startswith("x-stainless-"):
            del request.headers[h]


def _proxy_http_client(timeout: float = 120.0):
    """Build an httpx.AsyncClient with the WAF-bypass header normalizer
    installed. All bundle-mode requests go through this client so the
    SDK's bot-signature headers are scrubbed before reaching Cloudflare."""
    import httpx
    return httpx.AsyncClient(
        timeout=timeout,
        event_hooks={"request": [_normalize_headers_for_waf]},
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

    # OAuth token (Claude Code style) — needs the CLI identity headers.
    # See _CLAUDE_CLI_USER_AGENT / _CLAUDE_CLI_X_APP / _CLAUDE_CODE_BETA above.
    if "sk-ant-oat" in key:
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        import httpx
        return anthropic.AsyncAnthropic(
            auth_token=key,
            http_client=httpx.AsyncClient(
                headers={"user-agent": _CLAUDE_CLI_USER_AGENT},
            ),
            default_headers={
                "anthropic-beta": _CLAUDE_CODE_BETA,
                "x-app": _CLAUDE_CLI_X_APP,
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
