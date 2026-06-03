"""
KeyProvider — Single source of truth for LLM API keys at runtime.

All services that need API keys must use this module instead of reading
from settings/env directly. This enables live key updates when users
change their keys on the settings page without restarting the agent.

Flow:
  1. Agent starts → KeyProvider reads from settings (env vars / .env)
  2. User changes key on toup.ai → platform pushes config_update
  3. tunnel_client writes new .env, reloads settings, calls refresh()
  4. KeyProvider detects changed keys → marks clients stale
  5. Next LLM call → service gets fresh client with new key

Usage:
  from app.services.key_provider import keys
  key = keys.openai        # Returns current OpenAI key (or None)
  key = keys.anthropic     # Returns current Anthropic key (or None)
  keys.refresh()           # Re-read from settings (after config sync)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KeyProvider:
    """In-memory API key store with refresh support."""

    def __init__(self):
        self._openai: Optional[str] = None
        self._anthropic: Optional[str] = None
        self._google: Optional[str] = None
        self._mistral: Optional[str] = None
        self._xai: Optional[str] = None
        self._deepseek: Optional[str] = None
        self._brave: Optional[str] = None
        self._cohere: Optional[str] = None
        # Bundle credential. The bundle LLM client (make_anthropic_client /
        # make_openai_client) uses settings.toup_token as its api_key against
        # the platform proxy. A pool container boots GENERIC with a lobby/empty
        # toup_token; /admin/bind then maps the user's connect_token →
        # settings.toup_token and calls keys.refresh(). If refresh() doesn't
        # treat a toup_token change as a key change, the version never bumps,
        # the cached LLM client keeps the STALE token, and the user's first
        # message 401s ("Your API key is invalid") until something else happens
        # to change a provider key. Tracking it here closes that race.
        self._toup_token: Optional[str] = None
        self._llm_mode: Optional[str] = None
        self._version: int = 0  # Bumped on every refresh — services check this
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.refresh()

    def refresh(self):
        """Re-read keys from settings. Call after config sync."""
        from app.config import settings

        old_openai = self._openai
        old_anthropic = self._anthropic
        old_toup = self._toup_token
        old_mode = self._llm_mode

        self._openai = (settings.openai_api_key or "").strip() or None
        self._anthropic = (settings.anthropic_api_key or "").strip() or None
        self._google = (settings.google_api_key or "").strip() or None
        self._mistral = (settings.mistral_api_key or "").strip() or None
        self._xai = (settings.xai_api_key or "").strip() or None
        self._deepseek = (settings.deepseek_api_key or "").strip() or None
        self._brave = (settings.brave_api_key or "").strip() or None
        self._cohere = (settings.cohere_api_key or "").strip() or None
        # Bundle credential + mode — see __init__. A bind that flips the
        # container from GENERIC (empty token) to a bound user MUST bump the
        # version so the cached bundle client is rebuilt with the new token.
        self._toup_token = (getattr(settings, "toup_token", "") or "").strip() or None
        self._llm_mode = (getattr(settings, "llm_mode", "") or "").strip() or None
        self._loaded = True

        changed = (
            (old_openai != self._openai)
            or (old_anthropic != self._anthropic)
            or (old_toup != self._toup_token)
            or (old_mode != self._llm_mode)
        )
        if changed:
            self._version += 1
            logger.info(
                "[KEY-PROVIDER] Keys refreshed (v%d): openai=%s anthropic=%s bundle_token=%s mode=%s",
                self._version,
                "set" if self._openai else "none",
                "set" if self._anthropic else "none",
                "set" if self._toup_token else "none",
                self._llm_mode or "none",
            )

    @property
    def version(self) -> int:
        self._ensure_loaded()
        return self._version

    @property
    def openai(self) -> Optional[str]:
        self._ensure_loaded()
        return self._openai

    @property
    def anthropic(self) -> Optional[str]:
        self._ensure_loaded()
        return self._anthropic

    @property
    def google(self) -> Optional[str]:
        self._ensure_loaded()
        return self._google

    @property
    def mistral(self) -> Optional[str]:
        self._ensure_loaded()
        return self._mistral

    @property
    def xai(self) -> Optional[str]:
        self._ensure_loaded()
        return self._xai

    @property
    def deepseek(self) -> Optional[str]:
        self._ensure_loaded()
        return self._deepseek

    @property
    def brave(self) -> Optional[str]:
        self._ensure_loaded()
        return self._brave

    @property
    def cohere(self) -> Optional[str]:
        self._ensure_loaded()
        return self._cohere

    @property
    def has_openai(self) -> bool:
        return bool(self.openai)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic)


# ── Module-level singleton ────────────────────────────────────────
keys = KeyProvider()
