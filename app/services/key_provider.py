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

        self._openai = (settings.openai_api_key or "").strip() or None
        self._anthropic = (settings.anthropic_api_key or "").strip() or None
        self._google = (settings.google_api_key or "").strip() or None
        self._mistral = (settings.mistral_api_key or "").strip() or None
        self._xai = (settings.xai_api_key or "").strip() or None
        self._deepseek = (settings.deepseek_api_key or "").strip() or None
        self._brave = (settings.brave_api_key or "").strip() or None
        self._cohere = (settings.cohere_api_key or "").strip() or None
        self._loaded = True

        changed = (old_openai != self._openai) or (old_anthropic != self._anthropic)
        if changed:
            self._version += 1
            logger.info(
                "[KEY-PROVIDER] Keys refreshed (v%d): openai=%s anthropic=%s",
                self._version,
                "set" if self._openai else "none",
                "set" if self._anthropic else "none",
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
