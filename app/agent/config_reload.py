"""
Config Hot-Reload — Update settings at runtime without restarting.

Supports reloading a subset of safe config fields from environment
variables or in-memory overrides. Unsafe fields (database_url,
jwt_secret, etc.) cannot be hot-reloaded.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Fields that are safe to reload at runtime
RELOADABLE_FIELDS: Set[str] = {
    "temperature", "max_tokens", "agent_model", "agent_fallback_model",
    "agent_max_tokens", "agent_max_tool_iterations", "thinking_budget_default",
    "thinking_model_override", "memory_recall_limit", "similarity_threshold",
    "max_history_messages", "auto_extract_memories", "tts_auto_mode",
    "tts_default_voice", "tts_model", "tts_speed", "enable_reranker",
    "heartbeat_enabled", "heartbeat_interval_hours", "sandbox_enabled",
    "telegram_reactions_enabled", "telegram_inline_buttons",
    "telegram_polls_enabled", "moderation_enabled", "dm_policy",
    "group_policy", "group_require_mention", "tool_timeout_default",
}

# Fields that must NOT be reloaded (security-sensitive)
FROZEN_FIELDS: Set[str] = {
    "database_url", "jwt_secret", "jwt_algorithm", "openai_api_key",
    "anthropic_api_key", "telegram_bot_token", "discord_bot_token",
    "slack_bot_token", "slack_app_token", "whatsapp_access_token",
    "whatsapp_app_secret", "brave_api_key", "cohere_api_key",
}


def reload_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Reload config fields at runtime.

    Args:
        overrides: Dict of field_name → new_value. If None, re-reads from env.

    Returns:
        Dict of field_name → status ("updated" | "frozen" | "unknown" | "error: ...")
    """
    from app.config import settings

    results: Dict[str, str] = {}

    if overrides is None:
        # Re-read from environment for all reloadable fields
        overrides = {}
        for field_name in RELOADABLE_FIELDS:
            env_key = field_name.upper()
            env_val = os.environ.get(env_key)
            if env_val is not None:
                overrides[field_name] = env_val

    for field_name, new_value in overrides.items():
        if field_name in FROZEN_FIELDS:
            results[field_name] = "frozen"
            logger.warning(f"[HOT-RELOAD] Blocked reload of frozen field: {field_name}")
            continue

        if not hasattr(settings, field_name):
            results[field_name] = "unknown"
            continue

        try:
            current = getattr(settings, field_name)
            # Type coercion
            if isinstance(current, bool):
                if isinstance(new_value, str):
                    new_value = new_value.lower() in ("true", "1", "yes", "on")
            elif isinstance(current, int):
                new_value = int(new_value)
            elif isinstance(current, float):
                new_value = float(new_value)

            object.__setattr__(settings, field_name, new_value)
            results[field_name] = "updated"
            logger.info(f"[HOT-RELOAD] {field_name}: {current!r} → {new_value!r}")
        except Exception as e:
            results[field_name] = f"error: {e}"
            logger.error(f"[HOT-RELOAD] Failed to set {field_name}: {e}")

    return results


def get_reloadable_fields() -> List[str]:
    """Return sorted list of fields that can be hot-reloaded."""
    return sorted(RELOADABLE_FIELDS)


def get_current_config(fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return current values of config fields."""
    from app.config import settings

    if fields is None:
        fields = sorted(RELOADABLE_FIELDS)

    result = {}
    for f in fields:
        if hasattr(settings, f):
            result[f] = getattr(settings, f)
    return result
