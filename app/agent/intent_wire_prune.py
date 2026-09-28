"""Conservative, first-call-only tool schema pruning for an exact-user canary.

The runner has already computed the callable names through the existing intent,
channel, surface, and connector gates. This module only chooses which of those
definitions to serialize; it does not classify a message or grant capabilities.
"""

import re
from typing import Any, Dict, List, Optional, Sequence


_CATEGORIES = frozenset({"greeting", "question"})
_CHANNELS = frozenset({"web", "mobile"})
_MAX_FIRST_CALL_TOOLS = 16
_STANDALONE_GREETING = re.compile(r"\s*(?:hi|hello|hey)[!?.,;:]*\s*", re.IGNORECASE)


def select_first_call_tools(
    stable_tools: Sequence[Dict[str, Any]],
    allowed_names: Sequence[str],
    *,
    canary_enabled: bool,
    question_canary_enabled: bool = False,
    channel: Optional[str],
    intent_category: str,
    message_text: str,
    main_chat: bool,
    has_attachment: bool,
    has_job: bool,
) -> Optional[List[Dict[str, Any]]]:
    """Return exact allowed definitions in stable order, or None for legacy.

    Unknown/empty or duplicate names fail closed to the existing stable array.
    This limit also ensures the proxy's 128-tool cap cannot prune a named tool.
    """
    if (
        not canary_enabled
        or channel not in _CHANNELS
        or intent_category not in _CATEGORIES
        or (intent_category == "question" and not question_canary_enabled)
        or not main_chat
        or has_attachment
        or has_job
        # The classifier's greeting-prefix shortcut can classify a mixed
        # "Hi schedule..." as greeting before checking action words. A word
        # count also admits "Hi,schedule"; require a whole standalone match.
        or (intent_category == "greeting" and not _STANDALONE_GREETING.fullmatch(message_text or ""))
    ):
        return None
    names = set(allowed_names)
    if not names or len(names) != len(allowed_names) or len(names) > _MAX_FIRST_CALL_TOOLS:
        return None

    selected: List[Dict[str, Any]] = []
    seen = set()
    for tool in stable_tools:
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if name in names and name not in seen:
            selected.append(tool)
            seen.add(name)
    if seen != names or len(selected) >= len(stable_tools):
        return None
    return selected


def pruned_cache_key(user_id: str, intent_category: str) -> str:
    """Route the smaller tools-first prefix away from the legacy full head."""
    return f"{user_id}:iwp:{intent_category}"
