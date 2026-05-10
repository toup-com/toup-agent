"""
T1f' — Unified X-Agent-Key validation primitive.

Filed during T0c sign-off as a parked follow-up. Multiple call sites
re-implement variations of "look up X-Agent-Key → user_id":

  app/api/auth.py:62                   (HTTP /api/* via fallback)
  app/api/ws_chat.py:1083              (WS chat, header path)
  app/api/ws_chat.py:1093              (WS chat, query-param path)
  app/api/ws_browser.py:2305           (WS browser proxy)
  app/api/files.py:132                 (file proxy auth)
  app/mcp_auth.py                      (MCP transport — T0c)

Three behaviour shapes across them:

  1. **Tenant-DB lookup** (mcp_auth.py): query
     `agent_configs.agent_api_key == <key>` → user_id. Multi-tenant
     correct.
  2. **Local-settings compare** (auth.py / ws_chat / ws_browser /
     files): compare against `settings.agent_api_key` and bind
     `settings.user_id`. Single-tenant correct, multi-tenant wrong
     (returns the same user for any matching key).
  3. **Mixed** (a few sites do both depending on config).

This module exposes ONE async function — `resolve_user_from_agent_key`
— that does the right thing in both cases. Existing call sites
should migrate to it ONE AT A TIME, with smoke tests for each, in a
follow-up PR per the architecture doc.

WHY NOT ship the migration in this arc: each call site has subtle
differences (lazy user creation in auth.py:78, telegram bot context
in ws_chat.py:1037, file proxy in files.py:74) that need careful
case analysis. Bundling the unification with the connector arc
would couple two unrelated changes — risk per blast-radius rule.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


async def resolve_user_from_agent_key(agent_key: str) -> Optional[str]:
    """Return the user_id for this `X-Agent-Key`, or None if it doesn't
    match any tenant.

    Resolution order:
      1. **Tenant DB lookup** — query `agent_configs.agent_api_key`.
         Returns the matched user_id. This is the multi-tenant
         platform path.
      2. **Local settings fallback** — if step 1 misses AND
         `settings.agent_api_key` is set AND it matches AND
         `settings.user_id` is set, return that. Covers single-tenant
         agent containers where there's no `agent_configs` row for
         the local agent itself.

    Empty/whitespace key → None (don't burn a DB query on obviously
    bogus input).
    """
    if not agent_key or not agent_key.strip():
        return None

    # Step 1: tenant DB lookup. Reuse the MCP middleware's helper so
    # the two paths can never diverge.
    from app.mcp_auth import _resolve_agent_key_to_user_id
    try:
        user_id = await _resolve_agent_key_to_user_id(agent_key)
        if user_id:
            return user_id
    except Exception as e:
        # Don't fail-closed here — the local-settings fallback may
        # still let single-tenant agent containers proceed. Log
        # loudly so a sustained DB issue is visible.
        logger.warning(
            "[agent_key] tenant-DB lookup raised: %s — trying local fallback",
            e,
        )

    # Step 2: local settings fallback (single-tenant agent containers).
    if (
        settings.agent_api_key
        and agent_key == settings.agent_api_key
        and settings.user_id
    ):
        return settings.user_id

    return None
