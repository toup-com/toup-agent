"""Supervisor (`delegate`) enablement gate + the per-tenant override cache.

The tool is visible when, in order:

1. the tenant's override row says ``enabled`` (either way) — admin
   ``PUT /api/admin/users/{id}/subagent-settings``, read through
   ``GET /api/agent/runtime-flags`` and cached here for 60 s, so a flip takes
   effect within a minute with no container restart; else
2. ``SUPERVISOR_ENABLED=true`` (image-wide), or the bound ``user_id`` is in
   ``SUPERVISOR_USER_ALLOWLIST`` (the Autopilot pattern,
   ``autopilot_gate.py``).

and only on ATTENDED surfaces: a supervisor turn holds its caller for up to
``supervisor_timeout_seconds`` while it writes the merged answer, which only
makes sense where someone is waiting for that answer. Background channels
(trigger, routine, automation thread, autopilot, sub-agent) never see it.

The override read is per-process (one tenant per container) and fails to
``{}`` — "no override" — so an unreachable platform leaves the image gate in
charge rather than switching a feature off or on by accident.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SUPERVISOR_TOOL = "delegate"

#: Surfaces with a person waiting on the turn's own answer.
ATTENDED_CHANNELS = frozenset({
    "web", "app", "mobile", "voice", "desktop", "telegram", "whatsapp",
})

OVERRIDES_TTL_S = 60.0

# user_id -> (fetched_at_monotonic, overrides). Per process: one tenant.
_cache: Dict[str, tuple] = {}
_refreshing: set = set()


def _image_gate(settings: Any, user_id: Optional[str] = None) -> bool:
    if bool(getattr(settings, "supervisor_enabled", False)):
        return True
    allowlist = (getattr(settings, "supervisor_user_allowlist", "") or "").strip()
    user_id = (user_id or getattr(settings, "user_id", "") or "").strip()
    if not allowlist or not user_id:
        return False
    return user_id in {u.strip() for u in allowlist.split(",") if u.strip()}


def supervisor_enabled_for(
    settings: Any, overrides: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> bool:
    """The gate for one user. ``user_id`` names the user explicitly for a
    process that serves many (the platform relay); a tenant leaves it unset
    and the gate reads its own bound ``settings.user_id``."""
    enabled = (overrides or {}).get("enabled")
    if isinstance(enabled, bool):
        return enabled
    return _image_gate(settings, user_id)


def cached_overrides(user_id: str) -> Optional[Dict[str, Any]]:
    """The last fetched override object (any age), or None if never fetched."""
    hit = _cache.get(user_id or "")
    return dict(hit[1]) if hit else None


def _fresh(user_id: str) -> bool:
    hit = _cache.get(user_id or "")
    return bool(hit) and (time.monotonic() - hit[0]) < OVERRIDES_TTL_S


def _overrides_from_flags(flags: Dict[str, Any]) -> Dict[str, Any]:
    """`runtime-flags` additive fields → the override object's own keys."""
    from app.services.subagent_settings import normalize
    return normalize({
        "enabled": flags.get("supervisor_enabled"),
        "executor_model": flags.get("executor_model"),
        "supervisor_model": flags.get("supervisor_model"),
        "max_workers": flags.get("supervisor_max_workers"),
        "max_children_per_parent": flags.get("max_children_per_parent"),
        "max_per_user_concurrent": flags.get("max_per_user_concurrent"),
        "worker_timeout_seconds": flags.get("worker_timeout_seconds"),
    })


async def _fetch_flags(user_id: str) -> Optional[Dict[str, Any]]:
    """GET runtime-flags, same both-URL-variants rule as
    ``tool_executor._read_subagent_flag_for_user``. None on any failure."""
    from app.config import settings
    import httpx

    platform_url = (getattr(settings, "platform_api_url", "") or "").rstrip("/")
    agent_key = (getattr(settings, "agent_api_key", "") or "").strip()
    if not platform_url or not agent_key or not user_id:
        return None
    candidates = [f"{platform_url}/agent/runtime-flags"]
    if not platform_url.endswith("/api"):
        candidates.append(f"{platform_url}/api/agent/runtime-flags")
    async with httpx.AsyncClient(timeout=3.0) as c:
        for url in candidates:
            try:
                r = await c.get(url, headers={
                    "X-Agent-Key": agent_key, "X-Agent-User-Id": user_id,
                })
            except Exception:  # noqa: BLE001
                continue
            if r.status_code != 200:
                continue
            if "application/json" not in (r.headers.get("content-type") or "").lower():
                continue
            try:
                data = r.json()
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


async def get_overrides(user_id: str, *, force: bool = False) -> Dict[str, Any]:
    """The tenant's overrides, fetched at most once per ``OVERRIDES_TTL_S``.

    A FAILED fetch keeps the last override this process was told and only
    restamps it for the same window (no stampede on an unreachable
    platform): an admin `enabled: false` is the per-tenant kill switch, and
    erasing it to ``{}`` on a 3 s blip fell back to the image allowlist and
    put `delegate` back in front of the model (TW-7). ``{}`` only when
    nothing was ever fetched. A fetch that ANSWERS replaces the override,
    an empty answer included."""
    if not user_id:
        return {}
    if not force and _fresh(user_id):
        return cached_overrides(user_id) or {}
    try:
        flags = await _fetch_flags(user_id)
    except Exception:  # noqa: BLE001
        flags = None
    if flags is None:
        ov = cached_overrides(user_id) or {}
    else:
        ov = _overrides_from_flags(flags) if flags else {}
    _cache[user_id] = (time.monotonic(), ov)
    return dict(ov)


def _kick_refresh(user_id: str) -> None:
    if not user_id or _fresh(user_id) or user_id in _refreshing:
        return
    try:
        from app.services.background_tasks import spawn as _spawn_bg

        async def _go() -> None:
            try:
                await get_overrides(user_id, force=True)
            finally:
                _refreshing.discard(user_id)

        _refreshing.add(user_id)
        _spawn_bg(_go(), name="supervisor-overrides")
    except RuntimeError:  # no running loop (sync caller outside a turn)
        _refreshing.discard(user_id)


def supervisor_disabled_tools(channel: Optional[str], user_id: Optional[str]) -> frozenset:
    """``{"delegate"}`` when this turn may NOT see the tool, else empty.

    Synchronous on purpose — it is read on the turn's hot path. It decides on
    the CACHED override (any age) and refreshes a stale one in the background,
    so an admin flip is in force on the turn after the next fetch.
    """
    from app.config import settings

    if (channel or "").strip().lower() not in ATTENDED_CHANNELS:
        return frozenset({SUPERVISOR_TOOL})
    uid = (user_id or getattr(settings, "user_id", "") or "").strip()
    ov = cached_overrides(uid)
    _kick_refresh(uid)
    if supervisor_enabled_for(settings, ov):
        return frozenset()
    return frozenset({SUPERVISOR_TOOL})


def reset_for_test() -> None:
    _cache.clear()
    _refreshing.clear()
