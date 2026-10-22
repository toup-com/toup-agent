"""Per-tenant supervisor / executor sub-agent settings (no migration).

One ``platform_settings`` row per user, key ``subagent.settings.<user_id>``,
value a JSON object. Written by ``PUT /api/admin/users/{id}/subagent-settings``
and read by ``GET /api/agent/runtime-flags`` (the tenant's 60 s cache, see
``agent/supervisor_gate.py``) and by the voice relay for a delegation's
``supervisor_model``. Platform-side: ``platform_settings`` lives in the
platform DB.

Every reader goes through :func:`normalize`, so a hand-edited or older row
can only ever yield the known keys with their documented types — an unknown
key is dropped, a wrong-typed one is dropped, never coerced into something
the admin did not write.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KEY_PREFIX = "subagent.settings."

#: field -> (type, min, max). Bounds are sanity bounds; the tenant still
#: clamps every cap to its own image ceiling (``min(override, settings)``).
_INT_BOUNDS: Dict[str, tuple] = {
    "max_workers": (1, 6),
    "max_children_per_parent": (1, 20),
    "max_per_user_concurrent": (1, 10),
    "worker_timeout_seconds": (30, 900),
}
_STR_FIELDS = ("executor_model", "supervisor_model")
FIELDS = ("enabled",) + _STR_FIELDS + tuple(_INT_BOUNDS)


def setting_key(user_id: str) -> str:
    return f"{KEY_PREFIX}{user_id}"


def normalize(raw: Any) -> Dict[str, Any]:
    """The known keys of ``raw`` with valid values; everything else dropped."""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    if isinstance(raw.get("enabled"), bool):
        out["enabled"] = raw["enabled"]
    for k in _STR_FIELDS:
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()[:80]
    for k, (lo, hi) in _INT_BOUNDS.items():
        v = raw.get(k)
        if isinstance(v, bool) or not isinstance(v, int):
            continue
        if lo <= v <= hi:
            out[k] = v
    return out


async def load(db: Any, user_id: str) -> Dict[str, Any]:
    """The tenant's override object, ``{}`` when absent or unreadable."""
    if not user_id:
        return {}
    try:
        from app.db.models.platform import PlatformSetting
        row = await db.get(PlatformSetting, setting_key(user_id))
    except Exception:  # noqa: BLE001 — a missing table is "no override"
        logger.debug("[subagent_settings] read failed", exc_info=True)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return {}
    if row is None or not row.value:
        return {}
    try:
        return normalize(json.loads(row.value))
    except (ValueError, TypeError):
        logger.warning("[subagent_settings] malformed row for %s", user_id[:8])
        return {}


async def save(
    db: Any, user_id: str, values: Dict[str, Any],
    *, admin_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Replace the row with ``normalize(values)``; returns what was stored.
    Caller commits."""
    from app.db.models.platform import PlatformSetting

    clean = normalize(values)
    key = setting_key(user_id)
    row = await db.get(PlatformSetting, key)
    blob = json.dumps(clean, sort_keys=True)
    if row is None:
        db.add(PlatformSetting(key=key, value=blob, updated_by_user_id=admin_id))
    else:
        row.value = blob
        row.updated_by_user_id = admin_id
    return clean
