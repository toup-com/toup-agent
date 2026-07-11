"""Autopilot enablement gate (staged rollout).

Autopilot activates for a tenant when EITHER:
- ``AUTOPILOT_ENABLED=true`` (image-global master switch, the routines
  pattern — flips the feature for everyone), OR
- the container's bound ``user_id`` is in ``AUTOPILOT_USER_ALLOWLIST``
  (comma-separated full user ids, baked image-wide) — the staged
  path: admin/canary tenants test the full loop in production while
  everyone else stays off.

Pool containers evaluate correctly post-bind because ``settings.user_id``
is populated by the bind payload; pre-bind lobby containers have no
user_id and stay off.
"""
from __future__ import annotations


def autopilot_enabled_for(settings) -> bool:
    if bool(getattr(settings, "autopilot_enabled", False)):
        return True
    allowlist = (getattr(settings, "autopilot_user_allowlist", "") or "").strip()
    if not allowlist:
        return False
    user_id = (getattr(settings, "user_id", "") or "").strip()
    if not user_id:
        return False
    allowed = {u.strip() for u in allowlist.split(",") if u.strip()}
    return user_id in allowed
