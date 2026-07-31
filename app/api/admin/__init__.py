"""
app.api.admin — Admin package

Exports:
  system_router   — memory decay/consolidation, bot dashboard, gateway control
  users_router    — user & invite management (closed beta)
  invite_router   — public invite validation & signup (no auth)
  search_usage_router — per-user search-gateway usage rollups
  set_bot_refs    — wire Telegram bot into the system dashboard
  require_admin   — FastAPI dependency for admin-only endpoints
"""

from app.api.admin.system import router as system_router, set_bot_refs
from app.api.admin.users import router as users_router, invite_router
from app.api.admin.infrastructure import router as infra_router
from app.api.admin.rollouts import router as rollouts_router
from app.api.admin.provider_apps import router as provider_apps_router
from app.api.admin.live_activity import router as live_activity_debug_router
from app.api.admin.search_usage import router as search_usage_router
from app.api.admin.deps import require_admin

# media_router (admin media-proxy metrics) is in-flight work for the
# Phase 1 media-proxy ticket and ships with that arc — not bundled
# here. Re-export when admin/media.py + the media_proxy.py changes
# are committed together.

__all__ = [
    "system_router",
    "users_router",
    "invite_router",
    "infra_router",
    "rollouts_router",
    "provider_apps_router",
    "live_activity_debug_router",
    "search_usage_router",
    "set_bot_refs",
    "require_admin",
]
