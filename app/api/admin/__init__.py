"""
app.api.admin — Admin package

Exports:
  system_router   — memory decay/consolidation, bot dashboard, gateway control
  users_router    — user & invite management (closed beta)
  invite_router   — public invite validation & signup (no auth)
  set_bot_refs    — wire Telegram bot into the system dashboard
  require_admin   — FastAPI dependency for admin-only endpoints
"""

from app.api.admin.system import router as system_router, set_bot_refs
from app.api.admin.users import router as users_router, invite_router
from app.api.admin.deps import require_admin

__all__ = [
    "system_router",
    "users_router",
    "invite_router",
    "set_bot_refs",
    "require_admin",
]
