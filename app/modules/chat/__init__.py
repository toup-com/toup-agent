"""
Chat Module — WebSocket + REST chat and session management.

Routers:
  - router:          POST /api/chat (REST chat with streaming support)
  - ws_router:       WS /api/ws/chat (WebSocket real-time chat)
  - sessions_router: /api/sessions/* (session CRUD + message history)
"""
from app.modules.chat.router import router as chat_router
from app.modules.chat.ws_router import router as ws_chat_router
from app.modules.chat.ws_router import set_ws_refs
from app.modules.chat.sessions_router import router as sessions_router

__all__ = ["chat_router", "ws_chat_router", "sessions_router", "set_ws_refs"]
