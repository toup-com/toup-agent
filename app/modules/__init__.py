"""
HexBrain Modules — Separated feature modules.

Each module owns its own routers, schemas, and business logic.
The Brain core (memories, identities, embeddings, decay, consolidation)
remains in app/api/ and app/services/.

Modules:
  - chat:      WebSocket + REST chat, session management
"""
