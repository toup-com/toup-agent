# Brain Modules

Self-contained feature modules separated from the Brain core.

## Structure

```
modules/
└── chat/           ← Chat & Sessions (REST + WebSocket)
    ├── router.py           POST /api/chat, /api/chat/stream
    ├── ws_router.py        WS   /ws/chat
    └── sessions_router.py  CRUD /api/sessions/*

Note: Workspace has been fully separated into its own top-level
service at /workspace/ — see workspace/README.md.
```

## Design Principles

1. **Absolute imports** — All modules use `app.*` imports, no path hacking.
2. **Single FastAPI process** — Modules are just routers mounted in `main.py`.
3. **Shared infrastructure** — DB models, auth, services remain in `app/`.
4. **Clean `__init__.py`** — Each module re-exports its routers for easy mounting.

## Future

Each module can be extracted to a standalone microservice by:
- Moving shared deps (`app.db`, `app.services`) into a shared library
- Running each module as its own FastAPI app
- Adding inter-service communication (gRPC, message queue)
