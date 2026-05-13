"""Invariant #2 (bug sweep 2026-05-13): no API route returns HTML.

Ticket 6 was the symptom: `/api/triggers/*` returned the SPA's
index.html because no router was mounted at that prefix on
`platform_main`. The frontend tried to parse HTML as JSON and blew
up with `Unexpected token '<', "<!DOCTYPE "...`.

This test enforces the floor: the SPA catch-all in `platform_main`
MUST explicitly reject `/api/*` paths with a JSON 404 instead of
falling through to index.html. Pinned via source-grep so a future
"simplification" can't quietly drop the guard.
"""

from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_PLATFORM_MAIN = (BACKEND / "platform_main.py").read_text()


def test_spa_catchall_rejects_api_paths_as_json():
    """The SPA catch-all must short-circuit /api/* paths to a JSON
    404 before the index.html fallback. Pin both the path check and
    the JSONResponse — the bug class is "API caller gets HTML back."
    """
    assert 'path.startswith("api/")' in _PLATFORM_MAIN, (
        "platform_main's SPA catch-all must check `path.startswith"
        "(\"api/\")` to prevent /api/* from falling through to "
        "index.html. Without this, a missing API route returns HTML "
        "and the frontend parsing dies with 'Unexpected token <'."
    )
    assert "JSONResponse" in _PLATFORM_MAIN, (
        "platform_main must import JSONResponse for the /api/* 404 "
        "path — the catch-all MUST return JSON, never HTML."
    )


def test_triggers_proxy_router_mounted():
    """Ticket 6 — the original fix. triggers_proxy router must be
    mounted on platform_main so `/api/triggers/*` resolves to the
    proxy and forwards to the tenant agent. Pinned via source-grep so
    a future refactor can't silently drop the mount and re-introduce
    the HTML-instead-of-JSON symptom.
    """
    assert "from app.api.triggers_proxy import router as triggers_proxy_router" in _PLATFORM_MAIN, (
        "platform_main must import triggers_proxy_router."
    )
    assert "triggers_proxy_router, prefix=settings.api_prefix" in _PLATFORM_MAIN, (
        "platform_main must mount triggers_proxy_router at the api prefix."
    )
