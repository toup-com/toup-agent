"""Agent-side HTML-artifact routes — raw bytes out of the tenant container.

Mounted in ``agent_main`` behind the global ``X-Agent-Key`` middleware, so
these handlers carry no auth of their own. The only caller is the platform's
``artifact_proxy``, which adds the browser-facing security headers.

This half deliberately does NOT set a CSP: the sandbox contract belongs to
the origin the browser actually loads, and duplicating it here would mean two
places to keep in sync and one of them silently wrong.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Response

from app.agent.skills.builtins.app_html import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


@router.get("/")
async def list_artifacts() -> Dict[str, Any]:
    """Every registered single-file app, newest update first."""
    records = store.read_manifest()
    items: List[Dict[str, Any]] = []
    for slug, rec in records.items():
        d = rec.to_dict()
        d["exists"] = store.exists(slug)
        d["versions"] = len(store.list_versions(slug))
        items.append(d)
    items.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
    return {"apps": items, "root": store.apps_root(), "count": len(items)}


@router.get("/{slug}")
async def get_artifact(slug: str) -> Response:
    """Return one app's HTML verbatim.

    ``text/html; charset=utf-8`` explicitly: WKWebView decides encoding from
    the HTTP header before it parses ``<meta charset>``, and defaults to
    ASCII when the header omits it — which is how emoji turn into mojibake
    (the same trap the Expo preview proxy documents).
    """
    try:
        slug = store.normalise_slug(slug)
        html = store.read_app(slug)
    except store.AppStoreError as exc:
        raise HTTPException(404, str(exc))

    rec = store.read_manifest().get(slug)
    return Response(
        content=html.encode("utf-8"),
        media_type="text/html; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "X-Toup-Artifact-Revision": str(rec.revision if rec else 1),
            "X-Toup-Artifact-Title": (rec.title if rec else slug).encode(
                "ascii", "replace").decode("ascii"),
        },
    )


@router.get("/{slug}/meta")
async def get_artifact_meta(slug: str) -> Dict[str, Any]:
    try:
        slug = store.normalise_slug(slug)
    except store.AppStoreError as exc:
        raise HTTPException(400, str(exc))
    rec = store.read_manifest().get(slug)
    if rec is None or not store.exists(slug):
        raise HTTPException(404, f"no artifact named {slug!r}")
    d = rec.to_dict()
    d["path"] = store.app_path(slug)
    d["versions"] = store.list_versions(slug)
    return d


@router.get("/{slug}/state")
async def get_artifact_state(slug: str) -> Dict[str, Any]:
    """The app's persisted state blob (see store.read_state)."""
    try:
        slug = store.normalise_slug(slug)
    except store.AppStoreError as exc:
        raise HTTPException(400, str(exc))
    return {"slug": slug, "state": store.read_state(slug)}


@router.put("/{slug}/state")
async def put_artifact_state(slug: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Merge keys into the app's state. ``{"updates": {"k": null}}`` deletes.

    Merge rather than replace: two tabs of the same app would otherwise
    clobber each other's keys on every write.
    """
    try:
        slug = store.normalise_slug(slug)
        updates = body.get("updates")
        if updates is None:
            updates = body.get("state")
        written = store.merge_state(slug, updates or {})
    except store.AppStoreError as exc:
        raise HTTPException(400, str(exc))
    return {"slug": slug, "bytes": written, "state": store.read_state(slug)}


@router.delete("/{slug}")
async def delete_artifact(slug: str) -> Dict[str, Any]:
    try:
        slug = store.normalise_slug(slug)
    except store.AppStoreError as exc:
        raise HTTPException(400, str(exc))
    freed = 0
    try:
        freed = os.path.getsize(store.app_path(slug))
    except OSError:
        pass
    removed = store.delete_app(slug)
    return {"deleted": removed, "slug": slug, "bytes_freed": freed}
