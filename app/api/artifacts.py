"""Agent-side HTML-artifact routes — raw bytes out of the tenant container.

Mounted in ``agent_main`` behind the global ``X-Agent-Key`` middleware, so
these handlers carry no auth of their own. The only caller is the platform's
``artifact_proxy``, which adds the browser-facing security headers.

This half deliberately does NOT set a CSP: the sandbox contract belongs to
the origin the browser actually loads, and duplicating it here would mean two
places to keep in sync and one of them silently wrong.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, Response

from app.agent.skills.builtins.app_html import appskill, backfill, logo, runtime, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])

#: **There is no route here that returns an app's brief, and there must never
#: be one.** The brief is the model's working memory of an app — its stated
#: purpose, its audience, the reasoning behind its design — and it is written
#: in the second person, to a model. Every route in this module is reachable
#: from the platform, and the platform's artifact routes are reachable from a
#: signed-in client, so a route that served it would be one refactor away from
#: rendering it in somebody's Files list. The only reader is the skill, in
#: process, through `appskill.read`. See `appskill`'s module docstring for the
#: other three layers that keep it out of the library.


@router.get("/")
async def list_artifacts() -> Dict[str, Any]:
    """Every single-file app that is really there, newest update first.

    Reconciled against the disk BEFORE it is serialised. This route reports the
    manifest, and the manifest is an index — a record whose ``.html`` went away
    was listed anyway, with its last known size, as a row the user could tap and
    never open. `store.reconcile_all` restores it from ``.versions/`` where it
    can and drops the row where it cannot, so a broken entry repairs itself the
    first time anyone opens Files rather than needing a fleet-wide migration.

    ``size_bytes`` is then re-read from the FILE rather than trusted from the
    record: a size is a fact about bytes on disk, and reporting a remembered one
    is how a row comes to look healthy while being empty.
    """
    repaired = store.reconcile_all()
    if repaired:
        logger.info("[app_html] list repaired %s", repaired)
    # Round 20's two backfills, in the same place and for the same reason:
    # every tenant opens Files, so a repair that hangs off this route reaches
    # the whole fleet with no migration to write and nothing to sweep.
    #
    # Two speeds. The derived brief is string work and runs inline, so the
    # first list after an upgrade already leaves every app with a structure
    # map. Writing the NARRATIVE and drawing the icon takes a model call per
    # app, so it goes to the background — nobody should wait for twenty-two
    # of those to open a list — and runs at most once per container.
    repaired.update(appskill.backfill_missing())
    backfill.schedule(_user_id())

    records = store.read_manifest()
    items: List[Dict[str, Any]] = []
    for slug, rec in records.items():
        d = rec.to_dict()
        d["exists"] = store.exists(slug)
        d["versions"] = len(store.list_versions(slug))
        # The card needs to know an icon is worth fetching; it never needs the
        # icon inline, which would put an SVG per app into every list. A stat,
        # not a read — see `logo.has_icon`.
        d["has_icon"] = logo.has_icon(slug)
        try:
            d["size_bytes"] = os.path.getsize(store.app_path(slug))
        except (OSError, store.AppStoreError):
            d["size_bytes"] = 0
        items.append(d)
    items.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
    return {"apps": items, "root": store.apps_root(), "count": len(items),
            "repaired": repaired}


def _user_id() -> str:
    """This container's tenant, for the backfill's LLM accounting.

    Empty on a pool container that has not been bound yet, which is fine:
    bundle mode authenticates with ``TOUP_TOKEN``, not with this, and the
    field is only used for the usage row.
    """
    try:
        from app.config import settings
        return str(getattr(settings, "user_id", "") or "")
    except Exception:  # pragma: no cover
        return ""


@router.get("/{slug}")
async def get_artifact(slug: str) -> Response:
    """Return one app's HTML, wrapped for the sandbox it is about to run in.

    ``text/html; charset=utf-8`` explicitly: WKWebView decides encoding from
    the HTTP header before it parses ``<meta charset>``, and defaults to
    ASCII when the header omits it — which is how emoji turn into mojibake
    (the same trap the Expo preview proxy documents).

    :func:`runtime.wrap_for_runtime` is applied HERE and not at write time.
    The file on disk stays byte-for-byte what the model wrote, so
    ``view_app_file`` shows the truth, ``edit_app_file``'s exact-string match
    still lands, and the size on the card is the app's own size. What the
    browser gets additionally is a storage shim that cannot throw, error hooks
    that report the real message instead of ``"Script error."``, and
    ``crossorigin`` on the cdnjs tags that made that masking possible.
    """
    try:
        slug = store.normalise_slug(slug)
        html = store.read_app(slug)
    except store.AppStoreError as exc:
        raise HTTPException(404, str(exc))

    rec = store.read_manifest().get(slug)
    return Response(
        content=runtime.wrap_for_runtime(html).encode("utf-8"),
        media_type="text/html; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "X-Toup-Artifact-Revision": str(rec.revision if rec else 1),
            "X-Toup-Artifact-Title": (rec.title if rec else slug).encode(
                "ascii", "replace").decode("ascii"),
        },
    )


@router.get("/{slug}/icon")
async def get_artifact_icon(slug: str, request: Request) -> Response:
    """The app's designed mark, as SVG.

    **Never 404s for an app that exists.** If the designed icon has not been
    drawn yet — a fresh volume, a container that could not reach a model, an
    app mid-backfill — the deterministic monogram is generated and stored on
    the spot. A card rendering a broken image is worse than a card rendering
    a plain one, and "the icon arrives later" is not something a list view can
    express.

    ``image/svg+xml`` with ``X-Content-Type-Options: nosniff``: the platform
    serves this onward from the artifact origin, and the contents were drawn
    by a model. `logo.sanitize_svg` has already refused script, event
    attributes and external references — this header is the second lock, so
    the file cannot be talked into being interpreted as anything else.
    """
    try:
        slug = store.normalise_slug(slug)
    except store.AppStoreError as exc:
        raise HTTPException(400, str(exc))
    rec = store.read_manifest().get(slug)
    if rec is None:
        raise HTTPException(404, f"no artifact named {slug!r}")

    svg = logo.read_icon(slug)
    source = logo.read_sidecar(slug).get("source", "model" if svg else "")
    if not svg:
        # Synchronous, deterministic, no model: this route is on the path of
        # a list view and must never wait on a network call.
        svg, source = await logo.ensure_icon(
            slug, title=rec.title or slug, allow_model=False,
        )

    body = svg.encode("utf-8")
    etag = '"%s"' % hashlib.sha256(body).hexdigest()[:32]
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag,
                                                  "Cache-Control": "no-cache"})
    return Response(
        content=body,
        media_type="image/svg+xml",
        headers={
            # `no-cache` (revalidate), not `no-store`: an icon changes only
            # when the app changes what it is, so most requests should cost a
            # 304 rather than 6 KB — and a rename must still be visible at
            # once, which `no-store` and `max-age` cannot both give.
            "Cache-Control": "no-cache",
            "ETag": etag,
            "X-Content-Type-Options": "nosniff",
            "X-Toup-Icon-Source": source or "unknown",
        },
    )


@router.get("/{slug}/source")
async def get_artifact_source(slug: str) -> Dict[str, Any]:
    """The app's handle AND its body, exactly as the model wrote it.

    Round 19. A client that has been told "revision 4 is live" needs somewhere
    to go and get revision 4, and neither existing route answers that:
    ``/{slug}`` returns the RUNTIME-WRAPPED document (right for a browser
    frame, wrong for a client that applies its own sandbox wrapper — it would
    install two storage shims over each other), and ``/{slug}/meta`` returns
    the numbers without the file.

    So this returns the unwrapped bytes beside the revision they belong to, in
    one request. ``revision`` and ``html`` come from the same read, which is
    the property that matters: a client cannot cache a body under a revision
    that was current at a different moment.
    """
    try:
        slug = store.normalise_slug(slug)
        html = store.read_app(slug)
    except store.AppStoreError as exc:
        raise HTTPException(404, str(exc))
    rec = store.read_manifest().get(slug)
    return {
        "slug": slug,
        "title": (rec.title if rec else slug) or slug,
        "revision": rec.revision if rec else 1,
        "updated_at": (rec.updated_at if rec else None) or None,
        "size_bytes": len(html.encode("utf-8")),
        "html": html,
    }


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


@router.patch("/{slug}")
async def rename_artifact(slug: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Change an app's display name. Nothing else moves.

    Deliberately NOT a re-write: no new revision, no snapshot, no touch of the
    .html. The slug stays the identity that every chat card and every open
    runner is keyed on, so a rename cannot orphan a card the way a slug change
    would.
    """
    try:
        slug = store.normalise_slug(slug)
        rec = store.retitle_record(slug, str(body.get("title") or ""))
    except store.AppStoreError as exc:
        raise HTTPException(400, str(exc))
    if rec is None:
        raise HTTPException(404, f"no artifact named {slug!r}")
    return rec.to_dict()


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
