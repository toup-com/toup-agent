"""Route-order invariants for the jobs activity feed (2026-07-08 prod bug).

Starlette matches routes in registration order, and ``/apps/jobs/events``
is shadowed by any ``/apps/jobs/{job_id}`` route registered before it:

- AGENT side: ``apps_router`` (defines ``GET /jobs/{job_id}``) was
  included in ``agent_main`` BEFORE ``jobs_events_router``, so the feed
  resolved as ``db.get(BuildJob, "events")`` → 404 "Job not found".
- PLATFORM side: ``apps_proxy`` had NO literal ``/jobs/events`` route at
  all — the request fell into its ``GET /jobs/{job_id}`` proxy, which
  also drops the query string (``limit``/``before`` pagination).

Net effect: the Mission Control activity feed silently ran on the legacy
client-side ``steps_json`` flatten (the "everything is Nokia Snake
Arcade" mis-attribution path PR 6 of the unified-jobs arc was built to
replace).

Pinned via source-grep, matching the established entrypoint-invariant
style (see ``test_api_no_html_invariant.py``) — importing the real apps
is heavyweight and the bug class is purely declaration ORDER.
"""

from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_AGENT_MAIN = (BACKEND / "agent_main.py").read_text()
_APPS_PROXY = (BACKEND / "app" / "api" / "apps_proxy.py").read_text()


def test_agent_registers_jobs_events_before_apps_router():
    """The literal ``/apps/jobs/events`` router must be included before
    ``apps_router`` (which defines ``GET /jobs/{job_id}``) or the param
    route captures ``events`` as a job_id."""
    events_include = _AGENT_MAIN.index(
        "from app.api.jobs_events import router as jobs_events_router"
    )
    apps_include = _AGENT_MAIN.index(
        "app.include_router(apps_router, prefix=settings.api_prefix)"
    )
    assert events_include < apps_include, (
        "agent_main must include jobs_events_router BEFORE apps_router — "
        "Starlette matches in registration order, so apps.py's "
        "GET /jobs/{job_id} otherwise shadows the literal /jobs/events "
        "path and the activity feed 404s."
    )


def test_platform_proxy_has_literal_jobs_events_route_before_param_route():
    """apps_proxy must define ``GET /jobs/events`` BEFORE
    ``GET /jobs/{job_id}`` for the same registration-order reason."""
    literal = _APPS_PROXY.index('@router.get("/jobs/events")')
    param = _APPS_PROXY.index('@router.get("/jobs/{job_id}")')
    assert literal < param, (
        "apps_proxy's literal /jobs/events route must be defined before "
        "the /jobs/{job_id} param route or it is shadowed and the feed "
        "falls back to the legacy client-side flatten."
    )


def test_platform_proxy_forwards_feed_query_string():
    """The feed's keyset pagination lives in query params; the generic
    ``_proxy`` call sites don't carry them. Pin the forwarding."""
    assert 'request.url.query' in _APPS_PROXY, (
        "jobs_events_proxy must forward the query string "
        "(limit/before) to the agent — _proxy paths drop it otherwise."
    )
