"""Tests for the sentinel-image guard + backfill (2026-05-30).

Incident: a brand-new user (hosseininariman@gmail.com, created 18:03) was
provisioned with image_tag="toup-agent:latest", container_id=None — the
"sentinel" image the rollout pipeline never publishes — and couldn't chat
("the bridge bind never succeeded"). Two distinct sites stamp the broken
state:

  1. pool_service.claim_for_user used to write `settings.docker_agent_image`
     (the sentinel) to ManagedContainer.image_tag regardless of the real
     image the pool container was running. Fixed to use the last-known-good
     rollout SHA.
  2. provision_container's image-tag fallback chain ended at the sentinel
     when no rollout had completed. Fixed to REFUSE (HTTPException 503)
     rather than silently provision a non-deployable tag.

A boot-time backfill (`backfill_sentinel_image_containers`) cures any
existing stragglers — same pattern as the OpenAI project backfill from
PR #149. Source-level tests pin the guards so the next refactor can't
silently regress.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ENVIRONMENT", "test")


BACKEND_DIR = Path(__file__).resolve().parent.parent
_POOL_SRC = (BACKEND_DIR / "app/services/pool_service.py").read_text(encoding="utf-8")
_DHS_SRC = (BACKEND_DIR / "app/services/docker_host_service.py").read_text(encoding="utf-8")
_PLATFORM_MAIN_SRC = (BACKEND_DIR / "platform_main.py").read_text(encoding="utf-8")


def test_pool_service_stamps_real_image_not_sentinel():
    """claim_for_user must NOT write settings.docker_agent_image into
    ManagedContainer.image_tag (that's the sentinel, the entire root
    cause). It must call _latest_known_good_image_tag and only fall back
    to the sentinel as a LAST resort with a loud WARN marker."""
    # The line `container.image_tag = settings.docker_agent_image` was the
    # bug. It must be gone OR gated behind an explicit fallback log.
    # Conservatively: the file must reference the real-image helper near
    # the image_tag assignment.
    assert "_latest_known_good_image_tag" in _POOL_SRC, (
        "pool_service must call _latest_known_good_image_tag to get the "
        "real image instead of stamping the sentinel"
    )
    # The greppable warn marker that fires on the rare fallback path.
    assert "[POOL-IMAGE-MISS]" in _POOL_SRC, (
        "missing the [POOL-IMAGE-MISS] WARN marker — the rare fallback "
        "path must be visible/alertable, not silent"
    )


def test_provision_container_refuses_sentinel_image():
    """provision_container must REFUSE to ship a `:latest` sentinel —
    silently writing it (pre-fix behavior) produced container_id=None
    rows that look healthy in the DB but can't actually chat."""
    assert "[CONTAINER-SENTINEL-REFUSE]" in _DHS_SRC, (
        "provision_container must log a distinctive marker when refusing "
        "the sentinel image"
    )
    assert 'image_tag.endswith(":latest")' in _DHS_SRC, (
        "provision_container must check for the :latest sentinel suffix"
    )
    # And it should raise rather than silently provision — look for a 503
    # HTTPException near the sentinel check.
    assert "HTTPException(\n                503" in _DHS_SRC or "HTTPException(503" in _DHS_SRC, (
        "the sentinel guard must raise an HTTPException, not fall through"
    )


def test_container_backfill_function_exists():
    """The reconciler that cures any pre-existing sentinel rows."""
    from app.services import docker_host_service as svc
    assert hasattr(svc, "backfill_sentinel_image_containers"), (
        "backfill_sentinel_image_containers must exist as the boot-time "
        "+ admin remediation entry point"
    )


def test_backfill_predicate_covers_both_broken_shapes():
    """The backfill query must match BOTH broken shapes — sentinel image
    AND null container_id — because the pool fast-path produced rows
    with both characteristics."""
    assert "image_tag.endswith" in _DHS_SRC and ".container_id.is_(None)" in _DHS_SRC, (
        "backfill query must cover image_tag.endswith(':latest') OR "
        "container_id IS NULL"
    )


def test_boot_runs_container_backfill():
    """platform_main must invoke the container backfill on startup so a
    new deploy automatically cures any stragglers."""
    assert "backfill_sentinel_image_containers" in _PLATFORM_MAIN_SRC, (
        "platform_main.py must call backfill_sentinel_image_containers on boot"
    )
    # And the OpenAI backfill must still be wired (PR #149) — guard
    # against an accidental removal during this PR's edit.
    assert "backfill_missing_openai_projects" in _PLATFORM_MAIN_SRC
