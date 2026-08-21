"""Round 12 — the storage bridge.

A single-file app runs in a frame sandboxed WITHOUT ``allow-same-origin``, so
its origin is opaque and ``localStorage`` / ``sessionStorage`` / ``indexedDB``
/ ``document.cookie`` all **throw**, while every network call is blocked by
``connect-src`` against that opaque origin. An app therefore cannot persist
anything by itself.

The bridge is how it persists anyway:

    app  --postMessage-->  Toup shell  --HTTP + account bearer-->  agent  --> disk

Three properties, tested here end to end:

  1. **It actually persists.** Written through the HTTP route, readable back
     through the HTTP route, and present on disk — not memoized in a process
     that is about to be recreated by the next blue-green upgrade.
  2. **It merges rather than replaces**, so two open tabs of the same app do
     not clobber each other's keys.
  3. **The two halves of the protocol agree.** The host implementation
     (AppArtifactFrame.tsx) and the client snippet the model is told to inline
     (skills/toup-frontend-design.md) are written in different files by
     different hands; a rename in one that is not made in the other produces
     an app that silently never persists, which is invisible until a user
     reloads.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.skills.builtins.app_html import store

REPO = pathlib.Path(__file__).resolve().parents[2]
FRAME_TSX = REPO / "frontend" / "src" / "modules" / "workspace" / "AppArtifactFrame.tsx"
DESIGN_MD = REPO / "skills" / "toup-frontend-design.md"
PACKAGED_MD = (
    REPO / "backend" / "app" / "agent" / "skills" / "builtins" / "app_html"
    / "DESIGN_SKILL.md"
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    store.ensure_root()

    from app.api.artifacts import router
    api = FastAPI()
    api.include_router(router, prefix="/api")
    return TestClient(api)


# ── 1. Persistence ───────────────────────────────────────────────────

def test_state_written_through_the_route_is_readable_and_on_disk(client, tmp_path):
    assert client.get("/api/artifacts/budget/state").json()["state"] == {}

    r = client.put(
        "/api/artifacts/budget/state",
        json={"updates": {"rows": [{"label": "Rent", "amount": 1850}],
                          "currency": "EUR"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["bytes"] > 0

    got = client.get("/api/artifacts/budget/state").json()["state"]
    assert got["currency"] == "EUR"
    assert got["rows"][0]["amount"] == 1850

    # On disk — the property that survives a container recreation. The apps
    # root lives on the /app/workspace volume for exactly this reason.
    on_disk = json.loads((tmp_path / "apps" / ".state" / "budget.json").read_text())
    assert on_disk["rows"][0]["label"] == "Rent"


def test_a_second_app_gets_its_own_state(client):
    client.put("/api/artifacts/budget/state", json={"updates": {"k": "budget"}})
    client.put("/api/artifacts/workout/state", json={"updates": {"k": "workout"}})
    assert client.get("/api/artifacts/budget/state").json()["state"]["k"] == "budget"
    assert client.get("/api/artifacts/workout/state").json()["state"]["k"] == "workout"


# ── 2. Merge semantics ───────────────────────────────────────────────

def test_writes_merge_so_two_tabs_do_not_clobber_each_other(client):
    client.put("/api/artifacts/budget/state", json={"updates": {"a": 1, "b": 2}})
    # A second "tab" writes only its own key.
    client.put("/api/artifacts/budget/state", json={"updates": {"b": 20}})
    assert client.get("/api/artifacts/budget/state").json()["state"] == {"a": 1, "b": 20}


def test_null_deletes_a_key(client):
    client.put("/api/artifacts/budget/state", json={"updates": {"a": 1, "b": 2}})
    client.put("/api/artifacts/budget/state", json={"updates": {"a": None}})
    assert client.get("/api/artifacts/budget/state").json()["state"] == {"b": 2}


def test_oversized_state_is_refused_with_a_4xx_not_a_500(client):
    r = client.put(
        "/api/artifacts/budget/state",
        json={"updates": {"blob": "x" * (store.MAX_STATE_BYTES + 10)}},
    )
    assert r.status_code == 400, r.status_code


@pytest.mark.parametrize("slug", ["../escape", "manifest", "a b", "a_b", "x" * 61])
def test_bad_slugs_are_refused_by_the_route(client, slug):
    r = client.put(f"/api/artifacts/{slug}/state", json={"updates": {"k": "v"}})
    assert r.status_code in (400, 404), (slug, r.status_code)


def test_slug_case_is_normalised_not_rejected(client):
    """Case is the one thing the store forgives — a slug is a URL segment,
    and the same app must not be two apps because the model shifted from
    'Budget' to 'budget' between turns."""
    client.put("/api/artifacts/Budget/state", json={"updates": {"k": "v"}})
    assert client.get("/api/artifacts/budget/state").json()["state"] == {"k": "v"}
    assert client.get("/api/artifacts/BUDGET/state").json()["state"] == {"k": "v"}


# ── 3. Protocol lockstep ─────────────────────────────────────────────

def _js_string_literals(text: str) -> set:
    return set(re.findall(r"'([a-zA-Z0-9_-]+)'", text)) | set(
        re.findall(r'"([a-zA-Z0-9_-]+)"', text)
    )


def test_host_and_client_agree_on_the_message_envelope():
    host = FRAME_TSX.read_text()
    doc = DESIGN_MD.read_text()

    # The two channel names. Swapping either half breaks persistence
    # silently — the app just never gets a reply and falls back to memory.
    for name in ("toup-storage", "toup-storage-host"):
        assert name in host, f"{name} missing from the host"
        assert name in doc, f"{name} missing from the client snippet"

    host_literals = _js_string_literals(host)
    doc_literals = _js_string_literals(doc)
    # Envelope fields and ops the host branches on must exist in the snippet.
    for field in ("id", "op", "key", "value", "ok"):
        assert field in host, field
        assert field in doc, f"client snippet never mentions {field!r}"
    for op in ("get", "set"):
        assert op in host_literals, op
        assert op in doc_literals, f"client snippet cannot issue op {op!r}"


def test_the_host_identifies_the_frame_by_source_not_by_origin():
    """A sandboxed frame without allow-same-origin posts with origin "null",
    so an origin check cannot distinguish it from any other null-origin
    document. The host must compare `event.source` to the contentWindow it
    rendered."""
    host = FRAME_TSX.read_text()
    assert "event.source !== frame.contentWindow" in host


def test_the_client_snippet_never_touches_storage_unguarded():
    """The single most common way a generated app renders blank: an
    unguarded localStorage call on the first line of the script, in a frame
    where it throws."""
    doc = DESIGN_MD.read_text()
    assert "localStorage" in doc
    for m in re.finditer(r"localStorage\.\w+\(", doc):
        window = doc[max(0, m.start() - 220): m.start()]
        assert "try" in window, (
            f"unguarded localStorage call at offset {m.start()} in the design skill"
        )
    assert "THROW" in doc or "throw" in doc


def test_the_shipped_design_skill_matches_the_repo_copy():
    """Two copies exist on purpose — the repo one is the source of truth for
    review, the packaged one is what actually reaches the container image
    (the repo `skills/` directory is not COPYied into it). Drift between them
    means the guidance a reviewer reads is not the guidance the model gets."""
    assert PACKAGED_MD.read_text() == DESIGN_MD.read_text()
