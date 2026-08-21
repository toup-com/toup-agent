"""What `edit_image` returns, and what it refuses to guess.

The unit tests around this one pin the pieces; this drives the actual tool, so
the wiring is covered too — a resolver that picks the right image and a result
string that never mentions it would be the same bug wearing a different hat.

Four claims:
  1. The result names an `image_id`, so the NEXT edit can be exact.
  2. The result names the SOURCE it used, so a wrong pick is visible in the
     turn rather than in the picture.
  3. The result carries what a vision model saw, and says so loudly when that
     diverges from the request. This is the sentence that stops "Morty's now
     messing around with the portal machine" about a picture with no Morty.
  4. A `source_image_id` that does not resolve is an ERROR and the renderer is
     NEVER called. Round 16 was a silent substitution; substituting quietly
     after a typo would be the same class of failure.

Run: cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \
        pytest tests/test_edit_image_tool_contract.py -q
"""

from __future__ import annotations

import io
import json
import uuid

import pytest
from PIL import Image

from app.agent.image_artifacts import ImageArtifact, ORIGIN_GENERATED
from app.agent.tool_executor import ToolExecutor


def _png(colour=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buf, format="PNG")
    return buf.getvalue()


class _Backend:
    """Storage that answers with one PNG for any key."""

    def __init__(self, data: bytes):
        self._data = data

    def open(self, key):  # noqa: D401 - context-manager shim
        return io.BytesIO(self._data)

    async def put(self, key, data):
        return None


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """A ToolExecutor with every network hop stubbed and the render recorded."""
    from app.agent import doc_generators
    from app.config import settings
    from app.services import file_storage

    state = {"render_calls": [], "prompt": None, "verdict_json": json.dumps({
        "description": "A 2D cartoon of an elderly man beside a glowing machine.",
        "matches": True, "missing": [], "unexpected": [],
    })}

    monkeypatch.setattr(settings, "image_provider", "kie", raising=False)
    monkeypatch.setattr(settings, "image_edit_enabled", True, raising=False)
    monkeypatch.setattr(file_storage, "get_storage_backend",
                        lambda: _Backend(_png()), raising=False)

    async def _fake_persist(data, filename, mime, scope):
        return doc_generators.Attachment(
            id="a" * 32, filename=filename, mime_type=mime,
            size_bytes=len(data), storage_path=f"{scope}/{filename}",
            created_at="2026-08-21T00:00:00Z", width=16, height=16,
        )
    monkeypatch.setattr(doc_generators, "_persist", _fake_persist)

    ex = ToolExecutor(workspace=str(tmp_path))
    ex.set_user_id("u-contract")

    async def _render(mode, prompt, **kw):
        state["render_calls"].append(mode)
        state["prompt"] = prompt
        return _png((200, 40, 40))
    monkeypatch.setattr(ex, "_call_kie_image", _render)

    # The three advisory model calls. `_image_vision` is the single seam both
    # the source description and the verdict go through, so stubbing it keeps
    # the real parse/render path under test.
    async def _vision(img_bytes, mime, system, question, *, timeout, max_tokens=500):
        if "about to be edited" in question:
            return "A 2D cartoon illustration of two characters in a lab."
        return state["verdict_json"]
    monkeypatch.setattr(ex, "_image_vision", _vision)

    async def _expand(system, user):
        return ("A long constructed specification naming the medium, the "
                "subjects, what stays and what changes, at more length than "
                "the instruction it came from, so it is actually used.")
    monkeypatch.setattr(ex, "_expand_scene", _expand)

    async def _no_search(query, count):
        return "No results found."
    monkeypatch.setattr(ex, "_tool_web_search",
                        lambda inp: _no_search(inp.get("query"), inp.get("count")))

    return ex, state


def _generated(att_id: str) -> dict:
    return {"id": att_id, "filename": "image_ab12cd34.png",
            "mime_type": "image/png", "size_bytes": 99,
            "storage_path": f"scope/{att_id}_image_ab12cd34.png"}


async def test_result_names_the_id_the_source_and_what_it_shows(rig, monkeypatch):
    ex, state = rig
    # The Round 16 turn shape: the picture we generated is what "it" means.
    art = ImageArtifact(attachment=_generated("b" * 32), origin=ORIGIN_GENERATED,
                        role="assistant", turn_scope="this_turn")

    async def _resolve(**kw):
        return art
    monkeypatch.setattr("app.agent.image_artifacts.resolve_implicit", _resolve)

    out = await ex._tool_edit_image({"prompt": "Make morty playing with the portal machine"})

    assert not out.startswith("ERROR:"), out
    assert "image_id: " + "a" * 32 in out, "the NEXT edit needs a handle"
    assert "source:" in out and "b" * 32 in out, "which picture was edited must be stated"
    assert "generated" in out, "and where it came from"
    assert "A 2D cartoon of an elderly man" in out, "what the result actually shows"
    assert "Do NOT restate the request" in out


async def test_divergence_is_stated_and_a_retry_offered(rig, monkeypatch):
    ex, state = rig
    state["verdict_json"] = json.dumps({
        "description": "A 2D cartoon of an elderly man beside a glowing machine.",
        "matches": False,
        "missing": ["Morty, the boy in the yellow t-shirt"],
        "unexpected": [],
    })
    art = ImageArtifact(attachment=_generated("b" * 32), origin=ORIGIN_GENERATED,
                        role="assistant")

    async def _resolve(**kw):
        return art
    monkeypatch.setattr("app.agent.image_artifacts.resolve_implicit", _resolve)

    out = await ex._tool_edit_image({"prompt": "Make morty playing with the portal machine"})

    assert "DIVERGENCE" in out
    assert "Morty" in out
    assert "offer to try again" in out
    # The user-facing pill says so too — this is the job-card surface.
    from app.agent.tool_display import display_of
    assert "differs from the request" in (display_of(out) or "")


async def test_unknown_source_image_id_errors_without_rendering(rig, monkeypatch):
    """A typo must not become a different picture — and must not be billed."""
    ex, state = rig

    async def _none(*a, **kw):
        return None
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _none)

    async def _empty(**kw):
        return []
    monkeypatch.setattr("app.agent.image_artifacts.thread_images", _empty)

    out = await ex._tool_edit_image({
        "prompt": "make it night time", "source_image_id": uuid.uuid4().hex,
    })

    assert out.startswith("ERROR:")
    assert "Do not guess a different image" in out
    assert state["render_calls"] == [], "nothing was rendered, so nothing was charged"


async def test_no_image_in_thread_asks_rather_than_reaching_out(rig, monkeypatch):
    ex, state = rig

    async def _none(**kw):
        return None
    monkeypatch.setattr("app.agent.image_artifacts.resolve_implicit", _none)

    out = await ex._tool_edit_image({"prompt": "make it night time"})

    assert out.startswith("ERROR:")
    assert "no image in this conversation" in out.lower()
    assert "different chat" in out
    assert state["render_calls"] == []


async def test_the_constructed_spec_is_what_gets_rendered(rig, monkeypatch):
    """Not the eleven words. The whole point of Bug 4."""
    ex, state = rig
    art = ImageArtifact(attachment=_generated("b" * 32), origin=ORIGIN_GENERATED,
                        role="assistant")

    async def _resolve(**kw):
        return art
    monkeypatch.setattr("app.agent.image_artifacts.resolve_implicit", _resolve)

    await ex._tool_edit_image({"prompt": "Make morty playing with the portal machine"})

    assert state["prompt"] is not None
    assert "A long constructed specification" in state["prompt"], (
        "the expanded spec, not the raw instruction, reaches the renderer"
    )
    assert "MEDIUM LOCK" in state["prompt"], (
        "the source was described as a 2D cartoon, so the medium is pinned"
    )
    assert "2D cartoon illustration" in state["prompt"]


async def test_generate_result_carries_an_id_and_an_observation(rig):
    ex, state = rig
    out = await ex._tool_generate_image({"prompt": "a cat on a windowsill"})

    assert not out.startswith("ERROR:"), out
    assert "image_id: " + "a" * 32 in out
    assert "pass this as source_image_id" in out
    assert "A 2D cartoon of an elderly man" in out  # the stubbed observation
    assert "source:" not in out, "a generated image has no source to name"
