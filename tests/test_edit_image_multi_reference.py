"""`edit_image` with more than one picture: N sources in, one render out.

Incident 6 of round 46: the user attached a portrait and a group photo and
asked for one of them to be put into the other. The agent used one picture.
Nothing malfunctioned — `edit_image`'s schema had no way to name a second
image (`source_image_id` and `image` are both scalar strings), `_tool_edit_image`
resolved exactly one source, and `kie_client.edit` put exactly one URL into an
`image_input` field that has always been a JSON array.

What these pin, in the order the failure happened:

  1. Two pictures with roles reach the renderer as TWO sources, in the order
     the call named them, with a manifest saying which is which.
  2. The BASE is the picture the call named — not the last one attached, which
     is what `turn_artifacts` would have handed back (it walks inbound media
     reversed, so "here's me, here's the beach" resolved to the beach).
  3. More than one inbound image and no id at all is a REFUSAL, once, naming
     the pictures — never a guess, and never a render.
  4. A reference that does not resolve is an error before any spend. A render
     that silently dropped one of the user's photos is paid for and wrong.
  5. A follow-up turn can name references that are no longer attached, because
     `resolve_many` is scoped like `resolve_by_id`: this turn, then this thread.

Run: cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \
        PYTHONPATH=$(pwd) pytest tests/test_edit_image_multi_reference.py -q
"""

from __future__ import annotations

import io
import json
import uuid

import pytest
from PIL import Image

from app.agent.image_artifacts import ImageArtifact, ORIGIN_UPLOADED
from app.agent.tool_executor import ToolExecutor

pytestmark = pytest.mark.asyncio


def _png(colour=(10, 20, 30), size=(16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


class _Backend:
    """Storage that answers with a DIFFERENT PNG per key, so the order the
    sources arrive in is observable rather than assumed."""

    def __init__(self, by_key: dict):
        self._by_key = by_key

    def open(self, key):
        return io.BytesIO(self._by_key.get(key, _png()))

    async def put(self, key, data):
        return None


def _att(att_id: str, filename: str, *, mime="image/png") -> dict:
    return {"id": att_id, "filename": filename, "mime_type": mime,
            "size_bytes": 99, "storage_path": f"scope/{att_id}_{filename}"}


PORTRAIT = _att("a" * 32, "portrait.png")
GROUP = _att("b" * 32, "group.png")
JACKET = _att("c" * 32, "jacket.png")
ROOM = _att("d" * 32, "room.png")

# Distinct bytes per storage key: the render records what it was handed, so a
# swapped or dropped source shows up as different bytes, not as a soft claim.
_BYTES = {
    PORTRAIT["storage_path"]: _png((250, 0, 0), (40, 80)),      # portrait shape
    GROUP["storage_path"]: _png((0, 250, 0), (80, 40)),         # landscape shape
    JACKET["storage_path"]: _png((0, 0, 250), (30, 30)),
    ROOM["storage_path"]: _png((250, 250, 0), (60, 30)),
}


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """A ToolExecutor whose renderer records the SOURCES it was handed."""
    from app.agent import doc_generators
    from app.config import settings
    from app.services import file_storage

    state = {"render_calls": [], "prompt": None, "sources": None,
             "openai_calls": 0,
             "verdict_json": json.dumps({
                 "description": "A man standing in a group of people.",
                 "matches": True, "missing": [], "unexpected": [],
             })}

    monkeypatch.setattr(settings, "image_provider", "kie", raising=False)
    monkeypatch.setattr(settings, "image_edit_enabled", True, raising=False)
    monkeypatch.setattr(file_storage, "get_storage_backend",
                        lambda: _Backend(_BYTES), raising=False)

    async def _fake_persist(data, filename, mime, scope):
        return doc_generators.Attachment(
            id="f" * 32, filename=filename, mime_type=mime,
            size_bytes=len(data), storage_path=f"{scope}/{filename}",
            created_at="2026-09-15T00:00:00Z", width=16, height=16,
        )
    monkeypatch.setattr(doc_generators, "_persist", _fake_persist)

    ex = ToolExecutor(workspace=str(tmp_path))
    ex.set_user_id("u-multi")

    async def _render(mode, prompt, **kw):
        state["render_calls"].append(mode)
        state["prompt"] = prompt
        state["sources"] = list(kw.get("sources") or [])
        return _png((200, 40, 40))
    monkeypatch.setattr(ex, "_call_kie_image", _render)

    async def _openai_edit(*a, **kw):
        state["openai_calls"] += 1
        raise AssertionError("the OpenAI edit path must not run in these tests")
    monkeypatch.setattr(ex, "_openai_edit_image", _openai_edit)

    async def _vision(img_bytes, mime, system, question, *, timeout, max_tokens=500):
        if "about to be edited" in question:
            return "A photograph of one person, head and shoulders."
        return state["verdict_json"]
    monkeypatch.setattr(ex, "_image_vision", _vision)

    async def _expand(system, user):
        return "A long constructed specification of the finished picture."
    monkeypatch.setattr(ex, "_expand_scene", _expand)

    async def _no_search(query, count):
        return "No results found."
    monkeypatch.setattr(ex, "_tool_web_search",
                        lambda inp: _no_search(inp.get("query"), inp.get("count")))

    return ex, state


def _index(atts):
    """The by-id resolver the executor uses, backed by a fixed set."""
    by_id = {a["id"].lower(): a for a in atts}

    async def _by_id(image_id, **kw):
        a = by_id.get((image_id or "").strip().lower())
        return None if a is None else ImageArtifact(
            attachment=dict(a), origin=ORIGIN_UPLOADED, role="user",
            turn_scope="this_turn")
    return _by_id


async def test_portrait_and_group_render_from_two_sources_in_order(rig, monkeypatch):
    """The reported incident. Two pictures, two sources, base first."""
    ex, state = rig
    ex.set_inbound_media([PORTRAIT, GROUP])
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _index([PORTRAIT, GROUP]))

    out = await ex._tool_edit_image({
        "prompt": "put this man into the group photo",
        "source_image_id": GROUP["id"],
        "references": [{"image_id": PORTRAIT["id"],
                        "role": "the man whose face and body should be used"}],
    })

    assert not out.startswith("ERROR:"), out
    assert state["render_calls"] == ["edit"], "exactly one render, not two"
    assert len(state["sources"]) == 2, "the second picture reached the renderer"
    # Order is the contract: sources[0] is the base.
    assert state["sources"][0][0] == _BYTES[GROUP["storage_path"]]
    assert state["sources"][1][0] == _BYTES[PORTRAIT["storage_path"]]
    # And the renderer is TOLD which is which.
    assert "Image 1 is the base photograph" in state["prompt"]
    assert "Image 2 is a reference: the man whose face and body should be used." \
        in state["prompt"]
    # The result names what went in, so a dropped reference is visible in the
    # turn rather than only in the picture.
    assert "references used:" in out and PORTRAIT["id"] in out


async def test_full_body_and_scene_keeps_the_base_as_the_base(rig, monkeypatch):
    """The base is the picture NAMED, not the last one attached.

    `turn_artifacts` walks inbound media reversed, so the implicit resolver
    would have picked the scene here. Naming the base is what stops that.
    """
    ex, state = rig
    ex.set_inbound_media([PORTRAIT, ROOM])     # scene attached LAST
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _index([PORTRAIT, ROOM]))

    out = await ex._tool_edit_image({
        "prompt": "show him standing in this room",
        "source_image_id": PORTRAIT["id"],
        "references": [{"image_id": ROOM["id"], "role": "the room this should happen in"}],
    })

    assert not out.startswith("ERROR:"), out
    assert state["sources"][0][0] == _BYTES[PORTRAIT["storage_path"]], \
        "the named base must lead, even though the scene was attached last"
    assert state["sources"][1][0] == _BYTES[ROOM["storage_path"]]


async def test_reversed_order_is_honoured_not_normalised(rig, monkeypatch):
    """Same two pictures, opposite roles → opposite source order."""
    ex, state = rig
    ex.set_inbound_media([PORTRAIT, GROUP])
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _index([PORTRAIT, GROUP]))

    await ex._tool_edit_image({
        "prompt": "put the group behind him",
        "source_image_id": PORTRAIT["id"],
        "references": [{"image_id": GROUP["id"], "role": "the people to place behind him"}],
    })

    assert state["sources"][0][0] == _BYTES[PORTRAIT["storage_path"]]
    assert state["sources"][1][0] == _BYTES[GROUP["storage_path"]]


async def test_three_references_all_arrive_and_are_all_numbered(rig, monkeypatch):
    ex, state = rig
    ex.set_inbound_media([PORTRAIT, GROUP, JACKET, ROOM])
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id",
                        _index([PORTRAIT, GROUP, JACKET, ROOM]))

    out = await ex._tool_edit_image({
        "prompt": "put him in the room wearing the jacket, with the others behind",
        "source_image_id": PORTRAIT["id"],
        "references": [
            {"image_id": ROOM["id"], "role": "the room this should happen in"},
            {"image_id": JACKET["id"], "role": "the jacket to put on him"},
            {"image_id": GROUP["id"], "role": "the people to place behind him"},
        ],
    })

    assert not out.startswith("ERROR:"), out
    assert len(state["sources"]) == 4
    assert [s[0] for s in state["sources"]] == [
        _BYTES[PORTRAIT["storage_path"]], _BYTES[ROOM["storage_path"]],
        _BYTES[JACKET["storage_path"]], _BYTES[GROUP["storage_path"]],
    ]
    for n in ("Image 2 is a reference", "Image 3 is a reference", "Image 4 is a reference"):
        assert n in state["prompt"]


async def test_clear_roles_never_ask_a_question(rig, monkeypatch):
    """Two images WITH an explicit base is unambiguous — it must just render."""
    ex, state = rig
    ex.set_inbound_media([PORTRAIT, GROUP])
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _index([PORTRAIT, GROUP]))

    out = await ex._tool_edit_image({
        "prompt": "put this man into the group photo",
        "source_image_id": GROUP["id"],
        "references": [{"image_id": PORTRAIT["id"], "role": "the man to add"}],
    })

    assert not out.startswith("ERROR:")
    assert "ask" not in out.lower().split("references used:")[0][:200]
    assert state["render_calls"] == ["edit"]


async def test_ambiguous_turn_refuses_once_and_renders_nothing(rig, monkeypatch):
    """More than one picture and no id: name them, do not guess.

    This is the branch that replaces "edit whichever was attached last". It
    costs one free tool round-trip and zero charges.
    """
    ex, state = rig
    ex.set_inbound_media([PORTRAIT, GROUP])

    out = await ex._tool_edit_image({"prompt": "put me in there"})

    assert out.startswith("ERROR:")
    assert state["render_calls"] == [], "nothing rendered, so nothing charged"
    # The inventory the model was never shown, in ATTACH order.
    assert "portrait.png" in out and PORTRAIT["id"] in out
    assert "group.png" in out and GROUP["id"] in out
    assert out.index("portrait.png") < out.index("group.png")
    assert "source_image_id" in out and "references" in out
    # One question, only if it still cannot tell.
    assert "ONE short question" in out
    assert out.lower().count("ask them") == 1


async def test_one_inbound_image_still_resolves_implicitly(rig, monkeypatch):
    """The refusal must not fire on the ordinary single-photo turn."""
    ex, state = rig
    ex.set_inbound_media([PORTRAIT])

    async def _resolve(**kw):
        return ImageArtifact(attachment=dict(PORTRAIT), origin=ORIGIN_UPLOADED,
                             role="user", turn_scope="this_turn")
    monkeypatch.setattr("app.agent.image_artifacts.resolve_implicit", _resolve)

    out = await ex._tool_edit_image({"prompt": "make it night time"})

    assert not out.startswith("ERROR:"), out
    assert state["render_calls"] == ["edit"]
    assert len(state["sources"]) == 1, "no manifest, no references, one source"
    assert "Image 1 is the base" not in state["prompt"], (
        "a single-source edit must send the prompt it always sent"
    )


async def test_followup_turn_reuses_references_from_the_thread(rig, monkeypatch):
    """The pictures are no longer attached; the ids still resolve."""
    ex, state = rig
    ex.set_inbound_media([])          # nothing attached to THIS message

    async def _by_id(image_id, **kw):
        by = {PORTRAIT["id"]: PORTRAIT, GROUP["id"]: GROUP}
        a = by.get((image_id or "").strip().lower())
        return None if a is None else ImageArtifact(
            attachment=dict(a), origin=ORIGIN_UPLOADED, role="user")
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _by_id)

    async def _thread(**kw):
        return [ImageArtifact(attachment=dict(a), origin=ORIGIN_UPLOADED, role="user")
                for a in (GROUP, PORTRAIT)]
    monkeypatch.setattr("app.agent.image_artifacts.thread_images", _thread)

    out = await ex._tool_edit_image({
        "prompt": "same thing but at sunset",
        "source_image_id": GROUP["id"],
        "references": [{"image_id": PORTRAIT["id"], "role": "the man to add"}],
    })

    assert not out.startswith("ERROR:"), out
    assert len(state["sources"]) == 2


async def test_unresolvable_reference_errors_before_any_render(rig, monkeypatch):
    ex, state = rig
    ex.set_inbound_media([PORTRAIT, GROUP])
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _index([PORTRAIT, GROUP]))

    async def _thread(**kw):
        return []
    monkeypatch.setattr("app.agent.image_artifacts.thread_images", _thread)

    ghost = uuid.uuid4().hex
    out = await ex._tool_edit_image({
        "prompt": "put this man into the group photo",
        "source_image_id": GROUP["id"],
        "references": [{"image_id": ghost, "role": "the man to add"}],
    })

    assert out.startswith("ERROR:")
    assert ghost in out
    assert state["render_calls"] == [], "a dropped reference must not be billed"


async def test_a_reference_may_not_also_be_the_base(rig, monkeypatch):
    ex, state = rig
    ex.set_inbound_media([PORTRAIT, GROUP])
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _index([PORTRAIT, GROUP]))

    out = await ex._tool_edit_image({
        "prompt": "do the thing",
        "source_image_id": GROUP["id"],
        "references": [{"image_id": GROUP["id"], "role": "the group"}],
    })

    assert out.startswith("ERROR:")
    assert state["render_calls"] == []


async def test_more_references_than_the_provider_takes_is_refused_not_truncated(rig, monkeypatch):
    """Truncating would drop pictures the user asked for, and still bill."""
    ex, state = rig
    from app.services.kie_client import KIE_MAX_IMAGE_INPUT

    many = [_att(f"{i:032x}", f"ref{i}.png") for i in range(1, KIE_MAX_IMAGE_INPUT + 2)]
    ex.set_inbound_media([GROUP] + many)
    monkeypatch.setattr("app.agent.image_artifacts.resolve_by_id", _index([GROUP] + many))

    out = await ex._tool_edit_image({
        "prompt": "combine them all",
        "source_image_id": GROUP["id"],
        "references": [{"image_id": a["id"], "role": "a reference"} for a in many],
    })

    assert out.startswith("ERROR:")
    assert str(KIE_MAX_IMAGE_INPUT - 1) in out
    assert state["render_calls"] == []
