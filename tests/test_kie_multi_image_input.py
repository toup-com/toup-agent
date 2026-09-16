"""The provider half: `image_input` carries every source, in order, once.

`image_input` was always a JSON array and we always put exactly one URL in it
(kie_client.edit / start_task). These drive the real client against a recording
transport, so the claim under test is what goes ON THE WIRE, not what a helper
returns.

Pinned here:
  * N sources → N uploads and N downloadUrls, in the order given.
  * The aspect comes from the BASE (sources[0]) — a portrait subject dropped
    into a landscape scene must come out portrait.
  * One source is byte-identical to what shipped before references existed.
  * Above the cap it is a KieError with NOTHING created — an overflow must not
    cost an upload, let alone a render.

Run: cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \
        PYTHONPATH=$(pwd) pytest tests/test_kie_multi_image_input.py -q
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

pytestmark = pytest.mark.asyncio


def _png(size=(40, 80), colour=(1, 2, 3)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


class _Rec:
    """Records every request and answers the Kie flow deterministically."""

    def __init__(self):
        self.uploads: list[dict] = []
        self.creates: list[dict] = []
        self._n = 0

    class _R:
        def __init__(self, payload, status=200):
            self.status_code = status
            self._p = payload
            self.content = b"IMAGEBYTES"
            self.text = ""

        def json(self):
            return self._p

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None, **kw):
        if "file-base64-upload" in url:
            self.uploads.append(json or {})
            self._n += 1
            return self._R({"data": {"downloadUrl": f"https://kie/{self._n}.png"}})
        if "createTask" in url:
            self.creates.append(json or {})
            return self._R({"code": 200, "data": {"taskId": "task-1"}})
        raise AssertionError(f"unexpected POST {url}")

    async def get(self, url, params=None, headers=None, **kw):
        import json as _json
        if "recordInfo" in url:
            return self._R({"data": {
                "state": "success", "creditsConsumed": 18,
                "resultJson": _json.dumps({"resultUrls": ["https://kie/out.png"]}),
            }})
        return self._R({}, status=200)


@pytest.fixture
def rec(monkeypatch):
    from app.services import kie_client
    r = _Rec()
    monkeypatch.setattr(kie_client.settings, "kie_api_key", "test-key", raising=False)
    monkeypatch.setattr(kie_client.httpx, "AsyncClient", lambda *a, **k: r)
    return r


async def test_edit_uploads_every_source_and_keeps_the_order(rec):
    from app.services import kie_client

    base, ref1, ref2 = _png((40, 80)), _png((30, 30), (9, 9, 9)), _png((20, 20), (7, 7, 7))
    await kie_client.edit("compose these", sources=[
        (base, "image/png"), (ref1, "image/png"), (ref2, "image/jpeg"),
    ])

    assert len(rec.uploads) == 3, "one upload PER source, not one for the base"
    assert len(rec.creates) == 1, "one createTask — one render, one charge"
    inp = rec.creates[0]["input"]
    assert inp["image_input"] == [
        "https://kie/1.png", "https://kie/2.png", "https://kie/3.png",
    ], "image_input must carry every source in the order given"
    # Distinct remote names, or two references can collide on the file host.
    names = [u["fileName"] for u in rec.uploads]
    assert len(set(names)) == 3, names
    assert names[0] == "src.png", "the single-source name is unchanged"


async def test_aspect_comes_from_the_base_not_a_reference(rec):
    from app.services import kie_client

    portrait = _png((40, 80))        # 1:2  → nearest 9:16
    landscape = _png((80, 40))       # 2:1  → nearest 16:9
    await kie_client.edit("put him in the room", sources=[
        (portrait, "image/png"), (landscape, "image/png"),
    ])
    assert rec.creates[0]["input"]["aspect_ratio"] == "9:16"

    rec.creates.clear()
    rec.uploads.clear()
    await kie_client.edit("put the room around him", sources=[
        (landscape, "image/png"), (portrait, "image/png"),
    ])
    assert rec.creates[0]["input"]["aspect_ratio"] == "16:9"


async def test_single_source_scalar_form_still_works(rec):
    from app.services import kie_client

    await kie_client.edit("make it night", _png((80, 80)), "image/png")
    assert len(rec.uploads) == 1
    assert rec.creates[0]["input"]["image_input"] == ["https://kie/1.png"]
    assert rec.uploads[0]["fileName"] == "src.png"


async def test_start_task_edit_takes_a_sequence(rec):
    from app.services import kie_client

    task = await kie_client.start_task("edit", "compose", sources=[
        (_png((60, 60)), "image/png"), (_png((20, 20)), "image/png"),
    ])
    assert task == "task-1"
    assert len(rec.uploads) == 2
    assert rec.creates[0]["input"]["image_input"] == [
        "https://kie/1.png", "https://kie/2.png"]


async def test_over_the_cap_raises_before_any_upload(rec):
    from app.services import kie_client

    too_many = [(_png((10, 10)), "image/png")
                for _ in range(kie_client.KIE_MAX_IMAGE_INPUT + 1)]
    with pytest.raises(kie_client.KieError) as e:
        await kie_client.start_task("edit", "compose", sources=too_many)
    assert str(kie_client.KIE_MAX_IMAGE_INPUT) in str(e.value)
    assert rec.uploads == [], "an overflow must not cost an upload"
    assert rec.creates == [], "and must not create a billable task"


async def test_edit_with_no_source_is_an_error_not_a_generate(rec):
    from app.services import kie_client

    with pytest.raises(kie_client.KieError):
        await kie_client.edit("make something", sources=[])
    assert rec.creates == []


async def test_coerce_sources_drops_nothing_and_counts_everything():
    from app.services.kie_client import _coerce_sources

    out = _coerce_sources([(b"a", "image/png"), (b"b", "image/jpeg")], None, "image/png")
    assert out == [(b"a", "image/png"), (b"b", "image/jpeg")]
    # The scalar form is the 1-source shorthand, and only when there is no list.
    assert _coerce_sources(None, b"z", "image/webp") == [(b"z", "image/webp")]
    assert _coerce_sources([], None, "image/png") == []
