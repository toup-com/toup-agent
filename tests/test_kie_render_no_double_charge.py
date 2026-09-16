"""One request, one billed render — even when a poll does not answer.

`_call_kie_image` used to leave its poll loop on ANY non-200
(`if r.status_code != 200: break`), which fell through to the OpenAI edit path.
That is a SECOND billed picture for one request: the Kie job that was already
started is held against the user's credits at `/kie/image/start` precisely so an
abandoned render still stays paid for. A flaky gateway on one poll therefore
cost the user two pictures and delivered one.

The distinction these pin is between a lost QUESTION and a VERDICT:

  * 502 with no JSON body, 503, a transport error → the hop failed, the render
    did not. Retry. Never fall back.
  * 502 {code: "kie_failed"} → the platform's verdict; it refunded the hold.
    Terminal, and falling back costs exactly one picture in total.
  * the transient budget spent, or the job deadline passed with the render
    still live → `_KieRenderUnresolved`, and `_tool_edit_image` must SAY so
    rather than render again.

Run: cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \
        PYTHONPATH=$(pwd) pytest tests/test_kie_render_no_double_charge.py -q
"""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from app.agent.tool_executor import ToolExecutor, _KieRenderUnresolved

pytestmark = pytest.mark.asyncio


def _png(colour=(3, 3, 3)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buf, format="PNG")
    return buf.getvalue()


class _Resp:
    def __init__(self, status, payload=None, *, body_is_json=True):
        self.status_code = status
        self._p = payload if payload is not None else {}
        self._json = body_is_json
        self.text = "" if body_is_json else "<html>bad gateway</html>"

    def json(self):
        if not self._json:
            raise ValueError("not json")
        return self._p


class _Proxy:
    """A fake platform proxy: one /start, then a scripted sequence of /polls."""

    def __init__(self, polls):
        self.polls = list(polls)
        self.starts: list[dict] = []
        self.poll_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None, **kw):
        if url.endswith("/llm/kie/image/start"):
            self.starts.append(json or {})
            return _Resp(200, {"task_id": "task-1", "reservation_id": "res-1"})
        if url.endswith("/llm/kie/image/poll"):
            self.poll_count += 1
            step = self.polls.pop(0) if self.polls else _Resp(200, {"status": "pending"})
            if isinstance(step, Exception):
                raise step
            return step
        raise AssertionError(f"unexpected POST {url}")


@pytest.fixture
def ex(monkeypatch, tmp_path):
    from app.config import settings
    monkeypatch.setattr(settings, "platform_api_url", "https://plat.test", raising=False)
    monkeypatch.setattr(settings, "toup_token", "tok", raising=False)
    monkeypatch.setattr(settings, "kie_poll_interval_s", 0.0, raising=False)
    monkeypatch.setattr(settings, "kie_job_timeout_s", 5.0, raising=False)
    e = ToolExecutor(workspace=str(tmp_path))
    e.set_user_id("u-charge")
    return e


def _ok_poll():
    return _Resp(200, {"status": "success",
                       "b64": base64.b64encode(_png((9, 9, 9))).decode()})


async def test_a_transient_poll_error_retries_the_same_task(ex, monkeypatch):
    """The founder-visible bug: one flaky poll, two paid pictures."""
    from app.agent import tool_executor as te

    proxy = _Proxy([
        _Resp(502, body_is_json=False),      # a gateway, not the platform
        _Resp(503, {"detail": "redeploying"}),
        _ok_poll(),
    ])
    monkeypatch.setattr(te.httpx, "AsyncClient", lambda *a, **k: proxy)

    out = await ex._call_kie_image("edit", "make it night",
                                   sources=[(_png(), "image/png")])

    assert out, "the picture that was already rendering must be collected"
    assert len(proxy.starts) == 1, "one start — a retry must not create a second job"
    assert proxy.poll_count == 3


async def test_a_provider_verdict_is_terminal_not_retried(ex, monkeypatch):
    """502 {code: kie_failed} means the render failed and the hold was refunded.

    Retrying that would burn the whole job deadline for an answer we already
    have; falling back costs exactly one picture in total, which is correct.
    """
    from app.agent import tool_executor as te

    proxy = _Proxy([_Resp(502, {"detail": {"code": "kie_failed",
                                           "moderation": False,
                                           "message": "render failed"}})])
    monkeypatch.setattr(te.httpx, "AsyncClient", lambda *a, **k: proxy)

    with pytest.raises(RuntimeError) as err:
        await ex._call_kie_image("edit", "make it night",
                                 sources=[(_png(), "image/png")])
    assert not isinstance(err.value, _KieRenderUnresolved)
    assert proxy.poll_count == 1, "a verdict is not retried"


async def test_transient_budget_spent_refuses_rather_than_re_rendering(ex, monkeypatch):
    from app.agent import tool_executor as te

    monkeypatch.setattr(te, "_KIE_POLL_TRANSIENT_RETRIES", 2)
    proxy = _Proxy([_Resp(503, {}) for _ in range(10)])
    monkeypatch.setattr(te.httpx, "AsyncClient", lambda *a, **k: proxy)

    with pytest.raises(_KieRenderUnresolved):
        await ex._call_kie_image("edit", "make it night",
                                 sources=[(_png(), "image/png")])
    assert proxy.poll_count == 3, "budget + 1, then stop — not until the deadline"


async def test_a_dropped_connection_mid_render_is_unresolved_not_failed(ex, monkeypatch):
    from app.agent import tool_executor as te

    monkeypatch.setattr(te, "_KIE_POLL_TRANSIENT_RETRIES", 1)
    proxy = _Proxy([ConnectionError("reset"), ConnectionError("reset")])
    monkeypatch.setattr(te.httpx, "AsyncClient", lambda *a, **k: proxy)

    with pytest.raises(_KieRenderUnresolved):
        await ex._call_kie_image("edit", "x", sources=[(_png(), "image/png")])


async def test_a_still_rendering_job_at_the_deadline_is_unresolved(ex, monkeypatch):
    """It is still live and still held against the user's credits."""
    from app.agent import tool_executor as te
    from app.config import settings

    monkeypatch.setattr(settings, "kie_job_timeout_s", 0.05, raising=False)
    proxy = _Proxy([])            # every poll answers "pending"
    monkeypatch.setattr(te.httpx, "AsyncClient", lambda *a, **k: proxy)

    with pytest.raises(_KieRenderUnresolved):
        await ex._call_kie_image("edit", "x", sources=[(_png(), "image/png")])


async def test_the_start_payload_carries_the_sources_and_one_idempotency_key(ex, monkeypatch):
    from app.agent import tool_executor as te

    proxy = _Proxy([_ok_poll()])
    monkeypatch.setattr(te.httpx, "AsyncClient", lambda *a, **k: proxy)

    base, ref = _png((1, 1, 1)), _png((2, 2, 2))
    await ex._call_kie_image("edit", "compose",
                             sources=[(base, "image/png"), (ref, "image/jpeg")])

    body = proxy.starts[0]
    assert len(body["images_b64"]) == 2
    assert body["images_b64"][0]["b64"] == base64.b64encode(base).decode()
    assert body["images_b64"][1]["mime"] == "image/jpeg"
    # The scalar stays, so a platform that predates images_b64 still renders
    # the base rather than answering "edit mode requires image_b64".
    assert body["image_b64"] == base64.b64encode(base).decode()
    assert body.get("idempotency_key"), "a retry of /start must be dedupable"


async def test_edit_tool_reports_an_unresolved_render_and_never_falls_back(monkeypatch, tmp_path):
    """The whole point: a second render is a second charge."""
    from app.agent import doc_generators
    from app.agent.image_artifacts import ImageArtifact, ORIGIN_UPLOADED
    from app.config import settings
    from app.services import file_storage

    monkeypatch.setattr(settings, "image_provider", "kie", raising=False)
    monkeypatch.setattr(settings, "image_edit_enabled", True, raising=False)

    class _B:
        def open(self, key):
            return io.BytesIO(_png())

        async def put(self, key, data):
            return None
    monkeypatch.setattr(file_storage, "get_storage_backend", lambda: _B(), raising=False)

    async def _fake_persist(data, filename, mime, scope):
        return doc_generators.Attachment(
            id="e" * 32, filename=filename, mime_type=mime, size_bytes=len(data),
            storage_path=f"{scope}/{filename}", created_at="2026-09-15T00:00:00Z")
    monkeypatch.setattr(doc_generators, "_persist", _fake_persist)

    ex = ToolExecutor(workspace=str(tmp_path))
    ex.set_user_id("u-charge")
    ex.set_inbound_media([])

    calls = {"openai": 0}

    async def _unresolved(*a, **kw):
        raise _KieRenderUnresolved("the picture was still rendering after 420s")
    monkeypatch.setattr(ex, "_call_kie_image", _unresolved)

    async def _openai(*a, **kw):
        calls["openai"] += 1
        return base64.b64encode(_png()).decode()
    monkeypatch.setattr(ex, "_openai_edit_image", _openai)

    async def _vision(img, mime, system, question, *, timeout, max_tokens=500):
        return json.dumps({"description": "x", "matches": True,
                           "missing": [], "unexpected": []})
    monkeypatch.setattr(ex, "_image_vision", _vision)

    async def _expand(system, user):
        return "A constructed specification."
    monkeypatch.setattr(ex, "_expand_scene", _expand)
    monkeypatch.setattr(ex, "_tool_web_search", lambda inp: _no_results())

    async def _resolve(**kw):
        return ImageArtifact(
            attachment={"id": "d" * 32, "filename": "p.png",
                        "mime_type": "image/png", "size_bytes": 1,
                        "storage_path": "scope/p.png"},
            origin=ORIGIN_UPLOADED, role="user", turn_scope="this_turn")
    monkeypatch.setattr("app.agent.image_artifacts.resolve_implicit", _resolve)

    out = await ex._tool_edit_image({"prompt": "make it night"})

    assert out.startswith("ERROR:")
    assert calls["openai"] == 0, "a fallback render here bills the user twice"
    assert "charged twice" in out
    assert "may still arrive" in out


async def _no_results():
    return "No results found."


async def test_generate_tool_reports_an_unresolved_render_and_never_falls_back(
        monkeypatch, tmp_path):
    """The generate half of the same guarantee, and the half that had no test
    at all: mutation-verified, removing the `_KieRenderUnresolved` branch from
    `_tool_generate_image` left 54 image tests green while the OpenAI path
    below billed a SECOND picture for a Kie job already held against the
    user's credits."""
    from app.config import settings

    monkeypatch.setattr(settings, "image_provider", "kie", raising=False)

    ex = ToolExecutor(workspace=str(tmp_path))
    ex.set_user_id("u-charge")
    ex.set_inbound_media([])

    calls = {"openai": 0, "client": 0}

    async def _unresolved(*a, **kw):
        raise _KieRenderUnresolved("the picture was still rendering after 420s")
    monkeypatch.setattr(ex, "_call_kie_image", _unresolved)

    async def _openai(*a, **kw):
        calls["openai"] += 1
        return base64.b64encode(_png()).decode()
    monkeypatch.setattr(ex, "_openai_generate_image", _openai)

    def _client(*a, **kw):
        # Reaching the client factory at all means the branch fell through.
        calls["client"] += 1
        return None
    monkeypatch.setattr("app.services.bundle_client.make_openai_client", _client)

    async def _expand(system, user):
        return "A constructed specification."
    monkeypatch.setattr(ex, "_expand_scene", _expand)
    monkeypatch.setattr(ex, "_tool_web_search", lambda inp: _no_results())

    async def _ground(_p):
        return ""
    monkeypatch.setattr(ex, "_ground_image_terms", _ground)

    out = await ex._tool_generate_image({"prompt": "a quiet harbour at dawn"})
    text = out if isinstance(out, str) else out.content

    assert text.startswith("ERROR:")
    assert "charged twice" in text, "the honest sentence is the whole fix"
    assert "may still arrive" in text
    assert calls["openai"] == 0, "a fallback render here bills the user twice"
    assert calls["client"] == 0, "the branch must return before the client factory"
