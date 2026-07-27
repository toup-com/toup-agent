"""analyze_image must route through the platform LLM proxy in bundle mode.

Regression for audit W0.4a: `_tool_analyze_image` POSTed raw httpx to
https://api.openai.com/v1/chat/completions with a hardcoded "gpt-4o" and
`settings.openai_api_key` — bypassing the bundle proxy, credit metering,
and model governance entirely. The fix routes the call through
`bundle_client.make_openai_client` (same idiom as generate_image), so:

  - bundle mode  → {platform}/llm/openai/v1 with TOUP_TOKEN as auth; the
    tenant's raw OpenAI key never appears on the wire
  - manual mode  → api.openai.com direct with the tenant's own key
  - model comes from settings.analyze_image_model (default gpt-4o —
    behavior-preserving, cheap override possible)

No network: tests intercept httpx.AsyncHTTPTransport.handle_async_request
(the transport under BOTH the SDK's default client and the bundle proxy
client) and capture the outbound request.

Run: cd backend && python3 -m pytest tests/test_analyze_image_proxy.py -q
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

from app.agent.tool_executor import ToolExecutor


# Override the suite-wide DB autouse fixture; these tests don't touch DB.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield


_CANNED = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "A red square."},
        }
    ],
}


@pytest.fixture
def captured(monkeypatch):
    """Intercept the httpx transport layer and record outbound requests."""
    reqs: list[httpx.Request] = []

    async def _handle(self, request):
        reqs.append(request)
        return httpx.Response(
            200,
            json=_CANNED,
            headers={"content-type": "application/json"},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _handle)
    return reqs


@pytest.fixture
def keys_restore():
    """Re-sync the key_provider singleton with real settings after each
    test — the tests call keys.refresh() under patched settings."""
    from app.services.key_provider import keys
    yield keys
    keys.refresh()


def _settings_patch(**overrides):
    from app.config import settings
    return patch.multiple(settings, **overrides)


def _exec(tmp_path) -> ToolExecutor:
    ex = ToolExecutor(workspace=str(tmp_path))
    ex.set_user_id("u-analyze-1")
    return ex


async def test_bundle_mode_routes_through_proxy_with_toup_token(tmp_path, captured, keys_restore):
    """Bundle mode → proxy base_url + TOUP_TOKEN auth. The tenant's raw
    OpenAI key must never reach the wire, even when it is set."""
    sentinel = "sk-RAW-KEY-MUST-NOT-LEAK"
    with _settings_patch(
        llm_mode="bundle",
        toup_token="toup_ct_test_xyz",
        platform_api_url="https://toup.ai/api",
        openai_api_key=sentinel,
    ):
        keys_restore.refresh()
        out = await _exec(tmp_path)._tool_analyze_image(
            {"image": "https://example.com/pic.png", "question": "what is it?"}
        )

    assert out == "A red square."
    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == "https://toup.ai/api/llm/openai/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer toup_ct_test_xyz"
    # The raw tenant key must appear in NO header — proxy meters via toup_token.
    for name, value in req.headers.items():
        assert sentinel not in value, f"raw OpenAI key leaked in header {name}"

    body = json.loads(req.read())
    assert body["model"] == "gpt-4o"  # settings.analyze_image_model default
    assert body["messages"][0]["content"][0] == {"type": "text", "text": "what is it?"}
    assert body["messages"][0]["content"][1]["image_url"]["url"] == "https://example.com/pic.png"


async def test_bundle_mode_honors_analyze_image_model_override(tmp_path, captured, keys_restore):
    with _settings_patch(
        llm_mode="bundle",
        toup_token="toup_ct_test_xyz",
        platform_api_url="https://toup.ai/api",
        analyze_image_model="gpt-4o-mini",
    ):
        keys_restore.refresh()
        out = await _exec(tmp_path)._tool_analyze_image(
            {"image": "https://example.com/pic.png"}
        )

    assert out == "A red square."
    assert json.loads(captured[0].read())["model"] == "gpt-4o-mini"


async def test_manual_mode_falls_back_to_direct_openai_with_tenant_key(tmp_path, captured, keys_restore):
    """No bundle → direct api.openai.com with the tenant's own key
    (previous behavior, minus the raw httpx POST)."""
    with _settings_patch(
        llm_mode="manual",
        toup_token="",
        openai_api_key="sk-test-manual-key",
    ):
        keys_restore.refresh()
        out = await _exec(tmp_path)._tool_analyze_image(
            {"image": "https://example.com/pic.png", "question": "describe"}
        )

    assert out == "A red square."
    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test-manual-key"


async def test_manual_mode_without_key_errors_before_any_request(tmp_path, captured, keys_restore):
    with _settings_patch(llm_mode="manual", toup_token="", openai_api_key=""):
        keys_restore.refresh()
        out = await _exec(tmp_path)._tool_analyze_image(
            {"image": "https://example.com/pic.png"}
        )

    assert out.startswith("ERROR:")
    assert captured == [], "must not attempt a request with no access configured"


async def test_workspace_file_is_sent_as_data_uri_through_proxy(tmp_path, captured, keys_restore):
    """The local-file branch (base64 data URI) still works through the
    SDK client path."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfakepixels")

    with _settings_patch(
        llm_mode="bundle",
        toup_token="toup_ct_test_xyz",
        platform_api_url="https://toup.ai/api",
    ):
        keys_restore.refresh()
        out = await _exec(tmp_path)._tool_analyze_image({"image": "shot.png"})

    assert out == "A red square."
    body = json.loads(captured[0].read())
    url = body["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
