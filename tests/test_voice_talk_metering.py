"""W0.5 regression tests — voice attack-surface + metering.

(a) The unauthenticated /api/voice/ws/talk/{user_id} WebSocket and the
    /api/voice/talk/status endpoint are REMOVED (they accepted ANY caller,
    took user_id from the path, and ran unmetered STT/TTS on the platform
    key). The only voice conversation surface is /api/ws/realtime.
(b) Realtime voice metering fires for v1 sessions too: the gate is
    using_platform_key ONLY — previously `_v2_active() and using_platform_key`,
    which billed ZERO credits for every v1 platform-key session.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from app.config import settings
import app.api.ws_realtime as rt


@pytest.fixture
def v2_on(monkeypatch):
    monkeypatch.setattr(settings, "voice_realtime_v2", True)


@pytest.fixture
def v2_off(monkeypatch):
    monkeypatch.setattr(settings, "voice_realtime_v2", False)


# ── (a) Talk-mode surface removed ─────────────────────────────────────

def test_talk_routes_absent_from_voice_router():
    from app.api import voice

    paths = [r.path for r in voice.router.routes]
    assert not any("/ws/talk" in p for p in paths), paths
    assert not any("/talk/status" in p for p in paths), paths
    # The dead wiring went with it.
    assert not hasattr(voice, "set_voice_refs")
    assert not hasattr(voice, "talk_mode_ws")
    assert not hasattr(voice, "talk_mode_status")


async def test_talk_endpoints_404_when_mounted():
    """Mount the voice router exactly like the three entrypoints do and
    verify the old talk URLs resolve to nothing."""
    from app.api.voice import router as voice_router

    app = FastAPI()
    app.include_router(voice_router, prefix=settings.api_prefix)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"{settings.api_prefix}/voice/talk/status")
        assert r.status_code == 404
        r = await client.get(f"{settings.api_prefix}/voice/ws/talk/any-user-id")
        assert r.status_code == 404
    # No HTTP or WebSocket route may claim the old ws/talk path.
    ws_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert not any("/ws/talk" in p for p in ws_paths), ws_paths


def test_realtime_surface_untouched():
    ws_paths = [r.path for r in rt.router.routes]
    assert any(p.endswith("/ws/realtime") for p in ws_paths), ws_paths


# ── (b) v1 sessions on the platform key are metered ──────────────────

DOCS_USAGE = {
    "input_tokens": 132,
    "output_tokens": 121,
    "input_token_details": {
        "text_tokens": 119,
        "audio_tokens": 13,
        "cached_tokens": 64,
        "cached_tokens_details": {"text_tokens": 64, "audio_tokens": 0},
    },
    "output_token_details": {"text_tokens": 30, "audio_tokens": 91},
}

RESPONSE_DONE = {"id": "resp_abc123", "usage": DOCS_USAGE}


@pytest.fixture
def meter_calls(monkeypatch):
    calls = []

    async def _fake_meter(user_id, model, usage, response_id):
        calls.append((user_id, model, usage, response_id))

    monkeypatch.setattr(rt, "_meter_voice_turn", _fake_meter)
    return calls


async def test_v1_platform_key_session_is_metered(v2_off, meter_calls):
    task = rt._maybe_meter_response("u-1", RESPONSE_DONE, using_platform_key=True)
    assert task is not None
    await task
    assert meter_calls == [("u-1", "gpt-realtime", DOCS_USAGE, "resp_abc123")]


async def test_v2_platform_key_session_still_metered(v2_on, meter_calls):
    task = rt._maybe_meter_response("u-2", RESPONSE_DONE, using_platform_key=True)
    assert task is not None
    await task
    assert meter_calls == [
        ("u-2", settings.voice_realtime_model, DOCS_USAGE, "resp_abc123")
    ]


@pytest.mark.parametrize("v2_flag", [False, True])
async def test_byok_session_never_metered(v2_flag, meter_calls, monkeypatch):
    # BYOK pays OpenAI directly — charging credits too would double-bill.
    monkeypatch.setattr(settings, "voice_realtime_v2", v2_flag)
    assert rt._maybe_meter_response("u-3", RESPONSE_DONE, using_platform_key=False) is None
    assert meter_calls == []


async def test_response_without_usage_or_id_not_metered(v2_off, meter_calls):
    assert rt._maybe_meter_response("u-4", {"id": "resp_x"}, using_platform_key=True) is None
    assert rt._maybe_meter_response("u-4", {"usage": DOCS_USAGE}, using_platform_key=True) is None
    assert meter_calls == []


def test_v1_model_prices_nonzero(v2_off):
    """v1 bills under its real slug — 'gpt-realtime' must be in the pricing
    table (not silently zero, not the wrong-generation fallback)."""
    assert "gpt-realtime" in settings.voice_realtime_pricing_per_1m
    assert rt._usage_to_cost_cents("gpt-realtime", DOCS_USAGE) > 0
