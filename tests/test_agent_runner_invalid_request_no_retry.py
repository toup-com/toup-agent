"""A provider 400 that refuses the REQUEST is not retried.

Observed 2026-10-03 on the founder image (P2.7): six
``Function tools with reasoning_effort are not supported for gpt-6-sol in
/v1/chat/completions`` 400s (``type: invalid_request_error``), each sent three
times with the generic ladder's 2 s and 4 s sleeps before the gpt-4o fallback
— about 18 s of one voice job spent on guaranteed refusals. The identical
payload is refused identically; the runner now goes straight to the fallback
model path (or raises when there is none) without sleeping. 429 / 5xx /
timeouts keep the ladder exactly as before.

These drive the REAL ``AgentRunner.run()`` (harness from
``test_llm_trace_header.py``) with a stub LLM that refuses the primary model.

LANE: agent — ``run()`` touches ``conversations`` (AGENT_ONLY). One file per
process:

    cd backend && RUN_MODE=agent PYTHONPATH=. pytest -q -p no:cacheprovider \\
        tests/test_agent_runner_invalid_request_no_retry.py
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest

PRIMARY = "gpt-5.5-mini"
FALLBACK = "gpt-4o"

_P27_BODY = {
    "error": {
        "message": (
            "Function tools with reasoning_effort are not supported for gpt-6-sol "
            "in /v1/chat/completions. To use function tools, use /v1/responses or "
            "set reasoning_effort to 'none'."
        ),
        "type": "invalid_request_error",
        "param": "reasoning_effort",
        "code": None,
    }
}


class _SdkError(Exception):
    """The shape openai.APIStatusError exposes: status_code + body + text."""

    def __init__(self, message: str, *, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _proxy_400(body=_P27_BODY):
    # The platform llm-proxy relays the upstream JSON as a STRING under
    # `detail`; the OpenAI SDK keeps that dict as `.body`.
    detail = json.dumps(body, indent=2)
    return _SdkError(f"Error code: 400 - {{'detail': {detail!r}}}",
                     status_code=400, body={"detail": detail})


class _RefusingLLM:
    """Raises `make_error()` for the primary model, answers on any other."""

    def __init__(self, make_error):
        self.make_error = make_error
        self.models: list = []

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent

        model = kwargs.get("model")
        self.models.append(model)
        if model == PRIMARY:
            raise self.make_error()
        yield StreamEvent(type="text", text="done")
        yield StreamEvent(
            type="message_end", stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 2})


@pytest.fixture
def harness(monkeypatch, tmp_path):
    import app.agent.agent_runner as ar
    from app.agent.tool_executor import ToolExecutor
    from app.services import key_provider

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)
    monkeypatch.setattr(ar.settings, "agent_fallback_model", FALLBACK, raising=False)
    monkeypatch.setattr(key_provider.KeyProvider, "has_openai",
                        property(lambda self: True))
    monkeypatch.setattr(key_provider.KeyProvider, "has_anthropic",
                        property(lambda self: False))

    # Record the retry ladder's back-off sleeps without spending them.
    sleeps: list = []
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ar, "RETRY_DELAY", 0.25)

    async def recording_sleep(delay, *a, **kw):
        if delay in (0.25, 0.5):
            sleeps.append(delay)
            delay = 0
        return await real_sleep(delay, *a, **kw)

    monkeypatch.setattr(ar.asyncio, "sleep", recording_sleep)

    tools = ToolExecutor(workspace=str(tmp_path))

    async def fake_execute(name, payload):
        return "result"

    monkeypatch.setattr(tools, "execute", fake_execute)

    async def drive(llm):
        from app.db import User, async_session_maker

        user_id = str(uuid.uuid4())
        async with async_session_maker() as db:
            db.add(User(id=user_id, email=f"nr-{user_id[:8]}@example.test",
                        hashed_password="x" * 60, name="NoRetry"))
            await db.commit()
        runner = ar.AgentRunner(llm_service=llm, tool_executor=tools)
        runner.anthropic = llm
        return await runner.run(
            user_message="hello",
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            channel="app",
            model_override=PRIMARY,
            save_user_message=False,
            save_assistant_message=False,
            disable_post_processing=True,
        )

    return drive, sleeps


@pytest.mark.asyncio
async def test_agent_runner_does_not_retry_invalid_request_400(harness):
    """The incident error, as the proxy relays it: one primary call, no
    sleep, straight to the fallback model."""
    drive, sleeps = harness
    llm = _RefusingLLM(_proxy_400)
    await drive(llm)
    assert llm.models == [PRIMARY, FALLBACK], llm.models
    assert sleeps == [], f"slept {sleeps} before a deterministic 400"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["unsupported_parameter", "unsupported_value"])
async def test_unsupported_parameter_codes_are_not_retried(harness, code):
    drive, sleeps = harness
    body = {"error": {"message": "nope", "type": "some_other_type",
                      "param": "temperature", "code": code}}
    llm = _RefusingLLM(lambda: _SdkError("Error code: 400", status_code=400,
                                         body=body["error"]))
    await drive(llm)
    assert llm.models == [PRIMARY, FALLBACK], llm.models
    assert sleeps == []


@pytest.mark.asyncio
async def test_invalid_request_400_raises_without_sleeping_when_no_fallback(
    harness, monkeypatch,
):
    from app.services import key_provider

    drive, sleeps = harness
    monkeypatch.setattr(key_provider.KeyProvider, "has_openai",
                        property(lambda self: False))
    llm = _RefusingLLM(_proxy_400)
    with pytest.raises(_SdkError):
        await drive(llm)
    assert llm.models == [PRIMARY], llm.models
    assert sleeps == []


@pytest.mark.asyncio
@pytest.mark.parametrize("make_error", [
    lambda: _SdkError("Error code: 503 - upstream unavailable", status_code=503,
                      body={"error": {"type": "server_error", "message": "x"}}),
    lambda: _SdkError("Error code: 500 - invalid_request_error in a 5xx text",
                      status_code=500,
                      body={"error": {"type": "invalid_request_error"}}),
    lambda: _SdkError("Request timed out."),
], ids=["503", "500-even-if-typed-invalid", "timeout"])
async def test_transient_errors_keep_the_retry_ladder(harness, make_error):
    """Unchanged: three primary attempts with RETRY_DELAY then 2x, then the
    fallback model."""
    drive, sleeps = harness
    llm = _RefusingLLM(make_error)
    await drive(llm)
    assert llm.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK], llm.models
    assert sleeps == [0.25, 0.5]


# ── the classifier ────────────────────────────────────────────────────


def test_classifier_shapes():
    from app.agent.agent_runner import _is_invalid_request_400 as f

    assert f(_proxy_400())
    # OpenAI SDK body is the inner error object
    assert f(_SdkError("Error code: 400", status_code=400, body=_P27_BODY["error"]))
    # Anthropic SDK body
    assert f(_SdkError("400", status_code=400,
                       body={"type": "error", "error": {"type": "invalid_request_error",
                                                        "message": "bad"}}))
    # text-only fallback
    assert f(Exception("Error code: 400 - {'error': {'type': 'invalid_request_error'}}"))
    # never for a known non-400, a 400 of another type, or plain transport text
    assert not f(_SdkError("429 rate_limit invalid_request_error", status_code=429,
                           body={"error": {"type": "invalid_request_error"}}))
    assert not f(_SdkError("Error code: 400", status_code=400,
                           body={"detail": {"code": "hosted_tool_unsupported"}}))
    assert not f(Exception("APIConnectionError: connection reset"))
    assert not f(Exception("invalid_request_error but no status anywhere"))

    class _Hostile:
        status_code = 400
        body = {"detail": object()}

        def __str__(self):
            raise RuntimeError("boom")

    assert f(_Hostile()) is False
