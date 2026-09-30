"""The chat turn and its jobs under the proxy's monthly model budget refusal.

Lane: platform

Incident 2026-09-28, verified in production: the platform LLM proxy refused
every OpenAI call for one tenant with 429 "Monthly openai budget exceeded".
What the user got, and the piece of spec C that closes each:

* C1 — the turn died with ``UnboundLocalError: emitted_any`` (both OpenAI
  wires read a flag in their retry arms that was only assigned after
  ``create()``) and surfaced as "Something went wrong". A request-time 429 or
  connection error is a retry again; the budget refusal is raised at once.
* C2 — had it not crashed, the runner's ladder would have re-sent the refused
  call twice, slept, and hopped to the other provider through the same gate.
* C3 — the chat copy, where there was any, called it a rate limit.
* C4 — the ``create_job`` card closed as "the conversation ended … ask me to
  pick it up again", a promise the same gate refuses; job surfaces pushed the
  raw exception (cents, ISO timestamps, class names).

Fix pass v3 (E3) adds: the chat-intent ``job_update`` frames name the task;
the channels, system LLM calls, the retired app builder, routines and email
triggers treat the refusal as terminal and never store or serve its raw
text; a brand-new analysis while blocked gets exactly one reply; and an
ORDINARY request-time 429's retry multiplication is pinned end to end.

Fix pass v4 (F2) adds: Telegram's two agent-turn branches (and /compact,
/export) answer with the budget sentence and never the exception text;
``call_system_llm`` reports the refusal as ``model_budget`` with its reset
instead of swallowing it, so the email trigger's default summarize path and
the email briefing store the job sentence, not a "timeout"; every
``job_update`` frame that names a job also names its kind; and a dashboard
task's Logs line for the refusal is the sentence, not the raw 429. The
refusal there is the one the real openai SDK raises (``sdk_budget_429``).

BuildJob, JobEvent, DayChat, Conversation and Message are AGENT_ONLY tables,
so ``_reset_database`` below creates them on top of the lane's own schema.
All names, titles and filenames here are synthetic.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from openai import APIConnectionError, RateLimitError

from app.config import settings
from app.services import budget_refusal as br
from app.services import openai_agent_service as oas
from app.services.openai_agent_service import OpenAIAgentService

TORONTO = "America/Toronto"
RUNNER_SRC = (Path(oas.__file__).resolve().parents[1] / "agent" / "agent_runner.py").read_text()


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """The lane's schema plus the AGENT_ONLY tables these paths touch."""
    from app.db import drop_db, init_db
    from app.db.database import engine
    from app.db.models import BuildJob, Conversation, DayChat, JobEvent, Message

    await init_db()
    async with engine.begin() as conn:
        for table in (DayChat.__table__, Conversation.__table__, Message.__table__,
                      BuildJob.__table__, JobEvent.__table__):
            await conn.run_sync(lambda c, t=table: t.create(c, checkfirst=True))
    yield
    await drop_db()
    await engine.dispose()


# ── the refusal, as the SDK raises it ─────────────────────────────────


def _end_in(days: float, *, hour: int = 19, minute: int = 10) -> datetime:
    """An aware UTC ``period_end`` relative to the REAL clock — the copy is
    rendered against it, so a fixed date would rot into "no reset date"."""
    base = datetime.now(timezone.utc) + timedelta(days=days)
    return base.replace(hour=hour, minute=minute, second=40, microsecond=0)


def budget_429(period_end: object = "default", *, typed: bool = True) -> RateLimitError:
    """The proxy's 429 exactly as openai 2.53 hands it over (spec A2, with
    fix pass S2: no spend or budget figures in the detail)."""
    req = httpx.Request("POST", "http://platform.test/api/llm/openai/v1/chat/completions")
    if typed:
        end = _end_in(12) if period_end == "default" else period_end
        body = {"detail": {
            "error": br.REASON,
            "message": "Monthly openai budget exceeded",
            "provider": "openai",
            "period_start": br.iso_utc(end - timedelta(days=30)) if end else None,
            "period_end": br.iso_utc(end) if end else None,
        }}
        headers = {br.REASON_HEADER: br.REASON, "x-should-retry": "false",
                   "Retry-After": "604800"}
    else:  # a platform that predates the typed refusal
        body, headers = {"detail": "Monthly openai budget exceeded"}, {}
    resp = httpx.Response(429, request=req, json=body, headers=headers)
    return RateLimitError(f"Error code: 429 - {body}", response=resp, body=body)


def plain_429() -> RateLimitError:
    """An ordinary upstream rate limit — still a retry."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)


def conn_err() -> APIConnectionError:
    return APIConnectionError(
        request=httpx.Request("POST", "http://platform.test/api/llm/openai/v1/responses"))


def _assert_plain(copy: str) -> None:
    """User copy about the budget: no transport words, no paywall trigger,
    no cents, no timestamps, no exception names."""
    low = copy.lower()
    for banned in ("rate limit", "out of toup credits", "something went wrong",
                   "ratelimiterror", "error code", "429", "monthly_model_budget"):
        assert banned not in low, (banned, copy)
    assert "1003" not in copy and "1000" not in copy, copy
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d", copy), copy
    assert "+00:00" not in copy, copy


# ── C1: both OpenAI wires ─────────────────────────────────────────────


class _Outcomes:
    """`create()` raises / returns queued outcomes in order."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    async def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Empty:
    """A stream that ends at once — a successful, empty response."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


@pytest.fixture
def sleeps(monkeypatch):
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    monkeypatch.setattr(settings, "llm_stream_duplicate_guard", True, raising=False)
    import app.services.credit_reporter as cr
    monkeypatch.setattr(cr, "raise_if_exhausted", lambda *a, **k: None)
    seen = []

    async def no_sleep(delay, *a, **k):
        seen.append(delay)

    # The global asyncio.sleep; monkeypatch undoes it at teardown.
    monkeypatch.setattr(oas.asyncio, "sleep", no_sleep)
    return seen


WIRES = [pytest.param("chat", id="chat-wire"), pytest.param("responses", id="responses-wire")]


def _svc(wire: str, outcomes):
    svc = OpenAIAgentService.__new__(OpenAIAgentService)
    creates = _Outcomes(outcomes)
    if wire == "chat":
        svc.client = SimpleNamespace(chat=SimpleNamespace(completions=creates))
        svc.default_model, svc.default_max_tokens = "gpt-5.5", 256
    else:
        svc._responses_reasoning = {}
        svc.client = SimpleNamespace(responses=creates)
    svc._keys = SimpleNamespace(refresh=lambda: None)
    svc._ensure_client = lambda: None
    return svc, creates


async def _drain(wire: str, svc):
    messages = [{"role": "user", "content": "hi"}]
    if wire == "chat":
        stream = svc.create_message_stream(messages=messages, system="s")
    else:
        stream = svc._create_responses_stream(
            messages=messages, system="s", tools=None, model="gpt-5.6-terra",
            max_tokens=16)
    return [event async for event in stream]


@pytest.mark.parametrize("wire", WIRES)
async def test_a_request_time_429_is_retried_then_raised_as_itself(wire, sleeps):
    """Was `UnboundLocalError: emitted_any` on the first refusal."""
    svc, creates = _svc(wire, [plain_429(), plain_429(), plain_429()])
    with pytest.raises(RateLimitError):
        await _drain(wire, svc)
    assert creates.calls == 3 and sleeps == [1, 2]


@pytest.mark.parametrize("wire", WIRES)
async def test_a_request_time_429_then_success_recovers(wire, sleeps):
    svc, creates = _svc(wire, [plain_429(), _Empty()])
    events = await _drain(wire, svc)
    assert [e.type for e in events] == ["message_end"]
    assert creates.calls == 2 and sleeps == [1]


@pytest.mark.parametrize("wire", WIRES)
async def test_a_request_time_connection_error_is_retried_too(wire, sleeps):
    """The APIConnectionError arm read the same unbound flag."""
    svc, creates = _svc(wire, [conn_err(), _Empty()])
    events = await _drain(wire, svc)
    assert [e.type for e in events] == ["message_end"]
    assert creates.calls == 2 and sleeps == [1]


@pytest.mark.parametrize("typed", [True, False], ids=["typed", "legacy"])
@pytest.mark.parametrize("wire", WIRES)
async def test_a_budget_refusal_is_raised_at_once(wire, typed, sleeps):
    """One proxy call and no back-off: nothing changes until the window rolls."""
    refusal = budget_429(typed=typed)
    svc, creates = _svc(wire, [refusal, _Empty(), _Empty()])
    with pytest.raises(RateLimitError) as info:
        await _drain(wire, svc)
    assert info.value is refusal, "raised unconverted — ws_chat reads its detail"
    assert creates.calls == 1 and sleeps == []


# ── C2: the runner's retry ladder ─────────────────────────────────────


class RefusingLLM:
    def __init__(self):
        self.calls = []

    async def create_message_stream(self, **kwargs):
        self.calls.append(kwargs.get("model"))
        raise budget_429()
        yield  # pragma: no cover — makes this an async generator


async def _seed_user(tz: str | None = None) -> str:
    from app.db import User, async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"budget-{uid[:8]}@example.test",
                    hashed_password="x", timezone=tz))
        await db.commit()
    return uid


async def test_the_runner_raises_a_budget_refusal_after_one_call(monkeypatch, tmp_path):
    import app.agent.agent_runner as ar
    from app.agent.tool_executor import ToolExecutor
    from app.services import key_provider

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    sleeps, errors = [], []

    async def no_sleep(delay, *a, **k):
        sleeps.append(delay)

    async def record_error(self, **kw):
        errors.append(kw)

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar.AgentRunner, "_log_error", record_error)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)
    monkeypatch.setattr(ar.asyncio, "sleep", no_sleep)
    # Both providers keyed, so the fallback arm WOULD have a hop to make.
    monkeypatch.setattr(type(key_provider.keys), "has_openai", property(lambda self: True))
    monkeypatch.setattr(type(key_provider.keys), "has_anthropic", property(lambda self: True))

    uid = await _seed_user()
    llm = RefusingLLM()
    runner = ar.AgentRunner(llm_service=llm, tool_executor=ToolExecutor(workspace=str(tmp_path)))
    runner.anthropic = llm
    with pytest.raises(RateLimitError) as info:
        await runner.run(
            user_message="hi", user_id=uid, session_id=str(uuid.uuid4()),
            channel="app", model_override="gpt-5.5",
            save_user_message=False, save_assistant_message=False,
            disable_post_processing=True,
        )
    assert br.is_budget_refusal(info.value)
    assert llm.calls == ["gpt-5.5"], "retried, or hopped providers into the same gate"
    assert sleeps == []
    assert [e["error_type"] for e in errors] == ["llm_error"]
    assert errors[0]["context"]["terminal"] == "model_budget_exceeded"


def _handler() -> str:
    a = RUNNER_SRC.index("# Detect errors that warrant immediate cross-provider fallback")
    return RUNNER_SRC[a:RUNNER_SRC.index("if attempt < MAX_RETRIES:", a)]


def test_the_runner_budget_branch_is_ordered_before_the_hop():
    h = _handler()
    assert h.index("if _is_out_of_credits:") < h.index("budget_refusal(e)") \
        < h.index("_should_cross_provider =")


# ── C3: the chat bubble ───────────────────────────────────────────────


def test_the_bubble_names_the_reset_in_the_users_zone():
    from app.api.ws_chat import _friendly_error

    end = _end_in(12, hour=2, minute=30)   # 02:30 UTC: the evening before in Toronto
    exc = budget_429(end)
    local, utc = br.reset_when_phrase(end, TORONTO), br.reset_when_phrase(end, None)
    assert local != utc, "the fixture must straddle Toronto's midnight"

    msg = _friendly_error(exc, TORONTO)
    assert msg == br.chat_sentence(br.budget_refusal_detail(exc), TORONTO)
    assert f"resets on {local}." in msg and utc not in msg
    assert msg.startswith("Your agent’s monthly AI budget is used up")
    assert msg.endswith("Your credits aren’t affected.")
    _assert_plain(msg)
    # No zone: the same sentence, dated in UTC.
    assert _friendly_error(exc) == br.chat_sentence(br.budget_refusal_detail(exc), None)


def test_a_refusal_with_no_future_reset_gets_the_undated_sentence():
    from app.api.ws_chat import _friendly_error

    undated = ("Your agent’s AI budget is used up, so it can’t reply right now. "
               "Your credits aren’t affected.")
    assert _friendly_error(budget_429(typed=False), TORONTO) == undated   # legacy platform
    assert _friendly_error(budget_429(None)) == undated                   # no window end
    assert _friendly_error(budget_429(_end_in(-1))) == undated            # never a past date


def test_ordinary_errors_keep_their_copy():
    from app.api.ws_chat import _friendly_error

    assert _friendly_error(plain_429()).startswith("Rate limit reached")
    assert _friendly_error(RuntimeError("rate_limit exceeded")).startswith("Rate limit reached")
    assert "Toup credits" in _friendly_error(RuntimeError("insufficient_quota"))


def test_the_budget_branch_runs_before_the_keyword_buckets_and_the_429():
    from app.api import ws_chat

    src = inspect.getsource(ws_chat._friendly_error)
    i_budget = src.index("is_budget_refusal(exc)")
    assert src.index("REASON_DAILY_CAP_EXCEEDED") < i_budget, \
        "the structured credit block must stay first"
    assert i_budget < src.index('"out of extra"')
    assert i_budget < src.index('"insufficient_quota"')
    assert i_budget < src.index("status == 429")


def test_the_error_frame_is_plain_and_carries_the_budget_fields():
    """Pinned in the source because the handler needs a live socket; the
    behaviour of the fields themselves is asserted below it."""
    from app.api import ws_chat
    from app.api.ws_chat import _friendly_error

    src = inspect.getsource(ws_chat)
    block = src.split("user_msg = _friendly_error(e)")[1]
    i_plain = block.index('{"type": "error", "message": user_msg}')
    i_fields = block.index("_err_frame.update(_budget_frame_fields(_budget_detail))")
    assert i_plain < i_fields < block.index("await _safe_send(_err_frame)")

    end = _end_in(12)
    exc = budget_429(end)
    assert br.frame_fields(br.budget_refusal_detail(exc)) == {
        "code": "monthly_model_budget_exceeded", "retryable": False,
        "resets_at": br.iso_utc(end),
    }
    # The credit_exhausted route keys on this phrase; the budget copy never has it.
    assert "out of Toup credits" not in _friendly_error(exc)


def test_the_turns_job_row_and_push_carry_the_budget_sentence():
    """Pinned in the source for the same reason as the frame. The chat-intent
    job keeps the raw text for operators and gains the two fields a client
    renders (and the live frame that carries them); a gone client's push
    says when the budget resets instead of "Something went wrong"."""
    from app.api import ws_chat

    src = inspect.getsource(ws_chat)
    block = src[src.index("_budget_refused = is_budget_refusal(e)"):
                src.index("user_msg = _friendly_error(e)")]
    assert block.index("_fj.error_message = str(e)[:500]") \
        < block.index("_fj.error_class = _BUDGET_ERROR_CLASS") \
        < block.index("await _fdb.commit()")
    assert "_fj.user_message = _budget_job_msg" in block
    frame = block[block.index('"type": "job_update"'):]
    assert '"user_message": _budget_job_msg' in frame
    assert '"error_class": _BUDGET_ERROR_CLASS' in frame
    assert "_budget_push_body(_budget_detail, _copy_tz)" in block


def _chat_task_frames() -> dict:
    """status → the source of each chat-intent `job_update` frame literal."""
    from app.api import ws_chat

    src = inspect.getsource(ws_chat.ws_chat)
    frames = {}
    at = 0
    while True:
        at = src.find('"job_id": _chat_task_job_id,', at)
        if at < 0:
            return frames
        start = src.rfind("{", 0, at)
        end = src.index("})", at)
        literal = src[start:end + 1]
        status = re.search(r'"status": "(\w+)"', literal).group(1)
        frames[status] = literal
        at = end


def test_every_chat_intent_frame_names_the_task():
    """A client can meet a chat-intent job first in one of these frames (the
    web never handles `task_created`), and a frame with neither `name` nor
    `job_type` was drawn as "Couldn't build App Build" — for a task that was
    never a build. Pinned in the source: the frames need a live socket."""
    frames = _chat_task_frames()
    assert set(frames) == {"completed", "cancelled", "failed"}, sorted(frames)
    for status, literal in frames.items():
        assert '"type": "job_update"' in literal, status
        assert '"name": _chat_task_name' in literal, status
        assert '"job_type": _CHAT_TASK_JOB_TYPE' in literal, status


def test_the_frame_name_is_the_title_the_job_was_created_with():
    from app.api import ws_chat

    src = inspect.getsource(ws_chat.ws_chat)
    detect = src.index("_chat_task_job_id = await _detect_and_create_task(")
    assert "_chat_task_name = _chat_task_title(text)" in src[detect:detect + 400]
    assert ws_chat._CHAT_TASK_JOB_TYPE == "agent_task"
    creator = inspect.getsource(ws_chat._detect_and_create_task)
    assert "title = _chat_task_title(text)" in creator
    assert "job_type=_CHAT_TASK_JOB_TYPE" in creator
    text = "  Write me a short summary of the budget refusal copy for the release notes  "
    assert ws_chat._chat_task_title(text) == text.strip()[:60]


async def test_the_created_task_and_its_frames_share_title_and_type():
    """Behavioural half: the row `_detect_and_create_task` writes has the
    title and job_type the frames name."""
    from app.api import ws_chat

    uid = await _seed_user()
    queue: asyncio.Queue = asyncio.Queue()
    text = "Draft me a proposal outline for the spring workshop series"
    jid = await ws_chat._detect_and_create_task(text, uid, str(uuid.uuid4()), queue)
    assert jid
    row = await _row(jid)
    assert row.title == ws_chat._chat_task_title(text)
    assert row.job_type == ws_chat._CHAT_TASK_JOB_TYPE


async def test_the_bubble_zone_is_the_turns_then_the_stored_one():
    from app.api.ws_chat import _budget_copy_tz

    uid = await _seed_user(tz="Asia/Tehran")
    assert await _budget_copy_tz("Europe/Paris", uid) == "Europe/Paris"
    assert await _budget_copy_tz(None, uid) == "Asia/Tehran"
    assert await _budget_copy_tz(None, str(uuid.uuid4())) is None


# ── C4: job status taxonomy ───────────────────────────────────────────


#: Fix pass S1: "finish", not "run" — the refused job often did work first.
UNDATED_JOB_SENTENCE = ("This task couldn’t finish because your agent’s AI budget is "
                        "used up. Your credits aren’t affected.")


def test_the_budget_rule_is_terminal_and_beats_every_status_code_rule():
    from app.agent import job_status as js

    assert js.ERR_MODEL_BUDGET == br.ERROR_CLASS == "model_budget"
    typed = budget_429(_end_in(12))
    for raw in (
        typed,
        repr(typed),
        "Error code: 429 - {'detail': 'Monthly openai budget exceeded'}",
        "Error code: 429 - {'detail': 'Monthly Anthropic budget exceeded'}",
        # the trigger runner's exhausted row: a prefix, then repr(e)
        f"all_retries_exhausted: {typed!r}",
        # numbers `connector_auth` ("401") and `upstream` (5xx) would match
        "Error code: 429 - {'detail': {'error': 'monthly_model_budget_exceeded', "
        "'provider': 'openai', 'period_start': '2026-04-01T14:01:40+00:00', "
        "'period_end': '2026-05-01T14:01:40+00:00'}} request 401-503",
    ):
        verdict = js.classify(raw)
        assert verdict.error_class == js.ERR_MODEL_BUDGET, raw
        assert verdict.disposition == js.DISPOSITION_TERMINAL
        assert not js.is_retryable(raw)
        _assert_plain(verdict.user_message)
    for raw in ("Rate limit exceeded for this tenant token.",   # the G-20 limiter
                "Error code: 429 - rate_limit_exceeded", plain_429()):
        assert js.classify(raw).error_class == js.ERR_RATE_LIMITED, raw


def test_the_budget_rule_copy_is_the_undated_job_sentence():
    """S1, and the same sentence the web/app fallbacks show."""
    from app.agent import job_status as js

    verdict = js.classify(budget_429(_end_in(12)))
    assert verdict.user_message == UNDATED_JOB_SENTENCE == br.job_sentence(None)


@pytest.mark.parametrize("raw", [
    # the marker quoted in third-party text an exception carried along (S3)
    RuntimeError("summarizer failed on body: {'detail': {'error': "
                 "'monthly_model_budget_exceeded', 'provider': 'openai', "
                 "'period_end': '2099-01-01T00:00:00'}}"),
    "email body: {'error': 'monthly_model_budget_exceeded', 'provider': 'openai'}",
    ValueError("Customer wrote: our Monthly OpenAI budget exceeded the plan"),
    "Monthly Anthropic budget exceeded",
    # our own sentence, but not at the head of the text
    "Customer wrote: " + UNDATED_JOB_SENTENCE,
], ids=["exc-marker", "text-marker", "exc-legacy", "bare-legacy", "quoted-sentence"])
def test_third_party_text_carrying_the_marker_is_not_a_budget_stop(raw):
    from app.agent import job_status as js

    verdict = js.classify(raw)
    assert verdict.error_class != js.ERR_MODEL_BUDGET, raw
    assert js.model_budget_verdict(raw) is None


@pytest.mark.parametrize("stored", [
    UNDATED_JOB_SENTENCE,
    # dated, as a routine/trigger handler stores it
    "This task couldn’t finish because your agent’s monthly AI budget is used up. "
    "It resets on October 24. Your credits aren’t affected.",
    # the routine runner's retry gate: f"{error_class}: {error_detail}"
    "model_budget: " + UNDATED_JOB_SENTENCE,
    # the retired app builder's ModelBudgetStop, classified as an exception
    RuntimeError(UNDATED_JOB_SENTENCE),
], ids=["undated", "dated", "retry-gate", "exception"])
def test_our_own_job_sentence_stays_a_terminal_budget_stop(stored):
    """Routine and trigger handlers now store the sentence instead of the
    raw 429. Read as `unknown` it would be RETRY in the routine runner's
    gate and "Something went wrong" (with a retry) on the jobs API."""
    from app.agent import job_status as js

    verdict = js.classify(stored)
    assert verdict.error_class == js.ERR_MODEL_BUDGET, stored
    assert verdict.disposition == js.DISPOSITION_TERMINAL


def test_the_dated_verdict_names_the_local_date_and_nothing_else():
    from app.agent.job_status import ERR_MODEL_BUDGET, model_budget_verdict

    end = _end_in(12, hour=2, minute=30)
    refusal = budget_429(end)
    verdict = model_budget_verdict(refusal, TORONTO)
    assert verdict.error_class == ERR_MODEL_BUDGET
    assert verdict.user_message == br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    assert br.reset_when_phrase(end, TORONTO) in verdict.user_message
    _assert_plain(verdict.user_message)
    assert model_budget_verdict(plain_429()) is None
    assert model_budget_verdict(RuntimeError("boom")) is None


async def test_job_copy_zone_is_the_cache_then_the_users_row():
    from app.agent._user_tz_cache import invalidate_cached_user_tz, set_cached_user_tz
    from app.agent.job_status import user_tz_name

    uid = await _seed_user(tz="Asia/Tehran")
    assert await user_tz_name(uid) == "Asia/Tehran"
    set_cached_user_tz(uid, "Europe/Paris")
    try:
        assert await user_tz_name(uid) == "Europe/Paris"
    finally:
        invalidate_cached_user_tz(uid)
    assert await user_tz_name(None) is None
    assert await user_tz_name(str(uuid.uuid4())) is None


# ── C4: the job surfaces ──────────────────────────────────────────────


@pytest.fixture
def pushes(monkeypatch):
    calls = []

    async def fake_notify(**kwargs):
        calls.append(kwargs)
        return "nid"

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", fake_notify)
    return calls


@pytest.fixture
def frames(monkeypatch):
    seen = []

    async def fake_broadcast(user_id, event, *a, **k):
        seen.append(event)
        return 1

    import app.api.ws_chat as wsc
    monkeypatch.setattr(wsc, "broadcast_to_user", fake_broadcast)
    return seen


async def _seed_job(uid: str, *, title: str, job_type: str = "agent_task",
                    status: str = "running", **columns) -> str:
    from app.db import async_session_maker
    from app.db.models import BuildJob

    jid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(id=jid, user_id=uid, title=title, prompt="p",
                        job_type=job_type, status=status, **columns))
        await db.commit()
    return jid


async def _row(jid: str):
    from app.db import async_session_maker
    from app.db.models import BuildJob

    async with async_session_maker() as db:
        return await db.get(BuildJob, jid)


async def test_a_job_the_budget_stopped_says_so_on_the_row_and_the_push(pushes):
    from app.agent.job_runner import JobRunner, TaskSpec

    uid = await _seed_user(tz=TORONTO)
    end = _end_in(12, hour=2, minute=30)
    refusal = budget_429(end)

    async def refused(job, spec, _unused):
        raise refusal

    original = dict(JobRunner.HANDLERS)
    try:
        JobRunner.register("agent_task", refused)
        runner = JobRunner()
        spec = TaskSpec(user_id=uid, channel="web", source_kind="manual")
        job = await runner.create_job(job_type="agent_task", spec=spec,
                                      title="Draft proposal", status="running")
        with pytest.raises(RateLimitError):
            await runner.execute(job, spec)
    finally:
        JobRunner.HANDLERS.clear()
        JobRunner.HANDLERS.update(original)

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    assert br.reset_when_phrase(end, TORONTO) in sentence
    row = await _row(job.id)
    assert row.status == "failed" and row.error_message
    assert row.error_class == "model_budget" and row.user_message == sentence
    [push] = pushes
    assert push["event_kind"] == "mission_failed" and push["body"] == sentence
    for copy in (row.user_message, push["body"]):
        _assert_plain(copy)


async def test_an_interrupted_job_closed_by_the_budget_says_why(pushes, frames):
    """The incident's card shape: "Couldn't build …" with "Try again", over
    a push saying the conversation ended."""
    from app.agent.agent_runner import AgentRunner

    uid = await _seed_user(tz=TORONTO)
    jid = await _seed_job(uid, title="Draft proposal")
    refusal = budget_429(_end_in(12, hour=2, minute=30))
    runner = AgentRunner.__new__(AgentRunner)
    await runner._close_interrupted_jobs((jid,), uid, cause=refusal)

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    row = await _row(jid)
    assert row.status == "cancelled"
    assert row.error_class == "model_budget" and row.user_message == sentence
    [frame] = [f for f in frames if f.get("job_id") == jid]
    assert frame["user_message"] == sentence and frame["error_class"] == "model_budget"
    [push] = pushes
    assert push["event_kind"] == "mission_failed"
    assert push["title"] == "Couldn’t finish: Draft proposal"
    assert push["body"] == sentence
    _assert_plain(push["body"])


@pytest.mark.parametrize("cause", [None, RuntimeError("boom")], ids=["no-cause", "other-cause"])
async def test_an_ordinary_interrupted_job_keeps_its_story(pushes, frames, cause):
    from app.agent.agent_runner import AgentRunner
    from app.agent.job_status import ERR_TURN_INTERRUPTED, turn_interrupted

    uid = await _seed_user()
    jid = await _seed_job(uid, title="Check recent emails")
    await AgentRunner.__new__(AgentRunner)._close_interrupted_jobs((jid,), uid, cause=cause)
    row = await _row(jid)
    assert row.error_class == ERR_TURN_INTERRUPTED
    assert row.user_message == turn_interrupted().user_message
    [push] = pushes
    assert push["title"] == "Stopped: Check recent emails"
    assert push["body"].startswith("The conversation ended before this finished.")


async def test_a_build_closed_by_the_budget_settles_with_the_sentence(pushes, frames):
    from app.agent.agent_runner import AgentRunner
    from app.agent.skills.builtins.app_html import steps as steps_mod

    uid = await _seed_user(tz=TORONTO)
    jid = await _seed_job(
        uid, title="Build: Habit Garden", job_type="auto_builder",
        app_id=str(uuid.uuid4()), steps_json=json.dumps(steps_mod.initial_steps()),
        # a phase message already on the row: the caller's reason must win
        user_message="The app file couldn't be saved. Try again.",
    )
    refusal = budget_429(_end_in(12, hour=2, minute=30))
    await AgentRunner.__new__(AgentRunner)._close_interrupted_jobs((jid,), uid, cause=refusal)

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    row = await _row(jid)
    assert row.status == "failed"
    assert row.error_class == "model_budget" and row.user_message == sentence
    frame = [f for f in frames if f.get("job_id") == jid][-1]
    assert frame["user_message"] == sentence and frame["error_class"] == "model_budget"
    push = [p for p in pushes if p["event_kind"] == "mission_failed"][-1]
    assert push["body"] == sentence


async def test_a_rebuild_clears_a_stale_budget_sentence(pushes, frames):
    """One card per app: a later rebuild of the same row must not keep saying
    the budget is used up."""
    from app.agent.skills.builtins.app_html import steps as steps_mod

    uid = await _seed_user()
    jid = await _seed_job(
        uid, title="Build: Habit Garden", job_type="auto_builder", status="failed",
        app_id=str(uuid.uuid4()), steps_json=json.dumps(steps_mod.initial_steps()),
        error_class="model_budget",
        user_message=br.job_sentence(None),
    )
    await steps_mod.emit_step(user_id=uid, job_id=jid, step_type="create", status="running")
    row = await _row(jid)
    assert row.status == "running"
    assert row.error_class is None and row.user_message is None


async def test_a_subagent_the_budget_stopped_posts_the_sentence_not_the_exception(
    monkeypatch, pushes, frames,
):
    import app.agent.lanes as lanes
    import app.agent.subagent_message_writer as mw
    from app.agent import subagent_orchestrator as so
    from app.db.database import async_session_maker

    monkeypatch.setattr(lanes, "_lane_manager", None)
    posted = []

    async def fake_write(db, **kwargs):
        posted.append(kwargs)
        return ("msg-budget", "dc-budget")

    async def fake_broadcast(user_id, **kwargs):
        return {}

    monkeypatch.setattr(mw, "write_subagent_message", fake_write)
    monkeypatch.setattr(mw, "broadcast_subagent_message", fake_broadcast)

    uid = await _seed_user(tz=TORONTO)
    jid = await _seed_job(uid, title="Draft proposal",
                          job_type="subagent", status="queued")
    refusal = budget_429(_end_in(12, hour=2, minute=30))

    class _Refusing:
        async def run(self, **kwargs):
            raise refusal

    await so._run_child(
        job_id=jid, user_id=uid, task="Revise the draft proposal",
        label="Draft proposal", model=None, timeout_seconds=30,
        telegram_chat_id=None, parent_job_id=None, agent_runner=_Refusing(),
        session_maker=async_session_maker,
    )

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    [post] = posted
    assert post["content"] == sentence and post["outcome"] == "failed"
    row = await _row(jid)
    assert row.status == "failed" and row.error_message   # raw text kept for operators
    assert row.error_class == "model_budget" and row.user_message == sentence
    failed = [p for p in pushes if p["event_kind"] == "mission_failed"]
    assert failed and failed[-1]["body"] == sentence
    terminal = [f for f in frames if f.get("job_id") == jid and f.get("status") == "failed"]
    assert terminal[-1]["user_message"] == sentence
    assert terminal[-1]["error_class"] == "model_budget"


# ── E3: the surfaces outside the chat turn ────────────────────────────


def _recording_channel():
    """A BaseChannel that records what it sends (pattern:
    tests/test_whatsapp_message_handler.py)."""
    from app.agent.channels.base import BaseChannel, ChannelType

    class _Recording(BaseChannel):
        def __init__(self):
            super().__init__(ChannelType.WHATSAPP)
            self.sent = []

        async def start(self):
            return None

        async def stop(self):
            return None

        async def send_text(self, chat_id, text, parse_mode=None):
            self.sent.append(text)

        async def send_typing(self, chat_id):
            return None

    return _Recording()


class _RaisingRunner:
    def __init__(self, exc: BaseException):
        self.exc, self.calls = exc, 0

    async def run(self, **kwargs):
        self.calls += 1
        raise self.exc


async def _channel_turn(exc: BaseException, *, tz: str | None) -> list[str]:
    from app.agent._user_tz_cache import invalidate_cached_user_tz, set_cached_user_tz
    from app.agent.channels.base import ChannelType, InboundMessage
    from app.agent.channels.shared.message_handler import make_channel_handler

    uid = str(uuid.uuid4())
    if tz:
        set_cached_user_tz(uid, tz)
    try:
        channel = _recording_channel()
        handler = make_channel_handler(
            channel=channel, agent_runner=_RaisingRunner(exc), user_id=uid)
        await handler(InboundMessage(
            channel=ChannelType.WHATSAPP, channel_user_id="+15550100",
            channel_chat_id="+15550100", text="hi"))
    finally:
        invalidate_cached_user_tz(uid)
    return channel.sent


@pytest.mark.parametrize("typed", [True, False], ids=["typed", "legacy"])
async def test_a_channel_turn_the_budget_refused_says_so(typed):
    """Telegram/WhatsApp/Slack/Discord answered "Please try again in a
    moment" — false until the budget resets."""
    end = _end_in(12, hour=2, minute=30)
    refusal = budget_429(end) if typed else budget_429(typed=False)
    [sent] = await _channel_turn(refusal, tz=TORONTO)
    assert sent == br.chat_sentence(br.budget_refusal_detail(refusal), TORONTO)
    if typed:   # dated in the zone the runner cached for this user
        assert f"resets on {br.reset_when_phrase(end, TORONTO)}." in sent
    assert "try again" not in sent.lower()
    _assert_plain(sent)


async def test_a_channel_budget_refusal_with_a_cold_zone_cache_is_dated_in_utc():
    end = _end_in(12, hour=2, minute=30)
    refusal = budget_429(end)
    [sent] = await _channel_turn(refusal, tz=None)
    assert sent == br.chat_sentence(br.budget_refusal_detail(refusal), None)


async def test_any_other_channel_failure_keeps_the_generic_apology():
    from app.agent.channels.shared.message_handler import _GENERIC_ERROR_TEXT

    for exc in (RuntimeError("internal kaboom"), plain_429()):
        assert await _channel_turn(exc, tz=TORONTO) == [_GENERIC_ERROR_TEXT]


class _Completions:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    async def create(self, **kwargs):
        self.calls += 1
        raise self.outcomes.pop(0)


def _system_llm(outcomes):
    from app.services.llm_service import LLMService

    svc = LLMService.__new__(LLMService)
    completions = _Completions(outcomes)
    svc._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return svc, completions


@pytest.mark.parametrize("typed", [True, False], ids=["typed", "legacy"])
async def test_a_system_llm_call_raises_a_budget_refusal_at_once(typed, sleeps):
    """Memory extraction and the other system calls retried the refusal
    three times with back-off, ignoring `x-should-retry: false`."""
    refusal = budget_429(typed=typed)
    svc, completions = _system_llm([refusal, plain_429(), plain_429()])
    with pytest.raises(RateLimitError) as info:
        await svc._complete_openai([{"role": "user", "content": "hi"}], "gpt-4o-mini", 0.2, 64)
    assert info.value is refusal
    assert completions.calls == 1 and sleeps == []


async def test_a_system_llm_call_still_retries_an_ordinary_429(sleeps):
    svc, completions = _system_llm([plain_429(), plain_429(), plain_429()])
    with pytest.raises(RateLimitError):
        await svc._complete_openai([{"role": "user", "content": "hi"}], "gpt-4o-mini", 0.2, 64)
    assert completions.calls == 3 and sleeps == [1.0, 2.0]


# ── E3: the retired app builder ───────────────────────────────────────


def _anthropic_budget_429(end: datetime | None = None):
    """The same proxy refusal as the anthropic SDK raises it."""
    import anthropic

    end = end or _end_in(12)
    body = {"detail": {
        "error": br.REASON, "message": "Monthly anthropic budget exceeded",
        "provider": "anthropic",
        "period_start": br.iso_utc(end - timedelta(days=30)), "period_end": br.iso_utc(end),
    }}
    req = httpx.Request("POST", "http://platform.test/api/llm/anthropic/v1/messages")
    resp = httpx.Response(429, request=req, json=body, headers={
        br.REASON_HEADER: br.REASON, "x-should-retry": "false", "retry-after": "604800"})
    return anthropic.RateLimitError(f"Error code: 429 - {body}", response=resp, body=body)


def _builder():
    from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill

    return object.__new__(AppBuilderSkill)


def _refusing_openai_client(exc):
    completions = _Completions([exc])
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _refusing_anthropic_client(exc, calls):
    class _Stream:
        async def __aenter__(self):
            raise exc

        async def __aexit__(self, *a):
            return False

    def stream(**kwargs):
        calls.append(kwargs.get("model"))
        return _Stream()

    return SimpleNamespace(messages=SimpleNamespace(stream=stream))


async def test_the_app_builder_never_pauses_on_a_budget_refusal(monkeypatch):
    """Both arms converted the refusal into TokenLimitError — "OpenAI rate
    limit reached — retrying in 60s", and on the Anthropic arm a pause for
    the refusal's Retry-After (up to 7 days). It is the job sentence now."""
    import app.services.bundle_client as bc
    from app.agent.skills.builtins.app_builder import skill as sk

    refusal = budget_429()
    client, completions = _refusing_openai_client(refusal)
    monkeypatch.setattr(bc, "make_openai_client", lambda byok_key=None: client)
    with pytest.raises(sk.ModelBudgetStop) as info:
        await _builder()._call_openai("system", "user", openai_key="k", max_tokens=64)
    assert isinstance(info.value, RuntimeError)
    assert not isinstance(info.value, sk.TokenLimitError)
    assert str(info.value) == br.job_sentence(br.budget_refusal_detail(refusal))
    assert completions.calls == 1
    _assert_plain(str(info.value))

    anthropic_refusal, calls = _anthropic_budget_429(), []
    monkeypatch.setattr(bc, "make_anthropic_client",
                        lambda byok_key=None: _refusing_anthropic_client(anthropic_refusal, calls))
    with pytest.raises(sk.ModelBudgetStop) as info:
        await _builder()._call_anthropic("system", "user", model="claude-opus-4-7",
                                         anthropic_key="k", max_tokens=64)
    assert str(info.value) == br.job_sentence(br.budget_refusal_detail(anthropic_refusal))
    assert calls == ["claude-opus-4-7"]


async def test_the_app_builder_does_not_hop_providers_on_a_budget_refusal(monkeypatch):
    import app.services.bundle_client as bc
    from app.agent.skills.builtins.app_builder import skill as sk
    from app.services import key_provider

    monkeypatch.setattr(type(key_provider.keys), "anthropic", property(lambda self: "k-a"))
    monkeypatch.setattr(type(key_provider.keys), "openai", property(lambda self: "k-o"))
    anthropic_calls = []
    monkeypatch.setattr(bc, "make_anthropic_client", lambda byok_key=None:
                        _refusing_anthropic_client(_anthropic_budget_429(), anthropic_calls))
    openai_client, completions = _refusing_openai_client(RuntimeError("must not be called"))
    monkeypatch.setattr(bc, "make_openai_client", lambda byok_key=None: openai_client)

    with pytest.raises(sk.ModelBudgetStop):
        await _builder()._call_llm("system", "user", model="claude-opus-4-7")
    assert anthropic_calls == ["claude-opus-4-7"]
    assert completions.calls == 0, "fell back to the other provider on a budget refusal"


def test_the_app_builder_budget_checks_run_before_any_pause_or_fallback():
    """Source order, since the pipeline is retired (EXPO_PIPELINE_RETIRED)."""
    from app.agent.skills.builtins.app_builder import skill as sk

    anthropic_arm = inspect.getsource(sk.AppBuilderSkill._call_anthropic)
    arm = anthropic_arm[anthropic_arm.index("except anthropic.RateLimitError as e:"):]
    assert arm.index("if is_budget_refusal(e):") < arm.index("raise TokenLimitError(")
    assert "raise _model_budget_stop(e) from e" in arm[:arm.index("raise TokenLimitError(")]

    openai_arm = inspect.getsource(sk.AppBuilderSkill._call_openai)
    arm = openai_arm[openai_arm.index("except RateLimitError as e:"):]
    assert arm.index("if is_budget_refusal(e):") < arm.index("raise TokenLimitError(")

    router = inspect.getsource(sk.AppBuilderSkill._call_llm)
    first = router.index("except ModelBudgetStop:")
    assert first < router.index("except TokenLimitError:") < router.index("except Exception as _anth_err:")
    second = router.index("except ModelBudgetStop:", first + 1)
    assert second < router.index("except Exception as _oai_err:")


# ── E3: jobs, routines and triggers never serve the raw 429 ───────────


def _raw_budget_text() -> str:
    """What the writers store: str(e) of the typed refusal (S2 shape)."""
    return str(budget_429(_end_in(12)))[:500]


def _job(**columns):
    from app.db.models import BuildJob

    payload = dict(id=str(uuid.uuid4()), user_id="u1", title="Draft proposal", prompt="p",
                   job_type="agent_task", status="failed", created_at=datetime.utcnow())
    payload.update(columns)
    return BuildJob(**payload)


def test_the_jobs_api_serves_the_sentence_not_the_raw_429():
    from app.api.apps import _job_to_response

    raw = _raw_budget_text()
    assert "Error code: 429" in raw and "+00:00" in raw   # the leak being closed

    # A row with only the raw text (classified on read).
    legacy = _job_to_response(_job(error_message=raw))
    assert legacy.error_class == "model_budget"
    assert legacy.error_message == legacy.user_message == UNDATED_JOB_SENTENCE

    # A row whose writer stored the class and the dated sentence.
    dated = br.job_sentence(br.budget_refusal_detail(budget_429(_end_in(12))), TORONTO)
    stored = _job_to_response(_job(error_message=raw, error_class="model_budget",
                                   user_message=dated))
    assert stored.error_message == stored.user_message == dated

    # A routine/trigger row: the handler stored the sentence itself. It is
    # ours, so it is served as is and keeps its reset date (as the Routines
    # and Triggers panels serve it).
    handler_row = _job_to_response(_job(error_message=dated))
    assert handler_row.error_class == "model_budget"
    assert handler_row.error_message == handler_row.user_message == dated
    assert "It resets on" in dated

    for response in (legacy, stored, handler_row):
        _assert_plain(response.error_message)


def test_the_jobs_api_keeps_other_errors_for_legacy_clients():
    from app.api.apps import _job_to_response

    raw = "Error code: 402 - {'detail': {'error': 'out_of_credits'}}"
    response = _job_to_response(_job(error_message=raw))
    assert response.error_class == "credits_toup"
    assert response.error_message == raw   # unchanged: not a budget stop
    assert _job_to_response(_job(status="completed")).error_message is None


def test_the_triggers_api_serves_the_sentence_not_the_raw_429():
    from app.api.triggers import _job_to_event_response, _row_to_response

    raw = _raw_budget_text()
    now = datetime.utcnow()
    event = SimpleNamespace(
        id="ev-1", source_id="trig-1", idempotency_key="gm-1", created_at=now,
        completed_at=now, status="failed", outcome=None, error_message=raw,
        summary_message_id=None, coalesced_into_job_id=None, title="Inbox",
    )
    trigger = SimpleNamespace(
        id="trig-1", kind="email_received", action="summarize_and_post", name="Inbox",
        enabled=True, filter_json=None, config_json={}, provider_state_json={},
        last_fired_at=now, fire_count=1, last_status="failed", last_error=raw,
        created_at=now, updated_at=now,
    )
    served_event = _job_to_event_response(event)
    served = _row_to_response(trigger, recent_events=[event])
    assert served_event.error_detail == UNDATED_JOB_SENTENCE
    assert served.last_error == UNDATED_JOB_SENTENCE
    assert served.recent_events[0].error_detail == UNDATED_JOB_SENTENCE
    # The stored text is untouched (operators read it from the row).
    assert trigger.last_error == raw and event.error_message == raw

    other = "RuntimeError('internal_llm returned None (timeout / auth / parse failure)')"
    trigger.last_error, event.error_message = other, other
    assert _row_to_response(trigger).last_error == other
    assert _job_to_event_response(event).error_detail == other
    trigger.last_error = None
    assert _row_to_response(trigger).last_error is None


async def test_a_routine_the_budget_refused_stores_the_sentence(monkeypatch):
    from app.agent.job_status import DISPOSITION_TERMINAL, classify
    from app.agent.routines.agent_task_handler import AgentTaskHandler

    uid = await _seed_user(tz=TORONTO)
    end = _end_in(12, hour=2, minute=30)
    refusal = budget_429(end)
    runner = _RaisingRunner(refusal)
    routine = SimpleNamespace(id="rt-1", user_id=uid, prompt_text="Check my inbox",
                              config_json={})
    result = await AgentTaskHandler(agent_runner=runner)._run_via_agent_runner(
        routine, runner, "Check my inbox", db=None)

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    assert result.status == "failed"
    assert result.error_class == "model_budget"
    assert result.error_detail == sentence
    assert br.reset_when_phrase(end, TORONTO) in sentence
    _assert_plain(result.error_detail)
    # The routine runner's retry gate composes exactly this: terminal, so no
    # second and third attempt into the same refusal.
    gate = classify(f"{result.error_class or ''}: {result.error_detail or ''}")
    assert gate.error_class == "model_budget" and gate.disposition == DISPOSITION_TERMINAL

    boom = _RaisingRunner(RuntimeError("boom"))
    other = await AgentTaskHandler(agent_runner=boom)._run_via_agent_runner(
        routine, boom, "Check my inbox", db=None)
    assert (other.error_class, other.error_detail) == ("RuntimeError", "boom")


async def test_an_email_trigger_the_budget_refused_stores_the_sentence(monkeypatch):
    """The runner path must not fall back to the summarizer (the same proxy,
    or another provider's budget), and the result carries the sentence."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler, _FetchedEmail

    monkeypatch.setattr(settings, "trigger_turns_via_runner", True, raising=False)
    uid = await _seed_user(tz=TORONTO)
    end = _end_in(12, hour=2, minute=30)
    refusal = budget_429(end)
    runner = _RaisingRunner(refusal)
    summarizer_calls, writes = [], []

    async def summarizer(**kwargs):
        summarizer_calls.append(kwargs)
        return "summary"

    async def writer(db, **kwargs):
        writes.append(kwargs)
        return "msg-1", "day-1"

    async def broadcaster(user_id, **kwargs):
        return 1

    handler = EmailReceivedHandler(llm_fn=summarizer, writer=writer,
                                   broadcaster=broadcaster, agent_runner=runner)
    handler._mcp_client = object()
    email = _FetchedEmail(event_id="ev-1", gmail_id="gm-1",
                          headers={"From": "Sender <sender@example.test>", "Subject": "Hello"},
                          snippet="Hello", body="Hello there.", labels=["INBOX"], raw_message={})

    async def fetch_all(mcp, events):
        return [email], {}

    handler._fetch_all = fetch_all
    trigger = SimpleNamespace(id="trig-1", user_id=uid, name="Inbox", kind="email_received",
                              action="summarize_and_post", filter_json=None,
                              config_json={"delivery_channels": ["website"]})
    event = SimpleNamespace(id="ev-1", idempotency_key="gm-1", config_json=None)
    result = await handler._execute_core(trigger, [event], db=None)

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    assert runner.calls == 1
    assert summarizer_calls == [] and writes == []
    assert result.status == "failed" and result.per_event_status == {"ev-1": "failed"}
    assert result.error_class == "model_budget"
    assert result.error_detail == sentence
    _assert_plain(result.error_detail)


# ── C4: the turn hands its cause to the sweep ─────────────────────────


def test_the_sweep_hands_on_only_a_budget_cause(monkeypatch):
    """The doubles in test_turn_cancel_finalizes_jobs take no `cause`, and a
    TypeError there is swallowed into a stranded job — so every other ending
    must reach `_close_interrupted_jobs` exactly as before."""
    import app.agent.agent_runner as ar

    seen = []

    class _Tools:
        def __init__(self):
            self._ids = ("job-a",)

        def take_created_job_ids(self):
            ids, self._ids = self._ids, ()
            return ids

    def _close(ids, user_id, staged_action_id=None, **kwargs):
        seen.append(kwargs)

        async def _noop():
            return None
        return _noop()

    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **k: coro.close())
    runner = object.__new__(ar.AgentRunner)
    runner._close_interrupted_jobs = _close
    refusal = budget_429()
    for cause in (refusal, RuntimeError("boom"), asyncio.CancelledError(), None):
        runner.tools = _Tools()
        seen.clear()
        runner._sweep_unclosed_created_jobs("u1", cause=cause)
        assert len(seen) == 1, cause
        assert seen[0] == ({"cause": refusal} if cause is refusal else {}), cause


async def test_a_turn_ended_by_the_budget_hands_the_refusal_to_the_sweep(monkeypatch):
    import app.agent.agent_runner as ar

    monkeypatch.setattr(ar, "sweep_current_voice_job", lambda: None)
    refusal = budget_429()
    got = []

    runner = object.__new__(ar.AgentRunner)

    async def refused(**kwargs):
        raise refusal

    runner._run_inner = refused
    runner._sweep_unclosed_created_jobs = lambda user_id, cause=None: got.append((user_id, cause))
    with pytest.raises(RateLimitError):
        await runner.run(user_message="hi", user_id="u1")
    assert got == [("u1", refusal)]

    # A clean turn still calls it with the user id alone.
    async def answered(**kwargs):
        return "ok"

    runner._run_inner = answered
    runner._sweep_unclosed_created_jobs = lambda user_id: got.append((user_id,))
    assert await runner.run(user_message="hi", user_id="u2") == "ok"
    assert got[-1] == ("u2",)


# ── the analysis kickoff (spec B4, the agent_runner half) ─────────────


AID = "c" * 32


def _blocked_analysis(uid: str, session_id: str, *, task: str, end: datetime) -> dict:
    """The state `start_analysis` hands back for a retry the monthly budget
    still blocks (spec B4): not requeued, so still `failed`, with the
    proxy's reset recorded as `blocked_until`."""
    import app.agent.attachment_analysis as AA

    state = AA.new_state(uid, AID, AA.analysis_id_for(task), task,
                         "Handout.pdf", "page", 6, session_id, "app")
    state.update(
        status="failed", stage="failed",
        error={"code": "analysis_budget_exceeded", "message": "budget"},
        blocked_until=br.iso_utc(end), first_failed_at=br.iso_utc(datetime.now(timezone.utc)),
        units_attempted=0,
    )
    return state


async def _kickoff(monkeypatch, tmp_path, *, analysis: dict | None, task: str, uid: str,
                   session_id: str) -> tuple[str, list]:
    """Drive the REAL runner through the full-file kickoff (one stored PDF,
    an explicit whole-file request) with `start_analysis` answering
    `analysis` — or, with `analysis=None`, the REAL `start_analysis` (the
    caller sets up its storage and preflight). The model must never be
    asked: the kickoff IS the reply."""
    import app.agent.agent_runner as ar
    import app.agent.attachment_analysis as AA
    import app.api.chat_attachments as ca
    from app.agent.tool_executor import ToolExecutor
    from app.db import async_session_maker
    from app.db.models import Conversation

    starts = []
    real_start = AA.start_analysis

    async def fake_start(*args, **kwargs):
        starts.append(kwargs)
        if analysis is None:
            return await real_start(*args, **kwargs)
        return analysis

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    class NoModel:
        async def create_message_stream(self, **kwargs):
            raise AssertionError("the kickoff reply must not come from the model")
            yield  # pragma: no cover

    monkeypatch.setattr(AA, "start_analysis", fake_start)
    monkeypatch.setattr(ca, "load_attachment_record",
                        lambda user_id, aid: {"attachment_id": aid, "mime": "application/pdf",
                                              "name": "Handout.pdf"})
    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar, "intent_wire_prune_enabled", lambda uid: False)
    async with async_session_maker() as db:
        db.add(Conversation(id=session_id, user_id=uid, channel="app"))
        await db.commit()

    chunks = []

    async def capture(chunk):
        chunks.append(chunk)

    runner = ar.AgentRunner(NoModel(), ToolExecutor(workspace=str(tmp_path)))
    response = await runner.run(
        user_message=task, display_user_message=task, display_request=task,
        user_id=uid, session_id=session_id, channel="app", client_tz=TORONTO,
        attachment_records=[{
            "attachment_id": AID, "name": "Handout.pdf", "kind": "document",
            "mime": "application/pdf", "status": "ok", "page_count": 6,
        }],
        on_text_chunk=capture, disable_post_processing=True,
    )
    assert chunks == [response.text]
    return response.text, starts


@pytest.mark.parametrize("task", ["Summarize this", "این فایل را خلاصه کن"], ids=["en", "fa"])
async def test_a_kickoff_the_budget_still_blocks_says_so(monkeypatch, tmp_path, task):
    """The incident's second turn: the user asked again, the retry was
    declined (budget still spent), and the kickoff answered "I found the
    completed analysis and will post it in this chat" about a file it had
    not read a page of."""
    import app.agent.attachment_analysis as AA

    uid, session_id = await _seed_user(tz=TORONTO), str(uuid.uuid4())
    end = _end_in(12, hour=2, minute=30)   # the evening before, in Toronto
    blocked = _blocked_analysis(uid, session_id, task=task, end=end)
    persian = task != "Summarize this"
    reply, starts = await _kickoff(monkeypatch, tmp_path, analysis=blocked, task=task,
                                   uid=uid, session_id=session_id)

    # The analysis module's own sentence, dated in this turn's zone.
    assert reply == AA.blocked_reply_text(blocked, persian=persian, tz_name=TORONTO)
    assert br.reset_when_phrase(end, TORONTO, lang="fa" if persian else "en") in reply
    assert br.reset_when_phrase(end, None, lang="fa" if persian else "en") not in reply
    if persian:
        assert reply.startswith("فعلاً") and reply.endswith("این موضوع روی اعتبار شما اثری ندارد.")
    else:
        assert reply.startswith("I can’t read this file yet: my monthly AI budget is used up.")
        assert reply.endswith("Your credits aren’t affected.")
        _assert_plain(reply)
    for claim in ("I found the completed analysis", "قبلاً انجام شده"):
        assert claim not in reply
    # The reply is the message: no second, re-delivered status part.
    [kwargs] = starts
    assert kwargs["redeliver_when_blocked"] is False and kwargs["retry_failed"] is True


async def test_a_finished_analysis_kickoff_is_unchanged(monkeypatch, tmp_path):
    """Control: only the budget stop changes the kickoff's reply."""
    import app.agent.attachment_analysis as AA

    uid, session_id = await _seed_user(tz=TORONTO), str(uuid.uuid4())
    done = AA.new_state(uid, AID, AA.analysis_id_for("Summarize this"), "Summarize this",
                        "Handout.pdf", "page", 6, session_id, "app")
    done.update(status="completed", stage="completed")
    reply, _starts = await _kickoff(monkeypatch, tmp_path, analysis=done,
                                    task="Summarize this", uid=uid, session_id=session_id)
    assert reply == "I found the completed analysis and will post it in this chat."


def test_the_kickoff_names_the_limit_as_the_clients_do():
    assert "25 MiB" not in RUNNER_SRC, "the clients say 25 MB (format_limit_bytes)"


async def test_a_brand_new_analysis_while_blocked_gets_exactly_one_reply(monkeypatch, tmp_path):
    """The incident's FIRST turn: the budget was already spent when the file
    arrived, and the user got "I’m analyzing the full file…" and, seconds
    later, "not read yet". With the new-job stop (fix pass E1 item 14) the
    REAL start_analysis records the stop before anything runs, and the
    kickoff answers once, with the blocked sentence."""
    import app.agent.attachment_analysis as AA
    from app.services import file_storage

    uid, session_id = await _seed_user(tz=TORONTO), str(uuid.uuid4())
    end = _end_in(12, hour=2, minute=30)
    file_storage._backend = file_storage.LocalDiskBackend(root=str(tmp_path / "store"))
    AA._active.clear()
    AA._start_locks.clear()
    preflights, runs, deliveries = [], [], []

    async def blocked_preflight():
        preflights.append(1)
        return {"blocked": True, "known": True, "period_end": br.iso_utc(end),
                "remaining_cents": -3.5}

    async def no_delivery(state, *args, **kwargs):
        deliveries.append(state.get("analysis_id"))

    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "test-bundle-token", raising=False)
    monkeypatch.setattr(AA, "_agent_mode", lambda: False)   # file checkpoints in every lane
    monkeypatch.setattr(AA, "_budget_preflight", blocked_preflight)
    monkeypatch.setattr(AA, "read_original_bytes", lambda record: b"%PDF-1.4 synthetic")
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 6)
    monkeypatch.setattr(AA, "ensure_running", lambda *args: runs.append(args))
    monkeypatch.setattr(AA, "deliver_analysis", no_delivery)
    try:
        task = "Summarize this"
        reply, starts = await _kickoff(monkeypatch, tmp_path, analysis=None, task=task,
                                       uid=uid, session_id=session_id)
        await asyncio.sleep(0)   # a scheduled delivery would be created by now
        state = AA.load_state(uid, AID, AA.analysis_id_for(task))
    finally:
        AA._active.clear()
        AA._start_locks.clear()
        file_storage._backend = None

    assert preflights == [1] and len(starts) == 1
    assert state is not None and state["status"] == "failed"
    assert state["error"]["code"] == "analysis_budget_exceeded"
    assert all(t["delivered"] for t in state["delivery_targets"])
    # Exactly one reply (`_kickoff` asserts one chunk), and it is the
    # analysis module's blocked sentence dated in this turn's zone.
    assert reply == AA.blocked_reply_text(state, persian=False, tz_name=TORONTO)
    assert br.reset_when_phrase(end, TORONTO) in reply
    assert "I’m analyzing" not in reply and "not read yet" not in reply
    _assert_plain(reply)
    # Nothing runs, and nothing posts a second message.
    assert runs == [] and deliveries == []


# ── E3 item 9: an ORDINARY request-time 429, end to end ───────────────
#
# C1 restored the pre-#391 request-time retry (a 429 or connection error
# raised by `create()` itself is a retry again, not UnboundLocalError), and
# the budget refusal is excluded from it. What that costs on a PERSISTENT
# ordinary 429 is three layers multiplied: the SDK (max_retries=2, so 3
# requests per create), the service loop (3 creates), and the runner's
# ladder (MAX_RETRIES=2, so 3 service calls) — then one fallback. Pinned
# here, through the real bundle client and a counting proxy, so it cannot
# grow silently. max_retries is deliberately NOT changed in this pass.


class _CountingProxy:
    """The platform proxy as an httpx transport: every request is an ordinary
    429 (the G-20 limiter's shape, no budget reason), counted by wire."""

    def __init__(self):
        self.requests: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        wire = "responses" if request.url.path.endswith("/responses") else "chat"
        model = json.loads(request.content or b"{}").get("model")
        self.requests.append((wire, model))
        return httpx.Response(429, json={"detail": "Rate limit exceeded"},
                              headers={"retry-after": "1"})


class _CountingAnthropic:
    def __init__(self):
        self.calls = []

    async def create_message_stream(self, **kwargs):
        self.calls.append(kwargs.get("model"))
        raise RuntimeError("anthropic fallback refused too")
        yield  # pragma: no cover


@pytest.mark.parametrize(
    "has_openai,has_anthropic,fallback_requests,anthropic_calls",
    [
        # OpenAI key only: the configured fallback (gpt-4o, chat wire), which
        # goes back through the same proxy: 27 + 9.
        (True, False, [("chat", "gpt-4o")] * 9, []),
        # Both keys: a rate limit crosses to the Anthropic default instead.
        (True, True, [], ["claude-opus-4-7"]),
        # Bundle token only: no fallback is configured — the 429 is re-raised.
        (False, False, [], []),
    ],
    ids=["openai-key", "both-keys", "bundle-only"],
)
async def test_a_persistent_ordinary_429_costs_a_pinned_number_of_proxy_requests(
    monkeypatch, tmp_path, has_openai, has_anthropic, fallback_requests, anthropic_calls,
):
    import openai

    import app.agent.agent_runner as ar
    import app.services.bundle_client as bc
    import app.services.credit_reporter as cr
    from app.agent.tool_executor import ToolExecutor
    from app.services import key_provider

    proxy = _CountingProxy()
    assert settings.agent_fallback_model == "gpt-4o"      # the production defaults
    assert settings.anthropic_model == "claude-opus-4-7"
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "test-bundle-token", raising=False)
    monkeypatch.setattr(settings, "platform_api_url", "http://platform.test/api", raising=False)
    monkeypatch.setattr(bc, "_proxy_http_client", lambda timeout=120.0: httpx.AsyncClient(
        transport=httpx.MockTransport(proxy),
        event_hooks={"request": [bc._normalize_headers_for_waf]}))
    monkeypatch.setattr(type(key_provider.keys), "has_openai", property(lambda self: has_openai))
    monkeypatch.setattr(type(key_provider.keys), "has_anthropic",
                        property(lambda self: has_anthropic))

    sdk_retries, sleeps, errors = [], [], []

    async def no_sdk_sleep(self, **kwargs):   # the SDK's own back-off, not waited
        sdk_retries.append(kwargs.get("retries_taken"))

    async def no_sleep(delay, *args, **kwargs):   # service + runner back-off
        sleeps.append(delay)

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    async def record_error(self, **kwargs):
        errors.append(kwargs)

    async def fresh():
        return False

    monkeypatch.setattr(openai._base_client.AsyncAPIClient, "_sleep_for_retry", no_sdk_sleep)
    monkeypatch.setattr(ar.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(cr, "raise_if_exhausted", lambda *a, **k: None)
    monkeypatch.setattr(cr, "refresh_if_stale", fresh)
    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar.AgentRunner, "_log_error", record_error)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar, "intent_wire_prune_enabled", lambda uid: False)
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)

    service = OpenAIAgentService()   # the production client: bundle proxy, SDK retries on
    assert service.client.max_retries == openai.DEFAULT_MAX_RETRIES == 2
    anthropic = _CountingAnthropic()
    runner = ar.AgentRunner(llm_service=service, tool_executor=ToolExecutor(workspace=str(tmp_path)))
    runner.anthropic = anthropic

    uid = await _seed_user()
    try:
        with pytest.raises(Exception) as info:
            await runner.run(
                user_message="hi", user_id=uid, session_id=str(uuid.uuid4()),
                channel="app", model_override="gpt-6-sol",
                save_user_message=False, save_assistant_message=False,
                disable_post_processing=True,
            )
    finally:
        await service.client.close()
    assert not br.is_budget_refusal(info.value)

    # The primary: 3 runner attempts x 3 service creates x 3 SDK requests.
    primary = [r for r in proxy.requests if r == ("responses", "gpt-6-sol")]
    assert len(primary) == 27, proxy.requests
    assert proxy.requests == primary + fallback_requests
    assert len(proxy.requests) == 27 + len(fallback_requests)   # 36 / 27 / 27
    assert anthropic.calls == anthropic_calls
    # The SDK retried twice per create: 9 creates on the primary (+3 fallback).
    assert len(sdk_retries) == 2 * (len(proxy.requests) // 3)
    if has_openai and not has_anthropic:
        assert isinstance(info.value, RateLimitError)   # the fallback's own 429
    elif has_anthropic:
        assert "anthropic fallback refused too" in str(info.value)
    else:
        assert isinstance(info.value, RateLimitError)   # re-raised, no fallback


# ── Fix pass v4 (F2): Telegram, system LLM calls, email handlers, frames ──
#
# What the fix-pass verifiers still found: Telegram (which bypasses
# `make_channel_handler`) sent the raw 429; `call_system_llm` swallowed the
# refusal, so the email trigger's default summarize path and the email
# briefing stored a "timeout"; four `job_update` frames named a job but not
# its kind; and a dashboard task logged `Task failed: <raw 429>` into the
# Jobs page's Logs tab.


async def sdk_budget_429(period_end: datetime | None = None, *,
                         typed: bool = True) -> RateLimitError:
    """The proxy's refusal exactly as the REAL openai SDK raises it: an
    ``AsyncOpenAI`` client (no retries) over a mock transport answering the
    typed S2 429 — or, with ``typed=False``, a platform that predates it."""
    import openai

    end = period_end or _end_in(12, hour=2, minute=30)
    if typed:
        body = {"detail": {
            "error": br.REASON, "message": "Monthly openai budget exceeded",
            "provider": "openai",
            "period_start": br.iso_utc(end - timedelta(days=30)),
            "period_end": br.iso_utc(end),
        }}
        headers = {br.REASON_HEADER: br.REASON, "x-should-retry": "false",
                   "Retry-After": "604800"}
    else:
        body, headers = {"detail": "Monthly openai budget exceeded"}, {}

    client = openai.AsyncOpenAI(
        api_key="test-key", base_url="http://platform.test/api/llm/openai/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(429, json=body, headers=headers))),
        max_retries=0,
    )
    try:
        await client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    except RateLimitError as exc:
        return exc
    finally:
        await client.close()
    raise AssertionError("the mock proxy answered instead of refusing")


#: What the raw 429 carries and no user copy may: the status, the SDK's
#: prefix, the payload's key, the period timestamps' names, the enum.
RAW_429_MARKERS = ("429", "Error code", "detail", "period_", "monthly_model_budget_exceeded")


def _assert_no_raw_429(copy: str) -> None:
    for marker in RAW_429_MARKERS:
        assert marker not in copy, (marker, copy)
    _assert_plain(copy)


# ── F2.1: Telegram ────────────────────────────────────────────────────

_MISSING = object()


def _telegram_stubs() -> dict:
    """What `app.agent.telegram_bot` imports from python-telegram-bot, as
    inert stand-ins. The CI image deliberately leaves the library out (see
    `ToupTelegramBot._deliver_turn_attachments`), which would skip these
    tests there forever; the branches under test touch none of it — they run
    on the fake update below."""
    import types

    def _cls(name):
        return type(name, (), {"__init__": lambda self, *a, **k: None})

    tg = types.ModuleType("telegram")
    for name in ("BotCommand", "InlineKeyboardButton", "InlineKeyboardMarkup", "Update"):
        setattr(tg, name, _cls(name))
    constants = types.ModuleType("telegram.constants")
    constants.ChatAction = SimpleNamespace(TYPING="typing")
    ext = types.ModuleType("telegram.ext")
    for name in ("Application", "CallbackQueryHandler", "CommandHandler", "MessageHandler"):
        setattr(ext, name, _cls(name))
    ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    ext.filters = SimpleNamespace()
    tg.constants, tg.ext = constants, ext
    return {"telegram": tg, "telegram.constants": constants, "telegram.ext": ext}


@pytest.fixture
def telegram_bot(monkeypatch):
    """A fresh `app.agent.telegram_bot` bound to the stand-ins — the same in
    every lane, library installed or not — and gone again afterwards."""
    import importlib
    import sys

    import app.agent as agent_pkg

    for name, module in _telegram_stubs().items():
        monkeypatch.setitem(sys.modules, name, module)
    saved_module = sys.modules.pop("app.agent.telegram_bot", None)
    saved_attr = agent_pkg.__dict__.get("telegram_bot", _MISSING)
    try:
        yield importlib.import_module("app.agent.telegram_bot")
    finally:
        sys.modules.pop("app.agent.telegram_bot", None)
        if saved_module is not None:
            sys.modules["app.agent.telegram_bot"] = saved_module
        if saved_attr is _MISSING:
            agent_pkg.__dict__.pop("telegram_bot", None)
        else:
            agent_pkg.telegram_bot = saved_attr


class _TgMessage:
    def __init__(self):
        self.message_id = 7
        self.replies: list[str] = []

    async def set_reaction(self, *args, **kwargs):
        return None

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def _tg_update(*, callback_data: str | None = None):
    async def _noop(*args, **kwargs):
        return None

    message = _TgMessage()
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=4242, type="private", send_action=_noop),
        effective_user=SimpleNamespace(id=1001, full_name="Test User", first_name="Test"),
        message=message,
        callback_query=(SimpleNamespace(data=callback_data, answer=_noop, message=message)
                        if callback_data else None),
        get_bot=lambda: object(),
    )


def _tg_stream_recorder(monkeypatch, tb) -> list[str]:
    """Replace `TelegramStreamHandler`: what the chat finally reads is the
    text a branch hands to `finalize`."""
    finals: list[str] = []

    class _Stream:
        def __init__(self, chat_id, bot=None, reply_to_message_id=None):
            pass

        async def send_initial(self, text=""):
            return None

        def _start_typing(self):
            return None

        def _stop_typing(self):
            return None

        async def on_text_chunk(self, *args, **kwargs):
            return None

        async def on_tool_start(self, *args, **kwargs):
            return None

        async def on_tool_end(self, *args, **kwargs):
            return None

        async def finalize(self, final_text, reply_markup=None):
            finals.append(final_text)

    monkeypatch.setattr(tb, "TelegramStreamHandler", _Stream)
    return finals


def _tg_bot(tb, exc: BaseException, uid: str):
    bot = tb.ToupTelegramBot(token="test-token", agent_runner=_RaisingRunner(exc))

    async def toup_user_id(*args, **kwargs):
        return uid

    async def session_id(*args, **kwargs):
        return "session-test-1"

    bot._get_toup_user_id = toup_user_id
    bot._get_session_id = session_id
    bot._is_allowed = lambda telegram_user_id: True
    return bot


async def _tg_turn(tb, monkeypatch, exc: BaseException, *, branch: str,
                   tz: str | None = TORONTO) -> list[str]:
    """Drive one of the bot's two agent-turn branches into `exc`."""
    from app.agent._user_tz_cache import invalidate_cached_user_tz, set_cached_user_tz

    finals = _tg_stream_recorder(monkeypatch, tb)
    uid = str(uuid.uuid4())
    if tz:
        set_cached_user_tz(uid, tz)
    try:
        bot = _tg_bot(tb, exc, uid)
        if branch == "message":
            await bot._process_message(_tg_update(), "Summarize my week")
        else:
            await bot._handle_callback_query(_tg_update(callback_data="choice:1"),
                                             SimpleNamespace())
    finally:
        invalidate_cached_user_tz(uid)
    return finals


TG_BRANCHES = [pytest.param("message", id="message"), pytest.param("button", id="button")]


@pytest.mark.parametrize("typed", [True, False], ids=["typed", "legacy"])
@pytest.mark.parametrize("branch", TG_BRANCHES)
async def test_telegram_answers_a_budget_refusal_with_the_budget_sentence(
    telegram_bot, monkeypatch, branch, typed,
):
    """The verifier's reproduction: the bot bypasses `make_channel_handler`
    and sent `❌ Sorry, something went wrong:` + `str(e)[:200]` (a message)
    or `❌ Error: …` (a button tap) — the proxy's enum, provider and period
    timestamps, the incident's own symptom."""
    end = _end_in(12, hour=2, minute=30)
    refusal = await sdk_budget_429(end, typed=typed)
    assert "Error code: 429" in str(refusal)   # the text that used to leak

    [sent] = await _tg_turn(telegram_bot, monkeypatch, refusal, branch=branch)
    assert sent == br.chat_sentence(br.budget_refusal_detail(refusal), TORONTO)
    if typed:   # dated in the zone the runner cached for this user
        assert f"resets on {br.reset_when_phrase(end, TORONTO)}." in sent
    _assert_no_raw_429(sent)


@pytest.mark.parametrize("branch", TG_BRANCHES)
async def test_telegram_never_sends_other_exception_text(telegram_bot, monkeypatch, branch):
    from app.agent.channels.shared.message_handler import _GENERIC_ERROR_TEXT

    for exc in (RuntimeError("pool exhausted on db-7 token=test-secret-1"), plain_429()):
        [sent] = await _tg_turn(telegram_bot, monkeypatch, exc, branch=branch)
        assert sent == _GENERIC_ERROR_TEXT
        for leaked in ("pool exhausted", "test-secret-1", "rate limited", "429", "Error"):
            assert leaked not in sent, (leaked, sent)


@pytest.mark.parametrize("command,prefix", [
    ("_cmd_compact", "❌ Compaction failed: "),
    ("_cmd_export", "❌ Export failed: "),
], ids=["compact", "export"])
async def test_telegram_compact_and_export_name_a_budget_stop(
    telegram_bot, monkeypatch, command, prefix,
):
    """Neither command reaches the proxy today (compaction's summarizer
    swallows its errors, export makes no model call), so this is the belt:
    whatever raises inside the command, a budget refusal is answered with
    the budget sentence, never its text. Other errors keep their reply."""
    import app.db.database as database
    from app.agent._user_tz_cache import invalidate_cached_user_tz, set_cached_user_tz

    refusal = await sdk_budget_429()
    sentence = br.chat_sentence(br.budget_refusal_detail(refusal), TORONTO)
    for exc, expected in ((refusal, sentence),
                          (RuntimeError("synthetic outage"), prefix + "synthetic outage")):
        def raising_session_maker(exc=exc):
            raise exc

        monkeypatch.setattr(database, "async_session_maker", raising_session_maker)
        uid = str(uuid.uuid4())
        set_cached_user_tz(uid, TORONTO)
        update = _tg_update()
        try:
            await getattr(_tg_bot(telegram_bot, RuntimeError("unused"), uid), command)(
                update, SimpleNamespace())
        finally:
            invalidate_cached_user_tz(uid)
        assert update.message.replies == [expected]
    _assert_no_raw_429(sentence)


# ── F2.2: the system LLM call reports the refusal ─────────────────────


@pytest.fixture
def bundle_mode(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "test-bundle-token", raising=False)
    monkeypatch.setattr(settings, "platform_api_url", "http://platform.test/api", raising=False)


def _system_clients(monkeypatch, *, openai_outcome=None, anthropic_outcome=None) -> list:
    """`bundle_client` as `call_system_llm` meets it. An outcome is an
    exception to raise, a text to answer with, or None for "no client"."""
    import app.services.bundle_client as bc

    calls: list = []

    def _create(outcome, provider):
        async def create(**kwargs):
            calls.append((provider, kwargs.get("model")))
            if isinstance(outcome, BaseException):
                raise outcome
            if provider == "openai":
                return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=outcome))])
            return SimpleNamespace(usage=None, content=[SimpleNamespace(text=outcome)])
        return create

    def openai_client(*args, **kwargs):
        if openai_outcome is None:
            return None
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=_create(openai_outcome, "openai"))))

    def anthropic_client(*args, **kwargs):
        if anthropic_outcome is None:
            return None
        return SimpleNamespace(messages=SimpleNamespace(
            create=_create(anthropic_outcome, "anthropic")))

    monkeypatch.setattr(bc, "make_openai_client", openai_client)
    monkeypatch.setattr(bc, "make_anthropic_client", anthropic_client)
    return calls


async def _system_call(model: str) -> tuple:
    from app.services.internal_llm import call_system_llm

    failure: dict = {}
    text = await call_system_llm(
        user_id="user-test-1", operation_type="user.trigger.email_received",
        max_tokens=64, system="Summarize.", messages=[{"role": "user", "content": "hi"}],
        model=model, failure_out=failure,
    )
    return text, failure


@pytest.mark.parametrize("typed", [True, False], ids=["typed", "legacy"])
async def test_a_system_llm_budget_refusal_is_reported_not_swallowed(
    bundle_mode, monkeypatch, typed,
):
    """`call_system_llm` never raises; callers learn WHY from `failure_out`.
    The refusal read `rate_limit` there (a retry), and the email handlers
    went on to call it a timeout."""
    from app.services.internal_llm import _classify_failure_reason

    end = _end_in(12, hour=2, minute=30)
    refusal = await sdk_budget_429(end, typed=typed)
    calls = _system_clients(monkeypatch, openai_outcome=refusal)
    text, failure = await _system_call("gpt-4o-mini")
    assert text is None and calls == [("openai", "gpt-4o-mini")]
    if typed:
        assert failure == {"reason": br.ERROR_CLASS, "period_end": br.iso_utc(end)}
    else:   # a platform that predates the typed detail names no reset
        assert failure == {"reason": br.ERROR_CLASS}
    assert _classify_failure_reason(exception=refusal) == br.ERROR_CLASS
    # Every other reason is unchanged.
    assert _classify_failure_reason(exception=plain_429()) == "rate_limit"
    assert _classify_failure_reason(status_code=429) == "rate_limit"
    assert _classify_failure_reason(exception=RuntimeError("read timeout")) == "timeout"
    _system_clients(monkeypatch, openai_outcome=plain_429())
    assert await _system_call("gpt-4o-mini") == (None, {"reason": "rate_limit"})


@pytest.mark.parametrize("anthropic_leg,openai_leg,expected", [
    # Both budgets spent: the EARLIER reset is when a retry first goes through.
    ("budget-late", "budget-early", ("model_budget", "early")),
    # The Anthropic leg's reason wins over a non-generic fallback failure...
    ("budget-late", "rate-limit", ("model_budget", "late")),
    # ...and a generic Anthropic leg (Anthropic disabled) keeps the fallback's.
    ("no-client", "budget-early", ("model_budget", "early")),
    # Unchanged precedence: a real Anthropic failure wins, with no reset.
    ("rate-limit", "budget-early", ("rate_limit", None)),
    # The fallback answered: no failure at all.
    ("budget-late", "answers", None),
], ids=["both-budgets", "budget-then-429", "disabled-then-budget", "429-then-budget",
        "fallback-answers"])
async def test_the_anthropic_path_and_its_fallback_keep_the_budget_reason(
    bundle_mode, monkeypatch, anthropic_leg, openai_leg, expected,
):
    late, early = _end_in(20, hour=2, minute=30), _end_in(5, hour=2, minute=30)
    legs = {
        "budget-late": _anthropic_budget_429(late),
        "budget-early": await sdk_budget_429(early),
        "rate-limit": plain_429(),
        "no-client": None,
        "answers": "The summary.",
    }
    calls = _system_clients(monkeypatch, anthropic_outcome=legs[anthropic_leg],
                            openai_outcome=legs[openai_leg])
    text, failure = await _system_call("claude-haiku-4-5-20251001")
    assert calls[-1] == ("openai", "gpt-4o-mini")   # the fallback always ran
    if expected is None:
        assert (text, failure) == ("The summary.", {})
        return
    reason, which_end = expected
    ends = {"late": br.iso_utc(late), "early": br.iso_utc(early)}
    assert text is None
    assert failure == ({"reason": reason, "period_end": ends[which_end]} if which_end
                       else {"reason": reason})


async def test_the_day_summarizer_records_the_budget_reason(bundle_mode, monkeypatch):
    """The reason's consumers take any string: day_summarizer stores it in
    `day_chats.summary_last_failure_reason` (VARCHAR 50) and logs it."""
    from app.db.models.day_chat import DayChat
    from app.services import day_summarizer

    _system_clients(monkeypatch, anthropic_outcome=_anthropic_budget_429(), openai_outcome=None)
    text, reason = await day_summarizer._try_summarize("user-test-1", "A synthetic day.")
    assert (text, reason) == (None, br.ERROR_CLASS)
    assert len(reason) <= DayChat.__table__.c.summary_last_failure_reason.type.length


# ── F2.3 / F2.4: the email trigger's summarize path and the briefing ──


def _email_handler(**kwargs):
    from app.agent.triggers.email_received_handler import EmailReceivedHandler, _FetchedEmail

    writes: list = []

    async def writer(db, **kw):
        writes.append(kw)
        return "msg-1", "day-1"

    async def broadcaster(user_id, **kw):
        return 1

    handler = EmailReceivedHandler(writer=writer, broadcaster=broadcaster, **kwargs)
    handler._mcp_client = object()
    email = _FetchedEmail(event_id="ev-1", gmail_id="gm-1",
                          headers={"From": "Sender <sender@example.test>", "Subject": "Hello"},
                          snippet="Hello", body="Hello there.", labels=["INBOX"], raw_message={})

    async def fetch_all(mcp, events):
        return [email], {}

    handler._fetch_all = fetch_all
    return handler, writes


async def _email_fire(handler, uid: str):
    trigger = SimpleNamespace(id="trig-1", user_id=uid, name="Inbox", kind="email_received",
                              action="summarize_and_post", filter_json=None,
                              config_json={"delivery_channels": ["website"]})
    event = SimpleNamespace(id="ev-1", idempotency_key="gm-1", config_json=None)
    return await handler._execute_core(trigger, [event], db=None)


@pytest.mark.parametrize("typed", [True, False], ids=["typed", "legacy"])
@pytest.mark.parametrize("route", ["summarize", "runner-failed-then-summarize"])
async def test_an_email_trigger_on_the_summarize_path_names_the_budget(
    bundle_mode, monkeypatch, typed, route,
):
    """`trigger_turns_via_runner` ships OFF, so the summarize path is the
    default one — and the fallback after any runner failure. It stored
    "internal_llm returned None (timeout / auth / parse failure)": a
    timeout, served verbatim as the trigger's last_error and the run's
    error."""
    from app.agent.job_status import DISPOSITION_TERMINAL, classify
    from app.api.triggers import _served_error

    monkeypatch.setattr(settings, "trigger_turns_via_runner", route != "summarize",
                        raising=False)
    monkeypatch.setattr(settings, "trigger_turn_shadow", False, raising=False)
    uid = await _seed_user(tz=TORONTO)
    end = _end_in(12, hour=2, minute=30)
    refusal = await sdk_budget_429(end, typed=typed)
    calls = _system_clients(monkeypatch, openai_outcome=refusal)
    runner = _RaisingRunner(RuntimeError("runner unavailable")) if route != "summarize" else None
    handler, writes = _email_handler(agent_runner=runner)
    result = await _email_fire(handler, uid)

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    if typed:
        assert br.reset_when_phrase(end, TORONTO) in sentence
    assert calls == [("openai", "gpt-4o-mini")] and writes == []
    assert result.status == "failed" and result.per_event_status == {"ev-1": "failed"}
    assert result.error_class == br.ERROR_CLASS
    assert result.error_detail == sentence
    _assert_no_raw_429(result.error_detail)
    # Terminal wherever it is read back (the jobs and triggers APIs and the
    # runner's gate classify the stored text), and served as plain copy.
    verdict = classify(result.error_detail)
    assert verdict.error_class == br.ERROR_CLASS
    assert verdict.disposition == DISPOSITION_TERMINAL
    _assert_no_raw_429(_served_error(result.error_detail))


async def test_an_email_trigger_timeout_keeps_its_story(bundle_mode, monkeypatch):
    monkeypatch.setattr(settings, "trigger_turns_via_runner", False, raising=False)
    uid = await _seed_user(tz=TORONTO)
    _system_clients(monkeypatch, openai_outcome=RuntimeError("read timeout"))
    handler, _writes = _email_handler()
    result = await _email_fire(handler, uid)
    assert (result.status, result.error_class, result.error_detail) == (
        "failed", "RuntimeError", "internal_llm returned None (timeout / auth / parse failure)")


class _BriefingMCP:
    """fastmcp.Client as the briefing uses it: one list call, then one get
    per id (pattern: tests/test_email_briefing_handler.py)."""

    def __init__(self):
        self.calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name, args):
        self.calls.append(name)
        if name == "gmail__list_messages":
            payload = {"messages": [{"id": "m-1", "threadId": "t-1"}], "result_size": 1}
        else:
            payload = {"id": "m-1", "threadId": "t-1",
                       "headers": {"From": "sender@example.test", "Subject": "Hello"},
                       "snippet": "Hello there.", "body": "Hello there.",
                       "internalDate": 1780000000000}
        return SimpleNamespace(structured_content={"kind": "ok", "content": json.dumps(payload)})


async def _briefing_run(monkeypatch, uid: str):
    import app.services.credit_reporter as cr
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler

    async def not_exhausted(*args, **kwargs):
        return None

    monkeypatch.setattr(cr, "raise_if_exhausted_async", not_exhausted)
    routine = SimpleNamespace(id="rt-1", user_id=uid, name="Morning email briefing",
                              config_json={}, last_state_json=None)
    mcp = _BriefingMCP()
    result = await EmailBriefingHandler(mcp_client=mcp).execute(routine, None, None)
    return result, mcp


@pytest.mark.parametrize("typed", [True, False], ids=["typed", "legacy"])
async def test_an_email_briefing_the_budget_refused_says_so(bundle_mode, monkeypatch, typed):
    """It returned "call_system_llm returned None (timeout / auth / parse)",
    which the routine runner's gate read as a timeout and ran twice more."""
    from app.agent.job_status import DISPOSITION_TERMINAL, classify

    uid = await _seed_user(tz=TORONTO)
    end = _end_in(12, hour=2, minute=30)
    refusal = await sdk_budget_429(end, typed=typed)
    calls = _system_clients(monkeypatch, openai_outcome=refusal)
    result, mcp = await _briefing_run(monkeypatch, uid)

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    assert mcp.calls == ["gmail__list_messages", "gmail__get_message"]
    assert calls == [("openai", "gpt-4o-mini")]
    assert result.status == "failed"
    assert result.error_class == br.ERROR_CLASS and result.error_detail == sentence
    _assert_no_raw_429(result.error_detail)
    # The routine runner's retry gate composes exactly this: terminal.
    gate = classify(f"{result.error_class or ''}: {result.error_detail or ''}")
    assert gate.error_class == br.ERROR_CLASS and gate.disposition == DISPOSITION_TERMINAL


async def test_an_email_briefing_timeout_keeps_its_story(bundle_mode, monkeypatch):
    uid = await _seed_user(tz=TORONTO)
    _system_clients(monkeypatch, openai_outcome=RuntimeError("read timeout"))
    result, _mcp = await _briefing_run(monkeypatch, uid)
    assert (result.status, result.error_class, result.error_detail) == (
        "failed", "llm_returned_none", "call_system_llm returned None (timeout / auth / parse)")


# ── F2.5: every frame that names a job names its kind ─────────────────
#
# The web draws a card with no kind as an app build ("Couldn't build …"), and
# a tab meets a job first in whichever frame reaches it first.


@pytest.mark.parametrize("cause", ["budget", "none"])
async def test_an_interrupted_jobs_frame_names_its_kind(pushes, frames, cause):
    from app.agent.agent_runner import AgentRunner

    uid = await _seed_user(tz=TORONTO)
    jid = await _seed_job(uid, title="Draft proposal", config_json={"job_type": "write"})
    refusal = budget_429(_end_in(12, hour=2, minute=30)) if cause == "budget" else None
    await AgentRunner.__new__(AgentRunner)._close_interrupted_jobs((jid,), uid, cause=refusal)
    [frame] = [f for f in frames if f.get("job_id") == jid]
    assert frame["type"] == "job_update" and frame["name"] == "Draft proposal"
    assert frame["job_type"] == "write"


async def test_the_reapers_frames_name_the_jobs_kind(pushes, frames):
    from app.agent import job_reaper

    uid = await _seed_user()
    now = datetime.utcnow()
    stalled = await _seed_job(uid, title="Look up the ferry times",
                              config_json={"job_type": "search"},
                              created_at=now - timedelta(minutes=45))
    parked = await _seed_job(uid, title="Send the invoice", status="waiting_on_user",
                             error_class="awaiting_confirmation",
                             created_at=now - timedelta(hours=26))
    await job_reaper.sweep_stalled_jobs(now)   # runs the card-park sweep too
    kinds = {f["job_id"]: f.get("job_type") for f in frames if f.get("type") == "job_update"}
    # The create_job tag when the row has one, else the row's job_type.
    assert kinds == {stalled: "search", parked: "agent_task"}


async def test_a_resolved_card_parks_frame_names_the_jobs_kind(pushes, frames, monkeypatch):
    from app.api.agent import ResolvePendingActionRequest, resolve_job_for_pending_action

    monkeypatch.setattr(settings, "run_mode", "agent", raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "test-agent-key", raising=False)
    uid = await _seed_user()
    jid = await _seed_job(uid, title="Send the invoice", status="waiting_on_user",
                          error_class="awaiting_confirmation",
                          config_json={"pending_action_id": "act-test-1", "job_type": "write"})
    response = await resolve_job_for_pending_action(
        ResolvePendingActionRequest(action_id="act-test-1", outcome="executed"),
        SimpleNamespace(headers={"X-Agent-Key": "test-agent-key"}),
    )
    assert response.resolved == 1
    [frame] = [f for f in frames if f.get("job_id") == jid]
    assert frame["status"] == "completed" and frame["job_type"] == "write"


async def test_a_voice_tasks_frame_names_its_kind(frames):
    from app.agent import voice_tasks

    await voice_tasks.VoiceTaskService.__new__(voice_tasks.VoiceTaskService)._broadcast(
        "user-test-1",
        {"task_id": "task-test-1", "status": "failed", "title": "Look up the ferry times"},
    )
    [frame] = [f for f in frames if f.get("type") == "job_update"]
    assert frame["name"] == "Look up the ferry times"
    assert frame.get("job_type") == "agent_task"
    # The same constant every voice task row is created with.
    assert voice_tasks.JOB_TYPE == "agent_task"
    assert "job_type=JOB_TYPE" in Path(voice_tasks.__file__).read_text()


# ── F2.5 / F2.6: the dashboard task ───────────────────────────────────


async def _dashboard_task(monkeypatch, runner, uid: str) -> tuple:
    """`POST /apps/jobs/` — a task created from the dashboard — run to its
    end: the route creates the row and spawns the run, awaited here."""
    import app.api.apps as apps

    sent: list = []
    spawned: list = []

    async def ws_broadcast(user_id, event):
        sent.append(event)

    monkeypatch.setattr(settings, "user_id", uid, raising=False)
    monkeypatch.setattr(apps, "_agent_runner", runner)
    monkeypatch.setattr(apps, "_ws_broadcast", ws_broadcast)
    monkeypatch.setattr(apps, "_spawn_bg", lambda coro, **kwargs: spawned.append(coro))
    response = await apps.create_job(apps.CreateJobRequest(
        title="Draft proposal", description="Draft the spring workshop proposal"))
    [run] = spawned
    await run
    return response.id, sent


def _error_log_lines(row) -> list[str]:
    return [entry["message"] for entry in json.loads(row.build_logs_json or "[]")
            if entry.get("level") == "error"]


async def test_a_dashboard_tasks_completion_frame_names_its_kind(monkeypatch):
    class _Answering:
        async def run(self, **kwargs):
            return SimpleNamespace(tokens_total=0, model="")

    uid = await _seed_user()
    jid, sent = await _dashboard_task(monkeypatch, _Answering(), uid)
    [frame] = [f for f in sent if f.get("type") == "job_update"]
    assert frame == {"type": "job_update", "job_id": jid, "status": "completed",
                     "name": "Draft proposal", "job_type": "agent_task"}


async def test_a_dashboard_task_the_budget_stopped_logs_the_sentence(monkeypatch):
    """The Jobs page's Logs tab rendered `Task failed: Error code: 429 -
    {'detail': …}` for this stop (and the same line went out live as a
    `job_log` frame and into the job_events feed)."""
    from sqlalchemy import select

    from app.db import async_session_maker
    from app.db.models import JobEvent

    uid = await _seed_user(tz=TORONTO)
    end = _end_in(12, hour=2, minute=30)
    refusal = await sdk_budget_429(end)
    jid, sent = await _dashboard_task(monkeypatch, _RaisingRunner(refusal), uid)

    sentence = br.job_sentence(br.budget_refusal_detail(refusal), TORONTO)
    assert br.reset_when_phrase(end, TORONTO) in sentence
    row = await _row(jid)
    assert _error_log_lines(row) == [f"Task stopped: {sentence}"]
    live = [f["entry"]["message"] for f in sent if f.get("type") == "job_log"]
    async with async_session_maker() as db:
        labels = [e.label for e in (await db.execute(
            select(JobEvent).where(JobEvent.job_id == jid))).scalars() if e.label]
    assert f"Task stopped: {sentence}" in live and f"Task stopped: {sentence}" in labels
    for line in live + labels:
        _assert_no_raw_429(line)
    # The row and the live frame say the same dated sentence.
    assert row.status == "failed" and row.error_message   # raw text kept for operators
    assert row.error_class == br.ERROR_CLASS and row.user_message == sentence
    [frame] = [f for f in sent if f.get("type") == "job_update"]
    assert frame["user_message"] == sentence and frame["error_class"] == br.ERROR_CLASS


async def test_a_dashboard_tasks_other_failure_keeps_its_log_line(monkeypatch):
    uid = await _seed_user()
    jid, _sent = await _dashboard_task(monkeypatch, _RaisingRunner(RuntimeError("boom")), uid)
    row = await _row(jid)
    assert _error_log_lines(row) == ["Task failed: boom"]
    assert row.error_class != br.ERROR_CLASS
