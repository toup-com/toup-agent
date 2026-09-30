"""A routine run stopped by the monthly model budget is served as a sentence.

Lane: platform.

The runs-history endpoint projects ``build_jobs`` rows through
``routines._run_to_response`` and the dashboard renders ``error_detail``
verbatim (RoutinesSection tooltip). A run written by an agent image that
predates the 2026-09-28 incident fix stores the proxy's raw 429 there, with
its period timestamps. It must be served as the taxonomy's sentence, and
every other error must pass through untouched.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from types import SimpleNamespace

from app.api.routines import _run_to_response
from app.services import budget_refusal

RAW_429 = (
    "Error code: 429 - {'detail': {'error': 'monthly_model_budget_exceeded', "
    "'message': 'Monthly openai budget exceeded', 'provider': 'openai', "
    "'period_start': '2026-09-01T10:00:00+00:00', 'period_end': '2026-10-01T10:00:00+00:00'}}"
)
LEGACY_429 = "Error code: 429 - {'detail': 'Monthly openai budget exceeded'}"


def _job(error_message=None, error_json=None):
    return SimpleNamespace(
        id="run-1", idempotency_key=date(2026, 9, 28).isoformat(),
        created_at=datetime(2026, 9, 28, 12, 0), completed_at=datetime(2026, 9, 28, 12, 1),
        fire_instant=None, finished_local_at=None, status="failed", outcome="failure",
        error_message=error_message, error_json=error_json, channel_results_json=None,
        tools_invoked_json=None, emails_fetched=0, attempt=1, summary_message_id=None,
    )


def test_raw_budget_429_is_served_as_the_job_sentence():
    for raw in (RAW_429, LEGACY_429):
        resp = _run_to_response(_job(error_message=raw))
        assert resp.error_class == budget_refusal.ERROR_CLASS
        assert resp.error_detail == budget_refusal.job_sentence(None)
        assert not re.search(r"\d{4}-\d{2}-\d{2}|429|detail", resp.error_detail)


def test_budget_text_in_error_json_is_served_as_the_job_sentence():
    resp = _run_to_response(_job(error_json={"error_detail": RAW_429}))
    assert resp.error_detail == budget_refusal.job_sentence(None)
    # The structured blob itself is served as stored (operators read it).
    assert resp.error_json == {"error_detail": RAW_429}


def test_other_errors_pass_through_unchanged():
    for raw in ("Gmail read failed: token expired", "Rate limit exceeded for this tenant token.",
                "Customer wrote: our Monthly OpenAI budget exceeded the plan"):
        resp = _run_to_response(_job(error_message=raw))
        assert resp.error_detail == raw
        assert resp.error_class is None
    assert _run_to_response(_job()).error_detail is None


# -- the stored sentence keeps its reset date (correctness review, fix pass 2) --

from datetime import timezone as _tz  # noqa: E402

from app.agent.job_status import served_model_budget_text  # noqa: E402
from app.api.routines import _served_last_error  # noqa: E402
from app.api.triggers import _served_error  # noqa: E402

_NOW = datetime(2026, 10, 3, 12, 0, tzinfo=_tz.utc)


def _dated(period_end: str, tz_name) -> str:
    return budget_refusal.job_sentence({"period_end": period_end}, tz_name, now=_NOW)


# Every shape reset_when_phrase renders: plain, across a year, inside 48 h,
# inside 48 h with the UTC fallback zone.
DATED = [
    _dated("2026-10-24T15:00:00+00:00", "America/Chicago"),
    _dated("2027-01-03T15:00:00+00:00", "America/Chicago"),
    _dated("2026-10-04T15:10:00+00:00", "America/Chicago"),
    _dated("2026-10-04T15:10:00+00:00", "Not/AZone"),
]


def test_every_dated_sentence_shape_is_recognised():
    assert "October 24." in DATED[0] and "2027" in DATED[1]
    assert " at " in DATED[2] and DATED[3].count("UTC") == 1
    for sentence in DATED:
        assert served_model_budget_text(sentence) == sentence


def test_a_stored_dated_sentence_is_served_with_its_date():
    for sentence in DATED:
        for stored in (sentence, f"RuntimeError: {sentence}"):
            resp = _run_to_response(_job(error_message=stored))
            assert resp.error_class == budget_refusal.ERROR_CLASS
            assert resp.error_detail == sentence
            assert _served_error(stored) == sentence
            assert _served_last_error(stored) == sentence


def test_raw_429_on_every_surface_is_the_undated_sentence():
    for raw in (RAW_429, LEGACY_429):
        assert _served_error(raw) == budget_refusal.job_sentence(None)
        assert _served_last_error(raw) == budget_refusal.job_sentence(None)


def test_our_sentence_with_anything_appended_is_not_served_as_is():
    tail = DATED[0] + " detail: " + RAW_429
    assert served_model_budget_text(tail) == budget_refusal.job_sentence(None)
    assert "429" not in _served_error(tail)


def test_non_budget_text_passes_through_every_surface():
    for raw in ("Gmail read failed: token expired", None, ""):
        assert served_model_budget_text(raw) is None
        assert _served_error(raw) == raw
        assert _served_last_error(raw) == raw


# -- the email briefing says why when the budget stopped it (review note) ------
#
# The briefing summarises through `call_system_llm`, which never raises: a
# budget refusal comes back as None with `failure_out["reason"] ==
# "model_budget"` and the reset in `failure_out["period_end"]`. Before, the
# run read "call_system_llm returned None (timeout / auth / parse)", a
# retryable timeout served verbatim. test_email_briefing_handler.py covers
# the handler on the routines table but is excused from CI (COVERAGE_DEBT);
# this case needs no table, so it runs in this file's platform lane.

import pytest  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402


def _briefing_routine():
    return _NS(id="routine-1", user_id="budget-briefing-user",
               config_json={"mode": "latest_n", "max_emails": 1}, last_state_json=None)


async def _briefing_result(failure: dict):
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler
    from tests.test_email_briefing_handler import (
        _FakeMCP, _RecordingWriter, _get_message_ok, _list_messages_ok,
    )

    seen = {}

    async def _refused_llm(**kwargs):
        seen.update(kwargs)
        if kwargs.get("failure_out") is not None:
            kwargs["failure_out"].update(failure)
        return None

    mcp = _FakeMCP([
        ("gmail__list_messages", None, _list_messages_ok(["m1"])),
        ("gmail__get_message", None, _get_message_ok("m1")),
    ])
    writer = _RecordingWriter()
    handler = EmailBriefingHandler(mcp_client=mcp, llm_fn=_refused_llm, writer=writer)
    result = await handler.execute(_briefing_routine(), None, None)
    assert "failure_out" in seen, "the handler must ask call_system_llm why it failed"
    return result


@pytest.mark.asyncio
async def test_a_briefing_the_budget_stopped_says_so_with_its_reset():
    end = "2099-10-24T15:00:00+00:00"
    result = await _briefing_result({"reason": "model_budget", "period_end": end})
    assert result.status == "failed"
    assert result.error_class == budget_refusal.ERROR_CLASS
    assert result.error_detail == budget_refusal.job_sentence({"period_end": end}, None)
    assert "October 24" in result.error_detail
    assert not re.search(r"timeout|None|429|detail", result.error_detail)
    # The served text keeps the date on the Routines dashboard.
    resp = _run_to_response(_job(error_message=result.error_detail))
    assert resp.error_detail == result.error_detail


@pytest.mark.asyncio
async def test_a_briefing_the_budget_stopped_without_a_reset_is_undated():
    result = await _briefing_result({"reason": "model_budget"})
    assert result.error_class == budget_refusal.ERROR_CLASS
    assert result.error_detail == budget_refusal.job_sentence(None)


@pytest.mark.asyncio
async def test_a_briefing_that_timed_out_still_reads_as_a_timeout():
    result = await _briefing_result({"reason": "timeout"})
    assert result.error_class == "llm_returned_none"
    assert "timeout" in result.error_detail
