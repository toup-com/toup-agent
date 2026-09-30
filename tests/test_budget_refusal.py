"""app/services/budget_refusal — the one vocabulary for the proxy's monthly
model-budget refusal (incident 2026-09-28).

Pins: recognition on the typed header, the typed body, the legacy sentence
and the python-repr ``str(exc)`` of a 429; NO recognition of ordinary 429s,
the G-20 per-minute limiter, credit 402s, upstream ``insufficient_quota`` or
third-party text that merely quotes the marker (only a 429 — the exception's
status, or the SDKs' ``Error code: 429`` prefix — is trusted); the time
contract (naive == UTC, aware in, explicit +00:00 out, no microseconds); the
absolute, zone-aware reset phrase; the exact job sentence ("couldn’t
finish"); and that no rendered sentence leaks cents, timestamps, 'rate
limit' or the paywall trigger phrase.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException
from openai import InternalServerError, RateLimitError

from app.services import budget_refusal as br

PERIOD_END = "2026-10-24T18:25:30.250000"      # synthetic; microseconds like the naive column
# The proxy's wire detail: spend and budget are NOT on it (they reach job
# rows and model context through str(exc)); only the log line has them.
DETAIL = {
    "error": br.REASON,
    "message": "Monthly openai budget exceeded",
    "provider": "openai",
    "period_start": "2026-09-24T18:25:30+00:00",
    "period_end": PERIOD_END,
}


def _typed_429(*, header: bool = True, body: bool = True) -> RateLimitError:
    """Exactly what openai 2.53 raises for the proxy's typed refusal:
    ``.body == {"detail": {...}}`` and ``str(exc)`` is a python repr."""
    req = httpx.Request("POST", "http://platform.test/api/llm/openai/v1/chat/completions")
    payload = {"detail": dict(DETAIL)} if body else {"detail": "Monthly openai budget exceeded"}
    headers = {br.REASON_HEADER: br.REASON} if header else {}
    resp = httpx.Response(429, request=req, json=payload, headers=headers)
    return RateLimitError(f"Error code: 429 - {payload}", response=resp, body=payload)


def _plain_429(text: str) -> RateLimitError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req, json={"detail": text})
    return RateLimitError(f"Error code: 429 - {{'detail': '{text}'}}", response=resp, body={"detail": text})


# ── recognition ──────────────────────────────────────────────────────


def test_typed_refusal_is_recognised_by_header_alone():
    assert br.is_budget_refusal(_typed_429(header=True, body=False))


def test_typed_refusal_is_recognised_by_body_alone():
    assert br.is_budget_refusal(_typed_429(header=False, body=True))


def test_legacy_sentence_is_recognised_case_insensitively():
    assert br.is_budget_refusal(_plain_429("Monthly OpenAI budget exceeded"))
    assert br.is_budget_refusal("Error code: 429 - {'detail': 'Monthly anthropic budget exceeded'}")


def test_repr_text_is_recognised():
    assert br.is_budget_refusal(str(_typed_429()))
    assert br.is_budget_refusal(repr(_typed_429()))


@pytest.mark.parametrize("text", [
    "Rate limit exceeded for this tenant token.",       # G-20 per-minute limiter
    "Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o'}}",
    "out_of_credits:insufficient_message_credits:message",
    "Error code: 429 - {'error': {'code': 'insufficient_quota', 'type': 'insufficient_quota'}}",
    "",
])
def test_other_failures_are_not_budget_refusals(text):
    assert not br.is_budget_refusal(text)
    assert not br.is_budget_refusal(_plain_429(text or "x"))


def test_recognition_never_raises_on_odd_objects():
    class Weird(Exception):
        @property
        def response(self):
            raise RuntimeError("boom")

        def __str__(self):
            raise RuntimeError("boom")

    assert br.is_budget_refusal(Weird()) is False
    assert br.budget_refusal_detail(Weird()) is None
    assert br.is_budget_refusal(object()) is False
    assert br.is_budget_refusal(None) is False


def test_a_429_is_trusted_by_its_status_even_without_the_sdk_prefix():
    # The SDK's message is the raw body text when that body is not JSON; the
    # exception still says 429. A FastAPI HTTPException (in-process) too.
    req = httpx.Request("POST", "http://platform.test/api/llm/openai/v1/chat/completions")
    raw = RateLimitError("Monthly openai budget exceeded",
                         response=httpx.Response(429, request=req, text="Monthly openai budget exceeded"),
                         body="Monthly openai budget exceeded")
    assert br.is_budget_refusal(raw) and br.budget_refusal_detail(raw) is None
    in_process = HTTPException(status_code=429, detail=dict(DETAIL))
    assert br.is_budget_refusal(in_process)
    assert br.budget_refusal_detail(in_process) == DETAIL


def test_stored_error_text_is_trusted_only_with_the_sdk_prefix():
    # job_runner stores repr(e); ws_chat and the routine/trigger handlers str(e).
    typed = _typed_429(header=False)
    for stored in (repr(typed), f"RateLimitError: {typed}", str(typed)[:300]):
        assert br.is_budget_refusal(stored), stored
        assert br.budget_refusal_detail(stored) == DETAIL, stored
    # The same words with no "Error code: 429" are someone else's text.
    for stored in (br.REASON, "Monthly openai budget exceeded", repr({"detail": DETAIL}),
                   json.dumps({"detail": DETAIL}), "Error code: 500 - " + repr({"detail": DETAIL})):
        assert not br.is_budget_refusal(stored), stored
        assert br.budget_refusal_detail(stored) is None, stored


# ── third-party text is never a budget stop ──────────────────────────

_SPOOF = {"detail": {"error": br.REASON, "provider": "openai", "period_end": "2099-01-01T00:00:00"}}


@pytest.mark.parametrize("exc", [
    # a summariser that quotes the email it failed on
    RuntimeError(f"email_received: summarizer failed on body: {_SPOOF}"),
    # a customer's own words inside a tool error
    ValueError("Customer wrote: our Monthly OpenAI budget exceeded the plan, please advise"),
    KeyError(br.REASON),
    # an upstream 500 whose body quotes the marker is not a budget stop either
    InternalServerError(
        f"Error code: 500 - {_SPOOF}",
        response=httpx.Response(500, request=httpx.Request("POST", "https://api.example.test/v1")),
        body=None),
], ids=["quoted-dict", "quoted-sentence", "bare-marker", "upstream-500"])
def test_third_party_text_carrying_the_marker_is_not_a_refusal(exc):
    assert br.is_budget_refusal(exc) is False
    assert br.budget_refusal_detail(exc) is None
    # ...so nothing downstream can print the attacker's date.
    assert br.job_sentence(br.budget_refusal_detail(exc), "America/Toronto", now=NOW) == br.job_sentence(None)


def test_only_a_proxy_shaped_dict_is_lifted_out_of_text():
    """Inside a trusted 429 text, a marker dict still needs the proxy's shape
    (a provider the proxy budgets) before its period_end is believed: the
    text still reads as a refusal, but an undated one."""
    for provider in ("mallory", None, ["openai"]):
        spoof = {"detail": {"error": br.REASON, "provider": provider,
                            "period_end": "2099-01-01T00:00:00"}}
        text = f"Error code: 429 - {spoof}"
        assert br.is_budget_refusal(text), provider
        assert br.budget_refusal_detail(text) is None, provider
        assert br.job_sentence(br.budget_refusal_detail(text), "America/Toronto", now=NOW) \
            == br.job_sentence(None)
    for provider in ("openai", "anthropic"):
        text = f"Error code: 429 - {{'detail': {dict(DETAIL, provider=provider)}}}"
        assert br.budget_refusal_detail(text)["provider"] == provider


# ── detail extraction ────────────────────────────────────────────────


def test_detail_from_exception_body():
    d = br.budget_refusal_detail(_typed_429())
    assert d is not None and d["period_end"] == PERIOD_END and d["provider"] == "openai"


def test_detail_from_python_repr_text():
    d = br.budget_refusal_detail(str(_typed_429(header=False)))
    assert d is not None and d["error"] == br.REASON and d["period_end"] == PERIOD_END


def test_detail_from_json_text():
    d = br.budget_refusal_detail("Error code: 429 - " + json.dumps({"detail": DETAIL}) + " trailing")
    assert d is not None and d["provider"] == "openai"


def test_legacy_refusal_has_no_detail():
    assert br.budget_refusal_detail(_plain_429("Monthly openai budget exceeded")) is None


# ── time contract ────────────────────────────────────────────────────


def test_parse_utc_treats_naive_and_offsetless_as_utc():
    naive = datetime(2026, 10, 24, 18, 25, 30)
    assert br.parse_utc(naive) == naive.replace(tzinfo=timezone.utc)
    assert br.parse_utc("2026-10-24T18:25:30") == naive.replace(tzinfo=timezone.utc)
    assert br.parse_utc("2026-10-24T18:25:30Z") == naive.replace(tzinfo=timezone.utc)
    assert br.parse_utc("2026-10-24T14:25:30-04:00") == naive.replace(tzinfo=timezone.utc)


def test_parse_utc_never_raises():
    assert br.parse_utc("not a date") is None
    assert br.parse_utc(12345) is None
    assert br.parse_utc("") is None


def test_naive_and_iso_forms():
    assert br.to_naive_utc("2026-10-24T14:25:30-04:00") == datetime(2026, 10, 24, 18, 25, 30)
    assert br.iso_utc("2026-10-24T18:25:30.250000") == "2026-10-24T18:25:30+00:00"
    assert br.iso_utc(None) is None


def test_resets_at_omits_a_past_or_missing_end():
    now = datetime(2026, 10, 25, tzinfo=timezone.utc)
    assert br.resets_at(DETAIL, now=now) is None
    assert br.resets_at({"error": br.REASON}) is None
    assert br.resets_at(DETAIL, now=datetime(2026, 9, 29, tzinfo=timezone.utc)) == \
        datetime(2026, 10, 24, 18, 25, 30, 250000, tzinfo=timezone.utc)


# ── reset phrase ─────────────────────────────────────────────────────

NOW = datetime(2026, 9, 29, 1, 30, tzinfo=timezone.utc)


def test_reset_phrase_is_absolute_and_in_the_users_zone():
    # 18:25 UTC on Oct 24 is still Oct 24 in Toronto; a UTC-midnight anchor is not.
    assert br.reset_when_phrase(PERIOD_END, "America/Toronto", now=NOW) == "October 24"
    assert br.reset_when_phrase("2026-10-25T02:30:00", "America/Toronto", now=NOW) == "October 24"
    assert br.reset_when_phrase("2026-10-25T02:30:00", None, now=NOW) == "October 25"


def test_reset_phrase_adds_time_inside_48_hours_and_utc_when_zone_unknown():
    soon = NOW + timedelta(hours=5)
    assert br.reset_when_phrase(soon, "America/Toronto", now=NOW) == "September 29 at 2:30 AM"
    assert br.reset_when_phrase(soon, None, now=NOW) == "September 29 at 6:30 AM UTC"
    assert br.reset_when_phrase(soon, "Not/AZone", now=NOW) == "September 29 at 6:30 AM UTC"


def test_reset_phrase_names_the_year_across_a_boundary_and_never_the_past():
    assert br.reset_when_phrase("2027-01-03T12:00:00", "America/Toronto", now=NOW) == "January 3, 2027"
    assert br.reset_when_phrase("2026-09-01T00:00:00", "America/Toronto", now=NOW) is None
    assert br.reset_when_phrase(None, "America/Toronto", now=NOW) is None


def test_reset_phrase_persian_uses_persian_digits_and_months():
    assert br.reset_when_phrase(PERIOD_END, "America/Toronto", now=NOW, lang="fa") == "۲۴ اکتبر"
    soon = NOW + timedelta(hours=5)
    assert br.reset_when_phrase(soon, "America/Toronto", now=NOW, lang="fa") == "۲۹ سپتامبر ساعت ۰۲:۳۰"


# ── sentences ────────────────────────────────────────────────────────

_FORBIDDEN = re.compile(r"cents|\d{4}-\d{2}-\d{2}|rate limit|out of toup credits|something went wrong|\b5\d\d\b", re.I)


@pytest.mark.parametrize("render", [br.chat_sentence, br.job_sentence, br.push_body])
def test_sentences_name_the_date_and_leak_nothing(render):
    with_date = render(DETAIL, "America/Toronto", now=NOW)
    without = render(None, "America/Toronto", now=NOW)
    assert "October 24" in with_date and "monthly" in with_date
    assert "October" not in without and "monthly" not in without
    for text in (with_date, without):
        assert not _FORBIDDEN.search(text), text
        assert len(text) <= 300


def test_chat_sentence_wording():
    assert br.chat_sentence(DETAIL, "America/Toronto", now=NOW) == (
        "Your agent’s monthly AI budget is used up, so it can’t reply until the "
        "budget resets on October 24. Your credits aren’t affected."
    )
    assert br.chat_sentence(None) == (
        "Your agent’s AI budget is used up, so it can’t reply right now. "
        "Your credits aren’t affected."
    )


def test_job_sentence_wording():
    """The wording is "couldn’t finish", not "couldn’t run": the refused job
    often did work first. The undated form is also job_status's static
    model_budget copy and the web/app fallback (tests/test_budget_client_pins.py)."""
    assert br.job_sentence(DETAIL, "America/Toronto", now=NOW) == (
        "This task couldn’t finish because your agent’s monthly AI budget is "
        "used up. It resets on October 24. Your credits aren’t affected."
    )
    undated = ("This task couldn’t finish because your agent’s AI budget is used up. "
               "Your credits aren’t affected.")
    assert br.job_sentence(None) == undated
    assert br.job_sentence(dict(DETAIL, period_end=None), "America/Toronto", now=NOW) == undated
    assert br.job_sentence(DETAIL, "America/Toronto", now=datetime(2026, 11, 1, tzinfo=timezone.utc)) == undated


def test_frame_fields_carry_code_and_iso_reset():
    # frame_fields reads the real clock: the reset must stay in the future.
    end = (datetime.now(timezone.utc) + timedelta(days=20)).replace(microsecond=250000)
    fields = br.frame_fields(dict(DETAIL, period_end=end.replace(tzinfo=None).isoformat()))
    assert fields["code"] == br.REASON and fields["retryable"] is False
    assert fields["resets_at"] == end.replace(microsecond=0).isoformat()
    assert fields["resets_at"].endswith("+00:00")
    assert br.frame_fields(dict(DETAIL, period_end="2020-01-01T00:00:00")) == {
        "code": br.REASON, "retryable": False, "resets_at": None}
    assert br.frame_fields(None) == {"code": br.REASON, "retryable": False, "resets_at": None}
