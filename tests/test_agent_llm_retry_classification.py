"""A 402 is not retryable, and the turn-level ladder used to think it was.

`run_turn` handles the typed `OutOfCreditsError`, but that path only exists on
the manual-mode deduct route. A bundle turn meets `llm_proxy`'s credit
pre-flight, which answers HTTP 402 with `{"error": "out_of_credits", ...}` and
reaches the SDK as an unlisted `APIStatusError` — unclassified, so the generic
ladder re-sent the identical refused request twice more and then sent a THIRD
to the fallback model. Three guaranteed pre-flight refusals per refused turn,
a measured ~6.6 s median of extra delay before the user's paywall card, and
none of it could ever succeed: retries and the cross-provider hop both go back
through the same gate.

The wire-level ladder in `openai_agent_service.py` already escapes on the first
attempt and is deliberately untouched here.

Two halves, same as test_tool_choice_rejection_retry.py: the classifier is
executed directly, and the wiring inside `run_turn` is pinned by source — the
runner needs a full boot to execute, and a guard whose precondition something
above it destroys is invisible to every other check in this repo, so the pins
assert ORDER, not just presence.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. pytest tests/test_agent_llm_retry_classification.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.agent_runner import _is_credit_refusal

BACKEND = Path(__file__).resolve().parents[1]
RUNNER = (BACKEND / "app/agent/agent_runner.py").read_text()


class _SdkError(Exception):
    """The shape openai.APIStatusError exposes: status_code + body + text."""

    def __init__(self, message: str, *, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _proxy_402(reason: str = "insufficient_message_credits") -> _SdkError:
    """What the SDK raises when `llm_proxy`'s pre-flight returns 402 —
    FastAPI's HTTPException detail, relayed verbatim in the error text."""
    detail = {
        "error": "out_of_credits",
        "reason": reason,
        "bucket": "message",
        "balance_after": "0.00",
    }
    body = {"detail": detail}
    return _SdkError(f"Error code: 402 - {json.dumps(body)}",
                     status_code=402, body=body)


# ── the classifier ────────────────────────────────────────────────────


def test_the_proxy_402_as_the_sdk_raises_it_is_a_credit_refusal():
    assert _is_credit_refusal(_proxy_402())


def test_a_daily_cap_402_is_a_credit_refusal_too():
    """A cap refusal is a different copy on the phone but the same verdict
    here: the turn cannot run, and re-sending it is pure delay."""
    assert _is_credit_refusal(_proxy_402("daily_cap_exceeded"))


def test_the_exhausted_error_string_shape_is_a_credit_refusal():
    """`credit_exhausted.OutOfCreditsError.__str__` is
    `out_of_credits:<reason>:<bucket>` — it is handled by the typed branch
    above the ladder, but must not be misread if it ever arrives wrapped."""
    assert _is_credit_refusal(Exception("out_of_credits:daily_cap_exceeded:message"))


def test_a_bare_insufficient_reason_without_the_wrapper_is_a_credit_refusal():
    """`/credits/agent-deduct` and the integration bucket report the reason
    without the `out_of_credits` envelope."""
    assert _is_credit_refusal(Exception("charge refused: insufficient_message_credits"))
    assert _is_credit_refusal(Exception("insufficient_integration_credits"))


@pytest.mark.parametrize("err", [
    # the UPSTREAM provider's own balance error: a 400, a different fix, and
    # crossing to the other provider IS the right move for it
    _SdkError("Error code: 400 - Your credit balance is too low to access the "
              "Anthropic API", status_code=400),
    _SdkError("Error code: 429 - insufficient_quota", status_code=429),
    # the ordinary transient classes the ladder exists for
    _SdkError("Error code: 429 - rate_limit_exceeded", status_code=429),
    _SdkError("Error code: 401 - authentication_error", status_code=401),
    _SdkError("Error code: 500 - upstream error", status_code=500),
    _SdkError("Error code: 400 - This model's maximum context length is 128000 tokens.",
              status_code=400),
    Exception("APIConnectionError: connection reset"),
    # a 402 from something that is not the credit gate at all
    _SdkError("Error code: 402 - Payment Required", status_code=402),
])
def test_everything_else_is_not_a_credit_refusal(err):
    assert not _is_credit_refusal(err), str(err)


def test_the_classifier_never_raises_on_a_hostile_error():
    class _Weird(Exception):
        def __str__(self):
            return "Error code: 402 - OUT_OF_CREDITS"

    # Case-insensitive, and no attribute is touched but str().
    assert _is_credit_refusal(_Weird()) is True


# ── the run() wiring (source pins) ────────────────────────────────────


def _handler() -> str:
    """The generic classification block of run_turn's retry handler."""
    a = RUNNER.index("# Detect errors that warrant immediate cross-provider fallback")
    b = RUNNER.index("if attempt < MAX_RETRIES:", a)
    return RUNNER[a:b]


def test_the_ladder_asks_the_classifier_at_all():
    assert "_is_credit_refusal(e)" in _handler(), (
        "the retry loop never classifies a credit refusal — three proxy calls "
        "per refused turn is the behaviour that returns"
    )


def test_the_credit_branch_runs_before_the_cross_provider_decision():
    """ORDER, not presence. `_should_cross_provider` is what sends the third
    call to the other provider; classifying after it would leave the extra
    round-trip in place while the code read as fixed."""
    h = _handler()
    i_credit = h.index("_is_credit_refusal(e)")
    i_cross = h.index("_should_cross_provider =")
    assert i_credit < i_cross


def test_the_credit_branch_raises_instead_of_retrying():
    """`attempt = MAX_RETRIES` (the auth-error move) is NOT enough: it skips
    the retries but still falls into the fallback-model arm, which is a third
    proxy call through the same gate."""
    h = _handler()
    branch = h[h.index("if _is_out_of_credits:"):h.index("_should_cross_provider =")]
    code_only = "\n".join(
        l for l in branch.splitlines() if not l.strip().startswith("#")
    )
    assert "\n                        raise\n" in code_only, (
        "the branch does not re-raise — it must not reach the fallback arm"
    )
    assert "attempt = MAX_RETRIES" not in code_only
    assert "fallback" not in code_only
    assert "continue" not in code_only


def test_the_raise_is_bare_so_the_402_body_survives_to_ws_chat():
    """`ws_chat._extract_out_of_credits_detail(str(e))` lifts the proxy's own
    402 payload out of the stringified exception and builds the
    `credit_exhausted` frame from it. Wrapping the error in a RuntimeError (the
    context-overflow branch's move) would leave the user an error bubble with
    no door to the fix — the 2026-08-17 founder-device report."""
    h = _handler()
    branch = h[h.index("if _is_out_of_credits:"):h.index("_should_cross_provider =")]
    assert "raise RuntimeError" not in branch
    assert "from e" not in branch


def test_the_credit_branch_still_writes_the_error_row():
    """The no-fallback arm below writes an `llm_error` row before re-raising.
    The only thing this branch may remove is the two wasted attempts."""
    h = _handler()
    branch = h[h.index("if _is_out_of_credits:"):h.index("_should_cross_provider =")]
    assert "self._log_error(" in branch
    assert 'error_type="llm_error"' in branch


def test_the_log_line_names_no_content():
    """The model and the exception CLASS. A 402 body carries a balance, and the
    exception text can quote the request."""
    h = _handler()
    branch = h[h.index("if _is_out_of_credits:"):h.index("self._log_error(")]
    assert "type(e).__name__" in branch
    assert "%s" in branch
    # `str(e)` belongs in the DB row (2000 chars, operator-only), never in the
    # rolling log line.
    assert "str(e)" not in branch


def test_the_wire_level_ladder_retries_only_named_transient_classes():
    """`openai_agent_service` escapes on the FIRST attempt for a 402 because
    its retry ladder catches three named SDK classes and nothing wider — a 402
    arrives as an unlisted `APIStatusError`. Pinned by the catch list, not by
    grepping for "402": a broad `except APIStatusError`/`except APIError`
    there would silently re-open the loop under this fix."""
    import re

    wire = (BACKEND / "app/services/openai_agent_service.py").read_text()
    caught = set(re.findall(r"except\s+(\w+(?:Error|Exception))", wire))
    assert caught <= {
        "AuthenticationError", "RateLimitError", "APIConnectionError",
        "Exception", "JSONDecodeError", "CancelledError", "ValueError",
        "KeyError", "TypeError", "AttributeError", "RuntimeError",
    }, f"a wider SDK error class joined the wire-level ladder: {caught}"
    assert "APIStatusError" not in caught
    assert "APIError" not in caught
