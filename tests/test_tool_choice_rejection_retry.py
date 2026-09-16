"""A provider that refuses the `allowed_tools` tool_choice SHAPE gets one
unrestricted retry on the same model — not three identical 400s and a silent
fall-back to a weaker model.

Observed 2026-09-15 on the rollout canary (image ae6365a5f358, gpt-5.6-terra,
Responses wire): `Invalid value: 'allowed_tools'` on `tool_choice.type`, three
times, then gpt-4o over the chat wire — three minutes for two output tokens,
and the rollout aborted `aborted_canary_failed`. The restriction is intent
gating that rides outside the cached prefix (PR-1); the tool policy is
enforced at execute time either way, so dropping it costs nothing but one
round-trip and the model stays the one the tenant chose.

Pure helper tested directly; the run() wiring is pinned source-grep style
(the runner needs a full boot to execute — same pattern as
test_overflow_rollover.py and test_stable_prefix.py).

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. pytest tests/test_tool_choice_rejection_retry.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.agent.prefix_stability import is_tool_choice_rejection

BACKEND = Path(__file__).resolve().parents[1]
RUNNER = (BACKEND / "app/agent/agent_runner.py").read_text()

_OPENAI_BODY = {
    "error": {
        "message": (
            "Invalid value: 'allowed_tools'. Supported values are: 'code_interpreter', "
            "'programmatic_tool_calling', 'function', 'namespace', 'tool_search', "
            "'file_search', 'web_search_preview', 'image_generation', 'mcp', 'custom', "
            "'computer', 'shell', and 'apply_patch'."
        ),
        "type": "invalid_request_error",
        "param": "tool_choice.type",
        "code": "invalid_value",
    }
}


class _SdkError(Exception):
    """The shape openai.BadRequestError exposes: status_code + body + text."""

    def __init__(self, message: str, *, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _proxy_wrapped():
    # The platform proxy relays OpenAI's JSON as a STRING under `detail`,
    # which is exactly what the canary's log line shows.
    detail = json.dumps(_OPENAI_BODY, indent=2)
    return _SdkError(f"Error code: 400 - {{'detail': {detail!r}}}",
                     status_code=400, body={"detail": detail})


# ── the classifier ────────────────────────────────────────────────────


def test_the_canary_error_as_relayed_by_the_proxy_is_a_rejection():
    assert is_tool_choice_rejection(_proxy_wrapped())


def test_a_raw_openai_body_is_a_rejection():
    err = _SdkError("Error code: 400 - invalid tool_choice", status_code=400, body=_OPENAI_BODY)
    assert is_tool_choice_rejection(err)


def test_a_body_naming_any_tool_choice_param_is_a_rejection():
    body = {"error": {"message": "unknown field", "param": "tool_choice.tools[0].name",
                      "code": "unknown_parameter"}}
    assert is_tool_choice_rejection(_SdkError("Error code: 400", status_code=400, body=body))


def test_text_only_fallback_when_the_sdk_gives_no_body():
    err = Exception(
        "Error code: 400 - {'detail': '{\"error\": {\"message\": \"Invalid value: "
        "\\'allowed_tools\\'\", \"param\": \"tool_choice.type\", \"code\": \"invalid_value\"}}'}"
    )
    assert is_tool_choice_rejection(err)


@pytest.mark.parametrize("err", [
    # a context overflow is a 400 too, and has its own handling
    _SdkError("Error code: 400 - This model's maximum context length is 128000 tokens.",
              status_code=400, body={"error": {"code": "context_length_exceeded",
                                               "message": "too big", "param": "messages"}}),
    # a rate limit, however it is worded
    _SdkError("Error code: 429 - rate_limit_exceeded (tool_choice retry later)", status_code=429),
    # a 5xx whose text happens to mention the field
    _SdkError("Error code: 500 - upstream error on tool_choice", status_code=500,
              body={"error": {"param": "tool_choice.type", "code": "server_error"}}),
    # a 400 about the tools ARRAY, not the choice
    _SdkError("Error code: 400 - Invalid value", status_code=400,
              body={"error": {"param": "tools[3].function.name", "code": "invalid_value",
                              "message": "Invalid value: name too long"}}),
    # transport
    Exception("APIConnectionError: connection reset"),
    # a 400 with a body that is not JSON at all
    _SdkError("Error code: 400 - Bad Request", status_code=400, body={"detail": "<html>nope</html>"}),
])
def test_everything_else_is_not_a_rejection(err):
    assert not is_tool_choice_rejection(err), str(err)


def test_the_classifier_never_raises_on_a_hostile_body():
    class _Weird:
        status_code = 400
        body = {"detail": object()}

        def __str__(self):
            return "Error code: 400"

    assert is_tool_choice_rejection(_Weird()) is False  # noqa: E721 — not an exception, still safe


# ── the run() wiring (source pins) ────────────────────────────────────


def _between(start_pat: str, end_pat: str) -> str:
    a = RUNNER.index(start_pat)
    b = RUNNER.index(end_pat, a)
    return RUNNER[a:b]


def test_the_restriction_is_not_rebuilt_once_rejected():
    site = _between("_tool_choice = build_allowed_tools_choice(", "prompt_cache_key")
    head = RUNNER[RUNNER.rindex("if (", 0, RUNNER.index("_tool_choice = build_allowed_tools_choice(")):
                   RUNNER.index("_tool_choice = build_allowed_tools_choice(")]
    assert "and not _tool_choice_restriction_rejected" in head, (
        "the build site no longer consults the rejection flag — the retry "
        "would re-send the very shape the provider refused"
    )
    assert "_tool_choice_restriction_rejected = False" in RUNNER, "the flag is never initialised"
    assert site  # anchor sanity


def test_the_rejection_branch_retries_before_the_overflow_and_generic_ladders():
    handler = _between("except Exception as e:", "if attempt < MAX_RETRIES:")
    i_reject = handler.find("is_tool_choice_rejection(e)")
    i_overflow = handler.find("is_context_overflow_error(e)")
    assert i_reject > 0, "the retry loop never asks is_tool_choice_rejection"
    assert i_overflow > i_reject, "the rejection check must run BEFORE the overflow ladder"
    branch = handler[i_reject:i_overflow]
    assert "_tool_choice_restriction_rejected = True" in branch, "the flag is not set on rejection"
    assert re.search(r"\n\s+continue\n", branch), "the branch does not retry immediately (no `continue`)"
    assert "attempt < MAX_RETRIES" in handler[max(0, i_reject - 400):i_reject], (
        "the unrestricted retry is not bounded by the attempt budget"
    )
    code_only = "\n".join(l for l in branch.splitlines() if not l.strip().startswith("#"))
    assert "fallback" not in code_only and "active_model =" not in code_only, (
        "the unrestricted retry must stay on the SAME model — a model swap here "
        "is the silent downgrade this fix removes"
    )
    assert "isinstance(_tool_choice, dict)" in handler[max(0, i_reject - 400):i_reject], (
        "the branch must only fire when a restriction was actually sent"
    )


def test_the_retry_is_counted_and_the_signal_is_known():
    from app.services.health_signals import KNOWN_SIGNALS

    assert "llm_tool_choice_rejected" in KNOWN_SIGNALS
    handler = _between("is_tool_choice_rejection(e)", "is_context_overflow_error(e)")
    assert 'incr("llm_tool_choice_rejected")' in handler


def test_the_log_line_names_no_content():
    """The warning carries the model and the exception CLASS — never the
    exception text, which for a 400 can quote the request."""
    handler = _between("is_tool_choice_rejection(e)", "is_context_overflow_error(e)")
    m = re.search(r"logger\.warning\((.*?)\)\s*\n\s*try:", handler, re.S)
    assert m, "no warning in the branch"
    args = m.group(1)
    assert "type(e).__name__" in args
    assert not re.search(r"\bstr\(e\)|%s\", e\b|,\s*e\s*,|,\s*e\s*\)", args), args
