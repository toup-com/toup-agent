"""Regression tests for the 2026-05-12 system-prompt fixes.

When the user said "summarize my 12th email" the agent (gpt-5.5):
  1. Did NOT use `include_body=true` on `gmail__list_messages` —
     so 57.8 s for 4 sequential tool calls instead of one ~5 s call.
  2. After tokens expired, fell back to the `browser` tool — useless
     because Chromium isn't installed in tenant containers.

The MCP-side fix keeps the tool visible to the agent when expired.
The system-prompt fix tells the LLM exactly how to behave: surface
reauth errors clearly, and ALWAYS use the include_body fast path
for email reads.

Source-grep style tests so a future system-prompt refactor can't
quietly drop the guidance. The exact wording can drift; what MUST
stay is the SIGNAL — three behavioral rules embedded somewhere in
the connected-services block.
"""
from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_RUNNER = (BACKEND / "app/agent/agent_runner.py").read_text()


def test_system_prompt_tells_llm_to_surface_reauth_required():
    """The LLM must know what to do with `[reauth_required]` so it
    surfaces a clean reconnect message to the user instead of falling
    back to the browser tool."""
    assert "reauth_required" in _RUNNER, (
        "agent_runner system prompt must mention `reauth_required` so "
        "the LLM knows to handle that variant explicitly. Without it "
        "the LLM treats the error generically and may try `browser` "
        "next, which fails (no Chromium in the container)."
    )
    # And the prompt should mention reconnecting — that's the action
    # the user needs to take. Vague hand-wave like "try again" isn't
    # enough; the user has to know the recovery is reconnect, not
    # retry.
    assert "reconnect" in _RUNNER.lower(), (
        "system prompt must instruct the LLM to tell the user to "
        "reconnect when a tool returns reauth_required — otherwise "
        "the user sees a vague error and doesn't know what to do."
    )


def test_system_prompt_forbids_browser_fallback_for_connector_tasks():
    """The browser tool fails in production because Chromium isn't
    installed in tenant runtime containers. Even if it weren't, it's
    the wrong tool for authenticated services. Pin the negative."""
    # We grep for the explicit "do NOT use the browser tool"
    # instruction (case-insensitive). The exact phrasing can shift
    # but the signal must remain.
    text = _RUNNER.lower()
    assert (
        ("do not use the `browser`" in text)
        or ("do not use the browser" in text)
        or ("not use the `browser` tool to access" in text)
        or ("never use the browser" in text)
        or ("don't try the `browser` tool" in text)
        or ("not try the `browser` tool" in text)
    ), (
        "system prompt must explicitly tell the LLM not to fall back "
        "to the browser tool when connector tools exist for the same "
        "service — otherwise the agent picks browser when a connector "
        "errors and the user sees a useless 'Open Live Browser' link."
    )


def test_system_prompt_pushes_include_body_fast_path():
    """The `gmail__list_messages` include_body=true path is 4× faster
    for the read/summarize use case. The schema description already
    mentions it; pinning the system-prompt mention here because the
    LLM ignored the schema hint in production and did list→get→get
    sequentially anyway. Two reinforcing signals beat one."""
    assert "include_body" in _RUNNER, (
        "system prompt must mention include_body — otherwise the LLM "
        "falls back to the slow list+loop pattern (57.8 s for 12 "
        "emails vs ~5 s with the fast path)."
    )
    # And the hint must be wired to the Gmail connector specifically
    # — generic "use fast paths when available" is too vague.
    assert "gmail__list_messages" in _RUNNER, (
        "system prompt must name `gmail__list_messages` so the LLM "
        "knows WHICH tool gets the include_body parameter, not just "
        "the general concept."
    )


def test_system_prompt_includes_provider_down_and_rate_limited_guidance():
    """While we're surfacing reauth_required, also pin the other
    structured error variants so the LLM has clean handling for each.
    Drop either of these and the agent's response to a real outage
    becomes 'the tool failed' instead of 'Gmail is having an outage,
    try again in a moment'."""
    assert "provider_down" in _RUNNER, (
        "system prompt must mention provider_down so the LLM can "
        "tell the user 'the service is having an outage' instead of "
        "blaming the tool generically."
    )
    assert "rate_limited" in _RUNNER, (
        "system prompt must mention rate_limited so the LLM waits "
        "instead of hammering the provider and burning retries."
    )
