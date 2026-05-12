"""Regression tests for the 2026-05-12 "Gmail read is 53.5 s" fix.

The user's screenshot showed: 3 Gmail tool calls totalling ~30 s
(plus ~23 s LLM thinking) to summarise the 14th email. The right
shape is ONE call: `gmail__list_messages(max_results=14)` returns
all 14 with bodies inline.

Three contracts pin the fix:

  1. The manifest's `include_body` default is `true`. Pre-2026-05-12
     it was `false` and the LLM (gpt-5.5) picked the "safe" default
     every time. Flipping the default is the cheapest leverage point —
     it removes the LLM's need to guess.

  2. The dispatcher auto-injects `include_body=true` when the LLM
     omits the field. Safety net for any future LLM that still
     picks the slow default, OR for any caller (tests, scripts) that
     forgets to pass it.

  3. The connector dispatcher hands the pre-decrypted access_token
     to providers via `ConnectorContext.access_token`. Providers
     pull from `ctx.access_token` first instead of re-reading the
     vault — saves one DB round-trip + Fernet decrypt per tool call
     (~100-300 ms on Railway+pgbouncer).

If any of these regresses, the failure mode is non-obvious — the
tool calls still work, they're just slow. The user feels it as
"the agent takes forever to read my emails." Source-grep tests so
the contracts hold up under refactors.
"""
from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_DISPATCHER = (BACKEND / "app/services/connector_dispatcher.py").read_text()
_GMAIL_PROVIDER = (BACKEND / "app/connectors/gmail/provider.py").read_text()
_OUTLOOK_PROVIDER = (BACKEND / "app/connectors/outlook/provider.py").read_text()
_BASE = (BACKEND / "app/connectors/base.py").read_text()
_AGENT_RUNNER = (BACKEND / "app/agent/agent_runner.py").read_text()


# ── Manifest defaults ────────────────────────────────────────────────


def test_gmail_manifest_defaults_to_include_body_true():
    """The single highest-leverage fix: default include_body=true so
    the LLM doesn't have to guess. Drop the default and Gmail reads
    go back to 3-4 tool calls."""
    import yaml
    manifest = yaml.safe_load(
        (BACKEND / "app/connectors/gmail/manifest.yaml").read_text(),
    )
    list_tool = next(
        t for t in manifest["tools"] if t["name"] == "gmail__list_messages"
    )
    props = list_tool["input_schema"]["properties"]
    assert props["include_body"]["default"] is True, (
        "gmail__list_messages must default include_body to true. "
        "Defaulting to false was the root cause of the 53.5 s "
        "summarise-14th-email regression — gpt-5.5 picked the slow "
        "path because the safe schema default said to."
    )
    # max_results default must also cover typical "Nth email"
    # requests in one call. 25 is enough for "my last 5 / 10 / 20
    # emails" and the common "my Nth email" up to N=25.
    assert props["max_results"]["default"] >= 20, (
        "max_results default must be high enough that a single call "
        "handles common 'recent N emails' asks without a follow-up "
        "list call. Anything <20 forces a second list to find N=14 "
        "and up."
    )


def test_outlook_manifest_defaults_to_include_body_true():
    """Same contract for Outlook — Graph supports body inline via
    $select, so the speedup is even cleaner than Gmail's parallel
    fan-out."""
    import yaml
    manifest = yaml.safe_load(
        (BACKEND / "app/connectors/outlook/manifest.yaml").read_text(),
    )
    list_tool = next(
        t for t in manifest["tools"] if t["name"] == "outlook__list_messages"
    )
    props = list_tool["input_schema"]["properties"]
    assert props["include_body"]["default"] is True
    assert props["max_results"]["default"] >= 20


# ── Dispatcher safety net ────────────────────────────────────────────


def test_dispatcher_auto_injects_include_body_when_missing():
    """Some LLMs write `include_body: false` (or omit the field
    entirely) even when the schema says default-true, because their
    tool-choice strategy treats explicit defaults as the safest pick.
    The dispatcher's safety net forces include_body=true when the
    LLM omitted it — explicit `false` is still honoured for the rare
    bulk-enumeration use case."""
    # Both transports must be guarded.
    for tool_name in ("gmail__list_messages", "outlook__list_messages"):
        assert tool_name in _DISPATCHER, (
            f"dispatcher must mention {tool_name} in the auto-inject "
            f"guard. Drop the guard and the LLM regressions to "
            f"list→get pattern."
        )
    # The actual injection line — pin the shape so a future refactor
    # can't quietly null it out.
    assert (
        '"include_body" not in tool_input' in _DISPATCHER
        and "include_body" in _DISPATCHER
    ), (
        "dispatcher must auto-inject include_body=True when the field "
        "is missing from tool_input. Removing this guard re-opens the "
        "53.5 s regression."
    )


# ── Pre-decrypted access token threading ────────────────────────────


def test_connector_context_carries_access_token():
    """The dispatcher already paid the vault.get + Fernet decrypt
    cost during pre-flight. Re-doing it in every provider added
    ~100-300 ms per call. Threading the token through
    ConnectorContext.access_token eliminates the duplicate."""
    assert "access_token: Optional[str] = None" in _BASE, (
        "ConnectorContext must expose access_token. Without it "
        "providers re-fetch + re-decrypt the identity per call — "
        "the second DB round-trip was a measurable per-call latency."
    )


def test_dispatcher_passes_access_token_to_provider():
    """Wiring check: it's not enough for ConnectorContext to HAVE the
    field; the dispatcher must populate it. Pin the population so a
    future refactor of the ctx construction can't quietly drop it."""
    assert "access_token=identity.access_token" in _DISPATCHER, (
        "dispatcher must pass identity.access_token into "
        "ConnectorContext so providers can skip the duplicate "
        "vault.get."
    )


def test_all_oauth_providers_prefer_ctx_access_token():
    """Every OAuth provider should use `ctx.access_token` first and
    fall back to the local resolver only when None (covers tests
    that build ctx by hand). Without this each provider keeps doing
    the duplicate vault.get."""
    providers = ["gmail", "outlook", "calendar", "drive", "docs", "github", "linkedin"]
    for p in providers:
        src = (BACKEND / f"app/connectors/{p}/provider.py").read_text()
        assert "ctx.access_token" in src, (
            f"{p} provider must use ctx.access_token. Without it the "
            f"dispatcher's pre-decrypted token is wasted — provider "
            f"re-reads + re-decrypts the identity from scratch."
        )


# ── System prompt guidance ───────────────────────────────────────────


def test_system_prompt_pins_one_call_pattern_for_email_reads():
    """Belt-and-braces: even with manifest defaults + auto-inject,
    the system prompt should teach the LLM why one call is the right
    shape. The 'Nth email' pattern is the most common slip — pin a
    concrete example."""
    # Concrete example anchors the LLM's behaviour better than
    # abstract instruction.
    assert (
        "summarize my 14th email" in _AGENT_RUNNER
        or "summarize my Nth email" in _AGENT_RUNNER
        or "summarise my" in _AGENT_RUNNER.lower()
        or "my last N emails" in _AGENT_RUNNER
    ), (
        "system prompt must include a concrete 'Nth email' example "
        "so the LLM can pattern-match against the actual ask shape. "
        "Abstract guidance ('use include_body=true') wasn't enough — "
        "the production miss was on a literal 'summarize my 14th "
        "email' query."
    )
    # And it must warn against the wrong pattern by name.
    assert (
        "gmail__get_message" in _AGENT_RUNNER
        and "WRONG" in _AGENT_RUNNER
    ), (
        "system prompt must explicitly call out the list→get "
        "pattern as WRONG. Implicit hints didn't land — be explicit."
    )
