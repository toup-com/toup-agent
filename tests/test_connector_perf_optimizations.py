"""Regression test for the 2026-05-11 connector latency reduction.

Three contracts pinned by source-grep so a future refactor can't
silently regress the wins:

  1. `_google_base`, `_microsoft_base`, and the LinkedIn provider
     use a process-pooled httpx.AsyncClient — NOT `async with
     httpx.AsyncClient(...)` per request. Each per-request client
     ate ~200-400 ms of TLS handshake; pooling drops that to ~5 ms.

  2. The dispatcher fires `_resolve_user_preference` and `vault.get`
     in PARALLEL via `asyncio.gather`, using two separate AsyncSession
     instances. Pre-fix they ran serially against the same session
     and stacked ~50-150 ms each.

  3. `gmail__list_messages` and `outlook__list_messages` accept an
     `include_body` boolean. When true the provider returns full
     headers + body for every message inline — the agent does ONE
     tool call instead of `list → loop(get_message)`. This is the
     fast path the LLM should use whenever the user asks to read /
     summarize / show emails.
"""
from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_GOOGLE_BASE = (BACKEND / "app/connectors/_google_base.py").read_text()
_MS_BASE = (BACKEND / "app/connectors/_microsoft_base.py").read_text()
_LI_PROVIDER = (BACKEND / "app/connectors/linkedin/provider.py").read_text()
_DISPATCHER = (BACKEND / "app/services/connector_dispatcher.py").read_text()
_GMAIL_PROVIDER = (BACKEND / "app/connectors/gmail/provider.py").read_text()
_OUTLOOK_PROVIDER = (BACKEND / "app/connectors/outlook/provider.py").read_text()


# ── Pooled HTTP clients ──────────────────────────────────────────────


def _request_path_callsites(source: str) -> list[str]:
    """Lines in `source` that construct `httpx.AsyncClient(...)`
    OUTSIDE the pooled-getter / shutdown helpers. The pool's own
    fast-path + defensive-fallback both live inside `_get_*_client`;
    only constructions OUTSIDE those helpers are the bad pattern
    that re-introduces per-call TLS overhead."""
    # Walk the source line by line; track whether we're inside the
    # `_get_*_client` function body (between its def and the next
    # top-level def at column 0). Anything outside that scope is a
    # request-path callsite — which we want to be empty.
    in_pool_helper = False
    out: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        # Top-level def or async def ends the previous scope.
        if line.startswith(("def ", "async def ")):
            in_pool_helper = bool(
                "_get_google_client" in line
                or "_get_ms_client" in line
                or "_get_li_client" in line
                or "shutdown_" in line
            )
            continue
        if "httpx.AsyncClient(" in line and not stripped.startswith("#") and not in_pool_helper:
            out.append(line)
    return out


def test_google_base_uses_pooled_client_not_per_request():
    """Per-request `async with httpx.AsyncClient(...)` was the
    biggest single source of tool-call latency (200-400ms of TLS
    handshake per call). The pooled-client refactor must stick.

    The pooled getter (`_get_google_client`) can have multiple
    AsyncClient constructions internally (fast path + defensive
    fallback) — but no request-path helper should construct one.
    """
    assert "async def _get_google_client" in _GOOGLE_BASE, (
        "_google_base must expose _get_google_client — without it "
        "each Google call eats a fresh TLS handshake."
    )
    leaked = _request_path_callsites(_GOOGLE_BASE)
    assert leaked == [], (
        f"Found {len(leaked)} httpx.AsyncClient construction(s) "
        f"OUTSIDE the pooled getter — these re-introduce per-call "
        f"TLS handshake.\nOffending lines:\n" + "\n".join(leaked)
    )


def test_microsoft_base_uses_pooled_client():
    assert "async def _get_ms_client" in _MS_BASE
    leaked = _request_path_callsites(_MS_BASE)
    assert leaked == [], (
        f"Found {len(leaked)} httpx.AsyncClient construction(s) "
        f"OUTSIDE the pooled getter in _microsoft_base.py:\n"
        + "\n".join(leaked)
    )


def test_linkedin_provider_uses_pooled_client():
    assert "async def _get_li_client" in _LI_PROVIDER
    leaked = _request_path_callsites(_LI_PROVIDER)
    assert leaked == [], (
        f"Found {len(leaked)} httpx.AsyncClient construction(s) "
        f"OUTSIDE the pooled getter in linkedin/provider.py:\n"
        + "\n".join(leaked)
    )


# ── Parallel dispatcher pre-flight ───────────────────────────────────


def test_dispatcher_runs_preference_and_vault_in_parallel():
    """The two pre-flight DB reads (`_resolve_user_preference` +
    `vault.get`) ran serially before this fix. Parallelizing via
    asyncio.gather + a second AsyncSession saves one DB round-trip
    per tool call."""
    assert "asyncio.gather(" in _DISPATCHER, (
        "Dispatcher must use asyncio.gather to parallelize the "
        "pre-flight DB reads. Sequential awaits add ~50-150ms of "
        "Railway+pgbouncer latency per tool call."
    )
    # The gather must include both vault and preference resolution.
    # We grep for the exact symbols rather than rely on whitespace.
    gather_idx = _DISPATCHER.index("asyncio.gather(")
    block = _DISPATCHER[gather_idx:gather_idx + 600]
    assert "_resolve_user_preference" in block
    assert "vault.get" in block or "_vault_get_with_own_session" in block, (
        "The parallel gather must include the vault read. Without "
        "it the preference call is parallelised with itself and the "
        "win disappears."
    )


def test_dispatcher_uses_separate_session_for_parallel_vault_read():
    """asyncpg refuses two queries on the same connection. The
    parallel vault.get MUST use a separate AsyncSession or the
    dispatcher will throw `InterfaceError: another operation is in
    progress` under load."""
    assert "_vault_get_with_own_session" in _DISPATCHER, (
        "The parallel vault.get needs its own session — the "
        "dispatcher's caller-supplied `db` can't safely run a "
        "concurrent query (asyncpg one-query-per-connection rule)."
    )
    # The helper must open a fresh session via async_session_maker
    # (the proxy that resolves to the rebound engine post-bind).
    helper_idx = _DISPATCHER.index("async def _vault_get_with_own_session")
    helper_body = _DISPATCHER[helper_idx:helper_idx + 400]
    assert "async_session_maker" in helper_body or "_make_session" in helper_body


# ── Inline-body fast path ────────────────────────────────────────────


def test_gmail_list_messages_supports_include_body():
    """The include_body fast path turns the 2-call list+get pattern
    into a single tool call. Cuts both MCP round-trips AND LLM
    iterations in half for "summarize my latest N emails" — the
    most common Gmail use case."""
    # Manifest schema declares the parameter.
    import yaml
    manifest = yaml.safe_load(
        (BACKEND / "app/connectors/gmail/manifest.yaml").read_text(),
    )
    list_msg = next(
        t for t in manifest["tools"] if t["name"] == "gmail__list_messages"
    )
    assert "include_body" in list_msg["input_schema"]["properties"], (
        "gmail__list_messages must accept include_body. Without it "
        "the LLM has to do list → get_message → get_message → ... "
        "for every read, doubling tool-call latency."
    )

    # Provider actually honors the parameter.
    assert "include_body" in _GMAIL_PROVIDER
    assert "_fetch_messages_parallel" in _GMAIL_PROVIDER, (
        "Gmail provider must define a parallel-fetch helper. Doing "
        "per-message GETs serially inside the dispatcher just moves "
        "the per-message latency from MCP→agent to MCP→provider."
    )


def test_outlook_list_messages_supports_include_body():
    """Outlook gets the same fast path via Graph's `$select=body`.
    Unlike Gmail, Graph supports body inline in the list call
    directly — no per-message fan-out needed."""
    import yaml
    manifest = yaml.safe_load(
        (BACKEND / "app/connectors/outlook/manifest.yaml").read_text(),
    )
    list_msg = next(
        t for t in manifest["tools"] if t["name"] == "outlook__list_messages"
    )
    assert "include_body" in list_msg["input_schema"]["properties"]
    assert "include_body" in _OUTLOOK_PROVIDER
    # The provider must dynamically add `body` to the $select param —
    # otherwise Graph returns metadata only and include_body=true is a
    # no-op.
    assert "body" in _OUTLOOK_PROVIDER and "$select" in _OUTLOOK_PROVIDER


# ── Phase timing telemetry ───────────────────────────────────────────


def test_dispatcher_logs_phase_timings():
    """When tool-call latency regresses we need to attribute the
    extra time to a phase (DB pre-flight vs provider call vs audit
    write). Without per-phase logs we end up re-running with extra
    instrumentation under load — slow loop. Pin the phase
    instrumentation so it can't disappear in a future cleanup."""
    assert "phases:" in _DISPATCHER, "phase-timing dict missing"
    for required_phase in ("preflight", "audit_in", "provider", "audit_out"):
        assert f'"{required_phase}"' in _DISPATCHER or f"'{required_phase}'" in _DISPATCHER, (
            f"phase {required_phase!r} not marked in dispatcher. Lose "
            f"this and a future latency regression is impossible to "
            f"attribute without re-deploying with instrumentation."
        )
