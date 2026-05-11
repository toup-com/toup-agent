"""Regression test for the 2026-05-11 hotfix that re-enabled Gmail.

The perf commit set `http2=True` on the pooled httpx.AsyncClient.
httpx requires the `h2` extra (`pip install httpx[http2]`) for
HTTP/2, and our `requirements.platform.txt` only pins plain
`httpx>=0.28.1`. The unmet import killed `_get_google_client()` on
its first call — every Gmail dispatch threw ImportError, the agent
saw repeated tool failures, and `gpt-5.5` flipped to the
`browser` fallback strategy ("the live browser runtime is missing
its Chromium install — classic 'car is fine, engine isn't in it'"),
which itself failed because Playwright's Chromium isn't installed
in the runtime container either. Net effect: Gmail completely
unusable from chat.

Three contracts pinned here so the regression can't quietly come
back:

  1. None of the three pooled AsyncClient constructions sets
     http2=True. This dep would need an explicit requirements.txt
     change to be safe — pin the negative now.

  2. Every `_get_*_client` wraps construction in try/except and falls
     back to a per-call client (slower but functional) on any error.
     Loud ERROR log per fallback so the operator notices.

  3. The dispatcher's parallel pre-flight gather has a fallback to
     serial — so any future asyncpg/session-pool gotcha degrades
     latency, doesn't break dispatch.
"""
from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_GOOGLE = (BACKEND / "app/connectors/_google_base.py").read_text()
_MS = (BACKEND / "app/connectors/_microsoft_base.py").read_text()
_LI = (BACKEND / "app/connectors/linkedin/provider.py").read_text()
_DISPATCHER = (BACKEND / "app/services/connector_dispatcher.py").read_text()


def test_no_http2_on_pooled_clients():
    """`httpx[http2]` is not in requirements.platform.txt. Setting
    `http2=True` on the pooled client raises ImportError on first
    use and kills every Gmail call — see the 2026-05-11 hotfix
    commit for the original incident."""
    for fname, source in (
        ("_google_base.py", _GOOGLE),
        ("_microsoft_base.py", _MS),
        ("linkedin/provider.py", _LI),
    ):
        # Tolerate `http2` appearing inside comments / docstrings.
        # The forbidden pattern is `http2=True` as a kwarg.
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            assert "http2=True" not in stripped, (
                f"{fname}: pooled AsyncClient must NOT set "
                f"http2=True — the `h2` extra isn't installed in "
                f"requirements.platform.txt and the construction "
                f"hard-fails on first use. Pin the negative.\n"
                f"Offending line: {line!r}"
            )


def test_pooled_clients_have_defensive_fallback():
    """Each pooled-client helper wraps construction in try/except
    and falls back to a per-call client on failure. Without this
    fallback, ANY future config error in the pool path takes the
    connector down silently."""
    for fname, source in (
        ("_google_base", _GOOGLE),
        ("_microsoft_base", _MS),
        ("linkedin/provider", _LI),
    ):
        # The fallback is an `except Exception as e:` inside the
        # `_get_*_client` function that constructs a fresh
        # AsyncClient. Grep both halves.
        assert "except Exception" in source, (
            f"{fname}: pooled-client helper needs a try/except "
            f"around AsyncClient construction. Without it, any "
            f"future incompatible kwarg (like the http2 bug) "
            f"silently breaks every call until someone notices."
        )
        # The error message should be logged at ERROR so the alert
        # fires even though dispatch keeps succeeding.
        assert (
            "logger.error" in source.lower()
            or 'logger.error("' in source
        ), (
            f"{fname}: fallback must log at ERROR — silent fallback "
            f"is worse than a hard failure (slow drift, no signal)."
        )


def test_dispatcher_gather_has_serial_fallback():
    """The parallel pre-flight (asyncio.gather of preference +
    vault.get) is a real win but introduces a new failure mode:
    asyncpg contention, session-construction errors, etc. Wrap it
    in try/except so the serial path runs as fallback — slower but
    functional."""
    # Locate the gather block and verify the fallback comes
    # immediately after via the warning log line.
    gather_idx = _DISPATCHER.index("asyncio.gather(")
    after = _DISPATCHER[gather_idx:gather_idx + 1500]
    assert "except Exception" in after, (
        "dispatcher's parallel gather must be wrapped in a "
        "try/except — a runtime issue in the gather (asyncpg or "
        "session pool contention) should NOT take the entire "
        "tool-dispatch path down."
    )
    assert "_resolve_user_preference(" in after, (
        "serial fallback must call _resolve_user_preference — "
        "otherwise we lose the preference branch entirely on "
        "fallback."
    )
    assert "vault.get(" in after, (
        "serial fallback must also call vault.get — otherwise "
        "the dispatcher proceeds without an identity and the "
        "wrong-shape NoneType crashes downstream."
    )
