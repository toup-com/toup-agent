"""User content never reaches a log record.

Why this file exists
--------------------
Round 46 A15. The container trail is shipped to Loki, retained for days, and
read by operators and by agents — it is not a private scratch pad. A survey of
the 2026-09-15 fleet found user content on ORDINARY paths: the whole chat
message on every "play …" request (`[FAST-MEDIA]`), up to 100 characters of
every dictated phrase (`[AGENT] Transcription result`), 100 characters of the
assistant's answer on every turn (`[AGENT] Response`), phone numbers, WhatsApp
JIDs and a user's real name.

Two layers, because neither alone is enough:

  * EXECUTED — the two hot paths are driven with a distinctive token and
    `caplog` is asserted to contain it nowhere. This is the only kind of check
    that notices a NEW logging call added under an old, clean-looking one.
  * GREPPED — the leak sites are enumerated with their owners, and the source
    is searched for the exact patterns. Several of them belong to other lanes
    in this round; a grep stays true the moment their fix lands, and fails the
    moment anybody reintroduces the pattern. The alternative (importing every
    module and exercising it) is not available: some of these files need
    optional native deps that the platform test environment does not have.

`PENDING` is the honest part: a site that is known to still leak and is owned
elsewhere. It may only ever SHRINK. A leak that is not in it and not fixed
fails this file.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. python -m pytest -q \
        tests/test_log_privacy_user_content.py
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import re
from typing import List, Tuple

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]

# A token that cannot occur by accident and is not anybody's data.
TOKEN = "ЖqxZ-user-content-Ж"


# ══════════════════════════════════════════════════════════════════════
# 1. Executed: the two hot paths
# ══════════════════════════════════════════════════════════════════════

def test_fast_media_never_logs_the_users_message(caplog, monkeypatch):
    """`[FAST-MEDIA] Detected play request: %r → query=%r` logged the user's
    ENTIRE chat message, on the path that fires for every ordinary "play …"."""
    from app.api import ws_chat

    import httpx

    class _Boom:
        async def __aenter__(self):
            raise RuntimeError("no network in tests")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Boom())

    text = f"play {TOKEN} by {TOKEN}"
    with caplog.at_level(logging.DEBUG):
        asyncio.run(ws_chat._fast_media_check(text, "u1", asyncio.Queue()))

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "[FAST-MEDIA]" in blob, "the path did not run — the test proves nothing"
    assert TOKEN not in blob, blob
    assert "text_len=" in blob, blob


def test_transcription_never_logs_the_words(caplog, monkeypatch, tmp_path):
    """The dictation path. Whisper's answer is the user speaking."""
    from app.agent import voice_handler

    import httpx

    class _Resp:
        status_code = 200

        def json(self):
            return {"text": TOKEN}

        def raise_for_status(self):
            return None

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Client())
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"\x00" * 64)

    with caplog.at_level(logging.DEBUG):
        out = asyncio.run(voice_handler.transcribe_voice(str(clip), api_key="k"))

    assert out == TOKEN, "the path did not run — the test proves nothing"
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert TOKEN not in blob, blob
    assert "chars=" in blob, blob


# ══════════════════════════════════════════════════════════════════════
# 2. Grepped: the enumerated leak sites
# ══════════════════════════════════════════════════════════════════════
#
# (path, forbidden regex, what it leaks, owning lane)
LEAK_SITES: List[Tuple[str, str, str, str]] = [
    # L3 (this lane) — fixed here.
    ("app/api/ws_chat.py", r"Detected play request: %r",
     "the user's whole chat message", "L3"),
    ("app/api/ws_chat.py", r"No video found for: %s",
     "the title the user asked for", "L3"),
    ("app/agent/voice_handler.py", r"chars\): \{text\[:100\]\}",
     "100 characters of speech", "L3"),
    ("app/agent/voice_handler.py", r"text\[:100\]",
     "100 characters of speech", "L3"),
    # Narrow on purpose: the AgentConfig line further down logs the AGENT's
    # name, which the user chose for it and which is the diagnostic for the
    # Soul-save race. The two `Owner user row` lines log the PERSON's name.
    ("app/api/admin_pool.py", r"Owner user row \w+ for %s \(name=%r\)",
     "the user's real name", "L3"),
    ("app/agent/telegram_bot.py", r"Voice transcribed: \{transcription\[:200\]\}",
     "200 characters of speech", "L3"),
    # L7 — landed (assistant text at the end of every run).
    ("app/agent/agent_runner.py", r"\[AGENT\] Response: \{.*\[:100\]",
     "100 characters of the assistant's answer", "L7"),
    # The sidecar masks in its log helpers rather than at each call site, so
    # the invariant is "the helpers mask", not "no call site passes a phone".
    ("whatsapp_sidecar/sidecar.mjs", r"const log = \(msg, fields = \{\}\) => logger\.info\(fields, msg\);",
     "unmasked phone numbers, JIDs and display names", "L3"),
    # The message-WRITE paths. SQLAlchemy's DBAPIError.__str__ appends
    # `[SQL: INSERT INTO messages ...] [parameters: (...)]`, and those
    # parameters are the user's message text — so formatting the exception
    # into the line ships chat content to Loki. Observed live on an
    # `llm_proxy_events` INSERT in the 2026-09-15 trail.
    ("app/api/ws_chat.py", r"Failed to pre-save user message: \{",
     "the user's message text, via the SQL exception's bound parameters", "B-fix"),
    # The task-intent line printed `title`, which is `text.strip()[:60]` — the
    # first 60 characters of the user's own message — on every turn that
    # matched the task pattern. Same class as the [FAST-MEDIA] line one
    # function away, which this round already reduced to lengths.
    ("app/api/ws_chat.py", r"created agent_task job \{job_id\[:8\]\}: \{title\}",
     "the task title, derived from the user's message", "B-fix"),
]

# Known, still-leaking, owned elsewhere. MAY ONLY SHRINK.
#   (path, regex, what it leaks, owner)
PENDING: List[Tuple[str, str, str, str]] = [
    ("app/api/ws_browser.py", r"LLM returned TEXT \(no tool call\): %s",
     "300 characters of model output", "unowned — reported by L3 to the supervisor"),
    ("app/api/ws_browser.py", r"LLM THINKING: %s",
     "500 characters of model reasoning", "unowned — reported by L3"),
    ("app/services/email_service.py", r"HTTP %d for to=%s",
     "the recipient's email address", "unowned — reported by L3"),
]


def _read(rel: str) -> str:
    return (BACKEND / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("rel,pattern,what,owner", LEAK_SITES,
                         ids=[f"{s[0].split('/')[-1]}:{s[2][:24]}" for s in LEAK_SITES])
def test_known_leak_site_is_fixed(rel, pattern, what, owner):
    src = _read(rel)
    hits = [m for m in re.finditer(pattern, src)]
    # A comment naming the old pattern is allowed — that is how the incident
    # stays explained. Code is not.
    real = [
        m for m in hits
        if not src[:m.start()].rsplit("\n", 1)[-1].lstrip().startswith(("#", "//", "*"))
    ]
    assert not real, f"{rel} still leaks {what} (owner {owner}): {pattern}"


def test_pending_leaks_are_named_and_only_shrink():
    """Each pending entry must still be REAL. When its owner fixes it, this
    test fails and the entry is deleted — which is the point: a permanent
    allowlist is how a known leak becomes a forgotten one."""
    stale = []
    for rel, pattern, what, owner in PENDING:
        if not re.search(pattern, _read(rel)):
            stale.append(f"{rel}: {what} — FIXED, remove it from PENDING")
    assert not stale, "\n".join(stale)


def test_pending_list_has_not_grown():
    """Pinned. A new entry here is a new leak, and it must be a deliberate,
    reviewed act rather than a quiet append."""
    assert len(PENDING) == 3, (
        "PENDING changed — a leak was added (not allowed) or fixed "
        "(delete the entry and lower this number)"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. The wire, not just the log
# ══════════════════════════════════════════════════════════════════════

def test_a_failed_message_insert_cannot_print_the_message():
    """The structural half of the fix: every engine is built with
    `hide_parameters=True`, so even a call site that formats the exception
    (there are hundreds, in every repo) cannot print the bound parameters —
    which on a `messages` INSERT are the user's own words."""
    import inspect

    from app.db import database as db_mod

    src = inspect.getsource(db_mod._build_engine_inner)
    n_engines = src.count("create_async_engine(")
    assert n_engines >= 4, src
    n_flags = len(re.findall(r"^\s*hide_parameters=True,\s*$", src, re.M))
    assert n_flags == n_engines, (
        f"{n_engines} engines but {n_flags} hide_parameters=True — "
        "an engine was added without it"
    )

    # Executed: SQLAlchemy renders the placeholder, not the values.
    from sqlalchemy.exc import DBAPIError

    err = DBAPIError(
        "INSERT INTO messages (id, content) VALUES (?, ?)",
        ("m1", TOKEN),
        Exception("boom"),
        hide_parameters=True,
    )
    assert TOKEN not in str(err), str(err)
    assert "hidden" in str(err)


def test_the_ws_handlers_outer_catch_no_longer_sends_the_exception_text():
    """`send_json({"type":"error","message": str(e)})` put the exception on
    the wire. A SQLAlchemy error carries the statement AND its bound
    parameters — i.e. the user's own message, sent to the user's screen and
    to any proxy in between."""
    src = _read("app/api/ws_chat.py")
    assert '{"type": "error", "message": str(e)}' not in src
    assert "safe_send_fault_ws(websocket, AGENT_INTERNAL" in src


def test_turntrace_logs_a_hash_never_the_id():
    from app.api._turn_trace import cmid_hash, turntrace

    cmid = "82304e59-1111-4222-8333-444455556666"
    h = cmid_hash(cmid)
    assert cmid not in h and len(h) == 8
    assert h == cmid_hash(cmid)
    assert h != cmid_hash("82304e59-1111-4222-8333-444455556667")

    records: List[logging.LogRecord] = []

    class _Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("app.api._turn_trace")
    handler = _Grab()
    logger.addHandler(handler)
    old = logger.level
    logger.setLevel(logging.INFO)
    try:
        turntrace("dispatch", cmid, t_ms=12, outcome="ok")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)

    blob = "\n".join(r.getMessage() for r in records)
    assert "[TURNTRACE]" in blob
    assert cmid not in blob, blob
    assert h in blob, blob


# Produced by EXECUTING the app's own `cmidHash` (src/shared/turnTrace.ts) in
# node, not by re-implementing it here: a Python re-implementation compared to
# Python proves only that this file is self-consistent, and it hid a real
# divergence — the server dropped non-ASCII bytes (`encode("ascii","ignore")`)
# where the app masks each UTF-16 code unit to its low byte, so the three-hop
# join would have silently broken on the first non-ASCII id.
CMID_HASH_VECTORS = [
    ("", "00000000"),
    ("a", "e40c292c"),
    ("82304e59-1111-4222-8333-444455556666", "7a1d8f32"),
    ("0" * 36, "842f8155"),
    ("cm-é-1", "ee194a19"),
    ("παιδί", "17d12e58"),
    ("🙂-cm", "9e4d3e09"),
]


@pytest.mark.parametrize("cmid,expected", CMID_HASH_VECTORS)
def test_turntrace_matches_the_apps_hash(cmid, expected):
    """The join only exists if both sides compute the same number."""
    from app.api._turn_trace import cmid_hash

    assert cmid_hash(cmid) == expected
