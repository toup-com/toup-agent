"""`db_span` may never put SQL text — or a parameter — in a log line.

WHY THIS FILE EXISTS

`app/db/db_span.py` sits on `before_cursor_execute`, which is handed the
statement AND its parameters. The parameters of a `messages` INSERT ARE the
user's message: that is why `database.py` sets `hide_parameters=True` on every
engine, and why the 2026-09-15 trail shipped chat content to Loki through an
`llm_proxy_events` INSERT before it did. An instrument built on that hook is
one `%s` away from re-opening the same channel, on a hotter path.

So statement identity is a `verb:table` token where BOTH halves come from a
fixed whitelist. That is a structural guarantee, not a convenience: nothing
`_tok` can return is derived from free text, so no formatting mistake
downstream can leak one.

A privacy guard that cannot fail is decoration, so the mutation these tests
are written against is: make `_tok` return the statement. Every test below
goes RED.
"""
from __future__ import annotations

import logging
import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

SECRET = "my bank password is hunter2 and my address is 12 Elm St"


def test_a_messages_insert_renders_as_a_token_not_as_the_row():
    from app.db.db_span import _tok

    stmt = (
        "INSERT INTO messages (id, conversation_id, role, content) "
        f"VALUES ('m1', 'c1', 'user', '{SECRET}')"
    )
    assert _tok(stmt) == "insert:messages"


def test_no_statement_shape_can_return_free_text():
    """Every branch of the tokenizer, including the ones that find nothing."""
    from app.db.db_span import _TABLES, _VERBS, _tok

    statements = [
        "SELECT conversations.id FROM conversations WHERE id = $1",
        "SELECT day_chats.id FROM day_chats WHERE user_id = $1",
        "UPDATE conversations SET message_count = 1 WHERE id = $1",
        "UPDATE day_chats SET total_tokens = 7 WHERE id = $1",
        f"INSERT INTO messages (content) VALUES ('{SECRET}')",
        "SAVEPOINT sa_1",
        "RELEASE SAVEPOINT sa_1",
        "BEGIN ISOLATION LEVEL READ COMMITTED",
        "COMMIT",
        "SELECT 1",
        "SET statement_timeout = 30000",
        f"INSERT INTO some_table_nobody_whitelisted (x) VALUES ('{SECRET}')",
        f"DELETE FROM another_secret_table WHERE note = '{SECRET}'",
        f"EXPLAIN ANALYZE SELECT '{SECRET}'",
        "WITH RECURSIVE typeinfo_tree AS (SELECT 1) SELECT * FROM messages",
        "",
        None,
        12345,
        {"not": "a statement"},
    ]
    for stmt in statements:
        tok = _tok(stmt)
        assert ":" in tok, f"{tok!r} is not a verb:table token"
        verb, table = tok.split(":", 1)
        assert verb in set(_VERBS) | {"other"}, f"{verb!r} escaped the verb whitelist"
        assert table in _TABLES | {"other", "-"}, (
            f"{table!r} escaped the table whitelist — it came from the statement"
        )
        assert "hunter2" not in tok and "Elm" not in tok
        assert " " not in tok, "a token with a space breaks the key=value line"


def test_the_token_is_bounded_so_a_huge_insert_is_not_a_regex_walk():
    """The scan window is a bound, not a guess: an ORM `select(Conversation)`
    renders every mapped column before its `FROM`, so it has to be kilobytes —
    but a 200 KB `messages` INSERT must never become a 200 KB regex walk on
    the turn path."""
    from app.db.db_span import _SCAN_CHARS, _tok

    assert 1000 <= _SCAN_CHARS <= 8192
    huge = "INSERT INTO messages (content) VALUES ('" + ("x" * 500_000) + "')"
    assert _tok(huge) == "insert:messages"


@pytest.mark.asyncio
async def test_the_emitted_line_carries_no_parameter_text(caplog, tmp_path,
                                                          monkeypatch):
    """End to end: run a real INSERT carrying the secret through a real engine
    with the listeners installed, and read what reached the log."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    import app.db.db_span as ds

    monkeypatch.setattr(
        "app.config.settings.turn_db_span", True, raising=False,
    )
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="cmid-1", channel="web",
                  replace=True)

    eng = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/probe.db",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    ds.install(eng)
    try:
        with caplog.at_level(logging.INFO, logger="app.db.db_span"):
            async with ds.db_span("probe"), eng.begin() as conn:
                await conn.execute(text("CREATE TABLE messages (content TEXT)"))
                await conn.execute(
                    text("INSERT INTO messages (content) VALUES (:c)"),
                    {"c": SECRET},
                )
    finally:
        await eng.dispose()
        ds._reset_for_tests()

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("[PERF] db_span")]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "insert:messages" in line
    for fragment in ("hunter2", "Elm", "VALUES", "content", SECRET):
        assert fragment not in line, f"{fragment!r} reached the log: {line}"


@pytest.mark.asyncio
async def test_the_line_is_single_line_key_equals_value(caplog, monkeypatch):
    """Same grep contract as `[PERF] ws_pre_turn`: Loki splits it without a
    parser, so no spaces inside a value and no newlines anywhere. Emitted
    here from a span that ran NO statements, because that is the rendering
    most likely to produce an empty value."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)

    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        async with ds.db_span("empty"):
            pass
    ds._reset_for_tests()

    line = [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("[PERF] db_span")][0]
    assert "\n" not in line
    body = line[len("[PERF] db_span "):]
    for token in body.split(" "):
        assert "=" in token, token
        key, value = token.split("=", 1)
        assert key and " " not in value, token


@pytest.mark.asyncio
async def test_a_hostile_channel_cannot_forge_a_log_line(caplog, monkeypatch):
    """`ch=` is the one field on either line that comes from the CLIENT.

    `begin_turn` is called with `msg.get("channel")` straight off the WS
    frame — two lines above the handler's own "validate client_tz before ANY
    of it is believed" block — and it is rendered LAST, so a newline in it
    appends an attacker-controlled `[PERF] db_span` line to the stream this
    investigation reads, and a space in it breaks the key=value grep contract
    every consumer of these lines relies on. Truncating to 16 characters is
    not a defence: the payload below is exactly 16.

    Whitelisted, not escaped: `[a-z0-9_.-]` and nothing else."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    hostile = [
        "\n[PERF] db_span ",              # exactly 16 chars: a forged line
        "web nr_throttled=999",           # a forged FIELD on the real line
        "\r\nmission=whatever",
        "we\tb",
        "../../etc/passwd",
        {"not": "a string"},
        None,
        "",
    ]
    for ch in hostile:
        ds._reset_for_tests()
        ds.begin_turn(user_id="u1", client_msg_id="c", channel=ch,
                      replace=True)
        with caplog.at_level(logging.INFO, logger="app.db.db_span"):
            caplog.clear()
            async with ds.db_span("hostile"):
                pass
            host_line = ds.emit_turn_host()
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("[PERF] db_span")]
        assert len(lines) == 1, (ch, lines)
        for line in (lines[0], host_line):
            assert "\n" not in line and "\r" not in line, (ch, line)
            body = line.split(" ", 2)[2]
            for token in body.split(" "):
                assert "=" in token, (ch, token)
            # …and the forged key never becomes a key of its own.
            assert line.count("nr_throttled=") <= 1, (ch, line)
            assert line.endswith("ch=" + line.rsplit("ch=", 1)[1])
            assert line.rsplit("ch=", 1)[1] != ""
    ds._reset_for_tests()


def test_the_channel_sanitiser_keeps_the_real_channels_intact():
    """A whitelist that also mangles `web`/`mobile`/`app`/`whatsapp` would
    make every line unjoinable to the rest of the trail."""
    from app.db.db_span import _safe_channel

    for ch in ("web", "mobile", "app", "whatsapp", "telegram", "voice",
               "sms", "api", "chrome-ext", "x.y_z"):
        assert _safe_channel(ch) == ch
    assert _safe_channel("WEB") == "web"
    assert _safe_channel(None) == "-"
    assert _safe_channel("!!!") == "-"
    assert len(_safe_channel("a" * 100)) == 16


def test_a_turn_is_correlated_by_a_hash_never_by_a_raw_id():
    """`cmid_h` is the join key. A raw client_msg_id in a log line is the thing
    the three-hop trace exists to avoid."""
    import app.db.db_span as ds
    from app.services.cmid import cmid_hash

    ds._reset_for_tests()
    ds.begin_turn(user_id=None, client_msg_id="0cafe000-1111-4222-8333-444455556666",
                  channel="web", replace=True)
    h = ds.turn_cmid_h()
    assert h == cmid_hash("0cafe000-1111-4222-8333-444455556666")
    assert "0cafe000" not in h
    ds._reset_for_tests()


def test_the_hash_has_exactly_one_implementation():
    """It moved to `app/services/cmid.py` so the agent layer can stamp it
    without importing `app.api`. Two copies would drift, and the app-parity
    fixtures only test one of them."""
    from app.api import _turn_trace
    from app.services import cmid as shared

    assert _turn_trace.cmid_hash is shared.cmid_hash
    src = (
        __import__("pathlib").Path(_turn_trace.__file__)
    ).read_text()
    assert "_FNV_PRIME" not in src, "a second FNV implementation is back"
