"""P0: a spoken sentence may never reach a request line.

Every voice turn used to be persisted with `role` AND `content` as QUERY
PARAMETERS — a compatibility shim for agent images that predate the body-aware
`POST /api/sessions/{id}/messages` route. uvicorn logs the request line, and the
fleet ships those logs to Loki, so the shim wrote the user's full utterance,
percent-encoded but perfectly readable, into an operational log on every single
turn. Fifteen such lines were observed in one four-minute window on 2026-09-15
(container toup-agent-pool-88).

The on-device recogniser's entire premise is that the words do not leave the
device unnecessarily. A transcript in an access log undoes that quietly, forever,
on a surface nobody reads as a transcript.

These are pure-function tests: no DB, no network, platform sweep.
"""
from __future__ import annotations

import pytest


def test_message_payload_never_returns_query_params():
    from app.api.ws_realtime import _message_payload

    cases = [
        ("user", ""),
        ("user", "hi"),
        ("assistant", "Short reply."),
        # Long, non-Latin, emoji, and a URL-ish string — every shape that used
        # to decide whether the shim applied.
        ("user", "سلام حال شما چطور است" * 50),
        ("user", "a" * 20000),
        ("assistant", "https://example.test/?q=secret&x=1"),
        ("user", "🙂" * 100),
    ]
    for role, content in cases:
        body, params = _message_payload(role, content)
        assert params is None, f"query params returned for {role}/{len(content)} chars"
        assert body["role"] == role
        assert body["content"] == content


def test_message_payload_carries_every_field_in_the_body():
    from app.api.ws_realtime import _message_payload
    from datetime import datetime

    when = datetime(2026, 9, 15, 12, 0, 0)
    body, params = _message_payload(
        "assistant", "done", "gpt-realtime",
        {"type": "youtube", "video_id": "x"},
        [{"tool": "web_search"}],
        attachments=[{"id": "a1", "filename": "report.pdf"}],
        app_artifact={"slug": "snake"},
        client_msg_id="cmid-1",
        occurred_at=when,
    )
    assert params is None
    assert body["media"]["video_id"] == "x"
    assert body["tool_events"]
    assert body["attachments"][0]["id"] == "a1"
    assert body["app_artifact"]["slug"] == "snake"
    assert body["client_msg_id"] == "cmid-1"
    assert body["occurred_at"] == when.isoformat()


def test_message_payload_omits_absent_fields():
    """An older platform build and this one must be indistinguishable on the
    wire for a turn that produced nothing new."""
    from app.api.ws_realtime import _message_payload

    body, _ = _message_payload("user", "hello")
    for key in ("attachments", "app_artifact", "client_msg_id", "occurred_at",
                "media", "tool_events", "model_used"):
        assert key not in body


@pytest.mark.asyncio
async def test_save_voice_messages_issues_body_only_requests(monkeypatch):
    from app.api import ws_realtime

    seen: list[dict] = []

    async def _fake_vps_api(agent_url, key, method, path, params=None, json_body=None, **kw):
        seen.append({"params": params, "body": json_body, "path": path})
        return {"ok": True}

    async def _fake_vps_info(_uid):
        return ("https://agent.test", "key")

    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_vps_info)

    await ws_realtime._save_voice_messages(
        "u1", "s1", "what is my balance", "It is forty dollars.",
    )

    assert len(seen) == 2
    for call in seen:
        assert call["params"] is None, "the transcript must never ride the query string"
        # …and nothing identifying is in the PATH either.
        assert "what is my balance" not in call["path"]
        assert "forty dollars" not in call["path"]
    assert seen[0]["body"]["content"] == "what is my balance"
    assert seen[1]["body"]["content"] == "It is forty dollars."


# ── The same content, one layer up: the platform's own log lines ───────────

def test_the_voice_log_lines_carry_shapes_and_hashes_not_content():
    """Two neighbours of the P0 wrote the same user content to the same sink.

    `[REALTIME] Function call: %s(%s)` logged the FULL arguments of every tool
    call on the socket — and `think`'s `task` is the realtime model's verbatim
    synthesis of what the user just said, while `voice_task`'s `message` runs
    to 20 000 characters. `[REALTIME] Injected text:` logged 60 characters of
    whatever the app sent through `injectText`, which is a free-text public API
    of the hook. Both go to the process log and the fleet ships it.

    Source probes, because the alternative is asserting on a log record from a
    driven socket, and the property is about what the FORMAT STRING can carry.
    """
    import inspect
    import re

    from app.api import ws_realtime

    src = inspect.getsource(ws_realtime)

    # The FORMAT STRINGS, wherever they sit relative to the `logger.` call —
    # both are written across several lines.
    fn_call = [ln for ln in src.splitlines()
               if "[REALTIME] Function call:" in ln and ln.lstrip().startswith('"')]
    assert fn_call, "the function-call log line was renamed — update this probe"
    for ln in fn_call:
        assert "%s(%s)" not in ln, (
            "the tool arguments are logged verbatim; `think`'s are the user's words"
        )
        assert "sha=%s" in ln and "len=%d" in ln, ln

    injected = [ln for ln in src.splitlines() if "[REALTIME] Injected text:" in ln]
    assert injected, "the inject_text log line was renamed — update this probe"
    for ln in injected:
        assert re.search(r"len=%d sha=%s", ln), ln
        assert "[:60]" not in ln

    # …and the argument expressions that follow carry a digest, not a value.
    for marker in ("[REALTIME] Function call:", "[REALTIME] Injected text:"):
        block = src.split(marker, 1)[1][:400]
        assert "hashlib.sha256" in block, (marker, block[:200])


def test_the_platform_process_installs_the_identifier_redaction_too():
    """Whether the lines above are redacted at all is decided in the ENTRYPOINT.

    `install_content_redaction()` — the one that attaches the identifier filter
    and the redacting Formatter to the ROOT HANDLERS and to uvicorn's own — ran
    only in `agent_main`. `ws_realtime`, `ws_chat_proxy` and `agent_setup` all
    run in the PLATFORM process, which had query-token scrubbing and nothing
    else: no phone/JID/bot-token masking, and httpx left at INFO.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "platform_main.py").read_text("utf-8")
    assert "install_content_redaction()" in src, (
        "the platform process is where the voice relay runs and where an "
        "identifier is most likely to reach a log line"
    )
