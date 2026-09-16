"""No filename, no path, no file contents in any log line on the attachment
path (A15).

This was live, not hypothetical. `agent_runner._build_media_content` logged

    [AGENT] Image loaded: /tmp/toup_media_ne4m72s_/IMG_4893.jpg (…)

and the fleet log carried the camera-roll filenames of real users' photos —
the same privacy class as incident 3's voice_handler transcript logging.
Telemetry may carry a stage, a duration, a reason code, a count or a size
class. It may not carry the user's file.
"""

from __future__ import annotations

import ast
import os

import pytest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Names whose VALUE is a filename, a path, or a file's contents.
FORBIDDEN = {
    "filename", "fname", "_fname", "name", "_mname", "path", "_fpath", "_path",
    "text", "_text", "data", "_data", "_raw", "raw", "content", "body",
}

FILES = [
    ("app/agent/attachment_ingest.py", None),
    ("app/agent/attachment_limits.py", None),
    ("app/api/chat_attachments.py", None),
]


def _logging_calls(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            if fn.value.id in ("logger", "logging") and fn.attr in (
                "debug", "info", "warning", "error", "exception", "critical"
            ):
                yield node


def _names_in(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
    return out


@pytest.mark.parametrize("relpath,_", FILES)
def test_no_log_call_interpolates_a_filename_or_contents(relpath, _):
    src = open(os.path.join(BACKEND, relpath), "r", encoding="utf-8").read()
    tree = ast.parse(src)
    for call in _logging_calls(tree):
        used = set()
        for arg in call.args[1:] if call.args else []:
            used |= _names_in(arg)
        # f-strings in the FIRST arg are the classic leak
        if call.args:
            first = call.args[0]
            if isinstance(first, ast.JoinedStr):
                used |= _names_in(first)
        leaked = used & FORBIDDEN
        assert not leaked, (
            f"{relpath}:{call.lineno} logs {sorted(leaked)} — a user's filename "
            f"or file contents must never reach the fleet log"
        )


def test_the_attachment_blocks_builder_logs_nothing_about_the_file():
    """`_build_attachment_blocks` replaced a function whose three log lines
    each carried a path. It logs nothing at all now: the manifest it builds is
    MODEL-facing prompt content, which is a different surface from telemetry."""
    src = open(os.path.join(BACKEND, "app/agent/agent_runner.py"), "r", encoding="utf-8").read()
    start = src.index("def _build_attachment_blocks(")
    end = src.index("# ------------------------------------------------------------------\n    # Error logging")
    body = src[start:end]
    assert "[AGENT] Image loaded" not in body
    assert "Document extracted" not in body
    for line in body.splitlines():
        if "logger." in line and not line.strip().startswith("#"):
            pytest.fail(f"a log line returned to the blocks builder: {line.strip()}")


def test_the_upload_telemetry_carries_a_size_class_not_a_size():
    src = open(os.path.join(BACKEND, "app/api/chat_attachments.py"), "r", encoding="utf-8").read()
    assert "_size_class(" in src
    assert "size_class=%s" in src


def test_the_ingest_record_is_counts_only():
    from app.agent.attachment_ingest import IngestedAttachment

    ing = IngestedAttachment(
        name="private-medical-report.pdf",
        mime="application/pdf",
        kind="document",
        size_bytes=10,
        status="ok",
        text="DIAGNOSIS",
    )
    rec = ing.to_record()
    assert "private-medical-report" not in str(rec)
    assert "DIAGNOSIS" not in str(rec)
    assert rec["chars"] == len("DIAGNOSIS")


# ── a traceback is a path, too ───────────────────────────────────────────

#: Handlers on the delivery / persist path. Each of these wraps a storage read
#: whose exception renders the per-user workspace path — `FileNotFoundError:
#: [Errno 2] … '/app/workspace/<user_id>/generated/<uuid>_<real filename>'` —
#: so `logger.exception` puts the user id AND the filename into the trail that
#: ships to Loki, past a format string that looks clean.
EXC_FREE_REGIONS = [
    ("app/agent/channels/shared/message_handler.py", "async def _deliver_attachments"),
    ("app/agent/channels/shared/message_handler.py", "async def deliver_bot_attachments"),
    ("app/agent/tool_executor.py", "browser screenshot persist failed"),
    ("app/agent/tool_executor.py", "send_file/send_photo persist failed"),
    ("app/agent/tool_executor.py", "[TTS] persist failed"),
]


#: Call sites, as opposed to bodies. `_deliver_turn_attachments` is called from
#: a 200-line handler that legitimately logs tracebacks for the TURN, so the
#: unit here is the `try` around the call, not its enclosing function. Measured
#: against the old +-600-character window: this `except` sat 4083 characters
#: from the anchor it was supposed to be inside, i.e. on the delivery path the
#: test exists to keep traceback-free and invisible to it.
EXC_FREE_CALL_SITES = [
    ("app/agent/telegram_bot.py", "await self._deliver_turn_attachments("),
    ("app/agent/channels/shared/message_handler.py", "await _deliver_attachments("),
]


def _enclosing_function(src: str, anchor: str):
    """The SMALLEST function whose source contains `anchor`, as source text.

    The unit has to be a syntactic region, not a character window: a window
    passes or fails on how much unrelated code happens to sit nearby, so a
    benign edit that grows the region either turns it red for no behavioural
    reason or hides a newly added traceback log. Measured: in telegram_bot.py
    one `logger.exception` cleared the old +-600 window by 15 characters.
    """
    lines = src.splitlines(keepends=True)
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    tree = ast.parse(src)
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = offsets[node.lineno - 1]
        end = offsets[node.end_lineno - 1] + len(lines[node.end_lineno - 1])
        seg = src[start:end]
        if anchor in seg and (best is None or len(seg) < len(best[0])):
            best = (seg, node.name)
    return best


@pytest.mark.parametrize("relpath,anchor", EXC_FREE_REGIONS)
def test_no_traceback_is_logged_on_the_attachment_delivery_path(relpath, anchor):
    src = open(os.path.join(BACKEND, relpath), "r", encoding="utf-8").read()
    assert anchor in src, f"{relpath}: anchor {anchor!r} moved — re-point this test"
    found = _enclosing_function(src, anchor)
    assert found, f"{relpath}: {anchor!r} is not inside a function any more"
    body, name = found
    assert "logger.exception" not in body, (
        f"{relpath}:{name}: a traceback here renders the user's workspace path"
    )


@pytest.mark.parametrize("relpath,call", EXC_FREE_CALL_SITES)
def test_no_traceback_is_logged_around_the_delivery_call(relpath, call):
    src = open(os.path.join(BACKEND, relpath), "r", encoding="utf-8").read()
    assert call in src, f"{relpath}: {call!r} moved — the delivery step is GONE"
    i = src.rindex(call)
    # The `try`/`except` immediately around the call — the handler around THAT
    # may log tracebacks for the turn itself, which is not this path.
    assert "logger.exception" not in src[i: i + 500], (
        f"{relpath}: the delivery call's own except renders the workspace path"
    )


def test_no_exception_text_reaches_the_model_from_the_image_tools():
    """`attachment_ingest` states the rule for the ingest path: "no exception
    text ever reaches the model — a stack string in the prompt is both a leak
    and a lie about what the file is." The image tools returned `{exc}` straight
    into the tool result, i.e. into the context and the user's answer."""
    src = open(os.path.join(BACKEND, "app/agent/tool_executor.py"), "r", encoding="utf-8").read()
    # Every place a STORAGE read or write can fail on the image/attachment
    # path. Each of these sentences replaced a `{exc}` that rendered the
    # per-user workspace path (or an S3 endpoint and bucket) into the answer.
    for anchor in (
        "Reference image {_art.id} could not be loaded",
        "The image to edit could not be loaded",
        "That file could not be read",
        "The generated image could not be saved",
        "The edited image could not be saved",
        "Could not attach {name}; nothing was sent.",
        "Could not save the audio; nothing was attached.",
    ):
        assert anchor in src, f"{anchor!r} moved — re-point this test"
    for sentence in (
        "Could not load reference image",
        "Could not load the image to edit",
        "Could not save the generated image",
        "Could not save the edited image",
    ):
        assert sentence not in src, f"{sentence!r} still carries the exception text"
