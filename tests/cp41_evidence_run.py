"""CP4.1 evidence — runnable as a plain script (bypasses conftest DB setup).

Runs the five required checks and prints PASS/FAIL + captured log lines.
Exit code 0 iff all pass.

Invoke from backend/:
    DATABASE_URL="sqlite+aiosqlite:///:memory:" \
      RUN_MODE=platform JWT_SECRET=t ENCRYPTION_KEY=12345678901234567890123456789012 \
      PLATFORM_ENCRYPTION_KEY="SwlDSRHQctCGibJg-hQaXpX2QOIPJIOahm6Xgln38C8=" \
      PYTHONPATH=. python tests/cp41_evidence_run.py
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from pathlib import Path

from app.agent.agent_runner import (
    VAULT_TOOL_CHANNEL_BLOCK,
    VAULT_TOOL_NAME,
    strip_vault_tool_for_channel,
)
from app.agent import pending_credential_confirms as pcc
from app.agent.pending_credential_confirms import (
    RATE_LIMIT_MAX_ATTEMPTS,
    _peek_pending_count_for_tests,
    _reset_all_for_tests,
)
from app.agent.tool_definitions import get_extended_tools
from app.agent.tool_executor import ToolExecutor


FAILS: list[str] = []


def _ok(label: str):
    print(f"  ✓ {label}")


def _fail(label: str, detail: str = ""):
    msg = f"  ✗ {label}: {detail}" if detail else f"  ✗ {label}"
    print(msg)
    FAILS.append(msg)


def extended_tools():
    tools = get_extended_tools()
    names = {t.get("name") for t in tools}
    assert VAULT_TOOL_NAME in names, (
        f"{VAULT_TOOL_NAME} missing from get_extended_tools()"
    )
    return tools


# ── (1) + (2): channel filter inclusion/exclusion ─────────────────────────

def check_channel_filter():
    print("\n[1+2] channel filter inclusion/exclusion")
    tools = extended_tools()
    for ch in ("telegram", "voice", "mobile"):
        names = [t.get("name") for t in strip_vault_tool_for_channel(tools, ch)]
        if VAULT_TOOL_NAME in names:
            _fail(f"{ch}: tool was NOT stripped")
        else:
            _ok(f"{ch}: tool stripped")
    for ch in ("web", "app", "vibecoding", "discord", "slack", None):
        names = [t.get("name") for t in strip_vault_tool_for_channel(tools, ch)]
        if VAULT_TOOL_NAME not in names:
            _fail(f"{ch}: tool was incorrectly stripped")
        else:
            _ok(f"{ch}: tool present")
    if VAULT_TOOL_CHANNEL_BLOCK != frozenset({"telegram", "voice", "mobile"}):
        _fail("VAULT_TOOL_CHANNEL_BLOCK mismatch", str(VAULT_TOOL_CHANNEL_BLOCK))
    else:
        _ok("VAULT_TOOL_CHANNEL_BLOCK matches approval spec")


# ── (3): rate limit counts attempts ───────────────────────────────────────

async def check_rate_limit():
    print("\n[3] rate limit counts attempts (not saves)")
    await _reset_all_for_tests()

    executor = ToolExecutor(workspace=str(Path("/tmp/cp41-ws")))
    executor.set_user_id("u-ratelimit")
    executor.set_channel("web")

    sent_frames = []

    async def capture(payload):
        sent_frames.append(payload)

    executor._on_credential_confirm_request = capture

    for i in range(RATE_LIMIT_MAX_ATTEMPTS):
        r = await executor.execute(
            "save_streaming_credential", {"channel": "netflix"}
        )
        if r.startswith("ERROR"):
            _fail(f"attempt {i} errored", r)
            return

    if len(sent_frames) != RATE_LIMIT_MAX_ATTEMPTS:
        _fail(f"frame count after {RATE_LIMIT_MAX_ATTEMPTS} attempts",
              f"got {len(sent_frames)}")
    else:
        _ok(f"{RATE_LIMIT_MAX_ATTEMPTS} frames emitted for {RATE_LIMIT_MAX_ATTEMPTS} attempts")

    pending_before = await _peek_pending_count_for_tests()
    if pending_before != RATE_LIMIT_MAX_ATTEMPTS:
        _fail(f"pending count before 6th attempt", str(pending_before))
    else:
        _ok(f"pending count = {pending_before} after {RATE_LIMIT_MAX_ATTEMPTS} successful attempts")

    # 6th attempt: MUST fail rate_limited, MUST NOT emit frame, MUST NOT add pending
    r = await executor.execute(
        "save_streaming_credential", {"channel": "netflix"}
    )
    if not r.startswith("ERROR: rate_limited"):
        _fail("6th attempt return string", r)
    else:
        _ok("6th attempt → rate_limited")

    if len(sent_frames) != RATE_LIMIT_MAX_ATTEMPTS:
        _fail("6th attempt emitted a WS frame (should not)",
              f"frames={len(sent_frames)}")
    else:
        _ok("6th attempt emitted no WS frame")

    pending_after = await _peek_pending_count_for_tests()
    if pending_after != RATE_LIMIT_MAX_ATTEMPTS:
        _fail("6th attempt created a pending entry (should not)",
              str(pending_after))
    else:
        _ok(f"6th attempt created no pending entry (count stayed at {pending_after})")


# ── (4): unexpected field discard + key-name-only warning ────────────────

async def check_field_strip():
    print("\n[4] unexpected fields discarded with key-name-only warning")
    await _reset_all_for_tests()

    executor = ToolExecutor(workspace=str(Path("/tmp/cp41-ws")))
    executor.set_user_id("u-strip")
    executor.set_channel("web")

    captured = []

    async def capture(payload):
        captured.append(payload)

    executor._on_credential_confirm_request = capture

    SECRET = "UNMENTIONABLE_PW_1234567890"
    EVIL_EMAIL = "evil@example.com"

    # Capture warnings on the tool_executor logger.
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.WARNING)
    h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    tool_logger = logging.getLogger("app.agent.tool_executor")
    tool_logger.addHandler(h)
    tool_logger.setLevel(logging.WARNING)

    try:
        r = await executor.execute(
            "save_streaming_credential",
            {
                "channel": "netflix",
                "email_hint": "user@example.com",
                "password": SECRET,
                "extra_noise": EVIL_EMAIL,
            },
        )
    finally:
        tool_logger.removeHandler(h)

    if r.startswith("ERROR"):
        _fail("handler errored", r)
        return

    if len(captured) != 1:
        _fail(f"frame count", str(len(captured)))
        return

    frame = captured[0]
    if "password" in frame:
        _fail("outbound frame contains 'password' key")
    else:
        _ok("outbound frame has no 'password' key")

    frame_repr = repr(frame)
    if SECRET in frame_repr:
        _fail("outbound frame contains password VALUE")
    else:
        _ok("outbound frame does not contain password value")
    if EVIL_EMAIL in frame_repr:
        _fail("outbound frame contains extra_noise VALUE")
    else:
        _ok("outbound frame does not contain extra_noise value")

    log_text = buf.getvalue()
    if "unexpected field" not in log_text:
        _fail("no 'unexpected field' warning", log_text[:300])
    else:
        _ok("'unexpected field' warning fired")
    if "password" not in log_text:
        _fail("warning does not mention 'password' key name")
    else:
        _ok("warning mentions 'password' key name")
    if "extra_noise" not in log_text:
        _fail("warning does not mention 'extra_noise' key name")
    else:
        _ok("warning mentions 'extra_noise' key name")
    if SECRET in log_text:
        _fail("warning leaked the password VALUE", log_text[:500])
    else:
        _ok("warning does NOT leak the password value")
    if EVIL_EMAIL in log_text:
        _fail("warning leaked extra_noise VALUE", log_text[:500])
    else:
        _ok("warning does NOT leak extra_noise value")

    # Pending entry should carry only allowed fields.
    entry = await pcc.pop_pending(frame["request_id"], "u-strip")
    if entry is None:
        _fail("pending entry not found for the emitted request_id")
    else:
        if entry.channel != "netflix":
            _fail("entry.channel", str(entry.channel))
        else:
            _ok(f"entry.channel = {entry.channel}")
        if entry.email_hint != "user@example.com":
            _fail("entry.email_hint", str(entry.email_hint))
        else:
            _ok(f"entry.email_hint = {entry.email_hint}")


# ── (5): pending_credential_confirms.py contains the NOTE comment ────────

def check_note_comment():
    print("\n[5] pending-dict module contains single-replica NOTE comment")
    src = Path(pcc.__file__).read_text(encoding="utf-8")
    for phrase in (
        "single Railway replica",
        "scaled horizontally",
        "Redis",
    ):
        if phrase not in src:
            _fail(f"missing phrase", phrase)
        else:
            _ok(f"found: {phrase!r}")


# ── Defense-in-depth #2: handler rejects on blocked channels ─────────────

async def check_handler_gate():
    print("\n[gate] handler hard-check rejects blocked channels even if invoked")
    await _reset_all_for_tests()
    executor = ToolExecutor(workspace=str(Path("/tmp/cp41-ws")))
    executor.set_user_id("u-gate")
    for ch in ("telegram", "voice", "mobile"):
        executor.set_channel(ch)
        r = await executor.execute(
            "save_streaming_credential", {"channel": "netflix"}
        )
        if r.startswith("ERROR:") and "not available on this channel" in r:
            _ok(f"{ch}: handler rejected")
        else:
            _fail(f"{ch}: unexpected return", r)


# ── Defense-in-depth #1: tool description carries the warnings ────────────

def check_tool_description():
    print("\n[desc] tool description warns against Telegram/voice + no-password")
    t = next(t for t in get_extended_tools() if t.get("name") == VAULT_TOOL_NAME)
    desc = t.get("description", "")
    for phrase in ("NEVER", "Telegram", "voice", "NEVER include a password"):
        if phrase not in desc:
            _fail("description missing phrase", phrase)
        else:
            _ok(f"found: {phrase!r}")


async def amain():
    check_channel_filter()
    await check_rate_limit()
    await check_field_strip()
    check_note_comment()
    await check_handler_gate()
    check_tool_description()


def main():
    asyncio.run(amain())
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} checks failed")
        for f in FAILS:
            print(f"  {f}")
        sys.exit(1)
    print("ALL CP4.1 EVIDENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
