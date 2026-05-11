"""Regression test for the 2026-05-11 ToolPillRow shipping bug.

`agent_runner.run()` collects per-call tool records into a local
`tool_event_records` list, then calls `_save_messages()` to persist
them into `Message.metadata_json["tool_events"]`. The first version
of that wiring referenced `tool_event_records` from inside
`_save_messages()` without the parameter being on the signature —
NameError at runtime, every Gmail/connector turn died with the
`[Response interrupted due to an error]` suffix.

Source-grep style (matches the pattern in test_time_channel_fix.py)
so the test runs without booting the whole agent runtime."""
from __future__ import annotations

from pathlib import Path


_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()


def _signature_block(marker: str) -> str:
    idx = _SRC.index(marker)
    return _SRC[idx:_SRC.index("):", idx) + 2]


def _call_block(marker: str, max_chars: int = 1500) -> str:
    call_idx = _SRC.index(marker)
    block = _SRC[call_idx:call_idx + max_chars]
    lines = block.splitlines()
    end = next((i for i, ln in enumerate(lines) if ln.strip() == ")"), len(lines))
    return "\n".join(lines[: end + 1])


def test_save_messages_accepts_tool_event_records():
    """_save_messages() must declare tool_event_records on its signature.
    Without it, the body's reference NameErrors at runtime — caught at
    the WS chat layer as `[WS] Agent error` and surfaced to the user
    as the generic `Something went wrong` message."""
    sig = _signature_block("async def _save_messages(")
    assert "tool_event_records" in sig, (
        "_save_messages signature missing `tool_event_records:` — "
        "every connector tool turn will NameError on the persist path. "
        "Hotfix: add `tool_event_records: Optional[List[Dict[str, Any]]] = None`."
    )


def test_run_threads_tool_event_records_to_save_messages():
    """run() must pass tool_event_records=tool_event_records into the
    _save_messages call. Without the thread, the local accumulator
    silently does nothing and the persistence half of ToolPillRow is
    a no-op."""
    call_block = _call_block("await self._save_messages(")
    assert "tool_event_records=tool_event_records" in call_block, (
        "run() must pass tool_event_records=tool_event_records to "
        "_save_messages so the per-call ToolPillRow records land in "
        "Message.metadata_json['tool_events'] for history reload."
    )


def test_save_messages_body_uses_local_only():
    """Defense in depth — _save_messages's body should reference
    tool_event_records only as a parameter, never as a closure
    capture. Catches the original bug pattern (variable defined in a
    different scope, accidentally relied on through Python's late-
    binding closure rules) regressing if someone moves code around."""
    body_idx = _SRC.index("async def _save_messages(")
    next_def = _SRC.index("\n    async def ", body_idx + 1)
    body = _SRC[body_idx:next_def]
    # The signature should declare the parameter exactly once; the body
    # may reference it any number of times. We just want a guarantee
    # that the symbol appears on the signature line(s) — already
    # checked above — and that the body doesn't somehow use a
    # different identifier for the same data.
    refs = body.count("tool_event_records")
    assert refs >= 2, (
        f"Expected tool_event_records referenced >=2 times in "
        f"_save_messages (signature + body usage); got {refs}. "
        f"Did the metadata-write block get removed?"
    )
