"""Regression for the persistent duplicate-user-bubble bug (2026-05-24).

Symptom (operator-reproduced, mrvviinn@gmail.com): user sends 'HEY'
exactly once, but the chat UI renders TWO 'HEY' bubbles. Persists across
page refresh.

Diagnosis: the agent's DB has exactly ONE 'HEY' message
(verified via /api/day-chats/<today>/messages). The duplicate is a
frontend-side rendering issue — the 'user_message_persisted' handler
in ChatPage.tsx swaps the optimistic message's temp id (`cmsg-...`)
for the server uuid via `.map()`, but doesn't dedupe afterwards. If the
day-chat refetch landed before the WS ack and already added the server-
uuid entry, the map produces TWO entries with the same id. React renders
both, and the SWR `['day-messages', iso]` cache mirror writes both back
to localStorage, so the duplicate survives refresh.

These tests are source-level guards that fail if the dedup logic gets
removed or regressed. No live browser test — the fix is straightforward
to verify by hand in staging.
"""
from __future__ import annotations

import re
from pathlib import Path


CHATPAGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "frontend" / "src" / "modules" / "chat" / "ChatPage.tsx"
)


def _read_chatpage_source() -> str:
    return CHATPAGE_PATH.read_text(encoding="utf-8")


def test_user_message_persisted_handler_dedupes_by_id():
    """The handler must build the new messages array via a Set/Map that
    collapses duplicate ids, NOT via a plain .map() that produces two
    entries when the server id is already present in `prev`.
    """
    src = _read_chatpage_source()
    # Find the handler block.
    m = re.search(
        r"case 'user_message_persisted':(.+?)break;",
        src, flags=re.DOTALL,
    )
    assert m, "user_message_persisted handler block not found in ChatPage"
    body = m.group(1)

    # The buggy v1 pattern was: setMessages(prev => prev.map(m => ...));
    # The fixed v2 pattern uses a Set or Map to dedupe — accept either
    # `new Set<string>` or `new Map<string` as the dedup primitive.
    assert ("new Set" in body) or ("new Map" in body), (
        "user_message_persisted handler must dedupe after the temp-id "
        "swap. A plain .map() leaves two entries with the same id when "
        "the day-chat refetch already pulled the server uuid into "
        "`prev` — surfacing as duplicate user bubbles that persist "
        "across refresh via the SWR cache mirror."
    )

    # Belt-and-suspenders: the duplicate must not be re-introduced by a
    # naive `prev.map(m => m.id === client ? rename : m)` pattern alone.
    # We require an explicit guard against the seen id.
    assert "seen" in body.lower() or "byid" in body.lower(), (
        "Dedup must explicitly track seen ids so the iteration "
        "collapses the optimistic + server entries into one."
    )


def test_user_message_persisted_preserves_server_id():
    """The dedup must still rename the optimistic temp id to the server
    id — losing the rename would break the user_message_persisted
    contract (server id is the seed for the messages-since polling
    backstop on reconnect)."""
    src = _read_chatpage_source()
    m = re.search(
        r"case 'user_message_persisted':(.+?)break;",
        src, flags=re.DOTALL,
    )
    assert m, "handler block missing"
    body = m.group(1)
    # The handler must still reference both ids — client_msg_id to find
    # the optimistic entry, server_msg_id to install as the canonical id.
    assert "data.client_msg_id" in body
    assert "data.server_msg_id" in body
    # And the watermark for the reconnect-recovery path must still be
    # advanced.
    assert "lastSeenMessageIdRef.current" in body
