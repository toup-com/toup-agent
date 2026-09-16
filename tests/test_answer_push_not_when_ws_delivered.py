"""The out-of-band fallback asks "has the user SEEN this", not "could push reach a device".

Round 46, incident 2 (2026-09-15). The founder's in-app answer was delivered
on the WebSocket at 14:50:44.7 and then ALSO sent to their WhatsApp as a
`mission_completed` notification at 14:50:53. Nothing was broken: the phone was
seconds old and had not registered a push token, so
`notification_dispatcher` recorded `expo: skipped/no_active_devices`, `delivered`
stayed False, and the fallback condition — "push could not reach anyone" —
was satisfied. The fact that WOULD have answered the real question, whether the
answer reached a live socket, existed at the producer in `ws_chat` and was
logged and thrown away.

`data_json.delivered_in_app` is that fact, carried. The row is still QUEUED
unconditionally (it drives the Live Activity / card lane, founder decision
2026-07-17); only the Telegram/WhatsApp fan-out is gated.

Everything in THIS file is a source probe: it pins the gate's shape so it
cannot be quietly removed. The BEHAVIOUR — that a row carrying the flag is
finalised without the out-of-band fan-out, and that the same row without it
still falls back — is executed against the real dispatcher and real rows in
`tests/test_notification_dispatcher.py`
(`test_an_answer_already_delivered_in_app_is_not_pushed_out_of_band` and its
control). A grep alone could not see a second call site reaching
`_request_agent_channel_delivery` around the gate, nor the flag being
rejected at ingest before the dispatcher ever saw it.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. pytest \
      tests/test_answer_push_not_when_ws_delivered.py -q -p no:cacheprovider
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def dispatcher_src() -> str:
    return (BACKEND / "app/services/notification_dispatcher.py").read_text()


def fallback_block() -> str:
    src = dispatcher_src()
    i = src.index("# Out-of-band fallback only when push couldn't reach anyone.")
    return src[i:i + 2600]


def test_the_flag_gates_the_fallback():
    block = fallback_block()
    assert 'delivered_in_app' in block, block[:400]
    assert re.search(
        r'and not \(row\.data_json or \{\}\)\.get\("delivered_in_app"\)', block
    ), block[:1200]


def test_the_gate_is_inside_the_same_condition_as_the_other_suppressors():
    """`no_agent_fallback` and `silent` are the two existing suppressors and
    they live in one `if`. A separate early return above it would skip the
    `_finalize` bookkeeping the branch below depends on."""
    block = fallback_block()
    i_if = block.index("if (")
    i_call = block.index("_request_agent_channel_delivery")
    i_flag = block.index("delivered_in_app")
    assert i_if < i_flag < i_call, "the flag must be a term of the existing condition"


def test_a_delivered_row_is_finalized_as_sent_rather_than_retried():
    """Under-notifying is the risk this change takes on. A row whose answer
    reached the socket must be recorded as DELIVERED, not left to churn the
    backoff ladder as if nothing had happened."""
    block = fallback_block()
    assert 'channels["ws"] = {"status": "ok"' in block, block[-900:]
    assert re.search(r'elif \(row\.data_json or \{\}\)\.get\("delivered_in_app"\)', block)
    assert "delivered = True" in block.split('channels["ws"]')[1][:200]


def test_the_row_is_still_queued_unconditionally():
    """The founder's 2026-07-17 decision stands: every chat answer notifies the
    phone regardless of app state. Only the out-of-band fan-out is gated, and
    the producer's comment says so."""
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    i = src.index("Answer delivery push — UNCONDITIONAL")
    assert "UNCONDITIONAL" in src[i:i + 80]
    # …and the flag rides the SAME payload, not a second call site.
    block = src[i:i + 3000]
    assert '"delivered_in_app"' in block


def test_the_producer_sets_it_only_on_a_real_delivery():
    """The bound the design takes: `delivered_in_app` may only be true when a
    frame actually reached a socket — the turn's own `done`, or the late
    `message` broadcast to a reconnected one. Intent is not delivery."""
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    i = src.index('"delivered_in_app"')
    line = src[i:src.index("\n", i)]
    assert "_done_delivered" in line and "_late_sent" in line, line
    # Both are facts about a send that did not RAISE, which is the strongest
    # fact this process has: `_safe_send` returns True whenever `send_json`
    # returned, and `broadcast_to_user` counts the queues it put the event on.
    # Neither is an acknowledgement from the phone — no such signal exists on
    # this socket — so the claim these pin is "a frame left here for a live
    # socket", never "the user read it".
    assert "_done_delivered = await _safe_send(_done_payload)" in src
    assert "_late_sent = await broadcast_to_user(" in src


def test_late_sent_is_initialised_before_the_push_reads_it():
    """The late lane only runs when the `done` failed. If `_late_sent` were
    declared inside it, the happy path would raise NameError building the push
    payload — a crash on every successful turn."""
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    i_init = src.index("_late_sent = 0")
    i_branch = src.index("if not _done_delivered and (response.text or \"\").strip():")
    i_read = src.index('"delivered_in_app"')
    assert i_init < i_branch < i_read


def test_fallback_kinds_are_unchanged():
    """The gate narrows WHEN the fallback fires, never WHICH kinds may use it —
    a reminder that fires while the user is away still reaches their channels."""
    src = dispatcher_src()
    i = src.index("_FALLBACK_KINDS")
    block = src[i:i + 400]
    assert "mission_completed" in block


@pytest.mark.parametrize("key", ["no_agent_fallback", "silent", "delivered_in_app"])
def test_every_suppressor_reads_a_missing_data_json_as_absent(key):
    """`data_json` is nullable. `(row.data_json or {})` is what keeps a row
    with no payload from raising inside the dispatcher's hot loop."""
    block = fallback_block()
    assert f'(row.data_json or {{}}).get("{key}")' in block, key
