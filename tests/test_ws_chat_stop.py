"""A user must be able to stop a turn from any socket they still have.

`_wait_for_stop` reads only the connection that RECEIVED the message, so for
the whole life of the stop protocol a client could cancel its turn from exactly
one socket. The phone loses that socket on every backgrounding, every blip and
every container swap — while the turn keeps running headless, which is the
entire point of the in-flight registry — and its stop then landed in the main
receive loop and came back as `Unknown message type: stop`. The agent carried
on and the client was told its own frame was a protocol error.

These pin the out-of-band lane: who can cancel, what "nothing to cancel" means,
and the identity guard that stops one turn's teardown taking a newer turn's
handle with it.
"""
import asyncio
import inspect

import pytest

from app.api import ws_chat
from app.api.ws_chat import (
    _cancel_active_turn,
    _register_stoppable_turn,
    _retire_stoppable_turn,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    ws_chat._stoppable_turns.clear()
    yield
    ws_chat._stoppable_turns.clear()


async def _forever():
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_registered_turn_is_cancellable_from_another_socket():
    task = asyncio.create_task(_forever())
    _register_stoppable_turn("u1", "m1", task)

    assert _cancel_active_turn("u1") is True

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.asyncio
async def test_stop_with_nothing_running_is_a_no_op_not_an_error():
    # The client tears down locally the instant the button is pressed, so a
    # stop that arrives with no turn left is the ordinary late case — it must
    # answer, and it must not raise.
    assert _cancel_active_turn("nobody") is False


@pytest.mark.asyncio
async def test_a_finished_turn_is_not_cancelled_again():
    async def _done():
        return 1

    task = asyncio.create_task(_done())
    await task
    _register_stoppable_turn("u1", "m1", task)

    assert _cancel_active_turn("u1") is False
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_one_users_stop_cannot_reach_another_users_turn():
    task = asyncio.create_task(_forever())
    _register_stoppable_turn("u1", "m1", task)

    assert _cancel_active_turn("u2") is False
    assert not task.done()

    task.cancel()


def test_retire_is_identity_guarded_on_the_mission_id():
    # Both teardowns run — the per-turn `finally` and the connection-level
    # backstop — and a turn that ended while a NEWER one is already registered
    # must not take the newer one's handle with it. Same guard, and the same
    # reason, as `_clear_active_turn`.
    sentinel = object()
    _register_stoppable_turn("u1", "m2", sentinel)

    _retire_stoppable_turn("u1", "m1")          # the older turn's teardown
    assert ws_chat._stoppable_turns.get("u1") == ("m2", sentinel)

    _retire_stoppable_turn("u1", "m2")          # its own
    assert "u1" not in ws_chat._stoppable_turns


def test_retire_of_an_absent_entry_is_silent():
    _retire_stoppable_turn("ghost", "m1")
    assert "ghost" not in ws_chat._stoppable_turns


def test_the_receive_loop_answers_stop_before_the_unknown_type_fallback():
    # Order is the whole fix: the `stop` branch has to sit ABOVE the
    # `msg_type != "message"` catch-all, or the frame still comes back as
    # "Unknown message type: stop" and nothing about the code below it matters.
    src = inspect.getsource(ws_chat.ws_chat)
    stop_at = src.find('if msg_type == "stop":')
    unknown_at = src.find('if msg_type != "message":')
    assert stop_at > 0, "the receive loop has no `stop` branch"
    assert unknown_at > 0
    assert stop_at < unknown_at

    branch = src[stop_at:unknown_at]
    assert "_cancel_active_turn(user_id)" in branch
    # Answered either way — the turn's own CancelledError branch replies on the
    # socket that STARTED it, which in the case this lane exists for is exactly
    # the socket that is gone.
    assert '"type": "stopped"' in branch


def test_a_stopped_turn_retires_its_chat_task_job():
    # The answer branch completes the BuildJob and the error branch fails it;
    # the cancel branch did neither, so a stopped turn left a `running` row
    # behind and both clients drew it as an "In progress" card forever, under a
    # reply that had visibly stopped. Reproduced on the simulator 2026-08-25.
    src = inspect.getsource(ws_chat.ws_chat)
    # The TURN's cancel branch, not `_wait_for_stop`'s own — that one appears
    # first and swallows its cancellation with a bare `pass`.
    cancel_at = src.find("except asyncio.CancelledError:\n                    stop_task.cancel()")
    assert cancel_at > 0, "the turn's cancel branch is not where this expects it"
    nxt = src.find("except Exception as e:", cancel_at)
    branch = src[cancel_at:nxt if nxt > 0 else len(src)]
    assert "_chat_task_job_id" in branch, "the cancel branch never touches the task job"
    # The ROW is written, not merely announced — a broadcast alone leaves the
    # card `running` in the database and it comes straight back on the next
    # load. (Asserting only that the string "cancelled" appears somewhere in the
    # branch passes on the broadcast alone; this is that mutation, killed.)
    assert '.status = "cancelled"' in branch
    assert "completed_at" in branch
    # …and the live client is told, rather than left to its next poll.
    assert '"type": "job_update"' in branch


def test_the_partial_save_cannot_be_attempted_without_a_conversation():
    # `Message.conversation_id` is NOT NULL and a brand-new user's first
    # message arrives with session_id=null, so this write used to raise
    # IntegrityError into a bare `except: pass` — the partial lost, with no log
    # line to say so.
    src = inspect.getsource(ws_chat.ws_chat)
    marker = src.find("*[Generation stopped by user]*")
    assert marker > 0
    guard = src.rfind("if _partial and not session_id:", 0, marker)
    assert guard > 0, "the stop partial-save is not guarded on session_id"
    assert "except Exception:\n                            pass" not in src[guard:marker], \
        "the partial save still swallows its failure silently"
