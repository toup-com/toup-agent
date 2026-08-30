"""R38 · C — the automation thread gets a surface to elevate on.

A connector WRITE from `automation_thread` was denied outright, and the
deny was honest about its reason: the elevation card such a call needs
had nowhere to be drawn in a thread, and a confirmation the surface
cannot show is a wedge, not a safeguard.

The surface exists now — `thread_agent._append_approval_turns` draws one
`needs_you` turn per staged call, `fix="approve"`, carrying the
`pending_action_id` its buttons POST to. This file proves the half that
makes the button honest: the channel STAGES instead of denying, and an
approval on that channel actually executes the call. Either half alone
is worse than the deny it replaced — a card nobody can draw, or a button
that reports failure on every tap.

Platform lane: `connector_identities` / `connector_pending_actions` live
on the platform side, because the tokens that execute them do.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.api import connector_pending_actions as api
from app.connectors.base import (
    ConnectorConfirmationRequired, ConnectorToolError,
)
from app.db.database import async_session_maker
from app.db.models import ConnectorPendingAction
from app.services import connector_dispatcher as dispatcher

from tests.test_connector_elevation_gate import (  # noqa: F401 — fixtures
    CONNECTOR, DRAFT, TOOL, _StubUser, _provision_crypto, _seed_identity,
    alice_user_id, register_elevated,
)

THREAD = "automation_thread"


async def _dispatch(user_id: str, *, channel: str, elevation: bool = False):
    async with async_session_maker() as db:
        return await dispatcher.execute(
            db, user_id, CONNECTOR, TOOL,
            tool_input=dict(DRAFT), channel=channel,
        )


# ─── the policy, on its own ──────────────────────────────────────────

def test_a_staging_channel_must_also_be_a_confirmable_one():
    """The invariant the whole design rests on. A channel that stages a
    write and cannot draw a card stages into a void; a channel that
    draws a card and cannot execute the approval offers a button that
    lies. `stages_writes_for_approval` is the one predicate the surface
    and the executor both read."""
    assert dispatcher._MUTATES_CONFIRM_CHANNELS <= \
        dispatcher._CONFIRMABLE_CHANNELS
    for channel in dispatcher._MUTATES_CONFIRM_CHANNELS:
        assert dispatcher.stages_writes_for_approval(channel), channel
    assert not dispatcher.stages_writes_for_approval("web")
    assert not dispatcher.stages_writes_for_approval("")


def test_the_unattended_channels_still_deny_a_write(register_elevated):
    """R38 lifts the deny for ONE channel, the one that got a surface.
    Every other unattended channel keeps it — a routine tick, a trigger
    and a background turn have nobody watching to approve anything."""
    manifest, _ = register_elevated(elevation=False)
    tool = manifest.tools[0]

    for channel in ("routine", "trigger", "background", "voice",
                    "autopilot", "subagent"):
        refusal = dispatcher._resolve_channel_policy(tool, channel)
        assert isinstance(refusal, ConnectorToolError), channel

    assert dispatcher._resolve_channel_policy(tool, THREAD) is None


def test_an_approved_re_entry_is_not_denied_by_the_channel_it_staged_on(
    register_elevated,
):
    """The approve route re-enters `execute` with the row's own channel.
    Without honouring `approved_action_id` here, every tap on a thread
    card would come back a refusal — the button lying, which is the one
    outcome worse than the deny this replaced."""
    manifest, _ = register_elevated(elevation=False)
    tool = manifest.tools[0]
    # Decided by the CHANNEL alone: a staging channel keeps its mutating
    # tools on the first pass AND on the approve re-entry. The resolver used
    # to take an `approved_action_id` it never read, with a docstring saying
    # that parameter was what made the re-entry safe.
    assert dispatcher._resolve_channel_policy(tool, THREAD) is None
    import inspect
    assert "approved_action_id" not in inspect.signature(
        dispatcher._resolve_channel_policy).parameters, (
        "the resolver took a parameter it never read, and said in its "
        "docstring that the parameter was the safety")


# ─── the dispatcher, end to end ──────────────────────────────────────

@pytest.mark.asyncio
async def test_a_write_from_the_thread_is_staged_not_refused(
    register_elevated, alice_user_id,
):
    """Pre-R38 this returned `ConnectorToolError` — "no inline
    confirmation surface. User must invoke from web." — for a user who
    was sitting in the thread watching."""
    _, provider = register_elevated(elevation=False)
    await _seed_identity(alice_user_id)

    result = await _dispatch(alice_user_id, channel=THREAD)

    assert isinstance(result, ConnectorConfirmationRequired), result
    assert provider.execute_call_count == 0
    assert result.action_id

    async with async_session_maker() as db:
        row = await db.get(ConnectorPendingAction, result.action_id)
    assert row.status == "pending"
    assert row.channel == THREAD
    assert json.loads(row.payload_json) == DRAFT


@pytest.mark.asyncio
async def test_every_mutating_tool_stages_here_not_just_elevated_ones(
    register_elevated, alice_user_id,
):
    """The card is forced on `mutates`, not on `elevation`.

    Half the write surface — a draft, a label change — is `mutates:
    true` with `elevation: false`, and those are exactly the calls that
    would otherwise run unattended on a channel whose only licence to
    write is that the user approves each one.
    """
    _, provider = register_elevated(elevation=False)
    await _seed_identity(alice_user_id)
    assert provider.execute_call_count == 0

    result = await _dispatch(alice_user_id, channel=THREAD)
    assert isinstance(result, ConnectorConfirmationRequired)

    # ...and the same tool on the main chat, which does NOT force a
    # card for a non-elevation write, still runs. The forcing is scoped
    # to the staging channel and nothing else moved.
    ran = await _dispatch(alice_user_id, channel="web")
    assert not isinstance(ran, ConnectorConfirmationRequired), ran
    assert provider.execute_call_count == 1


@pytest.mark.asyncio
async def test_approving_a_thread_card_actually_makes_the_call(
    register_elevated, alice_user_id,
):
    """The button, tapped. This is the test that had to exist before
    `automation_thread` could join `_CONFIRMABLE_CHANNELS` at all."""
    _, provider = register_elevated(elevation=False)
    await _seed_identity(alice_user_id)

    staged = await _dispatch(alice_user_id, channel=THREAD)
    assert isinstance(staged, ConnectorConfirmationRequired)

    async with async_session_maker() as db:
        out = await api.approve_pending_action(
            staged.action_id,
            api.ApproveRequest(payload=None, decided_via="automation_thread"),
            current_user=_StubUser(alice_user_id),
            db=db,
        )

    assert provider.execute_call_count == 1
    assert provider.seen_inputs == [DRAFT]
    assert out.status == "executed"

    async with async_session_maker() as db:
        rows = (await db.execute(select(ConnectorPendingAction))).scalars().all()
    assert [r.status for r in rows] == ["executed"]


@pytest.mark.asyncio
async def test_rejecting_a_thread_card_makes_no_call(
    register_elevated, alice_user_id,
):
    _, provider = register_elevated(elevation=False)
    await _seed_identity(alice_user_id)
    staged = await _dispatch(alice_user_id, channel=THREAD)

    async with async_session_maker() as db:
        out = await api.reject_pending_action(
            staged.action_id,
            api.RejectRequest(reason="no"),
            current_user=_StubUser(alice_user_id),
            db=db,
        )

    assert out.status == "rejected"
    assert provider.execute_call_count == 0


def test_the_agent_path_cannot_hand_itself_an_approval():
    """`approved_action_id` is the one key that lifts both the channel
    deny and the elevation card, so the surface the MODEL drives must
    never be able to pass it. The MCP handler is that surface."""
    import inspect

    from app.services import connector_mcp

    src = inspect.getsource(connector_mcp)
    assert "approved_action_id" not in src

    sig = inspect.signature(dispatcher.execute)
    param = sig.parameters["approved_action_id"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
