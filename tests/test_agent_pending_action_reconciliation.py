"""Tenant-safe reconciliation reads and cancellation for durable voice tasks."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from app.api import agent as api
from app.db.database import async_session_maker
from app.db.models import AgentConfig, ConnectorPendingAction, User


async def _tenant(*, key: str) -> str:
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@example.com",
            hashed_password="x",
            name="Voice reconciliation test",
        ))
        db.add(AgentConfig(user_id=user_id, agent_api_key=key))
        await db.commit()
    return user_id


async def _action(user_id: str, *, status: str = "pending",
                  expires_at: datetime | None = None) -> str:
    action_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(ConnectorPendingAction(
            id=action_id,
            user_id=user_id,
            connector_id="gmail",
            tool_name="gmail__send_message",
            payload_json=json.dumps({"to": "test@example.com"}),
            channel="voice_task",
            status=status,
            expires_at=expires_at or datetime.utcnow() + timedelta(hours=1),
            decided_at=datetime.utcnow() if status != "pending" else None,
        ))
        await db.commit()
    return action_id


@pytest.mark.asyncio
async def test_reconciliation_auth_and_ownership_do_not_leak_actions():
    alice_id = await _tenant(key="alice-key")
    bob_id = await _tenant(key="bob-key")
    action_id = await _action(alice_id)

    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as wrong_key:
            await api.agent_pending_action_status(
                action_id, x_agent_key="wrong", x_agent_user_id=alice_id, db=db,
            )
        assert wrong_key.value.status_code == 403

        with pytest.raises(HTTPException) as other_tenant:
            await api.agent_pending_action_status(
                action_id, x_agent_key="bob-key", x_agent_user_id=bob_id, db=db,
            )
        assert other_tenant.value.status_code == 404


@pytest.mark.asyncio
async def test_reconciliation_read_expires_a_stale_pending_action_once():
    user_id = await _tenant(key="agent-key")
    action_id = await _action(
        user_id, expires_at=datetime.utcnow() - timedelta(seconds=1),
    )

    async with async_session_maker() as db:
        first = await api.agent_pending_action_status(
            action_id, x_agent_key="agent-key", x_agent_user_id=user_id, db=db,
        )
        second = await api.agent_pending_action_status(
            action_id, x_agent_key="agent-key", x_agent_user_id=user_id, db=db,
        )
        row = await db.get(ConnectorPendingAction, action_id)

    assert first.status == second.status == "expired"
    assert first.decided_at == second.decided_at
    assert row.status == "expired"
    assert row.decided_via == "voice_task_reconcile"


@pytest.mark.asyncio
async def test_cancel_rejects_pending_but_never_overwrites_winning_approval():
    user_id = await _tenant(key="agent-key")
    pending_id = await _action(user_id)
    approved_id = await _action(user_id, status="approved")

    async with async_session_maker() as db:
        cancelled = await api.agent_cancel_pending_action(
            pending_id, x_agent_key="agent-key", x_agent_user_id=user_id, db=db,
        )
        approved = await api.agent_cancel_pending_action(
            approved_id, x_agent_key="agent-key", x_agent_user_id=user_id, db=db,
        )
        pending_row = await db.get(ConnectorPendingAction, pending_id)
        approved_row = await db.get(ConnectorPendingAction, approved_id)

    assert cancelled.status == pending_row.status == "rejected"
    assert pending_row.decided_via == "voice_task_cancel"
    assert approved.status == approved_row.status == "approved"


@pytest.mark.asyncio
async def test_reconciliation_status_returns_the_final_approved_payload():
    user_id = await _tenant(key="agent-key")
    action_id = await _action(user_id)
    final_payload = {
        "to": "approved@example.test",
        "subject": "Approved subject",
        "body": "Approved body",
    }
    async with async_session_maker() as db:
        row = await db.get(ConnectorPendingAction, action_id)
        row.payload_json = json.dumps(final_payload, sort_keys=True)
        row.status = "executed"
        row.result_json = json.dumps({"kind": "ok", "id": "sent-1"})
        row.decided_at = datetime.utcnow()
        await db.commit()

        status = await api.agent_pending_action_status(
            action_id, x_agent_key="agent-key", x_agent_user_id=user_id, db=db,
        )

    assert status.payload == final_payload
    assert status.result == {"kind": "ok", "id": "sent-1"}
