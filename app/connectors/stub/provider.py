"""Stub connector provider — deterministic results, no network.

Loaded only with `include_experimental=True` (default off in
production). Used by T1c registry tests and T1d OAuth-flow tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import ClassVar, Optional

from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorResult,
    HealthResult,
    RefreshResult,
)


class StubProvider(BaseConnectorProvider):
    """Returns deterministic responses so tests can assert on shape."""

    manifest_id: ClassVar[str] = "stub"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        if tool_name == "stub__list_items":
            # Deterministic feed for the automations e2e harness
            # (scripts/e2e_automations.py): same three items every
            # poll, so the event-dedupe gate is what makes the second
            # poll a no-op — exactly the rail under test.
            return ConnectorOk(content=json.dumps({
                "items": [
                    {"id": "item-1", "title": "First stub item"},
                    {"id": "item-2", "title": "Second stub item"},
                    {"id": "item-3", "title": "Third stub item"},
                ],
            }))
        return ConnectorOk(
            content=json.dumps({
                "tool": tool_name,
                "input": tool_input,
                "user_id_prefix": ctx.user_id[:8] if ctx.user_id else None,
                "channel": ctx.channel,
            }),
        )

    async def revoke(
        self,
        user_id: str,
        access_token: str,
        refresh_token=None,
    ) -> None:
        # No-op — stub has no upstream. Tests that need to exercise
        # the "revoke fails" path monkey-patch this method to raise.
        return None

    async def refresh(
        self,
        refresh_token: str,
        *,
        scopes: Optional[list[str]] = None,
    ) -> RefreshResult:
        return RefreshResult(
            access_token=f"stub_refreshed_access_for_{refresh_token[:8]}",
            refresh_token=refresh_token,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        return HealthResult(ok=True, detail="stub always healthy")
